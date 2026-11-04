import subprocess
from pathlib import Path
from typing import Literal

import numpy as np
import pytest
from meedee import messages, smf
from test_synth import check_audio
from ufor import performance_binding, synth_trace
from ufor.events import PerformanceEvent, Release, Trigger
from ufor.instrument_trace import TraceAction
from ufor.samples import instrument, trace

from enge import fm, live, noise, sample_instrument, synth
from enge.midi import MidiPerformance, read_midi
from enge.performance import MidiConverter
from enge.presets import Patch
from enge.render import render_midi, write_flac


def decode(path: Path) -> np.ndarray:
    raw = subprocess.run(
        [
            "flac",
            "--silent",
            "--decode",
            "--stdout",
            "--force-raw-format",
            "--endian=little",
            "--sign=signed",
            str(path),
        ],
        check=True,
        capture_output=True,
    ).stdout
    octets = np.frombuffer(raw, dtype=np.uint8).reshape(-1, 3).astype(np.int32)
    samples = octets[:, 0] | octets[:, 1] << 8 | octets[:, 2] << 16
    samples = (samples ^ (1 << 23)) - (1 << 23)
    return samples.reshape(-1, 2).astype(np.float64) / (2**23 - 1)


def test_mixed_engine_render_preserves_notes_tails_and_block_boundaries(
    tmp_path: Path,
    backend: Literal["numpy", "native"],
) -> None:
    engines: list[Literal["fm", "synth", "sample", "noise"]] = [
        "fm",
        "synth",
        "sample",
        "noise",
    ]
    patches = {
        f"channel-{c}": Patch(engine=e, pan=(c - 1.5) / 3).prepare_score(48000)
        for c, e in enumerate(engines)
    }
    converters = []
    for c, (part, (score, _)) in enumerate(patches.items(), start=1):
        binding = performance_binding.PerformanceBindingScore.model_validate(
            {
                "name": part,
                "title": part,
                "body": {
                    "instrument": {"selector": f"test:{part}"},
                    "midi": [
                        {
                            "channels": [c],
                            "part": part,
                            "repeated_key_release": "oldest",
                            "controllers": [{"number": 11, "control": "expression"}],
                        }
                    ],
                },
            }
        )
        converters.append(MidiConverter(source=part, binding=binding, instrument=score))
    path = tmp_path / "mixed.mid"
    writer = smf.Writer(path, header=smf.Header(format=0, tracks=1, division=480))
    for c in range(4):
        writer.write(
            smf.Event(
                track=0,
                tick=c * 24,
                record=messages.NoteOn(channel=c + 1, note=60 + c * 4, velocity=102),
            )
        )
    for c in range(1, 5):
        writer.write(
            smf.Event(
                track=0,
                tick=360,
                record=messages.ControlChange(channel=c, controller=11, value=102),
            )
        )
    for c in range(4):
        writer.write(
            smf.Event(
                track=0,
                tick=800,
                record=messages.NoteOff(channel=c + 1, note=60 + c * 4, velocity=0),
            )
        )
    writer.write(smf.Event(track=0, tick=960, record=smf.Meta(kind=0x2F, data=b"")))
    writer.finish()
    performance = read_midi(path, converters=converters)
    full = tmp_path / "full.flac"
    partitioned = tmp_path / "partitioned.flac"
    report = render_midi(performance, patches, full, block_size=60000, backend=backend)
    other = render_midi(
        performance, patches, partitioned, block_size=997, backend=backend
    )
    actual, expected = decode(partitioned), decode(full)
    check_audio(tmp_path / "mixed-engines.wav", actual, expected)
    assert report.frames == other.frames == 60001
    assert report.notes == 4
    assert report.peak == pytest.approx(other.peak)
    assert 0 < report.peak < 1
    assert np.all(actual[44000:] == 0)


@pytest.mark.parametrize("filename", ["audio.flac", "output with spaces/café.flac"])
def test_flac_round_trips_pcm_and_preserves_destination_on_overload(
    tmp_path: Path,
    filename: str,
) -> None:
    t = np.arange(48000) / 48000
    samples = np.column_stack(
        [0.2 * np.sin(2 * np.pi * 440 * t), 0.3 * np.sin(2 * np.pi * 660 * t)]
    )
    path = tmp_path / filename
    assert write_flac([samples], path) == pytest.approx(0.3)
    expected = np.rint(samples * (2**23 - 1)) / (2**23 - 1)
    check_audio(tmp_path / "pcm.wav", decode(path), expected)
    original = path.read_bytes()
    with pytest.raises(ValueError, match="headroom"):
        write_flac([samples, np.full((48000, 2), 1.1)], path)
    assert path.read_bytes() == original


@pytest.mark.parametrize("engine", ["fm", "synth", "sample", "noise"])
def test_preset_velocity_scales_audio_and_survives_snapshot_restore(
    tmp_path: Path,
    backend: Literal["numpy", "native"],
    engine: Literal["fm", "synth", "sample", "noise"],
) -> None:
    score, assets = Patch(engine=engine).prepare_score(48000)

    def render(velocity: float) -> np.ndarray:
        events: list[PerformanceEvent] = [
            Trigger(
                tick=0,
                ordinal=0,
                part="main",
                trigger_id="note",
                key=69,
                pitch_hz=440,
                velocity=velocity,
            ),
            Release(tick=36000, ordinal=1, part="main", trigger_id="note"),
        ]
        actions: list[TraceAction]
        if isinstance(score, instrument.SampleInstrumentScore):
            renderer = sample_instrument.OfflineSampler(
                sample_instrument.prepare(score, assets), backend
            )
            actions = list(trace.prepare(score.body, events, seed=0).actions)
        else:
            actions = list(synth_trace.prepare(score.body, events, seed=0).actions)
            if engine == "fm":
                renderer = fm.OfflineFM(fm.prepare(score), backend)
            elif engine == "noise":
                renderer = noise.OfflineNoise(noise.prepare(score), backend)
            else:
                renderer = synth.OfflineSynth(synth.prepare(score), backend)
        first = renderer.advance([a for a in actions if a.tick < 24000], 0, 24000)
        assert any(c.velocity == velocity for c in renderer.snapshot().contexts)
        remaining = [a for a in actions if a.tick >= 24000]
        if isinstance(renderer, fm.OfflineFM):
            snapshot = renderer.snapshot()
            second = renderer.advance(remaining, 24000, 60000)
            renderer.restore(
                fm.FMSnapshot.model_validate_json(snapshot.model_dump_json())
            )
        elif isinstance(renderer, noise.OfflineNoise):
            snapshot = renderer.snapshot()
            second = renderer.advance(remaining, 24000, 60000)
            renderer.restore(
                noise.NoiseSnapshot.model_validate_json(snapshot.model_dump_json())
            )
        elif isinstance(renderer, sample_instrument.OfflineSampler):
            snapshot = renderer.snapshot()
            second = renderer.advance(remaining, 24000, 60000)
            renderer.restore(
                sample_instrument.SamplerSnapshot.model_validate_json(
                    snapshot.model_dump_json()
                )
            )
        else:
            snapshot = renderer.snapshot()
            second = renderer.advance(remaining, 24000, 60000)
            renderer.restore(
                synth.SynthSnapshot.model_validate_json(snapshot.model_dump_json())
            )
        repeated = renderer.advance(remaining, 24000, 60000)
        output = np.concatenate([first, second])
        check_audio(
            tmp_path / f"{engine}-{velocity}-restore.wav",
            np.concatenate([first, repeated]),
            output,
        )
        if isinstance(renderer, fm.OfflineFM):
            source = fm.PersistentFM(renderer.definition)
        elif isinstance(renderer, noise.OfflineNoise):
            source = noise.PersistentNoise(renderer.definition)
        elif isinstance(renderer, sample_instrument.OfflineSampler):
            source = sample_instrument.PersistentSampler(renderer.definition)
        else:
            source = synth.PersistentSynth(renderer.definition)
        native = live.NativeLiveEngine({"voice": source}, 60000)
        actual = np.empty_like(output)
        native.advance_into(
            {"voice": [a for a in actions if a.tick < 24000]},
            [],
            0,
            24000,
            actual[:24000],
        )
        saved = native.snapshot()
        native.advance_into({"voice": remaining}, [], 24000, 60000, actual[24000:])
        check_audio(tmp_path / f"{engine}-{velocity}-native.wav", actual, output)
        native.restore(saved)
        replay = np.empty_like(second)
        for start in range(24000, 60000, 997):
            end = min(start + 997, 60000)
            native.advance_into(
                {"voice": [a for a in remaining if start <= a.tick < end]},
                [],
                start,
                end,
                replay[start - 24000 : end - 24000],
            )
        np.testing.assert_array_equal(replay, actual[24000:])
        return output

    loud = render(1.0)
    quiet = render(0.25)
    assert np.max(np.abs(loud)) > 0.01
    check_audio(tmp_path / f"{engine}-velocity.wav", quiet, loud * 0.25)


@pytest.mark.parametrize("engine", ["fm", "synth", "noise"])
@pytest.mark.parametrize("binding", ["velocity", "key"])
def test_persistent_event_bindings_keep_reused_trigger_tails_separate(
    tmp_path: Path, engine: Literal["fm", "synth", "noise"], binding: str
) -> None:
    score, _ = Patch(engine=engine).prepare_score(48000)
    assert not isinstance(score, instrument.SampleInstrumentScore)
    if binding == "key":
        raw = score.model_dump(mode="json")
        voice = raw["body"]["voices"][0]
        voice["bindings"][0]["kind"] = "key"
        next(s for s in voice["modulation"]["sources"] if s["name"] == "velocity")[
            "maximum"
        ] = 127
        route = next(
            r for r in voice["modulation"]["routes"] if r["name"] == "velocity"
        )
        route["points"][-1]["input"] = 127
        score = type(score).model_validate(raw)
    if engine == "fm":
        definition = fm.prepare(score)
        reference = fm.OfflineFM(definition)
        renderer = fm.PersistentFM(definition)
    elif engine == "noise":
        definition = noise.prepare(score)
        reference = noise.OfflineNoise(definition)
        renderer = noise.PersistentNoise(definition)
    else:
        definition = synth.prepare(score)
        reference = synth.OfflineSynth(definition)
        renderer = synth.PersistentSynth(definition)
    events: list[PerformanceEvent] = [
        Trigger(
            tick=t,
            ordinal=0,
            part="main",
            trigger_id="note",
            key=int(v * 128) if binding == "key" else 69,
            pitch_hz=440,
            velocity=v,
        )
        for t, v in [(0, 0.25), (6001, 0.75), (24000, 0)]
    ]
    events.extend(
        Release(tick=t, ordinal=0, part="main", trigger_id="note")
        for t in (6000, 18000, 30000)
    )
    actions = list(synth_trace.prepare(score.body, events, seed=0).actions)
    expected = reference.advance(actions, 0, 48000)
    first = renderer.advance([a for a in actions if a.tick < 7001], 0, 7001)
    saved = renderer.snapshot()
    if binding == "velocity":
        assert sorted(saved.trigger_velocities.values()) == [0.25, 0.75]
    later = [a for a in actions if a.tick >= 7001]
    tail = renderer.advance(later, 7001, 48000)
    assert not renderer.snapshot().trigger_velocities
    check_audio(
        tmp_path / f"{engine}-velocity-tails.wav",
        np.concatenate([first, tail]),
        expected,
    )
    renderer.restore(saved)
    pieces = []
    for start in range(7001, 48000, 997):
        end = min(start + 997, 48000)
        pieces.append(
            renderer.advance([a for a in later if start <= a.tick < end], start, end)
        )
    np.testing.assert_array_equal(np.concatenate(pieces), tail)
    if binding == "key":
        native = live.NativeLiveEngine({"voice": type(renderer)(definition)}, 48000)
        actual = np.empty_like(expected)
        native.advance_into({"voice": actions}, [], 0, 48000, actual)
        check_audio(tmp_path / f"{engine}-key-callback.wav", actual, expected)


def test_render_rejects_incompatible_prepared_rate_before_replacing_output(
    tmp_path: Path,
) -> None:
    path = tmp_path / "existing.flac"
    path.write_bytes(b"existing")
    performance = MidiPerformance(sample_rate=44100, frames=44100, parts={"main": []})
    with pytest.raises(ValueError, match="performance rate"):
        render_midi(
            performance, {"main": Patch(engine="synth").prepare_score(48000)}, path
        )
    assert path.read_bytes() == b"existing"
