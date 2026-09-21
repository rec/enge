import json
from pathlib import Path
from typing import Literal

import numpy as np
import pytest
from test_filters import matrix_filter
from test_synth import check_audio
from test_synth_demo import publish_flac
from ufor import synth_trace
from ufor.envelope import Envelope, Segment
from ufor.events import ControlChange, Release, Trigger
from ufor.instrument_trace import VoiceRetirement
from ufor.samples.processing import ResonantFilter
from ufor.synth import SynthInstrumentScore

from enge import filters, noise, synth


def score() -> SynthInstrumentScore:
    return SynthInstrumentScore.model_validate_json(
        (Path(__file__).parents[1] / "conformance/noise-instrument.json").read_text()
    )


def trigger(frame: int = 0, name: str = "note") -> Trigger:
    return Trigger(tick=frame, ordinal=0, part="main", trigger_id=name, key=60)


def change(frame: int, value: float) -> ControlChange:
    return ControlChange(
        tick=frame,
        ordinal=1,
        control="color",
        value=value,
        scope="trigger",
        part="main",
        trigger_id="note",
    )


def scalar_noise(key: int, start: int, count: int) -> np.ndarray:
    result = []
    mask = 2**64 - 1
    for i in range(start, start + count):
        word = (key + (i + 1) * 0x9E3779B97F4A7C15) & mask
        word = ((word ^ (word >> 30)) * 0xBF58476D1CE4E5B9) & mask
        word = ((word ^ (word >> 27)) * 0x94D049BB133111EB) & mask
        word ^= word >> 31
        result.append(2 * ((word >> 11) / 2**53) - 1)
    return np.array(result)[:, None]


def definition() -> noise.PreparedVoice:
    return noise.PreparedVoice(
        sample_rate=48000,
        envelope=Envelope(
            segments=[Segment(duration=0, target=1)],
            release=[Segment(duration="1/10", target=0)],
        ),
        routes=[[1, 0.5]],
    )


def test_noise_raw_matches_oracle_and_statistics(
    tmp_path: Path, backend: Literal["numpy", "native"]
) -> None:
    renderer = noise.VoiceRenderer.start(definition(), 0, backend)
    actual = renderer.render(48000, np.ones(48000))
    expected = scalar_noise(0, 0, 48000) * [[1, 0.5]]
    check_audio(tmp_path / "noise-white.wav", actual, expected)
    np.testing.assert_array_equal(actual, expected)
    assert -1 <= actual[:, 0].min() < actual[:, 0].max() < 1
    # Fixed seed, 48k samples: generous statistical bounds, independent of parity.
    assert abs(actual[:, 0].mean()) < 0.015
    assert abs(actual[:, 0].var() - 1 / 3) < 0.015
    for lag in (1, 7, 31):
        assert abs(np.corrcoef(actual[:-lag, 0], actual[lag:, 0])[0, 1]) < 0.025
    assert renderer.frame_count == 48000


def test_noise_portable_vectors_and_exhaustion() -> None:
    from enge import _native

    vectors = json.loads(
        (Path(__file__).parents[1] / "conformance/noise-v1.json").read_text()
    )
    for v in vectors["samples"]:
        expected = 2 * ((v["word"] >> 11) / 2**53) - 1
        assert expected == v["sample"]
        assert noise.noise_samples(v["key"], v["index"], 1)[0, 0] == expected
        audio, _ = _native.render_noise(
            v["key"],
            v["index"],
            48000,
            np.ones(1),
            np.array([[0, 1, 1, 0, 0, 1, 0]], dtype=float),
            np.ones(1),
            ([], np.zeros((0, 0, 2)), np.zeros((1, 0, 2))),
        )
        assert audio[0, 0] == expected
    assert noise.noise_samples(0, 2**64, 0).shape == (0, 1)
    with pytest.raises(synth.EngineError, match="exhausted"):
        noise.noise_samples(0, 2**64, 1)


@pytest.mark.parametrize("block", [64, 128, 256, 997, 1024])
def test_noise_partitions_and_json_restore(
    tmp_path: Path, backend: Literal["numpy", "native"], block: int
) -> None:
    document = score()
    events = [
        trigger(),
        change(12001, 1),
        change(13000, 0.2),
        trigger(17003, "second"),
        Release(tick=35003, ordinal=0, part="main", trigger_id="note"),
    ]
    actions = synth_trace.prepare(document.body, events, seed=19).actions
    prepared = noise.prepare(document)
    whole = noise.OfflineNoise(prepared, backend)
    expected = whole.advance(actions, 0, 48000)
    split = noise.OfflineNoise(prepared, backend)
    actual = np.empty_like(expected)
    for start in range(0, 48000, block):
        end = min(48000, start + block)
        actual[start:end] = split.advance(
            [a for a in actions if start <= a.tick < end], start, end
        )
        snapshot = noise.NoiseSnapshot.model_validate_json(
            split.snapshot().model_dump_json()
        )
        split = noise.OfflineNoise(prepared, backend)
        split.restore(snapshot)
    check_audio(tmp_path / "noise-partitions.wav", actual, expected)
    assert split.snapshot() == whole.snapshot()


def test_noise_dynamic_filter_and_minimum_hold_match_oracle(
    tmp_path: Path, backend: Literal["numpy", "native"]
) -> None:
    prepared = definition().model_copy(
        update={
            "minimum_hold_seconds": 0.5,
            "filters": [
                ResonantFilter(name="tone", response="lowpass", cutoff_hz=1200, q=0.7)
            ],
        }
    )
    renderer = noise.VoiceRenderer.start(prepared, 42, backend)
    assert renderer.release()
    assert not renderer.release()
    assert renderer.release_frame == 24000
    count = 28800
    values = np.empty((count, 1, 2))
    values[:, 0, 0] = np.linspace(1200, 8000, count)
    values[:, 0, 1] = np.linspace(0.7, 2, count)
    gains = np.ones(count)
    gains[:1000] = 0
    actual = renderer.render(48000, gains, values)
    expected = np.zeros_like(actual)
    filtered = matrix_filter(prepared.filters, scalar_noise(42, 0, count), values)
    envelope = np.minimum(1, 1 - (np.arange(count) - 24000) / 4800)
    expected[:count] = filtered * (envelope * gains)[:, None] * [[1, 0.5]]
    check_audio(tmp_path / "noise-filter-release.wav", actual, expected)
    assert renderer.complete
    assert renderer.frame_count == count
    np.testing.assert_array_equal(renderer.render(0, np.empty(0)), np.empty((0, 2)))
    assert renderer.frame_count == count


def test_persistent_noise_matches_native_across_blocks_and_restore(
    tmp_path: Path,
) -> None:
    document = score()
    events = [
        trigger(),
        change(12001, 1),
        trigger(17003, "second"),
        Release(tick=35003, ordinal=0, part="main", trigger_id="note"),
    ]
    actions = synth_trace.prepare(document.body, events, seed=19).actions
    prepared = noise.prepare(document)
    expected = noise.OfflineNoise(prepared, "native").advance(actions, 0, 48000)
    engine = noise.PersistentNoise(prepared, voices=4)
    chunks = []
    boundaries = [0, 997, 12002, 17004, 24001, 35004, 48000]
    for start, end in zip(boundaries, boundaries[1:], strict=False):
        chunks.append(
            engine.advance([a for a in actions if start <= a.tick < end], start, end)
        )
        if end == 24001:
            snapshot = engine.snapshot()
    actual = np.concatenate(chunks)
    restored = noise.PersistentNoise(prepared, voices=4)
    restored.restore(snapshot)
    replay = restored.advance(
        [a for a in actions if 24001 <= a.tick < 48000], 24001, 48000
    )

    check_audio(tmp_path / "persistent-noise.wav", actual, expected)
    np.testing.assert_allclose(replay, actual[24001:], atol=0)


def test_noise_streams_ignore_pitch_and_other_renderers(
    tmp_path: Path, backend: Literal["numpy", "native"]
) -> None:
    document = score()
    events = [trigger(), trigger(0, "second").model_copy(update={"ordinal": 1})]
    actions = synth_trace.prepare(document.body, events, seed=0).actions
    pitched = synth_trace.prepare(
        document.body, [e.model_copy(update={"pitch_hz": 123}) for e in events], seed=0
    ).actions
    engine = noise.OfflineNoise(noise.prepare(document), backend)
    other = noise.OfflineNoise(noise.prepare(document), backend)
    actual = engine.advance(actions, 0, 48000)
    expected = other.advance(pitched, 0, 48000)
    check_audio(tmp_path / "noise-pitch.wav", actual, expected)
    keys = [v.renderer.stream_key for v in engine.snapshot().voices]
    assert keys[0] != keys[1]


def test_noise_demo_and_backend_parity(
    tmp_path: Path, pytestconfig: pytest.Config, backend: Literal["numpy", "native"]
) -> None:
    document = score()
    events = [
        trigger(),
        change(12000, 1),
        change(13200, 0.3),
        Release(tick=36000, ordinal=0, part="main", trigger_id="note"),
        trigger(48000, "wind"),
    ]
    actions = synth_trace.prepare(document.body, events, seed=5).actions
    actions.append(
        VoiceRetirement(
            tick=90000,
            ordinal=0,
            voice_id="voice-1",
            cause="transport_stop",
            action="stop",
        )
    )
    engine = noise.OfflineNoise(noise.prepare(document), backend)
    actual = engine.advance(actions, 0, 96000)
    expected = noise.OfflineNoise(noise.prepare(document)).advance(actions, 0, 96000)
    check_audio(tmp_path / "noise-demo.wav", actual, expected)
    assert engine.snapshot().voices == []
    publish_flac(
        tmp_path / "noise-demo-actual.wav",
        pytestconfig.cache.mkdir("audio") / f"noise-demo-{backend}.flac",
    )


def test_native_noise_has_no_python_dsp_fallback(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    prepared = definition()
    expected = noise.VoiceRenderer.start(prepared, 7).render(48000, np.ones(48000))

    def forbidden(*args: object, **kwargs: object) -> None:
        raise AssertionError("Native noise called Python DSP")

    monkeypatch.setattr(noise, "noise_samples", forbidden)
    monkeypatch.setattr(synth, "envelope_samples", forbidden)
    monkeypatch.setattr(synth, "route_samples", forbidden)
    monkeypatch.setattr(filters, "filter_samples", forbidden)
    actual = noise.VoiceRenderer.start(prepared, 7, "native").render(
        48000, np.ones(48000)
    )
    check_audio(tmp_path / "noise-native.wav", actual, expected)


def test_noise_snapshot_rejects_incompatible_state() -> None:
    prepared = noise.prepare(score())
    reference = noise.OfflineNoise(prepared)
    native = noise.OfflineNoise(prepared, "native")
    with pytest.raises(synth.EngineError, match="backend"):
        native.restore(reference.snapshot())
    with pytest.raises(synth.EngineError, match="prepared"):
        reference.restore(
            reference.snapshot().model_copy(
                update={
                    "definition": prepared.model_copy(update={"sample_rate": 44100})
                }
            )
        )
    with pytest.raises(synth.EngineError, match="Unknown noise backend"):
        noise.OfflineNoise(prepared, "invalid")  # type: ignore[arg-type]


def test_noise_muting_and_interleaving_preserve_stream(
    tmp_path: Path, backend: Literal["numpy", "native"]
) -> None:
    renderer = noise.VoiceRenderer.start(definition(), 19, backend)
    unrelated = noise.VoiceRenderer.start(definition(), 23, backend)
    actual = np.empty((48000, 2))
    actual[:12000] = renderer.render(12000, np.zeros(12000))
    unrelated.render(48000, np.ones(48000))
    actual[12000:] = renderer.render(36000, np.ones(36000))
    expected = scalar_noise(19, 0, 48000) * [[1, 0.5]]
    expected[:12000] = 0
    check_audio(tmp_path / "noise-muted.wav", actual, expected)
    assert renderer.frame_count == 48000


def test_noise_lfo_amplitude_matches_oracle(
    tmp_path: Path, backend: Literal["numpy", "native"]
) -> None:
    raw = score().model_dump()
    voice = raw["body"]["voices"][0]
    voice["processing"] = {"volume_db": -12}
    voice["envelope"] = {
        "segments": [{"duration": 0, "target": 1}],
        "release": [{"duration": 0, "target": 0}],
    }
    voice["lfos"] = {"pulse": {"waveform": "sine", "rate": 2}}
    voice["modulation"] = {
        "sources": [{"name": "pulse", "scope": "voice", "minimum": -1, "maximum": 1}],
        "parameters": [
            {
                "target": {"name": "processing", "parameter": "amplitude"},
                "unit": "ratio",
                "scope": "voice",
                "minimum": 0,
                "maximum": 2,
                "default": 1,
            }
        ],
        "routes": [
            {
                "name": "pulse",
                "source": "pulse",
                "target": {"name": "processing", "parameter": "amplitude"},
                "unit": "ratio",
                "operation": "add",
                "points": [{"input": -1, "amount": -1}, {"input": 1, "amount": 1}],
            }
        ],
    }
    voice["bindings"] = [{"name": "pulse", "kind": "lfo", "reference": "pulse"}]
    document = SynthInstrumentScore.model_validate(raw)
    actions = synth_trace.prepare(document.body, [trigger()], seed=1).actions
    start = next(a for a in actions if isinstance(a, synth_trace.VoiceStart))
    assert start.noise_key is not None
    actual = noise.OfflineNoise(noise.prepare(document), backend).advance(
        actions, 0, 48000
    )
    gain = (1 + np.sin(2 * np.pi * 2 * np.arange(48000) / 48000)) * 10 ** (-12 / 20)
    expected = scalar_noise(start.noise_key, 0, 48000) * gain[:, None] * [[1, 0.5]]
    check_audio(tmp_path / "noise-lfo.wav", actual, expected)


def test_noise_counter_survives_json_and_rejects_overflow(
    backend: Literal["numpy", "native"],
) -> None:
    renderer = noise.VoiceRenderer.start(definition(), 2**64 - 1, backend)
    renderer.frame_count = 2**64
    restored = noise.VoiceRenderer.model_validate_json(renderer.model_dump_json())
    assert restored == renderer
    assert restored.render(0, np.empty(0)).shape == (0, 2)
    with pytest.raises(synth.EngineError, match="exhausted"):
        restored.render(1, np.ones(1))
    assert restored == renderer


def test_native_noise_owns_strided_buffers(tmp_path: Path) -> None:
    from enge import _native

    gains = np.ones(96000)[::2]
    routes = np.array([1, 0, 0.5, 0])[::2]
    spans = np.array([[0, 48000, 1, 0, 0, 1, 0]], dtype=float)
    memory = np.zeros((1, 1, 2))
    values = np.tile([[[1200, 0.7]]], (96000, 1, 1))[::2]
    originals = [a.copy() for a in (gains, routes, spans, memory, values)]
    audio, state = _native.render_noise(
        13, 0, 48000, gains, spans, routes, ([0], memory, values)
    )
    expected = matrix_filter(
        [ResonantFilter(name="tone", response="lowpass", cutoff_hz=1200, q=0.7)],
        scalar_noise(13, 0, 48000),
        values,
    ) * [[1, 0.5]]
    check_audio(tmp_path / "noise-owned.wav", audio, expected)
    for original, current in zip(
        originals, (gains, routes, spans, memory, values), strict=True
    ):
        np.testing.assert_array_equal(original, current)
        assert not np.shares_memory(audio, current)
        assert not np.shares_memory(state, current)


@pytest.mark.parametrize("case", ["rate", "gain", "routes", "counter", "filter_shape"])
def test_native_noise_rejects_invalid_buffers(case: str) -> None:
    from enge import _native

    with pytest.raises(ValueError):
        _native.render_noise(
            0,
            2**64 if case == "counter" else 0,
            0 if case == "rate" else 48000,
            np.array([np.nan if case == "gain" else 1.0]),
            np.array([[0, 1, 1, 0, 0, 1, 0]], dtype=float),
            np.empty(0) if case == "routes" else np.ones(1),
            (
                [],
                np.zeros((0, 0, 2)),
                np.zeros((2 if case == "filter_shape" else 1, 0, 2)),
            ),
        )
