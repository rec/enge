from fractions import Fraction
from itertools import pairwise
from pathlib import Path

import numpy as np
import pytest
from test_dynamic_synth import change, dynamic_score, onset
from test_synth import check_audio, score
from ufor.envelope import Envelope, Segment
from ufor.events import Release
from ufor.instrument_trace import RetirementCause, VoiceRetirement
from ufor.samples import instrument, trace
from ufor.samples.enums import LoopMode
from ufor.samples.playback import Loop, Playback, Slice

from enge import sample_instrument, sampler
from enge.synth import EngineError


def sample_score(
    frames: int = 96000, native_rate: int = 48000
) -> instrument.SampleInstrumentScore:
    raw = score().model_dump(mode="json")
    raw["kind"] = "instrument"
    raw["timebases"].append(
        {"name": "native", "kind": "physical", "rate": {"numerator": native_rate}}
    )
    raw["assets"] = [
        {
            "name": "asset",
            "path": "sample.wav",
            "encoding": "WAV/PCM_16",
            "byte_length": 0,
            "sha256": "0" * 64,
            "audio": {"timebase": "native", "channels": ["a", "b"], "frames": frames},
        }
    ]
    raw["body"] = {
        "slices": [{"name": "slice", "asset": "asset", "end_frame": frames}],
        "settings": {},
        "slots": [
            {
                "name": "sample",
                "slice": "slice",
                "mapping": {
                    "lowest_key": 0,
                    "highest_key": 127,
                    "reference_pitch_hz": 440,
                },
                "channels": [
                    {"input": "a", "output": "left", "gain": 1},
                    {"input": "a", "output": "right", "gain": 0.25},
                    {"input": "b", "output": "right", "gain": 1},
                ],
            }
        ],
    }
    return instrument.SampleInstrumentScore.model_validate(raw)


@pytest.mark.parametrize("block", [64, 997, 1024])
def test_sampler_consumes_sustain_controls_and_group_envelope_with_restores(
    tmp_path: Path, block: int
) -> None:
    raw = sample_score(native_rate=24000).model_dump(mode="json")
    dynamic = dynamic_score().model_dump(mode="json")["body"]
    common = raw["body"]["settings"]
    common["controls"] = {**dynamic["controls"], "pedal": {"default": 0}}
    common["sustain"] = {"control": "pedal"}
    common["processing"] = {"volume_db": 6, "tuning_cents": 1200}
    raw["body"]["groups"] = [
        {
            "name": "group",
            "processing": {"volume_db": -6},
            "envelope": {
                "segments": [{"duration": "1/20", "target": 1}],
                "release": [{"duration": "1/4", "target": 0.25}],
            },
            "bindings": dynamic["voices"][0]["bindings"],
            "modulation": dynamic["voices"][0]["modulation"],
        }
    ]
    slot = raw["body"]["slots"][0]
    # Omission inherits the group's whole setting categories.
    for name in (
        "processing",
        "envelope",
        "envelopes",
        "lfos",
        "bindings",
        "modulation",
    ):
        slot.pop(name, None)
    slot["group"] = "group"
    document = instrument.SampleInstrumentScore.model_validate(raw)
    ramp = np.arange(96000, dtype=np.float64) / 96000
    audio = np.column_stack([ramp, 1 - ramp])
    prepared = sample_instrument.prepare(document, {"asset": audio})
    actions = trace.prepare(
        document.body,
        [
            onset(pitch=220),
            change(120, 1),
            change(1000, 1, "pedal", scope="part"),
            change(12000, 1, "bend"),
            Release(tick=18000, ordinal=0, part="main", trigger_id="note"),
            change(24000, 0, "pedal", scope="part"),
            change(27000, 0.2),
        ],
        seed=0,
    ).actions
    renderer = sample_instrument.OfflineSampler(prepared)
    boundaries = sorted(
        {
            0,
            48000,
            121,
            23999,
            24000,
            24001,
            27001,
            35999,
            36000,
            *range(0, 48000, block),
        }
    )
    chunks: list[np.ndarray] = []
    for start, end in pairwise(boundaries):
        chunks.append(
            renderer.advance([a for a in actions if start <= a.tick < end], start, end)
        )
        saved = renderer.snapshot()
        assert bool(saved.voices) == (end < 36000)
        if end in (121, 24000, 24001, 27001, 35999):
            encoded = saved.model_dump_json()
            restored = sample_instrument.OfflineSampler(prepared)
            restored.restore(
                sample_instrument.SamplerSnapshot.model_validate_json(encoded)
            )
            renderer.advance([], end, end + 1)
            assert restored.snapshot().model_dump_json() == encoded
            assert (
                restored.definition.samples["sample"].samples
                is prepared.samples["sample"].samples
            )
            renderer = restored
    frames = np.arange(48000)
    progress = np.minimum(frames, 12000) * 0.5 + np.maximum(frames - 12000, 0)
    envelope = np.minimum(frames / 2400, 1) * (
        1 - 0.75 * np.clip((frames - 24000) / 12000, 0, 1)
    )
    envelope[36000:] = 0
    gain = (
        0.5
        + 0.5 * np.clip((frames - 120) / 240, 0, 1)
        - 0.8 * np.clip((frames - 27000) / 240, 0, 1)
    )
    expected = (
        np.column_stack([progress / 96000, 1 - 0.75 * progress / 96000])
        * (envelope * gain)[:, None]
    )
    check_audio(
        tmp_path / "sample-sustain-controls.wav", np.concatenate(chunks), expected
    )


@pytest.mark.parametrize("mode", [LoopMode.until_release, LoopMode.through_release])
def test_sample_voice_shares_fractional_minimum_hold_with_loop_and_envelope(
    tmp_path: Path, mode: LoopMode
) -> None:
    sample = sampler.PreparedSample(
        samples=np.arange(6, dtype=np.float64)[:, None] / 10,
        native_rate=48000,
        sample_rate=48000,
        slice=Slice(
            name="slice",
            asset="asset",
            end_frame=6,
            loop=Loop(start_frame=1, end_frame=4, mode=mode),
        ),
        playback=Playback(direction="mirror"),
    )
    definition = sampler.PreparedSampleVoice(
        sample_rate=48000,
        slice=sample.slice,
        routes=[[1, -0.5, 0]],
        pitch_ratio=2,
        minimum_hold_seconds=Fraction(3, 96000),
        envelope=Envelope(
            segments=[Segment(duration=Fraction(4, 48000), target=1)],
            release=[Segment(duration=Fraction(5, 96000), target=0.25)],
        ),
    )
    renderer = sampler.SampleVoiceRenderer.start(definition, sample)
    assert renderer.release()
    assert not renderer.release()
    chunks: list[np.ndarray] = []
    for start, end in pairwise([0, 1, 2, 3, 4, 48000]):
        renderer = sampler.SampleVoiceRenderer.model_validate_json(
            renderer.model_dump_json()
        )
        chunks.append(renderer.render(sample, end - start))
        assert renderer.complete == (
            end >= (3 if mode == LoopMode.until_release else 4)
        )
    expected = np.zeros((48000, 3))
    prefix = (
        [0, 0.05, 0.14, 0] if mode == LoopMode.until_release else [0, 0.05, 0.07, 0.06]
    )
    expected[:4, 0] = prefix
    expected[:4, 1] = -0.5 * np.array(prefix)
    check_audio(tmp_path / "sample-minimum-hold.wav", np.concatenate(chunks), expected)
    assert renderer.complete
    saved = renderer.source
    assert not np.any(renderer.render(sample, 48000))
    assert renderer.source == saved


@pytest.mark.parametrize("mode", ["while_held", "one_shot"])
def test_source_exhaustion_and_one_shot_release_retire_at_exact_frames(
    tmp_path: Path, mode: str
) -> None:
    raw = sample_score(frames=4).model_dump(mode="json")
    raw["body"]["settings"]["playback"]["mode"] = mode
    document = instrument.SampleInstrumentScore.model_validate(raw)
    prepared = sample_instrument.prepare(document, {"asset": np.ones((4, 2))})
    actions = trace.prepare(
        document.body,
        [
            onset(pitch=440).model_copy(update={"controls": {}}),
            Release(tick=2, ordinal=0, part="main", trigger_id="note"),
        ],
        seed=0,
    ).actions
    renderer = sample_instrument.OfflineSampler(prepared)
    chunks: list[np.ndarray] = []
    for start, end in pairwise([0, 1, 2, 3, 4, 48000]):
        chunks.append(
            renderer.advance([a for a in actions if start <= a.tick < end], start, end)
        )
        if end >= (3 if mode == "while_held" else 4):
            assert renderer.snapshot().voices == []
    expected = np.zeros((48000, 2))
    expected[: 2 if mode == "while_held" else 4] = [1, 1.25]
    check_audio(tmp_path / "source-exhaustion.wav", np.concatenate(chunks), expected)


def test_preparation_shares_audio_but_restore_rejects_different_decoded_content() -> (
    None
):
    raw = sample_score(frames=4).model_dump(mode="json")
    raw["body"]["slots"].append({**raw["body"]["slots"][0], "name": "other"})
    document = instrument.SampleInstrumentScore.model_validate(raw)
    audio = np.ones((4, 2))
    prepared = sample_instrument.prepare(document, {"asset": audio})
    assert prepared.samples["sample"].samples is prepared.samples["other"].samples
    audio[:] = 0
    different = sample_instrument.prepare(document, {"asset": audio})
    snapshot = sample_instrument.OfflineSampler(prepared).snapshot()
    with pytest.raises(EngineError, match="decoded audio"):
        sample_instrument.OfflineSampler(different).restore(snapshot)


@pytest.mark.parametrize("field,value", [("delay_seconds", 0.5), ("offset_frames", 1)])
def test_unsupported_sample_start_variation_is_not_silently_ignored(
    field: str, value: float | int
) -> None:
    raw = sample_score(frames=4).model_dump(mode="json")
    raw["body"]["slots"][0]["variation"][field] = value
    document = instrument.SampleInstrumentScore.model_validate(raw)
    with pytest.raises(EngineError, match="Delayed or offset"):
        sample_instrument.prepare(document, {"asset": np.ones((4, 2))})


def test_prepared_replacement_and_transport_stop_apply_before_the_sample(
    tmp_path: Path,
) -> None:
    raw = sample_score(frames=100).model_dump(mode="json")
    raw["body"]["settings"]["voice_policy"] = {
        "same_key": "replace",
        "maximum_voices": 8,
    }
    raw["body"]["settings"]["playback"]["mode"] = "one_shot"
    document = instrument.SampleInstrumentScore.model_validate(raw)
    audio = np.column_stack([np.arange(100) / 100, np.zeros(100)])
    prepared = sample_instrument.prepare(document, {"asset": audio})
    actions = trace.prepare(
        document.body,
        [
            onset(pitch=440).model_copy(update={"controls": {}}),
            onset(2, "other", pitch=880).model_copy(update={"controls": {}}),
        ],
        seed=0,
    ).actions
    last = next(a for a in reversed(actions) if isinstance(a, trace.VoiceStart))
    actions.append(
        VoiceRetirement(
            tick=5,
            ordinal=0,
            voice_id=last.voice_id,
            cause=RetirementCause.transport_stop,
            action="stop",
        )
    )
    renderer = sample_instrument.OfflineSampler(prepared)
    actual = renderer.advance(actions, 0, 48000)
    expected = np.zeros_like(actual)
    expected[:5, 0] = [0, 0.01, 0, 0.02, 0.04]
    expected[:, 1] = expected[:, 0] / 4
    check_audio(tmp_path / "sample-stop.wav", actual, expected)
    assert renderer.snapshot().voices == []


def test_resolved_variation_and_static_tuning_are_applied_once(tmp_path: Path) -> None:
    raw = sample_score().model_dump(mode="json")
    raw["body"]["settings"]["processing"] = {"tuning_cents": 100, "volume_db": 3}
    slot = raw["body"]["slots"][0]
    slot["processing"] = {"tuning_cents": 200, "volume_db": -6}
    slot["variation"] = {"pitch_cents": 300, "gain_db": 3}
    document = instrument.SampleInstrumentScore.model_validate(raw)
    ramp = np.arange(96000, dtype=np.float64) / 96000
    prepared = sample_instrument.prepare(
        document, {"asset": np.column_stack([ramp, 1 - ramp])}
    )
    actions = trace.prepare(
        document.body,
        [
            onset(pitch=220).model_copy(update={"controls": {}}),
            Release(tick=10000, ordinal=0, part="main", trigger_id="note"),
        ],
        seed=123,
    ).actions
    start = next(a for a in actions if isinstance(a, trace.VoiceStart))
    actual = sample_instrument.OfflineSampler(prepared).advance(actions, 0, 48000)
    progress = (
        np.arange(10000) * 0.5 * 2 ** ((300 + start.variation.pitch_cents) / 1200)
    )
    expected = np.zeros_like(actual)
    expected[:10000] = np.column_stack(
        [progress / 96000, 1 - 0.75 * progress / 96000]
    ) * 10 ** ((-3 + start.variation.gain_db) / 20)
    check_audio(tmp_path / "sample-variation.wav", actual, expected)


def test_instrument_ramps_continue_in_silence_and_multiply_independent_trigger_gains(
    tmp_path: Path,
) -> None:
    raw = sample_score().model_dump(mode="json")
    shared = dynamic_score(scope="instrument").body
    local = dynamic_score().body.voices[0]
    raw["body"]["settings"].update(
        {
            "controls": {n: c.model_dump() for n, c in shared.controls.items()},
            "bindings": [b.model_dump() for b in shared.voices[0].bindings],
            "modulation": shared.voices[0].modulation.model_dump(),
        }
    )
    raw["body"]["slots"][0].update(
        {
            "bindings": [b.model_dump() for b in local.bindings],
            "modulation": local.modulation.model_dump(),
        }
    )
    document = instrument.SampleInstrumentScore.model_validate(raw)
    prepared = sample_instrument.prepare(document, {"asset": np.ones((96000, 2))})
    actions = trace.prepare(
        document.body,
        [
            change(0, 1, scope="instrument"),
            onset(120, "first", pitch=440, gain=0.25),
            onset(180, "second", pitch=440, gain=0.75),
            Release(tick=300, ordinal=0, part="main", trigger_id="first"),
            Release(tick=300, ordinal=1, part="main", trigger_id="second"),
        ],
        seed=0,
    ).actions
    actual = sample_instrument.OfflineSampler(prepared).advance(actions, 0, 48000)
    frames = np.arange(48000)
    gain = (0.5 + 0.5 * np.minimum(frames / 240, 1)) * np.where(
        frames < 120, 0, np.where(frames < 180, 0.25, np.where(frames < 300, 1, 0))
    )
    expected = np.column_stack([gain, 1.25 * gain])
    check_audio(tmp_path / "shared-and-trigger-controls.wav", actual, expected)
    independent = trace.prepare(
        document.body, [onset(gain=1, pitch=440)], seed=0
    ).actions
    isolated = sample_instrument.OfflineSampler(prepared).advance(independent, 0, 48000)
    check_audio(
        tmp_path / "independent-instance.wav",
        isolated,
        np.tile([0.5, 0.625], (48000, 1)),
    )


@pytest.mark.parametrize(
    "direction,prefix", [("backward", [3, 2, 1, 0]), ("mirror", [0, 1, 2, 3, 2, 1, 0])]
)
def test_unpitched_samples_inherit_direction_without_inventing_pitch(
    tmp_path: Path, direction: str, prefix: list[float]
) -> None:
    raw = sample_score(frames=4).model_dump(mode="json")
    raw["body"]["settings"]["playback"]["direction"] = direction
    raw["body"]["slots"][0]["mapping"].update(
        {"pitch_tracking": False, "reference_pitch_hz": None}
    )
    document = instrument.SampleInstrumentScore.model_validate(raw)
    prepared = sample_instrument.prepare(
        document,
        {"asset": np.column_stack([np.arange(4, dtype=np.float64), np.zeros(4)])},
    )
    actions = trace.prepare(
        document.body,
        [onset().model_copy(update={"controls": {}, "pitch_hz": None})],
        seed=0,
    ).actions
    renderer = sample_instrument.OfflineSampler(prepared)
    actual = renderer.advance(actions, 0, 48000)
    expected = np.zeros_like(actual)
    expected[: len(prefix), 0] = prefix
    expected[:, 1] = expected[:, 0] / 4
    check_audio(tmp_path / "unpitched-direction.wav", actual, expected)
    assert renderer.snapshot().voices == []


def test_preparation_rejects_filters_before_rendering() -> None:
    raw = sample_score(frames=4).model_dump(mode="json")
    raw["body"]["settings"]["processing"]["filters"] = [
        {"name": "low", "response": "lowpass", "cutoff_hz": 1000}
    ]
    document = instrument.SampleInstrumentScore.model_validate(raw)
    with pytest.raises(EngineError, match="Only sample volume and tuning"):
        sample_instrument.prepare(document, {"asset": np.ones((4, 2))})


@pytest.mark.parametrize("loop", [False, True])
def test_exhausted_sample_does_not_evaluate_later_voice_parameters(
    tmp_path: Path,
    loop: bool,
) -> None:
    raw = sample_score(frames=2).model_dump(mode="json")
    dynamic = dynamic_score(smoothing="1/12000").body
    raw["body"]["settings"]["controls"] = {
        n: c.model_dump() for n, c in dynamic.controls.items()
    }
    slot = raw["body"]["slots"][0]
    slot["modulation"] = dynamic.voices[0].modulation.model_dump(mode="json")
    slot["bindings"] = [b.model_dump(mode="json") for b in dynamic.voices[0].bindings]
    slot["modulation"]["parameters"][0]["maximum"] = 1.25
    slot["modulation"]["routes"][0]["operation"] = "add"
    if loop:
        raw["body"]["slices"][0]["loop"] = {"start_frame": 0, "end_frame": 2}
        raw["body"]["settings"]["envelope"]["release"] = [
            {"duration": "1/2", "target": 0}
        ]
    document = instrument.SampleInstrumentScore.model_validate(raw)
    prepared = sample_instrument.prepare(document, {"asset": np.ones((2, 2))})
    actions = trace.prepare(
        document.body,
        [
            onset(pitch=440, gain=0),
            change(0, 1),
            *(
                [Release(tick=2, ordinal=0, part="main", trigger_id="note")]
                if loop
                else []
            ),
        ],
        seed=0,
    ).actions
    expected = np.zeros((48000, 2))
    expected[:2] = [[1, 1.25], [1.25, 1.5625]]
    renderer = sample_instrument.OfflineSampler(prepared)
    actual = renderer.advance(actions, 0, 48000)
    check_audio(tmp_path / "exhausted-control-ramp.wav", actual, expected)
    assert renderer.snapshot().voices == []
