import numpy as np
import pytest
from ufor.envelope import Envelope, Segment
from ufor.events import Release, Trigger
from ufor.synth import SynthInstrumentScore
from ufor.synth_trace import prepare as prepare_trace

from enge.synth import EngineError, OfflineSynth, prepare, waveform_samples


def score() -> SynthInstrumentScore:
    return SynthInstrumentScore.model_validate(
        {
            "name": "triangle",
            "title": "Triangle",
            "kind": "synth_instrument",
            "timebases": [
                {"name": "output", "kind": "physical", "rate": {"numerator": 48000}}
            ],
            "body": {
                "voices": [
                    {
                        "name": "triangle",
                        "mapping": {
                            "lowest_key": 0,
                            "highest_key": 127,
                            "reference_pitch_hz": 440,
                        },
                        "channels": [
                            {"input": "mono", "output": "left", "gain": 1},
                        ],
                        "oscillator": {"waveform": "triangle"},
                    }
                ]
            },
            "inputs": [
                {
                    "name": "performance",
                    "stream": {
                        "family": "event",
                        "timebase": "output",
                        "kinds": ["trigger", "release", "control_change"],
                    },
                    "binding": {"performance": True},
                }
            ],
            "outputs": [
                {
                    "name": "audio",
                    "stream": {
                        "family": "sampled",
                        "quantity": "audio_amplitude",
                        "unit": "full_scale",
                        "timebase": "output",
                        "channels": ["left", "right"],
                    },
                    "binding": {"audio": True},
                }
            ],
        }
    )


def test_offline_synth_renders_one_second_with_explicit_channel_routes() -> None:
    document = score()
    trace = prepare_trace(
        document.body,
        [
            Trigger(
                tick=0,
                ordinal=0,
                part="main",
                trigger_id="note-a",
                key=69,
                pitch_hz=440,
            )
        ],
        seed=42,
    )
    output = OfflineSynth(prepare(document)).advance(trace.actions, 0, 48_000)
    assert output.shape == (48_000, 2)
    assert output[0, 0] == -1
    assert np.all(output[:, 1] == 0)


def test_offline_synth_is_partition_invariant_and_restorable() -> None:
    document = score()
    trace = prepare_trace(
        document.body,
        [
            Trigger(
                tick=0,
                ordinal=0,
                part="main",
                trigger_id="note-a",
                key=69,
                pitch_hz=440,
            ),
            Release(tick=24_000, ordinal=0, part="main", trigger_id="note-a"),
        ],
        seed=42,
    )
    definition = prepare(document)
    whole = OfflineSynth(definition).advance(trace.actions, 0, 48_000)
    split = OfflineSynth(definition)
    first = split.advance([a for a in trace.actions if a.tick < 24_000], 0, 24_000)
    snapshot = split.snapshot()
    second = split.advance(
        [a for a in trace.actions if a.tick >= 24_000], 24_000, 48_000
    )
    restored = OfflineSynth(definition)
    restored.restore(snapshot)
    replay = restored.advance(
        [a for a in trace.actions if a.tick >= 24_000], 24_000, 48_000
    )
    np.testing.assert_allclose(np.concatenate([first, second]), whole, atol=0)
    np.testing.assert_allclose(replay, second, atol=0)


def test_offline_synth_requires_resolved_pitch() -> None:
    document = score()
    trace = prepare_trace(
        document.body,
        [Trigger(tick=0, ordinal=0, part="main", trigger_id="note-a", key=69)],
        seed=42,
    )
    with pytest.raises(EngineError, match="pitch_hz"):
        OfflineSynth(prepare(document)).advance(trace.actions, 0, 1)


def test_offline_synth_defers_release_until_the_voice_minimum_hold() -> None:
    document = score()
    voice = document.body.voices[0].model_copy(
        update={
            "minimum_hold_seconds": 1,
            "envelope": Envelope(
                segments=[Segment(duration=1, target=1)],
                release=[Segment(duration=1, target=0)],
            ),
        }
    )
    instrument = document.body.model_copy(update={"voices": [voice]})
    document = document.model_copy(update={"body": instrument})
    trace = prepare_trace(
        instrument,
        [
            Trigger(
                tick=0,
                ordinal=0,
                part="main",
                trigger_id="note-a",
                key=69,
                pitch_hz=440,
            ),
            Release(tick=24_000, ordinal=0, part="main", trigger_id="note-a"),
        ],
        seed=42,
    )
    renderer = OfflineSynth(prepare(document))
    renderer.advance(trace.actions, 0, 48_000)
    snapshot = renderer.snapshot()
    assert snapshot.voices[0].release_frame == 48_000
    renderer.advance([], 48_000, 96_000)
    assert renderer.snapshot().voices == []


def test_offline_synth_can_use_transport_synchronized_offset_pitch() -> None:
    document = score()
    voice = document.body.voices[0].model_copy(
        update={"frequency_offset_hz": 1, "synchronize_oscillator": True}
    )
    instrument = document.body.model_copy(update={"voices": [voice]})
    document = document.model_copy(update={"body": instrument})
    trace = prepare_trace(
        instrument,
        [
            Trigger(
                tick=1,
                ordinal=0,
                part="main",
                trigger_id="note-a",
                key=69,
                pitch_hz=440,
            )
        ],
        seed=42,
    )
    output = OfflineSynth(prepare(document)).advance(trace.actions, 0, 2)
    assert output[1, 0] == pytest.approx(-1 + 4 * 441 / 48_000)


def test_waveform_samples_uses_tuneys_start_length_and_period_convention() -> None:
    oscillator = score().body.voices[0].oscillator
    actual = waveform_samples(oscillator, start=2, length=4, period=8)
    np.testing.assert_allclose(actual, [0, 0.5, 1, 0.5], atol=0)
