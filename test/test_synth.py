import numpy as np
import pytest
from ufor.events import Release, Trigger
from ufor.synth import SynthInstrumentScore
from ufor.synth_trace import prepare as prepare_trace

from enge.synth import EngineError, OfflineSynth, prepare


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
