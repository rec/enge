import wave
from pathlib import Path
from typing import Literal

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


def test_offline_synth_renders_one_second_with_explicit_channel_routes(
    tmp_path: Path,
    backend: Literal["numpy", "native"],
) -> None:
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
    output = OfflineSynth(prepare(document), backend).advance(trace.actions, 0, 48_000)
    assert output.shape == (48_000, 2)
    assert output[0, 0] == -1
    assert np.all(output[:, 1] == 0)
    phase = (np.arange(48000) * 440 / 48000) % 1
    expected = np.zeros_like(output)
    expected[:, 0] = np.where(phase < 0.5, 4 * phase - 1, 3 - 4 * phase)
    check_audio(tmp_path / "triangle.wav", output, expected)


def test_offline_synth_is_partition_invariant_and_restorable(
    tmp_path: Path, backend: Literal["numpy", "native"]
) -> None:
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
    whole = OfflineSynth(definition, backend).advance(trace.actions, 0, 48_000)
    split = OfflineSynth(definition, backend)
    first = split.advance([a for a in trace.actions if a.tick < 24_000], 0, 24_000)
    snapshot = split.snapshot()
    second = split.advance(
        [a for a in trace.actions if a.tick >= 24_000], 24_000, 48_000
    )
    restored = OfflineSynth(definition, backend)
    restored.restore(snapshot)
    replay = restored.advance(
        [a for a in trace.actions if a.tick >= 24_000], 24_000, 48_000
    )
    np.testing.assert_allclose(np.concatenate([first, second]), whole, atol=0)
    np.testing.assert_allclose(replay, second, atol=0)
    check_audio(tmp_path / "partition.wav", np.concatenate([first, second]), whole)
    check_audio(tmp_path / "restored.wav", np.concatenate([first, replay]), whole)


def test_offline_synth_requires_resolved_pitch(
    backend: Literal["numpy", "native"],
) -> None:
    document = score()
    trace = prepare_trace(
        document.body,
        [Trigger(tick=0, ordinal=0, part="main", trigger_id="note-a", key=69)],
        seed=42,
    )
    with pytest.raises(EngineError, match="pitch_hz"):
        OfflineSynth(prepare(document), backend).advance(trace.actions, 0, 1)


def test_offline_synth_defers_release_until_the_voice_minimum_hold(
    tmp_path: Path,
    backend: Literal["numpy", "native"],
) -> None:
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
    renderer = OfflineSynth(prepare(document), backend)
    first = renderer.advance(trace.actions, 0, 48_000)
    snapshot = renderer.snapshot()
    assert snapshot.voices[0].renderer.release_frame == 48_000
    second = renderer.advance([], 48_000, 96_000)
    assert renderer.snapshot().voices == []
    frames = np.arange(96000)
    phase = frames * 440 / 48000 % 1
    expected = np.zeros((96000, 2))
    expected[:, 0] = np.where(phase < 0.5, 4 * phase - 1, 3 - 4 * phase) * np.where(
        frames < 48000, frames / 48000, 2 - frames / 48000
    )
    check_audio(
        tmp_path / "minimum-hold.wav", np.concatenate([first, second]), expected
    )


def test_offline_synth_can_use_transport_synchronized_offset_pitch(
    tmp_path: Path,
    backend: Literal["numpy", "native"],
) -> None:
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
    output = OfflineSynth(prepare(document), backend).advance(trace.actions, 0, 48000)
    assert output[1, 0] == pytest.approx(-1 + 4 * 441 / 48_000)
    phase = np.arange(48000) * 441 / 48000 % 1
    expected = np.zeros_like(output)
    expected[1:, 0] = np.where(phase[1:] < 0.5, 4 * phase[1:] - 1, 3 - 4 * phase[1:])
    check_audio(tmp_path / "synchronized.wav", output, expected)


def test_waveform_samples_uses_tuneys_start_length_and_period_convention(
    tmp_path: Path,
) -> None:
    oscillator = score().body.voices[0].oscillator
    actual = waveform_samples(oscillator, start=2, length=4, period=8)
    np.testing.assert_allclose(actual, [0, 0.5, 1, 0.5], atol=0)
    actual = waveform_samples(oscillator, start=2, length=48000, period=8)[:, None]
    expected = np.tile([0, 0.5, 1, 0.5, 0, -0.5, -1, -0.5], 6000)[:, None]
    check_audio(tmp_path / "waveform-buffer.wav", actual, expected)


def check_audio(path: Path, actual: np.ndarray, expected: np.ndarray) -> None:
    """Keep one-second WAV regressions and compare before PCM quantization."""
    assert actual.shape == expected.shape
    assert actual.shape[0] >= 48000
    assert actual.dtype == np.float64
    assert np.all(np.isfinite(actual))
    peak = max(1.0, float(np.max(np.abs(actual))), float(np.max(np.abs(expected))))
    for name, values in (("actual", actual), ("expected", expected)):
        with wave.open(str(path.with_stem(f"{path.stem}-{name}")), "wb") as output:
            output.setnchannels(values.shape[1])
            output.setsampwidth(4)
            output.setframerate(48000)
            output.writeframes((values / peak * (2**31 - 1)).astype("<i4").tobytes())
    np.testing.assert_allclose(actual, expected, atol=1e-10, rtol=1e-9)
