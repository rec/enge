from pathlib import Path
from typing import Literal

import numpy as np
import pytest
from test_synth import check_audio
from ufor.oscillator import Oscillator, Waveform

from enge.synth import OscillatorState, oscillator_samples


@pytest.mark.parametrize("waveform", list(Waveform))
def test_phase_survives_long_pitch_changes_and_restoration(
    tmp_path: Path, waveform: Waveform, backend: Literal["numpy", "native"]
) -> None:
    frames = 480000
    rate = 48000
    origin = 2**53 + 125
    # Exact integer units provide an independent oracle for repeated pitch
    # ramps, including every discontinuity, without cumulative float error.
    increments = 51328 + np.arange(frames, dtype=np.int64) % 8192
    increments[360000:] = 51328
    frequencies = increments / 512
    initial = origin * 51328 % (rate * 512)
    positions = (initial + np.concatenate(([0], np.cumsum(increments[:-1])))) % (
        rate * 512
    )
    phase = positions / (rate * 512)
    if waveform == Waveform.sine:
        expected = np.sin(2 * np.pi * phase)
    elif waveform == Waveform.square:
        expected = np.where(positions < rate * 256, 1.0, -1.0)
    else:
        expected = np.where(phase < 0.5, 4 * phase - 1, 3 - 4 * phase)

    oscillator = Oscillator(waveform=waveform)
    state = OscillatorState.at_frame(origin, 100.25, rate)
    whole, ending = oscillator_samples(oscillator, state, frequencies, rate, backend)
    chunks: list[np.ndarray] = []
    start = 0
    while start < frames:
        end = min(frames, start + (64, 997, 1, 1024)[len(chunks) % 4])
        if start < 240001:
            end = min(end, 240001)
        chunk, state = oscillator_samples(
            oscillator, state, frequencies[start:end], rate, backend
        )
        chunks.append(chunk)
        start = end
        if end == 240001:
            state = OscillatorState.model_validate_json(state.model_dump_json())
    split = np.concatenate(chunks)
    check_audio(tmp_path / "long-phase.wav", whole[:, None], expected[:, None])
    check_audio(tmp_path / "restored-phase.wav", split[:, None], expected[:, None])
    assert state == ending
