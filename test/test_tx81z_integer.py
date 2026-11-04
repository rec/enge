from functools import cache
from pathlib import Path

import numpy as np
import pytest
from test_synth import check_audio

from enge.yamaha import TX81ZState, tx81z_operator_samples, tx81z_wave_samples


@cache
def reference_tables() -> tuple[np.ndarray, np.ndarray]:
    with np.load(Path(__file__).parent / "data/tx81z-opz-core.npz") as reference:
        return reference["waveforms"], reference["power"]


def opz_carrier(
    frequency: float | np.ndarray,
    gain: float | np.ndarray = 0.25,
    frames: int = 48000,
    waveform: int = 0,
    phase: int = 0,
) -> np.ndarray:
    """Independent carrier oracle using tables captured from pinned ymfm."""
    steps = np.broadcast_to(
        np.rint(np.asarray(frequency) * 2**20 / 48000), (frames,)
    ).astype(np.int64)
    phases = (phase + np.concatenate([[0], np.cumsum(steps[:-1])])) >> 10
    attenuation = np.rint(-64 * np.log2(np.maximum(gain, 2.0**-64))).astype(np.int64)
    waves, power = reference_tables()
    entries = waves[waveform, phases & 1023]
    levels = (entries & 32767).astype(np.int64) + 4 * attenuation
    values = power[levels & 255].astype(np.int64) >> (levels >> 8)
    return np.where(entries & 32768, -values, values) / 8192


@pytest.mark.parametrize("case", range(128))
def test_integer_core_matches_pinned_opz_and_replays(tmp_path: Path, case: int) -> None:
    with np.load(Path(__file__).parent / "data/tx81z-opz-core.npz") as reference:
        algorithm, feedback, profile = map(int, reference["parameters"][case])
        expected = reference[f"case{case}"].astype(float)[:, None] / 8192
    steps = [24000, 19200, 14400, 9600] if profile else [9600] * 4
    levels = np.array([48, 32, 16, 0] if profile else [0] * 4)
    waves = [7, 4, 1, 0] if profile else [0] * 4
    frequencies = np.tile(np.array(steps) * 48000 / 2**20, (48000, 1))
    gains = np.tile(2 ** (-levels.astype(float) / 8), (48000, 1))
    first, state = tx81z_operator_samples(
        algorithm,
        waves,
        TX81ZState(),
        frequencies[:17003],
        gains[:17003],
        feedback=feedback,
    )
    saved = TX81ZState.model_validate_json(state.model_dump_json())
    tail, final = tx81z_operator_samples(
        algorithm, waves, state, frequencies[17003:], gains[17003:], feedback=feedback
    )
    actual = np.concatenate([first, tail])
    check_audio(tmp_path / "opz-core.wav", actual, expected)
    np.testing.assert_array_equal(actual, expected)
    replay, replay_state = tx81z_operator_samples(
        algorithm, waves, saved, frequencies[17003:], gains[17003:], feedback=feedback
    )
    np.testing.assert_array_equal(tail, replay)
    assert final == replay_state


@pytest.mark.parametrize("waveform", range(8))
def test_quantized_waveform_matches_all_reference_cells(
    tmp_path: Path, waveform: int
) -> None:
    phases = np.arange(48000) / 1024 - 24
    waves, power = reference_tables()
    entries = waves[waveform, np.arange(48000) % 1024]
    levels = entries & 32767
    values = power[levels & 255] >> (levels >> 8)
    expected = np.where(entries & 32768, -values.astype(int), values) / 8192
    actual = tx81z_wave_samples(waveform, phases)
    check_audio(tmp_path / "opz-wave.wav", actual[:, None], expected[:, None])
    np.testing.assert_array_equal(actual, expected)
