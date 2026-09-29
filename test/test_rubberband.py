import wave
from pathlib import Path

import numpy as np
import pytest

from enge import _native


@pytest.fixture
def rubberband() -> object:
    if not hasattr(_native, "rubberband_stretch"):
        pytest.skip("build with maturin develop --features rubberband")
    return _native


def audio(frames: int = 48000, channels: int = 2) -> np.ndarray:
    time = np.arange(frames) / 48000
    return np.column_stack(
        [
            np.sin(2 * np.pi * frequency * time)
            for frequency in range(220, 220 + 110 * channels, 110)
        ]
    )


def write_wav(path: Path, values: np.ndarray) -> None:
    with wave.open(str(path), "wb") as output:
        output.setnchannels(values.shape[1])
        output.setsampwidth(4)
        output.setframerate(48000)
        peak = max(1.0, float(np.max(np.abs(values))))
        output.writeframes((values / peak * (2**31 - 1)).astype("<i4").tobytes())


@pytest.mark.parametrize("ratio,frames", [(0.5, 24000), (1.0, 48000), (2.0, 96000)])
def test_offline_rubberband_changes_time_without_changing_channel_layout(
    rubberband: object, ratio: float, frames: int, tmp_path: Path
) -> None:
    source = audio()
    actual = rubberband.rubberband_stretch(source, ratio, 1.0)

    assert actual.shape == (frames, 2)
    assert actual.dtype == np.float64
    assert np.all(np.isfinite(actual))
    assert np.max(np.abs(actual)) > 0.1
    write_wav(tmp_path / f"offline-time-{ratio}.wav", actual)


@pytest.mark.parametrize("pitch", [0.5, 1.0, 2.0])
def test_offline_rubberband_changes_pitch_without_changing_duration(
    rubberband: object, pitch: float, tmp_path: Path
) -> None:
    actual = rubberband.rubberband_stretch(audio(), 1.0, pitch)

    assert actual.shape == (48000, 2)
    assert np.all(np.isfinite(actual))
    write_wav(tmp_path / f"offline-pitch-{pitch}.wav", actual)


def test_offline_rubberband_identity_preserves_audible_signal(
    rubberband: object,
) -> None:
    source = audio(channels=1)
    actual = rubberband.rubberband_stretch(source, 1.0, 1.0)

    assert actual.shape == source.shape
    assert np.corrcoef(source[:, 0], actual[:, 0])[0, 1] > 0.99


@pytest.mark.parametrize("ratio,pitch", [(0.0, 1.0), (1.0, 0.0), (-1.0, 1.0)])
def test_offline_rubberband_rejects_nonpositive_ratios(
    rubberband: object, ratio: float, pitch: float
) -> None:
    with pytest.raises(ValueError, match="positive finite"):
        rubberband.rubberband_stretch(audio(), ratio, pitch)


def test_offline_rubberband_rejects_noncontiguous_audio(rubberband: object) -> None:
    with pytest.raises(ValueError, match="contiguous"):
        rubberband.rubberband_stretch(audio()[::2], 1.0, 1.0)


def test_live_rubberband_identity_streams_fixed_blocks(
    rubberband: object, tmp_path: Path
) -> None:
    shifter = rubberband.RubberBandLiveShifter(48000, 2, 1.0)
    source = audio(shifter.block_size * 94)
    blocks = [
        shifter.shift(source[start : start + shifter.block_size])
        for start in range(0, len(source), shifter.block_size)
    ]
    actual = np.vstack(blocks)

    assert shifter.block_size > 0
    assert shifter.start_delay > 0
    assert actual.shape == source.shape
    assert np.all(np.isfinite(actual))
    assert np.max(np.abs(actual[: shifter.start_delay])) < 1e-3
    assert np.max(np.abs(actual[shifter.start_delay :])) > 0.1
    write_wav(tmp_path / "live-identity.wav", actual)


def test_live_rubberband_shifts_pitch_without_changing_block_size(
    rubberband: object,
) -> None:
    shifter = rubberband.RubberBandLiveShifter(48000, 1, 2.0)
    block = audio(shifter.block_size, 1)

    actual = shifter.shift(block)

    assert actual.shape == block.shape
    assert np.all(np.isfinite(actual))


def test_live_rubberband_rejects_a_wrong_block_size(rubberband: object) -> None:
    shifter = rubberband.RubberBandLiveShifter(48000, 2, 1.0)

    with pytest.raises(ValueError, match="block size"):
        shifter.shift(audio(shifter.block_size - 1))
