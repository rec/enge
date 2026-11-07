import wave
from pathlib import Path

import numpy as np
import pytest

from enge import _native
from enge.rubberband import RealtimeRubberBand
from enge.synth import EngineError


@pytest.fixture
def rubberband(request: pytest.FixtureRequest) -> object:
    if not hasattr(_native, "rubberband_stretch"):
        if request.config.getoption("--require-rubberband"):
            pytest.fail("Rubber Band tests require the rubberband build feature")
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
    assert len(values) >= 48000
    with wave.open(str(path), "wb") as output:
        output.setnchannels(values.shape[1])
        output.setsampwidth(4)
        output.setframerate(48000)
        peak = max(1.0, float(np.max(np.abs(values))))
        output.writeframes((values / peak * (2**31 - 1)).astype("<i4").tobytes())


@pytest.mark.parametrize("ratio,frames", [(0.5, 48000), (1.0, 96000), (2.0, 192000)])
def test_offline_rubberband_changes_time_without_changing_channel_layout(
    rubberband: object, ratio: float, frames: int, tmp_path: Path
) -> None:
    source = audio(96000)
    actual = rubberband.rubberband_stretch(source, ratio, 1.0)

    assert actual.shape == (frames, 2)
    assert actual.dtype == np.float64
    assert np.all(np.isfinite(actual))
    assert np.max(np.abs(actual)) > 0.1
    assert abs(dominant_frequency(actual[:, 0]) - 220) < 2
    assert abs(dominant_frequency(actual[:, 1]) - 330) < 2
    write_wav(tmp_path / f"offline-time-{ratio}.wav", actual)


@pytest.mark.parametrize("pitch", [0.5, 1.0, 2.0])
def test_offline_rubberband_changes_pitch_without_changing_duration(
    rubberband: object, pitch: float, tmp_path: Path
) -> None:
    actual = rubberband.rubberband_stretch(audio(), 1.0, pitch)

    assert actual.shape == (48000, 2)
    assert np.all(np.isfinite(actual))
    assert abs(dominant_frequency(actual[:, 0]) - 220 * pitch) < 2
    assert abs(dominant_frequency(actual[:, 1]) - 330 * pitch) < 2
    write_wav(tmp_path / f"offline-pitch-{pitch}.wav", actual)


def test_offline_rubberband_identity_preserves_audible_signal(
    rubberband: object,
    tmp_path: Path,
) -> None:
    source = audio(channels=1)
    actual = rubberband.rubberband_stretch(source, 1.0, 1.0)

    assert actual.shape == source.shape
    assert np.corrcoef(source[:, 0], actual[:, 0])[0, 1] > 0.99
    write_wav(tmp_path / "offline-identity.wav", actual)


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
    tmp_path: Path,
) -> None:
    shifter = rubberband.RubberBandLiveShifter(48000, 1, 2.0)
    source = audio(shifter.block_size * 100, 1)
    actual = np.vstack(
        [
            shifter.shift(source[s : s + shifter.block_size])
            for s in range(0, len(source), shifter.block_size)
        ]
    )
    assert actual.shape == source.shape
    assert np.all(np.isfinite(actual))
    assert abs(dominant_frequency(actual[shifter.start_delay :, 0]) - 440) < 2
    write_wav(tmp_path / "live-octave.wav", actual)


def test_live_rubberband_rejects_a_wrong_block_size(rubberband: object) -> None:
    shifter = rubberband.RubberBandLiveShifter(48000, 2, 1.0)

    with pytest.raises(ValueError, match="block size"):
        shifter.shift(audio(shifter.block_size - 1))


@pytest.mark.parametrize("ratio", [0.5, 1.0, 2.0])
def test_realtime_rubberband_meets_finite_segment_boundary(
    rubberband: object, ratio: float, tmp_path: Path
) -> None:
    source = audio(channels=1)
    renderer = RealtimeRubberBand(
        sample_rate=48000,
        channels=1,
        ratio=ratio,
        source_frames=len(source),
        block_frames=512,
        max_buffer_frames=100000,
    )
    blocks = []
    while not renderer.complete:
        start = renderer.source_received
        end = min(start + renderer.block_frames, len(source))
        blocks.append(renderer.advance(source[start:end]))
    result = np.vstack(blocks)
    audible = result[
        renderer.startup_frames : renderer.startup_frames + renderer.output_frames
    ]

    assert renderer.source_received == len(source)
    assert result.shape[1] == 1
    assert len(audible) == round(len(source) * ratio)
    assert np.allclose(result[: renderer.startup_frames], 0)
    assert np.max(np.abs(audible)) > 0.5
    assert renderer.buffered_frames == 0
    assert renderer.complete
    assert renderer.underrun_frames <= 512
    center = audible[len(audible) // 4 : 3 * len(audible) // 4, 0]
    cycles = np.count_nonzero((center[:-1] <= 0) & (center[1:] > 0))
    assert 200 < cycles * 48000 / len(center) < 240
    write_wav(tmp_path / f"realtime-time-{ratio}.wav", result)
    with pytest.raises(EngineError, match="complete"):
        renderer.advance(np.empty((0, 1)))


def test_realtime_rubberband_rejects_buffer_growth_at_hard_limit(
    rubberband: object,
) -> None:
    source = audio(channels=1)
    renderer = RealtimeRubberBand(48000, 1, 2.0, len(source), 512, 512)

    with pytest.raises(EngineError, match="buffer limit"):
        while not renderer.complete:
            start = renderer.source_received
            renderer.advance(source[start : start + 512])

    assert renderer.buffered_frames == 0
    assert renderer.complete


def test_realtime_rubberband_requires_each_input_clock_block(
    rubberband: object,
) -> None:
    renderer = RealtimeRubberBand(48000, 1, 1.0, 48000, 512, 4096)

    with pytest.raises(EngineError, match="next clock block"):
        renderer.advance(audio(256, 1))


def test_native_realtime_stretcher_processes_stereo_incrementally(
    rubberband: object, tmp_path: Path
) -> None:
    source = audio()
    stretcher = rubberband.RubberBandRealtimeStretcher(48000, 2, 1.0, 1.0, 512)
    blocks = [
        stretcher.process(source[start : start + 512], start + 512 >= len(source))
        for start in range(0, len(source), 512)
    ]
    actual = np.vstack(blocks)

    assert stretcher.start_delay > 0
    assert actual.shape == source.shape
    assert np.all(np.isfinite(actual))
    assert np.corrcoef(source[:, 0], actual[:, 0])[0, 1] > 0.7
    write_wav(tmp_path / "realtime-native-stereo.wav", actual)
    with pytest.raises(ValueError, match="complete"):
        stretcher.process(source[:512], False)


def test_native_realtime_stretcher_rejects_oversized_blocks(
    rubberband: object,
) -> None:
    stretcher = rubberband.RubberBandRealtimeStretcher(48000, 1, 1.0, 1.0, 512)

    with pytest.raises(ValueError, match="block"):
        stretcher.process(audio(513, 1), False)


@pytest.mark.parametrize("frames,expected", [(96001, 48000), (96003, 48002)])
def test_offline_duration_rounds_half_frames_to_even(
    rubberband: object, frames: int, expected: int, tmp_path: Path
) -> None:
    result = rubberband.rubberband_stretch(audio(frames), 0.5, 1.0)
    assert result.shape == (expected, 2)
    write_wav(tmp_path / "rounded-duration.wav", result)


@pytest.mark.parametrize("frames,ratio", [(0, 1.0), (1, 0.1)])
def test_offline_rejects_empty_segments(
    rubberband: object, frames: int, ratio: float
) -> None:
    with pytest.raises(ValueError, match="nonempty|duration"):
        rubberband.rubberband_stretch(np.zeros((frames, 1)), ratio, 1.0)


@pytest.mark.parametrize("value", [float("nan"), float("inf"), 1e100])
def test_invalid_audio_does_not_consume_the_realtime_input_block(
    rubberband: object, value: float, tmp_path: Path
) -> None:
    renderer = RealtimeRubberBand(48000, 1, 1.0, 48000, 512, 100000)
    invalid = np.zeros((512, 1))
    invalid[0, 0] = value
    with pytest.raises(ValueError, match="float32"):
        rubberband.rubberband_stretch(invalid, 1.0, 1.0)
    shifter = rubberband.RubberBandLiveShifter(48000, 1, 1.0)
    with pytest.raises(ValueError, match="float32"):
        shifter.shift(np.full((shifter.block_size, 1), value))
    with pytest.raises(ValueError, match="float32"):
        renderer.advance(invalid)
    assert renderer.source_received == 0
    source = audio(channels=1)
    blocks = []
    while not renderer.complete:
        start = renderer.source_received
        blocks.append(renderer.advance(source[start : start + 512]))
    result = np.vstack(blocks)
    assert np.all(np.isfinite(result))
    assert np.max(np.abs(result)) > 0.5
    write_wav(tmp_path / "recovered-input.wav", result)


def test_offline_linked_channels_preserve_antiphase_stereo(
    rubberband: object, tmp_path: Path
) -> None:
    mono = audio(96000, 1)[:, 0]
    source = np.column_stack([mono, -mono])
    result = rubberband.rubberband_stretch(source, 0.5, 1.5)
    assert np.max(np.abs(result[:, 0] + result[:, 1])) < 1e-6
    assert abs(dominant_frequency(result[:, 0]) - 330) < 2
    write_wav(tmp_path / "linked-stereo.wav", result)


def dominant_frequency(values: np.ndarray) -> float:
    center = values[len(values) // 4 : 3 * len(values) // 4]
    spectrum = np.abs(np.fft.rfft(center * np.hanning(len(center))))
    return float(np.argmax(spectrum) * 48000 / len(center))
