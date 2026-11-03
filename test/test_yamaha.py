import wave
from pathlib import Path

import numpy as np
import pytest
from test_synth import check_audio
from ufor.dx7 import DX7Voice

from enge.synth import EngineError
from enge.yamaha import DX7Renderer


@pytest.mark.parametrize("coarse", [4, 31])
@pytest.mark.parametrize("detune", [0, 7, 14])
def test_fixed_operators_repeat_decades_and_ignore_played_key(
    tmp_path: Path, coarse: int, detune: int
) -> None:
    data = bytearray(voice().data)
    for i in range(0, 126, 21):
        data[i + 17] = 1
        data[i + 18] = coarse
        data[i + 19] = 37
        data[i + 20] = detune
    actual = DX7Renderer(DX7Voice(data=bytes(data)), note=57, velocity=100).render(
        48000
    )
    for i in range(0, 126, 21):
        data[i + 18] = coarse & 3
    expected = DX7Renderer(DX7Voice(data=bytes(data)), note=81, velocity=100).render(
        48000
    )
    check_audio(tmp_path / "fixed-frequency.wav", actual, expected)
    if detune <= 7:
        peak_hz = int(np.argmax(np.abs(np.fft.rfft(actual[:, 0]))))
        assert abs(peak_hz - 10 ** ((coarse & 3) + 0.37)) < 1
    if detune == 0:
        for i in range(0, 126, 21):
            data[i + 20] = 7
        neutral = DX7Renderer(DX7Voice(data=bytes(data)), note=69, velocity=100).render(
            48000
        )
        check_audio(tmp_path / "fixed-negative-detune.wav", actual, neutral)


def voice() -> DX7Voice:
    data = bytearray(155)
    for start in range(0, 126, 21):
        data[start : start + 4] = bytes([99, 99, 99, 99])
        data[start + 4 : start + 8] = bytes([99, 99, 99, 99])
        data[start + 16] = 99
        data[start + 18] = 1
        data[start + 20] = 7
    data[134] = 31
    return DX7Voice(data=bytes(data))


@pytest.mark.parametrize(
    "packed,field",
    [(False, f) for f in ("algorithm", "feedback", "output", "rate", "level")]
    + [(True, f) for f in ("algorithm", "output", "rate", "level")],
)
def test_malformed_parameters_remain_inspectable_but_cannot_render(
    packed: bool, field: str
) -> None:
    data = bytearray(128 if packed else 155)
    positions = {
        "algorithm": (110 if packed else 134, 32),
        "feedback": (111 if packed else 135, 8),
        "output": (13 if packed else 16, 100),
        "rate": (0, 100),
        "level": (4, 100),
    }
    index, value = positions[field]
    data[index] = value
    malformed = DX7Voice(data=bytes(data))
    assert malformed.data == bytes(data)
    assert len(malformed.title) == 10
    with pytest.raises(EngineError):
        DX7Renderer(malformed, note=69, velocity=100)


def test_dx7_renderer_has_deterministic_one_second_wav_regression(
    tmp_path: Path,
) -> None:
    whole = DX7Renderer(voice(), note=69, velocity=100).render(48000)
    split = DX7Renderer(voice(), note=69, velocity=100)
    actual = np.vstack((split.render(17321), split.render(30679)))

    assert whole.shape == (48000, 1)
    assert whole.dtype == np.float64
    assert np.all(np.isfinite(whole))
    assert np.max(np.abs(whole)) > 0
    with wave.open(str(tmp_path / "dx7-renderer.wav"), "wb") as output:
        output.setnchannels(1)
        output.setsampwidth(4)
        output.setframerate(48000)
        output.writeframes(
            (whole / np.max(np.abs(whole)) * (2**31 - 1)).astype("<i4").tobytes()
        )
    np.testing.assert_allclose(actual, whole, atol=1e-10, rtol=1e-9)


def test_dx7_renderer_releases_all_operators() -> None:
    data = bytearray(voice().data)
    for start in range(0, 126, 21):
        data[start + 7] = 0
    renderer = DX7Renderer(DX7Voice(data=bytes(data)), note=69, velocity=100)

    renderer.render(1)
    renderer.release()
    released = renderer.render(48000)

    assert np.max(np.abs(released[-1000:])) < 5e-4
