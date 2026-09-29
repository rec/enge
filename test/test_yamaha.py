import wave
from pathlib import Path

import numpy as np
from ufor.dx7 import DX7Voice

from enge.yamaha import DX7Renderer


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
