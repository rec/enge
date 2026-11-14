"""Independent bilinear band-pass transfer and compressor gain reference."""

import numpy as np


def deesser_audio(
    source: np.ndarray, threshold: float | np.ndarray = -24
) -> np.ndarray:
    size = 131072
    z = np.exp(-2j * np.pi * np.arange(size // 2 + 1) / size)
    g = np.tan(np.pi * 6000 / 48000)
    response = (
        g
        * (1 - z * z)
        / (1 + g + g * g + 2 * (g * g - 1) * z + (1 - g + g * g) * z * z)
    )
    detector = np.fft.irfft(
        np.fft.rfft(source, n=size, axis=0) * response[:, None], n=size, axis=0
    )[: len(source)]
    level = 20 * np.log10(np.maximum(np.max(np.abs(detector), axis=1), 1e-300))
    excess = level - threshold
    target = -0.75 * np.where(
        excess <= -3, 0, np.where(excess >= 3, excess, (excess + 3) ** 2 / 12)
    )
    reduction = 0.0
    gains = np.empty(len(source))
    for i, value in enumerate(target):
        seconds = 0.001 if value < reduction else 0.05
        reduction = value + np.exp(-1 / (48000 * seconds)) * (reduction - value)
        gains[i] = 10 ** (reduction / 20)
    return source * gains[:, None]
