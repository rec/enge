"""Independent frequency-domain LR4 transfers and per-band static gain oracle."""

import numpy as np


def bands(
    source: np.ndarray, lower: float = 200, upper: float = 4000
) -> list[np.ndarray]:
    size = 262144
    z = np.exp(-2j * np.pi * np.arange(size // 2 + 1) / size)
    split = []
    for frequency in [lower, upper]:
        g = np.tan(np.pi * frequency / 48000)
        denominator = (
            1
            + np.sqrt(2) * g
            + g * g
            + 2 * (g * g - 1) * z
            + (1 - np.sqrt(2) * g + g * g) * z * z
        )
        low = (g * g * (1 + z) ** 2 / denominator) ** 2
        high = ((1 - z) ** 2 / denominator) ** 2
        split.append((low, high, low + high))
    transforms = [
        split[0][0] * split[1][2],
        split[0][1] * split[1][0],
        split[0][1] * split[1][1],
    ]
    spectrum = np.fft.rfft(source, n=size, axis=0)
    return [
        np.fft.irfft(spectrum * h[:, None], n=size, axis=0)[: len(source)]
        for h in transforms
    ]


def compressed(
    source: np.ndarray,
    threshold: float | np.ndarray,
    ratio: float,
    knee: float,
    attack: float,
    release: float,
    makeup: float = 0,
) -> np.ndarray:
    level = 20 * np.log10(np.maximum(np.max(np.abs(source), axis=1), 1e-300))
    excess = level - threshold
    target = (
        (1 / ratio - 1)
        * np.where(
            excess <= -knee / 2,
            0,
            np.where(excess >= knee / 2, excess, (excess + knee / 2) ** 2 / (2 * knee)),
        )
        if knee
        else (1 / ratio - 1) * np.maximum(excess, 0)
    )
    reduction = 0.0
    gains = np.empty(len(source))
    for i, value in enumerate(target):
        seconds = attack if value < reduction else release
        reduction = (
            value
            if seconds == 0
            else value + np.exp(-1 / (48000 * seconds)) * (reduction - value)
        )
        gains[i] = 10 ** ((reduction + makeup) / 20)
    return source * gains[:, None]
