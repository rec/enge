"""Independent transfer-function and implicit state-space dynamic EQ references."""

import numpy as np


def static_band(source: np.ndarray, center: float = 1000, q: float = 1) -> np.ndarray:
    size = 131072
    z = np.exp(-2j * np.pi * np.arange(size // 2 + 1) / size)
    g = np.tan(np.pi * center / 48000)
    response = (
        (g / q)
        * (1 - z * z)
        / (1 + g / q + g * g + 2 * (g * g - 1) * z + (1 - g / q + g * g) * z * z)
    )
    return np.fft.irfft(
        np.fft.rfft(source, n=size, axis=0) * response[:, None], n=size, axis=0
    )[: len(source)]


def moving_band(source: np.ndarray, center: np.ndarray, q: np.ndarray) -> np.ndarray:
    memory = np.zeros((2, source.shape[1]))
    output = np.empty_like(source)
    for i, sample in enumerate(source):
        g = np.tan(np.pi * center[i] / 48000)
        matrix = np.array([[1 + g / q[i], g], [-g, 1]])
        right = memory.copy()
        right[0] += g * sample
        state = np.linalg.solve(matrix, right)
        output[i] = state[0] / q[i]
        memory = 2 * state - memory
        memory[np.abs(memory) < 1e-30] = 0
    return output


def bell_audio(
    source: np.ndarray,
    band: np.ndarray,
    ratio: float = 4,
    knee: float = 6,
    attack: float = 0.001,
    release: float = 0.05,
    maximum: float | np.ndarray = 12,
) -> np.ndarray:
    excess = 20 * np.log10(np.maximum(np.max(np.abs(band), axis=1), 1e-300)) + 24
    target = (1 / ratio - 1) * (
        np.maximum(excess, 0)
        if knee == 0
        else np.where(
            excess <= -knee / 2,
            0,
            np.where(excess >= knee / 2, excess, (excess + knee / 2) ** 2 / (2 * knee)),
        )
    )
    target = np.maximum(target, -np.broadcast_to(maximum, target.shape))
    reduction = 0.0
    gains = np.empty(len(source))
    for i, value in enumerate(target):
        seconds = attack if value < reduction else release
        reduction = (
            value
            if seconds == 0
            else value + np.exp(-1 / (48000 * seconds)) * (reduction - value)
        )
        gains[i] = 10 ** (reduction / 20)
    return source + (gains[:, None] - 1) * band
