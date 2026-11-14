"""Closed-form reference for piecewise-constant linked input peaks."""

import numpy as np


def transient_audio(
    source: np.ndarray,
    attack_db: float | np.ndarray = 0,
    sustain_db: float | np.ndarray = 0,
) -> np.ndarray:
    peaks = np.max(np.abs(source), axis=1)
    boundaries = np.r_[0, np.flatnonzero(np.diff(peaks)) + 1, len(peaks)]
    levels: list[np.ndarray] = []
    for attack, release in ((0.001, 0.01), (0.01, 0.1)):
        values = np.zeros(len(peaks))
        previous = 0.0
        for first, last in zip(boundaries[:-1], boundaries[1:], strict=True):
            target = peaks[first]
            seconds = attack if target > previous else release
            values[first:last] = target + (previous - target) * np.exp(
                -np.arange(1, last - first + 1) / (48000 * seconds)
            )
            previous = values[last - 1]
        levels.append(values)
    contrast = (levels[0] - levels[1]) / np.maximum(
        np.maximum(levels[0], levels[1]), 1e-12
    )
    gain = 10 ** (
        (attack_db * np.maximum(contrast, 0) + sustain_db * np.maximum(-contrast, 0))
        / 20
    )
    return source * gain[:, None]
