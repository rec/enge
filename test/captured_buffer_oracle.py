"""Whole-array traversal and mixing equations, independent of streaming storage."""

import numpy as np


def region_audio(
    source: np.ndarray, at: int, mode: str, positions: np.ndarray, length: int = 1200
) -> np.ndarray:
    captured = source[at - length : at]
    age = np.arange(len(source)) - at
    if mode == "reverse":
        return captured[np.clip(length - 1 - age, 0, length - 1)]
    if mode == "scrub":
        return np.column_stack(
            [
                np.interp(positions * (length - 1), np.arange(length), captured[:, c])
                for c in range(source.shape[1])
            ]
        )
    period = length - 64
    offset = age % period
    head = captured[offset]
    tail = captured[np.minimum(period + offset, length - 1)]
    weight = np.minimum(offset / 64, 1)[:, None]
    return tail * (1 - weight) + head * weight


def capture_gain(at: int, mode: str, length: int = 1200) -> np.ndarray:
    frames = np.arange(48000)
    gain = np.clip((frames - at) / 64, 0, 1)
    if mode == "reverse":
        gain *= np.clip((at + length - 1 - frames) / 64, 0, 1)
    return gain
