"""Linked peak analysis with sample-rate-aware attack and release."""

from math import expm1

import numpy as np


def process(
    level: float,
    audio: np.ndarray,
    attack: float,
    release: float,
    rate: int,
    floor: float,
) -> float:
    if attack < 0 or release < 0:
        raise ValueError("Follower attack/release must be nonnegative")
    peak = max(abs(float(v)) for v in audio)
    seconds = attack if peak > level else release
    level = (
        peak if seconds == 0 else level + (peak - level) * -expm1(-1 / (rate * seconds))
    )
    return 0.0 if level < floor else level
