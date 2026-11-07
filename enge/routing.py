"""Always-applied matrix routing with scalar compensated accumulation."""

from math import cos, fsum, pi, sin

import numpy as np


def process(coefficients: list[list[float]], source: np.ndarray) -> np.ndarray:
    try:
        output = np.asarray(
            [
                fsum(c * float(x) for c, x in zip(r, source, strict=True))
                for r in coefficients
            ]
        )
    except OverflowError as error:
        raise ValueError("Non-finite routing matrix sum") from error
    if not np.all(np.isfinite(output)):
        raise ValueError("Non-finite routing matrix output")
    return output


def crossfade(
    first: np.ndarray, second: np.ndarray, position: float, equal_power: bool
) -> np.ndarray:
    if not 0 <= position <= 1:
        raise ValueError("Crossfader position must be in [0, 1]")
    left, right = 1 - position, position
    if equal_power and 0 < position < 1:
        left, right = cos(pi * position / 2), sin(pi * position / 2)
    output = first * left + second * right
    if not np.all(np.isfinite(output)):
        raise ValueError("Non-finite crossfader output")
    return output
