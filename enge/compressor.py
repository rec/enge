"""Shared NumPy linked-peak soft-knee reduction recurrence."""

from math import exp, isfinite, log10

import numpy as np


def gain(
    peak: float, parameters: list[float], reduction: float, rate: int
) -> tuple[float, float]:
    threshold, ratio, knee, attack, release, makeup = parameters
    if (
        any(not isfinite(v) for v in parameters)
        or ratio < 1
        or min(knee, attack, release) < 0
    ):
        raise ValueError("Invalid compressor parameters")
    x = 20 * log10(peak) - threshold if peak else -np.inf
    if x <= -knee / 2:
        target = 0.0
    elif x >= knee / 2:
        target = (1 / ratio - 1) * x
    else:
        target = (0.5 * (1 / ratio - 1) * knee) * (x / knee + 0.5) ** 2
    time = attack if target < reduction else release
    reduction = (
        target if time == 0 else target + exp(-1 / (rate * time)) * (reduction - target)
    )
    value = float(np.power(10, (reduction + makeup) / 20))
    if not isfinite(value) or not isfinite(reduction):
        raise ValueError("Non-finite compressor gain")
    return value, reduction
