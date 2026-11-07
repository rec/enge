"""One-pole highpass with a normalized all-pass state."""

from math import isfinite, pi, sqrt, tan

import numpy as np
from pydantic import BaseModel, ConfigDict, Field


class HighpassState(BaseModel):
    memory: list[float] = Field(min_length=1)

    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)


def coefficient(cutoff_hz: float, sample_rate: int) -> float:
    if not isfinite(cutoff_hz) or not 0 < cutoff_hz < sample_rate / 2:
        raise ValueError("Highpass cutoff must be positive and below Nyquist")
    g = tan(pi * (cutoff_hz / sample_rate))
    a = (g - 1) / (g + 1)
    if not -1 < a < 1:
        raise ValueError("Unrepresentable highpass coefficient")
    return a


def process(
    state: HighpassState, source: np.ndarray, a: float, state_floor: float
) -> np.ndarray:
    memory = np.asarray(state.memory)
    b = sqrt((1 - a) * (1 + a))
    wet = (source - (a * source + b * memory)) / 2
    memory = b * source - a * memory
    if not np.all(np.isfinite(wet)) or not np.all(np.isfinite(memory)):
        raise ValueError("Non-finite highpass state")
    memory[np.abs(memory) < state_floor] = 0
    state.memory = memory.tolist()
    return wet
