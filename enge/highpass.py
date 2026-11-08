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
    allpass, memory = _step(np.asarray(state.memory), source, a, state_floor)
    wet = (source - allpass) / 2
    if not np.all(np.isfinite(wet)):
        raise ValueError("Non-finite highpass state")
    state.memory = memory.tolist()
    return wet


def process_allpass(
    state: HighpassState, source: np.ndarray, a: float, state_floor: float
) -> np.ndarray:
    wet, memory = _step(np.asarray(state.memory), source, a, state_floor)
    state.memory = memory.tolist()
    return wet


def _step(
    memory: np.ndarray, source: np.ndarray, a: float, state_floor: float
) -> tuple[np.ndarray, np.ndarray]:
    b = sqrt((1 - a) * (1 + a))
    wet = a * source + b * memory
    memory = b * source - a * memory
    if not np.all(np.isfinite(wet)) or not np.all(np.isfinite(memory)):
        raise ValueError("Non-finite highpass state")
    memory[np.abs(memory) < state_floor] = 0
    return wet, memory
