"""Fixed Blackman-windowed Hilbert FIR and continuous quadrature modulation."""

from math import cos, pi, sin

import numpy as np
from pydantic import BaseModel, ConfigDict, Field


class FrequencyShifterState(BaseModel):
    history: list[list[float]] = Field(min_length=1025, max_length=1025)
    write: int = Field(default=0, ge=0, lt=1025)
    phase: float = Field(default=0, ge=0, lt=1)
    remaining: int = Field(default=1024, ge=0, le=1024)

    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)


def initial_state(channels: int) -> FrequencyShifterState:
    return FrequencyShifterState(history=[[0.0] * channels for _ in range(1025)])


def coefficients() -> list[float]:
    return [
        2
        / (pi * (n - 512))
        * (0.42 - 0.5 * cos(2 * pi * n / 1024) + 0.08 * cos(4 * pi * n / 1024))
        if n % 2
        else 0.0
        for n in range(1025)
    ]


def process(
    state: FrequencyShifterState,
    source: np.ndarray,
    shift_hz: float,
    sample_rate: int,
    taps: np.ndarray,
    ended: bool,
) -> tuple[np.ndarray, np.ndarray, bool]:
    if not -2000 <= shift_hz <= 2000 or not np.all(np.isfinite(source)):
        raise ValueError("Invalid frequency shifter input or shift")
    state.history[state.write] = source.tolist()
    history = np.asarray(state.history)
    dry = history[(state.write - 512) % 1025].copy()
    quadrature = taps @ history[(state.write - np.arange(1025)) % 1025]
    angle = 2 * pi * state.phase
    wet = dry * cos(angle) - quadrature * sin(angle)
    if not np.all(np.isfinite(wet)):
        raise ValueError("Non-finite frequency shifter output")
    state.write = (state.write + 1) % 1025
    state.phase = (state.phase + shift_hz / sample_rate) % 1
    if state.phase == 1:
        state.phase = 0.0
    drained = ended and state.remaining == 0
    state.remaining = max(0, state.remaining - 1) if ended else 1024
    return dry, wet, drained
