"""Three-band LR4 crossover with phase correction and linked peak dynamics."""

from math import pi, sqrt, tan

import numpy as np
from pydantic import BaseModel, ConfigDict, Field
from ufor.audio_effects import MultibandCompressor

from . import compressor


class MultibandState(BaseModel):
    integrators: list[list[list[float]]] = Field(min_length=9, max_length=9)
    reductions: list[float] = Field(min_length=3, max_length=3)

    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)


def initial_state(channels: int) -> MultibandState:
    return MultibandState(
        integrators=[[[0.0, 0.0] for _ in range(channels)] for _ in range(9)],
        reductions=[0.0] * 3,
    )


def validate_state(state: MultibandState, channels: int) -> None:
    if (
        np.asarray(state.integrators).shape != (9, channels, 2)
        or len(state.reductions) != 3
        or not np.all(np.isfinite(state.integrators))
        or not np.all(np.isfinite(state.reductions))
        or any(v > 0 for v in state.reductions)
    ):
        raise ValueError("Invalid multiband snapshot")


def coefficients(
    definition: MultibandCompressor, rate: int
) -> list[tuple[float, float, float, float, float]]:
    if not 0 < definition.lower_crossover_hz < definition.upper_crossover_hz < rate / 2:
        raise ValueError("Multiband crossovers must strictly increase below Nyquist")
    result = []
    q = 1 / sqrt(2)
    for cutoff in [definition.lower_crossover_hz, definition.upper_crossover_hz]:
        g = tan(pi * cutoff / rate)
        d = q * (1 + g * g) + g
        a1 = q / d
        a2 = g * a1
        if not 0 < a1 < 1 or not 0 < a2 or not 0 < g * a2 < 1:
            raise ValueError("Unrepresentable multiband crossover coefficients")
        result.append((g, d, a1, a2, g * a2))
    return result


def process(
    state: MultibandState,
    source: np.ndarray,
    values: list[list[float]],
    prepared: list[tuple[float, float, float, float, float]],
    definition: MultibandCompressor,
    rate: int,
    ended: bool,
) -> tuple[np.ndarray, bool]:
    low = _stage(state, 0, source, prepared[0], 0, definition.state_floor)
    low = _stage(state, 1, low, prepared[0], 0, definition.state_floor)
    upper = _stage(state, 2, source, prepared[0], 1, definition.state_floor)
    upper = _stage(state, 3, upper, prepared[0], 1, definition.state_floor)
    mid = _stage(state, 4, upper, prepared[1], 0, definition.state_floor)
    mid = _stage(state, 5, mid, prepared[1], 0, definition.state_floor)
    high = _stage(state, 6, upper, prepared[1], 1, definition.state_floor)
    high = _stage(state, 7, high, prepared[1], 1, definition.state_floor)
    low = _stage(state, 8, low, prepared[1], 2, definition.state_floor)
    output = np.zeros_like(source)
    for i, band in enumerate([low, mid, high]):
        gain, state.reductions[i] = compressor.gain(
            float(np.max(np.abs(band))),
            values[i],
            state.reductions[i],
            rate,
            -float("inf"),
        )
        output += gain * band
    drained = ended and np.all(np.abs(state.integrators) < definition.tail_threshold)
    return (np.zeros_like(output) if drained else output), bool(drained)


def _stage(
    state: MultibandState,
    index: int,
    source: np.ndarray,
    coefficients: tuple[float, float, float, float, float],
    response: int,
    floor: float,
) -> np.ndarray:
    g, d, a1, a2, a3 = coefficients
    memory = np.asarray(state.integrators[index])
    s1, s2 = memory[:, 0], memory[:, 1]
    v3 = source - s2
    v1 = a1 * s1 + a2 * v3
    v2 = s2 + a2 * s1 + a3 * v3
    band = s1 / d + (g / d) * v3
    memory = np.column_stack([2 * v1 - s1, 2 * v2 - s2])
    if not np.all(np.isfinite(memory)):
        raise ValueError("Non-finite multiband filter state")
    memory[np.abs(memory) < floor] = 0
    state.integrators[index] = memory.tolist()
    return (
        v2
        if response == 0
        else source - band - v2
        if response == 1
        else source - 2 * band
    )
