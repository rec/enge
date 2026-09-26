"""Trapezoidal voice filters with explicit, serializable integrator state."""

from math import pi, tan
from typing import Literal

import numpy as np
from ufor.base import Model
from ufor.samples.processing import FilterBoundary, FilterResponse, ResonantFilter


class FilterState(Model, frozen=True):
    """One stage's [s1, s2] integrators for each source channel."""

    integrators: list[list[float]]


def initial_states(
    definitions: list[ResonantFilter], channels: int
) -> list[FilterState]:
    return [
        FilterState(integrators=[[0, 0] for _ in range(channels)])
        for f in definitions
        for _ in range(f.stages)
    ]


def parameters(
    definitions: list[ResonantFilter],
    rate: int,
    frames: int,
    values: np.ndarray | None = None,
    start: int = 0,
) -> np.ndarray:
    """Resolve (frames, filters, cutoff/Q), applying only authored boundaries."""
    if rate <= 0:
        raise ValueError("Filter sample rate must be positive")
    if values is None:
        values = np.tile(
            np.array([[f.cutoff_hz, f.q] for f in definitions]).reshape(-1, 2),
            (frames, 1, 1),
        )
    if values.shape != (frames, len(definitions), 2):
        raise ValueError("Filter parameters must have shape (frames, filters, 2)")
    values = np.array(values, dtype=np.float64, copy=True)
    for j, definition in enumerate(definitions):
        upper = rate / 2 * definition.nyquist_ratio
        if definition.minimum_hz >= upper:
            raise ValueError(f"Filter {definition.name}: invalid cutoff bounds")
        cutoff, q = values[:, j, 0], values[:, j, 1]
        invalid = ~np.isfinite(cutoff) | ~np.isfinite(q) | (q <= 0)
        outside = (cutoff < definition.minimum_hz) | (cutoff > upper)
        if definition.boundary == FilterBoundary.error:
            invalid |= outside
        if np.any(invalid):
            frame = start + int(np.flatnonzero(invalid)[0])
            raise ValueError(
                f"Filter {definition.name}: invalid cutoff/Q at frame {frame}"
            )
        np.clip(cutoff, definition.minimum_hz, upper, out=cutoff)
    return values


def native_inputs(
    definitions: list[ResonantFilter], states: list[FilterState], values: np.ndarray
) -> tuple[list[int], np.ndarray, np.ndarray]:
    """Expand the ordered stage cascade; Rust computes its own coefficients."""
    stages = [i for i, f in enumerate(definitions) for _ in range(f.stages)]
    responses = [list(FilterResponse).index(definitions[i].response) for i in stages]
    channels = len(states[0].integrators) if states else 0
    memory = np.array([s.integrators for s in states], dtype=np.float64).reshape(
        len(stages), channels, 2
    )
    return responses, memory, np.ascontiguousarray(values[:, stages, :])


def restored_states(memory: np.ndarray) -> list[FilterState]:
    return [FilterState(integrators=s.tolist()) for s in memory]


def filter_samples(
    definitions: list[ResonantFilter],
    states: list[FilterState],
    samples: np.ndarray,
    values: np.ndarray,
    rate: int,
    backend: Literal["numpy", "native"] = "numpy",
) -> tuple[np.ndarray, list[FilterState]]:
    """Filter source channels before gain/routing, without mutating inputs.

    Values must already have passed `parameters`, as in the instrument renderers.
    State and parameter arrays cross the same boundary for both realizations.
    """
    responses, memory, values = native_inputs(definitions, states, values)
    if backend == "native":
        from . import _native

        audio, memory = _native.render_filters(
            samples, rate, (responses, memory, values)
        )
        return audio, restored_states(memory)
    if backend != "numpy":
        raise ValueError(f"Unknown filter backend: {backend}")
    audio = np.array(samples, dtype=np.float64, copy=True)
    for j, response in enumerate(responses):
        s1, s2 = memory[j, :, 0].copy(), memory[j, :, 1].copy()
        for i, (cutoff, q) in enumerate(values[:, j]):
            g = tan(pi * float(cutoff) / rate)
            if q < 1:
                d = float(q) * (1 + g * g) + g
                a1 = float(q) / d
            else:
                a1 = 1 / (1 + g * (g + 1 / float(q)))
            a2 = g * a1
            a3 = g * a2
            v3 = audio[i] - s2
            v1 = a1 * s1 + a2 * v3
            v2 = s2 + a2 * s1 + a3 * v3
            band = s1 / d + (g / d) * v3 if q < 1 else v1 / q
            if response == 0:
                audio[i] = v2
            elif response == 1:
                audio[i] = audio[i] - band - v2
            elif response == 2:
                audio[i] = band
            else:
                audio[i] -= band
            s1, s2 = 2 * v1 - s1, 2 * v2 - s2
        memory[j, :, 0], memory[j, :, 1] = s1, s2
    if not np.all(np.isfinite(audio)) or not np.all(np.isfinite(memory)):
        raise ValueError("Non-finite filter output or state")
    return audio, restored_states(memory)
