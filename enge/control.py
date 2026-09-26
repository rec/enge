"""Array evaluation of uFor control ramps and modulation routes."""

from fractions import Fraction
from math import ceil

import numpy as np
from ufor import modulation
from ufor.samples.controls import ControlState


def control_samples(
    state: ControlState, start: int, frames: int, rate: int
) -> np.ndarray:
    """Resolve exact anchors once, then generate the linear ramp in arrays."""
    output = np.full(frames, state.target, dtype=np.float64)
    if not frames:
        return output
    elapsed = Fraction(start, rate) - state.at
    if elapsed < 0:
        raise ValueError("Control query precedes current state")
    if state.value == state.target or elapsed >= state.smoothing:
        return output
    count = min(frames, ceil((state.smoothing - elapsed) * rate))
    progress = (float(elapsed * rate) + np.arange(count)) / float(
        state.smoothing * rate
    )
    output[:count] = state.value + (state.target - state.value) * progress
    return output


def modulation_samples(
    definition: modulation.Modulation, signals: dict[str, np.ndarray], frames: int
) -> dict[tuple[str, str], np.ndarray]:
    """Evaluate value/weight columns without constructing per-sample models."""
    sources = {s.name: s for s in definition.sources}
    for name, signal in signals.items():
        if name not in sources:
            raise ValueError(f"Unknown modulation source {name}")
        source = sources[name]
        if signal.shape != (frames, 2) or not np.all(np.isfinite(signal)):
            raise ValueError(f"Invalid modulation signal {name}")
        if np.any((signal[:, 0] < source.minimum) | (signal[:, 0] > source.maximum)):
            raise ValueError(f"source {name} value is outside its domain")
        if np.any((signal[:, 1] < 0) | (signal[:, 1] > 1)):
            raise ValueError(f"Invalid modulation weight {name}")
    additions: dict[modulation.Target, list[np.ndarray]] = {
        p.target: [] for p in definition.parameters
    }
    products = {p.target: np.ones(frames) for p in definition.parameters}
    for route in sorted(definition.routes, key=lambda r: (r.source, r.name)):
        if route.source not in signals:
            raise ValueError(f"missing modulation source {route.source}")
        signal = signals[route.source]
        knots = np.array([p.input for p in route.points])
        amounts = np.array([p.amount for p in route.points])
        index = np.clip(
            np.searchsorted(knots, signal[:, 0], side="right") - 1, 0, len(knots) - 1
        )
        mapped = amounts[index]
        if route.interpolation == modulation.Interpolation.linear and len(knots) > 1:
            lower = np.minimum(index, len(knots) - 2)
            progress = (signal[:, 0] - knots[lower]) / (knots[lower + 1] - knots[lower])
            mapped = amounts[lower] + progress * (amounts[lower + 1] - amounts[lower])
            mapped = np.where(signal[:, 0] <= knots[0], amounts[0], mapped)
            mapped = np.where(signal[:, 0] >= knots[-1], amounts[-1], mapped)
        if route.operation == modulation.Operation.add:
            additions[route.target].append(signal[:, 1] * mapped)
        else:
            products[route.target] *= 1 + signal[:, 1] * (mapped - 1)
    output = {}
    for parameter in definition.parameters:
        value = (
            parameter.default + _sum(additions[parameter.target], frames)
        ) * products[parameter.target]
        if np.any(
            ~np.isfinite(value)
            | (value < parameter.minimum)
            | (value > parameter.maximum)
        ):
            raise ValueError(
                f"modulated {parameter.target.name}.{parameter.target.parameter} "
                "is outside its domain"
            )
        output[parameter.target.name, parameter.target.parameter] = value
    return output


def _sum(terms: list[np.ndarray], frames: int) -> np.ndarray:
    # Error-free two-sum expansions retain small terms under cancellation, as
    # scalar math.fsum does. Loop over routes, never over audio samples.
    partials: list[np.ndarray] = []
    for term in terms:
        value = term
        for i, partial in enumerate(partials):
            greater = np.abs(value) >= np.abs(partial)
            large = np.where(greater, value, partial)
            small = np.where(greater, partial, value)
            value = large + small
            partials[i] = small - (value - large)
        partials.append(value)
    total = np.zeros(frames)
    for partial in reversed(partials):
        total = total + partial
    return total
