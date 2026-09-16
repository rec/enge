"""Sample uFor LFO observations without changing their exact event anchors."""

from fractions import Fraction
from math import ceil
from typing import Literal

import numpy as np
from ufor import lfo
from ufor.oscillator import Waveform


def lfo_samples(
    definition: lfo.LFO,
    state: lfo.LFOState,
    start: int,
    frames: int,
    sample_rate: int,
    backend: Literal["numpy", "native"] = "numpy",
) -> np.ndarray:
    """Return bipolar value and activation weight columns on the seconds clock.

    Use uFor's lfo_event to apply rate/reset events before their addressed sample.
    Rate changes retain phase and activation age. Rendering does not re-anchor
    state, so silent intervals, partitions, and restores share the same timeline.
    """
    if definition.clock != "seconds" or sample_rate <= 0 or frames < 0:
        raise ValueError(
            "LFO rendering requires a seconds clock and valid frame layout"
        )
    at = Fraction(start, sample_rate)
    if at < state.at or at < state.started_at:
        raise ValueError("LFO query precedes its current state")
    if backend == "numpy":
        observations = [
            lfo.lfo_at(definition, state, Fraction(start + i, sample_rate))
            for i in range(frames)
        ]
        return np.array(
            [[v.value, v.weight] for v in observations], dtype=np.float64
        ).reshape(frames, 2)
    if backend != "native":
        raise ValueError(f"Unknown LFO backend: {backend}")
    from . import _native

    # Resolve discontinuities exactly before converting phase arithmetic to float.
    # Spans never straddle a wrap or duty edge. This also handles rates above the
    # output rate without iterating through unobserved cycles.
    phase = lfo.phase_at(state, at)
    step = (state.rate / sample_rate) % 1
    spans: list[list[float]] = []
    first = 0
    while first < frames:
        rising = phase < definition.duty_cycle
        boundary = definition.duty_cycle if rising else Fraction(1)
        count = (
            min(frames - first, ceil((boundary - phase) / step))
            if step
            else frames - first
        )
        origin, increment = phase, step
        if definition.waveform == Waveform.triangle:
            width = definition.duty_cycle if rising else 1 - definition.duty_cycle
            origin = (
                phase / width if rising else (phase - definition.duty_cycle) / width
            )
            increment = step / width if count > 1 else Fraction(0)
        spans.append(
            [first, first + count, float(origin), float(increment), float(rising)]
        )
        phase = (phase + count * step) % 1
        first += count
    age = at - state.started_at - definition.delay
    weights: list[list[float]] = [[0, frames, 1, 0, 0, 1, 0]]
    delayed = min(frames, max(0, ceil(-age * sample_rate)))
    weights.append([0, delayed, 0, 0, 0, 1, 0])
    if definition.fade_in:
        last = min(frames, ceil((definition.fade_in - age) * sample_rate))
        if delayed < last:
            weights.append(
                [
                    delayed,
                    last,
                    float((age + Fraction(delayed, sample_rate)) / definition.fade_in),
                    float(1 / (definition.fade_in * sample_rate)),
                    0,
                    1,
                    delayed,
                ]
            )
    return _native.render_lfo(
        [Waveform.sine, Waveform.square, Waveform.triangle].index(definition.waveform),
        np.array(spans, dtype=np.float64).reshape(-1, 5),
        np.array(weights, dtype=np.float64),
        frames,
    )
