"""Block LFO evaluation with exact event anchors and optional sine interpolation."""

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
    control_interval: int = 1,
) -> np.ndarray:
    """Return value/activation columns without re-anchoring persistent state.

    Sine knots use an event-anchored grid independent of render partitions.
    Linear waveforms, discontinuities, and activation weights remain exact.
    """
    if definition.clock != "seconds" or sample_rate <= 0 or frames < 0:
        raise ValueError(
            "LFO rendering requires a seconds clock and valid frame layout"
        )
    if type(control_interval) is not int or control_interval <= 0:
        raise ValueError("Control interval must be a positive integer")
    if backend not in ("numpy", "native"):
        raise ValueError(f"Unknown LFO backend: {backend}")
    at = Fraction(start, sample_rate)
    if at < state.at or at < state.started_at:
        raise ValueError("LFO query precedes its current state")
    if not frames:
        return np.empty((0, 2))
    interval = control_interval
    if definition.waveform != Waveform.sine or state.rate * interval * 2 >= sample_rate:
        interval = 1
    weights = _activation_spans(definition, state, at, frames, sample_rate)
    anchor = ceil(max(state.at, state.started_at) * sample_rate)
    offset = (start - anchor) % interval
    count = frames if interval == 1 else (offset + max(0, frames - 1)) // interval + 2
    first = start if interval == 1 else start - offset
    spans = _phase_spans(definition, state, first, count, sample_rate, interval)
    sample_weights = (
        weights if interval == 1 else np.array([[0, count, 1, 0, 0, 1, 0]], dtype=float)
    )
    if backend == "native":
        from . import _native

        output = _native.render_lfo(
            [Waveform.sine, Waveform.square, Waveform.triangle].index(
                definition.waveform
            ),
            spans,
            sample_weights,
            count,
        )
    else:
        output = np.empty((count, 2))
        for span in spans:
            begin, end = int(span[0]), int(span[1])
            phase = span[2] + np.arange(end - begin) * span[3]
            if definition.waveform == Waveform.sine:
                values = np.sin(2 * np.pi * phase)
            elif definition.waveform == Waveform.square:
                values = np.full(end - begin, 1 if span[4] else -1)
            else:
                values = 2 * phase - 1 if span[4] else 1 - 2 * phase
            output[begin:end, 0] = np.clip(values, -1, 1)
        output[:, 1] = _weights(sample_weights, count)
    if interval != 1:
        output = np.column_stack(
            (
                np.interp(
                    np.arange(frames) + offset,
                    np.arange(count) * interval,
                    output[:, 0],
                ),
                _weights(weights, frames),
            )
        )
    return output


def _phase_spans(
    definition: lfo.LFO,
    state: lfo.LFOState,
    start: int,
    frames: int,
    sample_rate: int,
    interval: int,
) -> np.ndarray:
    phase = lfo.phase_at(state, Fraction(start, sample_rate))
    step = (state.rate * interval / sample_rate) % 1
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
    return np.array(spans, dtype=np.float64).reshape(-1, 5)


def _activation_spans(
    definition: lfo.LFO,
    state: lfo.LFOState,
    at: Fraction,
    frames: int,
    sample_rate: int,
) -> np.ndarray:
    age = at - state.started_at - definition.delay
    spans: list[list[float]] = [[0, frames, 1, 0, 0, 1, 0]]
    delayed = min(frames, max(0, ceil(-age * sample_rate)))
    spans.append([0, delayed, 0, 0, 0, 1, 0])
    if definition.fade_in:
        last = min(frames, ceil((definition.fade_in - age) * sample_rate))
        if delayed < last:
            spans.append(
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
    return np.array(spans, dtype=np.float64)


def _weights(spans: np.ndarray, frames: int) -> np.ndarray:
    output = np.zeros(frames)
    for span in spans:
        first, last = int(span[0]), int(span[1])
        output[first:last] = span[2] + span[3] * (
            span[4] + (np.arange(first, last) - span[6]) / span[5]
        )
    return np.clip(output, 0, 1)
