"""Prepare exact envelope boundaries and owned buffers for the native kernel."""

from fractions import Fraction
from math import ceil

import numpy as np
from envelfo.runtime import envelfo_at
from ufor.envelfo import EnvelfoState, Envelope, EnvelopeState
from ufor.oscillator import Oscillator, Waveform
from ufor.segments import IndefiniteSegment, Segment


def render(
    oscillator: Oscillator,
    states: np.ndarray,
    frequencies: np.ndarray,
    sample_rate: int,
    gain: float,
    gains: np.ndarray,
    spans: np.ndarray,
    routes: list[list[float]],
    frames: int,
    filters: tuple[list[int], np.ndarray, np.ndarray] | None = None,
    filter_after_amplitude: bool = False,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return owned audio and next-state buffers without changing the inputs."""
    from . import _native

    return _native.render(
        [Waveform.sine, Waveform.square, Waveform.triangle].index(oscillator.waveform),
        float(oscillator.duty_cycle),
        sample_rate,
        gain,
        np.ascontiguousarray(frequencies, dtype=np.float64),
        states,
        np.ascontiguousarray(gains, dtype=np.float64),
        spans,
        np.asarray(routes, dtype=np.float64),
        frames,
        filters,
        filter_after_amplitude,
    )


def envelope_spans(
    envelope: Envelope,
    state: EnvelfoState,
    start: int,
    frames: int,
    sample_rate: int,
) -> np.ndarray:
    """Resolve half-open integer spans without rounding rational boundaries.

    Rows contain first, last, entry, delta, progress origin, denominator, and
    local index origin. Rust evaluates the linear arithmetic for each sample.
    Later rows overwrite earlier rows, including the release's terminal level.
    """
    if not frames:
        return np.empty((0, 7))
    runtime = state.runtime
    assert isinstance(runtime, EnvelopeState)
    position = runtime.position
    if position.direction == -1:
        return np.array(
            [
                [
                    i,
                    i + 1,
                    0
                    if (
                        v := envelfo_at(
                            envelope, state, Fraction(start + i, sample_rate)
                        )
                    ).status
                    == "complete"
                    else v.value,
                    0,
                    0,
                    1,
                    i,
                ]
                for i in range(frames)
            ],
            dtype=np.float64,
        )
    if position.paused or not position.rate:
        value = envelfo_at(envelope, state, Fraction(start, sample_rate))
        return np.array(
            [[0, frames, 0 if value.status == "complete" else value.value, 0, 0, 1, 0]],
            dtype=np.float64,
        )
    parts = envelope.release if runtime.phase == "release" else envelope.segments
    indefinite = any(
        s.duration is None for s in [*envelope.segments, *envelope.release]
    )
    parts = parts[runtime.segment_index :]
    total = sum((s.duration for s in parts if isinstance(s, Segment)), Fraction(0))
    scale = Fraction(1) if indefinite else total
    speed = position.rate * scale
    if not speed:
        value = envelfo_at(envelope, state, Fraction(start, sample_rate))
        return np.array(
            [[0, frames, 0 if value.status == "complete" else value.value, 0, 0, 1, 0]],
            dtype=np.float64,
        )
    elapsed = (
        position.coordinate * scale / speed + Fraction(start, sample_rate) - position.at
    )
    scaled = [
        s.model_copy(update={"duration": s.duration / speed})
        if isinstance(s, Segment)
        else s
        for s in parts
    ]
    return np.asarray(
        _spans(
            runtime.start_value,
            scaled,
            elapsed,
            frames,
            sample_rate,
            0,
            runtime.phase == "release",
        ),
        dtype=np.float64,
    )


def _spans(
    initial: float,
    segments: list[Segment | IndefiniteSegment],
    elapsed: Fraction,
    frames: int,
    sample_rate: int,
    offset: int,
    silence_after: bool,
) -> list[list[float]]:
    terminal = initial
    for segment in segments:
        if isinstance(segment, IndefiniteSegment):
            break
        terminal = segment.to
    spans = [[offset, offset + frames, terminal, 0, 0, 1, offset]]
    boundary = Fraction(0)
    entry = initial
    for segment in segments:
        if isinstance(segment, IndefiniteSegment):
            first = min(frames, max(0, ceil((boundary - elapsed) * sample_rate)))
            spans.append([first + offset, offset + frames, entry, 0, 0, 1, offset])
            return spans
        end = boundary + segment.duration
        first = max(0, ceil((boundary - elapsed) * sample_rate))
        last = min(frames, ceil((end - elapsed) * sample_rate))
        if segment.duration and first < last:
            spans.append(
                [
                    first + offset,
                    last + offset,
                    entry,
                    segment.to - entry,
                    float((elapsed - boundary) / segment.duration),
                    float(segment.duration * sample_rate),
                    offset,
                ]
            )
        boundary, entry = end, segment.to
    if silence_after:
        first = max(0, ceil((boundary - elapsed) * sample_rate))
        spans.append([min(first, frames) + offset, frames + offset, 0, 0, 0, 1, offset])
    return spans
