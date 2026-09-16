"""Prepare exact envelope boundaries and owned buffers for the native kernel."""

from fractions import Fraction
from math import ceil

import numpy as np
from ufor.envelope import Envelope, Segment
from ufor.oscillator import Oscillator, Waveform


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
) -> tuple[np.ndarray, np.ndarray]:
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
    )


def envelope_spans(
    envelope: Envelope,
    start: int,
    frames: int,
    sample_rate: int,
    release_frame: Fraction | None,
) -> np.ndarray:
    """Resolve half-open integer spans without rounding rational boundaries.

    Rows contain first, last, entry, delta, progress origin, denominator, and
    local index origin. Rust evaluates the linear arithmetic for each sample.
    Later rows overwrite earlier rows, including the release's terminal level.
    """
    spans = _spans(
        envelope.initial,
        envelope.segments,
        Fraction(start, sample_rate),
        frames,
        sample_rate,
        0,
    )
    if release_frame is not None:
        offset = max(0, ceil(release_frame) - start)
        if offset < frames:
            elapsed = release_frame / sample_rate
            boundary = Fraction(0)
            level = envelope.initial
            for segment in envelope.segments:
                end = boundary + segment.duration
                if elapsed < end:
                    level += (segment.target - level) * float(
                        (elapsed - boundary) / segment.duration
                    )
                    break
                boundary, level = end, segment.target
            spans.extend(
                _spans(
                    level,
                    envelope.release,
                    (start + offset - release_frame) / sample_rate,
                    frames - offset,
                    sample_rate,
                    offset,
                )
            )
    return np.asarray(spans, dtype=np.float64)


def _spans(
    initial: float,
    segments: list[Segment],
    elapsed: Fraction,
    frames: int,
    sample_rate: int,
    offset: int,
) -> list[list[float]]:
    spans = [[offset, offset + frames, segments[-1].target, 0, 0, 1, offset]]
    boundary = Fraction(0)
    entry = initial
    for segment in segments:
        end = boundary + segment.duration
        first = max(0, ceil((boundary - elapsed) * sample_rate))
        last = min(frames, ceil((end - elapsed) * sample_rate))
        if segment.duration and first < last:
            spans.append(
                [
                    first + offset,
                    last + offset,
                    entry,
                    segment.target - entry,
                    float((elapsed - boundary) / segment.duration),
                    float(segment.duration * sample_rate),
                    offset,
                ]
            )
        boundary, entry = end, segment.target
    return spans
