"""Pass exact release positions and owned parameter buffers to Rust sampling."""

from fractions import Fraction

import numpy as np
from ufor.number import MAX_INTEGER

from . import sampler


def render(
    definition: sampler.PreparedSample,
    state: sampler.SampleState,
    steps: np.ndarray,
    release_frame: Fraction | None,
    gain: float,
    gains: np.ndarray,
    spans: np.ndarray,
    routes: np.ndarray,
    frames: int,
    filters: tuple[list[int], np.ndarray, np.ndarray] | None = None,
    filter_after_amplitude: bool = False,
) -> tuple[np.ndarray, sampler.SampleState, np.ndarray]:
    from . import _native

    release = None
    if release_frame is not None and state.frame <= release_frame < state.frame + len(
        steps
    ):
        offset = release_frame - state.frame
        index = int(offset)
        fraction = offset - index
        if fraction.denominator > MAX_INTEGER:
            raise ValueError("Sample release fraction exceeds the signed 64-bit domain")
        release = (index, fraction.numerator, fraction.denominator)
    audio, values, filter_states = _native.render_sample(
        definition.native_traversal,
        state.native_state(),
        state.frame,
        np.ascontiguousarray(steps),
        release,
        gain,
        np.ascontiguousarray(gains, dtype=np.float64),
        spans,
        np.ascontiguousarray(routes, dtype=np.float64),
        frames,
        filters,
        filter_after_amplitude,
    )
    (
        index,
        position,
        error,
        direction,
        looping,
        overlap,
        released,
        exhaustion,
        repeats_remaining,
        plays_remaining,
        end_fade_at,
    ) = values
    return (
        audio,
        sampler.SampleState(
            frame=state.frame + len(steps),
            index=index,
            position=position,
            error=error,
            direction=direction,
            looping=looping,
            overlap=overlap,
            released=released,
            exhaustion_frame=exhaustion,
            repeats_remaining=repeats_remaining,
            plays_remaining=plays_remaining,
            end_fade_at=end_fade_at,
        ),
        filter_states,
    )
