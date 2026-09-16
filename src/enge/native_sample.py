"""Prepare exact release splits and owned parameter buffers for Rust sampling."""

from fractions import Fraction

import numpy as np
from ufor.samples.enums import Direction, LoopMode

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
) -> tuple[np.ndarray, sampler.SampleState]:
    from . import _native

    release = None
    if release_frame is not None and state.frame <= release_frame < state.frame + len(
        steps
    ):
        offset = release_frame - state.frame
        index = int(offset)
        fraction = offset - index
        before = Fraction(float(steps[index])) * fraction
        release = (
            index,
            bool(fraction),
            float(before),
            float(Fraction(float(steps[index])) - before),
        )
    selection = definition.slice
    loop = selection.loop
    audio, values = _native.render_sample(
        definition.native_buffer,
        (
            selection.start_frame,
            selection.end_frame,
            definition.playback.direction == Direction.mirror,
            None
            if loop is None
            else (
                loop.start_frame,
                loop.end_frame,
                loop.crossfade_frames,
                loop.mode == LoopMode.until_release,
            ),
        ),
        (
            state.index,
            state.position,
            state.error,
            state.direction,
            state.looping,
            state.overlap,
            state.released,
            state.exhaustion_frame,
        ),
        state.frame,
        definition.sample_rate,
        np.ascontiguousarray(steps),
        release,
        gain,
        np.ascontiguousarray(gains, dtype=np.float64),
        spans,
        np.ascontiguousarray(routes, dtype=np.float64),
        frames,
    )
    index, position, error, direction, looping, overlap, released, exhaustion = values
    return audio, sampler.SampleState(
        frame=state.frame + len(steps),
        index=index,
        position=position,
        error=error,
        direction=direction,
        looping=looping,
        overlap=overlap,
        released=released,
        exhaustion_frame=exhaustion,
    )
