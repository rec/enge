"""Per-channel Schmitt-trigger octave division with peak-scaled output."""

from math import isfinite

import numpy as np
from pydantic import BaseModel, ConfigDict, StrictBool

from . import follower


class OctaveState(BaseModel):
    levels: list[float]
    armed: list[StrictBool]
    positive: list[StrictBool]
    active: list[StrictBool]

    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)


def initial_state(channels: int) -> OctaveState:
    return OctaveState(
        levels=[0.0] * channels,
        armed=[False] * channels,
        positive=[False] * channels,
        active=[False] * channels,
    )


def validate_state(state: OctaveState, channels: int) -> None:
    if (
        any(
            len(v) != channels
            for v in (state.levels, state.armed, state.positive, state.active)
        )
        or any(not isfinite(v) or v < 0 for v in state.levels)
        or any(
            type(v) is not bool
            for r in (state.armed, state.positive, state.active)
            for v in r
        )
    ):
        raise ValueError("Invalid octave divider snapshot")


def process(
    state: OctaveState,
    source: np.ndarray,
    threshold: float,
    attack: float,
    release: float,
    rate: int,
    floor: float,
    tail: float,
    ended: bool,
) -> tuple[np.ndarray, bool]:
    if (
        not isfinite(threshold)
        or threshold <= 0
        or not all(isfinite(v) and v >= 0 for v in (attack, release))
    ):
        raise ValueError("Invalid octave divider controls")
    if not np.all(np.isfinite(source)):
        raise ValueError("Non-finite octave divider input")
    wet = np.zeros_like(source)
    for i, value in enumerate(source):
        state.levels[i] = follower.process(
            state.levels[i], source[i : i + 1], attack, release, rate, floor
        )
        if state.levels[i] == 0:
            state.armed[i] = state.positive[i] = state.active[i] = False
        elif value <= -threshold:
            state.armed[i] = True
        elif value >= threshold and state.armed[i]:
            state.positive[i] = not state.positive[i]
            state.active[i] = True
            state.armed[i] = False
        if state.active[i]:
            wet[i] = state.levels[i] * (1 if state.positive[i] else -1)
    drained = ended and all(v < tail for v in state.levels)
    if drained:
        state.levels = [0.0] * len(source)
        state.armed = [False] * len(source)
        state.positive = [False] * len(source)
        state.active = [False] * len(source)
        wet.fill(0)
    return wet, drained
