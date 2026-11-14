"""Linked sample-peak limiting with a fixed look-ahead window."""

from math import expm1, isfinite

import numpy as np
from pydantic import BaseModel, ConfigDict, Field


class LimiterState(BaseModel):
    history: list[list[float]]
    write: int = Field(default=0, ge=0, strict=True)
    remaining: int = Field(ge=0, strict=True)
    gain: float = Field(default=1.0, ge=0, le=1)

    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)


def initial_state(frames: int, channels: int) -> LimiterState:
    return LimiterState(
        history=[[0.0] * channels for _ in range(frames + 1)], remaining=frames
    )


def validate_state(state: LimiterState, frames: int, channels: int) -> None:
    if (
        len(state.history) != frames + 1
        or type(state.write) is not int
        or not 0 <= state.write <= frames
        or type(state.remaining) is not int
        or not 0 <= state.remaining <= frames
        or not isfinite(state.gain)
        or not 0 <= state.gain <= 1
        or any(
            len(r) != channels or any(not isfinite(v) for v in r) for r in state.history
        )
    ):
        raise ValueError("Invalid limiter snapshot")


def process(
    state: LimiterState,
    source: np.ndarray,
    ceiling_db: float,
    release: float,
    rate: int,
    ended: bool,
) -> tuple[np.ndarray, np.ndarray, bool]:
    if (
        not isfinite(ceiling_db)
        or ceiling_db > 0
        or not isfinite(release)
        or release < 0
    ):
        raise ValueError("Invalid limiter controls")
    if not np.all(np.isfinite(source)):
        raise ValueError("Non-finite limiter input")
    state.history[state.write] = source.tolist()
    state.write = (state.write + 1) % len(state.history)
    dry = np.asarray(state.history[state.write])
    peak = np.max(np.abs(state.history))
    ceiling = 10.0 ** (ceiling_db / 20)
    target = min(1.0, ceiling / peak) if peak else 1.0
    released = (
        1.0
        if release == 0
        else state.gain + (1 - state.gain) * -expm1(-1 / (rate * release))
    )
    state.gain = min(target, released)
    wet = np.clip(dry * state.gain, -ceiling, ceiling)
    drained = ended and state.remaining == 0
    state.remaining = max(0, state.remaining - 1) if ended else len(state.history) - 1
    return dry, wet, drained
