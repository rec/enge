"""Linked peak gate and downward expansion with linear gain smoothing."""

from math import ceil, expm1, isfinite, log10, pow
from sys import maxsize

from pydantic import BaseModel, ConfigDict, Field


class GateState(BaseModel):
    gain: float = Field(default=0.0, ge=0, le=1)
    opened: bool = False
    elapsed: int = Field(default=0, ge=0)

    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)


def hold_frames(seconds: float, rate: int) -> int:
    frames = seconds * rate
    if not isfinite(frames) or frames < 0 or frames >= maxsize + 1:
        raise ValueError("Gate hold cannot use the native frame format")
    return ceil(frames)


def gate_gain(
    state: GateState,
    peak: float,
    threshold: float,
    hysteresis: float,
    hold: float,
    attack: float,
    release: float,
    rate: int,
) -> float:
    closing = threshold - hysteresis
    if hysteresis < 0 or not isfinite(closing):
        raise ValueError("Invalid gate hysteresis or closing threshold")
    frames = hold_frames(hold, rate)
    level = 20 * log10(peak) if peak > 0 else -float("inf")
    if level >= threshold:
        state.opened = True
        state.elapsed = 0
    elif state.opened:
        if level >= closing:
            state.elapsed = 0
        elif state.elapsed >= frames:
            state.opened = False
            state.elapsed = 0
        else:
            state.elapsed += 1
    else:
        state.elapsed = 0
    state.gain = _smooth(state.gain, float(state.opened), attack, release, rate)
    return state.gain


def expander_gain(
    gain: float,
    peak: float,
    threshold: float,
    ratio: float,
    attack: float,
    release: float,
    rate: int,
) -> float:
    if ratio < 1:
        raise ValueError("Expander ratio must be at least one")
    level = 20 * log10(peak) if peak > 0 else -float("inf")
    target = (
        1.0
        if ratio == 1 or level >= threshold
        else pow(10, ((ratio - 1) * (level - threshold)) / 20)
    )
    return _smooth(gain, target, attack, release, rate)


def _smooth(
    gain: float, target: float, attack: float, release: float, rate: int
) -> float:
    if attack < 0 or release < 0:
        raise ValueError("Dynamics attack/release must be nonnegative")
    seconds = attack if target > gain else release
    return (
        target
        if seconds == 0
        else gain + (target - gain) * -expm1(-1 / (rate * seconds))
    )
