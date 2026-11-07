"""Named input mixing with continuously retimed linear gate fades."""

from math import fsum, isfinite, pow

import numpy as np
from pydantic import BaseModel, ConfigDict, Field


class InputFade(BaseModel):
    value: float = Field(ge=0, le=1)
    target: float = Field(ge=0, le=1)
    step: float = 0.0
    remaining: int = Field(default=0, ge=0)

    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)


class MixerState(BaseModel):
    fades: list[InputFade] = Field(min_length=1)

    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)


def initial_state(muted: list[bool], solo: list[bool]) -> MixerState:
    selected = any(solo)
    values = [
        float(not m and (not selected or s)) for m, s in zip(muted, solo, strict=True)
    ]
    return MixerState(fades=[InputFade(value=v, target=v) for v in values])


def process(
    state: MixerState,
    inputs: list[np.ndarray],
    parameters: list[list[float]],
    fade_frames: int,
) -> np.ndarray:
    if any(p[1] not in (0, 1) or p[2] not in (0, 1) for p in parameters):
        raise ValueError("Mixer selectors must be zero or one")
    selected = any(p[2] == 1 for p in parameters)
    for fade, values in zip(state.fades, parameters, strict=True):
        target = float(values[1] == 0 and (not selected or values[2] == 1))
        if target != fade.target:
            fade.target = target
            fade.remaining = fade_frames
            fade.step = (target - fade.value) / fade_frames
    try:
        gains = [
            pow(10, p[0] / 20) * f.value
            for p, f in zip(parameters, state.fades, strict=True)
        ]
        if not all(isfinite(g) for g in gains):
            raise ValueError("Non-finite mixer gain")
        output = np.asarray(
            [
                fsum(g * float(a[c]) for g, a in zip(gains, inputs, strict=True))
                for c in range(len(inputs[0]))
            ]
        )
    except OverflowError as error:
        raise ValueError("Non-finite mixer sum") from error
    if not np.all(np.isfinite(output)):
        raise ValueError("Non-finite mixer output")
    for fade in state.fades:
        if fade.remaining:
            fade.remaining -= 1
            fade.value = fade.target if fade.remaining == 0 else fade.value + fade.step
    return output
