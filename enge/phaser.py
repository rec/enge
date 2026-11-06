"""Six-stage all-pass phasing with one frame of bounded feedback."""

from math import exp2, isfinite, log2, pi, sqrt, tan

import numpy as np
from pydantic import BaseModel, ConfigDict, Field


class PhaserState(BaseModel):
    stages: list[list[float]] = Field(min_length=6, max_length=6)
    feedback: list[float] = Field(min_length=1)

    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)


def coefficient(
    center_hz: float, depth_cents: float, sweep: float, sample_rate: int
) -> float:
    if (
        not isfinite(center_hz)
        or center_hz <= 0
        or not isfinite(depth_cents)
        or depth_cents < 0
        or not -1 <= sweep <= 1
    ):
        raise ValueError("Invalid phaser center, depth or sweep")
    exponent = log2(center_hz) + depth_cents * sweep / 1200
    if not isfinite(exponent) or exponent >= log2(sample_rate / 2):
        raise ValueError("Phaser frequency must remain below Nyquist")
    g = tan(pi * exp2(exponent) / sample_rate)
    value = (g - 1) / (g + 1)
    if not -1 < value < 1:
        raise ValueError("Unrepresentable phaser coefficient")
    return value


def process(
    state: PhaserState,
    source: np.ndarray,
    a: float,
    feedback: float,
    state_floor: float,
) -> np.ndarray:
    if not -0.95 <= feedback <= 0.95 or not np.all(np.isfinite(source)):
        raise ValueError("Invalid phaser feedback or input")
    value = source + feedback * np.asarray(state.feedback)
    b = sqrt((1 - a) * (1 + a))
    for stage in state.stages:
        memory = np.asarray(stage)
        output = a * value + b * memory
        memory = b * value - a * memory
        if not np.all(np.isfinite(output)) or not np.all(np.isfinite(memory)):
            raise ValueError("Non-finite phaser state")
        memory[np.abs(memory) < state_floor] = 0
        stage[:] = memory.tolist()
        value = output
    state.feedback = [0.0 if abs(v) < state_floor else float(v) for v in value]
    return value
