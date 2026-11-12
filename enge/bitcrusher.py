"""Signed integer quantization with capture-latched frame holds."""

from math import isfinite

import numpy as np
from pydantic import BaseModel, ConfigDict, Field


class BitcrusherState(BaseModel):
    held: list[float]
    remaining: int = Field(default=0, ge=0, le=65535, strict=True)

    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)


def process(
    state: BitcrusherState,
    source: np.ndarray,
    bits: float,
    hold_frames: float,
    ended: bool,
) -> tuple[np.ndarray, bool]:
    if (
        not isfinite(bits)
        or not 1 <= bits <= 24
        or bits != int(bits)
        or not isfinite(hold_frames)
        or not 1 <= hold_frames <= 65536
        or hold_frames != int(hold_frames)
    ):
        raise ValueError("Bitcrusher bits and hold_frames must be bounded integers")
    if not np.all(np.isfinite(source)):
        raise ValueError("Non-finite bitcrusher input")
    if state.remaining == 0:
        if ended:
            return np.zeros_like(source), True
        step = 2.0 ** (1 - int(bits))
        scaled = np.clip(source, -1, 1) / step
        quantized = np.sign(scaled) * np.floor(np.abs(scaled) + 0.5)
        state.held = np.clip(quantized * step, -1, 1 - step).tolist()
        state.remaining = int(hold_frames)
    state.remaining -= 1
    return np.asarray(state.held), False
