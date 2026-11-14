"""Linked fast/slow peak contrast for transient attack and decay shaping."""

from math import isfinite

import numpy as np
from pydantic import BaseModel, ConfigDict, Field

from . import follower


class TransientState(BaseModel):
    fast: float = Field(default=0.0, ge=0)
    slow: float = Field(default=0.0, ge=0)

    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)


def process(
    state: TransientState,
    source: np.ndarray,
    attack_db: float,
    sustain_db: float,
    rate: int,
    ended: bool,
) -> tuple[np.ndarray, bool]:
    if not all(isfinite(v) and -24 <= v <= 24 for v in (attack_db, sustain_db)):
        raise ValueError("Transient shaper gains must be in [-24, 24] dB")
    if not np.all(np.isfinite(source)):
        raise ValueError("Non-finite transient shaper input")
    if ended:
        state.fast = state.slow = 0.0
        return np.zeros_like(source), True
    state.fast = follower.process(state.fast, source, 0.001, 0.01, rate, 1e-30)
    state.slow = follower.process(state.slow, source, 0.01, 0.1, rate, 1e-30)
    contrast = (state.fast - state.slow) / max(state.fast, state.slow, 1e-12)
    gain_db = attack_db * max(contrast, 0) + sustain_db * max(-contrast, 0)
    wet = source * 10 ** (gain_db / 20)
    if not np.all(np.isfinite(wet)):
        raise ValueError("Non-finite transient shaper output")
    return wet, False
