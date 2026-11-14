"""Four-times oversampling with a fixed linear-phase half-band profile."""

from math import cos, pi, sin
from typing import Literal

import numpy as np
from pydantic import BaseModel, ConfigDict
from ufor.audio_effects import WAVESHAPER_LATENCY_FRAMES


class WaveshaperState(BaseModel):
    histories: list[list[list[float]]]
    writes: list[int]
    dry_history: list[list[float]]
    dry_write: int = 0
    remaining: int = 96

    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)


def initial_state(channels: int) -> WaveshaperState:
    return WaveshaperState(
        histories=[[[0.0] * channels for _ in range(65)] for _ in range(4)],
        writes=[0] * 4,
        dry_history=[[0.0] * channels for _ in range(WAVESHAPER_LATENCY_FRAMES)],
    )


def halfband_coefficients() -> list[float]:
    values = [
        (0.5 if n == 32 else sin(pi * (n - 32) / 2) / (pi * (n - 32)) if n % 2 else 0.0)
        * (0.42 - 0.5 * cos(2 * pi * n / 64) + 0.08 * cos(4 * pi * n / 64))
        for n in range(65)
    ]
    total = sum(values)
    return [v / total for v in values]


def process(
    state: WaveshaperState,
    source: np.ndarray,
    drive: float,
    bias: float,
    output_gain_db: float,
    curve: Literal["soft", "hard", "fold", "half_wave", "full_wave"],
    coefficients: np.ndarray,
    ended: bool,
) -> tuple[np.ndarray, np.ndarray, bool]:
    if not np.all(np.isfinite(source)):
        raise ValueError("Non-finite waveshaper input")
    dry = np.asarray(state.dry_history[state.dry_write])
    state.dry_history[state.dry_write] = source.tolist()
    state.dry_write = (state.dry_write + 1) % len(state.dry_history)
    wet = np.zeros_like(source)
    zero = np.zeros_like(source)
    offset = _shape(np.asarray(bias), curve)
    for i in range(2):
        first = 2 * _filter(state, 0, source if i == 0 else zero, coefficients)
        for j in range(2):
            second = 2 * _filter(state, 1, first if j == 0 else zero, coefficients)
            shaped = drive * second + bias
            shaped = _shape(shaped, curve) - offset
            third = _filter(state, 2, shaped, coefficients)
            if j == 0:
                fourth = _filter(state, 3, third, coefficients)
                if i == 0:
                    wet = fourth
    gain = np.power(10.0, output_gain_db / 20)
    if not np.isfinite(gain):
        raise ValueError("Non-finite waveshaper output gain")
    drained = ended and state.remaining == 0
    state.remaining = max(0, state.remaining - 1) if ended else 96
    wet *= gain
    if not np.all(np.isfinite(wet)):
        raise ValueError("Non-finite waveshaper output")
    return dry, wet, drained


def _shape(
    values: np.ndarray, curve: Literal["soft", "hard", "fold", "half_wave", "full_wave"]
) -> np.ndarray:
    if curve == "hard":
        return np.clip(values, -1, 1)
    if curve == "soft":
        return np.tanh(values)
    if curve == "half_wave":
        return np.maximum(values, 0)
    if curve == "full_wave":
        return np.abs(values)
    return 1 - np.abs(np.remainder(values + 1, 4) - 2)


def _filter(
    state: WaveshaperState, stage: int, source: np.ndarray, coefficients: np.ndarray
) -> np.ndarray:
    history = state.histories[stage]
    write = state.writes[stage]
    if not np.all(np.isfinite(source)):
        raise ValueError("Non-finite waveshaper input")
    history[write] = source.tolist()
    values = np.asarray(history)
    # Dot products provide an independent vector realization of the Rust FIR.
    result = coefficients @ values[(write - np.arange(65)) % 65]
    if not np.all(np.isfinite(result)):
        raise ValueError("Non-finite waveshaper FIR output")
    state.writes[stage] = (write + 1) % 65
    return result
