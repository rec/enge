"""Provisional TX81Z portamento from measured firmware pitch steps."""

from math import isfinite, log2
from typing import Self

import numpy as np
from pydantic import BaseModel, ConfigDict, Field

from .synth import EngineError


class TX81ZPortamento(BaseModel):
    time: int = Field(ge=1, le=99)
    sample_rate: int = Field(gt=0)
    target_hz: float = Field(gt=0)
    current_hz: float = Field(gt=0)
    ascending: bool
    offset4: int
    remainder: int = Field(default=0, ge=0)

    @classmethod
    def start(
        cls, time: int, sample_rate: int, start_hz: float, target_hz: float
    ) -> Self:
        if any(not isfinite(x) or x <= 0 for x in (start_hz, target_hz)):
            raise EngineError("TX81Z portamento pitches must be finite and positive")
        offset = round(3072 * (log2(target_hz) - log2(start_hz)))
        return cls(
            time=time,
            sample_rate=sample_rate,
            target_hz=target_hz,
            current_hz=start_hz,
            ascending=start_hz < target_hz,
            offset4=offset if start_hz <= target_hz else offset - 256,
        )

    def ratios(self, frames: int) -> np.ndarray:
        if frames < 0:
            raise EngineError("TX81Z portamento frame count must be nonnegative")
        output = np.empty(frames)
        for frame in range(frames):
            output[frame] = self.current_hz / self.target_hz
            if self.current_hz == self.target_hz:
                continue
            ticks, self.remainder = divmod(self.remainder + 96, self.sample_rate)
            for _ in range(ticks):
                if self.current_hz == self.target_hz:
                    break
                step = (self.offset4 >> 10) + int(self.ascending)
                self.offset4 -= step * _MULTIPLIERS[self.time]
                offset = self.offset4 if self.ascending else self.offset4 + 256
                pitch_steps = (-offset) // 4
                pitch_steps = (
                    min(0, pitch_steps) if self.ascending else max(0, pitch_steps)
                )
                self.current_hz = self.target_hz * 2 ** (pitch_steps / 768)
        return output

    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)


# Measured register-step model at approximately 96 updates per second:
# https://nornand.hatenablog.com/entry/2020/12/14/185801
_MULTIPLIERS = [
    0,
    254,
    243,
    232,
    211,
    202,
    193,
    185,
    178,
    171,
    165,
    159,
    153,
    147,
    141,
    135,
    130,
    125,
    120,
    115,
    110,
    106,
    102,
    98,
    94,
    91,
    88,
    85,
    82,
    79,
    76,
    74,
    72,
    70,
    68,
    66,
    64,
    62,
    60,
    58,
    56,
    54,
    53,
    51,
    49,
    47,
    46,
    44,
    42,
    41,
    39,
    38,
    37,
    36,
    34,
    33,
    31,
    30,
    28,
    27,
    26,
    25,
    24,
    23,
    22,
    21,
    20,
    19,
    18,
    18,
    17,
    16,
    16,
    15,
    14,
    14,
    13,
    13,
    12,
    12,
    11,
    11,
    10,
    10,
    9,
    9,
    8,
    8,
    7,
    7,
    6,
    6,
    5,
    5,
    4,
    4,
    3,
    3,
    2,
    1,
]
