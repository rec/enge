"""Caller-owned DX7 LFO clock, shared by independently rendered notes.

Rate/delay calibration and waveform numbering follow the pinned MSFA model.
Source: google/music-synthesizer-for-android, f67d41d, lfo.cc (Google Inc.).
The clock advances per output frame; this does not reproduce block-held controls
or establish hardware equivalence.
"""

from math import pi, sin

import numpy as np
from pydantic import BaseModel, ConfigDict, Field
from ufor.dx7 import DX7Voice

from .synth import EngineError
from .yamaha import dx7_global_parameters


class DX7LFOSnapshot(BaseModel, frozen=True):
    speed: int = Field(ge=0, le=99)
    delay: int = Field(ge=0, le=99)
    sync: bool
    waveform: int = Field(ge=0, le=5)
    sample_rate: int = Field(gt=0)
    frame: int = Field(default=0, ge=0)
    phase: int = Field(default=0, ge=0, lt=2**32)
    phase_remainder: int = Field(default=0, ge=0, lt=64)
    delay_position: int = Field(default=0, ge=0, le=2**32)
    delay_remainder: int = Field(default=0, ge=0, lt=64)
    random_state: int = Field(default=0, ge=0, le=255)

    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)


class DX7LFOContext:
    """Advance once, then pass the same (waveform, delay) rows to every note.

    Both columns are normalized to 0..1. Key-down delivery is explicit so the
    caller owns shared sync/retrigger policy. Initial phase and random state
    are zero; snapshots retain them independently of note snapshots.
    """

    def __init__(self, voice: DX7Voice, sample_rate: int = 48000) -> None:
        parameters = dx7_global_parameters(voice)
        self.state = DX7LFOSnapshot(
            speed=parameters[1],
            delay=parameters[2],
            sync=bool(parameters[5]),
            waveform=parameters[6],
            sample_rate=sample_rate,
        )

    def snapshot(self) -> DX7LFOSnapshot:
        return self.state.model_copy()

    def restore(self, snapshot: DX7LFOSnapshot) -> None:
        if (
            snapshot.speed != self.state.speed
            or snapshot.delay != self.state.delay
            or snapshot.sync != self.state.sync
            or snapshot.waveform != self.state.waveform
            or snapshot.sample_rate != self.state.sample_rate
        ):
            raise EngineError(
                "DX7 LFO snapshot belongs to a different clock definition"
            )
        self.state = snapshot.model_copy()

    def keydown(self) -> None:
        updates = {"delay_position": 0, "delay_remainder": 0}
        if self.state.sync:
            updates.update(phase=2**31 - 1, phase_remainder=0)
        self.state = self.state.model_copy(update=updates)

    def advance(self, frames: int) -> np.ndarray:
        if frames < 0:
            raise EngineError("DX7 LFO frame count must be nonnegative")
        state = self.state
        unit = int(64 * 25190424 / state.sample_rate + 0.5)
        rate = 1 if state.speed == 0 else 165 * state.speed >> 6
        rate *= 11 if rate < 160 else 11 + ((rate - 160) >> 4)
        delta = unit * rate
        inverse_delay = 99 - state.delay
        delay_rate = (16 + (inverse_delay & 15)) << (1 + (inverse_delay >> 4))
        delay_increment = unit * delay_rate
        delay_increment_2 = unit * max(128, delay_rate & 0xFF80)
        phase, remainder = state.phase, state.phase_remainder
        delay_position, delay_remainder = state.delay_position, state.delay_remainder
        random_state = state.random_state
        output = np.empty((frames, 2))
        for frame in range(frames):
            increment, remainder = divmod(delta + remainder, 64)
            wraps, phase = divmod(phase + increment, 2**32)
            if state.waveform == 5:
                for _ in range(wraps):
                    random_state = (random_state * 179 + 17) & 255
            match state.waveform:
                case 0:
                    value = ((phase >> 7) ^ -(phase >> 31)) & (2**24 - 1)
                    level = value / 2**24
                case 1:
                    level = (((~phase ^ 2**31) & (2**32 - 1)) >> 8) / 2**24
                case 2:
                    level = ((phase ^ 2**31) >> 8) / 2**24
                case 3:
                    level = float(phase < 2**31)
                case 4:
                    level = 0.5 + 0.5 * sin(2 * pi * phase / 2**32)
                case _:
                    level = ((random_state ^ 128) + 1) / 256
            if state.delay == 0:
                delay_position = 2**32
            elif delay_position < 2**32:
                delay_delta = (
                    delay_increment if delay_position < 2**31 else delay_increment_2
                )
                step, delay_remainder = divmod(delay_delta + delay_remainder, 64)
                delay_position = min(2**32, delay_position + step)
            output[frame] = level, max(0, delay_position - 2**31) / 2**31
        self.state = state.model_copy(
            update={
                "frame": state.frame + frames,
                "phase": phase,
                "phase_remainder": remainder,
                "delay_position": delay_position,
                "delay_remainder": delay_remainder,
                "random_state": random_state,
            }
        )
        return output
