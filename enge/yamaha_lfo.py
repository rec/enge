# The OPZ LFO arithmetic below is adapted from ymfm under this license:
# BSD 3-Clause License
#
# Copyright (c) 2021, Aaron Giles
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
#    list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
#    this list of conditions and the following disclaimer in the documentation
#    and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
#    contributors may be used to endorse or promote products derived from
#    this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

"""Caller-owned Yamaha LFO clocks, shared by independently rendered notes.

Rate/delay calibration and waveform numbering follow the pinned MSFA model.
Source: google/music-synthesizer-for-android, f67d41d, lfo.cc (Google Inc.).
The DX7 clock advances per output frame, without reproducing block-held controls.
The TX81Z clock follows ymfm's provisional OPZ model and measured register maps.
Neither implementation establishes hardware equivalence.
"""

from math import pi, sin
from typing import Annotated

import numpy as np
from pydantic import BaseModel, ConfigDict, Field
from ufor import tx81z
from ufor.dx7 import DX7Voice

from .synth import EngineError
from .yamaha import dx7_global_parameters, tx81z_lfo_registers


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


class TX81ZLFOSnapshot(BaseModel, frozen=True):
    rate: int = Field(ge=0, le=255)
    sync: bool
    waveform: int = Field(ge=0, le=3)
    sample_rate: int = Field(gt=0)
    clock_hz: int = Field(gt=0)
    delay: int = Field(default=0, ge=0, le=99)
    delay_elapsed: int = Field(default=0, ge=0)
    frame: int = Field(default=0, ge=0)
    counter: int = Field(default=0, ge=0, lt=2**32)
    secondary_counter: int = Field(default=0, ge=0, lt=2**32)
    secondary_rate: int | None = Field(default=None, ge=0, le=255)
    secondary_waveform: int = Field(default=0, ge=0, le=3)
    secondary_sync: bool = False
    secondary_delay: int = Field(default=0, ge=0, le=99)
    secondary_delay_elapsed: int = Field(default=0, ge=0)
    secondary_wave_pitch: int = Field(default=0, ge=-128, le=127)
    secondary_wave_amplitude: int = Field(default=0, ge=0, le=255)
    secondary_delay_progress: int = Field(default=1, ge=0)
    clock_remainder: int = Field(default=0, ge=0)
    noise_lfsr: int = Field(default=1, ge=0, lt=2**32)
    noise_table: list[Annotated[int, Field(ge=0, le=255)]] = Field(
        default_factory=lambda: [0] * 256, min_length=256, max_length=256
    )
    wave_pitch: int = Field(default=0, ge=-128, le=127)
    wave_amplitude: int = Field(default=0, ge=0, le=255)
    delay_progress: int = Field(default=0, ge=0)

    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)


class TX81ZLFOContext:
    """OPZ hardware LFOs sharing one chip clock and noise table.

    Integer rows contain raw signed PM, raw unsigned AM, and the numerator and
    denominator of patch-delay progress. Progress is latched with the waveform
    at chip observations. Notes apply their own depths to these shared rows.
    The clock and noise model follow ymfm provisionally.
    With a secondary voice, advance returns (frames, 2, 4) rows in LFO order;
    otherwise it returns (frames, 4), leaving LFO2 at reset settings.
    """

    def __init__(
        self,
        voice: tx81z.TX81ZVoice,
        sample_rate: int = 48000,
        clock_hz: int = 3579545,
        *,
        secondary: tx81z.TX81ZVoice | None = None,
    ) -> None:
        rate, _, _ = tx81z_lfo_registers(voice)
        self.state = TX81ZLFOSnapshot(
            rate=rate,
            sync=bool(voice.data[58]),
            waveform=voice.data[59],
            sample_rate=sample_rate,
            clock_hz=clock_hz,
            delay=voice.data[55],
            delay_progress=0 if voice.data[55] else 1,
            secondary_rate=tx81z_lfo_registers(secondary)[0] if secondary else None,
            secondary_waveform=secondary.data[59] if secondary else 0,
            secondary_sync=bool(secondary.data[58]) if secondary else False,
            secondary_delay=secondary.data[55] if secondary else 0,
            secondary_delay_progress=0 if secondary and secondary.data[55] else 1,
        )

    def snapshot(self) -> TX81ZLFOSnapshot:
        return self.state.model_copy(deep=True)

    def restore(self, snapshot: TX81ZLFOSnapshot) -> None:
        if (
            snapshot.rate != self.state.rate
            or snapshot.sync != self.state.sync
            or snapshot.waveform != self.state.waveform
            or snapshot.sample_rate != self.state.sample_rate
            or snapshot.clock_hz != self.state.clock_hz
            or snapshot.delay != self.state.delay
            or snapshot.secondary_rate != self.state.secondary_rate
            or snapshot.secondary_waveform != self.state.secondary_waveform
            or snapshot.secondary_sync != self.state.secondary_sync
            or snapshot.secondary_delay != self.state.secondary_delay
        ):
            raise EngineError(
                "TX81Z LFO snapshot belongs to a different clock definition"
            )
        self.state = snapshot.model_copy(deep=True)

    def keydown(self) -> None:
        updates = {"delay_elapsed": 0}
        if self.state.delay:
            updates["delay_progress"] = 0
        if self.state.sync:
            updates["counter"] = 0
        if self.state.secondary_rate is not None:
            updates["secondary_delay_elapsed"] = 0
            if self.state.secondary_delay:
                updates["secondary_delay_progress"] = 0
            if self.state.secondary_sync:
                updates["secondary_counter"] = 0
        self.state = self.state.model_copy(update=updates)

    def advance(self, frames: int) -> np.ndarray:
        if frames < 0:
            raise EngineError("TX81Z LFO frame count must be nonnegative")
        state = self.state
        counter, secondary = state.counter, state.secondary_counter
        remainder, noise = state.clock_remainder, state.noise_lfsr
        table = state.noise_table.copy()
        count = 1 if state.secondary_rate is None else 2
        pitch = [state.wave_pitch, state.secondary_wave_pitch]
        amplitude = [state.wave_amplitude, state.secondary_wave_amplitude]
        progress = [state.delay_progress, state.secondary_delay_progress]
        waveforms = [state.waveform, state.secondary_waveform]
        elapsed = [state.delay_elapsed, state.secondary_delay_elapsed]
        increment = (16 | (state.rate & 15)) << (state.rate >> 4)
        secondary_rate = state.secondary_rate or 0
        secondary_increment = (16 | (secondary_rate & 15)) << (secondary_rate >> 4)
        output = np.empty((frames, count, 4), dtype=np.int64)
        durations = [
            _TX_DELAY_TICKS[x] * state.sample_rate
            for x in (state.delay, state.secondary_delay)
        ]
        for frame in range(frames):
            ticks, remainder = divmod(
                remainder + state.clock_hz, 64 * state.sample_rate
            )
            for _ in range(ticks):
                for _ in range(2):
                    noise = (noise << 1) & (2**32 - 1)
                    noise |= ((noise >> 17) ^ (noise >> 14) ^ 1) & 1
                counter = (counter + increment) & (2**32 - 1)
                secondary = (secondary + secondary_increment) & (2**32 - 1)
                index = (counter >> 22) & 255
                noise_value = (noise >> 17) & 255
                table[(index + 1) & 255] = noise_value
                table[(((secondary >> 22) & 255) + 1) & 255] = noise_value
                for lfo in range(count):
                    index = ((secondary if lfo else counter) >> 22) & 255
                    match waveforms[lfo]:
                        case 0:
                            am, pm = index ^ 255, index
                        case 1:
                            am = 0 if index & 128 else 255
                            pm = am ^ 128
                        case 2:
                            am = (
                                (index << 1) if index & 128 else ((index ^ 255) << 1)
                            ) & 255
                            pm = am if index & 64 else (am ^ 255)
                        case _:
                            am = pm = table[index]
                    pitch[lfo] = pm if pm < 128 else pm - 256
                    amplitude[lfo] = am
                    duration = durations[lfo]
                    progress[lfo] = (
                        min(duration, max(0, 96 * (elapsed[lfo] + frame) - duration))
                        if duration
                        else 1
                    )
            for lfo in range(count):
                output[frame, lfo] = (
                    pitch[lfo],
                    amplitude[lfo],
                    progress[lfo],
                    max(1, durations[lfo]),
                )
        self.state = state.model_copy(
            update={
                "frame": state.frame + frames,
                "delay_elapsed": state.delay_elapsed + frames,
                "counter": counter,
                "secondary_counter": secondary,
                "clock_remainder": remainder,
                "noise_lfsr": noise,
                "noise_table": table,
                "wave_pitch": pitch[0],
                "wave_amplitude": amplitude[0],
                "delay_progress": progress[0],
                "secondary_wave_pitch": pitch[1],
                "secondary_wave_amplitude": amplitude[1],
                "secondary_delay_progress": progress[1],
                "secondary_delay_elapsed": elapsed[1] + frames if count == 2 else 0,
            }
        )
        return output[:, 0] if count == 1 else output


# Measured completion times in 1/48 second, with reported uncertainty of one tick.
# https://nornand.hatenablog.com/entry/2020/11/30/185005
_TX_DELAY_TICKS = [
    0,
    15,
    16,
    16,
    17,
    18,
    18,
    19,
    19,
    20,
    21,
    22,
    23,
    24,
    25,
    26,
    27,
    29,
    31,
    32,
    34,
    35,
    36,
    37,
    38,
    40,
    41,
    43,
    45,
    47,
    49,
    52,
    54,
    57,
    61,
    64,
    67,
    69,
    71,
    74,
    76,
    79,
    82,
    86,
    90,
    94,
    98,
    103,
    108,
    114,
    121,
    128,
    133,
    137,
    142,
    147,
    152,
    158,
    164,
    171,
    179,
    187,
    196,
    205,
    216,
    228,
    241,
    256,
    265,
    273,
    283,
    293,
    304,
    315,
    328,
    341,
    356,
    372,
    390,
    409,
    431,
    455,
    481,
    511,
    528,
    546,
    564,
    585,
    606,
    630,
    655,
    682,
    711,
    744,
    780,
    818,
    861,
    909,
    963,
    1022,
]
