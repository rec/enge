"""Dedicated compatible renderers for Yamaha's rate-and-level FM voices."""

from math import exp, log2, pi, sin

import numpy as np
from pydantic import BaseModel, ConfigDict, Field
from ufor.dx7 import DX7Operator, DX7Voice

from .synth import EngineError


class DX7Envelope(BaseModel):
    """A DX7 operator envelope in the chip's logarithmic level units."""

    rates: list[int] = Field(min_length=4, max_length=4)
    levels: list[int] = Field(min_length=4, max_length=4)
    output_level: int = Field(ge=0)
    rate_scaling: int = Field(ge=0)
    sample_rate: int = Field(gt=0)
    level: int = 0
    target: int = 0
    stage: int = 0
    increment: int = 0
    rising: bool = False
    held: bool = True

    @classmethod
    def start(
        cls,
        operator: DX7Operator,
        note: int,
        velocity: int,
        sample_rate: int,
    ) -> "DX7Envelope":
        output_level = _scale_output_level(operator.output_level)
        output_level += _key_scale(operator, note)
        output_level = min(127, output_level) * 32
        output_level += _velocity_scale(velocity, operator.key_velocity_sensitivity)
        envelope = cls(
            rates=operator.rates,
            levels=operator.levels,
            output_level=max(0, output_level),
            rate_scaling=_rate_scale(note, operator.rate_scaling),
            sample_rate=sample_rate,
        )
        envelope._advance(0)
        return envelope

    def release(self) -> None:
        if self.held:
            self.held = False
            self._advance(3)

    def sample(self) -> float:
        """Advance one sample and return linear operator gain."""
        if self.stage < 3 or (self.stage < 4 and not self.held):
            if self.rising:
                if self.level < 1716 << 16:
                    self.level = 1716 << 16
                self.level += (((17 << 24) - self.level) >> 24) * self.increment
                if self.level >= self.target:
                    self.level = self.target
                    self._advance(self.stage + 1)
            else:
                self.level -= self.increment
                if self.level <= self.target:
                    self.level = self.target
                    self._advance(self.stage + 1)
        return 2 ** ((self.level / 65536 - 14 * 256) / 256)

    def _advance(self, stage: int) -> None:
        self.stage = stage
        if stage == 4:
            return
        target_level = (_scale_output_level(self.levels[stage]) >> 1) * 64
        target_level += self.output_level - 4256
        self.target = max(16, target_level) << 16
        self.rising = self.target > self.level
        qrate = min(63, (self.rates[stage] * 41 >> 6) + self.rate_scaling)
        self.increment = round(
            ((4 + (qrate & 3)) << (8 + (qrate >> 2))) * 44100 / self.sample_rate
        )

    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)


class DX7Renderer:
    """Render one DX7 note with native algorithms and Yamaha envelope timing."""

    def __init__(
        self,
        voice: DX7Voice,
        note: int,
        velocity: int,
        sample_rate: int = 48000,
    ) -> None:
        if not 0 <= note <= 127:
            raise EngineError("DX7 note must be from 0 through 127")
        if not 0 <= velocity <= 127:
            raise EngineError("DX7 velocity must be from 0 through 127")
        if sample_rate <= 0:
            raise EngineError("DX7 sample rate must be positive")
        self.voice = voice
        self.note = note
        self.velocity = velocity
        self.sample_rate = sample_rate
        self.envelopes = [
            DX7Envelope.start(operator, note, velocity, sample_rate)
            for operator in voice.operators
        ]
        self.frequencies = np.array(
            [_operator_frequency(operator, note) for operator in voice.operators]
        )
        self.phases = np.zeros(6)
        self.feedback = 0.0
        self.order = _operator_order(voice)

    def release(self) -> None:
        for envelope in self.envelopes:
            envelope.release()

    def render(self, frames: int) -> np.ndarray:
        if frames < 0:
            raise EngineError("DX7 frame count must be nonnegative")
        output = np.empty((frames, 1))
        topology = self.voice.topology
        carriers = [operator - 1 for operator in topology.carriers]
        feedback_operator = topology.feedback - 1
        for frame in range(frames):
            values = np.zeros(6)
            for operator in self.order:
                modulation = sum(
                    values[edge.source - 1]
                    for edge in topology.edges
                    if edge.destination - 1 == operator
                )
                if operator == feedback_operator:
                    modulation += self.feedback * self.voice.feedback / 7
                gain = self.envelopes[operator].sample()
                values[operator] = gain * sin(
                    2 * pi * self.phases[operator] + modulation
                )
            output[frame, 0] = sum(values[carrier] for carrier in carriers)
            self.feedback = values[feedback_operator]
            self.phases = np.remainder(
                self.phases + self.frequencies / self.sample_rate, 1
            )
        return output


def _operator_order(voice: DX7Voice) -> list[int]:
    pending = [0] * 6
    successors = [[] for _ in range(6)]
    for edge in voice.topology.edges:
        source = edge.source - 1
        destination = edge.destination - 1
        pending[destination] += 1
        successors[source].append(destination)
    ready = [index for index, degree in enumerate(pending) if degree == 0]
    result: list[int] = []
    while ready:
        operator = ready.pop()
        result.append(operator)
        for destination in successors[operator]:
            pending[destination] -= 1
            if pending[destination] == 0:
                ready.append(destination)
    if len(result) != 6:
        raise EngineError("DX7 algorithm contains a current-sample cycle")
    return result


def _operator_frequency(operator: DX7Operator, note: int) -> float:
    note_frequency = 440 * 2 ** ((note - 69) / 12)
    detune = 0.0209 * exp(-0.396 * log2(note_frequency)) * (operator.detune - 7) / 7
    if operator.frequency_mode.value == "fixed":
        return 10 ** ((operator.coarse * 100 + operator.fine) / 100) * (1 + detune)
    ratio = 0.5 if operator.coarse == 0 else operator.coarse
    return note_frequency * ratio * (1 + operator.fine / 100) * (1 + detune)


def _scale_output_level(level: int) -> int:
    if level >= 20:
        return 28 + level
    levels = [
        0,
        5,
        9,
        13,
        17,
        20,
        23,
        25,
        27,
        29,
        31,
        33,
        35,
        37,
        39,
        41,
        42,
        43,
        45,
        46,
    ]
    return levels[level]


def _velocity_scale(velocity: int, sensitivity: int) -> int:
    values = [
        0,
        70,
        86,
        97,
        106,
        114,
        121,
        126,
        132,
        138,
        142,
        148,
        152,
        156,
        160,
        163,
        166,
        170,
        173,
        174,
        178,
        181,
        184,
        186,
        189,
        190,
        194,
        196,
        198,
        200,
        202,
        205,
        206,
        209,
        211,
        214,
        216,
        218,
        220,
        222,
        224,
        225,
        227,
        229,
        230,
        232,
        233,
        235,
        237,
        238,
        240,
        241,
        242,
        243,
        244,
        246,
        246,
        248,
        249,
        250,
        251,
        252,
        253,
        254,
    ]
    return ((sensitivity * (values[velocity >> 1] - 239) + 7) >> 3) << 4


def _rate_scale(note: int, sensitivity: int) -> int:
    return sensitivity * min(31, max(0, note // 3 - 7)) >> 3


def _key_scale(operator: DX7Operator, note: int) -> int:
    offset = note - operator.breakpoint - 17
    if offset >= 0:
        return _key_curve((offset + 1) // 3, operator.right_depth, operator.right_curve)
    return _key_curve((-offset + 1) // 3, operator.left_depth, operator.left_curve)


def _key_curve(group: int, depth: int, curve: int) -> int:
    if curve in (0, 3):
        scale = group * depth * 329 >> 12
    else:
        exponent = [
            0,
            1,
            2,
            3,
            4,
            5,
            6,
            7,
            8,
            9,
            11,
            14,
            16,
            19,
            23,
            27,
            33,
            39,
            47,
            56,
            66,
            80,
            94,
            110,
            126,
            142,
            158,
            174,
            190,
            206,
            222,
            238,
            250,
        ][min(group, 32)]
        scale = exponent * depth * 329 >> 15
    return -scale if curve < 2 else scale
