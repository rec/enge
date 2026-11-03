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


class DX7PitchEnvelope(BaseModel):
    """Four-stage pitch envelope, in Q24 octaves with a fractional tick carry."""

    rates: list[int] = Field(min_length=4, max_length=4)
    levels: list[int] = Field(min_length=4, max_length=4)
    sample_rate: int = Field(gt=0)
    level: int
    stage: int = 0
    held: bool = True
    remainder: int = 0

    def release(self) -> None:
        if self.held:
            self.held = False
            self.stage = 3

    def sample(self) -> float:
        if self.stage < 3 or (self.stage == 3 and not self.held):
            target = _PITCH_LEVELS[self.levels[self.stage]] * 2**19
            unit = int(64 * 2**24 / (21.3 * self.sample_rate) + 0.5)
            increment, self.remainder = divmod(
                _PITCH_RATES[self.rates[self.stage]] * unit + self.remainder, 64
            )
            if self.level < target:
                self.level = min(target, self.level + increment)
            else:
                self.level = max(target, self.level - increment)
            if self.level == target:
                self.stage += 1
        return 2 ** (self.level / 2**24)

    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)


class DX7Snapshot(BaseModel, frozen=True):
    voice: DX7Voice
    note: int
    velocity: int
    sample_rate: int
    envelopes: list[DX7Envelope] = Field(min_length=6, max_length=6)
    pitch_envelope: DX7PitchEnvelope
    phases: list[float] = Field(min_length=6, max_length=6)
    feedback: list[float] = Field(min_length=2, max_length=2)

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
        if (diagnostic := voice.parameter_diagnostic) is not None:
            raise EngineError(diagnostic)
        if any(v > 99 for o in voice.operators for v in (*o.rates, *o.levels)):
            raise EngineError("DX7 envelope rates and levels must be from 0 through 99")
        if not 0 <= note <= 127:
            raise EngineError("DX7 note must be from 0 through 127")
        if not 0 <= velocity <= 127:
            raise EngineError("DX7 velocity must be from 0 through 127")
        if sample_rate <= 0:
            raise EngineError("DX7 sample rate must be positive")
        pitch_start = 102 if voice.packed else 126
        pitch = list(voice.data[pitch_start : pitch_start + 8])
        if any(v > 99 for v in pitch):
            raise EngineError(
                "DX7 pitch envelope rates and levels must be from 0 through 99"
            )
        self.pitch_envelope = DX7PitchEnvelope(
            rates=pitch[:4],
            levels=pitch[4:],
            sample_rate=sample_rate,
            level=_PITCH_LEVELS[pitch[7]] * 2**19,
        )
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
        self.feedback = [0.0, 0.0]
        self.order = _operator_order(voice)

    def snapshot(self) -> DX7Snapshot:
        return DX7Snapshot(
            voice=self.voice.model_copy(deep=True),
            note=self.note,
            velocity=self.velocity,
            sample_rate=self.sample_rate,
            envelopes=[e.model_copy(deep=True) for e in self.envelopes],
            pitch_envelope=self.pitch_envelope.model_copy(deep=True),
            phases=self.phases.tolist(),
            feedback=self.feedback.copy(),
        )

    def restore(self, snapshot: DX7Snapshot) -> None:
        if (
            snapshot.voice.data != self.voice.data
            or snapshot.note != self.note
            or snapshot.velocity != self.velocity
            or snapshot.sample_rate != self.sample_rate
        ):
            raise EngineError("DX7 snapshot belongs to a different note definition")
        self.envelopes = [e.model_copy(deep=True) for e in snapshot.envelopes]
        self.pitch_envelope = snapshot.pitch_envelope.model_copy(deep=True)
        self.phases = np.array(snapshot.phases)
        self.feedback = snapshot.feedback.copy()

    def release(self) -> None:
        self.pitch_envelope.release()
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
            pitch_ratio = self.pitch_envelope.sample()
            values = np.zeros(6)
            for operator in self.order:
                modulation = sum(
                    values[edge.source - 1]
                    for edge in topology.edges
                    if edge.destination - 1 == operator
                )
                if operator == feedback_operator and self.voice.feedback:
                    modulation += sum(self.feedback) / 2 ** (9 - self.voice.feedback)
                gain = self.envelopes[operator].sample()
                values[operator] = gain * sin(
                    2 * pi * (self.phases[operator] + modulation)
                )
            output[frame, 0] = sum(values[carrier] for carrier in carriers)
            self.feedback[0] = self.feedback[1]
            self.feedback[1] = float(values[feedback_operator])
            self.phases = np.remainder(
                self.phases + self.frequencies * pitch_ratio / self.sample_rate, 1
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
    if operator.frequency_mode.value == "fixed":
        # MSFA's fixed-mode logarithmic detune is key-independent and unipolar.
        detune_octaves = 13457 * max(0, operator.detune - 7) / 2**24
        return (
            10 ** (((operator.coarse & 3) * 100 + operator.fine) / 100)
            * 2**detune_octaves
        )
    note_frequency = 440 * 2 ** ((note - 69) / 12)
    detune = 0.0209 * exp(-0.396 * log2(note_frequency)) * (operator.detune - 7) / 7
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


# Measured DX7 rate/level data reported by MSFA, pitchenv.cc:
# https://github.com/google/music-synthesizer-for-android/blob/master/app/src/main/jni/pitchenv.cc
_PITCH_RATES = [
    1,
    2,
    3,
    3,
    4,
    4,
    5,
    5,
    6,
    6,
    7,
    7,
    8,
    8,
    9,
    9,
    10,
    10,
    11,
    11,
    12,
    12,
    13,
    13,
    14,
    14,
    15,
    16,
    16,
    17,
    18,
    18,
    19,
    20,
    21,
    22,
    23,
    24,
    25,
    26,
    27,
    28,
    30,
    31,
    33,
    34,
    36,
    37,
    38,
    39,
    41,
    42,
    44,
    46,
    47,
    49,
    51,
    53,
    54,
    56,
    58,
    60,
    62,
    64,
    66,
    68,
    70,
    72,
    74,
    76,
    79,
    82,
    85,
    88,
    91,
    94,
    98,
    102,
    106,
    110,
    115,
    120,
    125,
    130,
    135,
    141,
    147,
    153,
    159,
    165,
    171,
    178,
    185,
    193,
    202,
    211,
    232,
    243,
    254,
    255,
]
_PITCH_LEVELS = [
    -128,
    -116,
    -104,
    -95,
    -85,
    -76,
    -68,
    -61,
    -56,
    -52,
    -49,
    -46,
    -43,
    -41,
    -39,
    -37,
    -35,
    -33,
    -32,
    -31,
    -30,
    -29,
    -28,
    -27,
    -26,
    -25,
    -24,
    -23,
    -22,
    -21,
    -20,
    -19,
    -18,
    -17,
    -16,
    -15,
    -14,
    -13,
    -12,
    -11,
    -10,
    -9,
    -8,
    -7,
    -6,
    -5,
    -4,
    -3,
    -2,
    -1,
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
    10,
    11,
    12,
    13,
    14,
    15,
    16,
    17,
    18,
    19,
    20,
    21,
    22,
    23,
    24,
    25,
    26,
    27,
    28,
    29,
    30,
    31,
    32,
    33,
    34,
    35,
    38,
    40,
    43,
    46,
    49,
    53,
    58,
    65,
    73,
    82,
    92,
    103,
    115,
    127,
]
