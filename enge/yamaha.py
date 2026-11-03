"""Dedicated compatible renderers for Yamaha's rate-and-level FM voices."""

from math import exp, log2, pi, sin

import numpy as np
from pydantic import BaseModel, ConfigDict, Field
from ufor import tx81z
from ufor.dx7 import DX7Operator, DX7Voice

from .synth import EngineError


class TX81ZState(BaseModel, frozen=True):
    """Numeric operator-bank continuation, separate from patch/envelope state."""

    phases: list[float] = Field(
        default_factory=lambda: [0.0] * 4, min_length=4, max_length=4
    )
    feedback: list[float] = Field(
        default_factory=lambda: [0.0] * 2, min_length=2, max_length=2
    )

    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)


class TX81ZEnvelope(BaseModel):
    """OPZ attenuation state advanced by the shared envelope counter.

    Counter ticks are chip envelope cycles, not output audio frames. The host
    owns this clock and supplies the same counter to every operator.
    """

    rates: list[int] = Field(min_length=4, max_length=4)
    sustain: int = Field(ge=0, le=1023)
    shift: int = Field(ge=0, le=3)
    attenuation: int = Field(default=1023, ge=0, le=1023)
    stage: int = Field(default=0, ge=0, le=3)

    @classmethod
    def start(
        cls,
        operator: tx81z.TX81ZOperator,
        additional: tx81z.TX81ZAdditionalOperator,
        key_code: int,
    ) -> "TX81ZEnvelope":
        if operator.number != additional.number:
            raise EngineError("TX81Z envelope requires matching VCED/ACED operators")
        if not 0 <= key_code <= 31:
            raise EngineError("TX81Z key code must be from 0 through 31")
        scaling = key_code >> (operator.rate_scaling ^ 3)
        rates = [
            min(63, r + scaling) if r else 0
            for r in (
                operator.attack_rate * 2,
                operator.decay_1_rate * 2,
                operator.decay_2_rate * 2,
                operator.release_rate * 4 + 2,
            )
        ]
        sustain = 15 - operator.decay_1_level
        sustain |= (sustain + 1) & 16
        return cls(
            rates=rates,
            sustain=sustain * 32,
            shift=0 if operator.number == 1 else additional.eg_shift,
            attenuation=0 if rates[0] >= 62 else 1023,
        )

    def release(self) -> None:
        self.stage = 3

    def clock(self, counter: int) -> int:
        """Advance one shared envelope observation and return shifted attenuation."""
        if counter < 0:
            raise EngineError("TX81Z envelope counter must be nonnegative")
        if self.stage == 0 and self.attenuation == 0:
            self.stage = 1
        if self.stage == 1 and self.attenuation >= self.sustain:
            self.stage = 2
        rate = self.rates[self.stage]
        shift = rate >> 2
        scaled = counter << shift
        if scaled & 2047 == 0:
            index = (scaled >> max(11, shift)) & 7
            increment = (_TX_INCREMENTS[rate] >> (4 * index)) & 15
            if self.stage == 0:
                if rate < 62:
                    self.attenuation += (~self.attenuation * increment) >> 4
            else:
                self.attenuation = min(1023, self.attenuation + increment)
        return self.attenuation >> self.shift

    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)


class TX81ZEnvelopeBank(BaseModel):
    """Four envelopes on one clock; preserve counter/remainder across host calls.

    OPZ envelope ticks divide the input clock by 2 * 32 * 3. Rows observe the
    current attenuation before advancing one output-frame interval.
    """

    envelopes: list[TX81ZEnvelope] = Field(min_length=4, max_length=4)
    sample_rate: int = Field(default=48000, gt=0)
    clock_hz: int = Field(default=3579545, gt=0)
    counter: int = Field(default=0, ge=0)
    remainder: int = Field(default=0, ge=0)

    def release(self) -> None:
        for envelope in self.envelopes:
            envelope.release()

    def attenuation_samples(self, frames: int) -> np.ndarray:
        if frames < 0:
            raise EngineError("TX81Z frame count must be nonnegative")
        output = np.empty((frames, 4), dtype=np.int64)
        divisor = 192 * self.sample_rate
        for frame in range(frames):
            output[frame] = [e.attenuation >> e.shift for e in self.envelopes]
            ticks, self.remainder = divmod(self.remainder + self.clock_hz, divisor)
            for _ in range(ticks):
                self.counter += 1
                for envelope in self.envelopes:
                    envelope.clock(self.counter)
        return output

    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)


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


def tx81z_operator_samples(
    algorithm: int,
    waveforms: list[int],
    state: TX81ZState,
    frequencies: np.ndarray,
    gains: np.ndarray,
    sample_rate: int = 48000,
    feedback: int = 0,
) -> tuple[np.ndarray, TX81ZState]:
    """Render prepared OP1..OP4 Hz/gain rows and return raw carrier mix/state.

    Gains scale normalized chip operator output. Routing uses four cycles per
    full-scale modulation output; DAC saturation and patch-level scaling belong
    to the enclosing voice. This is the provisional floating-point OPZ model.
    """
    if not 1 <= algorithm <= 8 or not 0 <= feedback <= 7:
        raise EngineError("Invalid TX81Z algorithm or feedback")
    if len(waveforms) != 4 or any(w not in range(8) for w in waveforms):
        raise EngineError("TX81Z requires four waveform numbers from 0 through 7")
    if (
        sample_rate <= 0
        or frequencies.ndim != 2
        or frequencies.shape[1] != 4
        or gains.shape != frequencies.shape
        or not np.all(np.isfinite(frequencies))
        or not np.all(np.isfinite(gains))
        or np.any(frequencies <= 0)
        or np.any(gains < 0)
    ):
        raise EngineError("TX81Z frequencies/gains require finite (frames, 4) rows")
    edges, carriers = _TX_ALGORITHMS[algorithm - 1]
    phases = np.array(state.phases)
    history = state.feedback.copy()
    output = np.empty((len(frequencies), 1))
    for frame in range(len(frequencies)):
        values = [0.0] * 4
        for operator in (3, 2, 1, 0):
            modulation = 4 * sum(values[s - 1] for s, d in edges if d - 1 == operator)
            if operator == 3 and feedback:
                modulation += sum(history) * 2 ** (feedback - 7)
            values[operator] = gains[frame, operator] * float(
                tx81z_wave_samples(
                    waveforms[operator], np.asarray(phases[operator] + modulation)
                )
            )
        output[frame, 0] = sum(values[c - 1] for c in carriers)
        history = [history[1], values[3]]
        phases = np.remainder(phases + frequencies[frame] / sample_rate, 1)
    return output, TX81ZState(phases=phases.tolist(), feedback=history)


def tx81z_base_frequency(
    operator: tx81z.TX81ZOperator, additional: tx81z.TX81ZAdditionalOperator, note: int
) -> float:
    """Prepare nominal Hz before chip detune, LFO modulation and phase quantization."""
    if operator.number != additional.number:
        raise EngineError("TX81Z frequency requires matching VCED/ACED operators")
    if not 0 <= note <= 127:
        raise EngineError("TX81Z note must be from 0 through 127")
    if additional.fixed_frequency:
        return (
            max(0.5, operator.coarse >> 2) * 16 + additional.fine
        ) * 2**additional.fixed_frequency_range
    _, multiple, cents = _TX_RATIO_COARSE[operator.coarse]
    if multiple == 0.5 and additional.fine > 7:
        raise EngineError("TX81Z half-multiple ratios require fine from 0 through 7")
    ratio = (multiple + additional.fine / 16) * 2 ** (cents / 1200)
    return 440 * 2 ** ((note - 69) / 12) * ratio


def tx81z_wave_samples(waveform: int, phases: np.ndarray) -> np.ndarray:
    """Sample the provisional ymfm TX81Z shapes at phases measured in cycles.

    ACED waveform numbers are zero-based. This realizes the continuous shapes,
    not the chip's logarithmic lookup-table quantization or alias suppression.
    """
    if waveform not in range(8):
        raise EngineError("TX81Z waveform must be from 0 through 7")
    phase = np.remainder(phases, 1)
    signal = np.sin(2 * pi * phase * (2 if waveform >= 4 else 1))
    if waveform >= 6:
        signal = np.abs(signal)
    if waveform % 2:
        signal *= np.abs(signal)
    if waveform >= 2:
        signal = np.where(phase < 0.5, signal, 0)
    return signal


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


# TX81Z visible operator numbers; OP4 is the feedback operator in every graph.
_TX_ALGORITHMS = [
    ([(4, 3), (3, 2), (2, 1)], [1]),
    ([(4, 2), (3, 2), (2, 1)], [1]),
    ([(3, 2), (4, 1), (2, 1)], [1]),
    ([(4, 3), (3, 1), (2, 1)], [1]),
    ([(4, 3), (2, 1)], [1, 3]),
    ([(4, 3), (4, 2), (4, 1)], [1, 2, 3]),
    ([(4, 3)], [1, 2, 3]),
    ([], [1, 2, 3, 4]),
]
# VCED coarse indices follow the displayed ratios, including 14.10 before 14.13.
# Nominal synthesis uses OPZ's cent offsets rather than the rounded display.
_TX_RATIO_COARSE = sorted(
    (m * r, m, d)
    for m in [0.5, *range(1, 16)]
    for d, r in ((0, 1), (600, 1.41), (781, 1.57), (950, 1.73))
)
# OPZ attenuation increments reported by ymfm, ymfm_fm.ipp.
# Each hexadecimal nibble gives one step in the eight-counter sequence.
_TX_INCREMENTS = (
    [0, 0, 0x10101010, 0x10101010, 0x10101010, 0x10101010, 0x11101110, 0x11101110]
    + [0x10101010, 0x10111010, 0x11101110, 0x11111110] * 10
    + [
        0x11111111,
        0x21112111,
        0x21212121,
        0x22212221,
        0x22222222,
        0x42224222,
        0x42424242,
        0x44424442,
        0x44444444,
        0x84448444,
        0x84848484,
        0x88848884,
    ]
    + [0x88888888] * 4
)
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
