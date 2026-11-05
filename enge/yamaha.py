"""Dedicated compatible renderers for Yamaha's rate-and-level FM voices."""

from math import exp, isfinite, log2, pi, sin
from pathlib import Path

import numpy as np
from pydantic import BaseModel, ConfigDict, Field
from ufor import tx81z, wt11
from ufor.dx7 import DX7Operator, DX7Voice

from .synth import EngineError
from .yamaha_portamento import TX81ZPortamento


class TX81ZState(BaseModel, frozen=True):
    """OPZ Q20 phases and signed 14-bit feedback continuation."""

    phases: list[int] = Field(
        default_factory=lambda: [0] * 4, min_length=4, max_length=4
    )
    feedback: list[int] = Field(
        default_factory=lambda: [0] * 2, min_length=2, max_length=2
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
    reverb_rate: int = Field(default=0, ge=0, le=63)
    attenuation: int = Field(default=1023, ge=0, le=1023)
    stage: int = Field(default=0, ge=0, le=4)

    @classmethod
    def start(
        cls,
        operator: tx81z.TX81ZOperator,
        additional: tx81z.TX81ZAdditionalOperator,
        key_code: int,
        reverb: int = 0,
    ) -> "TX81ZEnvelope":
        if operator.number != additional.number:
            raise EngineError("TX81Z envelope requires matching VCED/ACED operators")
        if not 0 <= key_code <= 31:
            raise EngineError("TX81Z key code must be from 0 through 31")
        if not 0 <= reverb <= 7:
            raise EngineError("TX81Z reverb rate must be from 0 through 7")
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
            reverb_rate=min(rates[3], min(63, reverb * 4 + 2 + scaling))
            if reverb
            else 0,
            attenuation=0 if rates[0] >= 62 else 1023,
        )

    def release(self) -> None:
        if self.stage < 3:
            self.stage = 3

    def clock(self, counter: int) -> int:
        """Advance one shared envelope observation and return shifted attenuation."""
        if counter < 0:
            raise EngineError("TX81Z envelope counter must be nonnegative")
        if self.stage == 0 and self.attenuation == 0:
            self.stage = 1
        if self.stage == 1 and self.attenuation >= self.sustain:
            self.stage = 2
        rate = self.reverb_rate if self.stage == 4 else self.rates[self.stage]
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
                if self.stage == 3 and self.reverb_rate and self.attenuation >= 192:
                    self.stage = 4
        return self.attenuation >> self.shift

    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)


class TX81ZEnvelopeBank(BaseModel):
    """Four envelopes on one clock; preserve counter/remainder across host calls.

    OPZ envelopes advance once per three chip ticks (192 master clocks).
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
        """Observe raw attenuation at the start of each host frame."""
        if frames < 0:
            raise EngineError("TX81Z frame count must be nonnegative")
        return self._attenuation_boundaries(frames, self.clock_hz)[:-1]

    def attenuation_ticks(self, ticks: int) -> np.ndarray:
        """Observe raw attenuation after each completed native chip clock."""
        if ticks < 0:
            raise EngineError("TX81Z tick count must be nonnegative")
        return self._attenuation_boundaries(ticks, 64 * self.sample_rate)[1:]

    def _attenuation_boundaries(self, count: int, increment: int) -> np.ndarray:
        output = np.empty((count + 1, 4), dtype=np.int64)
        output[0] = [e.attenuation for e in self.envelopes]
        divisor = 192 * self.sample_rate
        for index in range(1, count + 1):
            ticks, self.remainder = divmod(self.remainder + increment, divisor)
            for _ in range(ticks):
                self.counter += 1
                for envelope in self.envelopes:
                    envelope.clock(self.counter)
            output[index] = [e.attenuation for e in self.envelopes]
        return output

    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)


class TX81ZSnapshot(BaseModel, frozen=True):
    voice: tx81z.TX81ZVoice
    additional: tx81z.TX81ZAdditional
    note: int
    velocity: int
    sample_rate: int
    frame: int = Field(ge=0)
    held_output: float
    pitch_hz: float | None = None
    start_pitch_hz: float | None = None
    portamento: TX81ZPortamento | None = None
    envelopes: TX81ZEnvelopeBank
    operators: TX81ZState

    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)


class TX81ZRenderer:
    """Offline provisional four-operator note renderer with explicit edit buffers.

    LFO observations carry signed PM, unsigned AM, a delay numerator and a
    delay denominator. Optional controller-depth rows hold
    summed integer PMD/AMD byte contributions in 0..255. These add to delayed
    patch bytes and saturate before register conversion. Controller modulation
    bypasses patch delay; conversion follows the provisional firmware model.
    Intrinsic key scaling
    uses the transposed key; snapshots retain the original trigger key.
    Native chip ticks clock the envelopes, Q20 phases and feedback before output.
    Optional render tuning rows are cents relative to the admitted pitch, applied
    to ratio operators before their additive chip-detune offsets.
    Breath rows are normalized positions, truncated to CC2 before measured
    EG-bias lookup. An omitted breath input means position zero.
    start_pitch_hz is the previous sounding fundamental after transpose and
    octave wrapping; portamento state uses those same sounding-Hz coordinates.
    """

    def __init__(
        self,
        voice: tx81z.TX81ZVoice,
        additional: tx81z.TX81ZAdditional,
        note: int,
        velocity: int,
        sample_rate: int = 48000,
        pitch_hz: float | None = None,
        start_pitch_hz: float | None = None,
        *,
        start_frame: int = 0,
    ) -> None:
        for diagnostic in (voice.parameter_diagnostic, additional.parameter_diagnostic):
            if diagnostic is not None:
                raise EngineError(diagnostic)
        for i in range(4):
            if voice.data[i * 13 + 8] > 1:
                raise EngineError(
                    f"TX81Z operator {4 - i} amplitude modulation enable must be 0 or 1"
                )
            if additional.data[i * 5] > 1:
                raise EngineError(
                    f"TX81Z operator {4 - i} fixed frequency flag must be 0 or 1"
                )
        if (
            not 0 <= note <= 127
            or not 0 <= velocity <= 127
            or sample_rate <= 0
            or start_frame < 0
        ):
            raise EngineError("Invalid TX81Z note, velocity or sample rate")
        _, self.lfo_pitch_depth, self.lfo_amplitude_depth = tx81z_lfo_registers(voice)
        for index, maximum in enumerate([7, 99, 99], start=20):
            if additional.data[index] > maximum:
                raise EngineError(
                    f"TX81Z ACED parameter {index} must be from 0 through {maximum}"
                )
        self.pitch_modulation_sensitivity = voice.data[60]
        self.amplitude_modulation_sensitivity = voice.data[61]
        self.amplitude_modulation_enabled = np.array(
            [o.amplitude_modulation_enable for o in voice.operators]
        )
        sounding_note = note + voice.data[62] - 24
        if sounding_note < 13:
            sounding_note = 13 + (sounding_note - 13) % 12
        elif sounding_note > 108:
            sounding_note = 97 + (sounding_note - 97) % 12
        self.sounding_note = sounding_note
        key_code = (sounding_note - 13) // 3
        self.voice = voice.model_copy(deep=True)
        self.additional = additional.model_copy(deep=True)
        self.note = note
        self.velocity = velocity
        self.sample_rate = sample_rate
        self.frame = start_frame
        self.held_output = 0.0
        if pitch_hz is not None and (not isfinite(pitch_hz) or pitch_hz <= 0):
            raise EngineError("Yamaha pitch_hz must be finite and positive")
        self.pitch_hz = pitch_hz
        self.sounding_pitch_hz = (
            pitch_hz if pitch_hz is not None else 440 * 2 ** ((note - 69) / 12)
        ) * 2 ** ((sounding_note - note) / 12)
        if start_pitch_hz is not None and (
            not isfinite(start_pitch_hz) or start_pitch_hz <= 0
        ):
            raise EngineError("TX81Z start_pitch_hz must be finite and positive")
        self.start_pitch_hz = start_pitch_hz
        self.portamento = (
            TX81ZPortamento.start(
                voice.data[66],
                sample_rate,
                start_pitch_hz,
                self.sounding_pitch_hz,
            )
            if start_pitch_hz is not None and voice.data[66]
            else None
        )
        self.envelopes = TX81ZEnvelopeBank(
            envelopes=[
                TX81ZEnvelope.start(
                    voice.operator(i),
                    additional.operator(i),
                    key_code,
                    additional.data[20],
                )
                for i in range(1, 5)
            ],
            sample_rate=sample_rate,
        )
        completed = start_frame * self.envelopes.clock_hz // (64 * sample_rate)
        self.envelopes.counter = completed // 3
        self.envelopes.remainder = completed % 3 * 64 * sample_rate
        frequencies = []
        detune_offsets = []
        for number in range(1, 5):
            operator = voice.operator(number)
            extension = additional.operator(number)
            frequency = tx81z_base_frequency(operator, extension, sounding_note)
            offset = 0.0
            if not extension.fixed_frequency:
                native = _tx81z_ratio_steps(
                    operator, extension, sounding_note, np.zeros(1, dtype=np.int64)
                )[0] * (self.envelopes.clock_hz / (64 * 2**20))
                offset = native - frequency
                if pitch_hz is not None:
                    frequency *= pitch_hz / (440 * 2 ** ((note - 69) / 12))
            frequencies.append(frequency + offset)
            detune_offsets.append(offset)
        self.frequencies = np.array(frequencies)
        self.detune_offsets = np.array(detune_offsets)
        self.ratio_operators = [
            not additional.operator(i).fixed_frequency for i in range(1, 5)
        ]
        self.levels = np.array(
            [
                tx81z_output_gain(
                    voice.operator(i), voice.algorithm, sounding_note, velocity
                )
                for i in range(1, 5)
            ]
        )
        self.waveforms = [additional.operator(i).waveform for i in range(1, 5)]
        self.operators = TX81ZState()

    def render(
        self,
        frames: int,
        modulation: np.ndarray | tuple[np.ndarray, np.ndarray] | None = None,
        tuning_cents: np.ndarray | None = None,
        controller_depths: np.ndarray | None = None,
        breath: np.ndarray | None = None,
        foot: np.ndarray | None = None,
        portamento_enabled: np.ndarray | None = None,
    ) -> np.ndarray:
        """Render raw audio; controller_depths are summed byte contributions.

        The two columns carry integer pitch/amplitude contributions in 0..255,
        before adding delayed patch depth and converting to OPZ registers.
        A modulation array supplies one value per host frame. The pair from
        TX81ZLFOContext.advance_ticks supplies individual chip observations.
        start_frame places a note on the instrument's absolute host timeline.
        """
        tuning = _tuning_ratios(tuning_cents, frames)
        if frames < 0:
            raise EngineError("TX81Z frame count must be nonnegative")
        divisor = 64 * self.sample_rate
        remainder = self.frame * self.envelopes.clock_hz % divisor
        ends = (
            np.arange(1, frames + 1) * self.envelopes.clock_hz + remainder
        ) // divisor
        ticks = int(ends[-1]) if frames else 0
        owners = np.repeat(np.arange(frames), np.diff(ends, prepend=0))
        if portamento_enabled is not None and (
            portamento_enabled.shape != (frames,)
            or not np.all(np.isfinite(portamento_enabled))
            or np.any((portamento_enabled < 0) | (portamento_enabled > 1))
        ):
            raise EngineError("TX81Z portamento rows must have shape (frames,) in 0..1")
        if breath is not None and (
            breath.shape != (frames,)
            or not np.all(np.isfinite(breath))
            or np.any((breath < 0) | (breath > 1))
        ):
            raise EngineError("TX81Z breath rows must have shape (frames,) in 0..1")
        if breath is not None and self.voice.data[75] != 50 and np.any(breath):
            raise EngineError("TX81Z breath pitch bias is not implemented")
        if foot is not None and (
            foot.shape != (frames,)
            or not np.all(np.isfinite(foot))
            or np.any((foot < 0) | (foot > 1))
        ):
            raise EngineError("TX81Z foot rows must have shape (frames,) in 0..1")
        if controller_depths is not None and (
            controller_depths.shape != (frames, 2)
            or not np.all(np.isfinite(controller_depths))
            or np.any(controller_depths != np.floor(controller_depths))
            or np.any((controller_depths < 0) | (controller_depths > 255))
        ):
            raise EngineError(
                "TX81Z controller depths must be integer contributions "
                "(frames, 2) in 0..255"
            )
        if modulation is None:
            if (
                (
                    self.lfo_pitch_depth
                    and self.pitch_modulation_sensitivity
                    and any(self.ratio_operators)
                )
                or (
                    self.lfo_amplitude_depth
                    and self.amplitude_modulation_sensitivity
                    and np.any(self.amplitude_modulation_enabled)
                )
                or (controller_depths is not None and np.any(controller_depths))
            ):
                raise EngineError("TX81Z modulation requires shared LFO rows")
        if modulation is not None:
            raw = _tx81z_modulation_ticks(modulation, ends, owners, ticks)
            # uint64 keeps the maximum permitted rational denominator exact.
            delay = (
                raw[:, 2].astype(np.uint64) * 255 // raw[:, 3].astype(np.uint64)
            ).astype(np.int64)
            pitch_depth = ((self.voice.data[56] * 165 >> 6) * delay) >> 8
            amplitude_depth = ((self.voice.data[57] * 165 >> 6) * delay) >> 8
            if controller_depths is not None:
                contributions = controller_depths[owners].astype(np.int64)
                pitch_depth += contributions[:, 0]
                amplitude_depth += contributions[:, 1]
            pitch_depth = np.minimum(pitch_depth, 255) >> 1
            amplitude_depth = (
                np.asarray(_TX_LFO_CURVE)[255 - np.minimum(amplitude_depth, 255)] >> 1
            )
            pm = raw[:, 0] * pitch_depth >> 7
            am = raw[:, 1] * amplitude_depth >> 7
        if self.portamento is not None:
            if (
                portamento_enabled is not None
                and not (self.voice.data[63] and self.voice.data[65])
                and (disabled := np.flatnonzero(portamento_enabled < 0.5)).size
            ):
                at = int(disabled[0])
                glide = np.ones(frames)
                glide[:at] = self.portamento.ratios(at)
                self.portamento.current_hz = self.portamento.target_hz
                self.portamento.offset4 = 0 if self.portamento.ascending else -256
                self.portamento.remainder = 0
            else:
                glide = self.portamento.ratios(frames)
            tuning = glide if tuning is None else tuning * glide
        frequencies = np.broadcast_to(self.frequencies, (ticks, 4))
        detune_offsets = np.broadcast_to(self.detune_offsets, (ticks, 4))
        attenuation = self.envelopes.attenuation_ticks(ticks)
        quiet = attenuation > 896
        attenuation >>= np.array([e.shift for e in self.envelopes.envelopes])
        if modulation is not None:
            sensitivity = self.pitch_modulation_sensitivity
            if sensitivity:
                delta = (
                    pm >> (6 - sensitivity)
                    if sensitivity < 6
                    else pm << (sensitivity - 5)
                )
                frequencies = frequencies.copy()
                detune_offsets = detune_offsets.copy()
                for i in range(4):
                    if self.ratio_operators[i]:
                        operator = self.voice.operator(i + 1)
                        extension = self.additional.operator(i + 1)
                        base = _tx81z_ratio_steps(
                            operator, extension, self.sounding_note, delta, detune=False
                        ) * (self.envelopes.clock_hz / (64 * 2**20))
                        native = _tx81z_ratio_steps(
                            operator, extension, self.sounding_note, delta
                        ) * (self.envelopes.clock_hz / (64 * 2**20))
                        detune_offsets[:, i] = native - base
                        if self.pitch_hz is not None:
                            base *= self.pitch_hz / (440 * 2 ** ((self.note - 69) / 12))
                        frequencies[:, i] = base + detune_offsets[:, i]
            if self.amplitude_modulation_sensitivity:
                am = am << (self.amplitude_modulation_sensitivity - 1)
                attenuation += am[:, None] * self.amplitude_modulation_enabled
        if tuning is not None:
            tuning = tuning[owners]
            frequencies = frequencies.copy()
            for i in range(4):
                if self.ratio_operators[i]:
                    frequencies[:, i] = (
                        frequencies[:, i] - detune_offsets[:, i]
                    ) * tuning + detune_offsets[:, i]
        levels = self.levels
        if self.voice.data[76]:
            cc = (
                np.zeros(frames, dtype=np.int64)
                if breath is None
                else (127 * breath).astype(np.int64)
            )
            bias = np.column_stack(
                [
                    _TX_BREATH_ATTENUATION[
                        self.voice.data[76], o.eg_bias_sensitivity, cc
                    ]
                    for o in self.voice.operators
                ]
            )
            levels = np.maximum(
                2 ** (-127 / 8), self.levels * 2 ** (-bias.astype(float) / 8)
            )
        if amount := self.voice.data[67]:
            cc = (
                np.zeros(frames, dtype=np.int64)
                if foot is None
                else (127 * foot).astype(np.int64)
            )
            # Provisional FC range mapping onto measured master-volume TL.
            volume = 127 - (127 - cc) * amount // 99
            carriers = np.array(
                [i in _TX_ALGORITHMS[self.voice.algorithm - 1][1] for i in range(1, 5)]
            )
            attenuation_tl = np.asarray(_TX_VOLUME_ATTENUATION)[volume, None] * carriers
            levels = np.maximum(
                2 ** (-127 / 8), levels * 2 ** (-attenuation_tl.astype(float) / 8)
            )
        gains = 2 ** (-attenuation / 64) * (
            levels[owners] if levels.ndim == 2 else levels
        )
        gains[quiet] = 0
        steps = np.rint(
            np.remainder(frequencies / (self.envelopes.clock_hz / 64), 1) * 2**20
        ).astype(np.int64)
        output, self.operators = tx81z_operator_ticks(
            self.voice.algorithm,
            self.waveforms,
            self.operators,
            steps,
            gains,
            self.voice.feedback,
        )
        observed = np.concatenate([[[self.held_output]], output])[ends]
        if ticks:
            self.held_output = float(output[-1, 0])
        self.frame += frames
        return observed

    def retarget(self, note: int, velocity: int, pitch_hz: float | None = None) -> None:
        """Select a legato key without restarting operators or envelope progress.

        Key-scaled rates and levels follow the selected key provisionally; the
        shared envelope counter, attenuation, stages and oscillator state persist.
        """
        start = (
            self.portamento.current_hz
            if self.portamento is not None
            else self.sounding_pitch_hz
        )
        target = TX81ZRenderer(
            self.voice,
            self.additional,
            note,
            velocity,
            self.sample_rate,
            pitch_hz,
            start,
        )
        self.note = target.note
        self.sounding_note = target.sounding_note
        self.velocity = target.velocity
        self.pitch_hz = target.pitch_hz
        self.sounding_pitch_hz = target.sounding_pitch_hz
        self.start_pitch_hz = target.start_pitch_hz
        self.portamento = target.portamento
        self.frequencies = target.frequencies
        self.detune_offsets = target.detune_offsets
        self.levels = target.levels
        for current, prepared in zip(
            self.envelopes.envelopes, target.envelopes.envelopes, strict=True
        ):
            current.rates = prepared.rates
            current.reverb_rate = prepared.reverb_rate

    def release(self) -> None:
        self.envelopes.release()

    def snapshot(self) -> TX81ZSnapshot:
        return TX81ZSnapshot(
            voice=self.voice.model_copy(deep=True),
            additional=self.additional.model_copy(deep=True),
            note=self.note,
            velocity=self.velocity,
            sample_rate=self.sample_rate,
            frame=self.frame,
            held_output=self.held_output,
            pitch_hz=self.pitch_hz,
            start_pitch_hz=self.start_pitch_hz,
            portamento=self.portamento.model_copy(deep=True)
            if self.portamento is not None
            else None,
            envelopes=self.envelopes.model_copy(deep=True),
            operators=self.operators.model_copy(deep=True),
        )

    def restore(self, snapshot: TX81ZSnapshot) -> None:
        if (
            snapshot.voice.data != self.voice.data
            or snapshot.additional.data != self.additional.data
            or snapshot.note != self.note
            or snapshot.velocity != self.velocity
            or snapshot.sample_rate != self.sample_rate
            or snapshot.pitch_hz != self.pitch_hz
            or snapshot.start_pitch_hz != self.start_pitch_hz
        ):
            raise EngineError("TX81Z snapshot belongs to a different note definition")
        bank = snapshot.envelopes
        completed = snapshot.frame * self.envelopes.clock_hz // (64 * self.sample_rate)
        if (
            bank.sample_rate != self.sample_rate
            or bank.clock_hz != self.envelopes.clock_hz
            or bank.counter != completed // 3
            or bank.remainder != completed % 3 * 64 * self.sample_rate
        ):
            raise EngineError("TX81Z snapshot envelope clock disagrees with its frame")
        self.envelopes = snapshot.envelopes.model_copy(deep=True)
        self.frame = snapshot.frame
        self.held_output = snapshot.held_output
        self.operators = snapshot.operators.model_copy(deep=True)
        self.portamento = (
            snapshot.portamento.model_copy(deep=True)
            if snapshot.portamento is not None
            else None
        )


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
    hold_frames: int = Field(default=0, ge=0)

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
            if self.hold_frames:
                self.hold_frames -= 1
                if not self.hold_frames:
                    self._advance(self.stage + 1)
            elif self.rising:
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
        self.hold_frames = 0
        if stage == 4:
            return
        target_level = (_scale_output_level(self.levels[stage]) >> 1) * 64
        target_level += self.output_level - 4256
        self.target = max(16, target_level) << 16
        self.rising = self.target > self.level
        if self.target == self.level or (stage == 0 and self.levels[0] == 0):
            rate = min(99, self.rates[stage] + self.rate_scaling)
            duration = _DX7_HOLD_SAMPLES[rate] if rate < 77 else 20 * (99 - rate)
            if rate < 77 and stage == 0 and self.levels[0] == 0:
                duration //= 20
            self.hold_frames = (duration * self.sample_rate + 44099) // 44100
        qrate = min(63, (self.rates[stage] * 41 >> 6) + self.rate_scaling)
        # Dexed includes LG_N=6 for its 64-frame updates; we advance every frame.
        self.increment = round(
            ((4 + (qrate & 3)) << (2 + (qrate >> 2))) * 44100 / self.sample_rate
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


class WT11Snapshot(TX81ZSnapshot, frozen=True):
    wt11_voice: wt11.WT11Voice
    wt11_additional: wt11.WT11Additional
    wt11_additional2: wt11.WT11Additional2
    pitch_envelope: DX7PitchEnvelope


class WT11Renderer(TX81ZRenderer):
    """WT11 pitch projection onto the shared OPZ core, provisionally.

    The envelope uses DXconvert's pinned three-stage-to-DX7 projection at the
    host frame rate. It affects ratio operators through the existing tuning
    path. Timing, fixed-mode interaction and level endpoints are unqualified.
    Active aftertouch assignments are explicitly unsupported in this slice.
    """

    def __init__(
        self,
        voice: wt11.WT11Voice,
        additional: wt11.WT11Additional,
        additional2: wt11.WT11Additional2,
        note: int,
        velocity: int,
        sample_rate: int = 48000,
        pitch_hz: float | None = None,
        start_pitch_hz: float | None = None,
        *,
        start_frame: int = 0,
    ) -> None:
        for diagnostic in (
            voice.parameter_diagnostic,
            additional.parameter_diagnostic,
            additional2.parameter_diagnostic,
        ):
            if diagnostic is not None:
                raise EngineError(diagnostic)
        assignment = additional2.aftertouch
        if (
            assignment.pitch
            or assignment.amplitude
            or assignment.eg_bias
            or assignment.pitch_bias != 50
        ):
            raise EngineError("WT11 aftertouch assignments are not implemented")
        data = bytearray(voice.data)
        extra = bytearray(additional.data)
        for index, source in enumerate((0, 2, 1, 3)):
            data[index * 13 : index * 13 + 13] = voice.data[
                source * 13 : source * 13 + 13
            ]
            extra[index * 5 : index * 5 + 5] = additional.data[
                source * 5 : source * 5 + 5
            ]
        # WT11 permits bias 100; the TX81Z core cannot interpret that domain.
        # Bias is guarded below, so the core receives only the neutral value.
        data[75] = 50
        super().__init__(
            tx81z.TX81ZVoice(data=bytes(data)),
            tx81z.TX81ZAdditional(data=bytes(extra)),
            note,
            velocity,
            sample_rate,
            pitch_hz,
            start_pitch_hz,
            start_frame=start_frame,
        )
        self.wt11_voice = voice.model_copy(deep=True)
        self.wt11_additional = additional.model_copy(deep=True)
        self.wt11_additional2 = additional2.model_copy(deep=True)
        pitch = voice.pitch_envelope
        self.pitch_envelope = DX7PitchEnvelope(
            rates=[
                _WT11_PITCH_RATES[pitch.rates[0]],
                min(98, _WT11_PITCH_RATES[pitch.rates[1]]),
                99,
                _WT11_PITCH_RATES[pitch.rates[2]],
            ],
            levels=[pitch.levels[0], pitch.levels[1], pitch.levels[1], pitch.levels[2]],
            sample_rate=sample_rate,
            level=_PITCH_LEVELS[pitch.levels[2]] * 2**19,
        )

    def render(
        self,
        frames: int,
        modulation: np.ndarray | tuple[np.ndarray, np.ndarray] | None = None,
        tuning_cents: np.ndarray | None = None,
        controller_depths: np.ndarray | None = None,
        breath: np.ndarray | None = None,
        foot: np.ndarray | None = None,
        portamento_enabled: np.ndarray | None = None,
    ) -> np.ndarray:
        if frames < 0:
            raise EngineError("WT11 frame count must be nonnegative")
        _tuning_ratios(tuning_cents, frames)
        if breath is not None and self.wt11_voice.data[75] != 50 and np.any(breath):
            raise EngineError("WT11 breath pitch bias is not implemented")
        pitch = self.pitch_envelope.model_copy(deep=True)
        cents = np.array([1200 * log2(pitch.sample()) for _ in range(frames)])
        if tuning_cents is not None:
            cents += tuning_cents
        output = super().render(
            frames,
            modulation,
            cents,
            controller_depths,
            breath,
            foot,
            portamento_enabled,
        )
        self.pitch_envelope = pitch
        return output

    def release(self) -> None:
        super().release()
        self.pitch_envelope.release()

    def snapshot(self) -> WT11Snapshot:
        return WT11Snapshot(
            **super().snapshot().model_dump(),
            wt11_voice=self.wt11_voice.model_copy(deep=True),
            wt11_additional=self.wt11_additional.model_copy(deep=True),
            wt11_additional2=self.wt11_additional2.model_copy(deep=True),
            pitch_envelope=self.pitch_envelope.model_copy(deep=True),
        )

    def restore(self, snapshot: TX81ZSnapshot) -> None:
        if (
            not isinstance(snapshot, WT11Snapshot)
            or snapshot.wt11_voice != self.wt11_voice
            or snapshot.wt11_additional != self.wt11_additional
            or snapshot.wt11_additional2 != self.wt11_additional2
            or snapshot.pitch_envelope.rates != self.pitch_envelope.rates
            or snapshot.pitch_envelope.levels != self.pitch_envelope.levels
            or snapshot.pitch_envelope.sample_rate != self.sample_rate
        ):
            raise EngineError("WT11 snapshot belongs to a different note definition")
        super().restore(snapshot)
        self.pitch_envelope = snapshot.pitch_envelope.model_copy(deep=True)


class DX7Snapshot(BaseModel, frozen=True):
    """Continuation with Q24 phase/increments and signed Q24 feedback."""

    voice: DX7Voice
    note: int
    velocity: int
    sample_rate: int
    pitch_hz: float | None = None
    envelopes: list[DX7Envelope] = Field(min_length=6, max_length=6)
    pitch_envelope: DX7PitchEnvelope
    phases: list[int] = Field(min_length=6, max_length=6)
    phase_steps: list[int] = Field(min_length=6, max_length=6)
    feedback: list[int] = Field(min_length=2, max_length=2)

    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)


class DX7Renderer:
    """Render one DX7 note with native algorithms and Yamaha envelope timing.

    Initial phases are cycles in operator order OP1 through OP6. Sync-on resets
    them; sync-off copies the supplied phases, or starts at zero when omitted.
    The caller owns idle oscillator evolution and subsequent note allocation.
    Optional render tuning rows are cents relative to the admitted pitch and
    affect ratio operators only. The caller supplies subsequent tuning rows
    after restoring a snapshot, just as it supplies shared modulation rows.
    Hardware pitch-bend cents rows affect both fixed and ratio operators.
    Controller rows contain normalized pitch depth, amplitude depth and EG
    attenuation after assignment combination; each column is in 0..1.
    """

    def __init__(
        self,
        voice: DX7Voice,
        note: int,
        velocity: int,
        sample_rate: int = 48000,
        pitch_hz: float | None = None,
        initial_phases: list[float] | None = None,
    ) -> None:
        if (diagnostic := voice.parameter_diagnostic) is not None:
            raise EngineError(diagnostic)
        if any(v > 99 for o in voice.operators for v in (*o.rates, *o.levels)):
            raise EngineError("DX7 envelope rates and levels must be from 0 through 99")
        parameters = dx7_global_parameters(voice)
        if not 0 <= note <= 127:
            raise EngineError("DX7 note must be from 0 through 127")
        sounding_note = note + parameters[8] - 24
        if not 0 <= sounding_note <= 127:
            raise EngineError("DX7 transposed note must be from 0 through 127")
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
        if pitch_hz is not None and (not isfinite(pitch_hz) or pitch_hz <= 0):
            raise EngineError("Yamaha pitch_hz must be finite and positive")
        self.pitch_hz = pitch_hz
        self.pitch_modulation_depth = parameters[3] * 165 >> 6
        self.pitch_modulation_sensitivity = [0, 10, 20, 33, 55, 92, 153, 255][
            parameters[7]
        ]
        # Hexter's provisional AMD fit, in 1/8-octave reduction units.
        depth = parameters[4]
        self.amplitude_modulation_level = (
            0.0
            if depth == 0
            else 52.75
            if depth == 99
            else round(exp(depth * 0.0428993 - 0.285189), 6)
        )
        self.amplitude_modulation_sensitivities = [
            _DX7_AMPLITUDE_SENSITIVITIES[o.amplitude_modulation_sensitivity]
            for o in voice.operators
        ]
        self.envelopes = [
            DX7Envelope.start(o, sounding_note, velocity, sample_rate)
            for o in voice.operators
        ]
        self.frequencies = np.array(
            [_operator_frequency(o, sounding_note) for o in voice.operators]
        )
        if pitch_hz is not None:
            for i, operator in enumerate(voice.operators):
                if operator.frequency_mode.value == "ratio":
                    self.frequencies[i] *= pitch_hz / (440 * 2 ** ((note - 69) / 12))
        if initial_phases is not None and (
            len(initial_phases) != 6
            or any(not isfinite(p) or not 0 <= p < 1 for p in initial_phases)
        ):
            raise EngineError("DX7 initial phases must be six finite values in [0, 1)")
        self.phases = (
            np.array([int(p * 2**24) for p in initial_phases], dtype=np.int64)
            if initial_phases is not None and not parameters[0]
            else np.zeros(6, dtype=np.int64)
        )
        self.feedback = [0, 0]
        self.order = _operator_order(voice)
        self.phase_steps = np.array(
            [
                round(float(f) * 2**24 / self.sample_rate) % 2**24
                for f in self.frequencies
            ],
            dtype=np.int64,
        )

    def snapshot(self) -> DX7Snapshot:
        return DX7Snapshot(
            voice=self.voice.model_copy(deep=True),
            note=self.note,
            velocity=self.velocity,
            sample_rate=self.sample_rate,
            pitch_hz=self.pitch_hz,
            envelopes=[e.model_copy(deep=True) for e in self.envelopes],
            pitch_envelope=self.pitch_envelope.model_copy(deep=True),
            phases=self.phases.tolist(),
            phase_steps=self.phase_steps.tolist(),
            feedback=self.feedback.copy(),
        )

    def restore(self, snapshot: DX7Snapshot) -> None:
        if (
            snapshot.voice.data != self.voice.data
            or snapshot.note != self.note
            or snapshot.velocity != self.velocity
            or snapshot.sample_rate != self.sample_rate
            or snapshot.pitch_hz != self.pitch_hz
        ):
            raise EngineError("DX7 snapshot belongs to a different note definition")
        self.envelopes = [e.model_copy(deep=True) for e in snapshot.envelopes]
        self.pitch_envelope = snapshot.pitch_envelope.model_copy(deep=True)
        self.phases = np.array(snapshot.phases)
        self.phase_steps = np.array(snapshot.phase_steps)
        self.feedback = snapshot.feedback.copy()

    def release(self) -> None:
        self.pitch_envelope.release()
        for envelope in self.envelopes:
            envelope.release()

    def render(
        self,
        frames: int,
        modulation: np.ndarray | None = None,
        tuning_cents: np.ndarray | None = None,
        pitch_bend_cents: np.ndarray | None = None,
        controllers: np.ndarray | None = None,
    ) -> np.ndarray:
        tuning = _tuning_ratios(tuning_cents, frames)
        bend = _tuning_ratios(pitch_bend_cents, frames)
        if frames < 0:
            raise EngineError("DX7 frame count must be nonnegative")
        if controllers is not None and (
            controllers.shape != (frames, 3)
            or not np.all(np.isfinite(controllers))
            or np.any((controllers < 0) | (controllers > 1))
        ):
            raise EngineError("DX7 controller rows must have shape (frames, 3) in 0..1")
        if modulation is None:
            if (
                (self.pitch_modulation_depth and self.pitch_modulation_sensitivity)
                or (
                    self.amplitude_modulation_level
                    and any(self.amplitude_modulation_sensitivities)
                )
                or (controllers is not None and np.any(controllers[:, :2]))
            ):
                raise EngineError("DX7 modulation requires shared LFO rows")
        elif (
            modulation.shape != (frames, 2)
            or not np.all(np.isfinite(modulation))
            or np.any((modulation < 0) | (modulation > 1))
        ):
            raise EngineError("DX7 shared LFO rows must have shape (frames, 2) in 0..1")
        output = np.empty((frames, 1))
        topology = self.voice.topology
        carriers = [operator - 1 for operator in topology.carriers]
        feedback_source = topology.feedback.source - 1
        feedback_destination = topology.feedback.destination - 1
        # Provisional Dexed Mk I scaling for the two long loops and algorithm 32.
        feedback_shift = (
            9 - self.voice.feedback + (2 if self.voice.algorithm in (4, 6, 32) else 0)
        )
        for frame in range(frames):
            pitch_ratio = self.pitch_envelope.sample()
            amplitude_reduction = (
                0.0 if controllers is None else 52.75 / 8 * controllers[frame, 2]
            )
            if modulation is not None:
                value, delay = modulation[frame]
                amplitude_reduction = max(
                    amplitude_reduction,
                    self.amplitude_modulation_level * value * delay / 8,
                )
                pitch_depth = self.pitch_modulation_depth * delay / 32768
                if controllers is not None:
                    pitch, amplitude, _ = controllers[frame]
                    amplitude_reduction = max(
                        amplitude_reduction, 52.75 / 8 * amplitude * value
                    )
                    pitch_depth = max(pitch_depth, pitch / 128)
                pitch_ratio *= 2 ** (
                    pitch_depth * self.pitch_modulation_sensitivity * (value - 0.5)
                )
            values = [0] * 6
            for operator in self.order:
                phase_modulation = sum(
                    values[edge.source - 1]
                    for edge in topology.edges
                    if edge.destination - 1 == operator
                )
                if operator == feedback_destination and self.voice.feedback:
                    phase_modulation += sum(self.feedback) >> feedback_shift
                envelope = self.envelopes[operator]
                envelope.sample()
                attenuation = 16384 - (envelope.level >> 14)
                if amplitude_reduction and (
                    sensitivity := self.amplitude_modulation_sensitivities[operator]
                ):
                    attenuation += int(amplitude_reduction * sensitivity * 1024)
                values[operator] = dx7_operator_sample(
                    int(self.phases[operator]) + phase_modulation, attenuation
                )
            output[frame, 0] = sum(values[carrier] for carrier in carriers) / 2**24
            self.feedback[0] = self.feedback[1]
            self.feedback[1] = values[feedback_source]
            steps = self.frequencies * pitch_ratio / self.sample_rate
            for i, operator in enumerate(self.voice.operators):
                if operator.frequency_mode.value == "fixed":
                    steps[i] = self.frequencies[i] / self.sample_rate
                elif tuning is not None:
                    steps[i] *= tuning[frame]
            if bend is not None:
                steps *= bend[frame]
            self.phase_steps = np.array(
                [round(float(s) * 2**24) % 2**24 for s in steps], dtype=np.int64
            )
            self.phases = (self.phases + self.phase_steps) % 2**24
        return output


def dx7_operator_sample(phase: int, attenuation: int) -> int:
    """Evaluate the provisional Mk I logarithmic sine in Q24 cycles/output.

    Attenuation is in 1/1024-octave steps. The quarter-wave lookup observes
    half-bin centers; negative output uses the chip model's one's complement.
    The reference's near-silence threshold is applied per frame.
    """
    if attenuation > 16284:
        return 0
    position = (phase >> 12) & 4095
    index = position & 1023
    if position & 1024:
        index = 1023 - index
    level = _DX7_LOG_SINE[index] + max(0, attenuation)
    magnitude = (4096 + _DX7_EXP[1023 - (level & 1023)]) >> (level >> 10)
    return (-magnitude - 1 if position & 2048 else magnitude) * 8192


def dx7_global_parameters(voice: DX7Voice) -> list[int]:
    """Decode and validate globals in unpacked parameter order 136..144."""
    if voice.packed:
        data = voice.data
        parameters = [
            (data[111] >> 3) & 1,
            *data[112:116],
            data[116] & 1,
            (data[116] >> 1) & 7,
            data[116] >> 4,
            data[117],
        ]
    else:
        parameters = list(voice.data[136:145])
        if any(voice.data[i + 17] > 1 for i in range(0, 126, 21)):
            raise EngineError("DX7 oscillator mode must be 0 or 1")
    for index, (value, maximum) in enumerate(
        zip(parameters, _DX7_GLOBAL_LIMITS, strict=True), start=136
    ):
        if value > maximum:
            raise EngineError(f"DX7 parameter {index} must be from 0 through {maximum}")
    return parameters


def tx81z_lfo_registers(
    voice: tx81z.TX81ZVoice | wt11.WT11Voice,
) -> tuple[int, int, int]:
    """Validate globals and prepare native LFS, PMD and AMD register values."""
    for index, maximum in enumerate(_TX_GLOBAL_LIMITS, start=54):
        if isinstance(voice, wt11.WT11Voice) and index == 75:
            maximum = 100
        if voice.data[index] > maximum:
            model = "WT11" if isinstance(voice, wt11.WT11Voice) else "TX81Z"
            raise EngineError(
                f"{model} VCED parameter {index} must be from 0 through {maximum}"
            )
    speed = voice.data[54]
    rate = (660 * speed) >> 8 if voice.data[59] == 3 else _TX_LFO_REGISTERS[speed][0]
    return (
        rate,
        _TX_LFO_REGISTERS[voice.data[56]][1],
        _TX_LFO_REGISTERS[voice.data[57]][2],
    )


def tx81z_detune_step(detune: int, key_code: int) -> int:
    """Return the signed OPZ phase-step offset before the operator multiplier.

    VCED detune is 0..6 with neutral 3. Key code is the chip's five-bit code,
    not a MIDI key. One unit is 1/2**20 cycle per native chip output interval.
    """
    if not 0 <= detune <= 6 or not 0 <= key_code <= 31:
        raise EngineError("Invalid TX81Z detune or chip key code")
    value = _TX_DETUNE_STEPS[key_code][abs(detune - 3)]
    return -value if detune < 3 else value


def tx81z_output_gain(
    operator: tx81z.TX81ZOperator, algorithm: int, note: int, velocity: int
) -> float:
    """Prepare total level from VCED, a MIDI key and note-on velocity.

    Sum output, carrier, key-level and velocity attenuation before TL saturation.
    Positive velocities follow androuet's measured tables; zero uses the
    provisional ROM-table extension. Envelope attenuation is separate; the
    renderer adds breath EG bias before TL saturation.
    """
    if not 1 <= algorithm <= 8 or not 0 <= note <= 127 or not 0 <= velocity <= 127:
        raise EngineError("Invalid TX81Z algorithm, key or note-on velocity")
    scaling_note = note if note <= 108 else 97 + (note - 109) % 12
    column = max(0, (scaling_note - 28) // 3)
    attenuation = _TX_OUTPUT_ATTENUATION[operator.output_level]
    attenuation += _TX_LEVEL_SCALING[operator.level_scaling][column]
    if sensitivity := operator.key_velocity_sensitivity:
        attenuation += _TX_VELOCITY_ATTENUATION[velocity][sensitivity - 1]
    if operator.number in _TX_ALGORITHMS[algorithm - 1][1]:
        attenuation += _TX_CARRIER_ATTENUATION[algorithm - 1]
    return 2 ** (-min(127, attenuation) / 8)


def tx81z_dac(samples: np.ndarray) -> np.ndarray:
    """Convert summed raw OPZ voices to full scale using pinned ymfm arithmetic.

    Raw notes use 8192 units per amplitude unit; the shared DAC output uses
    signed 16-bit full scale. Host fades round to integer units, ties to even.
    Positive overflow deliberately follows ymfm's roundtrip_fp helper.
    """
    if not np.all(np.isfinite(samples)):
        raise ValueError("TX81Z DAC samples must be finite")
    units = np.rint(np.clip(samples, -32769 / 8192, 4) * 8192).astype(np.int64)
    magnitude = units ^ (units >> 63)
    shift = np.maximum(0, np.frexp(magnitude)[1] - 9)
    decoded = (units >> shift) << shift
    decoded = np.where(units > 32767, 32767, np.maximum(decoded, -32768))
    return decoded / 32768


def tx81z_operator_samples(
    algorithm: int,
    waveforms: list[int],
    state: TX81ZState,
    frequencies: np.ndarray,
    gains: np.ndarray,
    sample_rate: int = 48000,
    feedback: int = 0,
) -> tuple[np.ndarray, TX81ZState]:
    """Render OP1..OP4 Hz/gain rows with the provisional integer OPZ core.

    Phase increments round to Q20 cycles and gains to 1/64-octave attenuation.
    Modulation uses integer phase indices and two signed feedback samples.
    Output is the raw carrier sum divided by 8192, before DAC saturation.
    """
    if (
        sample_rate <= 0
        or frequencies.ndim != 2
        or frequencies.shape[1] != 4
        or not np.all(np.isfinite(frequencies))
        or np.any(frequencies <= 0)
    ):
        raise EngineError("TX81Z requires positive frequencies and gains in 0..1")
    steps = np.rint(np.remainder(frequencies / sample_rate, 1) * 2**20).astype(np.int64)
    return _tx81z_operator_rows(
        algorithm, waveforms, state, steps, gains, feedback, native=False
    )


def tx81z_operator_ticks(
    algorithm: int,
    waveforms: list[int],
    state: TX81ZState,
    steps: np.ndarray,
    gains: np.ndarray,
    feedback: int = 0,
) -> tuple[np.ndarray, TX81ZState]:
    """Clock integer Q20 increments before observing each native operator row."""
    if (
        steps.ndim != 2
        or steps.shape[1] != 4
        or not np.issubdtype(steps.dtype, np.integer)
        or np.any(steps < 0)
    ):
        raise EngineError(
            "TX81Z native steps require four nonnegative integers per tick"
        )
    return _tx81z_operator_rows(
        algorithm, waveforms, state, steps, gains, feedback, native=True
    )


def tx81z_operator_output(waveform: int, phase: int, attenuation: int) -> int:
    """Evaluate an OPZ waveform at a 10-bit phase with 1/64-octave attenuation."""
    entry = _TX_WAVEFORMS[waveform][phase & 1023]
    level = (entry & 32767) + 4 * attenuation
    value = _TX_POWER[level & 255] >> (level >> 8)
    return -value if entry & 32768 else value


def tx81z_base_frequency(
    operator: tx81z.TX81ZOperator, additional: tx81z.TX81ZAdditionalOperator, note: int
) -> float:
    """Prepare Hz before chip detune: native ratio steps or nominal fixed Hz."""
    if operator.number != additional.number:
        raise EngineError("TX81Z frequency requires matching VCED/ACED operators")
    if not 0 <= note <= 127:
        raise EngineError("TX81Z note must be from 0 through 127")
    if operator.coarse < 4 and additional.fine > 7:
        raise EngineError(
            "TX81Z coarse values below four require fine from 0 through 7"
        )
    if additional.fixed_frequency:
        return (
            max(0.5, operator.coarse >> 2) * 16 + additional.fine
        ) * 2**additional.fixed_frequency_range
    return float(
        _tx81z_ratio_steps(
            operator, additional, note, np.zeros(1, dtype=np.int64), detune=False
        )[0]
    ) * (3579545 / (64 * 2**20))


def tx81z_wave_samples(waveform: int, phases: np.ndarray) -> np.ndarray:
    """Sample the quantized OPZ waveform at cycle phases, without attenuation."""
    if waveform not in range(8):
        raise EngineError("TX81Z waveform must be from 0 through 7")
    indices = np.floor(np.remainder(phases, 1) * 1024).astype(np.int64)
    table = np.array(
        [tx81z_operator_output(waveform, i, 0) / 8192 for i in range(1024)]
    )
    return table[indices]


def _tx81z_modulation_ticks(
    modulation: np.ndarray | tuple[np.ndarray, np.ndarray],
    ends: np.ndarray,
    owners: np.ndarray,
    ticks: int,
) -> np.ndarray:
    if isinstance(modulation, tuple):
        indices, rows = modulation
        if not np.array_equal(indices, ends) or rows.shape != (ticks + 1, 4):
            raise EngineError("Invalid TX81Z shared LFO tick grid")
    else:
        rows = modulation
        if rows.shape != (len(ends), 4):
            raise EngineError("Invalid TX81Z shared LFO rows")
    if (
        not np.all(np.isfinite(rows))
        or np.any(rows != np.floor(rows))
        or np.any((rows[:, 0] < -128) | (rows[:, 0] > 127))
        or np.any((rows[:, 1] < 0) | (rows[:, 1] > 255))
        or np.any((rows[:, 3] < 1) | (rows[:, 3] > 2**56))
        or np.any((rows[:, 2] < 0) | (rows[:, 2] > rows[:, 3]))
    ):
        raise EngineError("Invalid TX81Z shared LFO rows")
    return (rows[1:] if isinstance(modulation, tuple) else rows[owners]).astype(
        np.int64
    )


def _tx81z_operator_rows(
    algorithm: int,
    waveforms: list[int],
    state: TX81ZState,
    steps: np.ndarray,
    gains: np.ndarray,
    feedback: int,
    *,
    native: bool,
) -> tuple[np.ndarray, TX81ZState]:
    if not 1 <= algorithm <= 8 or not 0 <= feedback <= 7:
        raise EngineError("Invalid TX81Z algorithm or feedback")
    if len(waveforms) != 4 or any(w not in range(8) for w in waveforms):
        raise EngineError("TX81Z requires four waveform numbers from 0 through 7")
    if (
        gains.shape != steps.shape
        or not np.all(np.isfinite(gains))
        or np.any((gains < 0) | (gains > 1))
    ):
        raise EngineError("TX81Z requires positive frequencies and gains in 0..1")
    if not len(steps):
        return np.empty((0, 1)), state.model_copy(deep=True)
    edges, carriers = _TX_ALGORITHMS[algorithm - 1]
    mask = 2**20 - 1
    boundaries = np.empty((len(steps) + 1, 4), dtype=np.int64)
    boundaries[0] = [p & mask for p in state.phases]
    np.cumsum(steps.astype(np.int64) & mask, axis=0, out=boundaries[1:])
    boundaries[1:] += boundaries[0]
    boundaries &= mask
    phases = boundaries[1:] if native else boundaries[:-1]
    attenuation = np.rint(-64 * np.log2(np.maximum(gains, 2.0**-64))).astype(np.int64)
    history = state.feedback.copy()
    output = np.empty((len(steps), 1))
    for frame in range(len(steps)):
        values = [0] * 4
        for operator in (3, 2, 1, 0):
            modulation = sum(values[s - 1] for s, d in edges if d - 1 == operator) >> 1
            if operator == 3 and feedback:
                modulation += sum(history) >> (10 - feedback)
            values[operator] = tx81z_operator_output(
                waveforms[operator],
                (int(phases[frame, operator]) >> 10) + modulation,
                int(attenuation[frame, operator]),
            )
        output[frame, 0] = sum(values[c - 1] for c in carriers) / 8192
        history = [history[1], values[3]]
    return output, TX81ZState(phases=boundaries[-1].tolist(), feedback=history)


def _tx81z_ratio_steps(
    operator: tx81z.TX81ZOperator,
    additional: tx81z.TX81ZAdditionalOperator,
    note: int,
    delta: np.ndarray,
    *,
    detune: bool = True,
) -> np.ndarray:
    if note < 13:
        note = 13 + (note - 13) % 12
    elif note > 108:
        note = 97 + (note - 97) % 12
    _, multiple, cents = _TX_RATIO_COARSE[operator.coarse]
    position = np.clip((note - 13) * 64 + (cents * 64 + 50) // 100 + delta, 0, 6143)
    steps = _TX_PHASE_STEPS[position % 768] >> (7 - position // 768)
    if detune:
        steps = steps + tx81z_detune_step(operator.detune, (note - 13) // 3)
    return (steps * (int(multiple * 16) + additional.fine)) >> 4


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


def _tuning_ratios(cents: np.ndarray | None, frames: int) -> np.ndarray | None:
    if cents is None:
        return None
    if cents.shape != (frames,) or not np.all(np.isfinite(cents)):
        raise EngineError("Yamaha tuning rows must have shape (frames,) and be finite")
    with np.errstate(over="ignore", under="ignore"):
        ratios = np.exp2(cents / 1200)
    if np.any(~np.isfinite(ratios) | (ratios <= 0)):
        raise EngineError("Yamaha tuning ratio must be finite and positive")
    return ratios


def _operator_frequency(operator: DX7Operator, note: int) -> float:
    if operator.frequency_mode.value == "fixed":
        # MSFA's fixed-mode logarithmic detune is key-independent and unipolar.
        detune_octaves = 13457 * max(0, operator.detune - 7) / 2**24
        return (
            10 ** (((operator.coarse & 3) * 100 + operator.fine) / 100)
            * 2**detune_octaves
        )
    note_frequency = 440 * 2 ** ((note - 69) / 12)
    log_frequency = log2(note_frequency)
    # Dexed's measured fit scales log-frequency, not frequency directly.
    detune = (
        0.0209 * exp(-0.396 * log_frequency) * log_frequency * (operator.detune - 7) / 7
    )
    ratio = 0.5 if operator.coarse == 0 else operator.coarse
    return note_frequency * ratio * (1 + operator.fine / 100) * 2**detune


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


# OPZ key-code/detune magnitudes reported by ymfm's detune_adjustment.
_TX_DETUNE_STEPS = (
    [(0, 0, 1, 2)] * 4
    + [
        (0, 1, 2, 2),
        (0, 1, 2, 3),
        (0, 1, 2, 3),
        (0, 1, 2, 3),
        (0, 1, 2, 4),
        (0, 1, 3, 4),
        (0, 1, 3, 4),
        (0, 1, 3, 5),
        (0, 2, 4, 5),
        (0, 2, 4, 6),
        (0, 2, 4, 6),
        (0, 2, 5, 7),
        (0, 2, 5, 8),
        (0, 3, 6, 8),
        (0, 3, 6, 9),
        (0, 3, 7, 10),
        (0, 4, 8, 11),
        (0, 4, 8, 12),
        (0, 4, 9, 13),
        (0, 5, 10, 14),
        (0, 5, 11, 16),
        (0, 6, 12, 17),
        (0, 6, 13, 19),
        (0, 7, 14, 20),
    ]
    + [(0, 8, 16, 22)] * 4
)
# DXconvert's total-level curve and reported TX81Z carrier headroom measurements.
_TX_CARRIER_ATTENUATION = [0, 0, 0, 0, 8, 13, 13, 16]

# Master-volume register measurements, internal positions 0..127:
# https://nornand.hatenablog.com/entry/2020/11/21/201911
_TX_VOLUME_ATTENUATION = [
    127,
    102,
    90,
    83,
    77,
    72,
    68,
    65,
    62,
    60,
    57,
    55,
    53,
    51,
    50,
    48,
    47,
    45,
    44,
    43,
    42,
    41,
    40,
    39,
    38,
    37,
    36,
    35,
    34,
    33,
    33,
    32,
    31,
    30,
    30,
    29,
    28,
    28,
    27,
    27,
    26,
    25,
    25,
    24,
    24,
    23,
    23,
    22,
    22,
    21,
    21,
    21,
    20,
    20,
    19,
    19,
    18,
    18,
    18,
    17,
    17,
    16,
    16,
    16,
    15,
    15,
    15,
    14,
    14,
    14,
    13,
    13,
    13,
    12,
    12,
    12,
    11,
    11,
    11,
    10,
    10,
    10,
    10,
    9,
    9,
    9,
    9,
    8,
    8,
    8,
    8,
    7,
    7,
    7,
    7,
    6,
    6,
    6,
    6,
    5,
    5,
    5,
    5,
    4,
    4,
    4,
    4,
    3,
    3,
    3,
    3,
    3,
    2,
    2,
    2,
    2,
    2,
    1,
    1,
    1,
    1,
    1,
    0,
    0,
    0,
    0,
    0,
    0,
]
_TX_OUTPUT_ATTENUATION = [
    127,
    122,
    118,
    114,
    110,
    107,
    104,
    102,
    100,
    98,
    96,
    94,
    92,
    90,
    88,
    86,
    85,
    84,
    82,
    81,
] + list(range(79, -1, -1))
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
# Register synthesis quantizes these cent offsets to 1/64-semitone units.
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
# DXconvert fourop.py at 89aaf6cf48ff2eec7e11e82bb92e315b2e6e6621.
# Provisional conversion data, not measured WT11 timing.
_WT11_PITCH_RATES = [
    1,
    2,
    4,
    5,
    6,
    7,
    8,
    10,
    12,
    13,
    14,
    16,
    18,
    19,
    20,
    22,
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
    36,
    37,
    38,
    39,
    40,
    41,
    42,
    43,
    44,
    45,
    46,
    47,
    48,
    49,
    50,
    52,
    54,
    55,
    56,
    57,
    58,
    59,
    60,
    61,
    62,
    63,
    64,
    65,
    66,
    68,
    70,
    71,
    72,
    73,
    74,
    75,
    76,
    77,
    78,
    79,
    80,
    81,
    82,
    83,
    84,
    85,
    86,
    87,
    88,
    89,
    90,
    91,
    92,
    93,
    94,
    95,
    96,
    97,
    98,
    98,
    98,
    98,
    98,
    98,
    98,
    98,
    98,
    98,
    98,
    98,
    98,
    99,
]

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

# TX81Z register attenuation measurements by androuet (2020-11-21).
# https://nornand.hatenablog.com/entry/2020/11/21/201911
# Rows: LS 0..99; columns: keys <=30, 31..33, ..., 106..108.
# Each hex pair is one unsigned TL byte, preserving one measured row per line.
_TX_LEVEL_SCALING = [
    bytes.fromhex("000000000000000000000000000000000000000000000000000000"),
    bytes.fromhex("000000000000000000000000000000000000000000000000000000"),
    bytes.fromhex("000000000000000000000000000000000000000000010101010202"),
    bytes.fromhex("000000000000000000000000000000000000000101010102020303"),
    bytes.fromhex("000000000000000000000000000000000001010101020203030404"),
    bytes.fromhex("000000000000000000000000000000000101010102020303040505"),
    bytes.fromhex("000000000000000000000000000000010101010202030304050607"),
    bytes.fromhex("000000000000000000000000000001010102020203030405060708"),
    bytes.fromhex("000000000000000000000000000101010102020303040506070809"),
    bytes.fromhex("000000000000000000000000010101010202030304050607080a0b"),
    bytes.fromhex("000000000000000000000000010101010202030304050607090a0c"),
    bytes.fromhex("0000000000000000000000010101010202030304050607080a0c0d"),
    bytes.fromhex("0000000000000000000000010101010202030304050607090b0d0e"),
    bytes.fromhex("00000000000000000000010101010202030304050607080a0c0e10"),
    bytes.fromhex("00000000000000000000010101010202030404050607090b0d0f11"),
    bytes.fromhex("00000000000000000001010101020202030404050708090b0e1012"),
    bytes.fromhex("000000000000000000010101010202030304050607090a0c0f1114"),
    bytes.fromhex("000000000000000000010101010202030304050607090b0d0f1215"),
    bytes.fromhex("0000000000000000000101010202020304050607080a0c0e111416"),
    bytes.fromhex("0000000000000000010101010202030304050607080a0c0f111517"),
    bytes.fromhex("0000000000000000010101010202030304050607090b0d0f121619"),
    bytes.fromhex("00000000000000000101010202020304040607080a0b0e1014171a"),
    bytes.fromhex("00000000000000000101010202030304050607080a0c0e1114181b"),
    bytes.fromhex("00000000000000010101010202030304050607090a0d0f1215191d"),
    bytes.fromhex("00000000000000010101010202030304050607090b0d0f13161a1e"),
    bytes.fromhex("000000000000010101010202020304040507080a0b0e1014171c1f"),
    bytes.fromhex("000000000000010101010202030304050607080a0c0e1114181d21"),
    bytes.fromhex("000000000000010101010202030304050607090a0c0f1215191e22"),
    bytes.fromhex("000000000000010101010202030304050608090b0d0f12161a1f23"),
    bytes.fromhex("000000000001010101020202030404050608090b0d1013171b2024"),
    bytes.fromhex("0000000000010101010202020304040507080a0c0e1014181c2126"),
    bytes.fromhex("0000000000010101010202020304050607080a0c0e1114181d2227"),
    bytes.fromhex("0000000000010101010202030304050607090a0c0f1215191e2328"),
    bytes.fromhex("0000000000010101010202030304050607090b0d0f12161a1f252a"),
    bytes.fromhex("0000000001010101010202030304050607090b0d1013161b20262b"),
    bytes.fromhex("00000000010101010102020304040506080a0b0e1013171c21272c"),
    bytes.fromhex("00000000010101010102020304050507080a0c0e1114181c22282d"),
    bytes.fromhex("00000000010101010202020304050607080a0c0e1114181d23292f"),
    bytes.fromhex("00000000010101010202030304050607080a0c0f1115191e232a30"),
    bytes.fromhex("00000000010101010202030304050607090b0d0f12161a1f252b31"),
    bytes.fromhex("00000001010101010202030304050607090b0d1013161a20262d33"),
    bytes.fromhex("00000001010101010202030304050607090b0d1013171b20262d34"),
    bytes.fromhex("00000001010101010202030404050608090c0e1014171c21282f35"),
    bytes.fromhex("000000010101010102030304040607080a0c0e1114181c22283036"),
    bytes.fromhex("000000010101010102030304050607080a0c0e1114181d23293138"),
    bytes.fromhex("000000010101010202030304050607080a0c0f1215191e242b3239"),
    bytes.fromhex("000000010101010202030304050607080a0d0f12151a1e242b333a"),
    bytes.fromhex("000000010101010202030304050607090b0d0f12161a1f252c343c"),
    bytes.fromhex("000000010101010202030304050607090b0d1013161b20262d353d"),
    bytes.fromhex("000000010101010202030304050608090b0e1013171b20272e373e"),
    bytes.fromhex("000001010101020202030404050708090b0e1014171c21282f383f"),
    bytes.fromhex("000001010101020202030404050708090c0e1114181c2228303941"),
    bytes.fromhex("0000010101010202020304040607080a0c0e1114181d2329313a42"),
    bytes.fromhex("0000010101010202020304050607080a0c0f1115191e232a323b43"),
    bytes.fromhex("0000010101010202020304050607080a0c0f1215191e242b333c45"),
    bytes.fromhex("0000010101010202030304050607090a0c0f12161a1f242c343d46"),
    bytes.fromhex("0000010101010202030304050607090a0d1012161a1f252d353f47"),
    bytes.fromhex("0000010101010202030304050607090b0d1013161b20262d363f48"),
    bytes.fromhex("0000010101020202030404050608090b0d1013171b20262e37414a"),
    bytes.fromhex("0000010101020202030404050608090b0d1013171c21272f38424b"),
    bytes.fromhex("0000010101020202030404050608090b0e1114181c21283039434c"),
    bytes.fromhex("00000101010202020304040507080a0b0e1114181d2229313a444e"),
    bytes.fromhex("00000101010202020304040507080a0c0e1114181d2329313b454f"),
    bytes.fromhex("00000101010202020304050607080a0c0e1215191e232a323c4650"),
    bytes.fromhex("00000101010202020304050607090a0c0f1215191e242b333d4852"),
    bytes.fromhex("00000101010202020304050607090a0c0f12151a1e242b343d4953"),
    bytes.fromhex("00000101010202020304050607090a0c0f12161a1f252c353f4a54"),
    bytes.fromhex("00010101020202030304050607090b0d0f13161a1f252d353f4b55"),
    bytes.fromhex("00010101020202030304050607090b0d1013161b20262d36404c57"),
    bytes.fromhex("00010101020202030304050607090b0d1013171b20272e37414d58"),
    bytes.fromhex("00010101020202030304050608090b0d1014171c21272f38424e59"),
    bytes.fromhex("000101010202020303050506080a0b0d1014171c21282f3943505b"),
    bytes.fromhex("000101010202020303050506080a0b0e1014181c2228303944505c"),
    bytes.fromhex("000101010202020304050506080a0c0e1114181d2229313a45525d"),
    bytes.fromhex("000101010202020304050507080a0c0e1115181d2329313b46535e"),
    bytes.fromhex("000101010202030304050607080a0c0e1115191e232a323c475460"),
    bytes.fromhex("000101010202030304050607080a0c0e1115191e242b333c485561"),
    bytes.fromhex("000101010202030304050607080a0c0f1216191e242b333d495662"),
    bytes.fromhex("000101010202030304050607090a0c0f12161a1f252c343e4a5764"),
    bytes.fromhex("000101010202030304050607090b0d0f12161a1f252c353f4b5865"),
    bytes.fromhex("000101020202030304050607090b0d0f12161a20262d35404c5a66"),
    bytes.fromhex("000101020202030304050607090b0d0f13171b20262d36414d5b67"),
    bytes.fromhex("000101020202030304050607090b0d1013171b20272e37414e5c69"),
    bytes.fromhex("000101020202030304050607090b0d1013171b21272f37424f5d6a"),
    bytes.fromhex("000101020202030304050608090b0d1013181c21282f3843505e6b"),
    bytes.fromhex("000101020202030304050608090b0e1014181c2228303944515f6d"),
    bytes.fromhex("000101020203030304060608090c0e1014181c222930394552606e"),
    bytes.fromhex("0001010202030303040607080a0c0e1114181d2329313a4653626f"),
    bytes.fromhex("0001010202030303040607080a0c0e1114191d2329313b46536270"),
    bytes.fromhex("0001010202030304040607080a0c0e1115191d232a323b47546472"),
    bytes.fromhex("0001010202030304040607080a0c0e1115191e242b333c48566573"),
    bytes.fromhex("0001010202030304050607080a0c0f11151a1e242b333d49566674"),
    bytes.fromhex("0001010202030304050607080a0c0f12151a1f252b343e4a576776"),
    bytes.fromhex("0001010202030304050607080a0d0f12151a1f252c343e4a586877"),
    bytes.fromhex("0001010202030304050607080a0d0f12161a1f252c353f4b596978"),
    bytes.fromhex("0001010202030304050607090a0d0f12161b1f262d353f4c5a6a79"),
    bytes.fromhex("0001010202030304050607090b0d0f12161b20262d36404d5b6c7b"),
    bytes.fromhex("0001010202030304050607090b0d1013161b20272e37414e5c6d7c"),
    bytes.fromhex("0001010202030304050607090b0d1013171c20272e37414e5d6e7d"),
    bytes.fromhex("0001010202030304050607090b0d1013171c21272f38424f5e6f7f"),
]

# Rows: velocities 0..127; columns: KVS 1..7 (KVS 0 adds no TL).
# Zero provisionally applies the published KVS formula to ROM table entry FE.
# See plan/yamaha-emulation.md for the source audit and qualification boundary.
_TX_VELOCITY_ATTENUATION = [
    [22, 37, 52, 67, 81, 96, 111],
    [18, 29, 40, 51, 62, 73, 84],
    [17, 28, 38, 48, 58, 69, 79],
    [17, 27, 37, 47, 56, 66, 76],
    [17, 26, 36, 45, 55, 64, 74],
    [16, 25, 34, 44, 53, 62, 71],
    [16, 25, 34, 43, 51, 60, 69],
    [16, 24, 33, 41, 50, 58, 67],
    [15, 24, 32, 40, 48, 57, 65],
    [15, 23, 31, 39, 47, 55, 63],
    [15, 23, 30, 38, 46, 54, 62],
    [15, 22, 30, 38, 45, 53, 60],
    [14, 22, 29, 37, 44, 51, 59],
    [14, 21, 28, 36, 43, 50, 57],
    [14, 21, 28, 35, 42, 49, 56],
    [14, 21, 27, 34, 41, 48, 55],
    [14, 20, 27, 34, 40, 47, 53],
    [13, 20, 26, 33, 39, 46, 52],
    [13, 20, 26, 32, 38, 45, 51],
    [13, 19, 25, 32, 38, 44, 50],
    [13, 19, 25, 31, 37, 43, 49],
    [13, 19, 24, 30, 36, 42, 48],
    [13, 18, 24, 30, 35, 41, 47],
    [13, 18, 24, 29, 35, 40, 46],
    [12, 18, 23, 29, 34, 40, 45],
    [12, 18, 23, 28, 34, 39, 44],
    [12, 17, 23, 28, 33, 38, 43],
    [12, 17, 22, 27, 32, 37, 42],
    [12, 17, 22, 27, 32, 37, 42],
    [12, 17, 21, 26, 31, 36, 41],
    [12, 17, 21, 26, 31, 36, 40],
    [12, 16, 21, 26, 30, 35, 39],
    [12, 16, 21, 25, 30, 34, 39],
    [11, 16, 20, 25, 29, 33, 38],
    [11, 16, 20, 24, 29, 33, 37],
    [11, 15, 20, 24, 28, 32, 36],
    [11, 15, 19, 24, 28, 32, 36],
    [11, 15, 19, 23, 27, 31, 35],
    [11, 15, 19, 23, 27, 31, 35],
    [11, 15, 19, 23, 26, 30, 34],
    [11, 15, 18, 22, 26, 30, 33],
    [11, 14, 18, 22, 25, 29, 33],
    [11, 14, 18, 22, 25, 29, 32],
    [11, 14, 18, 21, 25, 28, 32],
    [10, 14, 17, 21, 24, 28, 31],
    [10, 14, 17, 21, 24, 27, 31],
    [10, 14, 17, 20, 23, 27, 30],
    [10, 13, 17, 20, 23, 26, 29],
    [10, 13, 16, 20, 23, 26, 29],
    [10, 13, 16, 19, 22, 25, 28],
    [10, 13, 16, 19, 22, 25, 28],
    [10, 13, 16, 19, 22, 25, 28],
    [10, 13, 16, 19, 21, 24, 27],
    [10, 13, 15, 18, 21, 24, 26],
    [10, 12, 15, 18, 20, 23, 26],
    [10, 12, 15, 18, 20, 23, 25],
    [10, 12, 15, 17, 20, 22, 25],
    [10, 12, 15, 17, 20, 22, 25],
    [9, 12, 14, 17, 19, 22, 24],
    [9, 12, 14, 17, 19, 21, 24],
    [9, 12, 14, 16, 19, 21, 23],
    [9, 12, 14, 16, 18, 21, 23],
    [9, 11, 13, 16, 18, 20, 22],
    [9, 11, 13, 15, 17, 19, 21],
    [9, 11, 13, 15, 17, 19, 21],
    [9, 11, 13, 15, 17, 19, 21],
    [9, 11, 13, 15, 16, 18, 20],
    [9, 11, 12, 14, 16, 18, 20],
    [9, 11, 12, 14, 16, 18, 19],
    [9, 10, 12, 14, 15, 17, 19],
    [9, 10, 12, 14, 15, 17, 18],
    [9, 10, 12, 13, 15, 16, 18],
    [9, 10, 12, 13, 15, 16, 18],
    [8, 10, 11, 13, 14, 16, 17],
    [8, 10, 11, 13, 14, 15, 17],
    [8, 10, 11, 12, 14, 15, 16],
    [8, 10, 11, 12, 13, 15, 16],
    [8, 9, 11, 12, 13, 14, 15],
    [8, 9, 10, 12, 13, 14, 15],
    [8, 9, 10, 11, 12, 13, 14],
    [8, 9, 10, 11, 12, 13, 14],
    [8, 9, 10, 11, 12, 13, 14],
    [8, 9, 10, 11, 11, 12, 13],
    [8, 9, 9, 10, 11, 12, 13],
    [8, 9, 9, 10, 11, 12, 12],
    [8, 8, 9, 10, 10, 11, 12],
    [8, 8, 9, 10, 10, 11, 11],
    [8, 8, 9, 10, 10, 11, 11],
    [8, 8, 9, 9, 10, 10, 11],
    [8, 8, 9, 9, 10, 10, 11],
    [7, 8, 8, 9, 9, 10, 10],
    [7, 8, 8, 9, 9, 9, 10],
    [7, 8, 8, 9, 9, 9, 10],
    [7, 8, 8, 8, 9, 9, 9],
    [7, 8, 8, 8, 8, 9, 9],
    [7, 7, 8, 8, 8, 8, 8],
    [7, 7, 7, 8, 8, 8, 8],
    [7, 7, 7, 8, 8, 8, 8],
    [7, 7, 7, 7, 7, 7, 7],
    [7, 7, 7, 7, 7, 7, 7],
    [7, 7, 7, 7, 7, 7, 7],
    [7, 7, 7, 7, 7, 7, 7],
    [7, 7, 7, 7, 6, 6, 6],
    [7, 7, 7, 7, 6, 6, 6],
    [7, 7, 6, 6, 6, 6, 6],
    [7, 7, 6, 6, 6, 6, 5],
    [7, 7, 6, 6, 6, 6, 5],
    [7, 6, 6, 6, 5, 5, 5],
    [7, 6, 6, 6, 5, 5, 5],
    [7, 6, 6, 6, 5, 5, 4],
    [7, 6, 6, 5, 5, 4, 4],
    [7, 6, 6, 5, 5, 4, 4],
    [7, 6, 6, 5, 5, 4, 4],
    [6, 6, 5, 5, 4, 4, 3],
    [6, 6, 5, 5, 4, 4, 3],
    [6, 6, 5, 5, 4, 3, 3],
    [6, 6, 5, 5, 4, 3, 3],
    [6, 6, 5, 4, 4, 3, 2],
    [6, 6, 5, 4, 4, 3, 2],
    [6, 6, 5, 4, 3, 3, 2],
    [6, 6, 5, 4, 3, 3, 2],
    [6, 5, 5, 4, 3, 2, 1],
    [6, 5, 5, 4, 3, 2, 1],
    [6, 5, 4, 4, 3, 2, 1],
    [6, 5, 4, 4, 3, 2, 1],
    [6, 5, 4, 3, 2, 1, 0],
    [6, 5, 4, 3, 2, 1, 0],
    [6, 5, 4, 3, 2, 1, 0],
]

# VCED global parameters 54..76, from the TX81Z manual's edit-buffer table.
_TX_GLOBAL_LIMITS = [
    99,  # LFO speed
    99,  # LFO delay
    99,  # Pitch modulation depth
    99,  # Amplitude modulation depth
    1,  # LFO sync
    3,  # LFO waveform
    7,  # Pitch modulation sensitivity
    3,  # Amplitude modulation sensitivity
    48,  # Transpose
    1,  # Poly/mono
    12,  # Pitch bend range
    1,  # Portamento mode
    99,  # Portamento time
    99,  # Foot control volume
    1,  # Sustain
    1,  # Portamento
    1,  # Chorus
    99,  # Modulation wheel pitch
    99,  # Modulation wheel amplitude
    99,  # Breath control pitch
    99,  # Breath control amplitude
    99,  # Breath control pitch bias
    99,  # Breath control EG bias
]

# Unpacked DX7 globals 136..144; packed globals decode to the same order.
_DX7_GLOBAL_LIMITS = [
    1,  # Oscillator sync
    99,  # LFO speed
    99,  # LFO delay
    99,  # Pitch modulation depth
    99,  # Amplitude modulation depth
    1,  # LFO sync
    5,  # LFO waveform, including sample-and-hold
    7,  # Pitch modulation sensitivity
    48,  # Transpose
]

# Hexter's provisional full-depth amplitude reduction ratios for AMS 0..3.
_DX7_AMPLITUDE_SENSITIVITIES = [0.0, 0.238058, 0.460510, 1.0]

# Measured VCED 0..99 mappings: LFS (waves 0..2), PMD, AMD.
# https://nornand.hatenablog.com/entry/2020/11/25/171659
# Firmware-derived lookup, ax81z 5848832989c7864092a277e9428d44f817c6173f Math.md.
_TX_LFO_CURVE = [
    255,
    255,
    224,
    205,
    192,
    181,
    173,
    166,
    160,
    154,
    149,
    145,
    141,
    137,
    134,
    130,
    128,
    125,
    122,
    120,
    117,
    115,
    113,
    111,
    109,
    107,
    105,
    103,
    102,
    100,
    98,
    97,
    96,
    94,
    93,
    91,
    90,
    89,
    88,
    86,
    85,
    84,
    83,
    82,
    81,
    80,
    79,
    78,
    77,
    76,
    75,
    74,
    73,
    72,
    71,
    70,
    70,
    69,
    68,
    67,
    66,
    66,
    65,
    64,
    64,
    63,
    62,
    61,
    61,
    60,
    59,
    59,
    58,
    57,
    57,
    56,
    56,
    55,
    54,
    54,
    53,
    53,
    52,
    51,
    51,
    50,
    50,
    49,
    49,
    48,
    48,
    47,
    47,
    46,
    46,
    45,
    45,
    44,
    44,
    43,
    43,
    42,
    42,
    42,
    41,
    41,
    40,
    40,
    39,
    39,
    38,
    38,
    38,
    37,
    37,
    36,
    36,
    36,
    35,
    35,
    34,
    34,
    34,
    33,
    33,
    33,
    32,
    32,
    32,
    31,
    31,
    30,
    30,
    30,
    29,
    29,
    29,
    28,
    28,
    28,
    27,
    27,
    27,
    26,
    26,
    26,
    25,
    25,
    25,
    24,
    24,
    24,
    24,
    23,
    23,
    23,
    22,
    22,
    22,
    21,
    21,
    21,
    21,
    20,
    20,
    20,
    19,
    19,
    19,
    19,
    18,
    18,
    18,
    18,
    17,
    17,
    17,
    17,
    16,
    16,
    16,
    16,
    15,
    15,
    15,
    15,
    14,
    14,
    14,
    14,
    13,
    13,
    13,
    13,
    12,
    12,
    12,
    12,
    11,
    11,
    11,
    11,
    10,
    10,
    10,
    10,
    9,
    9,
    9,
    9,
    8,
    8,
    8,
    8,
    8,
    7,
    7,
    7,
    7,
    6,
    6,
    6,
    6,
    6,
    5,
    5,
    5,
    5,
    5,
    4,
    4,
    4,
    4,
    4,
    3,
    3,
    3,
    3,
    3,
    2,
    2,
    2,
    2,
    2,
    2,
    1,
    1,
    1,
    1,
    1,
    0,
    0,
    0,
    0,
    0,
    0,
]

_TX_LFO_REGISTERS = [
    (0, 0, 0),
    (31, 0, 0),
    (74, 2, 0),
    (89, 3, 0),
    (106, 4, 0),
    (114, 5, 1),
    (125, 7, 1),
    (133, 8, 1),
    (138, 9, 1),
    (144, 11, 2),
    (148, 12, 2),
    (153, 13, 2),
    (157, 14, 2),
    (161, 16, 3),
    (165, 17, 3),
    (167, 18, 3),
    (171, 20, 3),
    (173, 21, 4),
    (176, 22, 4),
    (178, 23, 4),
    (181, 25, 5),
    (184, 26, 5),
    (185, 27, 5),
    (188, 29, 6),
    (189, 30, 6),
    (191, 31, 6),
    (194, 33, 7),
    (195, 34, 7),
    (197, 35, 7),
    (198, 36, 7),
    (200, 38, 8),
    (201, 39, 8),
    (203, 40, 8),
    (205, 42, 9),
    (206, 43, 9),
    (207, 44, 9),
    (208, 45, 10),
    (210, 47, 10),
    (211, 48, 10),
    (212, 49, 11),
    (213, 51, 11),
    (214, 52, 12),
    (216, 53, 12),
    (217, 54, 12),
    (218, 56, 13),
    (219, 57, 13),
    (220, 58, 14),
    (221, 60, 14),
    (222, 61, 15),
    (223, 62, 15),
    (223, 63, 16),
    (225, 65, 16),
    (226, 66, 17),
    (226, 67, 17),
    (227, 69, 18),
    (228, 70, 18),
    (229, 71, 19),
    (230, 72, 19),
    (231, 74, 20),
    (231, 75, 20),
    (232, 76, 21),
    (233, 78, 21),
    (234, 79, 22),
    (234, 80, 23),
    (235, 82, 23),
    (236, 83, 24),
    (237, 84, 25),
    (237, 85, 25),
    (238, 87, 26),
    (238, 88, 27),
    (239, 89, 28),
    (240, 91, 28),
    (240, 92, 29),
    (241, 93, 30),
    (242, 94, 31),
    (242, 96, 32),
    (243, 97, 33),
    (244, 98, 34),
    (244, 100, 35),
    (245, 101, 36),
    (246, 102, 37),
    (246, 103, 38),
    (247, 105, 40),
    (247, 106, 41),
    (248, 107, 42),
    (249, 109, 44),
    (249, 110, 45),
    (250, 111, 48),
    (250, 112, 49),
    (251, 114, 51),
    (251, 115, 54),
    (252, 116, 56),
    (252, 118, 60),
    (253, 119, 62),
    (253, 120, 67),
    (253, 121, 70),
    (254, 123, 77),
    (255, 124, 86),
    (255, 125, 96),
    (255, 127, 127),
]

# Native phase table from pinned ymfm_fm.ipp; provenance in opz-frequency.md.
_TX_PHASE_STEPS = np.load(
    Path(__file__).with_name("yamaha_phase.npy"), allow_pickle=False
)
# Measured TL attenuation, indexed by BC EG Bias, operator sensitivity, CC2.
# https://nornand.hatenablog.com/entry/2020/11/24/224423
_TX_BREATH_ATTENUATION = np.load(
    Path(__file__).with_name("yamaha_breath.npy"), allow_pickle=False
)

# Mathematical tables for the approved provisional Mk I model. Round the
# intermediate values to float32, matching the reference table precision.
_DX7_LOG_SINE = [
    int(-1024 * log2(float(np.float32(sin((i + 0.5) * pi / 2048)))) + 0.5)
    for i in range(1024)
]
_DX7_EXP = [
    int(float(np.float32((2 ** (i / 1024) - 1) * 4096)) + 0.5) for i in range(1024)
]

# Provisional equal-level durations measured in 44.1 kHz samples in Dexed's
# env.cc. Interpret these as durations, independent of the output sample rate.
_DX7_HOLD_SAMPLES = [
    1764000,
    1764000,
    1411200,
    1411200,
    1190700,
    1014300,
    992250,
    882000,
    705600,
    705600,
    584325,
    507150,
    502740,
    441000,
    418950,
    352800,
    308700,
    286650,
    253575,
    220500,
    220500,
    176400,
    145530,
    145530,
    125685,
    110250,
    110250,
    88200,
    88200,
    74970,
    61740,
    61740,
    55125,
    48510,
    44100,
    37485,
    31311,
    30870,
    27562,
    27562,
    22050,
    18522,
    17640,
    15435,
    14112,
    13230,
    11025,
    9261,
    9261,
    7717,
    6615,
    6615,
    5512,
    5512,
    4410,
    3969,
    3969,
    3439,
    2866,
    2690,
    2249,
    1984,
    1896,
    1808,
    1411,
    1367,
    1234,
    1146,
    926,
    837,
    837,
    705,
    573,
    573,
    529,
    441,
    441,
]

# Independent mathematical reconstruction of the OPZ logarithmic tables.
# The quarter-sine is sampled at cell midpoints; power mantissas include the
# implicit leading bit and the reference's two-bit output shift.
_TX_QUARTER_SINE = [round(-256 * log2(sin((i + 0.5) * pi / 512))) for i in range(256)]
_TX_SINE = [
    _TX_QUARTER_SINE[(i ^ (255 if i & 256 else 0)) & 255] | ((i & 512) << 6)
    for i in range(1024)
]
_TX_SQUARED = [min(2 * (i & 32767), _TX_SINE[0]) | (i & 32768) for i in _TX_SINE]
_TX_WAVEFORMS = [
    _TX_SINE,
    _TX_SQUARED,
    [_TX_SINE[i] if i < 512 else _TX_SINE[0] for i in range(1024)],
    [_TX_SQUARED[i] if i < 512 else _TX_SINE[0] for i in range(1024)],
    [_TX_SINE[2 * i] if i < 512 else _TX_SINE[0] for i in range(1024)],
    [_TX_SQUARED[2 * i] if i < 512 else _TX_SINE[0] for i in range(1024)],
    [_TX_SINE[(2 * i) & 511] if i < 512 else _TX_SINE[0] for i in range(1024)],
    [_TX_SQUARED[(2 * i) & 511] if i < 512 else _TX_SINE[0] for i in range(1024)],
]
_TX_POWER = [round(1024 * 2 ** ((255 - i) / 256)) * 4 for i in range(256)]
