"""Offline rendering for Ufor's oscillator synth profile."""

from fractions import Fraction
from math import ceil, isfinite
from typing import Literal, Self

import numpy as np
from ufor import instrument_trace, modulation
from ufor.base import Model
from ufor.envelope import Envelope, Segment
from ufor.oscillator import Oscillator, Waveform
from ufor.samples import controls
from ufor.samples.processing import ControlBinding, Processing
from ufor.streams import AudioType
from ufor.synth import SynthInstrument, SynthInstrumentScore, SynthVoice, frequency
from ufor.synth_trace import VoiceStart
from ufor.time import Timebase


class EngineError(ValueError):
    """The requested input is invalid or outside Enge's implemented profile."""


class PreparedSynth(Model, frozen=True):
    sample_rate: int
    channels: list[str]
    instrument: SynthInstrument


class ControlRamp(Model, frozen=True):
    control: str
    state: controls.ControlState


class ControlContext(Model, frozen=True):
    scope: Literal["instrument", "part", "trigger"]
    part: str | None
    trigger_id: str | None
    ramps: list[ControlRamp]


class OscillatorState(Model, frozen=True):
    """Phase in sample-rate units, with compensated-addition error."""

    position: float = 0
    error: float = 0

    @classmethod
    def at_frame(
        cls, frame: int | float, frequency_hz: float, sample_rate: int
    ) -> Self:
        """Initialize phase without losing precision at large frame coordinates."""
        return cls(position=(Fraction(frame) * Fraction(frequency_hz)) % sample_rate)


class VoiceSnapshot(Model, frozen=True):
    voice_id: str
    template: str
    frequency_hz: float
    oscillator: OscillatorState
    sources: dict[str, int]
    gain: float
    started_frame: int
    release_frame: Fraction | None = None


class SynthSnapshot(Model, frozen=True):
    definition: PreparedSynth
    frame: int
    voices: list[VoiceSnapshot]
    contexts: list[ControlContext]


def prepare(score: SynthInstrumentScore) -> PreparedSynth:
    """Validate the supported profile and isolate its prepared definitions."""
    score = SynthInstrumentScore.model_validate(score.model_dump())
    output = score.outputs[0].stream
    if not isinstance(output, AudioType):
        raise EngineError("Synth output must be sampled audio")
    timebase = next(t for t in score.timebases if t.name == output.timebase)
    for voice in score.body.voices:
        _validate_voice(voice)
    return PreparedSynth(
        sample_rate=_sample_rate(timebase),
        channels=output.channels,
        instrument=score.body,
    )


class OfflineSynth:
    """Render one prepared synth instance with exact frame-boundary actions."""

    def __init__(self, definition: PreparedSynth) -> None:
        self.definition = definition.model_copy(deep=True)
        self.frame = 0
        self.voices: dict[str, VoiceSnapshot] = {}
        self.contexts: list[ControlContext] = []
        self.templates = {v.name: v for v in self.definition.instrument.voices}
        self._new_context("instrument", None, None, 0, {})

    def advance(
        self, actions: list[instrument_trace.TraceAction], start: int, end: int
    ) -> np.ndarray:
        """Render [start, end), applying actions before their addressed sample."""
        if start != self.frame or end <= start:
            raise EngineError(
                "advance must continue from the current nonempty interval"
            )
        ordered = sorted(actions, key=lambda a: (a.tick, a.ordinal))
        if any(a.tick < start or a.tick >= end for a in ordered):
            raise EngineError("actions must belong to the rendered interval")
        output = np.zeros(
            (end - start, len(self.definition.channels)), dtype=np.float64
        )
        cursor = start
        for action in ordered:
            output[cursor - start : action.tick - start] = self._render(
                cursor, action.tick - cursor
            )
            self._apply(action)
            cursor = action.tick
        output[cursor - start :] = self._render(cursor, end - cursor)
        self.frame = end
        return output

    def snapshot(self) -> SynthSnapshot:
        return SynthSnapshot(
            definition=self.definition,
            frame=self.frame,
            voices=list(self.voices.values()),
            contexts=self.contexts,
        ).model_copy(deep=True)

    def restore(self, snapshot: SynthSnapshot) -> None:
        if snapshot.definition != self.definition:
            raise EngineError("Snapshot belongs to a different prepared synth")
        snapshot = snapshot.model_copy(deep=True)
        self.frame = snapshot.frame
        self.voices = {v.voice_id: v for v in snapshot.voices}
        self.contexts = snapshot.contexts

    def _new_context(
        self,
        scope: Literal["instrument", "part", "trigger"],
        part: str | None,
        trigger_id: str | None,
        frame: int,
        values: dict[str, float],
    ) -> int:
        ramps: list[ControlRamp] = []
        for voice in self.templates.values():
            sources = {s.name: s.scope for s in voice.modulation.sources}
            for binding in voice.bindings:
                assert isinstance(binding, ControlBinding)
                if sources[binding.name] != scope or any(
                    r.control == binding.control
                    and r.state.smoothing == binding.smoothing
                    for r in ramps
                ):
                    continue
                declaration = self.definition.instrument.controls[binding.control]
                ramps.append(
                    ControlRamp(
                        control=binding.control,
                        state=controls.initial_control(
                            declaration,
                            Fraction(frame, self.definition.sample_rate),
                            binding.smoothing,
                            values.get(binding.control),
                        ),
                    )
                )
        self.contexts.append(
            ControlContext(scope=scope, part=part, trigger_id=trigger_id, ramps=ramps)
        )
        return len(self.contexts) - 1

    def _context(
        self, scope: str, part: str | None, trigger_id: str | None
    ) -> int | None:
        return next(
            (
                i
                for i in range(len(self.contexts) - 1, -1, -1)
                if (
                    self.contexts[i].scope,
                    self.contexts[i].part,
                    self.contexts[i].trigger_id,
                )
                == (scope, part, trigger_id)
            ),
            None,
        )

    def _apply(self, action: instrument_trace.TraceAction) -> None:
        if isinstance(action, instrument_trace.TriggerContext):
            declarations = self.definition.instrument.controls
            if action.controls.keys() != declarations.keys():
                raise EngineError("Trigger context must contain all declared controls")
            for name, value in action.controls.items():
                declarations[name].validate_value(value)
            self._new_context(
                "trigger", action.part, action.trigger_id, action.tick, action.controls
            )
        elif isinstance(action, instrument_trace.ControlObservation):
            declaration = self.definition.instrument.require_control(action.control)
            declaration.validate_value(action.value)
            context_id = self._context(action.scope, action.part, action.trigger_id)
            if context_id is None:
                if action.scope == "trigger":
                    return
                context_id = self._new_context(
                    action.scope, action.part, action.trigger_id, 0, {}
                )
            context = self.contexts[context_id]
            event = controls.ControlValueEvent(
                at=Fraction(action.tick, self.definition.sample_rate),
                ordinal=action.ordinal,
                value=action.value,
            )
            self.contexts[context_id] = context.model_copy(
                update={
                    "ramps": [
                        r.model_copy(
                            update={
                                "state": controls.control_event(
                                    declaration, r.state, event
                                )
                            }
                        )
                        if r.control == action.control
                        else r
                        for r in context.ramps
                    ]
                }
            )
        elif isinstance(action, VoiceStart):
            self._start_voice(action)
        elif isinstance(action, instrument_trace.VoiceRetirement):
            if action.action == "fade":
                raise EngineError("Fade retirement is not implemented")
            if action.action == "stop":
                self.voices.pop(action.voice_id, None)
            elif (
                voice := self.voices.get(action.voice_id)
            ) and voice.release_frame is None:
                template = self.templates[voice.template]
                release = max(
                    Fraction(action.tick),
                    voice.started_frame
                    + template.minimum_hold_seconds * self.definition.sample_rate,
                )
                self.voices[action.voice_id] = voice.model_copy(
                    update={"release_frame": release}
                )
        else:
            raise EngineError(
                f"Unsupported synth action at frame {action.tick}: "
                f"{type(action).__name__}"
            )

    def _start_voice(self, action: VoiceStart) -> None:
        if (
            action.pitch_hz is None
            or not isfinite(action.pitch_hz)
            or action.pitch_hz <= 0
        ):
            raise EngineError("Synth voice requires positive resolved pitch_hz")
        if action.voice_id in self.voices:
            raise EngineError(f"Duplicate active voice: {action.voice_id}")
        template = self.templates.get(action.template)
        if (
            template is None
            or action.settings != template
            or action.oscillator != template.oscillator
            or action.channels != template.channels
        ):
            raise EngineError("Voice start must match its prepared synth template")
        sources: dict[str, int] = {}
        for source in template.modulation.sources:
            part = None if source.scope == "instrument" else action.part
            trigger_id = action.trigger_id if source.scope == "trigger" else None
            context_id = self._context(source.scope, part, trigger_id)
            if context_id is None:
                if source.scope == "trigger" and trigger_id is not None:
                    raise EngineError("Voice start is missing its trigger context")
                assert source.scope in ("instrument", "part", "trigger")
                context_id = self._new_context(source.scope, part, trigger_id, 0, {})
            sources[source.name] = context_id
        self.voices[action.voice_id] = VoiceSnapshot(
            voice_id=action.voice_id,
            template=template.name,
            frequency_hz=action.pitch_hz,
            oscillator=OscillatorState.at_frame(
                action.tick if template.synchronize_oscillator else 0,
                action.pitch_hz,
                self.definition.sample_rate,
            ),
            sources=sources,
            gain=template.oscillator.gain(action.key),
            started_frame=action.tick,
        )

    def _parameters(
        self, voice: VoiceSnapshot, start: int, frames: int
    ) -> tuple[np.ndarray, np.ndarray]:
        template = self.templates[voice.template]
        frequencies = np.full(
            frames, frequency(voice.frequency_hz, template.processing.tuning_cents)
        )
        gains = np.ones(frames)
        ramps: dict[str, controls.ControlState] = {}
        for binding in template.bindings:
            assert isinstance(binding, ControlBinding)
            context = self.contexts[voice.sources[binding.name]]
            ramps[binding.name] = next(
                r.state
                for r in context.ramps
                if r.control == binding.control
                and r.state.smoothing == binding.smoothing
            )
        if template.modulation.parameters:
            for i in range(frames):
                at = Fraction(start + i, self.definition.sample_rate)
                values = modulation.evaluate(
                    template.modulation,
                    {
                        n: modulation.SourceValue(value=controls.control_at(s, at))
                        for n, s in ramps.items()
                    },
                )
                for value in values:
                    if value.target.parameter == "amplitude":
                        gains[i] = value.value
                    else:
                        frequencies[i] = frequency(voice.frequency_hz, value.value)
        if not np.all(np.isfinite(frequencies) & (frequencies > 0)):
            raise EngineError(
                f"Invalid frequency for {voice.voice_id} at frame {start}"
            )
        return frequencies, gains

    def _render(self, start: int, frames: int) -> np.ndarray:
        output = np.zeros((frames, len(self.definition.channels)), dtype=np.float64)
        for voice in list(self.voices.values()):
            template = self.templates[voice.template]
            end = (
                None
                if voice.release_frame is None
                else voice.release_frame
                + _duration(template.envelope.release) * self.definition.sample_rate
            )
            count = frames if end is None else max(0, min(frames, ceil(end) - start))
            frequencies, gains = self._parameters(voice, start, count)
            gains *= (
                envelope_samples(
                    template.envelope,
                    start - voice.started_frame,
                    count,
                    self.definition.sample_rate,
                    None
                    if voice.release_frame is None
                    else voice.release_frame - voice.started_frame,
                )
                * voice.gain
            )
            wave, oscillator = oscillator_samples(
                template.oscillator,
                voice.oscillator,
                frequencies,
                self.definition.sample_rate,
            )
            wave *= gains
            for route in template.channels:
                output[:count, self.definition.channels.index(route.output)] += (
                    wave * route.gain
                )
            if end is not None and start + frames >= end:
                del self.voices[voice.voice_id]
            else:
                self.voices[voice.voice_id] = voice.model_copy(
                    update={"oscillator": oscillator}
                )
        return output


def waveform_samples(
    oscillator: Oscillator,
    start: float | np.ndarray,
    length: int,
    period: float | np.ndarray,
) -> np.ndarray:
    """Render Tuney's established sample-position oscillator convention."""
    end = start + length
    ratio = 2 * np.pi / period
    return _waveform_angles(
        oscillator, np.linspace(start * ratio, end * ratio, length, endpoint=False)
    )


def oscillator_samples(
    oscillator: Oscillator,
    state: OscillatorState,
    frequencies: np.ndarray,
    sample_rate: int,
) -> tuple[np.ndarray, OscillatorState]:
    """Render per-sample frequencies and return the phase for the next sample."""
    # Sum frequency before division, carrying the rounding correction across
    # calls so discontinuous waveforms switch on the same sample in every block.
    position, error = state.position, state.error
    phases = np.empty(len(frequencies))
    for i, value in enumerate(frequencies):
        phases[i] = position / sample_rate
        increment = float(value) - error
        total = position + increment
        error = (total - position) - increment
        position = total % sample_rate
    return _waveform_angles(oscillator, 2 * np.pi * phases), OscillatorState(
        position=position, error=error
    )


def envelope_samples(
    envelope: Envelope,
    start: int,
    length: int,
    sample_rate: int,
    release_frame: Fraction | None = None,
) -> np.ndarray:
    """Render a prepared held linear envelope in voice-relative sample frames.

    The caller resolves minimum hold into release_frame. Release begins at the
    exact fractional coordinate and captures the level there, even between
    samples. The final authored target is held; voice retirement is separate.
    """
    values = _envelope_values(
        envelope.initial,
        envelope.segments,
        Fraction(start, sample_rate),
        length,
        sample_rate,
    )
    if release_frame is not None:
        offset = max(0, ceil(release_frame) - start)
        if offset < length:
            release_gain = float(
                _envelope_values(
                    envelope.initial,
                    envelope.segments,
                    release_frame / sample_rate,
                    1,
                    sample_rate,
                )[0]
            )
            values[offset:] = _envelope_values(
                release_gain,
                envelope.release,
                (start + offset - release_frame) / sample_rate,
                length - offset,
                sample_rate,
            )
    return values


def _sample_rate(timebase: Timebase) -> int:
    if timebase.rate.denominator != 1:
        raise EngineError(
            "Synth output rate must be an integer number of frames per second"
        )
    return timebase.rate.numerator


def _validate_voice(voice: SynthVoice) -> None:
    envelope = voice.envelope
    if envelope.clock != "seconds" or envelope.scope != "voice":
        raise EngineError("Synth envelopes must use the voice seconds clock")
    if not envelope.hold or any(
        s.curve != 0 for s in [*envelope.segments, *envelope.release]
    ):
        raise EngineError("Only held linear synth envelopes are implemented")
    if voice.processing != Processing(tuning_cents=voice.processing.tuning_cents):
        raise EngineError("Only tuning processing is implemented")
    if voice.envelopes or voice.lfos:
        raise EngineError("Named synth generators are not implemented")
    if any(c.mode == "fade" for c in voice.chokes):
        raise EngineError("Fade retirement is not implemented")
    if any(not isinstance(b, ControlBinding) for b in voice.bindings):
        raise EngineError("Only control bindings are implemented")
    if any(
        p.target.name != "processing"
        or p.target.parameter not in ("amplitude", "tuning_cents")
        for p in voice.modulation.parameters
    ):
        raise EngineError("Only amplitude and tuning modulation are implemented")


def _envelope_values(
    initial: float,
    segments: list[Segment],
    elapsed: Fraction,
    frames: int,
    sample_rate: int,
) -> np.ndarray:
    values = np.full(frames, segments[-1].target)
    boundary = Fraction(0)
    entry = initial
    for segment in segments:
        end = boundary + segment.duration
        first = max(0, ceil((boundary - elapsed) * sample_rate))
        last = min(frames, ceil((end - elapsed) * sample_rate))
        if segment.duration and first < last:
            progress = float((elapsed - boundary) / segment.duration) + np.arange(
                first, last
            ) / float(segment.duration * sample_rate)
            values[first:last] = entry + (segment.target - entry) * progress
        boundary, entry = end, segment.target
    return values


def _duration(segments: list[Segment]) -> Fraction:
    return sum((s.duration for s in segments), Fraction(0))


def _waveform_angles(oscillator: Oscillator, angles: np.ndarray) -> np.ndarray:
    if oscillator.waveform == Waveform.sine:
        return np.sin(angles)
    phase = (angles / (2 * np.pi)) % 1
    if oscillator.waveform == Waveform.square:
        return np.where(phase < float(oscillator.duty_cycle), 1.0, -1.0)
    duty = float(oscillator.duty_cycle)
    if duty == 0:
        return 1 - 2 * phase
    if duty == 1:
        return 2 * phase - 1
    return np.where(
        phase < duty, 2 * phase / duty - 1, (1 + duty - 2 * phase) / (1 - duty)
    )
