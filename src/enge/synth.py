"""Offline rendering for Ufor's oscillator synth profile."""

from collections.abc import Callable
from fractions import Fraction
from functools import cached_property
from math import ceil, isfinite
from typing import Literal, Self

import numpy as np
from pydantic import BaseModel, ConfigDict, Field, model_validator
from ufor import instrument_trace, modulation
from ufor.base import Model
from ufor.envelope import Envelope, Segment
from ufor.oscillator import Oscillator, Waveform
from ufor.samples import controls
from ufor.samples.processing import ControlBinding, Processing, SoundSettings
from ufor.streams import AudioType
from ufor.synth import SynthInstrument, SynthInstrumentScore, SynthVoice, frequency
from ufor.synth_trace import VoiceStart
from ufor.time import Timebase

from . import native


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


class PreparedEnvelope(Model, frozen=True):
    """The shared amplitude lifetime of an oscillator or sample voice."""

    sample_rate: int = Field(gt=0)
    envelope: Envelope
    minimum_hold_seconds: Fraction = Field(default=Fraction(0), ge=0)

    @model_validator(mode="after")
    def envelope_profile(self) -> Self:
        validate_envelope(self.envelope)
        return self

    @cached_property
    def release_frames(self) -> Fraction:
        return _duration(self.envelope.release) * self.sample_rate

    @cached_property
    def minimum_hold_frames(self) -> Fraction:
        return self.minimum_hold_seconds * self.sample_rate


class EnvelopeRenderer(BaseModel):
    """Shared release timing; source state remains with the concrete renderer."""

    definition: PreparedEnvelope
    frame_count: int = 0
    release_frame: Fraction | None = None

    @property
    def complete(self) -> bool:
        return self.release_frame is not None and self.frame_count >= (
            self.release_frame + self.definition.release_frames
        )

    def release(self) -> bool:
        if self.release_frame is not None:
            return False
        self.release_frame = max(
            Fraction(self.frame_count), self.definition.minimum_hold_frames
        )
        return True

    def active_frames(self, frames: int) -> int:
        if self.release_frame is None:
            return frames
        end = self.release_frame + self.definition.release_frames
        return max(0, min(frames, ceil(end) - self.frame_count))

    model_config = ConfigDict(
        extra="forbid", allow_inf_nan=False, validate_default=True
    )


class PreparedVoice(PreparedEnvelope, frozen=True):
    """Resolved voice settings; route rows map oscillators to output channels."""

    oscillator: Oscillator
    frequencies: list[float] = Field(min_length=1)
    routes: list[list[float]] = Field(min_length=1)
    gain: float = 1

    @model_validator(mode="after")
    def rendering_profile(self) -> Self:
        if len(self.routes) != len(self.frequencies) or not self.routes[0]:
            raise EngineError(
                "Voice routes must map each oscillator to output channels"
            )
        if any(len(r) != len(self.routes[0]) for r in self.routes):
            raise EngineError("Voice route rows must have the same output channels")
        return self


class VoiceRenderer(EnvelopeRenderer):
    """One held voice with explicit oscillator routes and resumable audio state."""

    definition: PreparedVoice
    oscillators: list[OscillatorState]
    backend: Literal["numpy", "native"] = "numpy"

    @classmethod
    def start(
        cls,
        definition: PreparedVoice,
        phase_origin: int | float = 0,
        backend: Literal["numpy", "native"] = "numpy",
    ) -> Self:
        return cls(
            definition=definition,
            backend=backend,
            oscillators=[
                OscillatorState.at_frame(phase_origin, f, definition.sample_rate)
                for f in definition.frequencies
            ],
        )

    def render(
        self,
        frames: int,
        frequencies: np.ndarray | None = None,
        gains: np.ndarray | None = None,
    ) -> np.ndarray:
        """Render (frames, channels), padding completed tails with exact silence.

        Optional frequencies have shape (active frames, oscillators); gains have
        shape (active frames,). They replace pitch and multiply prepared gain.
        """
        count = self.active_frames(frames)
        definition = self.definition
        if count and self.backend == "native":
            states = np.array([[s.position, s.error] for s in self.oscillators])
            output, states = native.render(
                definition.oscillator,
                states,
                np.tile(definition.frequencies, (count, 1))
                if frequencies is None
                else frequencies[:count],
                definition.sample_rate,
                definition.gain,
                np.ones(count) if gains is None else gains[:count],
                native.envelope_spans(
                    definition.envelope,
                    self.frame_count,
                    count,
                    definition.sample_rate,
                    self.release_frame,
                ),
                definition.routes,
                frames,
            )
            self.oscillators = [
                OscillatorState(position=s[0], error=s[1]) for s in states
            ]
        elif count:
            waves: list[np.ndarray] = []
            for i, frequency_hz in enumerate(definition.frequencies):
                wave, self.oscillators[i] = oscillator_samples(
                    definition.oscillator,
                    self.oscillators[i],
                    np.full(count, frequency_hz)
                    if frequencies is None
                    else frequencies[:count, i],
                    definition.sample_rate,
                )
                waves.append(wave)
            amplitude = (
                envelope_samples(
                    definition.envelope,
                    self.frame_count,
                    count,
                    definition.sample_rate,
                    self.release_frame,
                )
                * definition.gain
            )
            if gains is not None:
                amplitude *= gains[:count]
            samples = waves[0][:, None] if len(waves) == 1 else np.column_stack(waves)
            output = route_samples(samples, definition.routes)
            output *= amplitude[:, None]
        else:
            output = np.zeros((0, len(definition.routes[0])))
        if count < frames and len(output) != frames:
            padded = np.zeros((frames, len(definition.routes[0])))
            padded[:count] = output
            output = padded
        self.frame_count += frames
        return output


class VoiceSnapshot(Model, frozen=True):
    voice_id: str
    template: str
    frequency_hz: float
    renderer: VoiceRenderer
    sources: dict[str, int]
    started_frame: int


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


class ControlRenderer:
    """Scoped control trajectories shared by synth and sampler instances."""

    def __init__(
        self,
        sample_rate: int,
        declarations: dict[str, controls.ControlDeclaration],
        settings: list[SoundSettings],
    ) -> None:
        self.sample_rate = sample_rate
        self.declarations = declarations
        self.settings = settings
        self.contexts: list[ControlContext] = []
        self.new_context("instrument", None, None, 0, {})

    def new_context(
        self,
        scope: Literal["instrument", "part", "trigger"],
        part: str | None,
        trigger_id: str | None,
        frame: int,
        values: dict[str, float],
    ) -> int:
        ramps: list[ControlRamp] = []
        for setting in self.settings:
            sources = {s.name: s.scope for s in setting.modulation.sources}
            for binding in setting.bindings:
                assert isinstance(binding, ControlBinding)
                if sources[binding.name] != scope or any(
                    r.control == binding.control
                    and r.state.smoothing == binding.smoothing
                    for r in ramps
                ):
                    continue
                ramps.append(
                    ControlRamp(
                        control=binding.control,
                        state=controls.initial_control(
                            self.declarations[binding.control],
                            Fraction(frame, self.sample_rate),
                            binding.smoothing,
                            values.get(binding.control),
                        ),
                    )
                )
        self.contexts.append(
            ControlContext(scope=scope, part=part, trigger_id=trigger_id, ramps=ramps)
        )
        return len(self.contexts) - 1

    def context(
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

    def apply(self, action: instrument_trace.TraceAction) -> bool:
        if isinstance(action, instrument_trace.TriggerContext):
            if action.controls.keys() != self.declarations.keys():
                raise EngineError("Trigger context must contain all declared controls")
            for name, value in action.controls.items():
                self.declarations[name].validate_value(value)
            self.new_context(
                "trigger", action.part, action.trigger_id, action.tick, action.controls
            )
        elif isinstance(action, instrument_trace.ControlObservation):
            if action.control not in self.declarations:
                raise EngineError(f"Unknown control: {action.control}")
            declaration = self.declarations[action.control]
            declaration.validate_value(action.value)
            context_id = self.context(action.scope, action.part, action.trigger_id)
            if context_id is None:
                if action.scope == "trigger":
                    return True
                context_id = self.new_context(
                    action.scope, action.part, action.trigger_id, 0, {}
                )
            context = self.contexts[context_id]
            event = controls.ControlValueEvent(
                at=Fraction(action.tick, self.sample_rate),
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
        else:
            return False
        return True

    def sources(
        self, settings: SoundSettings, action: instrument_trace.VoiceStart
    ) -> dict[str, int]:
        result: dict[str, int] = {}
        for source in settings.modulation.sources:
            part = None if source.scope == "instrument" else action.part
            trigger_id = action.trigger_id if source.scope == "trigger" else None
            context_id = self.context(source.scope, part, trigger_id)
            if context_id is None:
                if source.scope == "trigger" and trigger_id is not None:
                    raise EngineError("Voice start is missing its trigger context")
                assert source.scope in ("instrument", "part", "trigger")
                context_id = self.new_context(source.scope, part, trigger_id, 0, {})
            result[source.name] = context_id
        return result

    def parameters(
        self, settings: SoundSettings, sources: dict[str, int], start: int, frames: int
    ) -> tuple[np.ndarray, np.ndarray]:
        tuning = np.full(frames, settings.processing.tuning_cents, dtype=np.float64)
        gains = np.ones(frames)
        ramps: dict[str, controls.ControlState] = {}
        for binding in settings.bindings:
            assert isinstance(binding, ControlBinding)
            context = self.contexts[sources[binding.name]]
            ramps[binding.name] = next(
                r.state
                for r in context.ramps
                if r.control == binding.control
                and r.state.smoothing == binding.smoothing
            )
        if settings.modulation.parameters:
            for i in range(frames):
                at = Fraction(start + i, self.sample_rate)
                values = modulation.evaluate(
                    settings.modulation,
                    {
                        n: modulation.SourceValue(value=controls.control_at(s, at))
                        for n, s in ramps.items()
                    },
                )
                for value in values:
                    if value.target.parameter == "amplitude":
                        gains[i] = value.value
                    else:
                        tuning[i] = value.value
        return tuning, gains


class OfflineSynth:
    """Render one prepared synth instance with exact frame-boundary actions."""

    def __init__(
        self, definition: PreparedSynth, backend: Literal["numpy", "native"] = "numpy"
    ) -> None:
        if backend not in ("numpy", "native"):
            raise EngineError(f"Unknown synth backend: {backend}")
        self.backend = backend
        self.definition = definition.model_copy(deep=True)
        self.frame = 0
        self.voices: dict[str, VoiceSnapshot] = {}
        self.templates = {v.name: v for v in self.definition.instrument.voices}
        self.controls = ControlRenderer(
            self.definition.sample_rate,
            self.definition.instrument.controls,
            list(self.templates.values()),
        )

    def advance(
        self, actions: list[instrument_trace.TraceAction], start: int, end: int
    ) -> np.ndarray:
        """Render [start, end), applying actions before their addressed sample."""
        output = render_actions(
            actions,
            start,
            end,
            self.frame,
            len(self.definition.channels),
            self._render,
            self._apply,
        )
        self.frame = end
        return output

    def snapshot(self) -> SynthSnapshot:
        return SynthSnapshot(
            definition=self.definition,
            frame=self.frame,
            voices=list(self.voices.values()),
            contexts=self.controls.contexts,
        ).model_copy(deep=True)

    def restore(self, snapshot: SynthSnapshot) -> None:
        if snapshot.definition != self.definition:
            raise EngineError("Snapshot belongs to a different prepared synth")
        if any(v.renderer.backend != self.backend for v in snapshot.voices):
            raise EngineError("Snapshot belongs to a different synth backend")
        snapshot = snapshot.model_copy(deep=True)
        self.frame = snapshot.frame
        self.voices = {v.voice_id: v for v in snapshot.voices}
        self.controls.contexts = snapshot.contexts

    def _apply(self, action: instrument_trace.TraceAction) -> None:
        if self.controls.apply(action):
            return
        elif isinstance(action, VoiceStart):
            self._start_voice(action)
        elif isinstance(action, instrument_trace.VoiceRetirement):
            if action.action == "fade":
                raise EngineError("Fade retirement is not implemented")
            if action.action == "stop":
                self.voices.pop(action.voice_id, None)
            elif voice := self.voices.get(action.voice_id):
                voice.renderer.release()
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
        sources = self.controls.sources(template, action)
        self.voices[action.voice_id] = VoiceSnapshot(
            voice_id=action.voice_id,
            template=template.name,
            frequency_hz=action.pitch_hz,
            renderer=VoiceRenderer.start(
                PreparedVoice(
                    sample_rate=self.definition.sample_rate,
                    oscillator=template.oscillator,
                    envelope=template.envelope,
                    frequencies=[action.pitch_hz],
                    gain=template.oscillator.gain(action.key),
                    minimum_hold_seconds=template.minimum_hold_seconds,
                    routes=[
                        [
                            sum(r.gain for r in template.channels if r.output == c)
                            for c in self.definition.channels
                        ]
                    ],
                ),
                action.tick if template.synchronize_oscillator else 0,
                backend=self.backend,
            ),
            sources=sources,
            started_frame=action.tick,
        )

    def _parameters(
        self, voice: VoiceSnapshot, start: int, frames: int
    ) -> tuple[np.ndarray, np.ndarray]:
        template = self.templates[voice.template]
        tuning, gains = self.controls.parameters(template, voice.sources, start, frames)
        frequencies = np.array(
            [frequency(voice.frequency_hz, float(c)) for c in tuning]
        )
        if not np.all(np.isfinite(frequencies) & (frequencies > 0)):
            raise EngineError(
                f"Invalid frequency for {voice.voice_id} at frame {start}"
            )
        return frequencies, gains

    def _render(self, start: int, frames: int) -> np.ndarray:
        output = np.zeros((frames, len(self.definition.channels)), dtype=np.float64)
        for voice in list(self.voices.values()):
            count = voice.renderer.active_frames(frames)
            frequencies, gains = self._parameters(voice, start, count)
            output += voice.renderer.render(frames, frequencies[:, None], gains)
            if voice.renderer.complete:
                del self.voices[voice.voice_id]
        return output


def render_actions(
    actions: list[instrument_trace.TraceAction],
    start: int,
    end: int,
    frame: int,
    channels: int,
    render: Callable[[int, int], np.ndarray],
    apply: Callable[[instrument_trace.TraceAction], None],
) -> np.ndarray:
    """Shared half-open scheduling; preserve trace order for equal coordinates."""
    if start != frame or end <= start:
        raise EngineError("advance must continue from the current nonempty interval")
    ordered = sorted(actions, key=lambda a: (a.tick, a.ordinal))
    if any(a.tick < start or a.tick >= end for a in ordered):
        raise EngineError("actions must belong to the rendered interval")
    output = np.zeros((end - start, channels), dtype=np.float64)
    cursor = start
    for action in ordered:
        output[cursor - start : action.tick - start] = render(
            cursor, action.tick - cursor
        )
        apply(action)
        cursor = action.tick
    output[cursor - start :] = render(cursor, end - cursor)
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


def route_samples(samples: np.ndarray, routes: list[list[float]]) -> np.ndarray:
    """Mix explicit source-to-output gains without clipping or normalization."""
    return samples @ np.asarray(routes)


def oscillator_samples(
    oscillator: Oscillator,
    state: OscillatorState,
    frequencies: np.ndarray,
    sample_rate: int,
    backend: Literal["numpy", "native"] = "numpy",
) -> tuple[np.ndarray, OscillatorState]:
    """Render per-sample frequencies and return the phase for the next sample."""
    if backend == "native":
        states = np.array([[state.position, state.error]])
        frames = len(frequencies)
        output, states = native.render(
            oscillator,
            states,
            frequencies[:, None],
            sample_rate,
            1,
            np.ones(frames),
            np.array([[0, frames, 1, 0, 0, 1, 0]], dtype=np.float64),
            [[1]],
            frames,
        )
        return output[:, 0], OscillatorState(position=states[0, 0], error=states[0, 1])
    if backend != "numpy":
        raise EngineError(f"Unknown oscillator backend: {backend}")
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


def validate_envelope(envelope: Envelope) -> None:
    """Validate the amplitude-envelope profile shared by both voice renderers."""
    if envelope.clock != "seconds" or envelope.scope != "voice":
        raise EngineError("Envelopes must use the voice seconds clock")
    if not envelope.hold or any(
        s.curve != 0 for s in [*envelope.segments, *envelope.release]
    ):
        raise EngineError("Only held linear envelopes are implemented")


def _sample_rate(timebase: Timebase) -> int:
    if timebase.rate.denominator != 1:
        raise EngineError(
            "Synth output rate must be an integer number of frames per second"
        )
    return timebase.rate.numerator


def _validate_voice(voice: SynthVoice) -> None:
    validate_envelope(voice.envelope)
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
