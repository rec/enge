"""Offline rendering for Ufor's oscillator synth profile."""

from collections import deque
from collections.abc import Callable, Collection
from fractions import Fraction
from functools import cached_property
from math import ceil, floor, isfinite
from typing import Literal, Self

import numpy as np
from pydantic import BaseModel, ConfigDict, Field, model_validator
from ufor import instrument_trace, lfo, modulation
from ufor.base import Model
from ufor.control import TempoMap
from ufor.envelope import Envelope
from ufor.motion import (
    Contour,
    ContourState,
    Cycle,
    CycleState,
    EnterStage,
    Hold,
    MotionEvent,
    MotionOutputEvent,
    MotionState,
    MotionUse,
    Patch,
    PlaybackMode,
    Stages,
    StageState,
    advance_motion,
    cycle_lfo,
    initial_motion,
    motion_at,
    motion_event,
)
from ufor.oscillator import Oscillator, Waveform
from ufor.samples import controls, processing
from ufor.samples.processing import (
    ControlBinding,
    Processing,
    ReleaseTiming,
    SoundSettings,
)
from ufor.segments import Segment
from ufor.streams import AudioType
from ufor.synth import SynthInstrument, SynthInstrumentScore, SynthVoice, frequency
from ufor.synth_trace import VoiceStart
from ufor.time import Timebase

from . import _native, control, filters, native
from .lfo import lfo_samples


class EngineError(ValueError):
    """The requested input is invalid or outside Enge's implemented profile."""


StagedRuntimeDefinition = tuple[
    list[
        tuple[
            int,
            float,
            bool,
            list[tuple[float, float]],
            list[float],
            list[tuple[float, str]],
            int,
            float,
            float,
            int | None,
        ]
    ],
    list[tuple[int, str, int]],
    int,
    int | None,
    int,
    float,
    float,
    bool,
]


class PreparedSynth(Model, frozen=True):
    sample_rate: int
    channels: list[str]
    instrument: SynthInstrument
    source_instrument: SynthInstrument | None = None


class ControlRamp(Model, frozen=True):
    control: str
    state: controls.ControlState


class ControlContext(Model, frozen=True):
    scope: Literal["instrument", "part", "trigger"]
    part: str | None
    trigger_id: str | None
    ramps: list[ControlRamp]


class LFOSource(Model, frozen=True):
    setting: int
    name: str
    part: str | None
    trigger_id: str | None
    voice_id: str | None
    state: MotionState


class EnvelopeSource(Model, frozen=True):
    setting: int
    name: str
    binding_name: str
    part: str
    trigger_id: str | None
    voice_id: str
    state: MotionState
    pending_release: Fraction | None = None


class OscillatorState(Model, frozen=True):
    """Phase in sample-rate units, with compensated-addition error."""

    position: float = 0
    error: float = 0

    @classmethod
    def at_frame(
        cls, frame: int | float, frequency_hz: float, sample_rate: int
    ) -> Self:
        """Initialize phase without losing precision at large frame coordinates."""
        return cls(
            position=float((Fraction(frame) * Fraction(frequency_hz)) % sample_rate)
        )


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
    filters: list[processing.ResonantFilter] = Field(default_factory=list)

    @model_validator(mode="after")
    def rendering_profile(self) -> Self:
        filters.parameters(self.filters, self.sample_rate, 1)
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
    filter_states: list[filters.FilterState] = Field(default_factory=list)
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
            filter_states=filters.initial_states(
                definition.filters, len(definition.frequencies)
            ),
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
        filter_parameters: np.ndarray | None = None,
    ) -> np.ndarray:
        """Render (frames, channels), padding completed tails with exact silence.

        Optional frequencies have shape (active frames, oscillators); gains have
        shape (active frames,). They replace pitch and multiply prepared gain.
        Filter parameters have shape (active frames, filters, 2): cutoff Hz, Q.
        """
        count = self.active_frames(frames)
        definition = self.definition
        values = filters.parameters(
            definition.filters,
            definition.sample_rate,
            count,
            None if filter_parameters is None else filter_parameters[:count],
            self.frame_count,
        )
        if count and self.backend == "native":
            states = np.array([[s.position, s.error] for s in self.oscillators])
            output, states, memory = native.render(
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
                filters.native_inputs(definition.filters, self.filter_states, values),
            )
            self.filter_states = filters.restored_states(memory)
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
            if definition.filters:
                samples, self.filter_states = filters.filter_samples(
                    definition.filters,
                    self.filter_states,
                    samples,
                    values,
                    definition.sample_rate,
                )
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
    control_interval: int = Field(strict=True, gt=0)
    definition: PreparedSynth
    frame: int
    voices: list[VoiceSnapshot]
    contexts: list[ControlContext]
    lfos: list[LFOSource]
    envelopes: list[EnvelopeSource]
    tempo_map: TempoMap | None = None


class VoiceAddress(Model, frozen=True):
    part: str
    trigger_id: str | None


class PersistentSynthSnapshot(Model, frozen=True):
    definition: PreparedSynth
    action_capacity: int
    context_capacity: int
    frame: int
    voices: dict[str, int]
    voice_triggers: dict[str, VoiceAddress]
    part_contexts: dict[str, int]
    trigger_contexts: dict[tuple[str, str], int]
    state: _native.SynthRuntimeSnapshot

    model_config = ConfigDict(arbitrary_types_allowed=True)


def prepare(score: SynthInstrumentScore) -> PreparedSynth:
    """Validate the supported profile and isolate its prepared definitions."""
    score = SynthInstrumentScore.model_validate(score.model_dump())
    output = score.outputs[0].stream
    if not isinstance(output, AudioType):
        raise EngineError("Synth output must be sampled audio")
    timebase = next(t for t in score.timebases if t.name == output.timebase)
    voices: list[SynthVoice] = []
    for voice in score.body.voices:
        if not isinstance(voice, SynthVoice):
            raise EngineError("Oscillator engine requires oscillator voice templates")
        voice = _expand_patch_outputs(voice)
        _validate_voice(voice)
        filters.parameters(voice.processing.filters, _sample_rate(timebase), 1)
        voices.append(voice)
    return PreparedSynth(
        sample_rate=_sample_rate(timebase),
        channels=output.channels,
        instrument=score.body.model_copy(update={"voices": voices}),
        source_instrument=score.body if voices != score.body.voices else None,
    )


def _expand_patch_outputs(voice: SynthVoice) -> SynthVoice:
    motions = dict(voice.motions)
    bindings = list(voice.bindings)
    sources = list(voice.modulation.sources)
    changed = False
    for index, binding in enumerate(bindings):
        if not isinstance(binding, processing.GeneratorBinding):
            continue
        parent = motions[binding.reference]
        if not isinstance(parent.body, Patch):
            continue
        assert binding.output is not None
        child_name = parent.body.outputs[binding.output]
        name = f"patch-{binding.reference}-{child_name}"
        if name in voice.motions:
            raise EngineError(f"Patch output Motion name collides: {name}")
        child = parent.body.motions[child_name]
        if name not in motions:
            motions[name] = MotionUse(
                scope=parent.scope, clock=parent.clock, body=child
            )
        bindings[index] = binding.model_copy(update={"reference": name, "output": None})
        changed = True
    for parent_name, parent in voice.motions.items():
        if not isinstance(parent.body, Patch):
            continue
        if not any(
            isinstance(binding, processing.GeneratorBinding)
            and binding.reference == parent_name
            for binding in voice.bindings
        ):
            continue
        children = {
            name
            for connection in parent.body.events
            for name in (connection.source.partition(".")[0], connection.target)
        } | {source.partition(".")[0] for source in parent.body.event_outputs.values()}
        for child_name in children:
            name = f"patch-{parent_name}-{child_name}"
            if name in voice.motions:
                raise EngineError(f"Patch child Motion name collides: {name}")
            child = parent.body.motions[child_name]
            motions.setdefault(
                name, MotionUse(scope=parent.scope, clock=parent.clock, body=child)
            )
            if any(
                isinstance(binding, processing.GeneratorBinding)
                and binding.reference == name
                for binding in bindings
            ):
                continue
            if any(binding.name == name for binding in bindings):
                raise EngineError(f"Patch child binding name collides: {name}")
            bindings.append(processing.GeneratorBinding(name=name, reference=name))
            sources.append(
                modulation.Source(
                    name=name,
                    scope="voice",
                    minimum=0
                    if isinstance(child, Contour) and child.polarity == "unipolar"
                    else -1,
                    maximum=1,
                )
            )
            changed = True
    event_connections = list(voice.event_connections)
    original_bindings = {binding.name: binding for binding in voice.bindings}
    for index, connection in enumerate(event_connections):
        source_binding = original_bindings[connection.source]
        if not isinstance(source_binding, processing.GeneratorBinding):
            continue
        parent = voice.motions[source_binding.reference]
        if not isinstance(parent.body, Patch):
            continue
        child_name, _, port = parent.body.event_outputs[connection.port].partition(".")
        child_motion = f"patch-{source_binding.reference}-{child_name}"
        child_binding = next(
            binding.name
            for binding in bindings
            if isinstance(binding, processing.GeneratorBinding)
            and binding.reference == child_motion
        )
        event_connections[index] = connection.model_copy(
            update={"source": child_binding, "port": port}
        )
        changed = True
    for parent_name, parent in voice.motions.items():
        if not isinstance(parent.body, Patch) or not any(
            isinstance(binding, processing.GeneratorBinding)
            and binding.reference == parent_name
            for binding in voice.bindings
        ):
            continue
        for connection in parent.body.events:
            if connection.action != "cue":
                continue
            child_name, _, port = connection.source.partition(".")
            source_motion = f"patch-{parent_name}-{child_name}"
            target_motion = f"patch-{parent_name}-{connection.target}"
            source_binding = next(
                binding.name
                for binding in bindings
                if isinstance(binding, processing.GeneratorBinding)
                and binding.reference == source_motion
            )
            target_binding = next(
                binding.name
                for binding in bindings
                if isinstance(binding, processing.GeneratorBinding)
                and binding.reference == target_motion
            )
            assert connection.cue is not None
            event_connections.append(
                processing.MotionEventConnection(
                    source=source_binding,
                    port=port,
                    destination=target_binding,
                    cue=connection.cue,
                )
            )
            changed = True
    if not changed:
        return voice
    return SynthVoice.model_validate(
        voice.model_dump()
        | {
            "motions": motions,
            "bindings": bindings,
            "event_connections": event_connections,
            "modulation": voice.modulation.model_copy(update={"sources": sources}),
        }
    )


def _patch_child_names(settings: SoundSettings) -> set[str]:
    return {
        f"patch-{name}-{child}"
        for name, motion in settings.motions.items()
        if isinstance(motion.body, Patch)
        for child in motion.body.motions
    }


def _patch_start_connections(
    settings: SoundSettings, sources: dict[str, int]
) -> list[tuple[str, str, int, int]]:
    bindings = {
        binding.reference: binding.name
        for binding in settings.bindings
        if isinstance(binding, processing.GeneratorBinding)
    }
    connections: list[tuple[str, str, int, int]] = []
    for parent_name, parent in settings.motions.items():
        if not isinstance(parent.body, Patch):
            continue
        for order, connection in enumerate(parent.body.events):
            if connection.action != "start":
                continue
            child_name, _, port = connection.source.partition(".")
            source_name = f"patch-{parent_name}-{child_name}"
            target_name = f"patch-{parent_name}-{connection.target}"
            if (
                source_name not in settings.motions
                or target_name not in settings.motions
            ):
                continue
            connections.append(
                (bindings[source_name], port, sources[bindings[target_name]], order)
            )
    return connections


class ControlRenderer:
    """Resolve scoped controls and LFO sources for one instrument instance."""

    def __init__(
        self,
        sample_rate: int,
        declarations: dict[str, controls.ControlDeclaration],
        settings: list[SoundSettings],
        backend: Literal["numpy", "native"] = "numpy",
        control_interval: int = 1,
        tempo_map: TempoMap | None = None,
    ) -> None:
        if type(control_interval) is not int or control_interval <= 0:
            raise EngineError("Control interval must be a positive integer")
        self.control_interval = control_interval
        self.cached_span: tuple[int, int] | None = None
        self.cached_values: dict[
            tuple[int, tuple[tuple[str, int], ...]], dict[tuple[str, str], np.ndarray]
        ] = {}
        self.cached_lfos: dict[int, np.ndarray] = {}
        self.cached_envelopes: dict[int, np.ndarray] = {}
        self.cached_controls: dict[tuple[int, str, Fraction], np.ndarray] = {}
        self.sample_rate = sample_rate
        self.tempo_map = tempo_map.model_copy(deep=True) if tempo_map else None
        self.declarations = declarations
        self.settings = settings
        self.patch_children = [_patch_child_names(s) for s in settings]
        self.backend = backend
        self.contexts: list[ControlContext] = []
        self.lfos: list[LFOSource] = []
        self.envelopes: list[EnvelopeSource] = []
        self.new_context("instrument", None, None, 0, {})

    def motion_time(self, motion: MotionUse, seconds: Fraction) -> Fraction:
        if motion.clock == "seconds":
            return seconds
        if self.tempo_map is None:
            raise EngineError("Beat-clock Motion requires a host tempo map")
        if motion.position_driver == "transport":
            return self.tempo_map.beat_at(seconds)
        return self.tempo_map.elapsed_beats(Fraction(0), seconds)

    def motion_seconds(self, motion: MotionUse, at: Fraction) -> Fraction:
        if motion.clock == "seconds":
            return at
        assert self.tempo_map is not None
        seconds = self.tempo_map.time_for_elapsed_beat(at)
        if seconds is None:
            raise EngineError("Beat-clock Motion event has no host time")
        return seconds

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
                if not isinstance(binding, ControlBinding):
                    continue
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

    def apply(
        self, action: instrument_trace.TraceAction, active_voice_ids: Collection[str]
    ) -> bool:
        self.clear_cache()
        if isinstance(action, instrument_trace.LFOObservation):
            matches = [
                (index, source)
                for index, source in enumerate(self.lfos)
                if source.name == action.name
                and source.part is None
                and source.voice_id is None
            ]
            if not matches:
                raise EngineError(f"Unknown instrument LFO: {action.name}")
            for index, source in matches:
                definition = self.settings[source.setting].motions[source.name]
                self.lfos[index] = source.model_copy(
                    update={
                        "state": motion_event(
                            definition,
                            source.state,
                            MotionEvent(
                                at=self.motion_time(
                                    definition, Fraction(action.tick, self.sample_rate)
                                ),
                                ordinal=action.ordinal,
                                action=action.action,
                                rate=(
                                    None
                                    if action.rate is None
                                    else Fraction(action.rate)
                                ),
                                position=(
                                    None
                                    if action.position is None
                                    else Fraction(str(action.position))
                                ),
                                offset=(
                                    None
                                    if action.offset is None
                                    else Fraction(str(action.offset))
                                ),
                            ),
                        )
                    }
                )
        elif isinstance(action, instrument_trace.MotionObservation):
            if any(
                action.name in self.patch_children[index]
                or (
                    (motion := settings.motions.get(action.name)) is not None
                    and isinstance(motion.body, Patch)
                )
                for index, settings in enumerate(self.settings)
            ):
                raise EngineError("Patch-level Motion commands are not implemented")
            for index, source in enumerate(self.lfos):
                if (
                    source.name == action.name
                    and source.voice_id is not None
                    and source.voice_id in active_voice_ids
                    and (source.part, source.trigger_id)
                    == (action.part, action.trigger_id)
                ):
                    definition = self.settings[source.setting].motions[source.name]
                    self.lfos[index] = source.model_copy(
                        update={
                            "state": motion_event(
                                definition,
                                source.state,
                                self._motion_observation(action, definition),
                            )
                        }
                    )
            for index, source in enumerate(self.envelopes):
                if (
                    source.name != action.name
                    or source.voice_id not in active_voice_ids
                    or (source.part, source.trigger_id)
                    != (action.part, action.trigger_id)
                ):
                    continue
                settings = self.settings[source.setting]
                definition = settings.motions[source.name]
                event = self._motion_observation(action, definition)
                if isinstance(definition.body, Stages):
                    result = advance_motion(definition, source.state, event.at, event)
                    state = result.state
                    emitted = [(source.binding_name, e) for e in result.events]
                else:
                    state = motion_event(definition, source.state, event)
                    emitted = []
                self.envelopes[index] = source.model_copy(update={"state": state})
                if emitted:
                    sources = {
                        s.binding_name: i
                        for i, s in enumerate(self.envelopes)
                        if s.setting == source.setting and s.voice_id == source.voice_id
                    }
                    self._dispatch_staged_events(settings, sources, emitted)
        elif isinstance(action, instrument_trace.TriggerContext):
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

    def _motion_observation(
        self, action: instrument_trace.MotionObservation, motion: MotionUse
    ) -> MotionEvent:
        return MotionEvent(
            at=self.motion_time(motion, Fraction(action.tick, self.sample_rate)),
            ordinal=action.ordinal,
            action=action.action,
            position=None
            if action.position is None
            else Fraction(str(action.position)),
            offset=None if action.offset is None else Fraction(str(action.offset)),
        )

    def sources(
        self, settings: SoundSettings, action: instrument_trace.VoiceStart
    ) -> dict[str, int]:
        result: dict[str, int] = {}
        emitted: list[tuple[str, MotionOutputEvent]] = []
        bindings = {b.name: b for b in settings.bindings}
        for source in settings.modulation.sources:
            binding = bindings[source.name]
            if isinstance(binding, processing.GeneratorBinding):
                setting = next(i for i, s in enumerate(self.settings) if s is settings)
                motion = settings.motions[binding.reference]
                if isinstance(motion.body, (Contour, Stages)):
                    if motion.scope != "voice":
                        raise EngineError(
                            "Staged and contour Motions require voice scope"
                        )
                    existing = None
                    if binding.reference in self.patch_children[setting]:
                        existing = next(
                            (
                                i
                                for i, item in enumerate(self.envelopes)
                                if (item.setting, item.name, item.voice_id)
                                == (setting, binding.reference, action.voice_id)
                            ),
                            None,
                        )
                    if existing is not None:
                        result[source.name] = existing
                        continue
                    index = len(self.envelopes)
                    initial = initial_motion(
                        motion,
                        self.motion_time(
                            motion, Fraction(action.tick, self.sample_rate)
                        ),
                    )
                    if isinstance(motion.body, Stages):
                        advanced = advance_motion(
                            motion,
                            initial,
                            self.motion_time(
                                motion, Fraction(action.tick, self.sample_rate)
                            ),
                            MotionEvent(
                                at=self.motion_time(
                                    motion, Fraction(action.tick, self.sample_rate)
                                ),
                                ordinal=action.ordinal,
                                action="note_on",
                            ),
                        )
                        initial = advanced.state
                        emitted.extend((binding.name, e) for e in advanced.events)
                    elif motion.body.release:
                        initial = motion_event(
                            motion,
                            initial,
                            MotionEvent(
                                at=self.motion_time(
                                    motion, Fraction(action.tick, self.sample_rate)
                                ),
                                ordinal=action.ordinal,
                                action="note_on",
                            ),
                        )
                    self.envelopes.append(
                        EnvelopeSource(
                            setting=setting,
                            name=binding.reference,
                            binding_name=binding.name,
                            part=action.part,
                            trigger_id=action.trigger_id,
                            voice_id=action.voice_id,
                            state=initial,
                        )
                    )
                    result[source.name] = index
                    continue
                part = action.part if motion.scope in ("part", "voice") else None
                voice_id = action.voice_id if motion.scope == "voice" else None
                index = next(
                    (
                        i
                        for i, s in enumerate(self.lfos)
                        if (s.setting, s.name, s.part, s.voice_id)
                        == (setting, binding.reference, part, voice_id)
                    ),
                    None,
                )
                if index is None:
                    index = len(self.lfos)
                    self.lfos.append(
                        LFOSource(
                            setting=setting,
                            name=binding.reference,
                            part=part,
                            trigger_id=action.trigger_id
                            if voice_id is not None
                            else None,
                            voice_id=voice_id,
                            state=initial_motion(
                                motion,
                                self.motion_time(
                                    motion,
                                    Fraction(
                                        action.tick if voice_id is not None else 0,
                                        self.sample_rate,
                                    ),
                                ),
                            ),
                        )
                    )
                result[source.name] = index
                continue
            part = None if source.scope == "instrument" else action.part
            trigger_id = action.trigger_id if source.scope == "trigger" else None
            context_id = self.context(source.scope, part, trigger_id)
            if context_id is None:
                if source.scope == "trigger" and trigger_id is not None:
                    raise EngineError("Voice start is missing its trigger context")
                assert source.scope in ("instrument", "part", "trigger")
                context_id = self.new_context(source.scope, part, trigger_id, 0, {})
            result[source.name] = context_id
        self._dispatch_staged_events(settings, result, emitted)
        return result

    def release(
        self,
        settings: SoundSettings,
        sources: dict[str, int],
        action: instrument_trace.VoiceRetirement,
        renderer: EnvelopeRenderer,
    ) -> None:
        self.clear_cache()
        assert renderer.release_frame is not None
        voice_release_at = (
            Fraction(action.tick, self.sample_rate)
            + (renderer.release_frame - renderer.frame_count) / self.sample_rate
        )
        emitted: list[tuple[str, MotionOutputEvent]] = []
        released: set[int] = set()
        for binding in settings.bindings:
            if not isinstance(binding, processing.GeneratorBinding) or not isinstance(
                settings.motions[binding.reference].body, (Contour, Stages)
            ):
                continue
            index = sources[binding.name]
            if index in released:
                continue
            released.add(index)
            source = self.envelopes[index]
            definition = settings.motions[binding.reference]
            if (
                isinstance(definition.body, Contour)
                and binding.release_timing == ReleaseTiming.voice
                and voice_release_at > Fraction(action.tick, self.sample_rate)
            ):
                self.envelopes[index] = source.model_copy(
                    update={"pending_release": voice_release_at}
                )
                continue
            event = MotionEvent(
                at=self.motion_time(
                    definition, Fraction(action.tick, self.sample_rate)
                ),
                ordinal=action.ordinal,
                action="note_off",
            )
            if isinstance(definition.body, Stages):
                advanced = advance_motion(definition, source.state, event.at, event)
                state = advanced.state
                emitted.extend((binding.name, e) for e in advanced.events)
            else:
                state = motion_event(definition, source.state, event)
            self.envelopes[index] = source.model_copy(update={"state": state})
        self._dispatch_staged_events(settings, sources, emitted)

    def stop(self, voice_id: str) -> None:
        self.clear_cache()
        self.envelopes = [
            s.model_copy(update={"pending_release": None})
            if s.voice_id == voice_id and s.pending_release is not None
            else s
            for s in self.envelopes
        ]

    def values(
        self, settings: SoundSettings, sources: dict[str, int], start: int, frames: int
    ) -> dict[tuple[str, str], np.ndarray]:
        """Resolve every declared target once, independently of its DSP consumer."""
        span = (start, frames)
        if self.cached_span != span:
            self.clear_cache()
            self.cached_span = span
        key = (id(settings), tuple(sorted(sources.items())))
        if key in self.cached_values:
            return {n: v.copy() for n, v in self.cached_values[key].items()}
        self._motion_event_values(settings, sources, start, frames)
        signals: dict[str, np.ndarray] = {}
        for binding in settings.bindings:
            if isinstance(binding, processing.GeneratorBinding):
                index = sources[binding.name]
                motion = settings.motions[binding.reference]
                if isinstance(motion.body, (Contour, Stages)):
                    if index not in self.cached_envelopes:
                        source = self.envelopes[index]
                        state = source.state
                        assert isinstance(motion.body, Contour)
                        values = np.empty(frames)
                        for i in range(frames):
                            at = Fraction(start + i, self.sample_rate)
                            if (
                                source.pending_release is not None
                                and at >= source.pending_release
                            ):
                                state = motion_event(
                                    motion,
                                    state,
                                    MotionEvent(
                                        at=self.motion_time(
                                            motion, source.pending_release
                                        ),
                                        ordinal=0,
                                        action="note_off",
                                    ),
                                )
                                source = source.model_copy(
                                    update={"state": state, "pending_release": None}
                                )
                                self.envelopes[index] = source
                            values[i] = motion_at(
                                motion, state, self.motion_time(motion, at)
                            ).value
                        self.cached_envelopes[index] = np.column_stack(
                            (values, np.ones(frames))
                        )
                    signals[binding.name] = self.cached_envelopes[index]
                    continue
                if index not in self.cached_lfos:
                    state = self.lfos[index].state.runtime
                    assert isinstance(motion.body, Cycle)
                    assert isinstance(state, CycleState)
                    position = state.position
                    if (
                        motion.clock == "seconds"
                        and not position.paused
                        and position.direction == 1
                    ):
                        samples = lfo_samples(
                            cycle_lfo(motion),
                            lfo.LFOState(
                                at=position.at,
                                started_at=position.at - position.age,
                                phase=position.coordinate % 1,
                                rate=position.rate,
                            ),
                            start,
                            frames,
                            self.sample_rate,
                            self.backend,
                            self.control_interval,
                        )
                        samples[:, 0] = (
                            motion.body.center + motion.body.depth * samples[:, 0]
                        )
                    else:
                        samples = np.empty((frames, 2))
                        for i in range(frames):
                            value = motion_at(
                                motion,
                                self.lfos[index].state,
                                self.motion_time(
                                    motion, Fraction(start + i, self.sample_rate)
                                ),
                            )
                            samples[i] = value.value, value.weight
                    self.cached_lfos[index] = samples
                signals[binding.name] = self.cached_lfos[index]
                continue
            assert isinstance(binding, ControlBinding)
            context_id = sources[binding.name]
            source_key = (context_id, binding.control, binding.smoothing)
            if source_key not in self.cached_controls:
                state = next(
                    r.state
                    for r in self.contexts[context_id].ramps
                    if r.control == binding.control
                    and r.state.smoothing == binding.smoothing
                )
                self.cached_controls[source_key] = np.column_stack(
                    (
                        control.control_samples(state, start, frames, self.sample_rate),
                        np.ones(frames),
                    )
                )
            signals[binding.name] = self.cached_controls[source_key]
        output = control.modulation_samples(settings.modulation, signals, frames)
        self.cached_values[key] = output
        return {n: v.copy() for n, v in output.items()}

    def _motion_event_values(
        self, settings: SoundSettings, sources: dict[str, int], start: int, frames: int
    ) -> None:
        staged: list[tuple[str, int, MotionUse]] = []
        seen: set[int] = set()
        for binding in settings.bindings:
            if not isinstance(binding, processing.GeneratorBinding):
                continue
            index = sources[binding.name]
            if index in seen or not isinstance(
                settings.motions[binding.reference].body, Stages
            ):
                continue
            seen.add(index)
            staged.append((binding.name, index, settings.motions[binding.reference]))
        patch_starts = _patch_start_connections(settings, sources)
        contour_targets = {target for _, _, target, _ in patch_starts}
        motion_indices = {index for _, index, _ in staged} | contour_targets
        if not motion_indices or all(
            index in self.cached_envelopes for index in motion_indices
        ):
            return
        values = {index: np.empty(frames) for index in motion_indices}
        bindings = {binding.name: binding for binding in settings.bindings}
        cycle_ports = list(
            dict.fromkeys(
                [
                    (connection.source, connection.port)
                    for connection in settings.event_connections
                    if isinstance(
                        (binding := bindings[connection.source]),
                        processing.GeneratorBinding,
                    )
                    and isinstance(settings.motions[binding.reference].body, Cycle)
                ]
                + [
                    (source, port)
                    for source, port, _, _ in patch_starts
                    if isinstance(
                        (binding := bindings[source]),
                        processing.GeneratorBinding,
                    )
                    and isinstance(settings.motions[binding.reference].body, Cycle)
                ]
            )
        )
        for i in range(frames):
            boundary = Fraction(start + i, self.sample_rate)
            events_seen = 0
            sample_values: dict[int, float] = {}
            pending_starts: list[tuple[Fraction, int, int]] = []
            cycle_events: list[tuple[Fraction, int, str, str]] = []
            for order, (source_name, port) in enumerate(cycle_ports):
                binding = bindings[source_name]
                assert isinstance(binding, processing.GeneratorBinding)
                motion = settings.motions[binding.reference]
                assert isinstance(motion.body, Cycle)
                source = self.lfos[sources[source_name]]
                assert isinstance(source.state.runtime, CycleState)
                position = source.state.runtime.position
                origin = position.at
                previous = max(origin, boundary - Fraction(1, self.sample_rate))
                if boundary <= previous:
                    continue
                marker = next(m.position for m in motion.body.markers if m.name == port)
                first = position.coordinate + position.rate * (previous - origin)
                last = position.coordinate + position.rate * (boundary - origin)
                first_turn = floor(first - marker) + 1
                last_turn = floor(last - marker)
                if last_turn - first_turn + 1 > 4096 - len(cycle_events):
                    raise EngineError("Motion event capacity exceeded")
                for turn in range(first_turn, last_turn + 1):
                    at = origin + (turn + marker - position.coordinate) / position.rate
                    cycle_events.append((at, order, source_name, port))
            cycle_events.sort()
            cycle_event_index = 0
            while True:
                attempts = [
                    (
                        name,
                        index,
                        motion,
                        advance_motion(
                            motion,
                            self.envelopes[index].state,
                            self.motion_time(motion, boundary),
                        ),
                    )
                    for name, index, motion in staged
                ]
                next_staged_event = (
                    min(
                        (
                            self.motion_seconds(motion, event.at)
                            for _, _, motion, result in attempts
                            for event in result.events
                        ),
                        default=None,
                    )
                    if settings.event_connections or patch_starts
                    else None
                )
                next_cycle_event = (
                    cycle_events[cycle_event_index][0]
                    if cycle_event_index < len(cycle_events)
                    else None
                )
                next_event = min(
                    (
                        at
                        for at in (next_staged_event, next_cycle_event)
                        if at is not None
                    ),
                    default=None,
                )
                if next_event is None:
                    for _, index, _, result in attempts:
                        self.envelopes[index] = self.envelopes[index].model_copy(
                            update={"state": result.state}
                        )
                        sample_values[index] = result.value.value
                    break
                emitted: list[tuple[str, MotionOutputEvent]] = []
                for name, index, motion in staged:
                    result = advance_motion(
                        motion,
                        self.envelopes[index].state,
                        self.motion_time(motion, next_event),
                    )
                    self.envelopes[index] = self.envelopes[index].model_copy(
                        update={"state": result.state}
                    )
                    emitted.extend((name, event) for event in result.events)
                while (
                    cycle_event_index < len(cycle_events)
                    and cycle_events[cycle_event_index][0] == next_event
                ):
                    at, _, name, port = cycle_events[cycle_event_index]
                    emitted.append(
                        (
                            name,
                            MotionOutputEvent(
                                at=at,
                                port=port,
                                stage=name,
                                activation=0,
                            ),
                        )
                    )
                    cycle_event_index += 1
                events_seen += len(emitted)
                if events_seen > 4096:
                    raise EngineError("Motion event capacity exceeded")
                if emitted:
                    self._dispatch_staged_events(
                        settings, sources, emitted, pending_starts
                    )
            self._start_patch_contours(settings, pending_starts)
            for _, index, _ in staged:
                values[index][i] = sample_values[index]
            for index in contour_targets:
                motion = settings.motions[self.envelopes[index].name]
                values[index][i] = motion_at(
                    motion,
                    self.envelopes[index].state,
                    self.motion_time(motion, boundary),
                ).value
        for index, samples in values.items():
            self.cached_envelopes[index] = np.column_stack((samples, np.ones(frames)))

    def _dispatch_staged_events(
        self,
        settings: SoundSettings,
        sources: dict[str, int],
        emitted: list[tuple[str, MotionOutputEvent]],
        starts: list[tuple[Fraction, int, int]] | None = None,
    ) -> None:
        pending = deque(emitted)
        delivered = 0
        bindings = {b.name: b for b in settings.bindings}
        patch_starts = _patch_start_connections(settings, sources)
        pending_starts = [] if starts is None else starts
        while pending:
            name, event = pending.popleft()
            source_binding = bindings[name]
            assert isinstance(source_binding, processing.GeneratorBinding)
            source_motion = settings.motions[source_binding.reference]
            seconds = self.motion_seconds(source_motion, event.at)
            for source_name, port, target_index, order in patch_starts:
                if source_name == name and port == event.port:
                    pending_starts.append((seconds, order, target_index))
                    delivered += 1
                    if delivered > 4096:
                        raise EngineError("Motion event connection capacity exceeded")
            for connection in settings.event_connections:
                if connection.source != name or connection.port != event.port:
                    continue
                target_index = sources[connection.destination]
                binding = bindings[connection.destination]
                assert isinstance(binding, processing.GeneratorBinding)
                target_motion = settings.motions[binding.reference]
                at = self.motion_time(target_motion, seconds)
                state = self.envelopes[target_index].state
                assert isinstance(state.runtime, StageState)
                result = advance_motion(
                    target_motion,
                    state,
                    at,
                    MotionEvent(
                        at=at,
                        ordinal=state.runtime.ordinal + 1,
                        action="cue",
                        cue=connection.cue,
                    ),
                )
                self.envelopes[target_index] = self.envelopes[target_index].model_copy(
                    update={"state": result.state}
                )
                pending.extend((connection.destination, e) for e in result.events)
                delivered += 1
                if delivered > 4096:
                    raise EngineError("Motion event connection capacity exceeded")
        if starts is None:
            self._start_patch_contours(settings, pending_starts)

    def _start_patch_contours(
        self, settings: SoundSettings, starts: list[tuple[Fraction, int, int]]
    ) -> None:
        if len(starts) > 4096:
            raise EngineError("Patch event capacity exceeded")
        for seconds, _, index in sorted(starts):
            source = self.envelopes[index]
            motion = settings.motions[source.name]
            at = self.motion_time(motion, seconds)
            assert isinstance(source.state.runtime, ContourState)
            state = motion_event(
                motion,
                source.state,
                MotionEvent(
                    at=at,
                    ordinal=source.state.runtime.ordinal + 1,
                    action="start",
                ),
            )
            self.envelopes[index] = source.model_copy(update={"state": state})

    def clear_cache(self) -> None:
        """Discard ephemeral arrays after an event, restore, or new render span."""
        self.cached_span = None
        self.cached_values.clear()
        self.cached_lfos.clear()
        self.cached_envelopes.clear()
        self.cached_controls.clear()

    def parameters(
        self, settings: SoundSettings, sources: dict[str, int], start: int, frames: int
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        return processing_parameters(
            settings,
            self.values(settings, sources, start, frames),
            self.sample_rate,
            start,
            frames,
        )


class OfflineSynth:
    """Render one prepared synth instance with exact frame-boundary actions."""

    def __init__(
        self,
        definition: PreparedSynth,
        backend: Literal["numpy", "native"] = "numpy",
        control_interval: int = 1,
        tempo_map: TempoMap | None = None,
    ) -> None:
        if backend not in ("numpy", "native"):
            raise EngineError(f"Unknown synth backend: {backend}")
        self.backend: Literal["numpy", "native"] = backend
        self.definition = definition.model_copy(deep=True)
        self.frame = 0
        self.voices: dict[str, VoiceSnapshot] = {}
        self.templates = {
            v.name: v
            for v in self.definition.instrument.voices
            if isinstance(v, SynthVoice)
        }
        if tempo_map is None and any(
            m.clock == "beats"
            for v in self.templates.values()
            for m in v.motions.values()
        ):
            raise EngineError("Beat-clock Motion requires a host tempo map")
        self.controls = ControlRenderer(
            self.definition.sample_rate,
            self.definition.instrument.controls,
            list(self.templates.values()),
            backend,
            control_interval,
            tempo_map,
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
            control_interval=self.controls.control_interval,
            definition=self.definition,
            frame=self.frame,
            voices=list(self.voices.values()),
            contexts=self.controls.contexts,
            lfos=self.controls.lfos,
            envelopes=self.controls.envelopes,
            tempo_map=self.controls.tempo_map,
        ).model_copy(deep=True)

    def restore(self, snapshot: SynthSnapshot) -> None:
        if snapshot.control_interval != self.controls.control_interval:
            raise EngineError("Snapshot belongs to a different control interval")
        if snapshot.definition != self.definition:
            raise EngineError("Snapshot belongs to a different prepared synth")
        if snapshot.tempo_map != self.controls.tempo_map:
            raise EngineError("Snapshot belongs to a different host tempo map")
        if any(v.renderer.backend != self.backend for v in snapshot.voices):
            raise EngineError("Snapshot belongs to a different synth backend")
        snapshot = snapshot.model_copy(deep=True)
        self.frame = snapshot.frame
        self.voices = {v.voice_id: v for v in snapshot.voices}
        self.controls.clear_cache()
        self.controls.contexts = snapshot.contexts
        self.controls.lfos = snapshot.lfos
        self.controls.envelopes = snapshot.envelopes

    def _apply(self, action: instrument_trace.TraceAction) -> None:
        if self.controls.apply(action, self.voices):
            return
        elif isinstance(action, VoiceStart):
            self._start_voice(action)
        elif isinstance(action, instrument_trace.VoiceRetirement):
            if action.action == "fade":
                raise EngineError("Fade retirement is not implemented")
            if action.action == "stop":
                self.voices.pop(action.voice_id, None)
                self.controls.stop(action.voice_id)
            elif voice := self.voices.get(action.voice_id):
                if voice.renderer.release():
                    self.controls.release(
                        self.templates[voice.template],
                        voice.sources,
                        action,
                        voice.renderer,
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
        source_instrument = (
            self.definition.source_instrument or self.definition.instrument
        )
        source = next(
            (v for v in source_instrument.voices if v.name == action.template),
            None,
        )
        if (
            template is None
            or not isinstance(source, SynthVoice)
            or action.settings != source
            or action.oscillator != source.oscillator
            or action.channels != source.channels
        ):
            raise EngineError("Voice start must match its prepared synth template")
        if action.key is None:
            raise EngineError("Synth voice starts require a key for oscillator gain")
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
                    filters=template.processing.filters,
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
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        template = self.templates[voice.template]
        tuning, gains, filter_values = self.controls.parameters(
            template, voice.sources, start, frames
        )
        frequencies = np.array(
            [frequency(voice.frequency_hz, float(c)) for c in tuning]
        )
        if not np.all(np.isfinite(frequencies) & (frequencies > 0)):
            raise EngineError(
                f"Invalid frequency for {voice.voice_id} at frame {start}"
            )
        return frequencies, gains, filter_values

    def _render(self, start: int, frames: int) -> np.ndarray:
        output = np.zeros((frames, len(self.definition.channels)), dtype=np.float64)
        for voice in list(self.voices.values()):
            count = voice.renderer.active_frames(frames)
            frequencies, gains, filter_values = self._parameters(voice, start, count)
            output += voice.renderer.render(
                frames, frequencies[:, None], gains, filter_values
            )
            if voice.renderer.complete:
                del self.voices[voice.voice_id]
        return output


class PersistentSynth:
    """Native oscillator synth for one static uFor voice template."""

    def __init__(
        self,
        definition: PreparedSynth,
        voices: int = 16,
        action_capacity: int = 64,
        context_capacity: int = 64,
        tempo_map: TempoMap | None = None,
    ) -> None:
        if type(voices) is not int or voices <= 0:
            raise EngineError("Persistent synth voice capacity must be positive")
        if type(action_capacity) is not int or action_capacity <= 0:
            raise EngineError("Persistent synth action capacity must be positive")
        if type(context_capacity) is not int or context_capacity <= 0:
            raise EngineError("Persistent synth context capacity must be positive")
        templates = definition.instrument.voices
        if len(templates) != 1 or not isinstance(templates[0], SynthVoice):
            raise EngineError("Persistent synth requires one oscillator voice template")
        template = templates[0]
        if any(m.clock == "beats" for m in template.motions.values()):
            if tempo_map is None:
                raise EngineError("Beat-clock Motion requires a host tempo map")
        if template.processing != Processing(
            tuning_cents=template.processing.tuning_cents,
            filters=template.processing.filters,
        ):
            raise EngineError(
                "Persistent synth gain and spatial processing are not implemented"
            )
        route = [
            sum(r.gain for r in template.channels if r.output == c)
            for c in definition.channels
        ]
        rate = definition.sample_rate
        self.definition = definition.model_copy(deep=True)
        self.instrument = self.definition.instrument
        self.template = template
        self.frame = 0
        self.voices: dict[str, int] = {}
        self.voice_triggers: dict[str, VoiceAddress] = {}
        self.action_capacity = action_capacity
        self.context_capacity = context_capacity
        self._action_buffer = np.empty((action_capacity, 6), dtype=np.float64)
        (
            controls,
            smoothing,
            parameters,
            self.control_sources,
            self.control_scopes,
            self.source_controls,
            lfos,
            lfo_rationals,
            self.lfo_sources,
            self.voice_lfo_sources,
            envelope_initials,
            envelope_attacks,
            envelope_releases,
            envelope_parameters,
            self.named_motion_sources,
            self.source_scopes,
            runtime_filters,
            staged_motions,
            self.staged_motion_sources,
            event_connections,
        ) = _persistent_modulation(definition, template)
        self.part_contexts: dict[str, int] = {}
        self.trigger_contexts: dict[tuple[str, str], int] = {}
        self.runtime = _native.SynthRuntime(
            rate,
            [Waveform.sine, Waveform.square, Waveform.triangle].index(
                template.oscillator.waveform
            ),
            float(template.oscillator.duty_cycle),
            np.tile(np.asarray(route, dtype=np.float64), (voices, 1)),
            template.envelope.initial,
            _runtime_segments(template.envelope.segments, rate),
            _runtime_segments(template.envelope.release, rate),
            float(template.minimum_hold_seconds * rate),
            controls,
            smoothing,
            lfos,
            lfo_rationals,
            runtime_filters,
            parameters,
            context_capacity,
        )
        self.runtime.set_named_envelopes(
            envelope_initials,
            envelope_attacks,
            envelope_releases,
            envelope_parameters,
        )
        if tempo_map is not None:
            self.runtime.set_beat_clock(
                [
                    (
                        float(p.at_seconds * rate),
                        float(tempo_map.elapsed_beats(Fraction(0), p.at_seconds)),
                        float(p.beat),
                        float(p.bpm / (60 * rate)) if p.running else 0.0,
                    )
                    for p in tempo_map.points
                ]
            )
            self.runtime.set_beat_named_envelopes(
                [
                    i
                    for name, indices in self.named_motion_sources.items()
                    if template.motions[name].clock == "beats"
                    for i in indices
                ]
            )
        self.runtime.set_staged_motions(staged_motions, event_connections)
        patch_children = _patch_child_names(template)
        self.runtime.set_motion_state_owners(
            _motion_state_owners(
                len(lfos), patch_children, self.lfo_sources, self.voice_lfo_sources
            ),
            _motion_state_owners(
                len(envelope_initials), patch_children, self.named_motion_sources
            ),
            _motion_state_owners(
                len(staged_motions), patch_children, self.staged_motion_sources
            ),
        )
        for source_map in (
            self.lfo_sources,
            self.voice_lfo_sources,
            self.named_motion_sources,
            self.staged_motion_sources,
        ):
            for name, indices in source_map.items():
                if name in patch_children:
                    source_map[name] = indices[:1]
        patch_events: list[tuple[int, float, int, str | None, int]] = []
        patch_stage_starts: list[tuple[int, str, int, int]] = []
        for parent_name, parent in template.motions.items():
            if not isinstance(parent.body, Patch):
                continue
            if parent.body.events and (
                f"patch-{parent_name}-{parent.body.events[0].target}"
                not in template.motions
            ):
                continue
            for order, connection in enumerate(parent.body.events):
                if connection.action != "start":
                    continue
                child_name, _, port = connection.source.partition(".")
                child = parent.body.motions[child_name]
                destination = self.named_motion_sources[
                    f"patch-{parent_name}-{connection.target}"
                ][0]
                if isinstance(child, Cycle):
                    marker = next(m.position for m in child.markers if m.name == port)
                    patch_events.append(
                        (
                            self.voice_lfo_sources[f"patch-{parent_name}-{child_name}"][
                                0
                            ],
                            float(marker),
                            destination,
                            None,
                            order,
                        )
                    )
                else:
                    assert isinstance(child, Stages)
                    patch_stage_starts.append(
                        (
                            self.staged_motion_sources[
                                f"patch-{parent_name}-{child_name}"
                            ][0],
                            port,
                            destination,
                            order,
                        )
                    )
        bindings = {binding.name: binding for binding in template.bindings}
        for connection in template.event_connections:
            source_binding = bindings[connection.source]
            assert isinstance(source_binding, processing.GeneratorBinding)
            source_motion = template.motions[source_binding.reference]
            if not isinstance(source_motion.body, Cycle):
                continue
            marker = next(
                m.position
                for m in source_motion.body.markers
                if m.name == connection.port
            )
            target_binding = bindings[connection.destination]
            assert isinstance(target_binding, processing.GeneratorBinding)
            patch_events.append(
                (
                    self.voice_lfo_sources[source_binding.reference][0],
                    float(marker),
                    self.staged_motion_sources[target_binding.reference][0],
                    connection.cue,
                    len(patch_events),
                )
            )
        self.runtime.set_patch_events(patch_events)
        self.runtime.set_patch_stage_starts(patch_stage_starts)

    def advance(
        self, actions: list[instrument_trace.TraceAction], start: int, end: int
    ) -> np.ndarray:
        """Render one block into a newly owned output array."""
        if start != self.frame or end <= start:
            raise EngineError(
                "advance must continue from the current nonempty interval"
            )
        output = np.empty((end - start, len(self.definition.channels)))
        self.advance_into(actions, start, end, output)
        return output

    def advance_into(
        self,
        actions: list[instrument_trace.TraceAction],
        start: int,
        end: int,
        output: np.ndarray,
    ) -> None:
        """Render one block into a disjoint caller-owned output array."""
        if start != self.frame or end <= start:
            raise EngineError(
                "advance must continue from the current nonempty interval"
            )
        if (
            output.shape != (end - start, len(self.definition.channels))
            or output.dtype != np.float64
            or not output.flags.c_contiguous
            or not output.flags.writeable
        ):
            raise EngineError(
                "Persistent synth output must be writable C-contiguous float64 "
                "with shape (frames, channels)"
            )
        active = self.runtime.active_slots()
        contexts = self.runtime.active_contexts()
        encoded = self.native_live_actions(actions, start, end, active, contexts)
        self.runtime.process_actions_into(output, 0, 1, 0, encoded, len(encoded))
        self.finish_native_live_block(
            end, self.runtime.active_slots(), self.runtime.active_contexts()
        )

    def native_live_actions(
        self,
        actions: list[instrument_trace.TraceAction],
        start: int,
        end: int,
        active: list[bool],
        contexts: list[bool],
    ) -> np.ndarray:
        """Admit and encode one block without running its native DSP."""
        if start != self.frame or end <= start:
            raise EngineError(
                "advance must continue from the current nonempty interval"
            )
        ordered = sorted(actions, key=lambda a: (a.tick, a.ordinal))
        if any(a.tick < start or a.tick >= end for a in ordered):
            raise EngineError("actions must belong to the rendered interval")
        self.voices = {n: s for n, s in self.voices.items() if active[s]}
        self.voice_triggers = {
            n: address for n, address in self.voice_triggers.items() if n in self.voices
        }
        self.trigger_contexts = {
            n: s for n, s in self.trigger_contexts.items() if contexts[s]
        }
        count = 0
        for action in ordered:
            offset = action.tick - start
            if isinstance(action, instrument_trace.TriggerContext):
                if action.controls.keys() != self.instrument.controls.keys():
                    raise EngineError(
                        "Trigger context must contain all declared controls"
                    )
                for name, value in action.controls.items():
                    self.instrument.controls[name].validate_value(value)
                if "trigger" in self.source_scopes:
                    context, count = self._new_context(2, contexts, offset, count)
                    self.trigger_contexts[action.part, action.trigger_id] = context
                    for source, scope in enumerate(self.control_scopes):
                        if scope == "trigger":
                            value = action.controls[self.source_controls[source]]
                            self._encode_action(
                                count, offset, 7, context, source, value, 0
                            )
                            count += 1
            elif isinstance(action, instrument_trace.ControlObservation):
                declaration = self.instrument.controls.get(action.control)
                if declaration is None:
                    raise EngineError(f"Unknown control: {action.control}")
                declaration.validate_value(action.value)
                context = -1
                if action.scope == "part":
                    if action.part is None:
                        raise EngineError("Part control is missing its part")
                    context, count = self._part_context(
                        action.part, contexts, offset, count
                    )
                elif action.scope == "trigger":
                    if action.part is None or action.trigger_id is None:
                        raise EngineError("Trigger control is missing its context")
                    if (
                        context := self.trigger_contexts.get(
                            (action.part, action.trigger_id)
                        )
                    ) is None:
                        continue
                for source in self.control_sources.get(action.control, []):
                    if self.control_scopes[source] == action.scope:
                        self._encode_action(
                            count, offset, 4, source, action.value, context, 0
                        )
                        count += 1
            elif isinstance(action, instrument_trace.LFOObservation):
                sources = self.lfo_sources.get(action.name)
                if not sources:
                    raise EngineError(f"Unknown instrument LFO: {action.name}")
                for source in sources:
                    self._encode_action(
                        count,
                        offset,
                        8,
                        source,
                        (
                            "reset",
                            "rate",
                            "pause",
                            "resume",
                            "reverse",
                            "seek",
                            "shift",
                        ).index(action.action),
                        action.rate
                        if action.rate is not None
                        else action.position
                        if action.position is not None
                        else action.offset
                        if action.offset is not None
                        else 0,
                        0,
                    )
                    count += 1
            elif isinstance(action, instrument_trace.MotionObservation):
                motion = self.template.motions.get(action.name)
                if action.name in _patch_child_names(self.template) or (
                    motion is not None and isinstance(motion.body, Patch)
                ):
                    raise EngineError("Patch-level Motion commands are not implemented")
                command = ("pause", "resume", "reverse", "seek", "shift").index(
                    action.action
                )
                for voice_id, slot in self.voices.items():
                    if self.voice_triggers[voice_id] != VoiceAddress(
                        part=action.part, trigger_id=action.trigger_id
                    ):
                        continue
                    for kind, sources in (
                        (9, self.named_motion_sources),
                        (10, self.staged_motion_sources),
                        (11, self.voice_lfo_sources),
                    ):
                        for source in sources.get(action.name, []):
                            self._encode_action(
                                count,
                                offset,
                                kind,
                                slot,
                                source,
                                command,
                                action.position
                                if action.position is not None
                                else action.offset
                                if action.offset is not None
                                else 0,
                            )
                            count += 1
            elif isinstance(action, VoiceStart):
                part_context = -1
                trigger_context = -1
                if "part" in self.source_scopes:
                    part_context, count = self._part_context(
                        action.part, contexts, offset, count
                    )
                if "trigger" in self.source_scopes:
                    if (
                        action.trigger_id is None
                        or (
                            trigger_context := self.trigger_contexts.get(
                                (action.part, action.trigger_id)
                            )
                        )
                        is None
                    ):
                        raise EngineError("Voice start is missing its trigger context")
                slot, frequency_hz, gain, phase = self._start(action, active)
                self._encode_action(count, offset, 0, slot, frequency_hz, gain, phase)
                count += 1
                if part_context >= 0 or trigger_context >= 0:
                    self._encode_action(
                        count,
                        offset,
                        5,
                        slot,
                        part_context,
                        trigger_context,
                        0,
                    )
                    count += 1
            elif isinstance(action, instrument_trace.VoiceRetirement):
                if action.action == "fade":
                    raise EngineError("Fade retirement is not implemented")
                if (slot := self.voices.get(action.voice_id)) is not None:
                    kind = 2 if action.action == "stop" else 1
                    self._encode_action(count, offset, kind, slot, 0, 0, 0)
                    count += 1
                    if kind == 2:
                        active[slot] = False
                        del self.voices[action.voice_id]
                        del self.voice_triggers[action.voice_id]
            else:
                raise EngineError(
                    f"Unsupported persistent synth action at frame {action.tick}: "
                    f"{type(action).__name__}"
                )
        return self._action_buffer[:count]

    def finish_native_live_block(
        self, end: int, active: list[bool], contexts: list[bool]
    ) -> None:
        """Commit native liveness after a successfully processed block."""
        self.voices = {n: s for n, s in self.voices.items() if active[s]}
        self.voice_triggers = {
            n: address for n, address in self.voice_triggers.items() if n in self.voices
        }
        self.trigger_contexts = {
            n: s for n, s in self.trigger_contexts.items() if contexts[s]
        }
        self.frame = end

    def snapshot(self) -> PersistentSynthSnapshot:
        return PersistentSynthSnapshot(
            definition=self.definition.model_copy(deep=True),
            action_capacity=self.action_capacity,
            context_capacity=self.context_capacity,
            frame=self.frame,
            voices=self.voices.copy(),
            voice_triggers=self.voice_triggers.copy(),
            part_contexts=self.part_contexts.copy(),
            trigger_contexts=self.trigger_contexts.copy(),
            state=self.runtime.snapshot(),
        )

    def restore(self, snapshot: PersistentSynthSnapshot) -> None:
        if snapshot.definition != self.definition:
            raise EngineError("Snapshot belongs to a different prepared synth")
        if snapshot.action_capacity != self.action_capacity:
            raise EngineError("Snapshot belongs to a different action capacity")
        if snapshot.context_capacity != self.context_capacity:
            raise EngineError("Snapshot belongs to a different context capacity")
        self.runtime.restore(snapshot.state)
        self.frame = snapshot.frame
        self.voices = snapshot.voices.copy()
        self.voice_triggers = snapshot.voice_triggers.copy()
        self.part_contexts = snapshot.part_contexts.copy()
        self.trigger_contexts = snapshot.trigger_contexts.copy()

    def _new_context(
        self, kind: int, active: list[bool], offset: int, count: int
    ) -> tuple[int, int]:
        context = next((i for i, value in enumerate(active) if not value), None)
        if context is None:
            raise EngineError("Persistent synth context capacity exceeded")
        self._encode_action(count, offset, 6, context, kind, 0, 0)
        active[context] = True
        return context, count + 1

    def _part_context(
        self, part: str, active: list[bool], offset: int, count: int
    ) -> tuple[int, int]:
        if (context := self.part_contexts.get(part)) is not None:
            return context, count
        context, count = self._new_context(1, active, offset, count)
        self.part_contexts[part] = context
        return context, count

    def _start(
        self,
        action: VoiceStart,
        active: list[bool],
    ) -> tuple[int, float, float, float]:
        if (
            action.pitch_hz is None
            or not isfinite(action.pitch_hz)
            or action.pitch_hz <= 0
        ):
            raise EngineError("Synth voice requires positive resolved pitch_hz")
        if action.voice_id in self.voices:
            raise EngineError(f"Duplicate active voice: {action.voice_id}")
        source_instrument = (
            self.definition.source_instrument or self.definition.instrument
        )
        source = source_instrument.voices[0]
        assert isinstance(source, SynthVoice)
        if (
            action.template != self.template.name
            or action.settings != source
            or action.oscillator != source.oscillator
            or action.channels != source.channels
        ):
            raise EngineError("Voice start must match its prepared synth template")
        if action.key is None:
            raise EngineError("Synth voice starts require a key for oscillator gain")
        slot = next((i for i, value in enumerate(active) if not value), None)
        if slot is None:
            raise EngineError("Persistent synth voice capacity exceeded")
        active[slot] = True
        self.voices[action.voice_id] = slot
        self.voice_triggers[action.voice_id] = VoiceAddress(
            part=action.part, trigger_id=action.trigger_id
        )
        phase = (
            float(
                (Fraction(action.tick) * Fraction(action.pitch_hz))
                % self.definition.sample_rate
            )
            if self.template.synchronize_oscillator
            else 0.0
        )
        return (
            slot,
            action.pitch_hz,
            self.template.oscillator.gain(action.key),
            phase,
        )

    def _encode_action(
        self,
        row: int,
        offset: int,
        kind: int,
        slot: int,
        value_a: float,
        value_b: float,
        duration_or_phase: float,
    ) -> None:
        if row >= self.action_capacity:
            raise EngineError("Persistent synth action capacity exceeded")
        self._action_buffer[row, 0] = offset
        self._action_buffer[row, 1] = kind
        self._action_buffer[row, 2] = slot
        self._action_buffer[row, 3] = value_a
        self._action_buffer[row, 4] = value_b
        self._action_buffer[row, 5] = duration_or_phase


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


def processing_parameters(
    settings: SoundSettings,
    values: dict[tuple[str, str], np.ndarray],
    rate: int,
    start: int,
    frames: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Extract shared processing arrays from resolved instrument targets."""
    tuning = values.get(
        ("processing", "tuning_cents"),
        np.full(frames, settings.processing.tuning_cents, dtype=np.float64),
    )
    gains = values.get(("processing", "amplitude"), np.ones(frames))
    filter_values = np.tile(
        np.array([[f.cutoff_hz, f.q] for f in settings.processing.filters]).reshape(
            -1, 2
        ),
        (frames, 1, 1),
    )
    for j, definition in enumerate(settings.processing.filters):
        for k, parameter in enumerate(("cutoff_hz", "q")):
            if (key := (f"filter-{definition.name}", parameter)) in values:
                filter_values[:, j, k] = values[key]
    return (
        tuning,
        gains,
        filters.parameters(
            settings.processing.filters,
            rate,
            frames,
            filter_values,
            start,
        ),
    )


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
        output, states, _ = native.render(
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


def validate_generators(
    settings: SoundSettings, allow_beat_motions: bool = False
) -> None:
    """Validate the Motion clocks supported by the selected renderer."""
    if any(g.score is not None for g in settings.motions.values()):
        raise EngineError("Library Motion uses must be materialized before rendering")
    if any(
        g.scope != "voice"
        for g in settings.motions.values()
        if isinstance(g.body, (Contour, Stages))
    ):
        raise EngineError("Staged and contour Motions require voice scope")
    if any(
        g.clock != "seconds"
        and not (
            allow_beat_motions and isinstance(g.body, (Contour, Cycle, Stages, Patch))
        )
        for g in settings.motions.values()
    ):
        raise EngineError("Beat-clock Motions require a supported synth renderer")
    if any(
        not isinstance(b, ControlBinding)
        and not isinstance(b, processing.GeneratorBinding)
        for b in settings.bindings
    ):
        raise EngineError("Only control and motion bindings are implemented")


def validate_modulation(settings: SoundSettings) -> None:
    """Accept only targets realized by the common voice processing path."""
    targets = {f"filter-{f.name}" for f in settings.processing.filters}
    if any(
        not (
            (
                p.target.name == "processing"
                and p.target.parameter in ("amplitude", "tuning_cents")
            )
            or (p.target.name in targets and p.target.parameter in ("cutoff_hz", "q"))
        )
        for p in settings.modulation.parameters
    ):
        raise EngineError(
            "Only amplitude, tuning, and filter modulation are implemented"
        )


def _sample_rate(timebase: Timebase) -> int:
    if timebase.rate.denominator != 1:
        raise EngineError(
            "Synth output rate must be an integer number of frames per second"
        )
    return timebase.rate.numerator


def _validate_voice(voice: SynthVoice) -> None:
    validate_envelope(voice.envelope)
    if voice.processing != Processing(
        tuning_cents=voice.processing.tuning_cents, filters=voice.processing.filters
    ):
        raise EngineError("Only tuning and filter processing are implemented")
    validate_generators(voice, allow_beat_motions=True)
    if any(c.mode == "fade" for c in voice.chokes):
        raise EngineError("Fade retirement is not implemented")
    validate_modulation(voice)


def _envelope_values(
    initial: float,
    segments: list[Segment],
    elapsed: Fraction,
    frames: int,
    sample_rate: int,
) -> np.ndarray:
    values = np.full(frames, segments[-1].to)
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
            values[first:last] = entry + (segment.to - entry) * progress
        boundary, entry = end, segment.to
    return values


def _duration(segments: list[Segment]) -> Fraction:
    return sum((s.duration for s in segments), Fraction(0))


def _runtime_segments(segments: list[Segment], sample_rate: int) -> np.ndarray:
    return np.asarray(
        [[float(s.duration * sample_rate), s.to] for s in segments],
        dtype=np.float64,
    )


def _persistent_modulation(
    definition: PreparedSynth,
    template: SoundSettings,
    source_parameters: list[
        tuple[tuple[str, str], int, modulation.Operation, float, float, float]
    ]
    | None = None,
) -> tuple[
    np.ndarray,
    list[tuple[int, int]],
    list[float],
    dict[str, list[int]],
    list[str],
    list[str],
    np.ndarray,
    list[tuple[int, int]],
    dict[str, list[int]],
    dict[str, list[int]],
    list[float],
    list[list[tuple[float, float]]],
    list[list[tuple[float, float]]],
    list[tuple[int, int, float, float, bool, int, float, float, int | None, bool]],
    dict[str, list[int]],
    set[str],
    np.ndarray,
    list[StagedRuntimeDefinition],
    dict[str, list[int]],
    list[tuple[int, str, int, str]],
]:
    bindings = {b.name: b for b in template.bindings}
    if any(
        not isinstance(b, ControlBinding)
        and not isinstance(b, processing.GeneratorBinding)
        for b in bindings.values()
    ):
        raise EngineError("Persistent synth supports control and motion bindings")
    sources = {s.name: s for s in template.modulation.sources}
    if bindings.keys() != sources.keys():
        raise EngineError("Persistent synth modulation sources require bindings")
    parameters = {
        (p.target.name, p.target.parameter): p for p in template.modulation.parameters
    }
    source_parameters = source_parameters or [
        (
            ("processing", "amplitude"),
            0,
            modulation.Operation.multiply,
            1,
            -1e300,
            1e300,
        ),
        (
            ("processing", "tuning_cents"),
            1,
            modulation.Operation.add,
            template.processing.tuning_cents,
            -120000,
            120000,
        ),
    ]
    supported = {
        target: (index, operation)
        for target, index, operation, _, _, _ in source_parameters
    }
    parameter_values = [0.0] * (3 * len(source_parameters))
    for _, index, _, default, minimum, maximum in source_parameters:
        parameter_values[index * 3 : index * 3 + 3] = [default, minimum, maximum]
    runtime_filters: list[list[float]] = []
    for filter_definition in template.processing.filters:
        cutoff_maximum = definition.sample_rate / 2 * filter_definition.nyquist_ratio
        cutoff_default = min(
            max(filter_definition.cutoff_hz, filter_definition.minimum_hz),
            cutoff_maximum,
        )
        cutoff_index = len(parameter_values) // 3
        q_index = cutoff_index + 1
        supported[f"filter-{filter_definition.name}", "cutoff_hz"] = (
            cutoff_index,
            modulation.Operation.add,
        )
        supported[f"filter-{filter_definition.name}", "q"] = (
            q_index,
            modulation.Operation.add,
        )
        for target, default, minimum, maximum in (
            (
                (f"filter-{filter_definition.name}", "cutoff_hz"),
                cutoff_default,
                filter_definition.minimum_hz,
                cutoff_maximum,
            ),
            (
                (f"filter-{filter_definition.name}", "q"),
                filter_definition.q,
                np.finfo(np.float64).tiny,
                1e300,
            ),
        ):
            parameter = parameters.get(target)
            parameter_values.extend(
                [
                    default if parameter is None else parameter.default,
                    minimum if parameter is None else parameter.minimum,
                    maximum if parameter is None else parameter.maximum,
                ]
            )
        runtime_filters.append(
            [
                list(processing.FilterResponse).index(filter_definition.response),
                filter_definition.stages,
                list(processing.FilterBoundary).index(filter_definition.boundary),
                filter_definition.minimum_hz,
                cutoff_maximum,
                cutoff_index,
                q_index,
            ]
        )
    if parameters.keys() - supported.keys():
        raise EngineError("Persistent runtime has an unsupported modulation target")
    routes = sorted(template.modulation.routes, key=lambda r: (r.source, r.name))
    event_sources = {
        name
        for connection in template.event_connections
        for name in (connection.source, connection.destination)
    }
    patch_children = {
        f"patch-{parent_name}-{child}"
        for parent_name, parent in template.motions.items()
        if isinstance(parent.body, Patch)
        for child in (
            {
                name
                for connection in parent.body.events
                for name in (connection.source.partition(".")[0], connection.target)
            }
            | {
                source.partition(".")[0]
                for source in parent.body.event_outputs.values()
            }
        )
    }
    event_only_patch_outputs = {
        f"patch-{parent_name}-{child}"
        for parent_name, parent in template.motions.items()
        if isinstance(parent.body, Patch) and parent.body.event_outputs
        for child in parent.body.outputs.values()
    }
    event_sources.update(
        binding.name
        for binding in template.bindings
        if isinstance(binding, processing.GeneratorBinding)
        and binding.reference in patch_children | event_only_patch_outputs
    )
    routed_sources = {route.source for route in routes}
    if (
        len(routed_sources) != len(routes)
        or len({r.target for r in routes}) != len(routes)
        or sources.keys() - routed_sources != event_sources - routed_sources
    ):
        raise EngineError("Persistent synth requires one modulation route per target")
    rows: list[list[float]] = []
    smoothing: list[tuple[int, int]] = []
    control_sources: dict[str, list[int]] = {}
    control_scopes: list[str] = []
    source_controls: list[str] = []
    lfo_rows: list[list[float]] = []
    lfo_rationals: list[tuple[int, int]] = []
    lfo_sources: dict[str, list[int]] = {}
    voice_lfo_sources: dict[str, list[int]] = {}
    envelope_initials: list[float] = []
    envelope_attacks: list[list[tuple[float, float]]] = []
    envelope_releases: list[list[tuple[float, float]]] = []
    envelope_parameters: list[
        tuple[int, int, float, float, bool, int, float, float, int | None, bool]
    ] = []
    named_motion_sources: dict[str, list[int]] = {}
    staged_motions: list[StagedRuntimeDefinition] = []
    staged_bindings: dict[str, int] = {}
    staged_motion_sources: dict[str, list[int]] = {}
    source_scopes: set[str] = set()
    for route in routes:
        source = sources.get(route.source)
        binding = bindings.get(route.source)
        target = (route.target.name, route.target.parameter)
        if source is None or binding is None or target not in supported:
            raise EngineError("Persistent synth modulation route is unresolved")
        parameter, _ = supported[target]
        if (
            route.interpolation != modulation.Interpolation.linear
            or len(route.points) != 2
            or route.points[0].input != source.minimum
            or route.points[1].input != source.maximum
        ):
            raise EngineError(
                "Persistent synth modulation routes must be direct linear maps"
            )
        operation = route.operation
        source_scopes.add(source.scope)
        declaration = (
            definition.instrument.controls[binding.control]
            if isinstance(binding, ControlBinding)
            else None
        )
        input_minimum = (
            (-1 if declaration is not None and declaration.polarity == "bipolar" else 0)
            if declaration is not None
            else source.minimum
        )
        slope = (route.points[1].amount - route.points[0].amount) / (
            source.maximum - source.minimum
        )
        intercept = route.points[0].amount - slope * source.minimum
        target_parameter = parameters[target]
        mapped = [
            intercept + slope * input_minimum,
            intercept + slope,
        ]
        outcomes = (
            [target_parameter.default + v for v in mapped]
            if operation == modulation.Operation.add
            else [target_parameter.default * v for v in mapped]
        )
        if any(
            v < target_parameter.minimum or v > target_parameter.maximum
            for v in outcomes
        ):
            raise EngineError(
                "Persistent synth modulation route exceeds its target domain"
            )
        if isinstance(binding, ControlBinding):
            assert declaration is not None
            control_minimum = -1 if declaration.polarity == "bipolar" else 0
            if source.minimum > control_minimum or source.maximum < 1:
                raise EngineError(
                    "Persistent synth source must cover its control declaration"
                )
            index = len(rows)
            rows.append(
                [
                    declaration.default,
                    ["instrument", "part", "trigger"].index(source.scope),
                    parameter,
                    0 if operation == modulation.Operation.add else 1,
                    source.minimum,
                    source.maximum,
                    intercept,
                    slope,
                ]
            )
            duration = binding.smoothing * definition.sample_rate
            smoothing.append((duration.numerator, duration.denominator))
            control_sources.setdefault(binding.control, []).append(index)
            control_scopes.append(source.scope)
            source_controls.append(binding.control)
        else:
            assert isinstance(binding, processing.GeneratorBinding)
            motion = template.motions[binding.reference]
            if isinstance(motion.body, Contour):
                generator = motion.body
                assert isinstance(generator.initial, float)
                if source.scope != "voice" or any(
                    segment.curve != 0
                    for segment in [*generator.segments, *generator.release]
                ):
                    raise EngineError(
                        "Persistent named envelopes require linear voice segments"
                    )
                named_motion_sources.setdefault(binding.reference, []).append(
                    len(envelope_initials)
                )
                envelope_initials.append(generator.initial)
                duration_scale = (
                    1 if motion.clock == "beats" else definition.sample_rate
                )
                envelope_attacks.append(
                    [
                        (float(segment.duration * duration_scale), segment.to)
                        for segment in generator.segments
                    ]
                )
                envelope_releases.append(
                    [
                        (float(segment.duration * duration_scale), segment.to)
                        for segment in generator.release
                    ]
                )
                envelope_parameters.append(
                    (
                        parameter,
                        0 if operation == modulation.Operation.add else 1,
                        intercept,
                        slope,
                        binding.release_timing == ReleaseTiming.voice,
                        [
                            PlaybackMode.once,
                            PlaybackMode.loop,
                            PlaybackMode.ping_pong,
                        ].index(generator.playback),
                        float(
                            next(
                                m.position
                                for m in generator.markers
                                if m.name == generator.loop_start
                            )
                        )
                        if generator.loop_start is not None
                        else 0.0,
                        float(
                            next(
                                m.position
                                for m in generator.markers
                                if m.name == generator.loop_end
                            )
                        )
                        if generator.loop_end is not None
                        else 1.0,
                        generator.repeat_count,
                        generator.start == "event",
                    )
                )
                continue
            if isinstance(motion.body, Stages):
                staged_bindings[binding.name] = len(staged_motions)
                staged_motion_sources.setdefault(binding.reference, []).append(
                    len(staged_motions)
                )
                staged_motions.append(
                    _runtime_staged_motion(
                        motion.body,
                        definition.sample_rate,
                        parameter,
                        0 if operation == modulation.Operation.add else 1,
                        intercept,
                        slope,
                        motion.clock == "beats",
                    )
                )
                continue
            assert isinstance(motion.body, Cycle)
            intercept += slope * motion.body.center
            slope *= motion.body.depth
            generator = cycle_lfo(motion)
            index = len(lfo_rows)
            lfo_rows.append(
                [
                    ["instrument", "part", "trigger", "voice"].index(source.scope),
                    [Waveform.sine, Waveform.square, Waveform.triangle].index(
                        generator.waveform
                    ),
                    parameter,
                    0 if operation == modulation.Operation.add else 1,
                    intercept,
                    slope,
                    2.0
                    if motion.position_driver == "transport"
                    else 1.0
                    if motion.clock == "beats"
                    else 0.0,
                ]
            )
            for value in (
                generator.duty_cycle,
                generator.rate,
                generator.phase,
                generator.delay
                * (1 if motion.clock == "beats" else definition.sample_rate),
                generator.fade_in
                * (1 if motion.clock == "beats" else definition.sample_rate),
            ):
                lfo_rationals.append((value.numerator, value.denominator))
            if source.scope == "instrument":
                lfo_sources.setdefault(binding.reference, []).append(index)
            elif source.scope == "voice":
                voice_lfo_sources.setdefault(binding.reference, []).append(index)
    for binding in template.bindings:
        if binding.name not in event_sources or binding.name in routed_sources:
            continue
        assert isinstance(binding, processing.GeneratorBinding)
        motion = template.motions[binding.reference]
        if isinstance(motion.body, Cycle):
            generator = cycle_lfo(motion)
            voice_lfo_sources.setdefault(binding.reference, []).append(len(lfo_rows))
            lfo_rows.append(
                [
                    3,
                    [Waveform.sine, Waveform.square, Waveform.triangle].index(
                        generator.waveform
                    ),
                    0,
                    1,
                    1,
                    0,
                    0,
                ]
            )
            for value in (
                generator.duty_cycle,
                generator.rate,
                generator.phase,
                generator.delay * definition.sample_rate,
                generator.fade_in * definition.sample_rate,
            ):
                lfo_rationals.append((value.numerator, value.denominator))
            continue
        if isinstance(motion.body, Contour):
            generator = motion.body
            assert isinstance(generator.initial, float)
            named_motion_sources.setdefault(binding.reference, []).append(
                len(envelope_initials)
            )
            envelope_initials.append(generator.initial)
            envelope_attacks.append(
                [
                    (float(segment.duration * definition.sample_rate), segment.to)
                    for segment in generator.segments
                ]
            )
            envelope_releases.append([])
            envelope_parameters.append((0, 1, 1, 0, False, 0, 0, 1, None, True))
            continue
        assert isinstance(motion.body, Stages)
        source_scopes.add(motion.scope)
        staged_bindings[binding.name] = len(staged_motions)
        staged_motion_sources.setdefault(binding.reference, []).append(
            len(staged_motions)
        )
        staged_motions.append(
            _runtime_staged_motion(
                motion.body,
                definition.sample_rate,
                None,
                0,
                0,
                0,
                motion.clock == "beats",
            )
        )
    event_connections = [
        (
            staged_bindings[c.source],
            c.port,
            staged_bindings[c.destination],
            c.cue,
        )
        for c in template.event_connections
        if isinstance((binding := bindings[c.source]), processing.GeneratorBinding)
        if isinstance(template.motions[binding.reference].body, Stages)
    ]
    for target, index, _, _, _, _ in source_parameters:
        if (parameter := parameters.get(target)) is not None:
            parameter_values[index * 3 : index * 3 + 3] = [
                parameter.default,
                parameter.minimum,
                parameter.maximum,
            ]
    return (
        np.asarray(rows, dtype=np.float64).reshape(-1, 8),
        smoothing,
        parameter_values,
        control_sources,
        control_scopes,
        source_controls,
        np.asarray(lfo_rows, dtype=np.float64).reshape(-1, 7),
        lfo_rationals,
        lfo_sources,
        voice_lfo_sources,
        envelope_initials,
        envelope_attacks,
        envelope_releases,
        envelope_parameters,
        named_motion_sources,
        source_scopes,
        np.asarray(runtime_filters, dtype=np.float64).reshape(-1, 7),
        staged_motions,
        staged_motion_sources,
        event_connections,
    )


def _motion_state_owners(
    count: int, names: set[str], *source_maps: dict[str, list[int]]
) -> list[int]:
    owners = list(range(count))
    for source_map in source_maps:
        for name in names:
            if indices := source_map.get(name):
                for index in indices[1:]:
                    owners[index] = indices[0]
    return owners


def _runtime_staged_motion(
    body: Stages,
    sample_rate: int,
    parameter: int | None,
    operation: int,
    intercept: float,
    slope: float,
    beat_clock: bool,
) -> StagedRuntimeDefinition:
    names = {stage.name: index for index, stage in enumerate(body.stages)}
    stages: list[
        tuple[
            int,
            float,
            bool,
            list[tuple[float, float]],
            list[float],
            list[tuple[float, str]],
            int,
            float,
            float,
            int | None,
        ]
    ] = []
    for stage in body.stages:
        motion = stage.motion
        if isinstance(motion, Hold):
            stages.append((0, motion.value, False, [], [], [], 0, 0.0, 1.0, None))
        elif isinstance(motion, Contour):
            if any(s.curve != 0 for s in motion.segments):
                raise EngineError("Persistent staged contours require linear segments")
            stages.append(
                (
                    1,
                    0.0 if motion.initial == "current" else motion.initial,
                    motion.initial == "current",
                    [
                        (float(s.duration * (1 if beat_clock else sample_rate)), s.to)
                        for s in motion.segments
                    ],
                    [],
                    [(float(m.position), m.name) for m in motion.markers],
                    [
                        PlaybackMode.once,
                        PlaybackMode.loop,
                        PlaybackMode.ping_pong,
                    ].index(motion.playback),
                    float(
                        next(
                            m.position
                            for m in motion.markers
                            if m.name == motion.loop_start
                        )
                    )
                    if motion.loop_start is not None
                    else 0.0,
                    float(
                        next(
                            m.position
                            for m in motion.markers
                            if m.name == motion.loop_end
                        )
                    )
                    if motion.loop_end is not None
                    else 1.0,
                    motion.repeat_count,
                )
            )
        else:
            assert isinstance(motion, Cycle)
            assert isinstance(motion.rate, Fraction)
            stages.append(
                (
                    2,
                    0.0,
                    False,
                    [],
                    [
                        float(
                            [Waveform.sine, Waveform.square, Waveform.triangle].index(
                                motion.shape
                            )
                        ),
                        float(motion.rate / (1 if beat_clock else sample_rate)),
                        float(motion.phase),
                        float(motion.duty_cycle),
                        float(motion.delay * (1 if beat_clock else sample_rate)),
                        float(motion.fade_in * (1 if beat_clock else sample_rate)),
                        motion.center,
                        motion.depth,
                    ],
                    [(float(m.position), m.name) for m in motion.markers],
                    0,
                    0.0,
                    1.0,
                    None,
                )
            )
    transitions: list[tuple[int, str, int]] = []
    for transition in body.transitions:
        target = (
            names[transition.action.stage]
            if isinstance(transition.action, EnterStage)
            else len(stages)
        )
        for source in transition.from_stages:
            transitions.append((names[source], transition.event, target))
    return (
        stages,
        transitions,
        names[body.initial_stage],
        parameter,
        operation,
        intercept,
        slope,
        beat_clock,
    )


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
