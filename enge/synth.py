"""Offline rendering for Ufor's oscillator synth profile."""

from collections.abc import Callable, Collection, Iterable
from fractions import Fraction
from functools import cached_property
from math import ceil, isfinite
from typing import Literal, Self, cast

import numpy as np
from envelfo import patch
from envelfo.runtime import (
    envelfo_at,
    envelfo_event,
    envelope_complete_at,
    initial_envelfo,
)
from pydantic import BaseModel, ConfigDict, Field, model_validator
from ufor import envelfo_random, instrument_trace, modulation
from ufor.base import Model
from ufor.control import TempoMap
from ufor.envelfo import (
    LFO,
    Affine,
    EnterStage,
    EnveLFO,
    EnvelfoEvent,
    EnveLFOReference,
    EnvelfoState,
    Envelope,
    EnvelopeState,
    Latch,
    LFOState,
    Patch,
    PlaybackMode,
    Product,
    Quantize,
    SampleHold,
    SampleHoldState,
    Slew,
    Stages,
    StageState,
    Sum,
    Threshold,
)
from ufor.oscillator import Oscillator, Waveform
from ufor.samples import controls, processing
from ufor.samples.processing import (
    ControlBinding,
    Processing,
    ReleaseTiming,
    SoundSettings,
)
from ufor.segments import IndefiniteSegment, Segment
from ufor.streams import AudioType
from ufor.synth import (
    FMVoice,
    SynthInstrument,
    SynthInstrumentScore,
    SynthVoice,
    frequency,
)
from ufor.synth_trace import VoiceStart
from ufor.time import Timebase

from . import _native, control, envelfo_graph, filters, native
from .lfo import lfo_samples


class EngineError(ValueError):
    """The requested input is invalid or outside Enge's implemented profile."""


StagedRuntimeStage = tuple[
    int,
    float,
    bool,
    list[tuple[tuple[int, int] | None, float]],
    tuple[int, list[tuple[int, int]], float, float] | None,
    list[tuple[tuple[int, int], str]],
    int,
    tuple[int, int],
    tuple[int, int],
    int | None,
]
StagedRuntimeDefinition = tuple[
    list[StagedRuntimeStage],
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
    scope: Literal["instrument", "part", "trigger", "voice"]
    part: str | None
    trigger_id: str | None
    voice_id: str | None = None
    ramps: list[ControlRamp]
    velocity: float | None = Field(default=None, strict=True, ge=0, le=1)


class LFOSource(Model, frozen=True):
    setting: int
    name: str
    part: str | None
    trigger_id: str | None
    voice_id: str | None
    state: EnvelfoState | None = None


class EnvelopeSource(Model, frozen=True):
    setting: int
    name: str
    binding_name: str
    part: str
    trigger_id: str | None
    voice_id: str
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


class PatchState(Model, frozen=True):
    voice_id: str
    name: str


class PreparedVoiceEnvelope(Model, frozen=True):
    """The shared amplitude lifetime of an oscillator or sample voice."""

    sample_rate: int = Field(gt=0)
    envelope: Envelope
    filter_order: processing.FilterOrder = processing.FilterOrder.before_amplitude
    minimum_hold_seconds: Fraction = Field(default=Fraction(0), ge=0)

    @model_validator(mode="after")
    def envelope_profile(self) -> Self:
        validate_envelope(self.envelope)
        return self

    @cached_property
    def minimum_hold_frames(self) -> Fraction:
        return Fraction(str(self.minimum_hold_seconds)) * self.sample_rate

    @property
    def amplitude_definitions(self) -> dict[str, Envelope]:
        return {"": self.envelope}


class VoiceEnvelopeRenderer(BaseModel):
    """Shared release timing; source state remains with the concrete renderer."""

    definition: PreparedVoiceEnvelope
    frame_count: int = 0
    release_frame: Fraction | None = None
    amplitude_states: dict[str, EnvelfoState] = Field(default_factory=dict)
    lifetime_op: str = ""

    @model_validator(mode="after")
    def initialize_amplitudes(self) -> Self:
        if not self.amplitude_states:
            self.amplitude_states = {
                n: envelfo_event(
                    e,
                    initial_envelfo(e, Fraction(0)),
                    EnvelfoEvent(at=Fraction(0), ordinal=0, action="note_on"),
                )
                for n, e in self.definition.amplitude_definitions.items()
            }
        return self

    @property
    def complete(self) -> bool:
        self.settle_amplitudes(Fraction(self.frame_count))
        state = self.amplitude_states[self.lifetime_op]
        assert isinstance(state.runtime, EnvelopeState)
        return (
            state.runtime.phase == "release"
            and envelfo_at(
                self.definition.envelope,
                state,
                Fraction(self.frame_count, self.definition.sample_rate),
            ).status
            == "complete"
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
        state = self.amplitude_states[self.lifetime_op]
        assert isinstance(state.runtime, EnvelopeState)
        if state.runtime.phase == "on":
            state = envelfo_event(
                self.definition.envelope,
                state,
                EnvelfoEvent(
                    at=self.release_frame / self.definition.sample_rate,
                    ordinal=state.runtime.ordinal + 1,
                    action="note_off",
                ),
            )
        end = envelope_complete_at(self.definition.envelope, state)
        return (
            frames
            if end is None
            else max(
                0,
                min(frames, ceil(end * self.definition.sample_rate) - self.frame_count),
            )
        )

    def settle_amplitudes(self, frame: Fraction, op: str | None = None) -> None:
        if self.release_frame is None or frame < self.release_frame:
            return
        for name, body in self.definition.amplitude_definitions.items():
            if op is not None and name != op:
                continue
            state = self.amplitude_states[name]
            assert isinstance(state.runtime, EnvelopeState)
            if state.runtime.phase == "on":
                self.amplitude_states[name] = envelfo_event(
                    body,
                    state,
                    EnvelfoEvent(
                        at=self.release_frame / self.definition.sample_rate,
                        ordinal=state.runtime.ordinal + 1,
                        action="note_off",
                    ),
                )

    def amplitude_event(self, action: instrument_trace.EnvelfoObservation) -> None:
        name = self.lifetime_op if action.op is None else action.op
        if name not in self.amplitude_states:
            raise EngineError(f"Unknown amplitude operator: {name}")
        self.settle_amplitudes(Fraction(self.frame_count))
        state = self.amplitude_states[name]
        assert isinstance(state.runtime, EnvelopeState)
        self.amplitude_states[name] = envelfo_event(
            self.definition.amplitude_definitions[name],
            state,
            EnvelfoEvent(
                at=Fraction(self.frame_count, self.definition.sample_rate),
                ordinal=state.runtime.ordinal + 1,
                action=action.action,
                position=None
                if action.position is None
                else Fraction(str(action.position)),
                offset=None if action.offset is None else Fraction(str(action.offset)),
            ),
        )

    def amplitude_spans(self, frames: int, op: str | None = None) -> np.ndarray:
        name = self.lifetime_op if op is None else op
        body = self.definition.amplitude_definitions[name]
        rate = self.definition.sample_rate
        self.settle_amplitudes(Fraction(self.frame_count))
        boundary = (
            frames
            if self.release_frame is None
            else max(0, min(frames, ceil(self.release_frame) - self.frame_count))
        )
        spans = native.envelope_spans(
            body, self.amplitude_states[name], self.frame_count, boundary, rate
        )
        if boundary < frames:
            self.settle_amplitudes(Fraction(self.frame_count + boundary), name)
            tail = native.envelope_spans(
                body,
                self.amplitude_states[name],
                self.frame_count + boundary,
                frames - boundary,
                rate,
            )
            tail[:, :2] += boundary
            tail[:, 6] += boundary
            spans = np.concatenate((spans, tail))
        return spans

    def amplitude_samples(self, frames: int, op: str | None = None) -> np.ndarray:
        values = np.zeros(frames)
        for (
            first,
            last,
            entry,
            delta,
            origin,
            denominator,
            offset,
        ) in self.amplitude_spans(frames, op):
            a, b = int(first), int(last)
            values[a:b] = entry + delta * (
                origin + (np.arange(a, b) - offset) / denominator
            )
        return values

    model_config = ConfigDict(
        extra="forbid", allow_inf_nan=False, validate_default=True
    )


class PreparedVoice(PreparedVoiceEnvelope, frozen=True):
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


class VoiceRenderer(VoiceEnvelopeRenderer):
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
                self.amplitude_spans(count),
                definition.routes,
                frames,
                filters.native_inputs(definition.filters, self.filter_states, values),
                definition.filter_order == processing.FilterOrder.after_amplitude,
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
            amplitude = self.amplitude_samples(count) * definition.gain
            if gains is not None:
                amplitude *= gains[:count]
            samples = waves[0][:, None] if len(waves) == 1 else np.column_stack(waves)
            if definition.filter_order == processing.FilterOrder.after_amplitude:
                samples *= amplitude[:, None]
            if definition.filters:
                samples, self.filter_states = filters.filter_samples(
                    definition.filters,
                    self.filter_states,
                    samples,
                    values,
                    definition.sample_rate,
                )
            output = route_samples(samples, definition.routes)
            if definition.filter_order == processing.FilterOrder.before_amplitude:
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
    part: str
    trigger_id: str | None
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
    patches: list[PatchState] = Field(default_factory=list)
    graphs: dict[str, patch.PatchState] = Field(default_factory=dict)
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


def apply_amplitude_event(
    action: instrument_trace.TraceAction,
    voices: Iterable[tuple[VoiceEnvelopeRenderer, str, str | None]],
) -> bool:
    if (
        not isinstance(action, instrument_trace.EnvelfoObservation)
        or action.target != "envelope"
    ):
        return False
    for renderer, part, trigger in voices:
        if (part, trigger) == (action.part, action.trigger_id):
            renderer.amplitude_event(action)
    return True


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
    envelfos = dict(voice.envelfos)
    bindings = list(voice.bindings)
    sources = list(voice.modulation.sources)
    changed = False
    for index, binding in enumerate(bindings):
        if not isinstance(binding, processing.GeneratorBinding):
            continue
        parent = envelfos[binding.reference]
        if not isinstance(parent, Patch):
            continue
        assert binding.output is not None
        child_name = parent.outputs[binding.output]
        name = f"patch-{binding.reference}-{child_name}"
        if name in voice.envelfos:
            raise EngineError(f"Patch output Envelfo name collides: {name}")
        child = parent.envelfos[child_name]
        if isinstance(
            child, (Sum, Product, Affine, Quantize, Latch, Threshold, Slew)
        ) or any(
            isinstance(b, (Latch, Threshold, Slew)) for b in parent.envelfos.values()
        ):
            continue
        if name not in envelfos:
            envelfos[name] = type(child).model_validate(
                child.model_dump() | {"scope": parent.scope, "clock": parent.clock}
            )
        bindings[index] = binding.model_copy(update={"reference": name, "output": None})
        changed = True
    for parent_name, parent in voice.envelfos.items():
        if not isinstance(parent, Patch):
            continue
        if not any(
            isinstance(binding, processing.GeneratorBinding)
            and binding.reference == parent_name
            for binding in voice.bindings
        ):
            continue
        children = {
            name
            for connection in parent.events
            for name in (connection.source.partition(".")[0], connection.target)
        } | {source.partition(".")[0] for source in parent.event_outputs.values()}
        pending = [
            parent.outputs[b.output]
            for b in voice.bindings
            if isinstance(b, processing.GeneratorBinding)
            and b.reference == parent_name
            and b.output is not None
        ]
        pending.extend(children)
        children = set()
        visited: set[str] = set()
        while pending:
            child_name = pending.pop()
            if child_name in visited:
                continue
            visited.add(child_name)
            child = parent.envelfos[child_name]
            if isinstance(child, (Sum, Product)):
                pending.extend(child.inputs)
            elif isinstance(child, (Affine, Quantize, Latch, Threshold, Slew)):
                pending.append(child.input)
            else:
                children.add(child_name)
        for child_name in children:
            name = f"patch-{parent_name}-{child_name}"
            if name in voice.envelfos:
                raise EngineError(f"Patch child Envelfo name collides: {name}")
            child = parent.envelfos[child_name]
            envelfos.setdefault(
                name,
                type(cast(LFO | Envelope | Stages | SampleHold, child)).model_validate(
                    cast(LFO | Envelope | Stages | SampleHold, child).model_dump()
                    | {"scope": parent.scope, "clock": parent.clock}
                ),
            )
            if any(
                isinstance(binding, processing.GeneratorBinding)
                and binding.reference == name
                for binding in bindings
            ):
                continue
            if any(binding.name == name for binding in bindings):
                raise EngineError(f"Patch child binding name collides: {name}")
            bindings.append(
                processing.GeneratorBinding(
                    name=name,
                    reference=name,
                    release_timing=next(
                        (
                            b.release_timing
                            for b in voice.bindings
                            if isinstance(b, processing.GeneratorBinding)
                            and b.reference == parent_name
                            and (b.output is not None)
                            and (parent.outputs[b.output] == child_name)
                        ),
                        ReleaseTiming.event,
                    ),
                )
            )
            sources.append(
                modulation.Source(
                    name=name,
                    scope=parent.scope.value,
                    minimum=0
                    if isinstance(child, Envelope) and child.polarity == "unipolar"
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
        parent = voice.envelfos[source_binding.reference]
        if not isinstance(parent, Patch):
            continue
        child_name, _, port = parent.event_outputs[connection.port].partition(".")
        if isinstance(parent.envelfos[child_name], Threshold):
            continue
        child_envelfo = f"patch-{source_binding.reference}-{child_name}"
        child_binding = next(
            binding.name
            for binding in bindings
            if isinstance(binding, processing.GeneratorBinding)
            and binding.reference == child_envelfo
        )
        event_connections[index] = connection.model_copy(
            update={"source": child_binding, "port": port}
        )
        changed = True
    for parent_name, parent in voice.envelfos.items():
        if not isinstance(parent, Patch) or not any(
            isinstance(binding, processing.GeneratorBinding)
            and binding.reference == parent_name
            for binding in voice.bindings
        ):
            continue
        for connection in parent.events:
            if connection.action != "cue":
                continue
            child_name, _, port = connection.source.partition(".")
            if isinstance(parent.envelfos[child_name], Threshold):
                continue
            source_envelfo = f"patch-{parent_name}-{child_name}"
            target_envelfo = f"patch-{parent_name}-{connection.target}"
            source_binding = next(
                binding.name
                for binding in bindings
                if isinstance(binding, processing.GeneratorBinding)
                and binding.reference == source_envelfo
            )
            target_binding = next(
                binding.name
                for binding in bindings
                if isinstance(binding, processing.GeneratorBinding)
                and binding.reference == target_envelfo
            )
            assert connection.cue is not None
            event_connections.append(
                processing.EnvelfoEventConnection(
                    source=source_binding,
                    port=port,
                    destination=target_binding,
                    cue=connection.cue,
                    every=connection.every,
                    offset=connection.offset,
                    probability=connection.probability,
                    delay=connection.delay,
                )
            )
            changed = True
    if not changed:
        return voice
    return SynthVoice.model_validate(
        voice.model_dump()
        | {
            "envelfos": envelfos,
            "bindings": bindings,
            "event_connections": event_connections,
            "modulation": voice.modulation.model_copy(update={"sources": sources}),
        }
    )


def _patch_child_names(settings: SoundSettings) -> set[str]:
    return {
        f"patch-{name}-{child}"
        for name, envelfo in settings.envelfos.items()
        if isinstance(envelfo, Patch)
        for child in envelfo.envelfos
    }


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
        self.patches: list[PatchState] = []
        self.graphs: dict[str, patch.PatchState] = {}
        self.programs = [envelfo_graph.prepare_graph(s) for s in settings]
        self.new_context("instrument", None, None, 0, {})

    def envelfo_time(self, envelfo: EnveLFO, seconds: Fraction) -> Fraction:
        if envelfo.clock == "seconds":
            return seconds
        if self.tempo_map is None:
            raise EngineError("Beat-clock Envelfo requires a host tempo map")
        if envelfo.position_driver == "transport":
            return self.tempo_map.beat_at(seconds)
        return self.tempo_map.elapsed_beats(Fraction(0), seconds)

    def new_context(
        self,
        scope: Literal["instrument", "part", "trigger"],
        part: str | None,
        trigger_id: str | None,
        frame: int,
        values: dict[str, float],
        *,
        velocity: float | None = None,
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
            ControlContext(
                scope=scope,
                part=part,
                trigger_id=trigger_id,
                ramps=ramps,
                velocity=velocity,
            )
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
        if (
            isinstance(action, instrument_trace.EnvelfoObservation)
            and action.target == "envelope"
        ):
            return False
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
                definition = self.settings[source.setting].envelfos[source.name]
                assert source.state is not None
                self.lfos[index] = source.model_copy(
                    update={
                        "state": envelfo_event(
                            definition,
                            source.state,
                            EnvelfoEvent(
                                at=self.envelfo_time(
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
        elif isinstance(action, instrument_trace.EnvelfoObservation):
            assert action.name is not None
            if any(
                (
                    action.name in self.patch_children[index]
                    or (
                        (envelfo := settings.envelfos.get(action.name)) is not None
                        and isinstance(envelfo, Patch)
                    )
                    for index, settings in enumerate(self.settings)
                )
            ):
                raise EngineError("Patch-level Envelfo commands are not implemented")
            for source in self.lfos:
                if (
                    source.name == action.name
                    and source.voice_id in active_voice_ids
                    and (source.part, source.trigger_id)
                    == (action.part, action.trigger_id)
                ):
                    assert source.voice_id is not None
                    self._graph_event(
                        source.setting,
                        source.voice_id,
                        f"lfo-{source.name}",
                        self._envelfo_observation(action),
                    )
            for source in self.envelopes:
                if (
                    source.name == action.name
                    and source.voice_id in active_voice_ids
                    and (source.part, source.trigger_id)
                    == (action.part, action.trigger_id)
                ):
                    node = self.programs[source.setting].outputs[source.binding_name]
                    self._graph_event(
                        source.setting,
                        source.voice_id,
                        node,
                        self._envelfo_observation(action),
                    )
        elif isinstance(action, instrument_trace.TriggerContext):
            if action.controls.keys() != self.declarations.keys():
                raise EngineError("Trigger context must contain all declared controls")
            for name, value in action.controls.items():
                self.declarations[name].validate_value(value)
            self.new_context(
                "trigger",
                action.part,
                action.trigger_id,
                action.tick,
                action.controls,
                velocity=action.velocity,
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

    def _envelfo_observation(
        self, action: instrument_trace.EnvelfoObservation
    ) -> EnvelfoEvent:
        return EnvelfoEvent(
            at=Fraction(action.tick, self.sample_rate),
            ordinal=action.ordinal,
            action=action.action,
            position=None
            if action.position is None
            else Fraction(str(action.position)),
            offset=None if action.offset is None else Fraction(str(action.offset)),
        )

    def _graph_event(
        self, setting: int, voice_id: str, node: str, event: EnvelfoEvent
    ) -> None:
        key = f"{setting}:{voice_id}"
        state = self.graphs[key]
        current = state.children[node].runtime
        event = event.model_copy(
            update={"ordinal": max(event.ordinal, current.ordinal + 1)}
        )
        self.graphs[key], _ = patch.patch_event(
            self.programs[setting],
            state,
            node,
            event,
            tempo_map=self.tempo_map,
            inclusive=False,
        )

    def sources(
        self,
        settings: SoundSettings,
        action: instrument_trace.VoiceStart,
        velocity: float | None = None,
    ) -> dict[str, int]:
        result: dict[str, int] = {}
        setting = next(i for i, s in enumerate(self.settings) if s is settings)
        bindings = {b.name: b for b in settings.bindings}
        for source in settings.modulation.sources:
            binding = bindings[source.name]
            if isinstance(binding, processing.EventBinding):
                if velocity is None and action.trigger_id is not None:
                    context_id = self.context("trigger", action.part, action.trigger_id)
                    if context_id is not None:
                        velocity = self.contexts[context_id].velocity
                if velocity is None:
                    raise EngineError(
                        "Velocity binding requires resolved onset velocity"
                    )
                index = len(self.contexts)
                self.contexts.append(
                    ControlContext(
                        scope="voice",
                        part=action.part,
                        trigger_id=action.trigger_id,
                        voice_id=action.voice_id,
                        ramps=[
                            ControlRamp(
                                control="velocity",
                                state=controls.initial_control(
                                    controls.ControlDeclaration(default=velocity),
                                    Fraction(action.tick, self.sample_rate),
                                    Fraction(0),
                                ),
                            )
                        ],
                    )
                )
                result[source.name] = index
                continue
            if isinstance(binding, processing.GeneratorBinding):
                body = settings.envelfos[binding.reference]
                if isinstance(body, Patch):
                    index = next(
                        (
                            i
                            for i, p in enumerate(self.patches)
                            if (p.voice_id, p.name)
                            == (action.voice_id, binding.reference)
                        ),
                        None,
                    )
                    if index is None:
                        index = len(self.patches)
                        self.patches.append(
                            PatchState(voice_id=action.voice_id, name=binding.reference)
                        )
                    result[source.name] = -1 - index
                elif isinstance(body, (Envelope, Stages, SampleHold)):
                    if body.scope != "voice":
                        raise EngineError(
                            "Staged and envelope Envelfos require voice scope"
                        )
                    index = next(
                        (
                            i
                            for i, e in enumerate(self.envelopes)
                            if binding.reference in self.patch_children[setting]
                            and (e.setting, e.name, e.voice_id)
                            == (setting, binding.reference, action.voice_id)
                        ),
                        None,
                    )
                    if index is None:
                        index = len(self.envelopes)
                        self.envelopes.append(
                            EnvelopeSource(
                                setting=setting,
                                name=binding.reference,
                                binding_name=binding.name,
                                part=action.part,
                                trigger_id=action.trigger_id,
                                voice_id=action.voice_id,
                            )
                        )
                    result[source.name] = index
                else:
                    assert isinstance(body, LFO)
                    part = action.part if body.scope in ("part", "voice") else None
                    voice_id = action.voice_id if body.scope == "voice" else None
                    index = next(
                        (
                            i
                            for i, e in enumerate(self.lfos)
                            if (e.setting, e.name, e.part, e.voice_id)
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
                                state=initial_envelfo(
                                    body, self.envelfo_time(body, Fraction(0))
                                )
                                if voice_id is None
                                else None,
                            )
                        )
                    result[source.name] = index
                continue
            part = None if source.scope == "instrument" else action.part
            trigger_id = action.trigger_id if source.scope == "trigger" else None
            context_id = self.context(source.scope, part, trigger_id)
            if context_id is None:
                if source.scope == "trigger" and trigger_id is not None:
                    raise EngineError(f"Missing trigger context: {part}/{trigger_id}")
                assert source.scope in ("instrument", "part", "trigger")
                context_id = self.new_context(source.scope, part, trigger_id, 0, {})
            result[source.name] = context_id
        program = self.programs[setting]
        if program.envelfos:
            at = Fraction(action.tick, self.sample_rate)
            state = patch.initial_patch(
                program,
                at,
                name="voice",
                seed=action.envelfo_key,
                tempo_map=self.tempo_map,
            )
            key = f"{setting}:{action.voice_id}"
            self.graphs[key] = state
            for name in state.children:
                if isinstance(program.envelfos[name], (Envelope, Stages)):
                    self._graph_event(
                        setting,
                        action.voice_id,
                        name,
                        EnvelfoEvent(at=at, ordinal=action.ordinal, action="note_on"),
                    )
        return result

    def release(
        self,
        settings: SoundSettings,
        sources: dict[str, int],
        action: instrument_trace.VoiceRetirement,
        renderer: VoiceEnvelopeRenderer,
    ) -> None:
        self.clear_cache()
        assert renderer.release_frame is not None
        voice_release_at = (
            Fraction(action.tick, self.sample_rate)
            + (renderer.release_frame - renderer.frame_count) / self.sample_rate
        )
        released: set[int] = set()
        for binding in settings.bindings:
            if not isinstance(binding, processing.GeneratorBinding) or not isinstance(
                settings.envelfos[binding.reference], (Envelope, Stages)
            ):
                continue
            index = sources[binding.name]
            if index in released:
                continue
            released.add(index)
            source = self.envelopes[index]
            at = (
                voice_release_at
                if isinstance(settings.envelfos[binding.reference], Envelope)
                and binding.release_timing == ReleaseTiming.voice
                else Fraction(action.tick, self.sample_rate)
            )
            if at > Fraction(action.tick, self.sample_rate):
                self.envelopes[index] = source.model_copy(
                    update={"pending_release": at}
                )
            else:
                node = self.programs[source.setting].outputs[binding.name]
                self._graph_event(
                    source.setting,
                    source.voice_id,
                    node,
                    EnvelfoEvent(at=at, ordinal=action.ordinal, action="note_off"),
                )

    def stop(self, voice_id: str) -> None:
        self.clear_cache()
        for setting in range(len(self.settings)):
            self.graphs.pop(f"{setting}:{voice_id}", None)
        self.envelopes = [
            s.model_copy(update={"pending_release": None})
            if s.voice_id == voice_id
            else s
            for s in self.envelopes
        ]

    def values(
        self, settings: SoundSettings, sources: dict[str, int], start: int, frames: int
    ) -> dict[tuple[str, str], np.ndarray]:
        """Resolve every declared target once, independently of its DSP consumer."""
        if frames == 0:
            return {
                (p.target.name, p.target.parameter): np.empty(0)
                for p in settings.modulation.parameters
            }
        span = (start, frames)
        if self.cached_span != span:
            self.clear_cache()
            self.cached_span = span
        key = (id(settings), tuple(sorted(sources.items())))
        if key in self.cached_values:
            return {n: v.copy() for n, v in self.cached_values[key].items()}
        signals = self._graph_values(settings, sources, start, frames)
        for binding in settings.bindings:
            if isinstance(binding, processing.GeneratorBinding):
                continue
            if isinstance(binding, processing.EventBinding):
                value = self.contexts[sources[binding.name]].ramps[0].state.value
                signals[binding.name] = np.column_stack(
                    (np.full(frames, value), np.ones(frames))
                )
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

    def _graph_values(
        self, settings: SoundSettings, sources: dict[str, int], start: int, frames: int
    ) -> dict[str, np.ndarray]:
        bindings = [
            b for b in settings.bindings if isinstance(b, processing.GeneratorBinding)
        ]
        if not bindings:
            return {}
        setting = next(i for i, s in enumerate(self.settings) if s is settings)
        program = self.programs[setting]
        voice_id: str | None = None
        for binding in bindings:
            body = settings.envelfos[binding.reference]
            index = sources[binding.name]
            if isinstance(body, Patch):
                voice_id = self.patches[-1 - index].voice_id
            elif isinstance(body, (Envelope, Stages, SampleHold)):
                voice_id = self.envelopes[index].voice_id
            else:
                voice_id = self.lfos[index].voice_id
            if voice_id is not None:
                break
        external: dict[str, np.ndarray] = {}
        for binding in bindings:
            body = settings.envelfos[binding.reference]
            if isinstance(body, LFO) and body.scope != "voice":
                index = sources[binding.name]
                if index not in self.cached_lfos:
                    state = self.lfos[index].state
                    assert state is not None
                    self.cached_lfos[index] = self._source_samples(
                        body, state, start, frames
                    )
                external[program.outputs[binding.name]] = self.cached_lfos[index]
        if voice_id is None:
            return {b.name: external[program.outputs[b.name]] for b in bindings}
        key = f"{setting}:{voice_id}"
        result = {b.name: np.empty((frames, 2)) for b in bindings}
        retained = any(
            isinstance(b, (Latch, Threshold, Slew)) for b in program.envelfos.values()
        )
        transformed = any(
            isinstance(b, (Sum, Product, Affine, Quantize))
            for b in program.envelfos.values()
        )
        i = 0
        while i < frames:
            at = Fraction(start + i, self.sample_rate)
            due = sorted(
                (e.pending_release, j)
                for j, e in enumerate(self.envelopes)
                if e.setting == setting
                and e.voice_id == voice_id
                and e.pending_release is not None
                and e.pending_release <= at
            )
            for release_at, index in due:
                source = self.envelopes[index]
                self._graph_event(
                    setting,
                    voice_id,
                    program.outputs[source.binding_name],
                    EnvelfoEvent(at=release_at, ordinal=0, action="note_off"),
                )
                self.envelopes[index] = source.model_copy(
                    update={"pending_release": None}
                )
            if retained:
                advanced = patch.advance_patch(
                    program,
                    self.graphs[key],
                    at,
                    next_delivery=at + Fraction(1, self.sample_rate),
                    tempo_map=self.tempo_map,
                    inputs={n: float(v[i, 0] * v[i, 1]) for n, v in external.items()},
                )
                self.graphs[key] = advanced.state
                for binding in bindings:
                    body = settings.envelfos[binding.reference]
                    node = program.outputs[binding.name]
                    if isinstance(body, LFO):
                        result[binding.name][i] = (
                            external[node][i]
                            if node in external
                            else self._source_samples(
                                body, advanced.state.children[node], start + i, 1
                            )[0]
                        )
                    else:
                        result[binding.name][i] = advanced.values[binding.name], 1
                i += 1
                continue
            state, _ = patch.settle_patch(
                program, self.graphs[key], at, tempo_map=self.tempo_map
            )
            count = frames - i
            if count > 1:
                limit = Fraction(start + frames - 1, self.sample_rate)
                boundary = patch.next_patch_event(
                    program, state, limit, tempo_map=self.tempo_map
                )
                releases = [
                    e.pending_release
                    for e in self.envelopes
                    if e.setting == setting
                    and e.voice_id == voice_id
                    and e.pending_release is not None
                ]
                if boundary is not None:
                    releases.append(boundary)
                if releases:
                    count = min(
                        count,
                        max(1, ceil(min(releases) * self.sample_rate) - start - i),
                    )
            samples = {n: v[i : i + count] for n, v in external.items()}
            for name, child_state in state.children.items():
                body = program.envelfos[name]
                assert isinstance(body, EnveLFO)
                samples[name] = self._source_samples(
                    body, child_state, start + i, count
                )
            if transformed:
                observed = patch.observe_block(
                    program,
                    {n: (v[:, 0] * v[:, 1]).tolist() for n, v in samples.items()},
                )
                outputs = {n: np.asarray(v) for n, v in observed.items()}
            else:
                outputs = {}
            for binding in bindings:
                if isinstance(settings.envelfos[binding.reference], Patch):
                    node = program.outputs[binding.name]
                    result[binding.name][i : i + count, 0] = (
                        samples[node][:, 0] * samples[node][:, 1]
                        if node in samples
                        else outputs[binding.name]
                    )
                    result[binding.name][i : i + count, 1] = 1
                else:
                    result[binding.name][i : i + count] = samples[
                        program.outputs[binding.name]
                    ]
            if count > 1:
                state, _ = patch.settle_patch(
                    program,
                    state,
                    Fraction(start + i + count - 1, self.sample_rate),
                    tempo_map=self.tempo_map,
                )
            self.graphs[key] = state
            i += count
        return result

    def _source_samples(
        self, body: EnveLFO, state: EnvelfoState, start: int, frames: int
    ) -> np.ndarray:
        if isinstance(body, LFO):
            current = state.runtime
            assert isinstance(current, LFOState)
            position = current.position
            if (
                body.clock == "seconds"
                and not position.paused
                and position.direction == 1
            ):
                values = lfo_samples(
                    body,
                    state,
                    start,
                    frames,
                    self.sample_rate,
                    self.backend,
                    self.control_interval,
                )
                return values
            values = [
                envelfo_at(
                    body,
                    state,
                    self.envelfo_time(body, Fraction(start + i, self.sample_rate)),
                )
                for i in range(frames)
            ]
            return np.array([[v.value, v.weight] for v in values])
        if (
            body.clock == "seconds"
            and (span := self._envelfo_span(body, state, start, frames)) is not None
        ):
            return np.column_stack((span, np.ones(frames)))
        return np.array(
            [
                [
                    envelfo_at(
                        body,
                        state,
                        self.envelfo_time(body, Fraction(start + i, self.sample_rate)),
                    ).value,
                    1,
                ]
                for i in range(frames)
            ]
        )

    def _envelfo_span(
        self, envelfo: EnveLFO, state: EnvelfoState, start: int, frames: int
    ) -> np.ndarray | None:
        """Render an event-free interval; zero frames checks block support."""
        runtime = state.runtime
        body = envelfo
        if isinstance(runtime, StageState):
            assert isinstance(body, Stages)
            if runtime.completed:
                return np.full(frames, runtime.complete_value)
            body = next(s.envelfo for s in body.stages if s.name == runtime.stage)
            initial = runtime.entry_value
        elif isinstance(runtime, EnvelopeState):
            assert isinstance(body, Envelope)
            if runtime.phase == "idle" or runtime.complete_coordinate is not None:
                return np.full(
                    frames,
                    envelfo_at(envelfo, state, Fraction(start, self.sample_rate)).value,
                )
            initial = runtime.start_value
        elif isinstance(runtime, SampleHoldState):
            return np.full(frames, runtime.value)
        else:
            return None
        if isinstance(body, Envelope) and any(
            s.duration is None for s in [*body.segments, *body.release]
        ):
            observed = envelfo_at(envelfo, state, Fraction(start, self.sample_rate))
            if observed.status == "held" and (
                runtime.position.direction == 1
                or envelfo_at(
                    envelfo,
                    state,
                    Fraction(start + max(0, frames - 1), self.sample_rate),
                ).status
                == "held"
            ):
                return np.full(frames, observed.value)
        position = runtime.position
        if position.paused:
            return np.full(
                frames,
                envelfo_at(envelfo, state, Fraction(start, self.sample_rate)).value,
            )
        if position.direction != 1:
            return None
        if isinstance(body, LFO):
            if not frames:
                return np.empty(0)
            samples = lfo_samples(
                body,
                EnvelfoState(runtime=LFOState(at=position.at, position=position)),
                start,
                frames,
                self.sample_rate,
                self.backend,
                1,
            )
            return body.center + (samples[:, 0] - body.center) * samples[:, 1]
        assert isinstance(body, Envelope)
        if body.playback != PlaybackMode.once or position.coordinate < 0:
            return None
        segments = (
            body.release
            if isinstance(runtime, EnvelopeState) and runtime.phase == "release"
            else body.segments
        )
        if any(s.curve for s in segments if isinstance(s, Segment)):
            return None
        if not frames:
            return np.empty(0)
        indefinite = any(s.duration is None for s in [*body.segments, *body.release])
        if indefinite:
            segments = segments[runtime.segment_index :]
        speed = position.rate if indefinite else position.rate * _duration(segments)
        if not speed:
            return np.full(
                frames,
                envelfo_at(envelfo, state, Fraction(start, self.sample_rate)).value,
            )
        scaled = [
            s.model_copy(update={"duration": s.duration / speed})
            if isinstance(s, Segment)
            else s
            for s in segments
        ]
        elapsed = (
            position.coordinate * (1 if indefinite else _duration(segments)) / speed
            + Fraction(start, self.sample_rate)
            - position.at
        )
        return _envelope_values(initial, scaled, elapsed, frames, self.sample_rate)

    def clear_cache(self) -> None:
        """Discard ephemeral arrays after an event, restore, or new render span."""
        self.cached_span = None
        self.cached_values.clear()
        self.cached_lfos.clear()
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
            for m in v.envelfos.values()
        ):
            raise EngineError("Beat-clock Envelfo requires a host tempo map")
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
            patches=self.controls.patches,
            graphs=self.controls.graphs,
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
        self.controls.patches = snapshot.patches
        self.controls.graphs = snapshot.graphs

    def _apply(self, action: instrument_trace.TraceAction) -> None:
        if apply_amplitude_event(
            action, ((v.renderer, v.part, v.trigger_id) for v in self.voices.values())
        ):
            return
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
            part=action.part,
            trigger_id=action.trigger_id,
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
                    filter_order=template.processing.filter_order,
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
                self.controls.stop(voice.voice_id)
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
        if any(m.clock == "beats" for m in template.envelfos.values()):
            if tempo_map is None:
                raise EngineError("Beat-clock Envelfo requires a host tempo map")
        if template.processing != Processing(
            tuning_cents=template.processing.tuning_cents,
            filters=template.processing.filters,
            filter_order=template.processing.filter_order,
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
            self.named_envelfo_sources,
            self.source_scopes,
            runtime_filters,
            staged_envelfos,
            self.staged_envelfo_sources,
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
            cast(float, template.envelope.initial),
            _runtime_segments(template.envelope.segments, rate),
            _runtime_segments(template.envelope.release, rate),
            (
                (hold := template.minimum_hold_seconds * rate).numerator,
                hold.denominator,
            ),
            controls,
            smoothing,
            lfos,
            lfo_rationals,
            runtime_filters,
            parameters,
            context_capacity,
            template.processing.filter_order == processing.FilterOrder.after_amplitude,
        )
        self.runtime.set_named_envelopes(
            envelope_initials,
            envelope_attacks,
            envelope_releases,
            envelope_parameters,
        )
        _configure_sample_hold_envelfos(
            self.runtime, template, self.named_envelfo_sources
        )
        if tempo_map is not None:
            self.runtime.set_beat_clock(
                [
                    cast(
                        tuple[
                            tuple[int, int],
                            tuple[int, int],
                            tuple[int, int],
                            tuple[int, int],
                        ],
                        tuple(
                            (v.numerator, v.denominator)
                            for v in (
                                p.at_seconds * rate,
                                tempo_map.elapsed_beats(Fraction(0), p.at_seconds),
                                p.beat,
                                p.bpm / (60 * rate) if p.running else Fraction(0),
                            )
                        ),
                    )
                    for p in tempo_map.points
                ]
            )
            self.runtime.set_beat_named_envelopes(
                [
                    i
                    for name, indices in self.named_envelfo_sources.items()
                    if template.envelfos[name].clock == "beats"
                    for i in indices
                ]
            )
        self.runtime.set_staged_envelfos(staged_envelfos, event_connections)
        patch_children = _patch_child_names(template)
        self.runtime.set_envelfo_state_owners(
            _envelfo_state_owners(
                len(lfos), patch_children, self.lfo_sources, self.voice_lfo_sources
            ),
            _envelfo_state_owners(
                len(envelope_initials), patch_children, self.named_envelfo_sources
            ),
            _envelfo_state_owners(
                len(staged_envelfos), patch_children, self.staged_envelfo_sources
            ),
        )
        for source_map in (
            self.lfo_sources,
            self.voice_lfo_sources,
            self.named_envelfo_sources,
            self.staged_envelfo_sources,
        ):
            for name, indices in source_map.items():
                if name in patch_children:
                    source_map[name] = indices[:1]
        patch_events: list[
            tuple[
                int,
                tuple[int, int],
                int,
                str | None,
                int,
                int,
                int,
                int,
                int,
                tuple[int, int],
                bool,
            ]
        ] = []
        self.transform_nodes = _configure_patch_transforms(
            self.runtime,
            template,
            self.lfo_sources | self.voice_lfo_sources,
            self.named_envelfo_sources,
            self.staged_envelfo_sources,
            self.definition.sample_rate,
        )
        patch_stage_starts: list[
            tuple[int, str, int, int, int, int, int, int, tuple[int, int], bool]
        ] = []
        for parent_name, parent in template.envelfos.items():
            if not isinstance(parent, Patch):
                continue
            if (
                parent.events
                and f"patch-{parent_name}-{parent.events[0].target}"
                not in template.envelfos
                and not any(
                    isinstance(b, processing.GeneratorBinding)
                    and b.reference == parent_name
                    for b in template.bindings
                )
            ):
                continue
            for order, connection in enumerate(parent.events):
                if connection.action not in ("start", "sample", "capture"):
                    continue
                child_name, _, port = connection.source.partition(".")
                child = parent.envelfos[child_name]
                if isinstance(child, Threshold):
                    continue
                capture = connection.action == "capture"
                destination = (
                    self.transform_nodes[parent_name, connection.target]
                    if capture
                    else self.named_envelfo_sources[
                        f"patch-{parent_name}-{connection.target}"
                    ][0]
                )
                key = envelfo_random.stream_key(
                    0,
                    f"{parent_name}:capture-{connection.target}-{order}"
                    if capture
                    else f"patch-{parent_name}-{connection.target}:start-{order}",
                )
                if isinstance(child, LFO):
                    marker = next(m.position for m in child.markers if m.name == port)
                    patch_events.append(
                        (
                            self.voice_lfo_sources[f"patch-{parent_name}-{child_name}"][
                                0
                            ],
                            (marker.numerator, marker.denominator),
                            destination,
                            None,
                            order,
                            connection.every,
                            connection.offset,
                            int(connection.probability * 2**53),
                            key,
                            _runtime_event_delay(
                                connection.delay, self.definition.sample_rate
                            ),
                            capture,
                        )
                    )
                else:
                    assert isinstance(child, Stages)
                    patch_stage_starts.append(
                        (
                            self.staged_envelfo_sources[
                                f"patch-{parent_name}-{child_name}"
                            ][0],
                            port,
                            destination,
                            order,
                            connection.every,
                            connection.offset,
                            int(connection.probability * 2**53),
                            key,
                            _runtime_event_delay(
                                connection.delay, self.definition.sample_rate
                            ),
                            capture,
                        )
                    )
        bindings = {binding.name: binding for binding in template.bindings}
        for order, connection in enumerate(template.event_connections):
            source_binding = bindings[connection.source]
            assert isinstance(source_binding, processing.GeneratorBinding)
            source_envelfo = template.envelfos[source_binding.reference]
            if not isinstance(source_envelfo, LFO):
                continue
            marker = next(
                m.position for m in source_envelfo.markers if m.name == connection.port
            )
            target_binding = bindings[connection.destination]
            assert isinstance(target_binding, processing.GeneratorBinding)
            patch_events.append(
                (
                    self.voice_lfo_sources[source_binding.reference][0],
                    (marker.numerator, marker.denominator),
                    self.staged_envelfo_sources[target_binding.reference][0],
                    connection.cue,
                    len(patch_events),
                    connection.every,
                    connection.offset,
                    int(connection.probability * 2**53),
                    envelfo_random.stream_key(
                        0, f"{target_binding.reference}:cue-{order}"
                    ),
                    _runtime_event_delay(connection.delay, self.definition.sample_rate),
                    False,
                )
            )
        self.runtime.set_patch_events(patch_events)
        self.runtime.set_patch_stage_starts(patch_stage_starts)
        self.runtime.set_threshold_events(
            _threshold_runtime_connections(
                template,
                self.named_envelfo_sources,
                self.staged_envelfo_sources,
                self.transform_nodes,
                self.definition.sample_rate,
            )
        )

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
            elif isinstance(action, instrument_trace.EnvelfoObservation):
                if action.target == "envelope":
                    operator = 0
                    if isinstance(self.template, FMVoice):
                        name = (
                            self.template.fm.carrier if action.op is None else action.op
                        )
                        operator = next(
                            (
                                i
                                for i, o in enumerate(self.template.fm.operators)
                                if o.name == name
                            ),
                            -1,
                        )
                        if operator < 0:
                            raise EngineError(f"Unknown amplitude operator: {name}")
                    elif action.op is not None:
                        raise EngineError("op requires an FM voice")
                    command = (
                        "pause",
                        "resume",
                        "reverse",
                        "seek",
                        "shift",
                        "advance",
                    ).index(action.action)
                    for voice_id, slot in self.voices.items():
                        if self.voice_triggers[voice_id] == VoiceAddress(
                            part=action.part, trigger_id=action.trigger_id
                        ):
                            self._encode_action(
                                count,
                                offset,
                                13,
                                slot,
                                operator,
                                command,
                                action.position
                                if action.position is not None
                                else action.offset
                                if action.offset is not None
                                else 0,
                            )
                            count += 1
                    continue
                assert action.name is not None
                envelfo = self.template.envelfos.get(action.name)
                if action.name in _patch_child_names(self.template) or (
                    envelfo is not None and isinstance(envelfo, Patch)
                ):
                    raise EngineError(
                        "Patch-level Envelfo commands are not implemented"
                    )
                command = (
                    "pause",
                    "resume",
                    "reverse",
                    "seek",
                    "shift",
                    "advance",
                ).index(action.action)
                for voice_id, slot in self.voices.items():
                    if self.voice_triggers[voice_id] != VoiceAddress(
                        part=action.part, trigger_id=action.trigger_id
                    ):
                        continue
                    for kind, sources in (
                        (9, self.named_envelfo_sources),
                        (10, self.staged_envelfo_sources),
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
                if self.probabilistic_events:
                    self._encode_action(
                        count,
                        offset,
                        12,
                        slot,
                        action.envelfo_key & 0xFFFF_FFFF,
                        action.envelfo_key >> 32,
                        0,
                    )
                    count += 1
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

    @cached_property
    def probabilistic_events(self) -> bool:
        return (
            any(0 < c.probability < 1 for c in self.template.event_connections)
            or any(
                isinstance(m, Patch) and any(0 < c.probability < 1 for c in m.events)
                for m in self.template.envelfos.values()
            )
            or any(
                isinstance(self.template.envelfos[n], SampleHold)
                for n in self.named_envelfo_sources
            )
        )

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


def validate_envelope(envelope: Envelope) -> None:
    """Validate the amplitude-envelope profile shared by both voice renderers."""
    if envelope.clock != "seconds" or envelope.scope != "voice":
        raise EngineError("Envelopes must use the voice seconds clock")
    if not isinstance(envelope.initial, float) or envelope.start != "activation":
        raise EngineError(
            "Voice amplitude envelopes require a numeric initial and activation start"
        )
    if not envelope.release:
        raise EngineError("Voice amplitude envelopes require an explicit release")
    if envelope.playback != PlaybackMode.once or envelope.markers:
        raise EngineError(
            "Voice amplitude envelopes require once playback without markers"
        )
    if not envelope.hold or any(
        isinstance(s, Segment) and s.duration != 0 and s.curve != 0
        for s in [*envelope.segments, *envelope.release]
    ):
        raise EngineError("Only held linear envelopes are implemented")


def validate_generators(
    settings: SoundSettings,
    allow_beat_envelfos: bool = False,
    allow_velocity_bindings: bool = False,
) -> None:
    """Validate the Envelfo clocks supported by the selected renderer."""
    if any(isinstance(g, EnveLFOReference) for g in settings.envelfos.values()):
        raise EngineError("Library Envelfo uses must be materialized before rendering")
    if any(
        g.scope != "voice"
        for g in settings.envelfos.values()
        if isinstance(g, (Envelope, Stages, SampleHold))
    ):
        raise EngineError("Staged and envelope Envelfos require voice scope")
    if any(
        g.clock != "seconds"
        and (
            not (
                allow_beat_envelfos
                and isinstance(g, (Envelope, LFO, Stages, SampleHold, Patch))
            )
        )
        for g in settings.envelfos.values()
    ):
        raise EngineError("Beat-clock Envelfos require a supported synth renderer")
    if any(
        not isinstance(b, ControlBinding)
        and not isinstance(b, processing.GeneratorBinding)
        and not (
            allow_velocity_bindings
            and isinstance(b, processing.EventBinding)
            and b.kind == "velocity"
        )
        for b in settings.bindings
    ):
        raise EngineError("Unsupported binding for this renderer")


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
        tuning_cents=voice.processing.tuning_cents,
        filters=voice.processing.filters,
        filter_order=voice.processing.filter_order,
    ):
        raise EngineError("Only tuning and filter processing are implemented")
    validate_generators(voice, allow_beat_envelfos=True, allow_velocity_bindings=True)
    if any(c.mode == "fade" for c in voice.chokes):
        raise EngineError("Fade retirement is not implemented")
    validate_modulation(voice)


def _envelope_values(
    initial: float,
    segments: list[Segment | IndefiniteSegment],
    elapsed: Fraction,
    frames: int,
    sample_rate: int,
) -> np.ndarray:
    values = np.full(frames, initial)
    boundary = Fraction(0)
    entry = initial
    for segment in segments:
        if isinstance(segment, IndefiniteSegment):
            first = max(0, ceil((boundary - elapsed) * sample_rate))
            values[first:] = entry
            return values
        end = boundary + segment.duration
        first = max(0, ceil((boundary - elapsed) * sample_rate))
        last = min(frames, ceil((end - elapsed) * sample_rate))
        if segment.duration and first < last:
            progress = float((elapsed - boundary) / segment.duration) + np.arange(
                first, last
            ) / float(segment.duration * sample_rate)
            values[first:last] = entry + (segment.to - entry) * progress
        boundary, entry = end, segment.to
    values[max(0, ceil((boundary - elapsed) * sample_rate)) :] = entry
    return values


def _duration(segments: list[Segment | IndefiniteSegment]) -> Fraction:
    if any(s.duration is None for s in segments):
        raise EngineError("An indefinite Envelope has no finite duration")
    return sum((s.duration for s in segments if isinstance(s, Segment)), Fraction(0))


def _runtime_segments(
    segments: list[Segment | IndefiniteSegment], sample_rate: int
) -> list[tuple[tuple[int, int] | None, float]]:
    return [
        (((d := s.duration * sample_rate).numerator, d.denominator), float(s.to))
        if isinstance(s, Segment)
        else (None, 0.0)
        for s in segments
    ]


def _runtime_event_delay(seconds: Fraction, sample_rate: int) -> tuple[int, int]:
    frames = seconds * sample_rate
    if not -(1 << 63) <= frames.numerator < (1 << 63) or frames.denominator >= (
        1 << 63
    ):
        raise EngineError("Delayed Envelfo event time is not representable")
    return frames.numerator, frames.denominator


def _configure_sample_hold_envelfos(
    runtime: _native.SynthRuntime,
    settings: SoundSettings,
    sources: dict[str, list[int]],
) -> None:
    definitions = [
        (
            i,
            envelfo_random.stream_key(0, name + ":sample-hold"),
            body.minimum,
            body.maximum,
        )
        for name, indices in sources.items()
        if isinstance((body := settings.envelfos[name]), SampleHold)
        for i in indices
    ]
    runtime.set_sample_hold_envelfos(definitions)


def _configure_patch_transforms(
    runtime: _native.SynthRuntime,
    template: SoundSettings,
    lfos: dict[str, list[int]],
    envelopes: dict[str, list[int]],
    stages: dict[str, list[int]],
    rate: int,
) -> dict[tuple[str, str], int]:
    nodes: list[
        tuple[int, list[tuple[int, int, float, float]], float, float, float | None]
    ] = []
    outputs: dict[tuple[str, str], int] = {}
    detectors: dict[tuple[str, str], int] = {}
    for parent_name, parent in template.envelfos.items():
        if not isinstance(parent, Patch):
            continue
        children: dict[str, tuple[int, int, float, float]] = {}
        for name in parent.signal_order:
            body = parent.envelfos[name]
            reference = f"patch-{parent_name}-{name}"
            if isinstance(body, LFO) and reference in lfos:
                children[name] = (0, lfos[reference][0], body.center, body.depth)
            elif isinstance(body, (Envelope, SampleHold)) and reference in envelopes:
                children[name] = (1, envelopes[reference][0], 0.0, 1.0)
            elif isinstance(body, Stages) and reference in stages:
                children[name] = (2, stages[reference][0], 0.0, 1.0)
            elif isinstance(
                body, (Sum, Product, Affine, Quantize, Latch, Threshold, Slew)
            ):
                inputs = (
                    [body.input]
                    if isinstance(body, (Affine, Quantize, Latch, Threshold, Slew))
                    else body.inputs
                )
                if any(n not in children for n in inputs):
                    continue
                children[name] = (3, len(nodes), 0.0, 1.0)
                if isinstance(body, (Threshold, Latch)):
                    detectors[parent_name, name] = len(nodes)
                nodes.append(
                    (
                        6
                        if isinstance(body, Latch)
                        else 5
                        if isinstance(body, Quantize)
                        else 4
                        if isinstance(body, Slew)
                        else 3
                        if isinstance(body, Threshold)
                        else 2
                        if isinstance(body, Affine)
                        else 0
                        if isinstance(body, Sum)
                        else 1,
                        [children[n] for n in inputs],
                        body.step
                        if isinstance(body, Quantize)
                        else body.rise * float(Fraction(1, rate))
                        if isinstance(body, Slew)
                        else body.lower
                        if isinstance(body, Threshold)
                        else body.scale
                        if isinstance(body, Affine)
                        else 1.0,
                        body.origin
                        if isinstance(body, Quantize)
                        else body.fall * float(Fraction(1, rate))
                        if isinstance(body, Slew)
                        else body.upper
                        if isinstance(body, Threshold)
                        else body.offset
                        if isinstance(body, Affine)
                        else 0.0,
                        body.initial if isinstance(body, Slew) else None,
                    )
                )
        for output, name in parent.outputs.items():
            if (
                name in children
                and children[name][0] != 3
                and any(
                    isinstance(b, (Latch, Threshold, Slew))
                    for b in parent.envelfos.values()
                )
            ):
                nodes.append((2, [children[name]], 1.0, 0.0, None))
                outputs[parent_name, output] = len(nodes) - 1
            if name in children and children[name][0] == 3:
                outputs[parent_name, output] = children[name][1]
    parameters = {("processing", "amplitude"): 0, ("processing", "tuning_cents"): 1}
    for i, body in enumerate(template.processing.filters):
        parameters[f"filter-{body.name}", "cutoff_hz"] = 2 + 2 * i
        parameters[f"filter-{body.name}", "q"] = 3 + 2 * i
    routes: list[tuple[int, int, int, float, float]] = []
    bindings = {b.name: b for b in template.bindings}
    sources = {s.name: s for s in template.modulation.sources}
    for route in sorted(template.modulation.routes, key=lambda r: (r.source, r.name)):
        binding = bindings[route.source]
        if (
            not isinstance(binding, processing.GeneratorBinding)
            or binding.output is None
        ):
            continue
        source = sources[route.source]
        slope = (route.points[1].amount - route.points[0].amount) / (
            source.maximum - source.minimum
        )
        routes.append(
            (
                outputs[binding.reference, binding.output],
                parameters[route.target.name, route.target.parameter],
                0 if route.operation == modulation.Operation.add else 1,
                route.points[0].amount - slope * source.minimum,
                slope,
            )
        )
    runtime.set_envelfo_transforms(nodes, routes)
    return detectors


def _threshold_runtime_connections(
    template: SoundSettings,
    envelopes: dict[str, list[int]],
    stages: dict[str, list[int]],
    nodes: dict[tuple[str, str], int],
    rate: int,
) -> list[
    tuple[
        int,
        tuple[int, int],
        int,
        str | None,
        int,
        int,
        int,
        int,
        int,
        tuple[int, int],
        bool,
    ]
]:
    result: list[
        tuple[
            int,
            tuple[int, int],
            int,
            str | None,
            int,
            int,
            int,
            int,
            int,
            tuple[int, int],
            bool,
        ]
    ] = []
    bindings = {b.name: b for b in template.bindings}
    for (parent_name, child), node in nodes.items():
        parent = template.envelfos[parent_name]
        assert isinstance(parent, Patch)
        if not isinstance(parent.envelfos[child], Threshold):
            continue
        for order, connection in enumerate(parent.events):
            origin, _, port = connection.source.partition(".")
            if origin != child:
                continue
            target = f"patch-{parent_name}-{connection.target}"
            cue = connection.cue
            capture = connection.action == "capture"
            result.append(
                (
                    node,
                    (int(port == "rising"), 1),
                    nodes[parent_name, connection.target]
                    if capture
                    else (stages if cue is not None else envelopes)[target][0],
                    cue,
                    order,
                    connection.every,
                    connection.offset,
                    int(connection.probability * 2**53),
                    envelfo_random.stream_key(
                        0,
                        f"{parent_name}:capture-{connection.target}-{order}"
                        if capture
                        else f"{target}:"
                        f"{'cue' if cue is not None else 'start'}-{order}",
                    ),
                    _runtime_event_delay(connection.delay + Fraction(1, rate), rate),
                    capture,
                )
            )
        for order, connection in enumerate(template.event_connections):
            binding = bindings[connection.source]
            if (
                not isinstance(binding, processing.GeneratorBinding)
                or binding.reference != parent_name
            ):
                continue
            origin, _, port = parent.event_outputs[connection.port].partition(".")
            if origin != child:
                continue
            target_binding = bindings[connection.destination]
            assert isinstance(target_binding, processing.GeneratorBinding)
            target = target_binding.reference
            result.append(
                (
                    node,
                    (int(port == "rising"), 1),
                    stages[target][0],
                    connection.cue,
                    order,
                    connection.every,
                    connection.offset,
                    int(connection.probability * 2**53),
                    envelfo_random.stream_key(0, f"{target}:cue-{order}"),
                    _runtime_event_delay(connection.delay + Fraction(1, rate), rate),
                    False,
                )
            )
    return result


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
    list[list[tuple[tuple[int, int] | None, float]]],
    list[list[tuple[tuple[int, int] | None, float]]],
    list[
        tuple[
            int,
            int,
            float,
            float,
            bool,
            int,
            tuple[int, int],
            tuple[int, int],
            int | None,
            bool,
            tuple[bool, int, bool],
        ]
    ],
    dict[str, list[int]],
    set[str],
    np.ndarray,
    list[StagedRuntimeDefinition],
    dict[str, list[int]],
    list[tuple[int, str, int, str, int, int, int, int, tuple[int, int]]],
]:
    bindings = {b.name: b for b in template.bindings}
    if any(
        not isinstance(b, ControlBinding)
        and not isinstance(b, processing.GeneratorBinding)
        for b in bindings.values()
    ):
        raise EngineError("Persistent synth supports control and envelfo bindings")
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
        for parent_name, parent in template.envelfos.items()
        if isinstance(parent, Patch)
        for child in (
            {
                name
                for connection in parent.events
                for name in (connection.source.partition(".")[0], connection.target)
            }
            | {source.partition(".")[0] for source in parent.event_outputs.values()}
        )
    }
    event_only_patch_outputs = {
        f"patch-{parent_name}-{child}"
        for parent_name, parent in template.envelfos.items()
        if isinstance(parent, Patch) and parent.event_outputs
        for child in parent.outputs.values()
    }
    patch_children.update(_patch_child_names(template))
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
    envelope_attacks: list[list[tuple[tuple[int, int] | None, float]]] = []
    envelope_releases: list[list[tuple[tuple[int, int] | None, float]]] = []
    envelope_parameters: list[
        tuple[
            int,
            int,
            float,
            float,
            bool,
            int,
            tuple[int, int],
            tuple[int, int],
            int | None,
            bool,
            tuple[bool, int, bool],
        ]
    ] = []
    named_envelfo_sources: dict[str, list[int]] = {}
    staged_envelfos: list[StagedRuntimeDefinition] = []
    staged_bindings: dict[str, int] = {}
    staged_envelfo_sources: dict[str, list[int]] = {}
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
            intercept + slope * (1 if declaration is not None else source.maximum),
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
            envelfo = template.envelfos[binding.reference]
            if isinstance(envelfo, Patch):
                continue
            if isinstance(envelfo, SampleHold):
                named_envelfo_sources.setdefault(binding.reference, []).append(
                    len(envelope_initials)
                )
                envelope_initials.append(0.0)
                envelope_attacks.append([])
                envelope_releases.append([])
                envelope_parameters.append(
                    (
                        parameter,
                        0 if operation == modulation.Operation.add else 1,
                        intercept,
                        slope,
                        False,
                        0,
                        (0, 1),
                        (1, 1),
                        None,
                        True,
                        (True, 0, True),
                    )
                )
                continue
            if isinstance(envelfo, Envelope):
                generator = envelfo
                assert isinstance(generator.initial, float)
                if source.scope != "voice" or any(
                    isinstance(segment, Segment) and segment.curve != 0
                    for segment in [*generator.segments, *generator.release]
                ):
                    raise EngineError(
                        "Persistent named envelopes require linear voice segments"
                    )
                named_envelfo_sources.setdefault(binding.reference, []).append(
                    len(envelope_initials)
                )
                envelope_initials.append(generator.initial)
                duration_scale = (
                    1 if envelfo.clock == "beats" else definition.sample_rate
                )
                envelope_attacks.append(
                    _runtime_segments(generator.segments, duration_scale)
                )
                envelope_releases.append(
                    _runtime_segments(generator.release, duration_scale)
                )
                envelope_parameters.append(
                    (
                        parameter,
                        0 if operation == modulation.Operation.add else 1,
                        intercept,
                        slope,
                        binding.release_timing == ReleaseTiming.voice,
                        *_runtime_named_envelope_options(generator),
                    )
                )
                continue
            if isinstance(envelfo, Stages):
                staged_bindings[binding.name] = len(staged_envelfos)
                staged_envelfo_sources.setdefault(binding.reference, []).append(
                    len(staged_envelfos)
                )
                staged_envelfos.append(
                    _runtime_staged_envelfo(
                        envelfo,
                        definition.sample_rate,
                        parameter,
                        0 if operation == modulation.Operation.add else 1,
                        intercept,
                        slope,
                        envelfo.clock == "beats",
                    )
                )
                continue
            assert isinstance(envelfo, LFO)
            intercept += slope * envelfo.center
            slope *= envelfo.depth
            generator = envelfo
            index = len(lfo_rows)
            lfo_rows.append(
                [
                    ["instrument", "part", "trigger", "voice"].index(source.scope),
                    [Waveform.sine, Waveform.square, Waveform.triangle].index(
                        generator.shape
                    ),
                    parameter,
                    0 if operation == modulation.Operation.add else 1,
                    intercept,
                    slope,
                    2.0
                    if envelfo.position_driver == "transport"
                    else 1.0
                    if envelfo.clock == "beats"
                    else 0.0,
                ]
            )
            for value in (
                generator.duty_cycle,
                generator.rate,
                generator.phase,
                generator.delay
                * (1 if envelfo.clock == "beats" else definition.sample_rate),
                generator.fade_in
                * (1 if envelfo.clock == "beats" else definition.sample_rate),
            ):
                assert isinstance(value, Fraction)
                lfo_rationals.append((value.numerator, value.denominator))
            if source.scope == "instrument":
                lfo_sources.setdefault(binding.reference, []).append(index)
            elif source.scope == "voice":
                voice_lfo_sources.setdefault(binding.reference, []).append(index)
    for binding in template.bindings:
        if binding.name not in event_sources or binding.name in routed_sources:
            continue
        assert isinstance(binding, processing.GeneratorBinding)
        envelfo = template.envelfos[binding.reference]
        if isinstance(envelfo, Patch):
            continue
        if isinstance(envelfo, SampleHold):
            named_envelfo_sources.setdefault(binding.reference, []).append(
                len(envelope_initials)
            )
            envelope_initials.append(0.0)
            envelope_attacks.append([])
            envelope_releases.append([])
            envelope_parameters.append(
                (0, 1, 1, 0, False, 0, (0, 1), (1, 1), None, True, (True, 0, True))
            )
            continue
        if isinstance(envelfo, LFO):
            generator = envelfo
            source_scopes.add(envelfo.scope)
            if envelfo.scope != "voice":
                lfo_sources.setdefault(binding.reference, []).append(len(lfo_rows))
            else:
                voice_lfo_sources.setdefault(binding.reference, []).append(
                    len(lfo_rows)
                )
            lfo_rows.append(
                [
                    ["instrument", "part", "trigger", "voice"].index(envelfo.scope),
                    [Waveform.sine, Waveform.square, Waveform.triangle].index(
                        generator.shape
                    ),
                    0,
                    1,
                    1,
                    0,
                    1 if envelfo.clock == "beats" else 0,
                ]
            )
            for value in (
                generator.duty_cycle,
                generator.rate,
                generator.phase,
                generator.delay
                * (1 if envelfo.clock == "beats" else definition.sample_rate),
                generator.fade_in
                * (1 if envelfo.clock == "beats" else definition.sample_rate),
            ):
                assert isinstance(value, Fraction)
                lfo_rationals.append((value.numerator, value.denominator))
            continue
        if isinstance(envelfo, Envelope):
            generator = envelfo
            assert isinstance(generator.initial, float)
            if envelfo.scope != "voice" or any(
                isinstance(s, Segment) and s.curve != 0
                for s in [*generator.segments, *generator.release]
            ):
                raise EngineError(
                    "Persistent named envelopes require linear voice segments"
                )
            duration_scale = 1 if envelfo.clock == "beats" else definition.sample_rate
            named_envelfo_sources.setdefault(binding.reference, []).append(
                len(envelope_initials)
            )
            envelope_initials.append(generator.initial)
            envelope_attacks.append(
                _runtime_segments(generator.segments, duration_scale)
            )
            envelope_releases.append(
                _runtime_segments(generator.release, duration_scale)
            )
            envelope_parameters.append(
                (
                    0,
                    1,
                    1,
                    0,
                    binding.release_timing == ReleaseTiming.voice,
                    *_runtime_named_envelope_options(generator),
                )
            )
            continue
        assert isinstance(envelfo, Stages)
        source_scopes.add(envelfo.scope)
        staged_bindings[binding.name] = len(staged_envelfos)
        staged_envelfo_sources.setdefault(binding.reference, []).append(
            len(staged_envelfos)
        )
        staged_envelfos.append(
            _runtime_staged_envelfo(
                envelfo, definition.sample_rate, None, 0, 0, 0, envelfo.clock == "beats"
            )
        )
    event_connections = [
        (
            staged_bindings[c.source],
            c.port,
            staged_bindings[c.destination],
            c.cue,
            c.every,
            c.offset,
            int(c.probability * 2**53),
            envelfo_random.stream_key(
                0,
                cast(processing.GeneratorBinding, bindings[c.destination]).reference
                + f":cue-{order}",
            ),
            _runtime_event_delay(c.delay, definition.sample_rate),
        )
        for order, c in enumerate(template.event_connections)
        if isinstance((binding := bindings[c.source]), processing.GeneratorBinding)
        if isinstance(template.envelfos[binding.reference], Stages)
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
        named_envelfo_sources,
        source_scopes,
        np.asarray(runtime_filters, dtype=np.float64).reshape(-1, 7),
        staged_envelfos,
        staged_envelfo_sources,
        event_connections,
    )


def _envelfo_state_owners(
    count: int, names: set[str], *source_maps: dict[str, list[int]]
) -> list[int]:
    owners = list(range(count))
    for source_map in source_maps:
        for name in names:
            if indices := source_map.get(name):
                for index in indices[1:]:
                    owners[index] = indices[0]
    return owners


def _runtime_named_envelope_options(
    generator: Envelope,
) -> tuple[
    int, tuple[int, int], tuple[int, int], int | None, bool, tuple[bool, int, bool]
]:
    bounds = [
        next(m.position for m in generator.markers if m.name == n)
        if n is not None
        else f
        for n, f in (
            (generator.loop_start, Fraction(0)),
            (generator.loop_end, Fraction(1)),
        )
    ]
    return (
        list(PlaybackMode).index(generator.playback),
        (bounds[0].numerator, bounds[0].denominator),
        (bounds[1].numerator, bounds[1].denominator),
        generator.repeat_count,
        generator.start == "event",
        (
            generator.hold,
            ("current", "initial", "ignore").index(generator.retrigger),
            generator.polarity == "bipolar",
        ),
    )


def _runtime_staged_envelfo(
    body: Stages,
    sample_rate: int,
    parameter: int | None,
    operation: int,
    intercept: float,
    slope: float,
    beat_clock: bool,
) -> StagedRuntimeDefinition:
    names = {stage.name: index for index, stage in enumerate(body.stages)}
    stages: list[StagedRuntimeStage] = []
    scale = 1 if beat_clock else sample_rate
    for stage in body.stages:
        envelfo = stage.envelfo
        markers = [
            ((m.position.numerator, m.position.denominator), m.name)
            for m in envelfo.markers
        ]
        if isinstance(envelfo, Envelope):
            if any(s.curve != 0 for s in envelfo.segments if isinstance(s, Segment)):
                raise EngineError("Persistent staged envelopes require linear segments")
            options = _runtime_named_envelope_options(envelfo)
            stages.append(
                (
                    1,
                    0.0 if envelfo.initial == "current" else envelfo.initial,
                    envelfo.initial == "current",
                    _runtime_segments(envelfo.segments, scale),
                    None,
                    markers,
                    options[0],
                    options[1],
                    options[2],
                    envelfo.repeat_count,
                )
            )
        else:
            assert isinstance(envelfo, LFO)
            assert isinstance(envelfo.rate, Fraction)
            times = [
                envelfo.rate / scale,
                envelfo.phase,
                envelfo.duty_cycle,
                envelfo.delay * scale,
                envelfo.fade_in * scale,
            ]
            stages.append(
                (
                    2,
                    0.0,
                    False,
                    [],
                    (
                        [Waveform.sine, Waveform.square, Waveform.triangle].index(
                            envelfo.shape
                        ),
                        [(t.numerator, t.denominator) for t in times],
                        envelfo.center,
                        envelfo.depth,
                    ),
                    markers,
                    0,
                    (0, 1),
                    (1, 1),
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
