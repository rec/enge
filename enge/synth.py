"""Offline rendering for Ufor's oscillator synth profile."""

import json
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
    resolved_envelope,
)
from pydantic import BaseModel, ConfigDict, Field, model_validator
from ufor import envelfo_random, instrument_trace, modulation, synth_trace
from ufor.approach import ExponentialApproach
from ufor.base import Model
from ufor.control import TempoMap
from ufor.envelfo import (
    LFO,
    Affine,
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
from ufor.events import PerformanceEvent
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

from . import (
    _native,
    control,
    envelfo_graph,
    filters,
    native,
    voice_admission,
    voice_release,
)
from .lfo import lfo_samples


class EngineError(ValueError):
    """The requested input is invalid or outside Enge's implemented profile."""


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
            curve,
        ) in self.amplitude_spans(frames, op):
            a, b = int(first), int(last)
            values[a:b] = entry + delta * _curve_progress(
                origin + (np.arange(a, b) - offset) / denominator, curve
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
    trigger_velocities: dict[int, float | None]
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


def prepare_trace(
    definition: SynthInstrument,
    events: list[PerformanceEvent],
    seed: int,
    *,
    tempo_map: TempoMap | None = None,
    sample_rate: int | None = None,
    max_records: int = 10_000,
) -> synth_trace.SynthTrace:
    """Prepare synth-family actions with Rust uFor voice and release rules."""
    fields = {"name", "mapping", "trigger"}
    eligibility = _native.PreparedVoiceEligibility(
        json.dumps(
            [v.model_dump(mode="json", include=fields) for v in definition.voices]
        ),
        max_records=max_records,
    )

    def schedule(
        tempo: TempoMap, rate: int, coordinates: list[tuple[int, int, Fraction | None]]
    ) -> list[tuple[int, int, int]]:
        return _native.schedule_envelfo_commands(
            tempo.model_dump_json(),
            rate,
            [
                (t, o, (d.numerator, d.denominator) if d is not None else None)
                for t, o, d in coordinates
            ],
            max_records=max_records,
        )

    return synth_trace.prepare(
        definition,
        events,
        seed,
        tempo_map=tempo_map,
        sample_rate=sample_rate,
        admit_voices=voice_admission.native_retirements,
        release_plan=voice_release.native_release_plan,
        eligible_voices=eligibility.matching_voices,
        prepare_start=_native.prepare_voice_start,
        schedule=schedule,
    )


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
        voice = expand_patch_outputs(voice)
        _validate_voice(voice)
        filters.parameters(voice.processing.filters, _sample_rate(timebase), 1)
        voices.append(voice)
    return PreparedSynth(
        sample_rate=_sample_rate(timebase),
        channels=output.channels,
        instrument=score.body.model_copy(update={"voices": voices}),
        source_instrument=score.body if voices != score.body.voices else None,
    )


def expand_patch_outputs[T: processing.ModulationSettings](voice: T) -> T:
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
    return type(voice).model_validate(
        voice.model_dump()
        | {
            "envelfos": envelfos,
            "bindings": bindings,
            "event_connections": event_connections,
            "modulation": voice.modulation.model_copy(update={"sources": sources}),
        }
    )


def _patch_child_names(settings: processing.ModulationSettings) -> set[str]:
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
        settings: list[processing.ModulationSettings],
        backend: Literal["numpy", "native"] = "numpy",
        control_interval: int = 1,
        tempo_map: TempoMap | None = None,
        control_sources: list[tuple[str, str, Fraction]] | None = None,
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
        self.control_sources = list(control_sources or [])
        for setting in settings:
            scopes = {s.name: s.scope for s in setting.modulation.sources}
            self.control_sources.extend(
                (scopes[b.name], b.control, b.smoothing)
                for b in setting.bindings
                if isinstance(b, ControlBinding)
            )
        self.control_sources = list(dict.fromkeys(self.control_sources))
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
        for source_scope, name, smoothing in self.control_sources:
            if source_scope != scope:
                continue
            ramps.append(
                ControlRamp(
                    control=name,
                    state=controls.initial_control(
                        self.declarations[name],
                        Fraction(frame, self.sample_rate),
                        smoothing,
                        values.get(name),
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
        if isinstance(action, instrument_trace.TriggerRelease):
            return True
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
        settings: processing.ModulationSettings,
        action: instrument_trace.VoiceStart,
        velocity: float | None = None,
    ) -> dict[str, int]:
        result: dict[str, int] = {}
        setting = next(i for i, s in enumerate(self.settings) if s is settings)
        bindings = {b.name: b for b in settings.bindings}
        for source in settings.modulation.sources:
            binding = bindings[source.name]
            if isinstance(binding, processing.EventBinding):
                if binding.kind == "key":
                    if action.key is None:
                        raise EngineError("Key binding requires resolved onset key")
                    # Control snapshots store unit-range values; routes use MIDI keys.
                    value = action.key / 127
                else:
                    if velocity is None and action.trigger_id is not None:
                        context_id = self.context(
                            "trigger", action.part, action.trigger_id
                        )
                        if context_id is not None:
                            velocity = self.contexts[context_id].velocity
                    if velocity is None:
                        raise EngineError(
                            "Velocity binding requires resolved onset velocity"
                        )
                    value = velocity
                index = len(self.contexts)
                self.contexts.append(
                    ControlContext(
                        scope="voice",
                        part=action.part,
                        trigger_id=action.trigger_id,
                        voice_id=action.voice_id,
                        ramps=[
                            ControlRamp(
                                control=binding.kind,
                                state=controls.initial_control(
                                    controls.ControlDeclaration(default=value),
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
        settings: processing.ModulationSettings,
        sources: dict[str, int],
        action: instrument_trace.VoiceRetirement,
        voice_release_at: Fraction,
    ) -> None:
        """Release generator bindings at their event or absolute voice instant."""
        self.clear_cache()
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
        self,
        settings: processing.ModulationSettings,
        sources: dict[str, int],
        start: int,
        frames: int,
        defaults: dict[tuple[str, str], np.ndarray] | None = None,
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
        if defaults is None and key in self.cached_values:
            return {n: v.copy() for n, v in self.cached_values[key].items()}
        signals = self._graph_values(settings, sources, start, frames)
        for binding in settings.bindings:
            if isinstance(binding, processing.GeneratorBinding):
                continue
            if isinstance(binding, processing.EventBinding):
                value = self.contexts[sources[binding.name]].ramps[0].state.value
                if binding.kind == "key":
                    value *= 127
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
        output = control.modulation_samples(
            settings.modulation, signals, frames, defaults
        )
        if defaults is None:
            self.cached_values[key] = output
        return {n: v.copy() for n, v in output.items()}

    def _graph_values(
        self,
        settings: processing.ModulationSettings,
        sources: dict[str, int],
        start: int,
        frames: int,
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
            body = resolved_envelope(body, state)
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
            isinstance(s, IndefiniteSegment) for s in [*body.segments, *body.release]
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
        if not frames:
            return np.empty(0)
        indefinite = any(
            isinstance(s, IndefiniteSegment) for s in [*body.segments, *body.release]
        )
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
                    assert voice.renderer.release_frame is not None
                    self.controls.release(
                        self.templates[voice.template],
                        voice.sources,
                        action,
                        Fraction(action.tick, self.definition.sample_rate)
                        + (voice.renderer.release_frame - voice.renderer.frame_count)
                        / self.definition.sample_rate,
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
        prepared = _persistent_modulation(
            definition.sample_rate, definition.instrument.controls, template
        )
        (
            controls,
            smoothing,
            parameters,
            self.control_sources,
            self.control_scopes,
            self.source_controls,
            lfos,
            lfo_documents,
            self.lfo_sources,
            self.voice_lfo_sources,
            envelope_documents,
            envelope_parameters,
            self.named_envelfo_sources,
            self.source_scopes,
            runtime_filters,
            staged_envelfos,
            self.staged_envelfo_sources,
            event_connections,
            route_maps,
            self.event_bindings,
        ) = prepared[:20]
        if "velocity" in self.event_bindings.values():
            self.source_scopes.add("trigger")
        self.trigger_velocities: dict[int, float | None] = {}
        self.part_contexts: dict[str, int] = {}
        self.trigger_contexts: dict[tuple[str, str], int] = {}
        generators, self.transform_nodes, transform_maps = _prepare_native_generators(
            rate, voices, template, prepared, tempo_map
        )
        self.runtime = _native.SynthRuntime(
            rate,
            [Waveform.sine, Waveform.square, Waveform.triangle].index(
                template.oscillator.waveform
            ),
            float(template.oscillator.duty_cycle),
            np.tile(np.asarray(route, dtype=np.float64), (voices, 1)),
            template.envelope.model_dump_json(),
            (
                (hold := template.minimum_hold_seconds * rate).numerator,
                hold.denominator,
            ),
            controls,
            smoothing,
            generators,
            runtime_filters,
            parameters,
            context_capacity,
            template.processing.filter_order == processing.FilterOrder.after_amplitude,
        )
        for kind, mappings in enumerate(route_maps):
            self.runtime.set_route_maps(kind, mappings)
        self.runtime.set_route_maps(4, transform_maps)

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
        self.trigger_velocities = {
            i: v for i, v in self.trigger_velocities.items() if contexts[i]
        }
        count = 0
        for action in ordered:
            if isinstance(action, instrument_trace.TriggerRelease):
                continue
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
                    if "velocity" in self.event_bindings.values():
                        self.trigger_velocities[context] = action.velocity
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
                for source, kind in self.event_bindings.items():
                    value = (
                        action.key
                        if kind == "key"
                        else self.trigger_velocities.get(trigger_context)
                    )
                    if value is None:
                        raise EngineError(
                            f"{kind.title()} binding requires resolved onset {kind}"
                        )
                    self._encode_action(count, offset, 14, slot, source, value, 0)
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
        self.trigger_velocities = {
            i: v for i, v in self.trigger_velocities.items() if contexts[i]
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
            trigger_velocities=self.trigger_velocities.copy(),
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
        self.trigger_velocities = snapshot.trigger_velocities.copy()

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
            np.array([[0, frames, 1, 0, 0, 1, 0, 0]], dtype=np.float64),
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
    if any(isinstance(s, ExponentialApproach) for s in envelope.segments):
        raise EngineError("Captured amplitude stages are supported only in release")
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
    if not envelope.hold:
        raise EngineError("Only held envelopes are implemented")


def validate_generators(
    settings: SoundSettings,
    allow_beat_envelfos: bool = False,
    allow_velocity_bindings: bool = False,
    allow_key_bindings: bool = False,
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
            isinstance(b, processing.EventBinding)
            and (
                (allow_velocity_bindings and b.kind == "velocity")
                or (allow_key_bindings and b.kind == "key")
            )
        )
        for b in settings.bindings
    ):
        raise EngineError("Unsupported binding for this renderer")


def validate_modulation(
    settings: SoundSettings, additional_targets: tuple[tuple[str, str], ...] = ()
) -> None:
    """Accept only targets realized by the common voice processing path."""
    targets = {f"filter-{f.name}" for f in settings.processing.filters}
    if any(
        not (
            (
                p.target.name == "processing"
                and p.target.parameter in ("amplitude", "tuning_cents")
            )
            or (p.target.name in targets and p.target.parameter in ("cutoff_hz", "q"))
            or (p.target.name, p.target.parameter) in additional_targets
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
    validate_generators(
        voice,
        allow_beat_envelfos=True,
        allow_velocity_bindings=True,
        allow_key_bindings=True,
    )
    if any(c.mode == "fade" for c in voice.chokes):
        raise EngineError("Fade retirement is not implemented")
    validate_modulation(voice)


def _envelope_values(
    initial: float,
    segments: list[Segment | IndefiniteSegment | ExponentialApproach],
    elapsed: Fraction,
    frames: int,
    sample_rate: int,
) -> np.ndarray:
    values = np.full(frames, initial)
    boundary = Fraction(0)
    entry = initial
    for segment in segments:
        if isinstance(segment, ExponentialApproach):
            raise EngineError("Captured envelope must be resolved before sampling")
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
            values[first:last] = entry + (segment.to - entry) * _curve_progress(
                progress, segment.curve
            )
        boundary, entry = end, segment.to
    values[max(0, ceil((boundary - elapsed) * sample_rate)) :] = entry
    return values


def _curve_progress(progress: np.ndarray, curve: float) -> np.ndarray:
    # The portable normalized exponential, evaluated in arrays with stable
    # arithmetic for both signs, including very large curve magnitudes.
    if curve == 0:
        return progress
    if curve > 0:
        return (
            np.exp(curve * (progress - 1))
            * np.expm1(-curve * progress)
            / np.expm1(-curve)
        )
    return np.expm1(curve * progress) / np.expm1(curve)


def _duration(
    segments: list[Segment | IndefiniteSegment | ExponentialApproach],
) -> Fraction:
    if any(isinstance(s, IndefiniteSegment) for s in segments):
        raise EngineError("An indefinite Envelope has no finite duration")
    return sum((s.duration for s in segments if isinstance(s, Segment)), Fraction(0))


def _configure_sample_hold_keys(
    runtime: _native.GeneratorBank,
    settings: processing.ModulationSettings,
    sources: dict[str, list[int]],
) -> None:
    definitions = [
        (
            i,
            envelfo_random.stream_key(0, name + ":sample-hold"),
        )
        for name, indices in sources.items()
        if isinstance(settings.envelfos[name], SampleHold)
        for i in indices
    ]
    runtime.set_sample_hold_keys(definitions)


PersistentModulation = tuple[
    np.ndarray,
    list[tuple[int, int]],
    list[float],
    dict[str, list[int]],
    list[str],
    list[str],
    np.ndarray,
    list[str],
    dict[str, list[int]],
    dict[str, list[int]],
    list[str],
    list[tuple[int, int, bool]],
    dict[str, list[int]],
    set[str],
    np.ndarray,
    list[tuple[str, int | None, int]],
    dict[str, list[int]],
    list[tuple[int, str, int, str, int, int, int, int, tuple[int, int]]],
    list[list[tuple[int, list[tuple[float, float]]]]],
    dict[int, str],
    str,
    list[tuple[str, str]],
    list[tuple[str, str, int]],
]


def _configure_generator_owners(
    generators: _native.GeneratorBank, prepared: PersistentModulation
) -> None:
    generators.set_envelfo_state_owners(
        _envelfo_state_owners(len(prepared[6]), prepared[8], prepared[9]),
        _envelfo_state_owners(len(prepared[10]), prepared[12]),
        _envelfo_state_owners(len(prepared[15]), prepared[16]),
    )
    for sources in (prepared[8], prepared[9], prepared[12], prepared[16]):
        for name, indices in sources.items():
            sources[name] = indices[:1]


def _prepare_native_generators(
    rate: int,
    voices: int,
    template: processing.ModulationSettings,
    prepared: PersistentModulation,
    tempo_map: TempoMap | None = None,
) -> tuple[
    _native.GeneratorBank,
    dict[tuple[str, str], int],
    list[tuple[int, list[tuple[float, float]]]],
]:
    if tempo_map is None and any(
        m.clock == "beats" for m in template.envelfos.values()
    ):
        raise EngineError("Beat-clock Envelfo requires a host tempo map")
    lfos, lfo_documents, lfo_sources, voice_lfo_sources = prepared[6:10]
    (
        envelope_documents,
        envelope_parameters,
        named_envelfo_sources,
    ) = prepared[10:13]
    staged_envelfos, staged_envelfo_sources, event_connections = prepared[15:18]
    generators = _native.GeneratorBank(
        rate,
        voices,
        len(prepared[2]) // 3,
        lfos,
        lfo_documents,
        envelope_documents,
        envelope_parameters,
    )
    _configure_sample_hold_keys(generators, template, named_envelfo_sources)
    if tempo_map is not None:
        generators.set_beat_clock(tempo_map.model_dump_json())
    generators.set_staged_envelfos(staged_envelfos, event_connections)
    _configure_generator_owners(generators, prepared)
    transform_nodes, transform_maps = generators.configure_patches(
        prepared[20],
        prepared[21],
        (
            [(n, v[0]) for n, v in (lfo_sources | voice_lfo_sources).items()],
            [(n, v[0]) for n, v in named_envelfo_sources.items()],
            [(n, v[0]) for n, v in staged_envelfo_sources.items()],
        ),
        prepared[22],
        [
            (n, v.signal_order)
            for n, v in template.envelfos.items()
            if isinstance(v, Patch)
        ],
    )
    return generators, dict(transform_nodes), transform_maps


def _persistent_modulation(
    sample_rate: int,
    declarations: dict[str, controls.ControlDeclaration],
    template: processing.ModulationSettings,
    source_parameters: list[
        tuple[tuple[str, str], int, modulation.Operation, float | None, float, float]
    ]
    | None = None,
    *,
    fm: _native.PreparedFm | None = None,
) -> PersistentModulation:
    if source_parameters is None:
        if not isinstance(template, SoundSettings):
            raise EngineError(
                "Effect modulation requires explicit parameter definitions"
            )
        source_parameters = [
            (
                ("processing", "amplitude"),
                0,
                modulation.Operation.multiply,
                None,
                0 if fm is not None else -1e300,
                1e300,
            ),
            (
                ("processing", "tuning_cents"),
                1,
                modulation.Operation.add,
                None,
                -120000,
                120000,
            ),
        ]
    targets = [(n, p, i, d, m, x) for (n, p), i, _, d, m, x in source_parameters]
    return _native.prepare_modulation(
        sample_rate,
        json.dumps(
            {
                "modulation": template.modulation.model_dump(mode="json"),
                "bindings": [b.model_dump(mode="json") for b in template.bindings],
                "controls": {
                    n: v.model_dump(mode="json") for n, v in declarations.items()
                },
                "event_connections": [
                    c.model_dump(mode="json") for c in template.event_connections
                ],
            }
        ),
        [(n, v.model_dump_json()) for n, v in template.envelfos.items()],
        targets,
        sound_settings=(
            json.dumps({"processing": template.processing.model_dump(mode="json")})
            if isinstance(template, SoundSettings)
            else None
        ),
        fm=fm,
    )


def _envelfo_state_owners(count: int, *source_maps: dict[str, list[int]]) -> list[int]:
    owners = list(range(count))
    for source_map in source_maps:
        for indices in source_map.values():
            for index in indices[1:]:
                owners[index] = indices[0]
    return owners


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
