"""Independent NumPy realization of uFor's prepared audio-effect graphs."""

from collections.abc import Callable
from math import pow
from typing import Literal

import numpy as np
from pydantic import BaseModel, ConfigDict, Field
from ufor import audio_effects

from . import filters
from .synth import EngineError


class ParameterRamp(BaseModel):
    at: int
    start: float
    target: float
    duration: int = Field(ge=0)

    def value(self, frame: int) -> float:
        if frame >= self.at + self.duration or self.duration == 0:
            return self.target
        return self.start + (self.target - self.start) * (
            (frame - self.at) / self.duration
        )

    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)


class GrainState(BaseModel):
    samples: list[list[float]]
    index: int = 0
    gain: float

    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)


class GranulatorState(BaseModel):
    history: list[list[float]] = Field(default_factory=list)
    history_start: int = 0
    phase: float = 0
    launch_counter: int = 0
    grains: list[GrainState] = Field(default_factory=list)
    frozen: bool = False
    frozen_history: list[list[float]] = Field(default_factory=list)
    frozen_start: int = 0

    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)


class ProcessorState(BaseModel):
    parameters: dict[str, ParameterRamp]
    bypass: ParameterRamp
    filter_states: list[filters.FilterState] = Field(default_factory=list)
    granulator: GranulatorState | None = None
    ended_at: int | None = None

    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)


class EffectSnapshot(BaseModel, frozen=True):
    definition: audio_effects.EffectGraph
    sample_rate: int
    frame: int
    processors: dict[str, ProcessorState]
    ended_inputs: dict[str, int]
    stopped_at: int | None


class PreparedEffects(BaseModel, frozen=True):
    definition: audio_effects.EffectGraph
    sample_rate: int = Field(gt=0)
    order: list[str]
    channels: list[str]


class OfflineEffects:
    """Process one graph with exact frame actions and explicit input endings."""

    def __init__(
        self,
        definition: PreparedEffects,
        backend: Literal["numpy", "native"] = "numpy",
    ) -> None:
        if backend not in ("numpy", "native"):
            raise EngineError(f"Unknown effect backend: {backend}")
        self.definition = definition.model_copy(deep=True)
        self.backend = backend
        self.frame = 0
        self.ended_inputs: dict[str, int] = {}
        self.stopped_at: int | None = None
        layouts = _processor_layouts(definition.definition)
        self.processors = {
            p.name: _initial_state(p, len(layouts[p.name]))
            for p in definition.definition.processors
        }

    def advance(
        self,
        inputs: dict[str, np.ndarray],
        actions: list[audio_effects.EffectAction],
        start: int,
        end: int,
    ) -> np.ndarray:
        """Process `[start, end)`; every input covers that exact frame interval."""
        if start != self.frame or end <= start:
            raise EngineError("Effect advance must be contiguous and nonempty")
        frames = end - start
        graph = self.definition.definition
        expected = {i.name: len(i.channels) for i in graph.inputs}
        if inputs.keys() != expected.keys():
            raise EngineError("Effect inputs do not match the prepared graph")
        source_audio: dict[str, np.ndarray] = {}
        for name, channels in expected.items():
            value = np.asarray(inputs[name], dtype=np.float64)
            if value.shape != (frames, channels) or not np.all(np.isfinite(value)):
                raise EngineError(f"Invalid effect input {name}")
            source_audio[name] = value
        if any(a.tick < start or a.tick >= end for a in actions):
            raise EngineError("Effect action is outside the render interval")
        if any(
            (b.tick, b.ordinal) <= (a.tick, a.ordinal)
            for a, b in zip(actions, actions[1:], strict=False)
        ):
            raise EngineError("Effect actions must be strictly ordered")
        action_index = 0
        output = np.zeros((frames, len(self.definition.channels)))
        try:
            if (
                self.backend == "native"
                and not any(
                    isinstance(
                        a, (audio_effects.InputEndAction, audio_effects.StopAction)
                    )
                    for a in actions
                )
                and not self.ended_inputs
                and _native_supported(self.processors, graph, actions)
            ):
                output = self._advance_native(source_audio, actions, start, end)
                self.frame = end
                return output
            for i in range(frames):
                frame = start + i
                while (
                    action_index < len(actions) and actions[action_index].tick == frame
                ):
                    self._apply(actions[action_index])
                    action_index += 1
                if self.stopped_at is not None:
                    continue
                audio, ended = self._process_frame(source_audio, i, frame)
                if ended:
                    self.stopped_at = frame
                else:
                    output[i] = audio
        except (ValueError, FloatingPointError) as error:
            self.stopped_at = start
            output.fill(0)
            raise EngineError(
                f"Effect processing failed in block at frame {start}"
            ) from error
        if not np.all(np.isfinite(output)):
            self.stopped_at = start
            output.fill(0)
            raise EngineError(f"Non-finite effect output in block at frame {start}")
        self.frame = end
        return output

    def _advance_native(
        self,
        input_audio: dict[str, np.ndarray],
        actions: list[audio_effects.EffectAction],
        start: int,
        end: int,
    ) -> np.ndarray:
        from . import _native

        graph = self.definition.definition
        processors = {p.name: p for p in graph.processors}
        ordered = [processors[n] for n in self.definition.order]
        if len(ordered) == 1 and isinstance(
            processor := ordered[0], audio_effects.Granulator
        ):
            return self._advance_native_granulator(
                processor, input_audio, actions, start, end
            )
        input_indices = {v.name: i for i, v in enumerate(graph.inputs)}
        processor_indices = {
            p.name: len(graph.inputs) + i for i, p in enumerate(ordered)
        }
        connections = {(c.processor, c.port): c.source for c in graph.connections}
        sources = np.full((len(ordered), 2), -1, dtype=np.int64)
        kinds: list[int] = []
        state_floors: list[float] = []
        for i, processor in enumerate(ordered):
            ports = sorted(audio_effects.processor_ports(processor))
            if isinstance(processor, audio_effects.Multiply):
                ports = ["carrier", "modulator"]
            for j, port in enumerate(ports):
                source = connections[(processor.name, port)]
                sources[i, j] = (
                    input_indices[source.input]
                    if isinstance(source, audio_effects.InputSource)
                    else processor_indices[source.processor]
                )
            kinds.append(
                0
                if isinstance(processor, audio_effects.Gain)
                else 1
                if isinstance(processor, audio_effects.Multiply)
                else 2
            )
            state_floors.append(
                processor.state_floor
                if isinstance(processor, audio_effects.Filter)
                else 0
            )
        frames = end - start
        values = np.zeros((frames, len(ordered), 2))
        filter_values: list[np.ndarray | None] = [None] * len(ordered)
        action_index = 0
        for i in range(frames):
            frame = start + i
            while action_index < len(actions) and actions[action_index].tick == frame:
                self._apply(actions[action_index])
                action_index += 1
            for j, processor in enumerate(ordered):
                state = self.processors[processor.name]
                values[i, j, 1] = state.parameters["mix"].value(
                    frame
                ) * state.bypass.value(frame)
                if isinstance(processor, audio_effects.Gain):
                    values[i, j, 0] = state.parameters["gain_db"].value(frame)
                elif isinstance(processor, audio_effects.Filter):
                    if (node_filter_values := filter_values[j]) is None:
                        node_filter_values = np.zeros(
                            (frames, len(processor.filters), 2)
                        )
                        filter_values[j] = node_filter_values
                    node_filter_values[i] = [
                        [
                            state.parameters[f"{f.name}-cutoff_hz"].value(frame),
                            state.parameters[f"{f.name}-q"].value(frame),
                        ]
                        for f in processor.filters
                    ]
        native_filters = []
        for i, processor in enumerate(ordered):
            if isinstance(processor, audio_effects.Filter):
                assert filter_values[i] is not None
                native_filters.append(
                    filters.native_inputs(
                        processor.filters,
                        self.processors[processor.name].filter_states,
                        filters.parameters(
                            processor.filters,
                            self.definition.sample_rate,
                            frames,
                            filter_values[i],
                            start,
                        ),
                    )
                )
            else:
                native_filters.append(None)
        output_source = (
            input_indices[graph.output.input]
            if isinstance(graph.output, audio_effects.InputSource)
            else processor_indices[graph.output.processor]
        )
        output, memories = _native.render_effects(
            np.stack([input_audio[v.name] for v in graph.inputs]),
            kinds,
            sources,
            values,
            output_source,
            self.definition.sample_rate,
            native_filters,
            state_floors,
        )
        for i, processor in enumerate(ordered):
            if isinstance(processor, audio_effects.Filter):
                self.processors[processor.name].filter_states = filters.restored_states(
                    memories[i]
                )
        return output

    def _advance_native_granulator(
        self,
        processor: audio_effects.Granulator,
        input_audio: dict[str, np.ndarray],
        actions: list[audio_effects.EffectAction],
        start: int,
        end: int,
    ) -> np.ndarray:
        from . import _native

        state = self.processors[processor.name]
        granular = state.granulator
        assert granular is not None
        frames = end - start
        values = np.zeros((frames, 6))
        action_index = 0
        names = [
            "duration_seconds",
            "density_hz",
            "lookback_seconds",
            "playback_ratio",
            "position_jitter_seconds",
        ]
        for i in range(frames):
            frame = start + i
            while action_index < len(actions) and actions[action_index].tick == frame:
                self._apply(actions[action_index])
                action_index += 1
            values[i, :5] = [state.parameters[n].value(frame) for n in names]
            values[i, 5] = state.parameters["mix"].value(frame) * state.bypass.value(
                frame
            )
        source = next(iter(input_audio.values()))
        channels = source.shape[1]
        rendered = _native.render_granulator(
            source,
            values,
            start,
            self.definition.sample_rate,
            max(2, round(processor.history_seconds * self.definition.sample_rate)),
            processor.maximum_grains,
            np.asarray(granular.history, dtype=np.float64).reshape(-1, channels),
            granular.history_start,
            granular.phase,
            granular.launch_counter,
            [np.asarray(g.samples, dtype=np.float64) for g in granular.grains],
            [g.index for g in granular.grains],
            [g.gain for g in granular.grains],
        )
        (
            output,
            history,
            granular.history_start,
            granular.phase,
            granular.launch_counter,
            grain_samples,
            grain_indices,
            grain_gains,
        ) = rendered
        granular.history = history.tolist()
        granular.grains = [
            GrainState(samples=v.tolist(), index=i, gain=g)
            for v, i, g in zip(grain_samples, grain_indices, grain_gains, strict=True)
        ]
        return output

    def snapshot(self) -> EffectSnapshot:
        return EffectSnapshot(
            definition=self.definition.definition,
            sample_rate=self.definition.sample_rate,
            frame=self.frame,
            processors=self.processors,
            ended_inputs=self.ended_inputs,
            stopped_at=self.stopped_at,
        ).model_copy(deep=True)

    def restore(self, snapshot: EffectSnapshot) -> None:
        if (
            snapshot.definition != self.definition.definition
            or snapshot.sample_rate != self.definition.sample_rate
        ):
            raise EngineError("Snapshot belongs to different prepared effects")
        snapshot = snapshot.model_copy(deep=True)
        self.frame = snapshot.frame
        self.processors = snapshot.processors
        self.ended_inputs = snapshot.ended_inputs
        self.stopped_at = snapshot.stopped_at

    @property
    def drained(self) -> bool:
        return self.stopped_at is not None

    def _apply(self, action: audio_effects.EffectAction) -> None:
        graph = self.definition.definition
        audio_effects.validate_action(graph, action)
        if isinstance(action, audio_effects.InputEndAction):
            if action.input in self.ended_inputs:
                raise ValueError(f"Effect input {action.input} already ended")
            self.ended_inputs[action.input] = action.tick
        elif isinstance(action, audio_effects.StopAction):
            self.stopped_at = action.tick
        elif isinstance(action, audio_effects.ParameterAction):
            state = self.processors[action.processor]
            current = state.parameters[action.parameter]
            state.parameters[action.parameter] = ParameterRamp(
                at=action.tick,
                start=current.value(action.tick),
                target=action.value,
                duration=action.duration_frames,
            )
        elif isinstance(action, audio_effects.BypassAction):
            state = self.processors[action.processor]
            processor = _processor(graph, action.processor)
            state.bypass = ParameterRamp(
                at=action.tick,
                start=state.bypass.value(action.tick),
                target=0.0 if action.bypassed else 1.0,
                duration=processor.bypass_fade_frames,
            )
        elif isinstance(action, audio_effects.FreezeAction):
            state = self.processors[action.processor]
            granular = state.granulator
            if granular is None:
                raise ValueError("Freeze requires the granular processor profile")
            if action.frozen and not granular.frozen:
                granular.frozen_history = [list(v) for v in granular.history]
                granular.frozen_start = granular.history_start
            elif not action.frozen:
                granular.frozen_history = []
            granular.frozen = action.frozen
        else:
            raise ValueError("Unknown effect action")

    def _process_frame(
        self, input_audio: dict[str, np.ndarray], index: int, frame: int
    ) -> tuple[np.ndarray, bool]:
        graph = self.definition.definition
        processors = {p.name: p for p in graph.processors}
        connections = {(c.processor, c.port): c.source for c in graph.connections}
        audio: dict[str, np.ndarray] = {}
        ended: dict[str, bool] = {}
        for input in graph.inputs:
            audio[input.name] = (
                np.zeros(len(input.channels))
                if input.name in self.ended_inputs
                else input_audio[input.name][index]
            )
            ended[input.name] = input.name in self.ended_inputs
        processor_audio: dict[str, np.ndarray] = {}
        processor_ended: dict[str, bool] = {}
        for name in self.definition.order:
            processor = processors[name]
            ports = {}
            port_ended = {}
            for port in audio_effects.processor_ports(processor):
                source = connections[(name, port)]
                if isinstance(source, audio_effects.InputSource):
                    ports[port] = audio[source.input]
                    port_ended[port] = ended[source.input]
                else:
                    ports[port] = processor_audio[source.processor]
                    port_ended[port] = processor_ended[source.processor]
            processor_audio[name], processor_ended[name] = self._processor_frame(
                processor, ports, port_ended, frame
            )
        if isinstance(graph.output, audio_effects.InputSource):
            return audio[graph.output.input], ended[graph.output.input]
        return (
            processor_audio[graph.output.processor],
            processor_ended[graph.output.processor],
        )

    def _processor_frame(
        self,
        processor: audio_effects.Processor,
        ports: dict[str, np.ndarray],
        ended: dict[str, bool],
        frame: int,
    ) -> tuple[np.ndarray, bool]:
        state = self.processors[processor.name]
        authored_mix = state.parameters["mix"].value(frame)
        mix = authored_mix * state.bypass.value(frame)
        if isinstance(processor, audio_effects.Gain):
            dry = ports["input"]
            wet = dry * pow(10, state.parameters["gain_db"].value(frame) / 20)
            return _mix(dry, wet, mix), ended["input"]
        if isinstance(processor, audio_effects.Multiply):
            dry = ports["carrier"]
            wet = dry * ports["modulator"]
            return _mix(dry, wet, mix), ended["carrier"]
        if isinstance(processor, audio_effects.Filter):
            dry = ports["input"]
            values = np.array(
                [
                    [
                        state.parameters[f"{f.name}-cutoff_hz"].value(frame),
                        state.parameters[f"{f.name}-q"].value(frame),
                    ]
                    for f in processor.filters
                ]
            )[None, :, :]
            wet, filter_states = filters.filter_samples(
                processor.filters,
                state.filter_states,
                dry[None, :],
                filters.parameters(
                    processor.filters,
                    self.definition.sample_rate,
                    1,
                    values,
                    frame,
                ),
                self.definition.sample_rate,
            )
            state.filter_states = [
                s.model_copy(
                    update={
                        "integrators": [
                            [0 if abs(v) < processor.state_floor else v for v in row]
                            for row in s.integrators
                        ]
                    }
                )
                for s in filter_states
            ]
            drained = ended["input"] and all(
                abs(v) < processor.tail_threshold
                for s in state.filter_states
                for row in s.integrators
                for v in row
            )
            return (
                np.zeros_like(dry) if drained else _mix(dry, wet[0], mix),
                drained,
            )
        if isinstance(processor, audio_effects.Granulator):
            dry = ports["input"]
            wet = self._granulator_frame(processor, state, dry, frame, ended["input"])
            granular = state.granulator
            assert granular is not None
            drained = ended["input"] and not granular.frozen and not granular.grains
            return (np.zeros_like(dry) if drained else _mix(dry, wet, mix), drained)
        raise ValueError("Unknown effect processor")

    def _granulator_frame(
        self,
        processor: audio_effects.Granulator,
        state: ProcessorState,
        dry: np.ndarray,
        frame: int,
        ended: bool,
    ) -> np.ndarray:
        granular = state.granulator
        assert granular is not None
        rate = self.definition.sample_rate
        if not granular.frozen and not ended:
            granular.history.append(dry.tolist())
            maximum = max(2, round(processor.history_seconds * rate))
            if len(granular.history) > maximum:
                granular.history.pop(0)
                granular.history_start += 1
        density = state.parameters["density_hz"].value(frame)
        launch, granular.phase = audio_effects.grain_launches(
            granular.phase, density, rate
        )
        if launch:
            counter = granular.launch_counter
            granular.launch_counter += 1
            if (not ended or granular.frozen) and len(
                granular.grains
            ) < processor.maximum_grains:
                history = (
                    granular.frozen_history if granular.frozen else granular.history
                )
                history_start = (
                    granular.frozen_start if granular.frozen else granular.history_start
                )
                duration = max(
                    2, round(state.parameters["duration_seconds"].value(frame) * rate)
                )
                ratio = state.parameters["playback_ratio"].value(frame)
                lookback = state.parameters["lookback_seconds"].value(frame) * rate
                jitter = (
                    state.parameters["position_jitter_seconds"].value(frame)
                    * rate
                    * audio_effects.grain_jitter(counter)
                )
                end_position = frame - lookback + jitter
                start_position = end_position - (duration - 1) * ratio
                captured = _capture_grain(
                    history, history_start, start_position, ratio, duration
                )
                if captured is not None:
                    granular.grains.append(
                        GrainState(
                            samples=captured,
                            gain=audio_effects.grain_gain(
                                density,
                                state.parameters["duration_seconds"].value(frame),
                            ),
                        )
                    )
        wet = np.zeros_like(dry)
        active: list[GrainState] = []
        for grain in granular.grains:
            wet += (
                np.asarray(grain.samples[grain.index])
                * audio_effects.grain_window(grain.index, len(grain.samples))
                * grain.gain
            )
            grain.index += 1
            if grain.index < len(grain.samples):
                active.append(grain)
        granular.grains = active
        return wet


class OfflineEffectAttachment:
    """Apply a one-input graph to any offline engine with `advance`."""

    def __init__(self, source: object, effects: OfflineEffects) -> None:
        graph = effects.definition.definition
        main = [
            i for i in graph.inputs if isinstance(i.source, audio_effects.MainStream)
        ]
        if len(graph.inputs) != 1 or len(main) != 1:
            raise EngineError("Offline attachment requires a main-only effect graph")
        self.source = source
        self.effects = effects
        self.input = main[0].name

    def advance(
        self,
        source_actions: list[object],
        effect_actions: list[audio_effects.EffectAction],
        start: int,
        end: int,
    ) -> np.ndarray:
        advance = getattr(self.source, "advance", None)
        if not isinstance(advance, Callable):
            raise EngineError("Attached source has no advance method")
        return self.effects.advance(
            {self.input: advance(source_actions, start, end)},
            effect_actions,
            start,
            end,
        )


def process_audio(
    definition: PreparedEffects,
    inputs: dict[str, np.ndarray],
    actions: list[audio_effects.EffectAction] | None = None,
    backend: Literal["numpy", "native"] = "numpy",
    block_frames: int | None = None,
) -> np.ndarray:
    """Process complete named streams through one prepared graph in blocks."""
    frames = {v.shape[0] for v in inputs.values()}
    if len(frames) != 1:
        raise EngineError("Effect input streams must have equal lengths")
    total = frames.pop()
    block_frames = block_frames or definition.definition.maximum_block_frames
    if block_frames <= 0 or block_frames > definition.definition.maximum_block_frames:
        raise EngineError("Effect block size exceeds its prepared maximum")
    renderer = OfflineEffects(definition, backend)
    actions = actions or []
    output = np.zeros((total, len(definition.channels)))
    cursor = 0
    for start in range(0, total, block_frames):
        end = min(total, start + block_frames)
        first = cursor
        while cursor < len(actions) and actions[cursor].tick < end:
            cursor += 1
        output[start:end] = renderer.advance(
            {n: v[start:end] for n, v in inputs.items()},
            actions[first:cursor],
            start,
            end,
        )
    return output


def prepare(definition: audio_effects.EffectGraph, sample_rate: int) -> PreparedEffects:
    if sample_rate <= 0:
        raise EngineError("Effect sample rate must be positive")
    for processor in definition.processors:
        if isinstance(processor, audio_effects.Filter):
            filters.parameters(processor.filters, sample_rate, 1)
    return PreparedEffects(
        definition=definition,
        sample_rate=sample_rate,
        order=definition.order(),
        channels=definition.output_channels(),
    )


def serial_graph(
    scope: audio_effects.AttachmentScope,
    input: audio_effects.GraphInput,
    processors: list[audio_effects.Processor],
    maximum_block_frames: int,
    owner: str | None = None,
) -> audio_effects.EffectGraph:
    """Normalize single-input processors into the canonical graph model."""
    connections: list[audio_effects.Connection] = []
    source: audio_effects.AudioSource = audio_effects.InputSource(input=input.name)
    for processor in processors:
        if audio_effects.processor_ports(processor) != {"input"}:
            raise EngineError("Serial shorthand requires one-input processors")
        connections.append(
            audio_effects.Connection(
                processor=processor.name, port="input", source=source
            )
        )
        source = audio_effects.ProcessorSource(processor=processor.name)
    return audio_effects.EffectGraph(
        scope=scope,
        owner=owner,
        inputs=[input],
        processors=processors,
        connections=connections,
        output=source,
        maximum_block_frames=maximum_block_frames,
    )


def _initial_state(processor: audio_effects.Processor, channels: int) -> ProcessorState:
    values = {"mix": processor.mix}
    filter_states: list[filters.FilterState] = []
    if isinstance(processor, audio_effects.Gain):
        values["gain_db"] = processor.gain_db
    elif isinstance(processor, audio_effects.Filter):
        filter_states = filters.initial_states(processor.filters, channels)
        for definition in processor.filters:
            values[f"{definition.name}-cutoff_hz"] = definition.cutoff_hz
            values[f"{definition.name}-q"] = definition.q
    elif isinstance(processor, audio_effects.Granulator):
        values.update(
            duration_seconds=processor.duration_seconds,
            density_hz=processor.density_hz,
            lookback_seconds=processor.lookback_seconds,
            playback_ratio=processor.playback_ratio,
            position_jitter_seconds=processor.position_jitter_seconds,
        )
    return ProcessorState(
        parameters={
            name: ParameterRamp(at=0, start=value, target=value, duration=0)
            for name, value in values.items()
        },
        bypass=ParameterRamp(at=0, start=1, target=1, duration=0),
        filter_states=filter_states,
        granulator=(
            GranulatorState()
            if isinstance(processor, audio_effects.Granulator)
            else None
        ),
    )


def _processor(graph: audio_effects.EffectGraph, name: str) -> audio_effects.Processor:
    return next(p for p in graph.processors if p.name == name)


def _mix(dry: np.ndarray, wet: np.ndarray, mix: float) -> np.ndarray:
    return (1 - mix) * dry + mix * wet


def _capture_grain(
    history: list[list[float]],
    history_start: int,
    start: float,
    ratio: float,
    frames: int,
) -> list[list[float]] | None:
    if not history:
        return None
    positions = start + np.arange(frames) * ratio
    lower = np.floor(positions).astype(np.int64)
    upper = lower + 1
    if lower[0] < history_start or upper[-1] >= history_start + len(history):
        return None
    values = np.asarray(history)
    fraction = positions - lower
    first = values[lower - history_start]
    second = values[upper - history_start]
    return (first + (second - first) * fraction[:, None]).tolist()


def _native_supported(
    states: dict[str, ProcessorState],
    graph: audio_effects.EffectGraph,
    actions: list[audio_effects.EffectAction],
) -> bool:
    granular = [p for p in graph.processors if isinstance(p, audio_effects.Granulator)]
    if not granular:
        return True
    if len(graph.processors) != 1 or any(
        isinstance(a, audio_effects.FreezeAction) for a in actions
    ):
        return False
    state = states[granular[0].name].granulator
    return state is not None and not state.frozen


def _processor_layouts(
    graph: audio_effects.EffectGraph,
) -> dict[str, list[str]]:
    inputs = {i.name: i for i in graph.inputs}
    processors = {p.name: p for p in graph.processors}
    connections = {(c.processor, c.port): c.source for c in graph.connections}
    layouts: dict[str, list[str]] = {}
    for name in graph.order():
        processor = processors[name]
        layouts[name] = audio_effects.processor_channels(
            processor,
            {
                port: audio_effects.source_channels(
                    connections[(name, port)], inputs, layouts
                )
                for port in audio_effects.processor_ports(processor)
            },
        )
    return layouts
