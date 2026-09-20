"""Independent NumPy realization of uFor's prepared audio-effect graphs."""

from collections.abc import Callable
from math import pow

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


class ProcessorState(BaseModel):
    parameters: dict[str, ParameterRamp]
    bypass: ParameterRamp
    filter_states: list[filters.FilterState] = Field(default_factory=list)
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

    def __init__(self, definition: PreparedEffects) -> None:
        self.definition = definition.model_copy(deep=True)
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
        else:
            raise ValueError("Freeze requires the granular processor profile")

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
        raise ValueError("Granulator is not implemented in this processing step")


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


def prepare(definition: audio_effects.EffectGraph, sample_rate: int) -> PreparedEffects:
    if sample_rate <= 0:
        raise EngineError("Effect sample rate must be positive")
    if any(isinstance(p, audio_effects.Granulator) for p in definition.processors):
        raise EngineError("Granulator is not implemented in this processing step")
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
    return ProcessorState(
        parameters={
            name: ParameterRamp(at=0, start=value, target=value, duration=0)
            for name, value in values.items()
        },
        bypass=ParameterRamp(at=0, start=1, target=1, duration=0),
        filter_states=filter_states,
    )


def _processor(graph: audio_effects.EffectGraph, name: str) -> audio_effects.Processor:
    return next(p for p in graph.processors if p.name == name)


def _mix(dry: np.ndarray, wet: np.ndarray, mix: float) -> np.ndarray:
    return (1 - mix) * dry + mix * wet


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
