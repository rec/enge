"""Independent NumPy realization of uFor's prepared audio-effect graphs."""

from collections.abc import Callable
from fractions import Fraction
from math import pow
from typing import Literal

import numpy as np
from envelfo import patch
from pydantic import BaseModel, ConfigDict, Field
from ufor import audio_effects, instrument_trace, modulation
from ufor.control import TempoMap
from ufor.oscillator import Waveform
from ufor.samples.processing import FilterResponse

from . import _native, filters, synth
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


class TapDelayState(BaseModel):
    history: list[list[float]]
    write: int = 0

    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)


class ModulatedDelayState(TapDelayState):
    phase: float = 0


class ProcessorState(BaseModel):
    parameters: dict[str, ParameterRamp]
    bypass: ParameterRamp
    filter_states: list[filters.FilterState] = Field(default_factory=list)
    granulator: GranulatorState | None = None
    tap_delay: TapDelayState | None = None
    modulated_delay: ModulatedDelayState | None = None
    ended_at: int | None = None

    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)


class EffectSnapshot(BaseModel, frozen=True):
    definition: audio_effects.EffectGraph
    sample_rate: int
    frame: int
    processors: dict[str, ProcessorState]
    ended_inputs: dict[str, int]
    stopped_at: int | None
    tempo_map: TempoMap | None = None
    lfos: list[synth.LFOSource] = Field(default_factory=list)
    envelopes: list[synth.EnvelopeSource] = Field(default_factory=list)
    patches: list[synth.PatchState] = Field(default_factory=list)
    graphs: dict[str, patch.PatchState] = Field(default_factory=dict)


class PreparedEffects(BaseModel, frozen=True):
    definition: audio_effects.EffectGraph
    sample_rate: int = Field(gt=0)
    order: list[str]
    channels: list[str]
    tempo_map: TempoMap | None = None


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
            p.name: _initial_state(p, len(layouts[p.name]), self.definition.sample_rate)
            for p in definition.definition.processors
        }
        self.controls = synth.ControlRenderer(
            definition.sample_rate,
            {},
            [self.definition.definition],
            tempo_map=definition.tempo_map,
        )
        self.sources = self.controls.sources(
            self.definition.definition,
            instrument_trace.VoiceStart(
                tick=0,
                ordinal=0,
                voice_id="effects",
                part="effects",
                trigger_id=None,
                template="effects",
                key=None,
            ),
        )
        self.modulated: dict[tuple[str, str], float] = {}

    def advance(
        self,
        inputs: dict[str, np.ndarray],
        actions: list[audio_effects.EffectAction],
        start: int,
        end: int,
    ) -> np.ndarray:
        """Process `[start, end)`; every input covers that exact frame interval."""
        if start != self.frame or end < start:
            raise EngineError("Effect advance must be contiguous")
        frames = end - start
        graph = self.definition.definition
        if frames > graph.maximum_block_frames:
            raise EngineError("Effect block exceeds its prepared maximum")
        expected = {i.name: len(i.channels) for i in graph.inputs}
        if inputs.keys() != expected.keys():
            raise EngineError("Effect inputs do not match the prepared graph")
        source_audio: dict[str, np.ndarray] = {}
        for name, channels in expected.items():
            value = np.asarray(inputs[name], dtype=np.float64)
            if value.shape != (frames, channels) or not np.all(np.isfinite(value)):
                raise EngineError(f"Invalid effect input {name}")
            source_audio[name] = value
        if frames == 0:
            if actions:
                raise EngineError("Zero-frame effect calls cannot contain actions")
            return np.zeros((0, len(self.definition.channels)))
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
                self._modulate(frame)
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
                if isinstance(processor, audio_effects.Filter)
                else 3
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
            self._modulate(frame)
            for j, processor in enumerate(ordered):
                state = self.processors[processor.name]
                values[i, j, 1] = self._value(
                    processor, "mix", frame
                ) * state.bypass.value(frame)
                if isinstance(processor, audio_effects.Gain):
                    values[i, j, 0] = self._value(processor, "gain_db", frame)
                elif isinstance(processor, audio_effects.SoftClip):
                    values[i, j, 0] = self._value(processor, "drive", frame)
                elif isinstance(processor, audio_effects.Filter):
                    if (node_filter_values := filter_values[j]) is None:
                        node_filter_values = np.zeros(
                            (frames, len(processor.filters), 2)
                        )
                        filter_values[j] = node_filter_values
                    node_filter_values[i] = [
                        [
                            self._value(processor, f"{f.name}-cutoff_hz", frame),
                            self._value(processor, f"{f.name}-q", frame),
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
        values = np.zeros((frames, 7))
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
            self._modulate(frame)
            values[i, :5] = [self._value(processor, n, frame) for n in names]
            values[i, 5] = self._value(processor, "mix", frame) * state.bypass.value(
                frame
            )
            values[i, 6] = granular.frozen
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
            processor.freeze_crossfade_frames,
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
        if granular.frozen:
            granular.frozen_history = granular.history
            granular.frozen_start = granular.history_start
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
            tempo_map=self.controls.tempo_map,
            lfos=self.controls.lfos,
            envelopes=self.controls.envelopes,
            patches=self.controls.patches,
            graphs=self.controls.graphs,
        ).model_copy(deep=True)

    def restore(self, snapshot: EffectSnapshot) -> None:
        if (
            snapshot.definition != self.definition.definition
            or snapshot.sample_rate != self.definition.sample_rate
            or snapshot.tempo_map != self.controls.tempo_map
        ):
            raise EngineError("Snapshot belongs to different prepared effects")
        snapshot = snapshot.model_copy(deep=True)
        self.frame = snapshot.frame
        self.processors = snapshot.processors
        self.ended_inputs = snapshot.ended_inputs
        self.stopped_at = snapshot.stopped_at
        self.controls.clear_cache()
        self.controls.lfos = snapshot.lfos
        self.controls.envelopes = snapshot.envelopes
        self.controls.patches = snapshot.patches
        self.controls.graphs = snapshot.graphs
        self.modulated = {}

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
            if isinstance(
                next(i.source for i in graph.inputs if i.name == action.input),
                audio_effects.MainStream,
            ):
                self.controls.release(
                    graph,
                    self.sources,
                    instrument_trace.VoiceRetirement(
                        tick=action.tick,
                        ordinal=action.ordinal + 1,
                        voice_id="effects",
                        cause=instrument_trace.RetirementCause.logical_release,
                        action="release",
                    ),
                    Fraction(action.tick, self.definition.sample_rate),
                )
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

    def _modulate(self, frame: int) -> None:
        graph = self.definition.definition
        if not graph.modulation.parameters:
            return
        values = self.controls.values(
            graph,
            self.sources,
            frame,
            1,
            {
                (p.target.name, p.target.parameter): np.array(
                    [
                        self.processors[p.target.name]
                        .parameters[p.target.parameter]
                        .value(frame)
                    ]
                )
                for p in graph.modulation.parameters
            },
        )
        self.modulated = {n: float(v[0]) for n, v in values.items()}

    def _value(
        self, processor: audio_effects.Processor, name: str, frame: int
    ) -> float:
        return self.modulated.get(
            (processor.name, name),
            self.processors[processor.name].parameters[name].value(frame),
        )

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
        authored_mix = self._value(processor, "mix", frame)
        mix = authored_mix * state.bypass.value(frame)
        if isinstance(processor, audio_effects.Gain):
            dry = ports["input"]
            wet = dry * pow(10, self._value(processor, "gain_db", frame) / 20)
            return _mix(dry, wet, mix), ended["input"]
        if isinstance(processor, audio_effects.SoftClip):
            dry = ports["input"]
            drive = self._value(processor, "drive", frame)
            wet = np.tanh(drive * dry) / np.tanh(drive)
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
                        self._value(processor, f"{f.name}-cutoff_hz", frame),
                        self._value(processor, f"{f.name}-q", frame),
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
        if isinstance(processor, audio_effects.TapDelay):
            dry = ports["input"]
            delay = state.tap_delay
            assert delay is not None
            frames = len(delay.history)
            delay_frames = max(
                1,
                round(
                    self._value(processor, "delay_seconds", frame)
                    * self.definition.sample_rate
                ),
            )
            wet = np.asarray(delay.history[(delay.write - delay_frames) % frames])
            feedback = self._value(processor, "feedback", frame)
            value = dry if not ended["input"] else np.zeros_like(dry)
            history = value + feedback * wet
            history[np.abs(history) < processor.state_floor] = 0
            delay.history[delay.write] = history.tolist()
            delay.write = (delay.write + 1) % frames
            drained = ended["input"] and all(
                abs(v) < processor.tail_threshold for row in delay.history for v in row
            )
            return (np.zeros_like(dry) if drained else _mix(dry, wet, mix), drained)
        if isinstance(processor, audio_effects.ModulatedDelay):
            dry = ports["input"]
            delay = state.modulated_delay
            assert delay is not None
            frames = len(delay.history)
            phase_offsets = _channel_phase_offsets(processor, len(dry))
            phase = (delay.phase + phase_offsets) % 1
            lfo = _delay_waveform(processor.waveform, phase)
            delays = np.maximum(
                1,
                np.rint(
                    (
                        self._value(processor, "base_delay_seconds", frame)
                        + self._value(processor, "depth_seconds", frame) * lfo
                    )
                    * self.definition.sample_rate
                ).astype(np.int64),
            )
            wet = np.asarray(
                [
                    delay.history[(delay.write - value) % frames][channel]
                    for channel, value in enumerate(delays)
                ]
            )
            feedback = self._value(processor, "feedback", frame)
            value = dry if not ended["input"] else np.zeros_like(dry)
            history = value + feedback * wet
            history[np.abs(history) < processor.state_floor] = 0
            delay.history[delay.write] = history.tolist()
            delay.write = (delay.write + 1) % frames
            delay.phase = (
                delay.phase
                + self._value(processor, "rate_hz", frame) / self.definition.sample_rate
            ) % 1
            drained = ended["input"] and all(
                abs(v) < processor.tail_threshold for row in delay.history for v in row
            )
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
        density = self._value(processor, "density_hz", frame)
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
                    2, round(self._value(processor, "duration_seconds", frame) * rate)
                )
                ratio = self._value(processor, "playback_ratio", frame)
                lookback = self._value(processor, "lookback_seconds", frame) * rate
                jitter = (
                    self._value(processor, "position_jitter_seconds", frame)
                    * rate
                    * audio_effects.grain_jitter(counter)
                )
                end_position = frame - lookback + jitter
                start_position = end_position - (duration - 1) * ratio
                captured = _capture_grain(
                    history,
                    history_start,
                    start_position,
                    ratio,
                    duration,
                    processor.freeze_crossfade_frames if granular.frozen else None,
                )
                if captured is not None:
                    granular.grains.append(
                        GrainState(
                            samples=captured,
                            gain=audio_effects.grain_gain(
                                density,
                                self._value(processor, "duration_seconds", frame),
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


def prepare(
    definition: audio_effects.EffectGraph,
    sample_rate: int,
    tempo_map: TempoMap | None = None,
) -> PreparedEffects:
    if sample_rate <= 0:
        raise EngineError("Effect sample rate must be positive")
    definition = synth.expand_patch_outputs(definition)
    if tempo_map is None and any(
        g.clock == "beats" for g in definition.envelfos.values()
    ):
        raise EngineError("Beat-clock Envelfo requires a host tempo map")
    for processor in definition.processors:
        if isinstance(processor, audio_effects.Filter):
            filters.parameters(processor.filters, sample_rate, 1)
        elif isinstance(processor, audio_effects.ModulatedDelay):
            _channel_phase_offsets(processor, len(definition.output_channels()))
    return PreparedEffects(
        definition=definition,
        sample_rate=sample_rate,
        order=definition.order(),
        channels=definition.output_channels(),
        tempo_map=tempo_map,
    )


def native_live_graph(
    definition: PreparedEffects,
) -> tuple[
    list[int],
    np.ndarray,
    np.ndarray,
    int,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
]:
    """Encode a supported prepared graph for the allocation-free live owner."""
    graph = definition.definition
    if len(graph.inputs) != 1:
        raise EngineError("Native live effects require exactly one graph input")
    processors = {p.name: p for p in graph.processors}
    ordered = [processors[n] for n in definition.order]
    input_name = graph.inputs[0].name
    processor_indices = {p.name: i + 1 for i, p in enumerate(ordered)}
    connections = {(c.processor, c.port): c.source for c in graph.connections}
    widths = [
        3 + 2 * len(p.filters)
        if isinstance(p, audio_effects.Filter)
        else 8
        if isinstance(p, audio_effects.Granulator)
        else 4
        if isinstance(p, audio_effects.TapDelay)
        else 6
        if isinstance(p, audio_effects.ModulatedDelay)
        else 3
        for p in ordered
    ]
    parameters = np.zeros((len(ordered), max(widths, default=3)), dtype=np.float64)
    sources = np.full((len(ordered), 2), -1, dtype=np.int64)
    kinds: list[int] = []
    filter_rows: list[list[float]] = []
    granulator_rows: list[list[int]] = []
    tap_delay_rows: list[list[int]] = []
    modulated_delay_rows: list[list[float]] = []
    for node, processor in enumerate(ordered):
        ports = (
            ["carrier", "modulator"]
            if isinstance(processor, audio_effects.Multiply)
            else ["input"]
        )
        for port, name in enumerate(ports):
            source = connections[(processor.name, name)]
            if isinstance(source, audio_effects.InputSource):
                if source.input != input_name:
                    raise EngineError("Native live graph references an unknown input")
                sources[node, port] = 0
            else:
                sources[node, port] = processor_indices[source.processor]
        parameters[node, 1:3] = [processor.mix, 1]
        if isinstance(processor, audio_effects.Gain):
            kinds.append(0)
            parameters[node, 0] = processor.gain_db
        elif isinstance(processor, audio_effects.Multiply):
            kinds.append(1)
        elif isinstance(processor, audio_effects.SoftClip):
            kinds.append(4)
            parameters[node, 0] = processor.drive
        elif isinstance(processor, audio_effects.Filter):
            kinds.append(2)
            values = filters.parameters(processor.filters, definition.sample_rate, 1)[0]
            for index, (filter_definition, value) in enumerate(
                zip(processor.filters, values, strict=True)
            ):
                parameter = 3 + 2 * index
                parameters[node, parameter : parameter + 2] = value
                response = list(FilterResponse).index(filter_definition.response)
                filter_rows.extend(
                    [node, response, parameter, processor.state_floor]
                    for _ in range(filter_definition.stages)
                )
        elif isinstance(processor, audio_effects.Granulator):
            kinds.append(3)
            parameters[node, 3:8] = [
                processor.duration_seconds,
                processor.density_hz,
                processor.lookback_seconds,
                processor.playback_ratio,
                processor.position_jitter_seconds,
            ]
            granulator_rows.append(
                [
                    node,
                    max(2, round(processor.history_seconds * definition.sample_rate)),
                    processor.maximum_grains,
                    processor.freeze_crossfade_frames,
                ]
            )
        elif isinstance(processor, audio_effects.TapDelay):
            kinds.append(5)
            parameters[node, [0, 3]] = [processor.delay_seconds, processor.feedback]
            tap_delay_rows.append(
                [
                    node,
                    max(
                        1,
                        round(processor.maximum_delay_seconds * definition.sample_rate),
                    ),
                ]
            )
        elif isinstance(processor, audio_effects.ModulatedDelay):
            kinds.append(6)
            parameters[node, [0, 3, 4, 5]] = [
                processor.base_delay_seconds,
                processor.depth_seconds,
                processor.rate_hz,
                processor.feedback,
            ]
            phases = _channel_phase_offsets(processor, len(definition.channels))
            modulated_delay_rows.append(
                [
                    node,
                    max(
                        1,
                        round(processor.maximum_delay_seconds * definition.sample_rate),
                    ),
                    list(Waveform).index(processor.waveform),
                    *phases,
                ]
            )
        else:
            raise EngineError("Unsupported native live effect processor")
    output_source = (
        0
        if isinstance(graph.output, audio_effects.InputSource)
        else processor_indices[graph.output.processor]
    )
    return (
        kinds,
        sources,
        parameters,
        output_source,
        np.asarray(filter_rows, dtype=np.float64).reshape(-1, 4),
        np.asarray(granulator_rows, dtype=np.int64).reshape(-1, 4),
        np.asarray(tap_delay_rows, dtype=np.int64).reshape(-1, 2),
        np.asarray(modulated_delay_rows, dtype=np.float64).reshape(
            -1, len(definition.channels) + 3
        ),
    )


def configure_native_generators(
    runtime: _native.LiveRuntime, definition: PreparedEffects
) -> None:
    """Prepare the shared Rust generator executor for one effect instance."""
    graph = definition.definition
    if not graph.bindings:
        return
    parameters = graph.modulation.parameters
    for processor in graph.processors:
        if isinstance(processor, audio_effects.Filter):
            for body in processor.filters:
                for parameter in parameters:
                    if (parameter.target.name, parameter.target.parameter) == (
                        processor.name,
                        f"{body.name}-cutoff_hz",
                    ) and not (
                        body.minimum_hz <= parameter.minimum
                        and parameter.maximum
                        <= definition.sample_rate / 2 * body.nyquist_ratio
                    ):
                        raise EngineError(
                            "Native effect cutoff modulation must remain "
                            "within filter limits"
                        )
    indices = {(p.target.name, p.target.parameter): i for i, p in enumerate(parameters)}
    prepared = synth._persistent_modulation(
        definition.sample_rate,
        {},
        graph,
        [
            (
                (p.target.name, p.target.parameter),
                i,
                modulation.Operation.add,
                p.default,
                p.minimum,
                p.maximum,
            )
            for i, p in enumerate(parameters)
        ],
    )
    bank, _, transforms = synth._prepare_native_generators(
        definition.sample_rate,
        1,
        graph,
        prepared,
        definition.tempo_map,
        indices,
    )
    processors = {p.name: p for p in graph.processors}
    nodes = {n: i for i, n in enumerate(definition.order)}
    runtime.set_effect_generators(
        bank,
        [*prepared[20][1:], transforms],
        [
            (
                nodes[p.target.name],
                _native_parameter(processors[p.target.name], p.target.parameter),
                p.minimum,
                p.maximum,
            )
            for p in parameters
        ],
    )


def native_live_actions(
    definition: PreparedEffects,
    actions: list[audio_effects.EffectAction],
    start: int,
    end: int,
) -> np.ndarray:
    """Encode one ordered block of portable effect actions for `LiveRuntime`."""
    if end <= start:
        raise EngineError("Native live effect interval must be nonempty")
    ordered_actions = sorted(actions, key=lambda a: (a.tick, a.ordinal))
    if any(a.tick < start or a.tick >= end for a in ordered_actions):
        raise EngineError("Effect action is outside the native live interval")
    graph = definition.definition
    processors = {p.name: p for p in graph.processors}
    nodes = {name: i for i, name in enumerate(definition.order)}
    rows: list[list[float]] = []
    for action in ordered_actions:
        audio_effects.validate_action(graph, action)
        if isinstance(action, audio_effects.ParameterAction):
            processor = processors[action.processor]
            parameter = _native_parameter(processor, action.parameter)
            target = action.value
            duration = action.duration_frames
        elif isinstance(action, audio_effects.BypassAction):
            processor = processors[action.processor]
            parameter = 2
            target = 0.0 if action.bypassed else 1.0
            duration = processor.bypass_fade_frames
        elif isinstance(action, audio_effects.FreezeAction) and isinstance(
            processors[action.processor], audio_effects.Granulator
        ):
            rows.append(
                [
                    action.tick - start,
                    1,
                    nodes[action.processor],
                    0,
                    int(action.frozen),
                    0,
                ]
            )
            continue
        else:
            raise EngineError("Unsupported native live effect action")
        rows.append(
            [
                action.tick - start,
                0,
                nodes[action.processor],
                parameter,
                target,
                duration,
            ]
        )
    return np.asarray(rows, dtype=np.float64).reshape(-1, 6)


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


def _initial_state(
    processor: audio_effects.Processor, channels: int, sample_rate: int
) -> ProcessorState:
    values = {"mix": processor.mix}
    filter_states: list[filters.FilterState] = []
    if isinstance(processor, audio_effects.Gain):
        values["gain_db"] = processor.gain_db
    elif isinstance(processor, audio_effects.SoftClip):
        values["drive"] = processor.drive
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
    elif isinstance(processor, audio_effects.TapDelay):
        values.update(
            delay_seconds=processor.delay_seconds,
            feedback=processor.feedback,
        )
    elif isinstance(processor, audio_effects.ModulatedDelay):
        values.update(
            base_delay_seconds=processor.base_delay_seconds,
            depth_seconds=processor.depth_seconds,
            rate_hz=processor.rate_hz,
            feedback=processor.feedback,
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
        tap_delay=(
            TapDelayState(
                history=[
                    [0.0] * channels
                    for _ in range(
                        max(1, round(processor.maximum_delay_seconds * sample_rate))
                    )
                ]
            )
            if isinstance(processor, audio_effects.TapDelay)
            else None
        ),
        modulated_delay=(
            ModulatedDelayState(
                history=[
                    [0.0] * channels
                    for _ in range(
                        max(1, round(processor.maximum_delay_seconds * sample_rate))
                    )
                ]
            )
            if isinstance(processor, audio_effects.ModulatedDelay)
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
    freeze_crossfade_frames: int | None = None,
) -> list[list[float]] | None:
    if not history:
        return None
    positions = start + np.arange(frames) * ratio
    values = np.asarray(history)
    if freeze_crossfade_frames is not None:
        crossfade = min(freeze_crossfade_frames, (len(history) - 1) // 2)
        period = len(history) - crossfade
        offsets = np.remainder(positions - history_start, period)
        lower = np.floor(offsets).astype(np.int64)
        upper = (lower + 1) % period
        fraction = offsets - lower
        first = _frozen_history_values(values, lower, period, crossfade)
        second = _frozen_history_values(values, upper, period, crossfade)
        return (first + (second - first) * fraction[:, None]).tolist()
    lower = np.floor(positions).astype(np.int64)
    upper = lower + 1
    if lower[0] < history_start or upper[-1] >= history_start + len(history):
        return None
    fraction = positions - lower
    first = values[lower - history_start]
    second = values[upper - history_start]
    return (first + (second - first) * fraction[:, None]).tolist()


def _frozen_history_values(
    history: np.ndarray, offsets: np.ndarray, period: int, crossfade: int
) -> np.ndarray:
    output = history[offsets].copy()
    mask = offsets < crossfade
    if np.any(mask):
        weight = offsets[mask] / crossfade
        output[mask] = (1 - weight[:, None]) * history[period + offsets[mask]] + weight[
            :, None
        ] * history[offsets[mask]]
    return output


def _native_supported(
    states: dict[str, ProcessorState],
    graph: audio_effects.EffectGraph,
    actions: list[audio_effects.EffectAction],
) -> bool:
    granular = [p for p in graph.processors if isinstance(p, audio_effects.Granulator)]
    if any(
        isinstance(p, (audio_effects.TapDelay, audio_effects.ModulatedDelay))
        for p in graph.processors
    ):
        return False
    if not granular:
        return True
    if len(graph.processors) != 1:
        return False
    state = states[granular[0].name].granulator
    return state is not None


def _channel_phase_offsets(
    processor: audio_effects.ModulatedDelay, channels: int
) -> np.ndarray:
    if not processor.channel_phase_offsets:
        return np.zeros(channels)
    if len(processor.channel_phase_offsets) != channels:
        raise EngineError("Modulated delay phase offsets must match output channels")
    return np.asarray(processor.channel_phase_offsets)


def _native_parameter(processor: audio_effects.Processor, name: str) -> int:
    if name == "mix":
        return 1
    if isinstance(processor, (audio_effects.Gain, audio_effects.SoftClip)):
        return 0
    if isinstance(processor, audio_effects.Filter):
        return 3 + [
            n for f in processor.filters for n in (f"{f.name}-cutoff_hz", f"{f.name}-q")
        ].index(name)
    if isinstance(processor, audio_effects.Granulator):
        return 3 + [
            "duration_seconds",
            "density_hz",
            "lookback_seconds",
            "playback_ratio",
            "position_jitter_seconds",
        ].index(name)
    if isinstance(processor, audio_effects.TapDelay):
        return {"delay_seconds": 0, "feedback": 3}[name]
    if isinstance(processor, audio_effects.ModulatedDelay):
        return {
            "base_delay_seconds": 0,
            "depth_seconds": 3,
            "rate_hz": 4,
            "feedback": 5,
        }[name]
    raise EngineError("Unsupported native live effect parameter")


def _delay_waveform(waveform: Waveform, phase: np.ndarray) -> np.ndarray:
    if waveform == Waveform.sine:
        return np.sin(2 * np.pi * phase)
    if waveform == Waveform.square:
        return np.where(phase < 0.5, 1.0, -1.0)
    return np.where(phase < 0.5, 4 * phase - 1, 3 - 4 * phase)


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
