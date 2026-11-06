"""Prepared effect graphs with independent NumPy and persistent Rust rendering."""

from collections.abc import Callable
from fractions import Fraction
from math import ceil, exp, floor, log10, pow, sqrt
from typing import Literal

import numpy as np
from envelfo import patch
from pydantic import BaseModel, ConfigDict, Field
from ufor import audio_effects, instrument_trace, modulation
from ufor.control import TempoMap
from ufor.oscillator import Waveform
from ufor.samples.processing import FilterBoundary, FilterResponse

from . import _native, filters, phaser, synth, waveshaper
from .phaser import PhaserState
from .synth import EngineError
from .waveshaper import WaveshaperState


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


class DelayState(BaseModel):
    history: list[list[float]]
    write: int = 0

    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)


class DelayHead(BaseModel):
    delay_frames: float = 1.0
    gain: float = 0.0
    step: float = 0.0
    remaining: int = 0
    active: bool = False

    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)


class TapDelayState(DelayState):
    heads: list[DelayHead]
    current: int = 0


class ModulatedDelayState(DelayState):
    phase: float = 0


class AlignmentState(DelayState):
    remaining: int = Field(ge=0)


class ControlDelayState(BaseModel):
    history: list[float]
    write: int = 0

    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)


class ReverbState(BaseModel):
    predelay: TapDelayState
    lines: list[list[float]]
    writes: list[int]
    damping: list[float]
    non_silent: int = Field(default=0, ge=0)

    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)


class ProcessorState(BaseModel):
    parameters: dict[str, ParameterRamp]
    bypass: ParameterRamp
    filter_states: list[filters.FilterState] = Field(default_factory=list)
    granulator: GranulatorState | None = None
    tap_delay: TapDelayState | None = None
    modulated_delay: ModulatedDelayState | None = None
    reverb: ReverbState | None = None
    waveshaper: WaveshaperState | None = None
    phaser: PhaserState | None = None
    alignment: dict[str, AlignmentState] = Field(default_factory=dict)
    control_delays: dict[str, ControlDelayState] = Field(default_factory=dict)
    freeze_target: bool = False
    reduction_db: float = 0.0
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
    latency_frames: int = Field(ge=0)
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
        latencies = definition.definition.latencies()
        inputs = {i.name: i.channels for i in definition.definition.inputs}
        connections = {
            (c.processor, c.port): c.source for c in definition.definition.connections
        }
        for processor in definition.definition.processors:
            ports = {
                p: connections[(processor.name, p)]
                for p in audio_effects.processor_ports(processor)
            }
            delays = {
                p: 0
                if isinstance(s, audio_effects.InputSource)
                else latencies[s.processor]
                for p, s in ports.items()
            }
            arrival = max(delays.values())
            self.processors[processor.name].alignment = {
                p: AlignmentState(
                    history=[
                        [0.0]
                        * len(
                            inputs[s.input]
                            if isinstance(s, audio_effects.InputSource)
                            else layouts[s.processor]
                        )
                        for _ in range(arrival - delays[p])
                    ],
                    remaining=arrival - delays[p],
                )
                for p, s in ports.items()
                if delays[p] < arrival
            }
            state = self.processors[processor.name]
            commands = [("freeze", 0.0)] if state.granulator is not None else []
            for name, value in (
                [(n, r.value(0)) for n, r in state.parameters.items()]
                + [("bypass", 1.0)]
                + commands
            ):
                index = 2 if name == "bypass" else _native_parameter(processor, name)
                frames = (
                    arrival + (24 if name in {"drive", "bias"} else 48)
                    if isinstance(processor, audio_effects.Waveshaper)
                    else arrival
                )
                if frames:
                    state.control_delays[str(index)] = ControlDelayState(
                        history=[value] * frames
                    )
        self.waveshaper_coefficients = np.asarray(waveshaper.halfband_coefficients())
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
        self.delayed: dict[tuple[str, str], float] = {}
        self.native_runtime: _native.EffectRuntime | None = None
        self.native_parameters: np.ndarray | None = None
        if backend == "native":
            encoded = native_live_graph(self.definition)
            self.native_runtime = _native.EffectRuntime(
                *encoded, definition.sample_rate
            )
            self.native_parameters = encoded[2]

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
            self._sync_native()
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
        self._sync_native()
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
        if self.native_runtime is not None:
            self.native_runtime.import_state(
                [snapshot.processors[n].model_dump() for n in self.definition.order],
                snapshot.frame,
            )
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
        self.delayed = {}

    @property
    def drained(self) -> bool:
        return self.stopped_at is not None

    def _sync_native(self) -> None:
        if self.native_runtime is not None:
            self.processors = {
                n: ProcessorState.model_validate(self.processors[n].model_dump() | s)
                for n, s in zip(
                    self.definition.order,
                    self.native_runtime.export_state(),
                    strict=True,
                )
            }

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
            state.freeze_target = action.frozen
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
        return self.delayed.get(
            (processor.name, name),
            self.modulated.get(
                (processor.name, name),
                self.processors[processor.name].parameters[name].value(frame),
            ),
        )

    def _process_frame(
        self, input_audio: dict[str, np.ndarray], index: int, frame: int
    ) -> tuple[np.ndarray, bool]:
        graph = self.definition.definition
        processors = {p.name: p for p in graph.processors}
        self.delayed.clear()
        if self.native_runtime is not None:
            assert self.native_parameters is not None
            for node, name in enumerate(self.definition.order):
                processor = processors[name]
                state = self.processors[name]
                for parameter in state.parameters:
                    self.native_parameters[
                        node, _native_parameter(processor, parameter)
                    ] = self._value(processor, parameter, frame)
                self.native_parameters[node, 2] = state.bypass.value(frame)
            values, drained = self.native_runtime.render_frame(
                [input_audio[i.name][index] for i in graph.inputs],
                self.native_parameters,
                [i.name in self.ended_inputs for i in graph.inputs],
                [self.processors[n].freeze_target for n in self.definition.order],
            )
            return np.asarray(values, dtype=np.float64), drained
        for name, state in self.processors.items():
            processor = processors[name]
            parameters = [*state.parameters, "bypass"]
            if state.granulator is not None:
                parameters.append("freeze")
            for parameter in parameters:
                parameter_index = (
                    2
                    if parameter == "bypass"
                    else _native_parameter(processor, parameter)
                )
                if (
                    delay := state.control_delays.get(str(parameter_index))
                ) is not None:
                    value = (
                        state.bypass.value(frame)
                        if parameter == "bypass"
                        else float(state.freeze_target)
                        if parameter == "freeze"
                        else self._value(processor, parameter, frame)
                    )
                    self.delayed[(name, parameter)] = delay.history[delay.write]
                    delay.history[delay.write] = value
                    delay.write = (delay.write + 1) % len(delay.history)
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
                if (delay := self.processors[name].alignment.get(port)) is not None:
                    value = ports[port]
                    ports[port] = np.asarray(delay.history[delay.write])
                    delay.history[delay.write] = value.tolist()
                    delay.write = (delay.write + 1) % len(delay.history)
                    incoming_ended = port_ended[port]
                    port_ended[port] = incoming_ended and delay.remaining == 0
                    if incoming_ended:
                        delay.remaining = max(0, delay.remaining - 1)
                    else:
                        delay.remaining = len(delay.history)
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
        mix = authored_mix * self.delayed.get(
            (processor.name, "bypass"), state.bypass.value(frame)
        )
        if isinstance(processor, audio_effects.Gain):
            dry = ports["input"]
            wet = dry * pow(10, self._value(processor, "gain_db", frame) / 20)
            return _mix(dry, wet, mix), ended["input"]
        if isinstance(processor, audio_effects.Waveshaper):
            assert state.waveshaper is not None
            dry, wet, drained = waveshaper.process(
                state.waveshaper,
                ports["input"],
                self._value(processor, "drive", frame),
                self._value(processor, "bias", frame),
                self._value(processor, "output_gain_db", frame),
                processor.curve == "hard",
                self.waveshaper_coefficients,
                ended["input"],
            )
            return (np.zeros_like(dry) if drained else _mix(dry, wet, mix), drained)
        if isinstance(processor, audio_effects.Phaser):
            dry = ports["input"]
            assert state.phaser is not None
            a = phaser.coefficient(
                self._value(processor, "center_hz", frame),
                self._value(processor, "depth_cents", frame),
                self._value(processor, "sweep", frame),
                self.definition.sample_rate,
            )
            wet = phaser.process(
                state.phaser,
                dry,
                a,
                self._value(processor, "feedback", frame),
                processor.state_floor,
            )
            drained = ended["input"] and all(
                abs(v) < processor.tail_threshold
                for s in [*state.phaser.stages, state.phaser.feedback]
                for v in s
            )
            return (np.zeros_like(dry) if drained else _mix(dry, wet, mix), drained)
        if isinstance(processor, audio_effects.Multiply):
            dry = ports["carrier"]
            wet = dry * ports["modulator"]
            return _mix(dry, wet, mix), ended["carrier"]
        if isinstance(processor, audio_effects.Compressor):
            dry = ports["input"]
            peak = float(np.max(np.abs(ports["detector"])))
            parameters = [
                self._value(processor, p, frame)
                for p in (
                    "threshold_db",
                    "ratio",
                    "knee_db",
                    "attack_seconds",
                    "release_seconds",
                    "makeup_db",
                )
            ]
            threshold, ratio, knee, attack, release, makeup = parameters
            x = 20 * log10(peak) - threshold if peak else -np.inf
            if x <= -knee / 2:
                target = 0.0
            elif x >= knee / 2:
                target = (1 / ratio - 1) * x
            else:
                target = (0.5 * (1 / ratio - 1) * knee) * (x / knee + 0.5) ** 2
            time = attack if target < state.reduction_db else release
            reduction = (
                target
                if time == 0
                else target
                + exp(-1 / (self.definition.sample_rate * time))
                * (state.reduction_db - target)
            )
            gain = float(np.power(10, (reduction + makeup) / 20))
            if not np.isfinite(gain) or not np.isfinite(reduction):
                raise ValueError("Non-finite compressor gain")
            state.reduction_db = reduction
            return _mix(dry, dry * gain, mix), ended["input"]
        if isinstance(processor, audio_effects.Reverb):
            dry = ports["input"]
            if len(dry) == 1:
                dry = np.repeat(dry / sqrt(2), 2)
            reverb = state.reverb
            assert reverb is not None
            predelay = reverb.predelay
            reverb.non_silent -= sum(
                abs(v) >= processor.tail_threshold
                for v in predelay.history[predelay.write]
            )
            reverb.non_silent += sum(
                int(abs(v) >= processor.tail_threshold) for v in dry
            )
            predelay.history[predelay.write] = dry.tolist()
            _retime_delay(
                predelay,
                self._value(processor, "predelay_seconds", frame)
                * self.definition.sample_rate,
                processor.retime_fade_frames,
            )
            delayed = _tap_read(predelay, 2)
            predelay.write = (predelay.write + 1) % len(predelay.history)
            _advance_heads(predelay)
            a, gains = _reverb_coefficients(
                self._value(processor, "decay_seconds", frame),
                self._value(processor, "damping_hz", frame),
                self.definition.sample_rate,
                [len(r) for r in reverb.lines],
            )
            values = [r[w] for r, w in zip(reverb.lines, reverb.writes, strict=True)]
            scale = 1 / sqrt(8)
            wet = np.array(
                [
                    sum((-1) ** (i % 2) * scale * v for i, v in enumerate(values)),
                    sum(
                        (-1) ** i.bit_count() * scale * v for i, v in enumerate(values)
                    ),
                ]
            )
            reverb.damping = [
                (1 - a) * v + a * d for v, d in zip(values, reverb.damping, strict=True)
            ]
            reverb.damping = [
                0.0 if abs(v) < processor.state_floor else v for v in reverb.damping
            ]
            total = sum(reverb.damping)
            for i, (line, write, gain) in enumerate(
                zip(reverb.lines, reverb.writes, gains, strict=True)
            ):
                value = (
                    (-1) ** (i // 4) * scale * delayed[0]
                    + (-1) ** (i // 2) * scale * delayed[1]
                    + gain * (reverb.damping[i] - total / 4)
                )
                reverb.non_silent -= int(abs(line[write]) >= processor.tail_threshold)
                reverb.non_silent += int(abs(value) >= processor.tail_threshold)
                line[write] = 0.0 if abs(value) < processor.state_floor else value
                reverb.writes[i] = (write + 1) % len(line)
            drained = (
                ended["input"]
                and reverb.non_silent == 0
                and all(abs(v) < processor.tail_threshold for v in reverb.damping)
            )
            return (np.zeros(2) if drained else _mix(dry, wet, mix), drained)
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
            granular = state.granulator
            assert granular is not None
            frozen = bool(
                self.delayed.get((processor.name, "freeze"), float(state.freeze_target))
            )
            if frozen and not granular.frozen:
                granular.frozen_history = [list(v) for v in granular.history]
                granular.frozen_start = granular.history_start
            elif not frozen:
                granular.frozen_history = []
            granular.frozen = frozen
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
                1.0,
                self._value(processor, "delay_seconds", frame)
                * self.definition.sample_rate,
            )
            _retime_delay(delay, delay_frames, processor.retime_fade_frames)
            wet = _tap_read(delay, len(dry))
            feedback = self._value(processor, "feedback", frame)
            value = dry if not ended["input"] else np.zeros_like(dry)
            history = value + feedback * wet
            history[np.abs(history) < processor.state_floor] = 0
            delay.history[delay.write] = history.tolist()
            delay.write = (delay.write + 1) % frames
            _advance_heads(delay)
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
                (
                    self._value(processor, "base_delay_seconds", frame)
                    + self._value(processor, "depth_seconds", frame) * lfo
                )
                * self.definition.sample_rate,
            )
            wet = np.asarray(
                [
                    _delay_read(delay, float(value), channel)
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
    layouts = _processor_layouts(definition)
    if tempo_map is None and any(
        g.clock == "beats" for g in definition.envelfos.values()
    ):
        raise EngineError("Beat-clock Envelfo requires a host tempo map")
    for processor in definition.processors:
        if isinstance(processor, audio_effects.Filter):
            filters.parameters(processor.filters, sample_rate, 1)
        elif isinstance(processor, audio_effects.ModulatedDelay):
            _channel_phase_offsets(processor, len(layouts[processor.name]))
        elif isinstance(processor, audio_effects.Reverb):
            _reverb_coefficients(
                processor.decay_seconds,
                processor.damping_hz,
                sample_rate,
                audio_effects.reverb_delay_frames(sample_rate),
            )
        elif isinstance(processor, audio_effects.Phaser):
            phaser.coefficient(
                processor.center_hz, processor.depth_cents, processor.sweep, sample_rate
            )
    return PreparedEffects(
        definition=definition,
        sample_rate=sample_rate,
        order=definition.order(),
        channels=definition.output_channels(),
        latency_frames=(
            0
            if isinstance(definition.output, audio_effects.InputSource)
            else definition.latencies()[definition.output.processor]
        ),
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
    list[int],
    int,
    list[float],
]:
    """Encode a supported prepared graph for the allocation-free live owner."""
    graph = definition.definition
    processors = {p.name: p for p in graph.processors}
    ordered = [processors[n] for n in definition.order]
    input_indices = {p.name: i for i, p in enumerate(graph.inputs)}
    processor_indices = {p.name: i + len(graph.inputs) for i, p in enumerate(ordered)}
    layouts = _processor_layouts(graph)
    channel_widths = [len(p.channels) for p in graph.inputs] + [
        len(layouts[p.name]) for p in ordered
    ]
    connections = {(c.processor, c.port): c.source for c in graph.connections}
    widths = [
        3 + 2 * len(p.filters)
        if isinstance(p, audio_effects.Filter)
        else 10
        if isinstance(p, audio_effects.Granulator)
        else 16
        if isinstance(p, audio_effects.Reverb)
        else 71
        if isinstance(p, audio_effects.Waveshaper)
        else 8
        if isinstance(p, audio_effects.Compressor)
        else 5
        if isinstance(p, audio_effects.TapDelay)
        else 7
        if isinstance(p, (audio_effects.ModulatedDelay, audio_effects.Phaser))
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
            else ["input", "detector"]
            if isinstance(processor, audio_effects.Compressor)
            else ["input"]
        )
        for port, name in enumerate(ports):
            source = connections[(processor.name, name)]
            if isinstance(source, audio_effects.InputSource):
                sources[node, port] = input_indices[source.input]
            else:
                sources[node, port] = processor_indices[source.processor]
        parameters[node, 1:3] = [processor.mix, 1]
        if isinstance(processor, audio_effects.Gain):
            kinds.append(0)
            parameters[node, 0] = processor.gain_db
        elif isinstance(processor, audio_effects.Multiply):
            kinds.append(1)
        elif isinstance(processor, audio_effects.Waveshaper):
            kinds.append(4)
            parameters[node, [0, 3, 4, 5]] = [
                processor.drive,
                processor.bias,
                processor.output_gain_db,
                int(processor.curve == "hard"),
            ]
            parameters[node, 6:71] = waveshaper.halfband_coefficients()
        elif isinstance(processor, audio_effects.Phaser):
            kinds.append(9)
            parameters[node, [0, 3, 4, 5, 6]] = [
                processor.center_hz,
                processor.depth_cents,
                processor.sweep,
                processor.feedback,
                processor.state_floor,
            ]
        elif isinstance(processor, audio_effects.Compressor):
            kinds.append(7)
            parameters[node, [0, 3, 4, 5, 6, 7]] = [
                processor.threshold_db,
                processor.ratio,
                processor.knee_db,
                processor.attack_seconds,
                processor.release_seconds,
                processor.makeup_db,
            ]
        elif isinstance(processor, audio_effects.Reverb):
            kinds.append(8)
            parameters[node, [0, 3, 4, 5, 6]] = [
                processor.predelay_seconds,
                processor.decay_seconds,
                processor.state_floor,
                processor.damping_hz,
                processor.maximum_predelay_seconds,
            ]
            parameters[node, 8:16] = audio_effects.reverb_delay_frames(
                definition.sample_rate
            )
            tap_delay_rows.append(
                [
                    node,
                    ceil(processor.maximum_predelay_seconds * definition.sample_rate)
                    + 1,
                    processor.retime_fade_frames,
                ]
            )
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
                    [
                        node,
                        response,
                        parameter,
                        processor.state_floor,
                        filter_definition.minimum_hz,
                        definition.sample_rate / 2 * filter_definition.nyquist_ratio,
                        int(filter_definition.boundary == FilterBoundary.clamp),
                    ]
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
            parameters[node, 8] = processor.maximum_duration_seconds
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
            parameters[node, 4] = processor.state_floor
            tap_delay_rows.append(
                [
                    node,
                    max(
                        1,
                        ceil(processor.maximum_delay_seconds * definition.sample_rate),
                    ),
                    processor.retime_fade_frames,
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
            parameters[node, 6] = processor.state_floor
            phases = _channel_phase_offsets(processor, len(layouts[processor.name]))
            modulated_delay_rows.append(
                [
                    node,
                    max(
                        1,
                        ceil(processor.maximum_delay_seconds * definition.sample_rate),
                    ),
                    list(Waveform).index(processor.waveform),
                    *phases,
                    *([0.0] * (max(channel_widths) - len(phases))),
                ]
            )
        else:
            raise EngineError("Unsupported native live effect processor")
    output_source = (
        input_indices[graph.output.input]
        if isinstance(graph.output, audio_effects.InputSource)
        else processor_indices[graph.output.processor]
    )
    return (
        kinds,
        sources,
        parameters,
        output_source,
        np.asarray(filter_rows, dtype=np.float64).reshape(-1, 7),
        np.asarray(granulator_rows, dtype=np.int64).reshape(-1, 4),
        np.asarray(tap_delay_rows, dtype=np.int64).reshape(-1, 3),
        np.asarray(modulated_delay_rows, dtype=np.float64).reshape(
            -1, max(channel_widths) + 3
        ),
        channel_widths,
        next(
            i
            for i, v in enumerate(graph.inputs)
            if isinstance(v.source, audio_effects.MainStream)
        ),
        [
            p.tail_threshold
            if isinstance(
                p,
                (
                    audio_effects.Filter,
                    audio_effects.TapDelay,
                    audio_effects.ModulatedDelay,
                    audio_effects.Reverb,
                    audio_effects.Phaser,
                ),
            )
            else 0.0
            for p in ordered
        ],
    )


def configure_native_generators(
    runtime: _native.LiveRuntime, definition: PreparedEffects
) -> None:
    """Prepare the shared Rust generator executor for one effect instance."""
    graph = definition.definition
    if not graph.bindings:
        return
    parameters = graph.modulation.parameters
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
        elif isinstance(action, audio_effects.InputEndAction):
            rows.append(
                [
                    action.tick - start,
                    2,
                    next(
                        i for i, v in enumerate(graph.inputs) if v.name == action.input
                    ),
                    0,
                    0,
                    0,
                ]
            )
            continue
        elif isinstance(action, audio_effects.StopAction):
            rows.append([action.tick - start, 3, 0, 0, 0, 0])
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
    elif isinstance(processor, audio_effects.Waveshaper):
        values.update(
            drive=processor.drive,
            bias=processor.bias,
            output_gain_db=processor.output_gain_db,
        )
    elif isinstance(processor, audio_effects.Phaser):
        values.update(
            center_hz=processor.center_hz,
            depth_cents=processor.depth_cents,
            sweep=processor.sweep,
            feedback=processor.feedback,
        )
    elif isinstance(processor, audio_effects.Compressor):
        values.update(
            threshold_db=processor.threshold_db,
            ratio=processor.ratio,
            knee_db=processor.knee_db,
            attack_seconds=processor.attack_seconds,
            release_seconds=processor.release_seconds,
            makeup_db=processor.makeup_db,
        )
    elif isinstance(processor, audio_effects.Reverb):
        values.update(
            predelay_seconds=processor.predelay_seconds,
            decay_seconds=processor.decay_seconds,
            damping_hz=processor.damping_hz,
        )
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
        waveshaper=waveshaper.initial_state(channels)
        if isinstance(processor, audio_effects.Waveshaper)
        else None,
        phaser=PhaserState(
            stages=[[0.0] * channels for _ in range(6)], feedback=[0.0] * channels
        )
        if isinstance(processor, audio_effects.Phaser)
        else None,
        granulator=(
            GranulatorState()
            if isinstance(processor, audio_effects.Granulator)
            else None
        ),
        tap_delay=(
            TapDelayState(
                heads=[
                    DelayHead(
                        delay_frames=max(1.0, processor.delay_seconds * sample_rate),
                        gain=1.0,
                        active=True,
                    ),
                    *[DelayHead() for _ in range(processor.retime_fade_frames)],
                ],
                history=[
                    [0.0] * channels
                    for _ in range(
                        max(1, ceil(processor.maximum_delay_seconds * sample_rate))
                    )
                ],
            )
            if isinstance(processor, audio_effects.TapDelay)
            else None
        ),
        modulated_delay=(
            ModulatedDelayState(
                history=[
                    [0.0] * channels
                    for _ in range(
                        max(1, ceil(processor.maximum_delay_seconds * sample_rate))
                    )
                ]
            )
            if isinstance(processor, audio_effects.ModulatedDelay)
            else None
        ),
        reverb=(
            ReverbState(
                predelay=TapDelayState(
                    heads=[
                        DelayHead(
                            delay_frames=processor.predelay_seconds * sample_rate,
                            gain=1.0,
                            active=True,
                        ),
                        *[
                            DelayHead(delay_frames=0)
                            for _ in range(processor.retime_fade_frames)
                        ],
                    ],
                    history=[
                        [0.0, 0.0]
                        for _ in range(
                            ceil(processor.maximum_predelay_seconds * sample_rate) + 1
                        )
                    ],
                ),
                lines=[
                    [0.0] * n for n in audio_effects.reverb_delay_frames(sample_rate)
                ],
                writes=[0] * 8,
                damping=[0.0] * 8,
            )
            if isinstance(processor, audio_effects.Reverb)
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


def _channel_phase_offsets(
    processor: audio_effects.ModulatedDelay, channels: int
) -> np.ndarray:
    if not processor.channel_phase_offsets:
        return np.zeros(channels)
    if len(processor.channel_phase_offsets) != channels:
        raise EngineError("Modulated delay phase offsets must match output channels")
    return np.asarray(processor.channel_phase_offsets)


def _delay_read(state: DelayState, delay_frames: float, channel: int) -> float:
    whole = floor(delay_frames)
    fraction = delay_frames - whole
    index = (state.write - whole) % len(state.history)
    older = (index - 1) % len(state.history)
    return (1 - fraction) * state.history[index][channel] + fraction * state.history[
        older
    ][channel]


def _retime_delay(state: TapDelayState, delay_frames: float, fade_frames: int) -> None:
    current = state.heads[state.current]
    current.gain = max(
        0.0,
        1
        - sum(
            h.gain for i, h in enumerate(state.heads) if h.active and i != state.current
        ),
    )
    if current.delay_frames == delay_frames:
        return
    current.remaining = fade_frames
    current.step = current.gain / fade_frames
    index = next((i for i, h in enumerate(state.heads) if not h.active), None)
    if index is None:
        raise EngineError("Tap retiming exceeds prepared read-head capacity")
    state.current = index
    state.heads[index] = DelayHead(delay_frames=delay_frames, active=True)


def _advance_heads(state: TapDelayState) -> None:
    for i, head in enumerate(state.heads):
        if not head.active or i == state.current:
            continue
        head.remaining -= 1
        if head.remaining == 0:
            head.active, head.gain = False, 0.0
        else:
            head.gain = max(0.0, head.gain - head.step)


def _tap_read(state: TapDelayState, channels: int) -> np.ndarray:
    return np.array(
        [
            sum(
                h.gain * _delay_read(state, h.delay_frames, c)
                for h in state.heads
                if h.active
            )
            for c in range(channels)
        ]
    )


def _reverb_coefficients(
    decay: float, damping_hz: float, sample_rate: int, lengths: list[int]
) -> tuple[float, list[float]]:
    a = exp(-2 * np.pi * damping_hz / sample_rate)
    gains = [pow(10, -3 * n / (sample_rate * decay)) for n in lengths]
    if not 0 <= a < 1 or any(not 0 <= g < 1 for g in gains):
        raise ValueError("Unrepresentable reverb decay or damping")
    return a, gains


def _native_parameter(processor: audio_effects.Processor, name: str) -> int:
    if name == "mix":
        return 1
    if isinstance(processor, audio_effects.Gain):
        return 0
    if isinstance(processor, audio_effects.Waveshaper):
        return {"drive": 0, "bias": 3, "output_gain_db": 4}[name]
    if isinstance(processor, audio_effects.Phaser):
        return {"center_hz": 0, "depth_cents": 3, "sweep": 4, "feedback": 5}[name]
    if isinstance(processor, audio_effects.Compressor):
        return {
            "threshold_db": 0,
            "ratio": 3,
            "knee_db": 4,
            "attack_seconds": 5,
            "release_seconds": 6,
            "makeup_db": 7,
        }[name]
    if isinstance(processor, audio_effects.Reverb):
        return {"predelay_seconds": 0, "decay_seconds": 3, "damping_hz": 5}[name]
    if isinstance(processor, audio_effects.Filter):
        return 3 + [
            n for f in processor.filters for n in (f"{f.name}-cutoff_hz", f"{f.name}-q")
        ].index(name)
    if isinstance(processor, audio_effects.Granulator):
        if name == "freeze":
            return 9
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
