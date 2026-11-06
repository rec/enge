"""Block-oriented ownership for heterogeneous persistent sound engines."""

from collections.abc import Callable

import numpy as np
from pydantic import BaseModel, ConfigDict
from ufor import audio_effects, instrument_trace

from . import _native, effects, fm, noise, sample_instrument, synth


class LiveEngineSnapshot(BaseModel, frozen=True):
    frame: int
    source_names: list[str]
    source_states: dict[str, object]
    effect_state: effects.EffectSnapshot | None

    model_config = ConfigDict(arbitrary_types_allowed=True)


class NativeLiveEngineSnapshot(BaseModel, frozen=True):
    frame: int
    source_names: list[str]
    source_states: dict[str, object]
    runtime: object

    model_config = ConfigDict(arbitrary_types_allowed=True)


class NativeLiveEngine:
    """Prepare actions in Python for one native generated-source callback owner."""

    def __init__(
        self,
        sources: dict[
            str,
            synth.PersistentSynth
            | fm.PersistentFM
            | noise.PersistentNoise
            | sample_instrument.PersistentSampler,
        ],
        maximum_block_frames: int,
        effect_chain: effects.PreparedEffects | None = None,
        action_capacity: int = 64,
        batch_capacity: int = 4,
    ) -> None:
        if not sources:
            raise synth.EngineError("Native live engine requires at least one source")
        definitions = [s.definition for s in sources.values()]
        rates = {d.sample_rate for d in definitions}
        channels = {tuple(d.channels) for d in definitions}
        if len(rates) != 1 or len(channels) != 1:
            raise synth.EngineError(
                "Native live engine sources must share rate and channels"
            )
        self.sources = sources.copy()
        generated = {
            name: source
            for name, source in sources.items()
            if not isinstance(source, sample_instrument.PersistentSampler)
        }
        self.source_indices = {name: i for i, name in enumerate(generated)}
        self.sample_rate = rates.pop()
        self.channels = list(channels.pop())
        self.source_channels = self.channels.copy()
        self.host_widths: dict[str, int] = {}
        self.maximum_block_frames = maximum_block_frames
        self.effect_chain = effect_chain
        self.runtime = (
            _native.LiveRuntime(
                [s.runtime for s in generated.values()],
                maximum_block_frames,
                action_capacity,
                batch_capacity,
            )
            if generated
            else _native.LiveRuntime.empty(
                self.sample_rate,
                len(self.channels),
                maximum_block_frames,
                action_capacity,
            )
        )
        for source in sources.values():
            if isinstance(source, sample_instrument.PersistentSampler):
                source.add_to_native_live(self.runtime, batch_capacity)
        if effect_chain is not None:
            self.host_widths = _prepare_live_inputs(
                effect_chain, self.sample_rate, self.source_channels, list(sources)
            )
            source_routes = {
                n: list(s.native_sources.values())
                if isinstance(s, sample_instrument.PersistentSampler)
                else [self.source_indices[n]]
                for n, s in self.sources.items()
            }
            routes = [
                [-1]
                if isinstance(i.source, audio_effects.MainStream)
                else [-2]
                if isinstance(i.source, audio_effects.HostStream)
                else source_routes[i.source.instrument]
                for i in effect_chain.definition.inputs
            ]
            self.runtime.set_effect_graph(
                *effects.native_live_graph(effect_chain),
                batch_capacity,
                input_routes=routes,
            )
            effects.configure_native_generators(self.runtime, effect_chain)
            self.channels = effect_chain.channels.copy()
        self.frame = 0

    def advance_into(
        self,
        actions: dict[str, list[instrument_trace.TraceAction]],
        effect_actions: list[audio_effects.EffectAction],
        start: int,
        end: int,
        output: np.ndarray,
        host_inputs: dict[str, np.ndarray] | None = None,
    ) -> None:
        if actions.keys() != self.sources.keys():
            raise synth.EngineError("Native live actions must address every source")
        if start != self.frame or end <= start:
            raise synth.EngineError(
                "Native live advance must be contiguous and nonempty"
            )
        if end - start > self.maximum_block_frames:
            raise synth.EngineError("Native live block exceeds its prepared maximum")
        if (
            output.shape != (end - start, len(self.channels))
            or output.dtype != np.float64
            or not output.flags.c_contiguous
            or not output.flags.writeable
        ):
            raise synth.EngineError(
                "Native live output must be writable C-contiguous float64 with "
                "shape (frames, channels)"
            )
        host = _validate_host_inputs(self.host_widths, host_inputs, end - start)
        for name, source in self.sources.items():
            if isinstance(source, sample_instrument.PersistentSampler):
                for index, encoded in source.native_live_actions(
                    actions[name], start, end, self.runtime
                ).items():
                    if len(encoded):
                        self.runtime.submit(index, encoded)
            else:
                index = self.source_indices[name]
                encoded = source.native_live_actions(
                    actions[name],
                    start,
                    end,
                    self.runtime.active_slots(index),
                    self.runtime.active_contexts(index),
                )
                if len(encoded):
                    self.runtime.submit(index, encoded)
        if self.effect_chain is None:
            if effect_actions:
                raise synth.EngineError("Native live engine has no effect chain")
        else:
            encoded = effects.native_live_actions(
                self.effect_chain, effect_actions, start, end
            )
            if len(encoded):
                self.runtime.submit_effects(encoded)
        self.runtime.process_into(
            output,
            []
            if self.effect_chain is None
            else [
                host[i.source.stream]
                for i in self.effect_chain.definition.inputs
                if isinstance(i.source, audio_effects.HostStream)
            ],
        )
        for name, source in self.sources.items():
            if isinstance(source, sample_instrument.PersistentSampler):
                source.finish_native_live_block(end, self.runtime)
            else:
                index = self.source_indices[name]
                source.finish_native_live_block(
                    end,
                    self.runtime.active_slots(index),
                    self.runtime.active_contexts(index),
                )
        self.frame = end

    def snapshot(self) -> NativeLiveEngineSnapshot:
        return NativeLiveEngineSnapshot(
            frame=self.frame,
            source_names=list(self.sources),
            source_states={n: s.snapshot() for n, s in self.sources.items()},
            runtime=self.runtime.snapshot(),
        )

    def restore(self, snapshot: NativeLiveEngineSnapshot) -> None:
        if snapshot.source_names != list(self.sources):
            raise synth.EngineError("Snapshot belongs to different native live sources")
        for name, source in self.sources.items():
            source.restore(snapshot.source_states[name])  # ty: ignore[invalid-argument-type]
        self.runtime.restore(snapshot.runtime)  # ty: ignore[invalid-argument-type]
        self.frame = snapshot.frame


class LiveEngine:
    """Own and mix prepared persistent sources and an optional effect graph."""

    def __init__(
        self,
        sources: dict[
            str,
            synth.PersistentSynth
            | fm.PersistentFM
            | noise.PersistentNoise
            | sample_instrument.PersistentSampler,
        ],
        effect_chain: effects.OfflineEffects | None = None,
    ) -> None:
        if not sources:
            raise synth.EngineError("Live engine requires at least one source")
        definitions = [s.definition for s in sources.values()]
        rates = {d.sample_rate for d in definitions}
        channels = {tuple(d.channels) for d in definitions}
        if len(rates) != 1 or len(channels) != 1:
            raise synth.EngineError("Live engine sources must share rate and channels")
        self.sources = sources.copy()
        self.sample_rate = rates.pop()
        self.channels = list(channels.pop())
        self.source_channels = self.channels.copy()
        self.effects = effect_chain
        self.host_widths: dict[str, int] = {}
        if effect_chain is not None:
            self.host_widths = _prepare_live_inputs(
                effect_chain.definition,
                self.sample_rate,
                self.source_channels,
                list(sources),
            )
            self.channels = effect_chain.definition.channels.copy()
        self.frame = 0

    def advance(
        self,
        actions: dict[str, list[instrument_trace.TraceAction]],
        effect_actions: list[audio_effects.EffectAction],
        start: int,
        end: int,
        host_inputs: dict[str, np.ndarray] | None = None,
    ) -> np.ndarray:
        if actions.keys() != self.sources.keys():
            raise synth.EngineError("Live engine actions must address every source")
        if start != self.frame or end <= start:
            raise synth.EngineError(
                "Live engine advance must be contiguous and nonempty"
            )
        host = _validate_host_inputs(self.host_widths, host_inputs, end - start)
        output = np.zeros((end - start, len(self.source_channels)))
        rendered: dict[str, np.ndarray] = {}
        for name, source in self.sources.items():
            advance = getattr(source, "advance", None)
            if not isinstance(advance, Callable):
                raise synth.EngineError(f"Live source {name} has no advance method")
            rendered[name] = advance(actions[name], start, end)
            output += rendered[name]
        if self.effects is not None:
            inputs = {
                i.name: output
                if isinstance(i.source, audio_effects.MainStream)
                else host[i.source.stream]
                if isinstance(i.source, audio_effects.HostStream)
                else rendered[i.source.instrument]
                for i in self.effects.definition.definition.inputs
            }
            output = self.effects.advance(inputs, effect_actions, start, end)
        elif effect_actions:
            raise synth.EngineError("Live engine has no effect chain")
        self.frame = end
        return output

    def advance_into(
        self,
        actions: dict[str, list[instrument_trace.TraceAction]],
        effect_actions: list[audio_effects.EffectAction],
        start: int,
        end: int,
        output: np.ndarray,
        host_inputs: dict[str, np.ndarray] | None = None,
    ) -> None:
        if (
            output.shape != (end - start, len(self.channels))
            or output.dtype != np.float64
            or not output.flags.c_contiguous
            or not output.flags.writeable
        ):
            raise synth.EngineError(
                "Live engine output must be writable C-contiguous float64 with "
                "shape (frames, channels)"
            )
        output[:] = self.advance(actions, effect_actions, start, end, host_inputs)

    def snapshot(self) -> LiveEngineSnapshot:
        return LiveEngineSnapshot(
            frame=self.frame,
            source_names=list(self.sources),
            source_states={n: s.snapshot() for n, s in self.sources.items()},
            effect_state=None if self.effects is None else self.effects.snapshot(),
        )

    def restore(self, snapshot: LiveEngineSnapshot) -> None:
        if snapshot.source_names != list(self.sources):
            raise synth.EngineError("Snapshot belongs to different live sources")
        if (snapshot.effect_state is None) != (self.effects is None):
            raise synth.EngineError("Snapshot belongs to a different live effect chain")
        for name, source in self.sources.items():
            source.restore(snapshot.source_states[name])  # ty: ignore[invalid-argument-type]
        if self.effects is not None:
            assert snapshot.effect_state is not None
            self.effects.restore(snapshot.effect_state)
        self.frame = snapshot.frame


def _prepare_live_inputs(
    definition: effects.PreparedEffects,
    rate: int,
    channels: list[str],
    source_names: list[str],
) -> dict[str, int]:
    if definition.sample_rate != rate:
        raise synth.EngineError("Live effects must match the source sample rate")
    hosts: dict[str, int] = {}
    for input in definition.definition.inputs:
        source = input.source
        if isinstance(source, audio_effects.HostStream):
            width = len(input.channels)
            if source.stream in hosts and hosts[source.stream] != width:
                raise synth.EngineError(
                    "Live host stream has inconsistent channel widths"
                )
            hosts[source.stream] = width
        else:
            if input.channels != channels:
                raise synth.EngineError(
                    "Live source input must match its channel layout"
                )
            if isinstance(source, audio_effects.InstrumentStream):
                if source.instrument not in source_names:
                    raise synth.EngineError(
                        "Live effect input references an unknown source"
                    )
                if source.tap != "pre_effect":
                    raise synth.EngineError(
                        "Live sources have no post-effect instrument tap"
                    )
    return hosts


def _validate_host_inputs(
    widths: dict[str, int], inputs: dict[str, np.ndarray] | None, frames: int
) -> dict[str, np.ndarray]:
    values = {} if inputs is None else inputs
    if values.keys() != widths.keys():
        raise synth.EngineError("Live host inputs do not match the prepared graph")
    for name, width in widths.items():
        value = values[name]
        if (
            value.shape != (frames, width)
            or value.dtype != np.float64
            or not np.all(np.isfinite(value))
        ):
            raise synth.EngineError(f"Invalid live host input {name}")
    return values
