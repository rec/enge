"""Block-oriented ownership for heterogeneous persistent sound engines."""

from typing import Callable

import numpy as np
from pydantic import BaseModel, ConfigDict
from ufor import audio_effects, instrument_trace

from . import effects, fm, noise, sample_instrument, synth


class LiveEngineSnapshot(BaseModel, frozen=True):
    frame: int
    source_names: list[str]
    source_states: dict[str, object]
    effect_state: effects.EffectSnapshot | None

    model_config = ConfigDict(arbitrary_types_allowed=True)


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
        self.effects = effect_chain
        self.effect_input: str | None = None
        if effect_chain is not None:
            graph = effect_chain.definition.definition
            main = [
                i
                for i in graph.inputs
                if isinstance(i.source, audio_effects.MainStream)
            ]
            if (
                effect_chain.definition.sample_rate != self.sample_rate
                or effect_chain.definition.channels != self.channels
                or len(graph.inputs) != 1
                or len(main) != 1
            ):
                raise synth.EngineError(
                    "Live engine effects require one compatible main input"
                )
            self.effect_input = main[0].name
        self.frame = 0

    def advance(
        self,
        actions: dict[str, list[instrument_trace.TraceAction]],
        effect_actions: list[audio_effects.EffectAction],
        start: int,
        end: int,
    ) -> np.ndarray:
        if actions.keys() != self.sources.keys():
            raise synth.EngineError("Live engine actions must address every source")
        if start != self.frame or end <= start:
            raise synth.EngineError(
                "Live engine advance must be contiguous and nonempty"
            )
        output = np.zeros((end - start, len(self.channels)))
        for name, source in self.sources.items():
            advance = getattr(source, "advance", None)
            if not isinstance(advance, Callable):
                raise synth.EngineError(f"Live source {name} has no advance method")
            output += advance(actions[name], start, end)
        if self.effects is not None:
            assert self.effect_input is not None
            output = self.effects.advance(
                {self.effect_input: output}, effect_actions, start, end
            )
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
        output[:] = self.advance(actions, effect_actions, start, end)

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
