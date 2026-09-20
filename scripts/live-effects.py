"""Render four enge sources through per-source and shared live-effect chains."""

from pathlib import Path
from typing import Literal

import numpy as np
import tyro
from pydantic import BaseModel, Field
from ufor import audio_effects, synth_trace
from ufor.events import ControlChange, Release, Trigger
from ufor.instrument_trace import TraceAction
from ufor.samples import instrument, trace
from ufor.samples.processing import FilterResponse, ResonantFilter

from enge import effects, fm, noise, sample_instrument, synth
from enge.presets import Patch
from enge.render import write_flac


class Options(BaseModel, frozen=True):
    output: Path = Path("live-effects.flac")
    backend: Literal["numpy", "native"] = "native"
    block_frames: int = Field(default=4096, gt=0, le=4096)


def main() -> None:
    options = tyro.cli(Options)
    rate, frames = 48000, 6 * 48000
    sources = [
        render_patch(Patch(engine="synth", gain=0.09, pan=-0.5), 48, options.backend),
        render_patch(Patch(engine="sample", gain=0.08, pan=0.5), 55, options.backend),
        render_patch(
            Patch(engine="fm", gain=0.07, ratio=2, index=1.1), 60, options.backend
        ),
        np.repeat(noise.noise_samples(19, 0, frames), 2, axis=1) * 0.025,
    ]
    prepared = effects.prepare(effect_chain(options.block_frames), rate)
    individual = sum(
        (
            effects.process_audio(prepared, {"main": v}, backend=options.backend)
            for v in sources
        ),
        np.zeros((frames, 2)),
    )
    shared = effects.process_audio(
        prepared, {"main": sum(sources, np.zeros((frames, 2)))}, backend=options.backend
    )
    output = np.column_stack([individual[:, 0], shared[:, 1]])
    peak = float(np.max(np.abs(output)))
    write_flac([output / max(1, peak / 0.95)], options.output, rate)
    print(f"{options.output}: left per-source, right shared; pre-limit peak {peak:.3f}")


def render_patch(
    patch: Patch, key: int, backend: Literal["numpy", "native"]
) -> np.ndarray:
    rate, frames = 48000, 6 * 48000
    score, assets = patch.prepare_score(rate)
    events: list[Trigger | Release | ControlChange] = []
    for i, offset in enumerate((0, 4, 7, 12, 7)):
        start, name = i * rate, f"note-{i}"
        events.extend(
            [
                Trigger(
                    tick=start,
                    ordinal=0,
                    part="main",
                    trigger_id=name,
                    key=key + offset,
                    pitch_hz=440 * 2 ** ((key + offset - 69) / 12),
                ),
                Release(
                    tick=start + 36000,
                    ordinal=0,
                    part="main",
                    trigger_id=name,
                ),
            ]
        )
    if isinstance(score, instrument.SampleInstrumentScore):
        renderer = sample_instrument.OfflineSampler(
            sample_instrument.prepare(score, assets), backend
        )
        sample_actions: list[TraceAction] = list(
            trace.prepare(score.body, events, seed=0).actions
        )
        return renderer.advance(sample_actions, 0, frames)
    synth_actions: list[TraceAction] = list(
        synth_trace.prepare(score.body, events, seed=0).actions
    )
    if patch.engine == "fm":
        return fm.OfflineFM(fm.prepare(score), backend).advance(
            synth_actions, 0, frames
        )
    return synth.OfflineSynth(synth.prepare(score), backend).advance(
        synth_actions, 0, frames
    )


def effect_chain(block_frames: int) -> audio_effects.EffectGraph:
    return effects.serial_graph(
        audio_effects.AttachmentScope.master,
        audio_effects.GraphInput(
            name="main",
            channels=["left", "right"],
            source=audio_effects.MainStream(),
        ),
        [
            audio_effects.Filter(
                name="warmth",
                filters=[
                    ResonantFilter(
                        name="low",
                        response=FilterResponse.lowpass,
                        cutoff_hz=5200,
                        q=0.75,
                    )
                ],
            ),
            audio_effects.Granulator(
                name="shimmer",
                duration_seconds=0.045,
                density_hz=38,
                lookback_seconds=0.07,
                playback_ratio=0.75,
                position_jitter_seconds=0.012,
                history_seconds=0.14,
                maximum_grains=4,
                mix=0.32,
            ),
        ],
        block_frames,
    )


if __name__ == "__main__":
    main()
