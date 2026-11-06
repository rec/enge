"""Measure the callback-shaped heterogeneous engine path without device I/O."""

from pathlib import Path
from statistics import median
from time import perf_counter_ns
from typing import Literal

import numpy as np
import tyro
from pydantic import BaseModel, ConfigDict, Field
from ufor import audio_effects
from ufor.events import PerformanceEvent, Trigger
from ufor.instrument_trace import TraceAction
from ufor.samples.processing import FilterResponse, ResonantFilter
from ufor.synth import SynthInstrumentScore
from ufor.synth_trace import prepare as prepare_trace

from enge import effects, live, synth
from enge.render import write_flac


class Settings(BaseModel):
    block_frames: int = Field(default=64, gt=0)
    seconds: int = Field(default=10, gt=0)
    voices: int = Field(default=16, gt=0)
    effect_actions: int = Field(default=64, ge=1, le=64)
    granular: bool = False
    compressor: bool = False
    delay: Literal["none", "tap", "modulated"] = "none"
    audio_output: Path | None = None

    model_config = ConfigDict(frozen=True)


def main(settings: Settings) -> None:
    document = _score()
    events: list[PerformanceEvent] = [
        Trigger(
            tick=0,
            ordinal=i,
            part="main",
            trigger_id=f"note-{i}",
            key=48 + i,
            pitch_hz=110 * 2 ** (i / 12),
        )
        for i in range(settings.voices)
    ]
    actions: list[TraceAction] = list(
        prepare_trace(document.body, events, seed=0).actions
    )
    processors: list[audio_effects.Processor] = [
        audio_effects.Gain(name="trim", gain_db=-6),
        audio_effects.Filter(
            name="tone",
            filters=[
                ResonantFilter(
                    name="low",
                    response=FilterResponse.lowpass,
                    cutoff_hz=3600,
                    q=0.8,
                )
            ],
        ),
    ]
    if settings.granular:
        processors.append(
            audio_effects.Granulator(
                name="cloud",
                duration_seconds=0.015,
                density_hz=90,
                lookback_seconds=0.025,
                playback_ratio=1.25,
                position_jitter_seconds=0.003,
                history_seconds=0.06,
                maximum_grains=3,
                mix=0.8,
            )
        )
    if settings.delay == "tap":
        processors.append(
            audio_effects.TapDelay(
                name="delay",
                delay_seconds=0.005,
                maximum_delay_seconds=0.03,
                feedback=0.6,
                mix=0.5,
            )
        )
    elif settings.delay == "modulated":
        processors.append(
            audio_effects.ModulatedDelay(
                name="delay",
                base_delay_seconds=0.006,
                depth_seconds=0.001,
                maximum_delay_seconds=0.03,
                rate_hz=5,
                channel_phase_offsets=[0, 0.5],
                feedback=0.6,
                mix=0.5,
            )
        )
    graph = effects.serial_graph(
        audio_effects.AttachmentScope.master,
        audio_effects.GraphInput(
            name="main",
            channels=["left", "right"],
            source=audio_effects.MainStream(),
        ),
        processors,
        settings.block_frames,
    )
    if settings.compressor:
        graph = audio_effects.EffectGraph(
            scope=graph.scope,
            inputs=[
                *graph.inputs,
                audio_effects.GraphInput(
                    name="sidechain",
                    channels=["mono"],
                    source=audio_effects.HostStream(stream="detector"),
                ),
            ],
            processors=[*graph.processors, audio_effects.Compressor(name="duck")],
            connections=[
                *graph.connections,
                audio_effects.Connection(
                    processor="duck", port="input", source=graph.output
                ),
                audio_effects.Connection(
                    processor="duck",
                    port="detector",
                    source=audio_effects.InputSource(input="sidechain"),
                ),
            ],
            output=audio_effects.ProcessorSource(processor="duck"),
            maximum_block_frames=settings.block_frames,
        )
    engine = live.NativeLiveEngine(
        {"synth": synth.PersistentSynth(synth.prepare(document), settings.voices)},
        settings.block_frames,
        effects.prepare(graph, 48000),
    )
    output = np.empty((settings.block_frames, 2))
    durations = []
    callbacks = settings.seconds * 48000 // settings.block_frames
    detector = (
        (
            0.5
            + 0.5
            * np.sin(
                2
                * np.pi
                * np.arange(callbacks * settings.block_frames)[:, None]
                * 4
                / 48000
            )
        )
        if settings.compressor
        else None
    )
    recording = (
        np.empty((callbacks * settings.block_frames, 2))
        if settings.audio_output is not None
        else None
    )
    for i in range(callbacks):
        start = i * settings.block_frames
        batch = effect_actions(settings, start)
        host = (
            None
            if detector is None
            else {"detector": detector[start : start + settings.block_frames]}
        )
        before = perf_counter_ns()
        engine.advance_into(
            {"synth": actions if i == 0 else []},
            batch,
            start,
            start + settings.block_frames,
            output,
            host,
        )
        durations.append((perf_counter_ns() - before) / 1000)
        if recording is not None:
            recording[start : start + settings.block_frames] = output
    ordered = sorted(durations[1:])
    budget = settings.block_frames / 48000 * 1e6
    p99 = ordered[int(0.99 * (len(ordered) - 1))]
    print(
        f"callbacks={callbacks} block={settings.block_frames} voices={settings.voices} "
        f"effect_actions={settings.effect_actions} granular={settings.granular}"
        f" delay={settings.delay}"
        f" compressor={settings.compressor}"
    )
    print(f"median_us={median(ordered):.1f} p99_us={p99:.1f} max_us={max(ordered):.1f}")
    print(f"budget_us={budget:.1f} max_fraction={max(ordered) / budget:.3f}")
    print(
        f"first_us={durations[0]:.1f} "
        f"misses={sum(v > budget for v in durations)}/{callbacks}"
    )
    if settings.audio_output is not None and recording is not None:
        settings.audio_output.parent.mkdir(parents=True, exist_ok=True)
        write_flac([recording], settings.audio_output, 48000)
        print(settings.audio_output)


def effect_actions(settings: Settings, start: int) -> list[audio_effects.EffectAction]:
    if settings.delay == "tap":
        return [
            audio_effects.ParameterAction(
                tick=start + i,
                ordinal=0,
                processor="delay",
                parameter="delay_seconds",
                value=0.003 + ((start + i) % 1000) / 480000,
            )
            for i in range(min(settings.effect_actions, settings.block_frames))
        ]
    actions: list[audio_effects.EffectAction] = []
    for ordinal in range(settings.effect_actions):
        processor, parameter, value = (
            ("trim", "gain_db", -8 + (start + ordinal) % 9)
            if ordinal % 3 == 0
            else ("tone", "low-cutoff_hz", 600 + (start + ordinal * 53) % 6000)
            if ordinal % 3 == 1 or not settings.granular
            else ("cloud", "density_hz", 40 + (start + ordinal) % 50)
        )
        if settings.compressor and ordinal % 4 == 3:
            processor, parameter, value = (
                "duck",
                "threshold_db",
                -24 + (start + ordinal) % 18,
            )
        actions.append(
            audio_effects.ParameterAction(
                tick=start,
                ordinal=ordinal,
                processor=processor,
                parameter=parameter,
                value=value,
                duration_frames=ordinal % 31,
            )
        )
    if settings.granular and start > 0 and start % 48000 == 0:
        actions[-1] = audio_effects.FreezeAction(
            tick=start,
            ordinal=settings.effect_actions - 1,
            processor="cloud",
            frozen=(start // 48000) % 2 == 1,
        )
    return actions


def _score() -> SynthInstrumentScore:
    return SynthInstrumentScore.model_validate(
        {
            "name": "benchmark",
            "title": "Benchmark",
            "kind": "synth_instrument",
            "timebases": [
                {"name": "output", "kind": "physical", "rate": {"numerator": 48000}}
            ],
            "body": {
                "voices": [
                    {
                        "name": "tone",
                        "mapping": {
                            "lowest_key": 0,
                            "highest_key": 127,
                            "reference_pitch_hz": 440,
                        },
                        "channels": [
                            {"input": "mono", "output": "left", "gain": 0.05},
                            {"input": "mono", "output": "right", "gain": 0.05},
                        ],
                        "oscillator": {"waveform": "triangle"},
                    }
                ]
            },
            "inputs": [
                {
                    "name": "performance",
                    "stream": {
                        "family": "event",
                        "timebase": "output",
                        "kinds": ["trigger", "release", "control_change"],
                    },
                    "binding": {"performance": True},
                }
            ],
            "outputs": [
                {
                    "name": "audio",
                    "stream": {
                        "family": "sampled",
                        "quantity": "audio_amplitude",
                        "unit": "full_scale",
                        "timebase": "output",
                        "channels": ["left", "right"],
                    },
                    "binding": {"audio": True},
                }
            ],
        }
    )


if __name__ == "__main__":
    main(tyro.cli(Settings))
