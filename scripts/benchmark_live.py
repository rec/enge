"""Measure the callback-shaped heterogeneous engine path without device I/O."""

from pathlib import Path
from statistics import median
from time import perf_counter_ns
from typing import Annotated, Literal, Self

import numpy as np
import tyro
from pydantic import BaseModel, ConfigDict, Field, model_validator
from reccy.configuration import units
from ufor import audio_effects
from ufor.base import Frequency
from ufor.events import PerformanceEvent, Trigger
from ufor.instrument_trace import TraceAction
from ufor.samples.processing import FilterResponse, ResonantFilter
from ufor.synth import SynthInstrumentScore
from ufor.synth_trace import prepare as prepare_trace

from enge import effects, live, synth
from enge.render import write_flac


class Settings(BaseModel):
    block_frames: int = Field(default=64, gt=0)
    seconds: Annotated[
        int, units.unit_validator("second"), tyro.conf.arg(constructor=str)
    ] = Field(default=10, gt=0)
    voices: int = Field(default=16, gt=0)
    effect_actions: int = Field(default=64, ge=1, le=64)
    granular: bool = False
    compressor: bool = False
    dynamics: Literal["none", "gate", "expander"] = "none"
    reverb: bool = False
    waveshaper: bool = False
    phaser: bool = False
    highpass: bool = False
    matrix: bool = False
    mixer: bool = False
    follower: bool = False
    crossfader: Literal["none", "linear", "equal_power"] = "none"
    delay: Literal[
        "none", "tap", "modulated", "multitap", "matrix", "ensemble", "resonant"
    ] = "none"
    damping_hz: Annotated[Frequency | None, tyro.conf.arg(constructor=str)] = None
    audio_output: Path | None = None

    @model_validator(mode="after")
    def damping_profile(self) -> Self:
        if self.damping_hz is not None and self.delay not in {"tap", "matrix"}:
            raise ValueError("damping_hz requires a fixed delay profile")
        return self

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
    if settings.delay == "resonant":
        processors.append(
            audio_effects.ResonantDelay(
                name="string",
                frequency_hz=440,
                damping_hz=6000,
                decay_seconds=0.04,
                mix=0.5,
            )
        )
    elif settings.delay == "matrix":
        processors.append(
            audio_effects.MatrixDelay(
                name="delay",
                delay_seconds=0.005,
                maximum_delay_seconds=0.03,
                feedback=0.6,
                damping_hz=settings.damping_hz,
                mix=0.5,
                feedback_matrix=[[0, 1], [1, 0]],
            )
        )
    elif settings.delay == "tap":
        processors.append(
            audio_effects.TapDelay(
                name="delay",
                delay_seconds=0.005,
                maximum_delay_seconds=0.03,
                feedback=0.6,
                damping_hz=settings.damping_hz,
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
    if settings.reverb:
        processors.append(audio_effects.Reverb(name="room"))
    if settings.waveshaper:
        processors.append(
            audio_effects.Waveshaper(name="shape", drive=3, bias=0.2, mix=0.6)
        )
    if settings.matrix:
        processors.append(audio_effects.width_matrix("route", 1.5))
    if settings.highpass:
        processors.append(audio_effects.Highpass(name="dc"))
    if settings.phaser:
        processors.append(audio_effects.Phaser(name="phase", feedback=0.5))
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
    if settings.delay in {"multitap", "ensemble"}:
        if settings.delay == "ensemble":
            multi = audio_effects.modulated_multitap_graph(
                "echo",
                graph.scope,
                graph.inputs[0],
                [
                    audio_effects.ModulatedTap(
                        delay=audio_effects.ModulatedDelay(
                            name=n,
                            base_delay_seconds=b,
                            depth_seconds=d,
                            maximum_delay_seconds=0.05,
                            rate_hz=r,
                            channel_phase_offsets=[p, (p + 0.37) % 1],
                        ),
                        gain_db=-12,
                        pan=s,
                    )
                    for n, b, d, r, p, s in [
                        ("a", 0.015, 0.002, 0.37, 0.05, -1),
                        ("b", 0.019, 0.0025, 0.53, 0.27, -0.35),
                        ("c", 0.023, 0.003, 0.71, 0.51, 0.35),
                        ("d", 0.027, 0.0018, 0.89, 0.78, 1),
                    ]
                ],
                settings.block_frames,
                dry_gain_db=-3,
            )
        else:
            multi = audio_effects.multitap_graph(
                "echo",
                graph.scope,
                graph.inputs[0],
                [
                    audio_effects.DelayTap(
                        name=n,
                        delay_seconds=d,
                        maximum_delay_seconds=0.03,
                        gain_db=-6,
                        pan=p,
                    )
                    for n, d, p in [
                        ("a", 0.003, -1),
                        ("b", 0.005, -0.3),
                        ("c", 0.007, 0.3),
                        ("d", 0.009, 1),
                    ]
                ],
                settings.block_frames,
                dry_gain_db=-6,
            )
        graph = audio_effects.EffectGraph(
            scope=graph.scope,
            inputs=graph.inputs,
            processors=[*graph.processors, *multi.processors],
            connections=[
                *graph.connections,
                *[
                    c.model_copy(update={"source": graph.output})
                    if isinstance(c.source, audio_effects.InputSource)
                    else c
                    for c in multi.connections
                ],
            ],
            output=multi.output,
            maximum_block_frames=settings.block_frames,
        )
    detectors: list[audio_effects.Processor] = []
    if settings.compressor:
        detectors.append(audio_effects.Compressor(name="duck"))
    if settings.dynamics == "gate":
        detectors.append(audio_effects.Gate(name="dynamics"))
    elif settings.dynamics == "expander":
        detectors.append(audio_effects.Expander(name="dynamics"))
    if detectors:
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
            processors=[*graph.processors, *detectors],
            connections=[
                *graph.connections,
                *[
                    c
                    for i, p in enumerate(detectors)
                    for c in [
                        audio_effects.Connection(
                            processor=p.name,
                            port="input",
                            source=graph.output
                            if i == 0
                            else audio_effects.ProcessorSource(
                                processor=detectors[i - 1].name
                            ),
                        ),
                        audio_effects.Connection(
                            processor=p.name,
                            port="detector",
                            source=audio_effects.InputSource(input="sidechain"),
                        ),
                    ]
                ],
            ],
            output=audio_effects.ProcessorSource(processor=detectors[-1].name),
            maximum_block_frames=settings.block_frames,
        )
    if settings.follower:
        graph = audio_effects.EffectGraph.model_validate(
            graph.model_dump()
            | {
                "processors": [
                    *graph.processors,
                    audio_effects.Follower(name="level"),
                    audio_effects.Gain(name="follow_gain"),
                ],
                "connections": [
                    *graph.connections,
                    audio_effects.Connection(
                        processor="level", port="input", source=graph.output
                    ),
                    audio_effects.Connection(
                        processor="follow_gain", port="input", source=graph.output
                    ),
                ],
                "output": audio_effects.ProcessorSource(processor="follow_gain"),
                "bindings": [
                    {"kind": "follower", "name": "amplitude", "processor": "level"}
                ],
                "modulation": {
                    "sources": [
                        {
                            "name": "amplitude",
                            "scope": "voice",
                            "minimum": 0,
                            "maximum": 4,
                        }
                    ],
                    "parameters": [
                        {
                            "target": {"name": "follow_gain", "parameter": "gain_db"},
                            "scope": "voice",
                            "unit": "db",
                            "minimum": -12,
                            "maximum": 0,
                            "default": 0,
                        }
                    ],
                    "routes": [
                        {
                            "name": "follow",
                            "source": "amplitude",
                            "target": {"name": "follow_gain", "parameter": "gain_db"},
                            "operation": "add",
                            "unit": "db",
                            "points": [
                                {"input": 0, "amount": -12},
                                {"input": 1, "amount": 0},
                            ],
                        }
                    ],
                },
            }
        )
    extra_streams = []
    if settings.mixer:
        extra_streams.extend(["mix_b", "mix_c", "mix_d"])
        graph = audio_effects.EffectGraph(
            scope=graph.scope,
            inputs=[
                *graph.inputs,
                *[
                    audio_effects.GraphInput(
                        name=n,
                        channels=["left", "right"],
                        source=audio_effects.HostStream(stream=n),
                    )
                    for n in extra_streams
                ],
            ],
            processors=[
                *graph.processors,
                audio_effects.Mixer(
                    name="sum",
                    inputs=[
                        audio_effects.MixerInput(name=n, gain_db=-12)
                        for n in ["a", "b", "c", "d"]
                    ],
                ),
            ],
            connections=[
                *graph.connections,
                audio_effects.Connection(
                    processor="sum", port="a", source=graph.output
                ),
                *[
                    audio_effects.Connection(
                        processor="sum",
                        port=n,
                        source=audio_effects.InputSource(input=f"mix_{n}"),
                    )
                    for n in ["b", "c", "d"]
                ],
            ],
            output=audio_effects.ProcessorSource(processor="sum"),
            maximum_block_frames=settings.block_frames,
        )
    if settings.crossfader != "none":
        extra_streams.append("fade_input")
        graph = audio_effects.EffectGraph(
            scope=graph.scope,
            inputs=[
                *graph.inputs,
                audio_effects.GraphInput(
                    name="fade_input",
                    channels=["left", "right"],
                    source=audio_effects.HostStream(stream="fade_input"),
                ),
            ],
            processors=[
                *graph.processors,
                audio_effects.Crossfader(name="fade", law=settings.crossfader),
            ],
            connections=[
                *graph.connections,
                audio_effects.Connection(
                    processor="fade", port="first", source=graph.output
                ),
                audio_effects.Connection(
                    processor="fade",
                    port="second",
                    source=audio_effects.InputSource(input="fade_input"),
                ),
            ],
            output=audio_effects.ProcessorSource(processor="fade"),
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
        if settings.compressor or settings.dynamics != "none"
        else None
    )
    extra_audio = {
        n: np.repeat(
            (
                0.1
                * np.sin(
                    2
                    * np.pi
                    * np.arange(callbacks * settings.block_frames)[:, None]
                    * (220 + i * 73)
                    / 48000
                )
            ),
            2,
            axis=1,
        )
        for i, n in enumerate(extra_streams)
    }
    recording = (
        np.empty((callbacks * settings.block_frames, 2))
        if settings.audio_output is not None
        else None
    )
    for i in range(callbacks):
        start = i * settings.block_frames
        batch = effect_actions(settings, start)
        host = {
            n: a[start : start + settings.block_frames] for n, a in extra_audio.items()
        }
        if detector is not None:
            host["detector"] = detector[start : start + settings.block_frames]
        before = perf_counter_ns()
        engine.advance_into(
            {"synth": actions if i == 0 else []},
            batch,
            start,
            start + settings.block_frames,
            output,
            host or None,
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
        f" damping_hz={settings.damping_hz}"
        f" compressor={settings.compressor}"
        f" dynamics={settings.dynamics}"
        f" reverb={settings.reverb}"
        f" waveshaper={settings.waveshaper}"
        f" phaser={settings.phaser}"
        f" highpass={settings.highpass}"
        f" matrix={settings.matrix}"
        f" mixer={settings.mixer} crossfader={settings.crossfader}"
        f" follower={settings.follower}"
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
    if settings.delay == "resonant":
        return [
            audio_effects.ParameterAction(
                tick=start + i,
                ordinal=0,
                processor="string",
                parameter="mix" if i % 4 == 0 else "decay_seconds",
                value=0.25 + 0.5 * (i % 16) / 15
                if i % 4 == 0
                else 0.02 + ((start + i) % 100) / 1000,
                duration_frames=8,
            )
            for i in range(min(settings.effect_actions, settings.block_frames))
        ]
    if settings.delay in {"multitap", "ensemble"}:
        return [
            audio_effects.ParameterAction(
                tick=start + i,
                ordinal=0,
                processor="echo" if i % 3 == 0 else f"echo-tap-{'abcd'[i % 4]}",
                parameter=f"tap-{'abcd'[i % 4]}-gain_db"
                if i % 3 == 0
                else "base_delay_seconds"
                if settings.delay == "ensemble"
                else "delay_seconds",
                value=-9 + i % 6
                if i % 3 == 0
                else (0.015 if settings.delay == "ensemble" else 0.003)
                + ((start + i) % 1000) / 480000,
                duration_frames=8 if i % 3 == 0 else 0,
            )
            for i in range(min(settings.effect_actions, settings.block_frames))
        ]
    if settings.delay in {"tap", "matrix"}:
        return [
            audio_effects.ParameterAction(
                tick=start + i,
                ordinal=0,
                processor="delay",
                parameter="damping_hz"
                if settings.damping_hz is not None and i % 3 == 0
                else "delay_seconds",
                value=settings.damping_hz * (0.5 + 0.5 * ((start + i) % 100) / 100)
                if settings.damping_hz is not None and i % 3 == 0
                else 0.003 + ((start + i) % 1000) / 480000,
            )
            for i in range(min(settings.effect_actions, settings.block_frames))
        ]
    actions: list[audio_effects.EffectAction] = []
    for ordinal in range(settings.effect_actions):
        duration = ordinal % 31
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
        if settings.dynamics != "none" and ordinal % 4 == 3:
            processor, parameter, value = (
                "dynamics",
                "threshold_db",
                -24 + (start + ordinal) % 18,
            )
        if settings.reverb and ordinal % 4 == 2:
            processor, parameter, value = (
                "room",
                "predelay_seconds",
                0.003 + ((start + ordinal) % 1000) / 480000,
            )
        if settings.waveshaper and ordinal % 4 == 2:
            processor, parameter, value = (
                "shape",
                "drive",
                1 + ((start + ordinal) % 100) / 20,
            )
        if settings.highpass and ordinal % 4 == 2:
            processor, parameter, value = (
                "dc",
                "cutoff_hz",
                20 + (start + ordinal) % 200 * 10,
            )
        if settings.phaser and ordinal % 4 == 2:
            processor, parameter, value = (
                "phase",
                "sweep",
                ((start + ordinal) % 200) / 100 - 1,
            )
        if settings.follower and ordinal % 4 == 2:
            processor, parameter, value = (
                "level",
                "attack_seconds",
                0.001 + (start + ordinal) % 1000 / 1000000,
            )
        if settings.mixer and ordinal % 4 == 2:
            processor, parameter, value = (
                "sum",
                f"{'bcd'[ordinal % 3]}-{'muted' if ordinal % 8 == 2 else 'solo'}",
                float((start // settings.block_frames + ordinal) % 2),
            )
            duration = 0
        if settings.crossfader != "none" and ordinal % 4 == 2:
            processor, parameter, value = (
                "fade",
                "position",
                ((start + ordinal) % 200) / 200,
            )
            duration = ordinal % 31
        actions.append(
            audio_effects.ParameterAction(
                tick=start,
                ordinal=ordinal,
                processor=processor,
                parameter=parameter,
                value=value,
                duration_frames=duration,
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
