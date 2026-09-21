"""Benchmark source-to-output effect blocks and write a reproducible report."""

import platform
from pathlib import Path
from time import perf_counter_ns

import numpy as np
import tyro
from pydantic import BaseModel, Field
from ufor import audio_effects, synth_trace
from ufor.events import ControlChange, Release, Trigger
from ufor.instrument_trace import TraceAction
from ufor.samples.processing import FilterResponse, ResonantFilter
from ufor.synth import SynthInstrumentScore

from enge import _native, effects, synth
from enge.presets import Patch


class Options(BaseModel, frozen=True):
    output: Path = Path("doc/live-effects-benchmark.md")
    seconds: float = Field(default=1, gt=0)
    chains: int = Field(default=4, ge=1)
    block_sizes: list[int] = Field(default_factory=lambda: [64, 256, 1024])


class Result(BaseModel, frozen=True):
    case: str
    block: int
    blocks: int
    worst_us: float
    p99_us: float
    deadline_us: float
    misses: int


def main() -> None:
    options = tyro.cli(Options)
    results = [
        benchmark_persistent(block, options) for block in options.block_sizes
    ] + [
        benchmark(case, block, options)
        for case in (
            "filter",
            "filter-actions",
            "fanout",
            "granulator",
            "filter+granulator",
            "freeze",
            "subnormal",
            "filter-tail",
        )
        for block in options.block_sizes
    ]
    options.output.parent.mkdir(parents=True, exist_ok=True)
    options.output.write_text(report(options, results))
    print(options.output)


def benchmark_persistent(block: int, options: Options) -> Result:
    frames = round(options.seconds * 48000)
    frequencies = [55 * 2 ** (i / 12) for i in range(16)]
    runtime = _native.OscillatorFilterRuntime(
        48000,
        frequencies,
        [0.02] * 16,
        np.tile([[0.8, 0.6]], (16, 1)),
    )
    elapsed = []
    for start in range(0, frames, block):
        began = perf_counter_ns()
        runtime.process(
            min(block, frames - start),
            700 + start % 4000,
            0.9,
            -6 + start % 3,
        )
        elapsed.append(perf_counter_ns() - began)
    values = np.asarray(elapsed) / 1000
    deadline = block / 48000 * 1_000_000
    return Result(
        case="persistent-16-voice",
        block=block,
        blocks=len(values),
        worst_us=float(np.max(values)),
        p99_us=float(np.percentile(values, 99)),
        deadline_us=deadline,
        misses=int(np.count_nonzero(values > deadline)),
    )


def benchmark(case: str, block: int, options: Options) -> Result:
    rate, frames = 48000, round(options.seconds * 48000)
    sources = [source(frames) for _ in range(options.chains)]
    graph = effects.prepare(effect_graph(case, max(options.block_sizes)), rate)
    processors = [effects.OfflineEffects(graph, "native") for _ in sources]
    source_cursors = [0] * len(sources)
    elapsed: list[int] = []
    for start in range(0, frames, block):
        end = min(frames, start + block)
        began = perf_counter_ns()
        for i, ((renderer, actions), processor) in enumerate(
            zip(sources, processors, strict=True)
        ):
            first = source_cursors[i]
            while (
                source_cursors[i] < len(actions)
                and actions[source_cursors[i]].tick < end
            ):
                source_cursors[i] += 1
            effect_actions = controls(case, start, end, frames)
            audio = renderer.advance(actions[first : source_cursors[i]], start, end)
            if case == "subnormal":
                audio *= 1e-310
            processor.advance({"main": audio}, effect_actions, start, end)
        elapsed.append(perf_counter_ns() - began)
    values = np.asarray(elapsed) / 1000
    deadline = block / rate * 1_000_000
    return Result(
        case=case,
        block=block,
        blocks=len(values),
        worst_us=float(np.max(values)),
        p99_us=float(np.percentile(values, 99)),
        deadline_us=deadline,
        misses=int(np.count_nonzero(values > deadline)),
    )


def source(frames: int) -> tuple[synth.OfflineSynth, list[TraceAction]]:
    score, _ = Patch(engine="synth", gain=0.04).prepare_score(48000)
    assert isinstance(score, SynthInstrumentScore)
    events: list[Trigger | Release | ControlChange] = [
        Trigger(
            tick=0,
            ordinal=i,
            part="main",
            trigger_id=f"voice-{i}",
            key=key,
            pitch_hz=440 * 2 ** ((key - 69) / 12),
        )
        for i, key in enumerate((48, 55, 60, 64))
    ]
    actions: list[TraceAction] = list(
        synth_trace.prepare(score.body, events, seed=0).actions
    )
    return synth.OfflineSynth(synth.prepare(score), "native"), actions


def controls(
    case: str, frame: int, end: int, frames: int
) -> list[audio_effects.EffectAction]:
    transitions = [v for v in (frames // 3, 2 * frames // 3) if frame <= v < end]
    if case == "freeze" and transitions:
        return [
            audio_effects.FreezeAction(
                tick=frame,
                ordinal=0,
                processor="cloud",
                frozen=transitions[0] < frames // 2,
            )
        ]
    if case == "filter-tail" and frame <= frames // 2 < end:
        return [audio_effects.InputEndAction(tick=frame, ordinal=0, input="main")]
    if case == "filter-actions":
        return [
            audio_effects.ParameterAction(
                tick=frame,
                ordinal=i,
                processor="tone",
                parameter="low-cutoff_hz",
                value=700 + (frame + i * 47) % 4000,
                duration_frames=i % 31,
            )
            for i in range(64)
        ]
    processor, parameter, value = (
        ("tone", "low-cutoff_hz", 700 + frame % 4000)
        if case in ("filter", "filter+granulator", "subnormal", "filter-tail")
        else ("cloud", "density_hz", 40 + frame % 20)
        if case in ("granulator", "freeze")
        else ("left", "gain_db", -6 + frame % 3)
    )
    return [
        audio_effects.ParameterAction(
            tick=frame,
            ordinal=0,
            processor=processor,
            parameter=parameter,
            value=value,
            duration_frames=frame % 31,
        )
    ]


def effect_graph(case: str, maximum: int) -> audio_effects.EffectGraph:
    input = audio_effects.GraphInput(
        name="main",
        channels=["left", "right"],
        source=audio_effects.MainStream(),
    )
    tone = audio_effects.Filter(
        name="tone",
        filters=[
            ResonantFilter(
                name="low",
                response=FilterResponse.lowpass,
                cutoff_hz=4000,
                q=0.9,
            )
        ],
    )
    cloud = audio_effects.Granulator(
        name="cloud",
        duration_seconds=0.035,
        density_hz=48,
        lookback_seconds=0.05,
        playback_ratio=0.8,
        position_jitter_seconds=0.008,
        history_seconds=0.1,
        maximum_grains=4,
        mix=0.4,
    )
    if case in ("filter", "filter-actions", "subnormal", "filter-tail"):
        return effects.serial_graph(
            audio_effects.AttachmentScope.master, input, [tone], maximum
        )
    if case in ("granulator", "freeze"):
        return effects.serial_graph(
            audio_effects.AttachmentScope.master, input, [cloud], maximum
        )
    if case == "filter+granulator":
        return effects.serial_graph(
            audio_effects.AttachmentScope.master, input, [tone, cloud], maximum
        )
    processors: list[audio_effects.Processor] = [
        audio_effects.Gain(name="left", gain_db=-6),
        audio_effects.Gain(name="right", gain_db=-3),
        audio_effects.Multiply(name="ring"),
    ]
    return audio_effects.EffectGraph(
        scope=audio_effects.AttachmentScope.master,
        inputs=[input],
        processors=processors,
        connections=[
            audio_effects.Connection(
                processor=name,
                port="input",
                source=audio_effects.InputSource(input="main"),
            )
            for name in ("left", "right")
        ]
        + [
            audio_effects.Connection(
                processor="ring",
                port=port,
                source=audio_effects.ProcessorSource(processor=name),
            )
            for port, name in (("carrier", "left"), ("modulator", "right"))
        ],
        output=audio_effects.ProcessorSource(processor="ring"),
        maximum_block_frames=maximum,
    )


def report(options: Options, results: list[Result]) -> str:
    rows = "\n".join(
        f"| {r.case} | {r.block} | {r.worst_us:.1f} | {r.p99_us:.1f} | "
        f"{r.deadline_us:.1f} | {r.misses}/{r.blocks} |"
        for r in results
    )
    return f"""# Live effects benchmark

Generated by `scripts/benchmark-effects.py` on {platform.platform()} with Python
{platform.python_version()}. The run used 48 kHz, {options.chains} parallel chains,
four active oscillator voices per chain, changing controls in every block, and
{options.seconds:g} second per case. The committed measurements used a Rust release
build produced by `maturin develop --release`.

| Case | Frames | Worst us | p99 us | Deadline us | Misses |
|---|---:|---:|---:|---:|---:|
{rows}

`persistent-16-voice` owns all oscillator phases, routing, filter integrators, and
gain state in one Rust object and crosses Python once per block. `filter`,
`filter-actions`, `subnormal`, and `fanout` execute one whole graph per
Rust call. The action case submits the maximum 64 actions at every block boundary.
The subnormal case scales source output by `1e-310`. `filter-tail` ends every graph
halfway through the run and measures the reference tail path. `granulator` executes
the Rust granular kernel and preserves state across calls. `filter+granulator`
currently uses the NumPy graph because the native graph kernel does not yet compose
granulation with other processors. `freeze` changes frozen history in the NumPy
reference because native freeze transition state is not implemented. These two
rows expose incomplete native paths and cannot support an allocation-free callback
claim.

This harness includes Python source/action orchestration around native DSP and
therefore measures the current callable system rather than a GIL-free native host.
It does not test a lock-free action queue, bounded callback error status, tail-pool
retirement, a physical audio device, or thread-local subnormal handling. Those
items remain required before enge can claim live callback readiness.
"""


if __name__ == "__main__":
    main()
