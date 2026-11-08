from pathlib import Path

import numpy as np
import pytest
from test_effects import graph_input
from test_envelfo_slew import slew_score
from test_live import native_oscillator_runtime
from test_synth import check_audio
from ufor import audio_effects
from ufor.control import TempoMap
from ufor.samples import processing
from waveshaper_oracle import waveshaper_audio

from enge import _native, effects


def modulated_graph(kind: str) -> tuple[audio_effects.EffectGraph, str, float, float]:
    processors: dict[str, audio_effects.Processor] = {
        "gain": audio_effects.Gain(name="fx", gain_db=-6),
        "clip": audio_effects.Waveshaper(name="fx", drive=2),
        "phaser": audio_effects.Phaser(name="fx", feedback=0.5),
        "highpass": audio_effects.Highpass(name="fx", cutoff_hz=1000),
        "gate": audio_effects.Gate(name="fx"),
        "expander": audio_effects.Expander(name="fx"),
        "filter": audio_effects.Filter(
            name="fx",
            filters=[
                processing.ResonantFilter(
                    name="low", response="lowpass", cutoff_hz=1000, q=0.7
                )
            ],
        ),
        "grain": audio_effects.Granulator(
            name="fx",
            duration_seconds=0.02,
            density_hz=30,
            history_seconds=0.05,
            maximum_grains=4,
        ),
        "tap": audio_effects.TapDelay(
            name="fx", delay_seconds=0.005, maximum_delay_seconds=0.01, feedback=0.25
        ),
        "damped-tap": audio_effects.TapDelay(
            name="fx",
            delay_seconds=0.005,
            maximum_delay_seconds=0.01,
            feedback=0.5,
            damping_hz=1200,
        ),
        "chorus": audio_effects.ModulatedDelay(
            name="fx",
            base_delay_seconds=0.005,
            depth_seconds=0,
            maximum_delay_seconds=0.01,
            rate_hz=1,
            feedback=0.25,
        ),
        "multiply": audio_effects.Multiply(name="fx", mix=0.5),
    }
    parameter, default, depth, unit = {
        "gain": ("gain_db", -6, 3, "db"),
        "clip": ("drive", 2, 0.5, "ratio"),
        "phaser": ("sweep", 0, 0.5, "ratio"),
        "highpass": ("cutoff_hz", 1000, 250, "hz"),
        "gate": ("hold_seconds", 0.01, 0.005, "seconds"),
        "expander": ("ratio", 2, 0.5, "ratio"),
        "filter": ("low-cutoff_hz", 1000, 250, "hz"),
        "grain": ("duration_seconds", 0.02, 0.005, "seconds"),
        "tap": ("delay_seconds", 0.005, 0.001, "seconds"),
        "damped-tap": ("damping_hz", 1200, 400, "hz"),
        "chorus": ("base_delay_seconds", 0.005, 0.001, "seconds"),
        "multiply": ("mix", 0.5, 0.25, "ratio"),
    }[kind]
    raw = {
        "scope": "master",
        "inputs": [graph_input().model_dump()],
        "processors": [processors[kind].model_dump()],
        "connections": [
            {"processor": "fx", "port": p, "source": {"kind": "input", "input": "main"}}
            for p in sorted(audio_effects.processor_ports(processors[kind]))
        ],
        "output": {"kind": "processor", "processor": "fx"},
        "maximum_block_frames": 48000,
        "envelfos": {"motion": {"kind": "lfo", "shape": "square", "rate": 2}},
        "bindings": [{"kind": "envelfo", "name": "motion", "reference": "motion"}],
        "modulation": {
            "sources": [
                {"name": "motion", "scope": "voice", "minimum": -1, "maximum": 1}
            ],
            "parameters": [
                {
                    "target": {"name": "fx", "parameter": parameter},
                    "unit": unit,
                    "scope": "voice",
                    "minimum": default - 2 * depth,
                    "maximum": default + 2 * depth,
                    "default": default,
                }
            ],
            "routes": [
                {
                    "name": "motion",
                    "source": "motion",
                    "target": {"name": "fx", "parameter": parameter},
                    "unit": unit,
                    "operation": "add",
                    "points": [
                        {"input": -1, "amount": -depth},
                        {"input": 1, "amount": depth},
                    ],
                }
            ],
        },
    }
    return audio_effects.EffectGraph.model_validate(raw), parameter, default, depth


@pytest.mark.parametrize(
    "kind",
    [
        "gain",
        "clip",
        "phaser",
        "highpass",
        "gate",
        "expander",
        "filter",
        "grain",
        "tap",
        "damped-tap",
        "chorus",
        "multiply",
    ],
)
def test_effect_envelfo_matches_automation_and_native_callback_replay(
    tmp_path: Path, kind: str
) -> None:
    graph, parameter, default, depth = modulated_graph(kind)
    raw = graph.model_dump()
    raw.update(envelfos={}, bindings=[], modulation={})
    static = effects.prepare(audio_effects.EffectGraph.model_validate(raw), 48000)
    start_action = np.array([[0, 0, 0, 220, 0.2, 0]], dtype=np.float64)
    source = native_oscillator_runtime().process_actions(48000, 0, 1, 0, start_action)
    actions = [
        audio_effects.ParameterAction(
            tick=t,
            ordinal=0,
            processor="fx",
            parameter=parameter,
            value=default + depth * s,
        )
        for t, s in [(0, 1), (12000, -1), (24000, 1), (36000, -1)]
    ]
    expected = effects.OfflineEffects(static).advance(
        {"main": source}, actions, 0, 48000
    )
    prepared = effects.prepare(graph, 48000)
    for backend in ("numpy", "native"):
        renderer = effects.OfflineEffects(prepared, backend)
        head = renderer.advance({"main": source[:17999]}, [], 0, 17999)
        saved = effects.EffectSnapshot.model_validate_json(
            renderer.snapshot().model_dump_json()
        )
        renderer.restore(saved)
        actual = np.concatenate(
            [head, renderer.advance({"main": source[17999:]}, [], 17999, 48000)]
        )
        check_audio(tmp_path / f"{backend}-envelfo.wav", actual, expected)

    runtime = _native.LiveRuntime([native_oscillator_runtime()], 997, 64, 4)
    runtime.set_effect_graph(*effects.native_live_graph(prepared), 4)
    effects.configure_native_generators(runtime, prepared)
    runtime.submit(0, start_action)
    actual = np.empty_like(source)
    for first, end in [(0, 17999), (17999, 48000)]:
        if first:
            saved_native = runtime.snapshot()
        for start in range(first, end, 997):
            runtime.process_into(actual[start : min(start + 997, end)])
    runtime.restore(saved_native)
    replay = actual.copy()
    for start in range(17999, 48000, 997):
        runtime.process_into(replay[start : min(start + 997, 48000)])
    check_audio(tmp_path / "callback-envelfo.wav", actual, expected)
    check_audio(tmp_path / "callback-replay.wav", replay, actual)


@pytest.mark.parametrize("clock", ["seconds", "beats"])
def test_effect_patch_retains_slew_and_tempo_clock_across_backends(
    tmp_path: Path, clock: str
) -> None:
    graph, _, _, _ = modulated_graph("clip")
    raw = graph.model_dump()
    raw["envelfos"] = {
        "motion": slew_score(8, 4, 0.5, clock)
        .body.voices[0]
        .envelfos["gesture"]
        .model_dump()
    }
    raw["bindings"][0]["output"] = "value"
    raw["modulation"]["sources"][0]["minimum"] = 0
    raw["modulation"]["parameters"][0]["maximum"] = 4
    raw["modulation"]["routes"][0]["points"] = [
        {"input": 0, "amount": 0},
        {"input": 1, "amount": 1},
    ]
    graph = audio_effects.EffectGraph.model_validate(raw)
    tempo = (
        TempoMap.model_validate(
            {
                "points": [
                    {"at_seconds": "0", "beat": "0", "bpm": "120"},
                    {
                        "at_seconds": "1/8",
                        "beat": "1/4",
                        "bpm": "120",
                        "running": False,
                    },
                ]
            }
        )
        if clock == "beats"
        else None
    )
    prepared = effects.prepare(graph, 48000, tempo)
    inputs = (
        np.where(np.arange(48000) >= 6000, 1.0, -1.0)
        if clock == "beats"
        else np.tile(np.repeat([-1.0, 1.0], 12000), 2)
    )
    smooth = np.empty(48000)
    smooth[0] = 0.5
    for i in range(1, 48000):
        smooth[i] = (
            min(inputs[i], smooth[i - 1] + 8 / 48000)
            if inputs[i] >= smooth[i - 1]
            else max(inputs[i], smooth[i - 1] - 4 / 48000)
        )
    start_action = np.array([[0, 0, 0, 220, 0.2, 0]], dtype=np.float64)
    source = native_oscillator_runtime().process_actions(48000, 0, 1, 0, start_action)
    drive = (2.5 + 0.5 * smooth)[:, None]
    expected = waveshaper_audio(source, drive)
    for backend in ("numpy", "native"):
        renderer = effects.OfflineEffects(prepared, backend)
        head = renderer.advance({"main": source[:14003]}, [], 0, 14003)
        saved = effects.EffectSnapshot.model_validate_json(
            renderer.snapshot().model_dump_json()
        )
        renderer.restore(saved)
        actual = np.concatenate(
            [head, renderer.advance({"main": source[14003:]}, [], 14003, 48000)]
        )
        check_audio(tmp_path / f"slew-{backend}.wav", actual, expected)
    runtime = _native.LiveRuntime([native_oscillator_runtime()], 997, 64, 4)
    runtime.set_effect_graph(*effects.native_live_graph(prepared), 4)
    effects.configure_native_generators(runtime, prepared)
    runtime.submit(0, start_action)
    actual = np.empty_like(source)
    for first, end in [(0, 14003), (14003, 48000)]:
        if first:
            saved_native = runtime.snapshot()
        for start in range(first, end, 997):
            runtime.process_into(actual[start : min(start + 997, end)])
    runtime.restore(saved_native)
    replay = actual.copy()
    for start in range(14003, 48000, 997):
        runtime.process_into(replay[start : min(start + 997, 48000)])
    check_audio(tmp_path / "slew-callback.wav", actual, expected)
    check_audio(tmp_path / "slew-replay.wav", replay, actual)


def test_effect_routes_compose_over_an_automated_base_value(tmp_path: Path) -> None:
    graph, _, _, _ = modulated_graph("gain")
    raw = graph.model_dump()
    raw["modulation"]["parameters"][0].update(minimum=-24, maximum=6)
    raw["modulation"]["routes"].append(
        {
            "name": "scale",
            "source": "motion",
            "target": {"name": "fx", "parameter": "gain_db"},
            "unit": "ratio",
            "operation": "multiply",
            "points": [{"input": -1, "amount": 0.5}, {"input": 1, "amount": 1}],
        }
    )
    prepared = effects.prepare(audio_effects.EffectGraph.model_validate(raw), 48000)
    start_action = np.array([[0, 0, 0, 220, 0.2, 0]], dtype=np.float64)
    source = native_oscillator_runtime().process_actions(48000, 0, 1, 0, start_action)
    action = audio_effects.ParameterAction(
        tick=10001,
        ordinal=0,
        processor="fx",
        parameter="gain_db",
        value=-9,
        duration_frames=6000,
    )
    frames = np.arange(48000)
    base = -6 - 3 * np.clip((frames - 10001) / 6000, 0, 1)
    positive = (frames // 12000) % 2 == 0
    gain = (base + np.where(positive, 3, -3)) * np.where(positive, 1, 0.5)
    expected = source * 10 ** (gain[:, None] / 20)
    actual = effects.OfflineEffects(prepared).advance(
        {"main": source}, [action], 0, 48000
    )
    check_audio(tmp_path / "automated-base.wav", actual, expected)
    runtime = _native.LiveRuntime([native_oscillator_runtime()], 997, 64, 4)
    runtime.set_effect_graph(*effects.native_live_graph(prepared), 4)
    effects.configure_native_generators(runtime, prepared)
    runtime.submit(0, start_action)
    actual = np.empty_like(source)
    for start in range(0, 48000, 997):
        end = min(start + 997, 48000)
        if start <= action.tick < end:
            runtime.submit_effects(
                effects.native_live_actions(prepared, [action], start, end)
            )
        runtime.process_into(actual[start:end])
    check_audio(tmp_path / "automated-base-callback.wav", actual, expected)


@pytest.mark.parametrize("end_tick", [0, 24000])
def test_effect_envelope_releases_over_a_delay_tail_and_restores(
    tmp_path: Path,
    end_tick: int,
) -> None:
    graph, _, _, _ = modulated_graph("gain")
    raw = graph.model_dump()
    raw["processors"].insert(
        0,
        audio_effects.TapDelay(
            name="echo", delay_seconds=0.05, maximum_delay_seconds=0.05, feedback=0.8
        ).model_dump(),
    )
    raw["connections"] = [
        {
            "processor": "echo",
            "port": "input",
            "source": {"kind": "input", "input": "main"},
        },
        {
            "processor": "fx",
            "port": "input",
            "source": {"kind": "processor", "processor": "echo"},
        },
    ]
    raw["envelfos"] = {
        "motion": {
            "kind": "envelope",
            "initial": 0,
            "segments": [{"duration": "1/4 s", "to": 1}, {"duration": None}],
            "release": [{"duration": "1/4 s", "to": 0}],
        }
    }
    raw["modulation"]["sources"][0]["minimum"] = 0
    raw["modulation"]["routes"][0]["points"] = [
        {"input": 0, "amount": -6},
        {"input": 1, "amount": 6},
    ]
    raw["modulation"]["parameters"][0].update(minimum=-12, maximum=0)
    prepared = effects.prepare(audio_effects.EffectGraph.model_validate(raw), 48000)
    raw.update(envelfos={}, bindings=[], modulation={})
    raw["processors"][1]["gain_db"] = 0
    static = effects.prepare(audio_effects.EffectGraph.model_validate(raw), 48000)
    source = np.column_stack([np.sin(2 * np.pi * np.arange(48000) / 97)] * 2)
    action = audio_effects.InputEndAction(tick=end_tick, ordinal=0, input="main")
    expected = effects.OfflineEffects(static).advance(
        {"main": source}, [action], 0, 48000
    )
    env = np.minimum(np.arange(48000) / 12000, 1)
    env[24000:] = np.maximum(1 - np.arange(24000) / 12000, 0)
    if end_tick == 0:
        env[:] = 0
    expected *= 10 ** ((-12 + 12 * env[:, None]) / 20)
    for backend in ("numpy", "native"):
        renderer = effects.OfflineEffects(prepared, backend)
        head = renderer.advance({"main": source[:27001]}, [action], 0, 27001)
        saved = effects.EffectSnapshot.model_validate_json(
            renderer.snapshot().model_dump_json()
        )
        renderer.restore(saved)
        actual = np.concatenate(
            [head, renderer.advance({"main": source[27001:]}, [], 27001, 48000)]
        )
        check_audio(tmp_path / f"release-{backend}.wav", actual, expected)


def test_modulated_callback_failure_is_silent_and_retained(tmp_path: Path) -> None:
    graph, _, _, _ = modulated_graph("gain")
    prepared = effects.prepare(graph, 48000)
    runtime = _native.LiveRuntime([native_oscillator_runtime()], 48000, 64, 4)
    runtime.set_effect_graph(*effects.native_live_graph(prepared), 4)
    effects.configure_native_generators(runtime, prepared)
    runtime.submit(0, np.array([[0, 0, 0, 220, 0.2, 0]], dtype=np.float64))
    action = audio_effects.ParameterAction(
        tick=0, ordinal=0, processor="fx", parameter="gain_db", value=0
    )
    runtime.submit_effects(effects.native_live_actions(prepared, [action], 0, 48000))
    actual = np.ones((48000, 2))
    with pytest.raises(ValueError, match="outside its domain"):
        runtime.process_into(actual)
    assert runtime.failed()
    check_audio(tmp_path / "failed-callback.wav", actual, np.zeros_like(actual))


@pytest.mark.parametrize("minimum,maximum", [(1, 1500), (500, 30000)])
def test_native_filter_modulation_uses_authored_clamping_limits(
    tmp_path: Path, minimum: int, maximum: int
) -> None:
    graph, _, _, _ = modulated_graph("filter")
    raw = graph.model_dump()
    raw["processors"][0]["filters"][0].update(
        boundary="clamp", minimum_hz=20, nyquist_ratio=0.9
    )
    raw["modulation"]["parameters"][0].update(minimum=minimum, maximum=maximum)
    raw["modulation"]["routes"][0]["points"] = [
        {"input": -1, "amount": minimum - 1000},
        {"input": 1, "amount": maximum - 1000},
    ]
    prepared = effects.prepare(audio_effects.EffectGraph.model_validate(raw), 48000)
    runtime = _native.LiveRuntime([native_oscillator_runtime()], 997, 64, 4)
    runtime.set_effect_graph(*effects.native_live_graph(prepared), 4)
    effects.configure_native_generators(runtime, prepared)
    onset = np.array([[0, 0, 0, 220, 0.2, 0]], dtype=np.float64)
    source = native_oscillator_runtime().process_actions(48000, 0, 1, 0, onset)
    expected = effects.OfflineEffects(prepared).advance({"main": source}, [], 0, 48000)
    runtime.submit(0, onset)
    actual = np.empty_like(expected)
    for start in range(0, 48000, 997):
        runtime.process_into(actual[start : min(start + 997, 48000)])
    check_audio(tmp_path / "clamped-filter.wav", actual, expected)
