from pathlib import Path
from typing import Literal

import numpy as np
import pytest
from test_effects import graph_input
from test_synth import check_audio
from ufor import audio_effects

from enge import _native, effects


def compressor_graph(
    detector_channels: int = 1, attack: float = 0, release: float = 0
) -> audio_effects.EffectGraph:
    return audio_effects.EffectGraph(
        scope=audio_effects.AttachmentScope.master,
        inputs=[
            graph_input(),
            audio_effects.GraphInput(
                name="side",
                channels=["mono"] if detector_channels == 1 else ["left", "right"],
                source=audio_effects.HostStream(stream="external"),
            ),
        ],
        processors=[
            audio_effects.Compressor(
                name="duck",
                threshold_db=-12,
                ratio=4,
                knee_db=6,
                attack_seconds=attack,
                release_seconds=release,
            )
        ],
        connections=[
            audio_effects.Connection(
                processor="duck", port=p, source=audio_effects.InputSource(input=i)
            )
            for p, i in (("input", "main"), ("detector", "side"))
        ],
        output=audio_effects.ProcessorSource(processor="duck"),
        maximum_block_frames=997,
    )


def compressor_audio(
    prepared: effects.PreparedEffects,
    inputs: dict[str, np.ndarray],
    actions: list[audio_effects.EffectAction],
    backend: Literal["numpy", "native", "callback"],
) -> tuple[np.ndarray, np.ndarray]:
    runtime = _native.LiveRuntime.empty(48000, 2, 997, 128)
    renderer = effects.OfflineEffects(
        prepared, "numpy" if backend == "numpy" else "native"
    )
    if backend == "callback":
        runtime.set_effect_graph(
            *effects.native_live_graph(prepared), 4, input_routes=[[-2], [-2]]
        )
        effects.configure_native_generators(runtime, prepared)
    actual = np.empty((48000, 2))
    for start in range(0, 48000, 997):
        end = min(start + 997, 48000)
        batch = [a for a in actions if start <= a.tick < end]
        if backend == "callback":
            if batch:
                runtime.submit_effects(
                    effects.native_live_actions(prepared, batch, start, end)
                )
            runtime.process_into(
                actual[start:end],
                [inputs[i.name][start:end] for i in prepared.definition.inputs],
            )
        else:
            actual[start:end] = renderer.advance(
                {n: a[start:end] for n, a in inputs.items()}, batch, start, end
            )
        if end == 23928:
            saved = runtime.snapshot() if backend == "callback" else renderer.snapshot()
    if backend == "callback":
        runtime.restore(saved)
    else:
        renderer.restore(
            effects.EffectSnapshot.model_validate_json(saved.model_dump_json())
        )
    replay = actual.copy()
    for start in range(23928, 48000, 997):
        end = min(start + 997, 48000)
        batch = [a for a in actions if start <= a.tick < end]
        if backend == "callback":
            if batch:
                runtime.submit_effects(
                    effects.native_live_actions(prepared, batch, start, end)
                )
            runtime.process_into(
                replay[start:end],
                [inputs[i.name][start:end] for i in prepared.definition.inputs],
            )
        else:
            replay[start:end] = renderer.advance(
                {n: a[start:end] for n, a in inputs.items()}, batch, start, end
            )
    return actual, replay


@pytest.mark.parametrize("implementation", ["numpy", "native", "callback"])
@pytest.mark.parametrize("detector_channels", [1, 2])
def test_compressor_soft_knee_matches_static_transfer_and_links_channels(
    tmp_path: Path,
    implementation: Literal["numpy", "native", "callback"],
    detector_channels: int,
) -> None:
    peaks = np.repeat(10 ** (np.array([-18, -15, -12, -9, 0]) / 20), 8000)
    peaks = np.r_[peaks, np.zeros(8000)]
    detector = (
        peaks[:, None]
        if detector_channels == 1
        else np.column_stack((-peaks, peaks / 2))
    )
    time = np.arange(48000) / 48000
    carrier = np.column_stack(
        (0.2 * np.sin(2 * np.pi * 220 * time), 0.1 * np.cos(2 * np.pi * 330 * time))
    )
    reduction = np.repeat([0, 0, -0.5625, -2.25, -9, 0], 8000)
    expected = carrier * 10 ** (reduction[:, None] / 20)
    actual, replay = compressor_audio(
        effects.prepare(compressor_graph(detector_channels), 48000),
        {"main": carrier, "side": detector},
        [],
        implementation,
    )
    check_audio(tmp_path / "knee.wav", actual, expected)
    check_audio(tmp_path / "replay.wav", replay, actual)


@pytest.mark.parametrize("implementation", ["numpy", "native", "callback"])
def test_compressor_attack_and_release_follow_declared_time_constants(
    tmp_path: Path, implementation: Literal["numpy", "native", "callback"]
) -> None:
    attack = np.exp(-1 / (48000 * 0.01))
    release = np.exp(-1 / (48000 * 0.1))
    first = -9 * (1 - attack ** np.arange(1, 16001))
    second = first[-1] * release ** np.arange(1, 16001)
    third = -9 + (second[-1] + 9) * attack ** np.arange(1, 16001)
    carrier = np.tile([0.25, -0.5], (48000, 1))
    detector = np.r_[np.ones(16000), np.zeros(16000), np.ones(16000)][:, None]
    expected = carrier * 10 ** (np.r_[first, second, third][:, None] / 20)
    actual, replay = compressor_audio(
        effects.prepare(compressor_graph(attack=0.01, release=0.1), 48000),
        {"main": carrier, "side": detector},
        [],
        implementation,
    )
    check_audio(tmp_path / "ballistics.wav", actual, expected)
    check_audio(tmp_path / "replay.wav", replay, actual)


@pytest.mark.parametrize("implementation", ["numpy", "native", "callback"])
def test_compressor_envelfo_threshold_and_automation_follow_same_frame_clock(
    tmp_path: Path, implementation: Literal["numpy", "native", "callback"]
) -> None:
    raw = compressor_graph().model_dump(mode="json")
    raw["envelfos"] = {
        "rise": {
            "kind": "envelope",
            "initial": 0,
            "segments": [{"duration": "1 s", "to": 1}],
        }
    }
    raw["bindings"] = [{"kind": "envelfo", "name": "rise", "reference": "rise"}]
    raw["modulation"] = {
        "sources": [{"name": "rise", "scope": "voice", "minimum": 0, "maximum": 1}],
        "parameters": [
            {
                "target": {"name": "duck", "parameter": "threshold_db"},
                "scope": "voice",
                "unit": "db",
                "minimum": -12,
                "maximum": 0,
                "default": -12,
            }
        ],
        "routes": [
            {
                "name": "rise",
                "source": "rise",
                "target": {"name": "duck", "parameter": "threshold_db"},
                "unit": "db",
                "operation": "add",
                "points": [{"input": 0, "amount": 0}, {"input": 1, "amount": 12}],
            }
        ],
    }
    raw["processors"][0]["knee_db"] = 0
    graph = audio_effects.EffectGraph.model_validate(raw)
    makeup = np.minimum(np.maximum(np.arange(48000) - 12000, 0), 6000) / 6000 * 6
    bypass = np.ones(48000)
    bypass[20000:26000] = 1 - np.minimum(np.arange(6000) / 64, 1)
    bypass[26000:] = np.minimum(np.arange(22000) / 64, 1)
    wet_gain = 10 ** ((0.75 * (-12 + 12 * np.arange(48000) / 48000) + makeup) / 20)
    carrier = np.tile([0.25, -0.5], (48000, 1))
    expected = carrier * ((1 - bypass) + bypass * wet_gain)[:, None]
    actions: list[audio_effects.EffectAction] = [
        audio_effects.ParameterAction(
            tick=12000,
            ordinal=0,
            processor="duck",
            parameter="makeup_db",
            value=6,
            duration_frames=6000,
        ),
        audio_effects.BypassAction(
            tick=20000, ordinal=0, processor="duck", bypassed=True
        ),
        audio_effects.BypassAction(
            tick=26000, ordinal=0, processor="duck", bypassed=False
        ),
    ]
    actual, replay = compressor_audio(
        effects.prepare(graph, 48000),
        {"main": carrier, "side": np.ones((48000, 1))},
        actions,
        implementation,
    )
    check_audio(tmp_path / "modulation.wav", actual, expected)
    check_audio(tmp_path / "replay.wav", replay, actual)


def test_native_compressor_failure_silences_block_and_snapshot_recovers(
    tmp_path: Path,
) -> None:
    graph = compressor_graph().model_copy(update={"maximum_block_frames": 48000})
    prepared = effects.prepare(graph, 48000)
    runtime = _native.LiveRuntime.empty(48000, 2, 48000, 64)
    runtime.set_effect_graph(
        *effects.native_live_graph(prepared), 2, input_routes=[[-2], [-2]]
    )
    saved = runtime.snapshot()
    runtime.submit_effects(np.array([[25000, 0, 0, 3, 0.5, 0]], dtype=np.float64))
    inputs = [np.full((48000, 2), 0.25), np.ones((48000, 1))]
    actual = np.ones((48000, 2))
    with pytest.raises(ValueError, match="Invalid compressor"):
        runtime.process_into(actual, inputs)
    assert runtime.failed()
    check_audio(tmp_path / "failed.wav", actual, np.zeros_like(actual))
    runtime.restore(saved)
    assert not runtime.failed()
    runtime.process_into(actual, inputs)
    check_audio(tmp_path / "recovered.wav", actual, inputs[0] * 10 ** (-9 / 20))
