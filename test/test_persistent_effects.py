from pathlib import Path
from typing import Literal

import numpy as np
import pytest
from test_compressor import compressor_graph
from test_effect_envelfo import modulated_graph
from test_effects import graph_input
from test_synth import check_audio
from ufor import audio_effects

from enge import _native, effects


def graph_audio(graph: audio_effects.EffectGraph) -> dict[str, np.ndarray]:
    time = np.arange(48000) / 48000
    return {
        i.name: np.column_stack(
            [
                0.5 + 0.5 * np.sin(2 * np.pi * (7 + c) * time)
                if isinstance(i.source, audio_effects.HostStream)
                else 0.2 * np.sin(2 * np.pi * (220 + 110 * c) * time)
                for c in range(len(i.channels))
            ]
        )
        for i in graph.inputs
    }


def callback_audio(
    prepared: effects.PreparedEffects,
    inputs: dict[str, np.ndarray],
    actions: list[audio_effects.EffectAction],
) -> tuple[np.ndarray, np.ndarray, bool]:
    graph = prepared.definition
    channels = len(
        next(
            i.channels
            for i in graph.inputs
            if isinstance(i.source, audio_effects.MainStream)
        )
    )
    runtime = _native.LiveRuntime.empty(48000, channels, 997, 128)
    runtime.set_effect_graph(
        *effects.native_live_graph(prepared),
        4,
        input_routes=[[-2] for _ in graph.inputs],
    )
    effects.configure_native_generators(runtime, prepared)
    actual = np.empty((48000, len(prepared.channels)))
    for first, end in ((0, 23928), (23928, 48000)):
        if first:
            saved = runtime.snapshot()
        for start in range(first, end, 997):
            stop = min(start + 997, end)
            batch = [a for a in actions if start <= a.tick < stop]
            if batch:
                runtime.submit_effects(
                    effects.native_live_actions(prepared, batch, start, stop)
                )
            runtime.process_into(
                actual[start:stop], [inputs[i.name][start:stop] for i in graph.inputs]
            )
    drained = runtime.effects_drained
    runtime.restore(saved)
    replay = actual.copy()
    for start in range(23928, 48000, 997):
        stop = min(start + 997, 48000)
        batch = [a for a in actions if start <= a.tick < stop]
        if batch:
            runtime.submit_effects(
                effects.native_live_actions(prepared, batch, start, stop)
            )
        runtime.process_into(
            replay[start:stop], [inputs[i.name][start:stop] for i in graph.inputs]
        )
    assert runtime.effects_drained == drained
    return actual, replay, drained


@pytest.mark.parametrize(
    "kind", ["gain", "clip", "phaser", "filter", "grain", "tap", "chorus", "multiply"]
)
def test_native_input_end_releases_generators_and_drains_the_graph(
    tmp_path: Path, kind: str
) -> None:
    graph, _, _, depth = modulated_graph(kind)
    raw = graph.model_dump(mode="json")
    if kind == "grain":
        raw["processors"][0].update(lookback_seconds=0.005, density_hz=90)
    raw["envelfos"]["motion"] = {
        "kind": "envelope",
        "initial": 0,
        "segments": [{"duration": "1/4 s", "to": 1}, {"duration": None}],
        "release": [{"duration": "1/4 s", "to": 0}],
    }
    raw["modulation"]["sources"][0]["minimum"] = 0
    raw["modulation"]["routes"][0]["points"] = [
        {"input": 0, "amount": -depth},
        {"input": 1, "amount": depth},
    ]
    graph = audio_effects.EffectGraph.model_validate(raw)
    prepared = effects.prepare(graph, 48000)
    inputs = graph_audio(graph)
    actions: list[audio_effects.EffectAction] = [
        audio_effects.InputEndAction(tick=12003, ordinal=0, input="main")
    ]
    reference = effects.OfflineEffects(prepared)
    expected = reference.advance(inputs, actions, 0, 48000)
    assert np.max(np.abs(expected[:12003])) > 0.01
    assert reference.drained
    native = effects.OfflineEffects(prepared, "native")
    offline = native.advance(inputs, actions, 0, 48000)
    assert native.drained
    actual, replay, drained = callback_audio(prepared, inputs, actions)
    assert drained
    check_audio(tmp_path / "offline-end.wav", offline, expected)
    check_audio(tmp_path / "callback-end.wav", actual, expected)
    check_audio(tmp_path / "callback-replay.wav", replay, actual)


def test_native_stop_ends_a_frozen_tail_and_keeps_later_blocks_silent(
    tmp_path: Path,
) -> None:
    graph, _, _, _ = modulated_graph("grain")
    raw = graph.model_dump(mode="json")
    raw["processors"][0].update(lookback_seconds=0.005, density_hz=90)
    graph = audio_effects.EffectGraph.model_validate(raw)
    prepared = effects.prepare(graph, 48000)
    inputs = graph_audio(graph)
    actions: list[audio_effects.EffectAction] = [
        audio_effects.FreezeAction(tick=9001, ordinal=0, processor="fx", frozen=True),
        audio_effects.InputEndAction(tick=12003, ordinal=0, input="main"),
        audio_effects.StopAction(tick=36017, ordinal=0),
    ]
    expected = effects.OfflineEffects(prepared).advance(inputs, actions, 0, 48000)
    assert np.max(np.abs(expected[20000:36017])) > 0.01
    assert not np.any(expected[36017:])
    actual, replay, drained = callback_audio(prepared, inputs, actions)
    assert drained
    check_audio(tmp_path / "frozen-stop.wav", actual, expected)
    check_audio(tmp_path / "frozen-replay.wav", replay, actual)


@pytest.mark.parametrize(
    "kind", ["filter", "grain", "tap", "chorus", "compressor", "phaser"]
)
@pytest.mark.parametrize(
    "origin,target", [("numpy", "native"), ("native", "numpy"), ("native", "native")]
)
def test_effect_json_snapshots_preserve_dsp_across_backends(
    tmp_path: Path,
    kind: str,
    origin: Literal["numpy", "native"],
    target: Literal["numpy", "native"],
) -> None:
    graph = (
        compressor_graph(attack=0.01, release=0.1)
        if kind == "compressor"
        else modulated_graph(kind)[0]
    )
    raw = graph.model_dump(mode="json")
    raw["maximum_block_frames"] = 48000
    if kind == "grain":
        raw["processors"][0].update(
            density_hz=150,
            lookback_seconds=0.005,
            history_seconds=0.02,
            playback_ratio=0.25,
            maximum_duration_seconds=0.08,
        )
        raw["modulation"]["parameters"][0]["maximum"] = 0.08
    graph = audio_effects.EffectGraph.model_validate(raw)
    prepared = effects.prepare(graph, 48000)
    inputs = graph_audio(graph)
    name = graph.processors[0].name
    actions: list[audio_effects.EffectAction] = [
        audio_effects.ParameterAction(
            tick=17003,
            ordinal=0,
            processor=name,
            parameter="mix",
            value=0.35,
            duration_frames=7000,
        ),
        audio_effects.ParameterAction(
            tick=31007,
            ordinal=0,
            processor=name,
            parameter="mix",
            value=0.8,
            duration_frames=2300,
        ),
    ]
    if kind == "grain":
        actions.extend(
            [
                audio_effects.ParameterAction(
                    tick=15000,
                    ordinal=0,
                    processor=name,
                    parameter="duration_seconds",
                    value=0.06,
                ),
                audio_effects.FreezeAction(
                    tick=18000, ordinal=0, processor=name, frozen=True
                ),
                audio_effects.FreezeAction(
                    tick=27003, ordinal=0, processor=name, frozen=False
                ),
            ]
        )
    actions.sort(key=lambda a: (a.tick, a.ordinal))
    expected = effects.OfflineEffects(prepared, origin).advance(
        inputs, actions, 0, 48000
    )
    first = effects.OfflineEffects(prepared, origin)
    head = first.advance(
        {n: a[:23928] for n, a in inputs.items()},
        [a for a in actions if a.tick < 23928],
        0,
        23928,
    )
    saved = effects.EffectSnapshot.model_validate_json(
        first.snapshot().model_dump_json()
    )
    restored = effects.OfflineEffects(prepared, target)
    restored.restore(saved)
    tail = restored.advance(
        {n: a[23928:] for n, a in inputs.items()},
        [a for a in actions if a.tick >= 23928],
        23928,
        48000,
    )
    actual = np.concatenate((head, tail))
    check_audio(tmp_path / "portable-state.wav", actual, expected)
    if origin == target:
        np.testing.assert_array_equal(actual, expected)


def test_native_snapshot_rejection_preserves_the_running_graph(tmp_path: Path) -> None:
    graph = modulated_graph("tap")[0]
    prepared = effects.prepare(graph, 48000)
    inputs = graph_audio(graph)
    expected = effects.OfflineEffects(prepared, "native").advance(inputs, [], 0, 48000)
    renderer = effects.OfflineEffects(prepared, "native")
    head = renderer.advance({"main": inputs["main"][:23928]}, [], 0, 23928)
    saved = renderer.snapshot()
    bad = saved.model_copy(deep=True)
    delay = bad.processors["fx"].tap_delay
    assert delay is not None
    delay.history = []
    with pytest.raises(ValueError, match="snapshot matrix"):
        renderer.restore(bad)
    assert renderer.frame == 23928
    assert renderer.snapshot() == saved
    tail = renderer.advance({"main": inputs["main"][23928:]}, [], 23928, 48000)
    actual = np.concatenate((head, tail))
    check_audio(tmp_path / "rejected-state.wav", actual, expected)
    np.testing.assert_array_equal(actual, expected)


def test_mono_detector_delay_drains_independently_of_the_stereo_signal(
    tmp_path: Path,
) -> None:
    graph = audio_effects.EffectGraph(
        scope=audio_effects.AttachmentScope.master,
        inputs=[
            graph_input(),
            audio_effects.GraphInput(
                name="side",
                channels=["mono"],
                source=audio_effects.HostStream(stream="external"),
            ),
        ],
        processors=[
            audio_effects.ModulatedDelay(
                name="movement",
                base_delay_seconds=4 / 48000,
                depth_seconds=1 / 48000,
                maximum_delay_seconds=6 / 48000,
                rate_hz=2,
                channel_phase_offsets=[0.25],
                feedback=0.6,
            ),
            audio_effects.Compressor(
                name="duck",
                threshold_db=-12,
                attack_seconds=0.005,
                release_seconds=0.02,
            ),
        ],
        connections=[
            audio_effects.Connection(
                processor="movement",
                port="input",
                source=audio_effects.InputSource(input="side"),
            ),
            audio_effects.Connection(
                processor="duck",
                port="input",
                source=audio_effects.InputSource(input="main"),
            ),
            audio_effects.Connection(
                processor="duck",
                port="detector",
                source=audio_effects.ProcessorSource(processor="movement"),
            ),
        ],
        output=audio_effects.ProcessorSource(processor="duck"),
        maximum_block_frames=48000,
    )
    prepared = effects.prepare(graph, 48000)
    inputs = graph_audio(graph)
    actions: list[audio_effects.EffectAction] = [
        audio_effects.InputEndAction(tick=12003, ordinal=0, input="side"),
        audio_effects.InputEndAction(tick=42003, ordinal=0, input="main"),
    ]
    expected = effects.OfflineEffects(prepared).advance(inputs, actions, 0, 48000)
    actual, replay, drained = callback_audio(prepared, inputs, actions)
    assert drained
    native = effects.OfflineEffects(prepared, "native").advance(
        inputs, actions, 0, 48000
    )
    check_audio(tmp_path / "mono-detector.wav", actual, expected)
    check_audio(tmp_path / "offline-detector.wav", native, expected)
    check_audio(tmp_path / "detector-replay.wav", replay, actual)


@pytest.mark.parametrize("duration_frames", [2.5, 4.5])
def test_granular_duration_rounds_half_frames_to_even(
    tmp_path: Path, duration_frames: float
) -> None:
    graph = effects.serial_graph(
        audio_effects.AttachmentScope.master,
        graph_input(),
        [
            audio_effects.Granulator(
                name="cloud",
                duration_seconds=duration_frames / 48000,
                density_hz=8000,
                lookback_seconds=3 / 48000,
                history_seconds=16 / 48000,
                maximum_grains=2,
            )
        ],
        48000,
    )
    prepared = effects.prepare(graph, 48000)
    inputs = graph_audio(graph)
    expected = effects.OfflineEffects(prepared).advance(inputs, [], 0, 48000)
    assert np.max(np.abs(expected)) > 0.01
    native = effects.OfflineEffects(prepared, "native").advance(inputs, [], 0, 48000)
    actual, replay, drained = callback_audio(prepared, inputs, [])
    assert not drained
    check_audio(tmp_path / "rounded-offline.wav", native, expected)
    check_audio(tmp_path / "rounded-callback.wav", actual, expected)
    check_audio(tmp_path / "rounded-replay.wav", replay, actual)


def test_repeated_native_input_end_silences_failure_and_restores(
    tmp_path: Path,
) -> None:
    graph = modulated_graph("tap")[0]
    prepared = effects.prepare(graph, 48000)
    inputs = graph_audio(graph)
    runtime = _native.LiveRuntime.empty(48000, 2, 48000, 64)
    runtime.set_effect_graph(
        *effects.native_live_graph(prepared), 2, input_routes=[[-2]]
    )
    effects.configure_native_generators(runtime, prepared)
    saved = runtime.snapshot()
    actions: list[audio_effects.EffectAction] = [
        audio_effects.InputEndAction(tick=12003, ordinal=0, input="main"),
        audio_effects.InputEndAction(tick=24003, ordinal=0, input="main"),
    ]
    runtime.submit_effects(effects.native_live_actions(prepared, actions, 0, 48000))
    actual = np.ones((48000, 2))
    with pytest.raises(ValueError, match="repeated"):
        runtime.process_into(actual, [inputs["main"]])
    assert runtime.failed()
    check_audio(tmp_path / "repeated-end-failure.wav", actual, np.zeros_like(actual))
    runtime.restore(saved)
    assert not runtime.failed()
    assert not runtime.effects_drained
    runtime.submit_effects(effects.native_live_actions(prepared, actions[:1], 0, 48000))
    runtime.process_into(actual, [inputs["main"]])
    assert runtime.effects_drained
    expected = effects.OfflineEffects(prepared).advance(inputs, actions[:1], 0, 48000)
    check_audio(tmp_path / "repeated-end-recovery.wav", actual, expected)


@pytest.mark.parametrize("automated", [False, True])
def test_slow_grains_can_outlast_the_retained_source_history(
    tmp_path: Path, automated: bool
) -> None:
    graph = effects.serial_graph(
        audio_effects.AttachmentScope.master,
        graph_input(),
        [
            audio_effects.Granulator(
                name="cloud",
                duration_seconds=0.01 if automated else 0.04,
                maximum_duration_seconds=0.04,
                density_hz=50,
                lookback_seconds=0.005,
                playback_ratio=0.25,
                history_seconds=0.02,
                maximum_grains=4,
            )
        ],
        48000,
    )
    prepared = effects.prepare(graph, 48000)
    inputs = graph_audio(graph)
    actions: list[audio_effects.EffectAction] = (
        [
            audio_effects.ParameterAction(
                tick=12003,
                ordinal=0,
                processor="cloud",
                parameter="duration_seconds",
                value=0.04,
                duration_frames=1300,
            )
        ]
        if automated
        else []
    )
    expected = effects.OfflineEffects(prepared).advance(inputs, actions, 0, 48000)
    assert np.max(np.abs(expected)) > 0.01
    native = effects.OfflineEffects(prepared, "native").advance(
        inputs, actions, 0, 48000
    )
    actual, replay, _ = callback_audio(prepared, inputs, actions)
    check_audio(tmp_path / "slow-offline.wav", native, expected)
    check_audio(tmp_path / "slow-callback.wav", actual, expected)
    check_audio(tmp_path / "slow-replay.wav", replay, actual)


def test_over_capacity_native_grain_failure_preserves_snapshot_recovery(
    tmp_path: Path,
) -> None:
    graph = modulated_graph("grain")[0]
    prepared = effects.prepare(graph, 48000)
    inputs = graph_audio(graph)
    runtime = _native.LiveRuntime.empty(48000, 2, 48000, 64)
    runtime.set_effect_graph(
        *effects.native_live_graph(prepared), 2, input_routes=[[-2]]
    )
    saved = runtime.snapshot()
    action = audio_effects.ParameterAction(
        tick=12003,
        ordinal=0,
        processor="fx",
        parameter="duration_seconds",
        value=0.04,
    )
    encoded = effects.native_live_actions(prepared, [action], 0, 48000)
    encoded[0, 4] = 0.06
    runtime.submit_effects(encoded)
    actual = np.ones((48000, 2))
    with pytest.raises(ValueError, match="granulator parameter"):
        runtime.process_into(actual, [inputs["main"]])
    assert runtime.failed()
    check_audio(tmp_path / "grain-cap-failure.wav", actual, np.zeros_like(actual))
    runtime.restore(saved)
    assert not runtime.failed()
    runtime.submit_effects(effects.native_live_actions(prepared, [action], 0, 48000))
    runtime.process_into(actual, [inputs["main"]])
    raw = graph.model_dump()
    raw.update(envelfos={}, bindings=[], modulation={})
    expected = effects.OfflineEffects(
        effects.prepare(audio_effects.EffectGraph.model_validate(raw), 48000)
    ).advance(inputs, [action], 0, 48000)
    check_audio(tmp_path / "grain-cap-recovery.wav", actual, expected)
