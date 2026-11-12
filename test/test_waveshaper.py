from pathlib import Path
from typing import Literal

import numpy as np
import pytest
from test_effects import graph_input
from test_persistent_effects import callback_audio
from test_synth import check_audio
from ufor import audio_effects
from waveshaper_oracle import waveshaper_audio

from enge import _native, effects


@pytest.mark.parametrize("curve", ["soft", "hard", "fold"])
@pytest.mark.parametrize("bias", [0.0, 0.4])
def test_waveshaper_matches_independent_oversampling_and_drains(
    tmp_path: Path, curve: Literal["soft", "hard", "fold"], bias: float
) -> None:
    graph = effects.serial_graph(
        audio_effects.AttachmentScope.master,
        graph_input(),
        [
            audio_effects.Waveshaper(
                name="shape",
                curve=curve,
                drive=3,
                bias=bias,
                output_gain_db=-3,
                mix=0.6,
            )
        ],
        48000,
    )
    prepared = effects.prepare(graph, 48000)
    assert prepared.latency_frames == 48
    time = np.arange(48000) / 48000
    source = np.column_stack(
        (1.3 * np.cos(2 * np.pi * 997 * time), 0.9 * np.sin(2 * np.pi * 13001 * time))
    )
    source[40000:] = 0
    dry = np.concatenate((np.zeros((48, 2)), source[:-48]))
    expected = 0.4 * dry + 0.6 * waveshaper_audio(source, 3, bias, -3, curve)
    actions: list[audio_effects.EffectAction] = [
        audio_effects.InputEndAction(tick=40000, ordinal=0, input="main")
    ]
    for backend in ("numpy", "native"):
        renderer = effects.OfflineEffects(prepared, backend)
        actual = renderer.advance({"main": source}, actions, 0, 48000)
        assert renderer.drained
        assert renderer.stopped_at == 40096
        check_audio(tmp_path / f"oversampled-{backend}.wav", actual, expected)
    actual, replay, drained = callback_audio(prepared, {"main": source}, actions)
    assert drained
    check_audio(tmp_path / "oversampled-callback.wav", actual, expected)
    check_audio(tmp_path / "oversampled-replay.wav", replay, actual)


@pytest.mark.parametrize(
    "origin,target", [("numpy", "native"), ("native", "numpy"), ("native", "native")]
)
def test_waveshaper_aligns_branch_audio_controls_and_portable_state(
    tmp_path: Path,
    origin: Literal["numpy", "native"],
    target: Literal["numpy", "native"],
) -> None:
    graph = audio_effects.EffectGraph(
        scope=audio_effects.AttachmentScope.master,
        inputs=[graph_input()],
        processors=[
            audio_effects.Waveshaper(name="first", mix=0),
            audio_effects.Waveshaper(name="second", mix=0),
            audio_effects.Multiply(name="product", mix=0),
        ],
        connections=[
            audio_effects.Connection(
                processor="first",
                port="input",
                source=audio_effects.InputSource(input="main"),
            ),
            audio_effects.Connection(
                processor="second",
                port="input",
                source=audio_effects.ProcessorSource(processor="first"),
            ),
            audio_effects.Connection(
                processor="product",
                port="carrier",
                source=audio_effects.ProcessorSource(processor="second"),
            ),
            audio_effects.Connection(
                processor="product",
                port="modulator",
                source=audio_effects.InputSource(input="main"),
            ),
        ],
        output=audio_effects.ProcessorSource(processor="product"),
        maximum_block_frames=48000,
    )
    prepared = effects.prepare(graph, 48000)
    assert prepared.latency_frames == 96
    source = np.column_stack([np.cos(np.arange(48000) * f / 48000) for f in (631, 997)])
    source[36000:] = 0
    delayed = np.concatenate((np.zeros((96, 2)), source[:-96]))
    expected = delayed.copy()
    expected[12096:] = delayed[12096:] ** 2
    actions: list[audio_effects.EffectAction] = [
        audio_effects.ParameterAction(
            tick=12000, ordinal=0, processor="product", parameter="mix", value=1
        ),
        audio_effects.InputEndAction(tick=36000, ordinal=0, input="main"),
    ]
    first = effects.OfflineEffects(prepared, origin)
    head = first.advance({"main": source[:37]}, [], 0, 37)
    saved = effects.EffectSnapshot.model_validate_json(
        first.snapshot().model_dump_json()
    )
    second = effects.OfflineEffects(prepared, target)
    if origin == target == "native":
        before = second.snapshot()
        for field in ("alignment", "control", "fir"):
            invalid = saved.model_copy(deep=True)
            if field == "alignment":
                invalid.processors["product"].alignment["modulator"].history.pop()
            elif field == "control":
                invalid.processors["first"].control_delays["0"].history.pop()
            else:
                state = invalid.processors["first"].waveshaper
                assert state is not None
                state.histories[0].pop()
            with pytest.raises(ValueError, match="snapshot|compensation"):
                second.restore(invalid)
            assert second.snapshot() == before
    second.restore(saved)
    tail = second.advance({"main": source[37:]}, actions, 37, 48000)
    actual = np.concatenate((head, tail))
    check_audio(tmp_path / "aligned-portable.wav", actual, expected)
    if origin == target:
        whole = effects.OfflineEffects(prepared, origin).advance(
            {"main": source}, actions, 0, 48000
        )
        np.testing.assert_array_equal(actual, whole)
    callback, replay, _ = callback_audio(prepared, {"main": source}, actions)
    check_audio(tmp_path / "aligned-callback.wav", callback, expected)
    check_audio(tmp_path / "aligned-replay.wav", replay, callback)


def test_waveshaper_control_changes_follow_their_dsp_stage(tmp_path: Path) -> None:
    graph = effects.serial_graph(
        audio_effects.AttachmentScope.master,
        graph_input(),
        [audio_effects.Waveshaper(name="shape", drive=2)],
        48000,
    )
    prepared = effects.prepare(graph, 48000)
    time = np.arange(48000)
    source = np.column_stack(
        [0.8 * np.sin(2 * np.pi * f * time / 48000) for f in (997, 9001)]
    )
    drive = 2 + 3 * np.clip((time - 10000) / 512, 0, 1)
    bias = 0.4 * np.clip((time - 18000) / 512, 0, 1)
    output_db = -6 * np.clip((time - 28000) / 64, 0, 1)
    mix = 1 - 0.5 * np.clip((time - 12048) / 64, 0, 1)
    bypass = 1 - np.clip((time - 36048) / 64, 0, 1)
    wet = waveshaper_audio(source, drive, bias, output_db)
    dry = np.concatenate((np.zeros((48, 2)), source[:-48]))
    expected = (1 - mix * bypass)[:, None] * dry + (mix * bypass)[:, None] * wet
    actions: list[audio_effects.EffectAction] = [
        audio_effects.ParameterAction(
            tick=t,
            ordinal=0,
            processor="shape",
            parameter=p,
            value=v,
            duration_frames=d,
        )
        for t, p, v, d in [
            (10000, "drive", 5, 512),
            (12000, "mix", 0.5, 64),
            (18000, "bias", 0.4, 512),
            (28000, "output_gain_db", -6, 64),
        ]
    ]
    actions.append(
        audio_effects.BypassAction(
            tick=36000, ordinal=0, processor="shape", bypassed=True
        )
    )
    for backend in ("numpy", "native"):
        actual = effects.OfflineEffects(prepared, backend).advance(
            {"main": source}, actions, 0, 48000
        )
        check_audio(tmp_path / f"controls-{backend}.wav", actual, expected)
    actual, replay, _ = callback_audio(prepared, {"main": source}, actions)
    check_audio(tmp_path / "controls-callback.wav", actual, expected)
    check_audio(tmp_path / "controls-replay.wav", replay, actual)


def test_oversampling_reduces_the_third_harmonic_foldback(tmp_path: Path) -> None:
    graph = effects.serial_graph(
        audio_effects.AttachmentScope.master,
        graph_input(),
        [audio_effects.Waveshaper(name="shape", curve="hard", drive=4)],
        48000,
    )
    source = np.column_stack([np.sin(2 * np.pi * 11000 * np.arange(96000) / 48000)] * 2)
    actual = effects.process_audio(
        effects.prepare(graph, 48000), {"main": source}, [], backend="native"
    )
    expected = waveshaper_audio(source, 4, curve="hard")
    check_audio(tmp_path / "alias-reduction.wav", actual, expected)
    spectrum = np.abs(np.fft.rfft(actual[-48000:, 0]))
    naive = np.abs(np.fft.rfft(np.clip(4 * source[-48000:, 0], -1, 1)))
    assert spectrum[15000] < 0.1 * naive[15000]
    assert spectrum[11000] > 0.5 * naive[11000]


def test_native_snapshot_rejects_a_different_fixed_curve() -> None:
    runtimes: list[_native.LiveRuntime] = []
    curves: list[Literal["soft", "fold"]] = ["soft", "fold"]
    for curve in curves:
        prepared = effects.prepare(
            effects.serial_graph(
                audio_effects.AttachmentScope.master,
                graph_input(),
                [audio_effects.Waveshaper(name="shape", curve=curve)],
                997,
            ),
            48000,
        )
        runtime = _native.LiveRuntime.empty(48000, 2, 997, 64)
        runtime.set_effect_graph(
            *effects.native_live_graph(prepared), 2, input_routes=[[-2]]
        )
        runtimes.append(runtime)
    with pytest.raises(ValueError, match="[Ss]napshot"):
        runtimes[1].restore(runtimes[0].snapshot())


@pytest.mark.parametrize("failure", ["gain", "input", "curve"])
@pytest.mark.parametrize("curve", ["soft", "fold"])
def test_waveshaper_callback_failure_can_restore(
    tmp_path: Path, failure: str, curve: Literal["soft", "fold"]
) -> None:
    graph = effects.serial_graph(
        audio_effects.AttachmentScope.master,
        graph_input(),
        [
            *(
                [audio_effects.Gain(name="input-gain", gain_db=20)]
                if failure == "input"
                else []
            ),
            audio_effects.Waveshaper(name="shape", curve=curve),
        ],
        48000,
    )
    prepared = effects.prepare(graph, 48000)
    source = np.column_stack(
        [0.2 * np.sin(np.arange(48000) * f / 48000) for f in (631, 997)]
    )
    runtime = _native.LiveRuntime.empty(48000, 2, 48000, 64)
    runtime.set_effect_graph(
        *effects.native_live_graph(prepared), 2, input_routes=[[-2]]
    )
    saved = runtime.snapshot()
    if failure == "gain":
        action = audio_effects.ParameterAction(
            tick=12003,
            ordinal=0,
            processor="shape",
            parameter="output_gain_db",
            value=7000,
        )
        runtime.submit_effects(
            effects.native_live_actions(prepared, [action], 0, 48000)
        )
    elif failure == "curve":
        runtime.submit_effects(np.array([[12003, 0, 0, 5, 2, 0]], dtype=np.float64))
    actual = np.zeros_like(source)
    frames = 1 if failure == "input" else 48000
    actual[:frames] = 1
    with pytest.raises(ValueError, match="waveshaper"):
        runtime.process_into(
            actual[:frames], [np.full((1, 2), 1e308) if failure == "input" else source]
        )
    assert runtime.failed()
    check_audio(tmp_path / "gain-failure.wav", actual, np.zeros_like(actual))
    runtime.restore(saved)
    assert not runtime.failed()
    runtime.process_into(actual, [source])
    expected = waveshaper_audio(
        source * (10 if failure == "input" else 1), 1, curve=curve
    )
    check_audio(tmp_path / "gain-recovery.wav", actual, expected)


def test_waveshaper_aligns_a_mono_detector_with_stereo_audio(tmp_path: Path) -> None:
    graph = audio_effects.EffectGraph(
        scope=audio_effects.AttachmentScope.master,
        inputs=[
            graph_input(),
            audio_effects.GraphInput(
                name="detector",
                channels=["mono"],
                source=audio_effects.HostStream(stream="detector"),
            ),
        ],
        processors=[
            audio_effects.Waveshaper(name="shape", mix=0),
            audio_effects.Compressor(
                name="duck",
                threshold_db=-6,
                ratio=2,
                knee_db=0,
                attack_seconds=0,
                release_seconds=0,
            ),
        ],
        connections=[
            audio_effects.Connection(
                processor="shape",
                port="input",
                source=audio_effects.InputSource(input="detector"),
            ),
            audio_effects.Connection(
                processor="duck",
                port="input",
                source=audio_effects.InputSource(input="main"),
            ),
            audio_effects.Connection(
                processor="duck",
                port="detector",
                source=audio_effects.ProcessorSource(processor="shape"),
            ),
        ],
        output=audio_effects.ProcessorSource(processor="duck"),
        maximum_block_frames=48000,
    )
    prepared = effects.prepare(graph, 48000)
    assert prepared.latency_frames == 48
    source = np.column_stack(
        [0.3 * np.cos(np.arange(48000) * f / 48000) for f in (631, 997)]
    )
    detector = np.full((48000, 1), 0.25)
    detector[12000:] = 1
    source[40000:] = 0
    detector[40000:] = 0
    expected = np.concatenate((np.zeros((48, 2)), source[:-48]))
    expected[12048:18048] *= 10 ** (-3 / 20)
    expected[18048:] *= 10 ** (-6 / 20)
    actions: list[audio_effects.EffectAction] = [
        audio_effects.ParameterAction(
            tick=18000, ordinal=0, processor="duck", parameter="threshold_db", value=-12
        ),
        audio_effects.InputEndAction(tick=40000, ordinal=0, input="main"),
        audio_effects.InputEndAction(tick=40000, ordinal=1, input="detector"),
    ]
    inputs = {"main": source, "detector": detector}
    for backend in ("numpy", "native"):
        renderer = effects.OfflineEffects(prepared, backend)
        actual = renderer.advance(inputs, actions, 0, 48000)
        assert renderer.stopped_at == 40048
        check_audio(tmp_path / f"detector-{backend}.wav", actual, expected)
    actual, replay, drained = callback_audio(prepared, inputs, actions)
    assert drained
    check_audio(tmp_path / "detector-callback.wav", actual, expected)
    check_audio(tmp_path / "detector-replay.wav", replay, actual)


@pytest.mark.parametrize(
    "origin,target", [("numpy", "native"), ("native", "numpy"), ("native", "native")]
)
def test_waveshaper_delays_granular_freeze_and_preserves_pending_commands(
    tmp_path: Path,
    origin: Literal["numpy", "native"],
    target: Literal["numpy", "native"],
) -> None:
    grain = audio_effects.Granulator(
        name="cloud",
        duration_seconds=0.01,
        density_hz=100,
        lookback_seconds=0.005,
        history_seconds=0.04,
        maximum_grains=4,
    )
    graph = effects.serial_graph(
        audio_effects.AttachmentScope.master,
        graph_input(),
        [audio_effects.Waveshaper(name="shape", mix=0), grain],
        48000,
    )
    reference = effects.serial_graph(
        audio_effects.AttachmentScope.master,
        graph_input(),
        [audio_effects.Gain(name="trim"), grain],
        48000,
    )
    source = np.column_stack(
        [0.3 * np.cos(np.arange(48000) * f / 48000) for f in (631, 997)]
    )
    source[36000:] = 0
    delayed = np.concatenate((np.zeros((48, 2)), source[:-48]))
    actions: list[audio_effects.EffectAction] = [
        audio_effects.FreezeAction(
            tick=23920, ordinal=0, processor="cloud", frozen=True
        ),
        audio_effects.FreezeAction(
            tick=25000, ordinal=0, processor="cloud", frozen=False
        ),
        audio_effects.InputEndAction(tick=36000, ordinal=0, input="main"),
    ]
    expected = effects.OfflineEffects(effects.prepare(reference, 48000)).advance(
        {"main": delayed},
        [
            a.model_copy(
                update={
                    "tick": a.tick
                    + (96 if isinstance(a, audio_effects.InputEndAction) else 48)
                }
            )
            for a in actions
        ],
        0,
        48000,
    )
    prepared = effects.prepare(graph, 48000)
    first = effects.OfflineEffects(prepared, origin)
    head = first.advance({"main": source[:23928]}, actions[:1], 0, 23928)
    saved = effects.EffectSnapshot.model_validate_json(
        first.snapshot().model_dump_json()
    )
    assert saved.processors["cloud"].freeze_target
    state = saved.processors["cloud"].granulator
    assert state is not None and not state.frozen
    second = effects.OfflineEffects(prepared, target)
    second.restore(saved)
    actual = np.concatenate(
        (head, second.advance({"main": source[23928:]}, actions[1:], 23928, 48000))
    )
    check_audio(tmp_path / "freeze-portable.wav", actual, expected)
    callback, replay, drained = callback_audio(prepared, {"main": source}, actions)
    assert drained
    check_audio(tmp_path / "freeze-callback.wav", callback, expected)
    check_audio(tmp_path / "freeze-replay.wav", replay, callback)
