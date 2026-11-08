from pathlib import Path

import numpy as np
import pytest
from test_persistent_effects import callback_audio
from test_synth import check_audio
from ufor import audio_effects

from enge import _native, effects
from enge.highpass import HighpassState


def delayed_audio(source: np.ndarray, frames: float) -> np.ndarray:
    grid = np.arange(-1, 48001)
    return np.column_stack(
        [
            np.interp(
                np.arange(48000) - frames,
                grid,
                np.concatenate([[0], source[:, c], [0]]),
                left=0,
                right=0,
            )
            for c in range(source.shape[1])
        ]
    )


@pytest.mark.parametrize("stereo", [False, True])
def test_multitap_pan_gain_retime_end_and_portable_callback_replay(
    tmp_path: Path, stereo: bool
) -> None:
    frames = np.arange(48000)
    source = np.column_stack(
        [
            0.2 * np.cos(2 * np.pi * f * frames / 48000 + i * 0.3)
            for i, f in enumerate([220, 330] if stereo else [220])
        ]
    )
    source[36000:] = 0
    taps = [
        audio_effects.DelayTap(
            name=n,
            delay_seconds=d / 48000,
            maximum_delay_seconds=0.01,
            gain_db=g,
            pan=p,
        )
        for n, d, g, p in [
            ("one", 37.5, -3, -1),
            ("two", 67, -6, -0.25),
            ("three", 121.25, -9, 0.5),
            ("four", 205, -12, 1),
        ]
    ]
    graph = audio_effects.multitap_graph(
        "echo",
        audio_effects.AttachmentScope.master,
        audio_effects.GraphInput(
            name="main",
            channels=["left", "right"] if stereo else ["centre"],
            source=audio_effects.MainStream(),
        ),
        taps,
        48000,
        dry_gain_db=-6,
    )
    prepared = effects.prepare(graph, 48000)
    expected = (
        source.copy() if stereo else np.repeat(source / np.sqrt(2), 2, axis=1)
    ) * 10 ** (-6 / 20)
    for tap in taps:
        delayed = delayed_audio(source, tap.delay_seconds * 48000)
        gain = np.full(48000, tap.gain_db)
        if tap.name == "one":
            progress = np.clip((frames - 23900) / 64, 0, 1)
            delayed = (
                delayed * (1 - progress[:, None])
                + delayed_audio(source, 83.25) * progress[:, None]
            )
            gain -= 9 * np.clip((frames - 24000) / 1200, 0, 1)
        if stereo:
            weights = [
                1 - max(0, tap.pan),
                1 + min(0, tap.pan),
            ]
            placed = delayed * weights
        else:
            angle = (tap.pan + 1) * np.pi / 4
            placed = delayed * np.array([np.cos(angle), np.sin(angle)])
        expected += placed * (10 ** (gain / 20))[:, None]
    actions: list[audio_effects.EffectAction] = [
        audio_effects.ParameterAction(
            tick=23900,
            ordinal=0,
            processor="echo-tap-one",
            parameter="delay_seconds",
            value=83.25 / 48000,
        ),
        audio_effects.ParameterAction(
            tick=24000,
            ordinal=0,
            processor="echo",
            parameter="tap-one-gain_db",
            value=-12,
            duration_frames=1200,
        ),
        audio_effects.InputEndAction(tick=36000, ordinal=0, input="main"),
    ]
    inputs = {"main": source}
    for origin, target in [("numpy", "native"), ("native", "numpy")]:
        renderer = effects.OfflineEffects(prepared, origin)
        before = renderer.advance(
            {"main": source[:23928]}, [a for a in actions if a.tick < 23928], 0, 23928
        )
        saved = effects.EffectSnapshot.model_validate_json(
            renderer.snapshot().model_dump_json()
        )
        restored = effects.OfflineEffects(prepared, target)
        restored.restore(saved)
        after = restored.advance(
            {"main": source[23928:]},
            [a for a in actions if a.tick >= 23928],
            23928,
            48000,
        )
        assert restored.drained
        check_audio(
            tmp_path / f"multitap-{origin}-{target}.wav",
            np.vstack([before, after]),
            expected,
        )
    actual, replay, drained = callback_audio(prepared, inputs, actions)
    assert drained
    check_audio(tmp_path / "multitap-callback.wav", actual, expected)
    check_audio(tmp_path / "multitap-replay.wav", replay, actual)


def damping_graph(
    delay_frames: float = 37, cutoff: float | None = 1200, feedback: float = 0.5
) -> audio_effects.EffectGraph:
    return effects.serial_graph(
        audio_effects.AttachmentScope.master,
        audio_effects.GraphInput(
            name="main", channels=["left", "right"], source=audio_effects.MainStream()
        ),
        [
            audio_effects.TapDelay(
                name="echo",
                delay_seconds=delay_frames / 48000,
                maximum_delay_seconds=96 / 48000,
                feedback=feedback,
                damping_hz=cutoff,
            )
        ],
        48000,
    )


@pytest.mark.parametrize("delay", [37, 37.25])
def test_damped_echoes_match_feedback_transfer_and_replay_filter_tail(
    tmp_path: Path, delay: float
) -> None:
    prepared = effects.prepare(damping_graph(delay), 48000)
    source = np.zeros((48000, 2))
    source[[0, 23840]] = [0.2, -0.1]
    size = 131072
    z = np.exp(-2j * np.pi * np.arange(size // 2 + 1) / size)
    g = np.tan(np.pi * 1200 / 48000)
    a = (g - 1) / (g + 1)
    lowpass = (1 + a) / 2 * (1 + z) / (1 + a * z)
    whole = int(delay)
    fraction = delay - whole
    read = z**whole * (1 - fraction + fraction * z)
    transfer = read / (1 - 0.5 * lowpass * read)
    expected = np.fft.irfft(
        np.fft.rfft(source, n=size, axis=0) * transfer[:, None], n=size, axis=0
    )[:48000]
    actions: list[audio_effects.EffectAction] = [
        audio_effects.InputEndAction(tick=23900, ordinal=0, input="main")
    ]
    for origin, target in [("numpy", "native"), ("native", "numpy")]:
        renderer = effects.OfflineEffects(prepared, origin)
        before = renderer.advance({"main": source[:23928]}, actions, 0, 23928)
        saved = effects.EffectSnapshot.model_validate_json(
            renderer.snapshot().model_dump_json()
        )
        delay_state = saved.processors["echo"].tap_delay
        assert delay_state is not None and delay_state.damping is not None
        assert max(abs(v) for v in delay_state.damping.memory) > 1e-9
        restored = effects.OfflineEffects(prepared, target)
        restored.restore(saved)
        after = restored.advance({"main": source[23928:]}, [], 23928, 48000)
        assert restored.drained
        check_audio(
            tmp_path / f"damped-{origin}-{target}.wav",
            np.vstack([before, after]),
            expected,
        )
    actual, replay, drained = callback_audio(prepared, {"main": source}, actions)
    assert drained
    assert actual[whole, 0] == pytest.approx(0.2 * (1 - fraction))
    check_audio(tmp_path / "damped-callback.wav", actual, expected)
    check_audio(tmp_path / "damped-replay.wav", replay, actual)


@pytest.mark.parametrize("enabled", [False, True])
def test_tap_snapshot_rejects_damping_mode_and_memory_mismatches(
    tmp_path: Path, enabled: bool
) -> None:
    prepared = effects.prepare(damping_graph(cutoff=1200 if enabled else None), 48000)
    other = effects.prepare(damping_graph(cutoff=None if enabled else 1200), 48000)
    runtime = _native.LiveRuntime.empty(48000, 2, 48000, 64)
    runtime.set_effect_graph(
        *effects.native_live_graph(prepared), 2, input_routes=[[-2]]
    )
    wrong = _native.LiveRuntime.empty(48000, 2, 48000, 64)
    wrong.set_effect_graph(*effects.native_live_graph(other), 2, input_routes=[[-2]])
    with pytest.raises(ValueError, match="different live effect graph"):
        runtime.restore(wrong.snapshot())
    source = np.zeros((48000, 2))
    source[[0, 23840]] = [0.2, -0.1]
    expected = effects.OfflineEffects(prepared).advance({"main": source}, [], 0, 48000)
    actual = np.empty_like(source)
    runtime.process_into(actual, [source])
    check_audio(tmp_path / "preserved.wav", actual, expected)
    for backend in ["numpy", "native"]:
        renderer = effects.OfflineEffects(prepared, backend)
        before = renderer.advance({"main": source[:23928]}, [], 0, 23928)
        saved = renderer.snapshot()
        broken = saved.model_copy(deep=True)
        delay_state = broken.processors["echo"].tap_delay
        assert delay_state is not None
        if enabled:
            assert delay_state.damping is not None
            delay_state.damping.memory.pop()
        else:
            delay_state.damping = HighpassState(memory=[0.0, 0.0])
        with pytest.raises(ValueError, match="damping"):
            renderer.restore(broken)
        assert renderer.snapshot() == saved
        after = renderer.advance({"main": source[23928:]}, [], 23928, 48000)
        check_audio(
            tmp_path / f"preserved-{backend}.wav", np.vstack([before, after]), expected
        )


@pytest.mark.parametrize("enabled", [False, True])
def test_nonfinite_tap_feedback_fails_before_retaining_history(
    tmp_path: Path, enabled: bool
) -> None:
    prepared = effects.prepare(
        damping_graph(1, cutoff=16000 if enabled else None, feedback=0.99), 48000
    )
    runtime = _native.LiveRuntime.empty(48000, 2, 48000, 64)
    runtime.set_effect_graph(
        *effects.native_live_graph(prepared), 2, input_routes=[[-2]]
    )
    saved = runtime.snapshot()
    output = np.ones((48000, 2))
    with pytest.raises(ValueError, match="Non-finite tap delay feedback"):
        runtime.process_into(output, [np.full_like(output, 1e308)])
    assert runtime.failed()
    check_audio(tmp_path / "overflow.wav", output, np.zeros_like(output))
    runtime.restore(saved)
    source = np.full_like(output, 0.01)
    expected = effects.OfflineEffects(prepared).advance({"main": source}, [], 0, 48000)
    runtime.process_into(output, [source])
    check_audio(tmp_path / "recovered.wav", output, expected)


@pytest.mark.parametrize("enabled", [False, True])
def test_invalid_damping_action_silences_callback_and_restores(
    tmp_path: Path, enabled: bool
) -> None:
    prepared = effects.prepare(damping_graph(cutoff=1200 if enabled else None), 48000)
    runtime = _native.LiveRuntime.empty(48000, 2, 48000, 64)
    runtime.set_effect_graph(
        *effects.native_live_graph(prepared), 2, input_routes=[[-2]]
    )
    saved = runtime.snapshot()
    runtime.submit_effects(
        np.array([[12003, 0, 0, 5, 24000 if enabled else 1200, 0]], dtype=np.float64)
    )
    source = np.zeros((48000, 2))
    source[0] = [0.2, -0.1]
    output = np.ones_like(source)
    with pytest.raises(ValueError, match="Nyquist|not prepared"):
        runtime.process_into(output, [source])
    assert runtime.failed()
    check_audio(tmp_path / "failed.wav", output, np.zeros_like(output))
    runtime.restore(saved)
    runtime.process_into(output, [source])
    expected = effects.OfflineEffects(prepared).advance({"main": source}, [], 0, 48000)
    check_audio(tmp_path / "recovered.wav", output, expected)


def test_damping_preparation_rejects_nyquist_and_unrepresentable_domains() -> None:
    for cutoff in [24000, 1e-300]:
        with pytest.raises(ValueError, match="Nyquist|coefficient"):
            effects.prepare(damping_graph(cutoff=cutoff), 48000)
