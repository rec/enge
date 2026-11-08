from pathlib import Path

import numpy as np
import pytest
from test_persistent_effects import callback_audio
from test_synth import check_audio
from ufor import audio_effects

from enge import _native, effects


def matrix_graph(
    matrix: list[list[float]], damping: float | None = None
) -> audio_effects.EffectGraph:
    return effects.serial_graph(
        audio_effects.AttachmentScope.master,
        audio_effects.GraphInput(
            name="main",
            channels=[f"channel-{i}" for i in range(len(matrix))],
            source=audio_effects.MainStream(),
        ),
        [
            audio_effects.MatrixDelay(
                name="echo",
                delay_seconds=37.25 / 48000,
                maximum_delay_seconds=0.01,
                feedback=0.5,
                feedback_matrix=matrix,
                damping_hz=damping,
            )
        ],
        48000,
    )


@pytest.mark.parametrize(
    "matrix,damping",
    [
        ([[0, 1], [1, 0]], None),
        ([[0.25, -0.75], [-1, 0]], 1200),
        ([[0, 0, -1], [1, 0, 0], [0, 1, 0]], None),
        ([[1, 0], [0, 1]], 1200),
    ],
)
def test_matrix_echoes_match_transfer_and_portable_callback_replay(
    tmp_path: Path, matrix: list[list[float]], damping: float | None
) -> None:
    prepared = effects.prepare(matrix_graph(matrix, damping), 48000)
    source = np.zeros((48000, len(matrix)))
    source[[0, 23840], 0] = 0.2
    size = 131072
    z = np.exp(-2j * np.pi * np.arange(size // 2 + 1) / size)
    read = z**37 * (0.75 + 0.25 * z)
    lowpass = np.ones_like(z)
    if damping is not None:
        g = np.tan(np.pi * damping / 48000)
        a = (g - 1) / (g + 1)
        lowpass = (1 + a) / 2 * (1 + z) / (1 + a * z)
    denominator = (
        np.eye(len(matrix))[None]
        - (0.5 * lowpass * read)[:, None, None] * np.asarray(matrix)[None]
    )
    spectrum = np.linalg.solve(
        denominator, (read[:, None] * np.fft.rfft(source, n=size, axis=0))[:, :, None]
    )[:, :, 0]
    expected = np.fft.irfft(spectrum, n=size, axis=0)[:48000]
    actions: list[audio_effects.EffectAction] = [
        audio_effects.InputEndAction(tick=23900, ordinal=0, input="main")
    ]
    for origin, target in [("numpy", "native"), ("native", "numpy")]:
        renderer = effects.OfflineEffects(prepared, origin)
        before = renderer.advance({"main": source[:23928]}, actions, 0, 23928)
        saved = effects.EffectSnapshot.model_validate_json(
            renderer.snapshot().model_dump_json()
        )
        restored = effects.OfflineEffects(prepared, target)
        restored.restore(saved)
        after = restored.advance({"main": source[23928:]}, [], 23928, 48000)
        assert restored.drained
        check_audio(
            tmp_path / f"matrix-{origin}-{target}.wav",
            np.vstack([before, after]),
            expected,
        )
    actual, replay, drained = callback_audio(prepared, {"main": source}, actions)
    assert drained
    check_audio(tmp_path / "matrix-callback.wav", actual, expected)
    check_audio(tmp_path / "matrix-replay.wav", replay, actual)


def test_matrix_delay_retimes_and_automates_with_shared_snapshot_state(
    tmp_path: Path,
) -> None:
    prepared = effects.prepare(matrix_graph([[0, 1], [1, 0]], 1200), 48000)
    frames = np.arange(48000)
    source = np.column_stack(
        [0.1 * np.sin(2 * np.pi * f * frames / 48000) for f in [220, 330]]
    )
    source[30000:] = 0
    actions: list[audio_effects.EffectAction] = [
        audio_effects.ParameterAction(
            tick=23880,
            ordinal=0,
            processor="echo",
            parameter="damping_hz",
            value=2400,
            duration_frames=64,
        ),
        audio_effects.ParameterAction(
            tick=23900,
            ordinal=0,
            processor="echo",
            parameter="delay_seconds",
            value=81.5 / 48000,
        ),
        audio_effects.ParameterAction(
            tick=24500,
            ordinal=0,
            processor="echo",
            parameter="feedback",
            value=0.6,
            duration_frames=64,
        ),
        audio_effects.InputEndAction(tick=30000, ordinal=0, input="main"),
    ]
    expected = effects.OfflineEffects(prepared).advance(
        {"main": source}, actions, 0, 48000
    )
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
            tmp_path / f"automated-{origin}-{target}.wav",
            np.vstack([before, after]),
            expected,
        )
    actual, replay, drained = callback_audio(prepared, {"main": source}, actions)
    assert drained
    check_audio(tmp_path / "automated-callback.wav", actual, expected)
    check_audio(tmp_path / "automated-replay.wav", replay, actual)


def test_matrix_snapshot_rejects_different_feedback_without_mutating_voice(
    tmp_path: Path,
) -> None:
    prepared = effects.prepare(matrix_graph([[0, 1], [1, 0]]), 48000)
    other = effects.prepare(matrix_graph([[1, 0], [0, 1]]), 48000)
    runtime = _native.LiveRuntime.empty(48000, 2, 48000, 64)
    runtime.set_effect_graph(
        *effects.native_live_graph(prepared), 2, input_routes=[[-2]]
    )
    wrong = _native.LiveRuntime.empty(48000, 2, 48000, 64)
    wrong.set_effect_graph(*effects.native_live_graph(other), 2, input_routes=[[-2]])
    with pytest.raises(ValueError, match="different live effect graph"):
        runtime.restore(wrong.snapshot())
    source = np.zeros((48000, 2))
    source[0, 0] = 0.2
    actual = np.empty_like(source)
    runtime.process_into(actual, [source])
    expected = effects.OfflineEffects(prepared).advance({"main": source}, [], 0, 48000)
    check_audio(tmp_path / "preserved-matrix.wav", actual, expected)


def test_matrix_coefficient_action_fails_silently_and_restores(tmp_path: Path) -> None:
    prepared = effects.prepare(matrix_graph([[0, 1], [1, 0]]), 48000)
    runtime = _native.LiveRuntime.empty(48000, 2, 48000, 64)
    runtime.set_effect_graph(
        *effects.native_live_graph(prepared), 2, input_routes=[[-2]]
    )
    saved = runtime.snapshot()
    runtime.submit_effects(np.array([[0, 0, 0, 6, 0.5, 0]], dtype=np.float64))
    source = np.zeros((48000, 2))
    source[0, 0] = 0.2
    actual = np.ones_like(source)
    with pytest.raises(ValueError, match="Invalid tap delay parameter action"):
        runtime.process_into(actual, [source])
    assert runtime.failed()
    check_audio(tmp_path / "failed-matrix.wav", actual, np.zeros_like(source))
    runtime.restore(saved)
    runtime.process_into(actual, [source])
    expected = effects.OfflineEffects(prepared).advance({"main": source}, [], 0, 48000)
    check_audio(tmp_path / "recovered-matrix.wav", actual, expected)


@pytest.mark.parametrize("coefficients", [[1.1, 0, 0, 1], [1e308, 1e308, 0, 0], []])
def test_native_preparation_rejects_unbounded_or_missing_delay_matrix(
    coefficients: list[float],
) -> None:
    prepared = effects.prepare(matrix_graph([[0, 1], [1, 0]]), 48000)
    plan = list(effects.native_live_graph(prepared))
    parameters = np.zeros((1, 6 + len(coefficients)))
    parameters[:, :6] = plan[2][:, :6]
    parameters[0, 6:] = coefficients
    plan[2] = parameters
    runtime = _native.LiveRuntime.empty(48000, 2, 48000, 64)
    with pytest.raises(ValueError, match="feedback matrix"):
        runtime.set_effect_graph(*plan, 2, input_routes=[[-2]])
