from math import pi, tan
from pathlib import Path

import numpy as np
import pytest
from test_effects import graph_input
from test_persistent_effects import callback_audio
from test_synth import check_audio
from ufor import audio_effects

from enge import _native, effects


def highpass_graph(cutoff: float = 20) -> audio_effects.EffectGraph:
    return effects.serial_graph(
        audio_effects.AttachmentScope.master,
        graph_input(),
        [audio_effects.Highpass(name="dc", cutoff_hz=cutoff)],
        48000,
    )


@pytest.mark.parametrize("cutoff,frequency", [(20, 20), (800, 800), (800, 6400)])
def test_highpass_matches_the_one_pole_transfer_function(
    tmp_path: Path, cutoff: float, frequency: int
) -> None:
    prepared = effects.prepare(highpass_graph(cutoff), 48000)
    assert prepared.latency_frames == 0
    time = np.arange(48000) / 48000
    source = np.column_stack(
        [v * np.cos(2 * pi * frequency * time) for v in (0.3, 0.2)]
    )
    expected = effects.OfflineEffects(prepared).advance({"main": source}, [], 0, 48000)
    g = tan(pi * cutoff / 48000)
    pole = (1 - g) / (1 + g)
    z = np.exp(-2j * pi * frequency / 48000)
    transfer = (1 + pole) / 2 * (1 - z) / (1 - pole * z)
    for c, amplitude in enumerate((0.3, 0.2)):
        measured = 2 * np.mean(
            expected[24000:, c] * np.exp(-2j * pi * frequency * time[24000:])
        )
        assert abs(measured - amplitude * transfer) < 1e-10
    actual = effects.OfflineEffects(prepared, "native").advance(
        {"main": source}, [], 0, 48000
    )
    callback, replay, _ = callback_audio(prepared, {"main": source}, [])
    check_audio(tmp_path / "response-native.wav", actual, expected)
    check_audio(tmp_path / "response-callback.wav", callback, expected)
    check_audio(tmp_path / "response-replay.wav", replay, callback)


def test_dc_removal_matches_the_step_response_and_drains(tmp_path: Path) -> None:
    prepared = effects.prepare(highpass_graph(), 48000)
    source = np.zeros((48000, 2))
    source[:24000] = [0.5, -0.25]
    pole = (1 - tan(pi * 20 / 48000)) / (1 + tan(pi * 20 / 48000))
    step = (1 + pole) / 2 * np.power(pole, np.arange(48000))
    step[24000:] -= (1 + pole) / 2 * np.power(pole, np.arange(24000))
    expected = step[:, None] * [0.5, -0.25]
    actions = [audio_effects.InputEndAction(tick=24000, ordinal=0, input="main")]
    for backend in ("numpy", "native"):
        renderer = effects.OfflineEffects(prepared, backend)
        actual = renderer.advance({"main": source}, actions, 0, 48000)
        assert renderer.drained
        check_audio(tmp_path / f"step-{backend}.wav", actual, expected)
    callback, replay, drained = callback_audio(prepared, {"main": source}, actions)
    assert drained
    check_audio(tmp_path / "step-callback.wav", callback, expected)
    check_audio(tmp_path / "step-replay.wav", replay, callback)


def test_highpass_cutoff_and_warm_bypass_replay_across_backends(tmp_path: Path) -> None:
    prepared = effects.prepare(highpass_graph(), 48000)
    source = np.column_stack(
        [0.3 * np.cos(np.arange(48000) * v / 48000) + 0.2 for v in (631, 997)]
    )
    actions = [
        audio_effects.ParameterAction(
            tick=23800,
            ordinal=0,
            processor="dc",
            parameter="cutoff_hz",
            value="1600 Hz",
            duration_frames=1024,
        )
    ]
    wet = effects.OfflineEffects(prepared).advance({"main": source}, actions, 0, 48000)
    weight = (
        1
        - np.clip((np.arange(48000) - 12000) / 64, 0, 1)
        + np.clip((np.arange(48000) - 24000) / 64, 0, 1)
    )
    expected = (1 - weight[:, None]) * source + weight[:, None] * wet
    actions.extend(
        [
            audio_effects.BypassAction(
                tick=12000, ordinal=0, processor="dc", bypassed=True
            ),
            audio_effects.BypassAction(
                tick=24000, ordinal=0, processor="dc", bypassed=False
            ),
        ]
    )
    actions.sort(key=lambda a: (a.tick, a.ordinal))
    for origin, target in (
        ("numpy", "native"),
        ("native", "numpy"),
        ("native", "native"),
    ):
        renderer = effects.OfflineEffects(prepared, origin)
        first = renderer.advance(
            {"main": source[:23928]}, [a for a in actions if a.tick < 23928], 0, 23928
        )
        saved = effects.EffectSnapshot.model_validate_json(
            renderer.snapshot().model_dump_json()
        )
        restored = effects.OfflineEffects(prepared, target)
        restored.restore(saved)
        last = restored.advance(
            {"main": source[23928:]},
            [a for a in actions if a.tick >= 23928],
            23928,
            48000,
        )
        check_audio(
            tmp_path / f"warm-{origin}-{target}.wav", np.vstack([first, last]), expected
        )
    callback, replay, _ = callback_audio(prepared, {"main": source}, actions)
    check_audio(tmp_path / "warm-callback.wav", callback, expected)
    check_audio(tmp_path / "warm-replay.wav", replay, callback)


@pytest.mark.parametrize("cutoff", [24000, 1e-300])
def test_highpass_invalid_coefficients_fail_silently_and_restore(
    tmp_path: Path, cutoff: float
) -> None:
    prepared = effects.prepare(highpass_graph(), 48000)
    runtime = _native.LiveRuntime.empty(48000, 2, 48000, 64)
    runtime.set_effect_graph(
        *effects.native_live_graph(prepared), 2, input_routes=[[-2]]
    )
    saved = runtime.snapshot()
    source = np.ones((48000, 2)) * 0.25
    action = audio_effects.ParameterAction(
        tick=12003, ordinal=0, processor="dc", parameter="cutoff_hz", value=20
    )
    encoded = effects.native_live_actions(prepared, [action], 0, 48000)
    encoded[0, 4] = cutoff
    runtime.submit_effects(encoded)
    actual = np.ones_like(source)
    with pytest.raises(ValueError, match="[Hh]ighpass"):
        runtime.process_into(actual, [source])
    assert runtime.failed()
    check_audio(tmp_path / "invalid.wav", actual, np.zeros_like(actual))
    runtime.restore(saved)
    runtime.process_into(actual, [source])
    expected = effects.OfflineEffects(prepared).advance({"main": source}, [], 0, 48000)
    check_audio(tmp_path / "recovered.wav", actual, expected)
