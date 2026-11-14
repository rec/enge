from pathlib import Path
from typing import Literal

import numpy as np
import pytest
from deesser_oracle import deesser_audio
from test_effect_envelfo import modulated_graph
from test_effects import graph_input
from test_persistent_effects import callback_audio
from test_synth import check_audio
from ufor import audio_effects

from enge import effects


def deesser_graph() -> audio_effects.EffectGraph:
    return audio_effects.deesser_graph(
        "ess", audio_effects.AttachmentScope.master, graph_input(), 48000
    )


def sibilant_audio() -> np.ndarray:
    frames = np.arange(48000)
    time = frames / 48000
    left = 0.15 * np.sin(2 * np.pi * 220 * time)
    right = 0.1 * np.sin(2 * np.pi * 330 * time)
    burst = 0.4 * np.sin(2 * np.pi * 6000 * time)
    right += burst * ((frames >= 12003) & (frames < 18007))
    left += burst * ((frames >= 24007) & (frames < 30011))
    source = np.column_stack([left, right])
    source[36003:] = 0
    return source


def test_deesser_matches_filtered_detection_and_linked_gain_across_replay(
    tmp_path: Path,
) -> None:
    source = sibilant_audio()
    expected = deesser_audio(source)
    # The low-band channel is reduced too when only the other channel sibilates.
    assert np.linalg.norm(expected[15000:18000, 0]) < 0.6 * np.linalg.norm(
        source[15000:18000, 0]
    )
    np.testing.assert_array_equal(expected[:12003], source[:12003])
    prepared = effects.prepare(deesser_graph(), 48000)
    assert prepared.channels == ["left", "right"] and prepared.latency_frames == 0
    actions: list[audio_effects.EffectAction] = [
        audio_effects.InputEndAction(tick=36003, ordinal=0, input="main")
    ]
    pairs: list[tuple[Literal["numpy", "native"], Literal["numpy", "native"]]] = [
        ("numpy", "native"),
        ("native", "numpy"),
    ]
    for origin, target in pairs:
        first = effects.OfflineEffects(prepared, origin)
        head = first.advance({"main": source[:23928]}, [], 0, 23928)
        saved = effects.EffectSnapshot.model_validate_json(
            first.snapshot().model_dump_json()
        )
        second = effects.OfflineEffects(prepared, target)
        second.restore(saved)
        tail = second.advance({"main": source[23928:]}, actions, 23928, 48000)
        assert second.drained
        actual = np.vstack([head, tail])
        np.testing.assert_array_equal(actual[36003:], 0)
        check_audio(tmp_path / f"{origin}-{target}.wav", actual, expected)
    actual, replay, drained = callback_audio(prepared, {"main": source}, actions)
    assert drained
    check_audio(tmp_path / "callback.wav", actual, expected)
    check_audio(tmp_path / "replay.wav", replay, actual)


def test_deesser_threshold_envelfo_and_warm_bypass_keep_detector_state(
    tmp_path: Path,
) -> None:
    raw = deesser_graph().model_dump()
    modulation = modulated_graph("gain")[0].model_dump()
    raw.update({n: modulation[n] for n in ("envelfos", "bindings", "modulation")})
    target = {"name": "ess-compressor", "parameter": "threshold_db"}
    parameter = raw["modulation"]["parameters"][0]
    parameter.update(target=target, unit="db", minimum=-36, maximum=-12, default=-24)
    route = raw["modulation"]["routes"][0]
    route.update(target=target, unit="db")
    route["points"] = [{"input": -1, "amount": -6}, {"input": 1, "amount": 6}]
    graph = audio_effects.EffectGraph.model_validate(raw)
    prepared = effects.prepare(graph, 48000)
    frames = np.arange(48000)
    source = sibilant_audio()
    threshold = -24 + 6 * np.where((frames // 12000) % 2 == 0, 1, -1)
    wet = deesser_audio(source, threshold)
    mix = (
        1 - np.clip((frames - 15003) / 64, 0, 1) + np.clip((frames - 27007) / 64, 0, 1)
    )
    expected = (1 - mix[:, None]) * source + mix[:, None] * wet
    actions: list[audio_effects.EffectAction] = [
        audio_effects.BypassAction(
            tick=15003, ordinal=0, processor="ess-compressor", bypassed=True
        ),
        audio_effects.BypassAction(
            tick=27007, ordinal=0, processor="ess-compressor", bypassed=False
        ),
        audio_effects.InputEndAction(tick=36003, ordinal=0, input="main"),
    ]
    for backend in BACKENDS:
        actual = effects.process_audio(
            prepared, {"main": source}, actions, backend, 997
        )
        check_audio(tmp_path / f"{backend}.wav", actual, expected)
    actual, replay, drained = callback_audio(prepared, {"main": source}, actions)
    assert drained
    check_audio(tmp_path / "callback.wav", actual, expected)
    check_audio(tmp_path / "replay.wav", replay, actual)


def test_deesser_unity_ratio_keeps_detector_out_of_the_signal_path(
    tmp_path: Path,
) -> None:
    source = sibilant_audio()
    prepared = effects.prepare(deesser_graph(), 48000)
    actions: list[audio_effects.EffectAction] = [
        audio_effects.ParameterAction(
            tick=0, ordinal=0, processor="ess-compressor", parameter="ratio", value=1
        ),
        audio_effects.InputEndAction(tick=36003, ordinal=0, input="main"),
    ]
    for backend in BACKENDS:
        actual = effects.process_audio(
            prepared, {"main": source}, actions, backend, 997
        )
        np.testing.assert_array_equal(actual, source)
        check_audio(tmp_path / f"{backend}.wav", actual, source)
    actual, replay, drained = callback_audio(prepared, {"main": source}, actions)
    assert drained
    np.testing.assert_array_equal(actual, source)
    check_audio(tmp_path / "callback.wav", actual, source)
    check_audio(tmp_path / "replay.wav", replay, actual)


def test_deesser_rejects_detector_center_outside_sample_rate_bounds() -> None:
    with pytest.raises(ValueError, match="cutoff"):
        effects.prepare(deesser_graph(), 8000)


BACKENDS: list[Literal["numpy", "native"]] = ["numpy", "native"]
