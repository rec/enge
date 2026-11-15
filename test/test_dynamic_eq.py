from pathlib import Path
from typing import Literal

import numpy as np
import pytest
from dynamic_eq_oracle import bell_audio, moving_band, static_band
from test_effect_envelfo import modulated_graph
from test_effects import graph_input
from test_persistent_effects import callback_audio
from test_synth import check_audio
from ufor import audio_effects

from enge import _native, effects


def graph(ratio: float = 4) -> audio_effects.EffectGraph:
    return effects.serial_graph(
        audio_effects.AttachmentScope.master,
        graph_input(),
        [
            audio_effects.DynamicEQ(
                name="bell",
                threshold_db=-24,
                ratio=ratio,
                attack_seconds=0.001,
                release_seconds=0.05,
            )
        ],
        48000,
    )


def source_audio() -> np.ndarray:
    frames = np.arange(48000)
    time = frames / 48000
    low = 0.1 * np.sin(2 * np.pi * 100 * time)
    peak = np.where((frames >= 12003) & (frames < 30007), 0.6, 0.02)
    band = peak * np.sin(2 * np.pi * 1000 * time)
    source = np.column_stack([low + band, low * 0.5 + band * 0.1])
    source[36003:] = 0
    return source


@pytest.mark.parametrize("ratio", [1, 4])
def test_dynamic_eq_matches_transfer_and_capped_dynamics_across_replay(
    tmp_path: Path, ratio: float
) -> None:
    source = source_audio()
    expected = bell_audio(source, static_band(source), ratio=ratio)
    prepared = effects.prepare(graph(ratio), 48000)
    assert prepared.latency_frames == 0
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
        if ratio == 1:
            np.testing.assert_array_equal(actual, source)
        check_audio(tmp_path / f"{origin}-{target}.wav", actual, expected)
    actual, replay, drained = callback_audio(prepared, {"main": source}, actions)
    assert drained
    check_audio(tmp_path / "callback.wav", actual, expected)
    check_audio(tmp_path / "replay.wav", replay, actual)


def test_dynamic_eq_sweeps_q_and_scoped_frequency_through_warm_bypass(
    tmp_path: Path,
) -> None:
    raw = graph().model_dump()
    raw["processors"][0]["q"] = 0.5
    modulation = modulated_graph("gain")[0].model_dump()
    raw.update({n: modulation[n] for n in ("envelfos", "bindings", "modulation")})
    target = {"name": "bell", "parameter": "center_hz"}
    raw["modulation"]["parameters"][0].update(
        target=target, unit="hz", minimum=500, maximum=6000, default=1000
    )
    raw["modulation"]["routes"][0].update(
        target=target,
        unit="hz",
        points=[{"input": -1, "amount": -250}, {"input": 1, "amount": 250}],
    )
    prepared = effects.prepare(audio_effects.EffectGraph.model_validate(raw), 48000)
    source = source_audio()
    frames = np.arange(48000)
    center = (
        1000
        + 3000 * np.clip((frames - 15003) / 18000, 0, 1)
        + 250 * np.where((frames // 12000) % 2 == 0, 1, -1)
    )
    q = 0.5 + 1.5 * np.clip((frames - 6001) / 18000, 0, 1)
    maximum = np.where(frames < 24007, 12, 3)
    wet = bell_audio(source, moving_band(source, center, q), maximum=maximum)
    mix = (
        1 - np.clip((frames - 15003) / 64, 0, 1) + np.clip((frames - 27007) / 64, 0, 1)
    )
    expected = source + mix[:, None] * (wet - source)
    actions: list[audio_effects.EffectAction] = [
        audio_effects.ParameterAction(
            tick=6001,
            ordinal=0,
            processor="bell",
            parameter="q",
            value=2,
            duration_frames=18000,
        ),
        audio_effects.ParameterAction(
            tick=15003,
            ordinal=0,
            processor="bell",
            parameter="center_hz",
            value=4000,
            duration_frames=18000,
        ),
        audio_effects.BypassAction(
            tick=15003, ordinal=1, processor="bell", bypassed=True
        ),
        audio_effects.ParameterAction(
            tick=24007, ordinal=0, processor="bell", parameter="maximum_cut_db", value=3
        ),
        audio_effects.BypassAction(
            tick=27007, ordinal=0, processor="bell", bypassed=False
        ),
        audio_effects.InputEndAction(tick=36003, ordinal=0, input="main"),
    ]
    for backend in BACKENDS:
        renderer = effects.OfflineEffects(prepared, backend)
        head = renderer.advance({"main": source[:23928]}, actions[:3], 0, 23928)
        saved = effects.EffectSnapshot.model_validate_json(
            renderer.snapshot().model_dump_json()
        )
        other = effects.OfflineEffects(
            prepared, "native" if backend == "numpy" else "numpy"
        )
        other.restore(saved)
        actual = np.vstack(
            [head, other.advance({"main": source[23928:]}, actions[3:], 23928, 48000)]
        )
        assert other.drained
        check_audio(tmp_path / f"{backend}.wav", actual, expected)
    actual, replay, drained = callback_audio(prepared, {"main": source}, actions)
    assert drained
    check_audio(tmp_path / "callback.wav", actual, expected)
    check_audio(tmp_path / "replay.wav", replay, actual)


def test_dynamic_eq_mono_instantaneous_cap_stop_and_snapshot_validation(
    tmp_path: Path,
) -> None:
    raw = graph().model_dump()
    raw["inputs"][0]["channels"] = ["mono"]
    raw["processors"][0].update(
        attack_seconds=0, release_seconds=0, knee_db=0, maximum_cut_db=6
    )
    prepared = effects.prepare(audio_effects.EffectGraph.model_validate(raw), 48000)
    source = source_audio()[:, :1]
    expected = bell_audio(
        source, static_band(source), knee=0, attack=0, release=0, maximum=6
    )
    expected[24007:] = 0
    actions: list[audio_effects.EffectAction] = [
        audio_effects.StopAction(tick=24007, ordinal=0)
    ]
    for backend in BACKENDS:
        renderer = effects.OfflineEffects(prepared, backend)
        original = renderer.snapshot()
        invalid = original.model_copy(deep=True)
        invalid.processors["bell"].filter_states.clear()
        with pytest.raises(ValueError, match="dynamic EQ"):
            renderer.restore(invalid)
        invalid = original.model_copy(deep=True)
        invalid.processors["bell"].reduction_db = 1
        with pytest.raises(ValueError, match="dynamic EQ"):
            renderer.restore(invalid)
        assert renderer.snapshot() == original
        actual = renderer.advance({"main": source}, actions, 0, 48000)
        check_audio(tmp_path / f"{backend}.wav", actual, expected)
    actual, replay, drained = callback_audio(prepared, {"main": source}, actions)
    assert drained
    check_audio(tmp_path / "callback.wav", actual, expected)
    check_audio(tmp_path / "replay.wav", replay, actual)


@pytest.mark.parametrize(
    "parameter,value,message",
    [(4, 0, "filter"), (5, 0.5, "compressor"), (9, -1, "compressor")],
)
def test_failed_dynamic_eq_callbacks_silence_latch_and_restore(
    tmp_path: Path, parameter: int, value: float, message: str
) -> None:
    prepared = effects.prepare(graph(), 48000)
    runtime = _native.LiveRuntime.empty(48000, 2, 48000, 64)
    runtime.set_effect_graph(
        *effects.native_live_graph(prepared), 2, input_routes=[[-2]]
    )
    saved = runtime.snapshot()
    runtime.submit_effects(
        np.array([[12003, 0, 0, parameter, value, 0]], dtype=np.float64)
    )
    source = source_audio()
    actual = np.ones_like(source)
    with pytest.raises(ValueError, match=message):
        runtime.process_into(actual, [source])
    assert runtime.failed()
    check_audio(tmp_path / "failed.wav", actual, np.zeros_like(source))
    runtime.restore(saved)
    runtime.process_into(actual, [source])
    check_audio(
        tmp_path / "restored.wav", actual, bell_audio(source, static_band(source))
    )


def test_multiple_dynamic_eq_nodes_keep_independent_band_and_reduction_state(
    tmp_path: Path,
) -> None:
    definition = effects.serial_graph(
        audio_effects.AttachmentScope.master,
        graph_input(),
        [
            audio_effects.DynamicEQ(
                name="one",
                threshold_db=-24,
                ratio=4,
                attack_seconds=0.001,
                release_seconds=0.05,
            ),
            audio_effects.DynamicEQ(
                name="two",
                center_hz=4000,
                q=2,
                threshold_db=-24,
                ratio=4,
                attack_seconds=0.001,
                release_seconds=0.05,
            ),
        ],
        48000,
    )
    source = source_audio()
    frames = np.arange(48000)
    tone = 0.3 * np.sin(2 * np.pi * 4000 * frames / 48000) * (frames < 36003)
    source += np.column_stack([tone, tone * 0.5])
    first = bell_audio(source, static_band(source))
    expected = bell_audio(first, static_band(first, 4000, 2))
    prepared = effects.prepare(definition, 48000)
    actions: list[audio_effects.EffectAction] = [
        audio_effects.InputEndAction(tick=36003, ordinal=0, input="main")
    ]
    renderer = effects.OfflineEffects(prepared, "numpy")
    head = renderer.advance({"main": source[:23928]}, [], 0, 23928)
    saved = effects.EffectSnapshot.model_validate_json(
        renderer.snapshot().model_dump_json()
    )
    other = effects.OfflineEffects(prepared, "native")
    other.restore(saved)
    actual = np.vstack(
        [head, other.advance({"main": source[23928:]}, actions, 23928, 48000)]
    )
    assert other.drained
    check_audio(tmp_path / "portable.wav", actual, expected)
    actual, replay, drained = callback_audio(prepared, {"main": source}, actions)
    assert drained
    check_audio(tmp_path / "callback.wav", actual, expected)
    check_audio(tmp_path / "replay.wav", replay, actual)


def test_dynamic_eq_rejects_initial_tuning_outside_render_bounds() -> None:
    with pytest.raises(ValueError, match="cutoff"):
        effects.prepare(graph(), 2000)


BACKENDS: list[Literal["numpy", "native"]] = ["numpy", "native"]
