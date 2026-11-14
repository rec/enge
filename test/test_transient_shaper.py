from pathlib import Path
from typing import Literal

import numpy as np
import pytest
from test_effect_envelfo import modulated_graph
from test_effects import graph_input
from test_persistent_effects import callback_audio
from test_synth import check_audio
from transient_oracle import transient_audio
from ufor import audio_effects, envelfo

from enge import _native, effects


def shaper_graph(attack: float = 6, sustain: float = -3) -> audio_effects.EffectGraph:
    return effects.serial_graph(
        audio_effects.AttachmentScope.master,
        graph_input(),
        [
            audio_effects.TransientShaper(
                name="shape", attack_gain_db=attack, sustain_gain_db=sustain
            )
        ],
        48000,
    )


def stepped_audio() -> np.ndarray:
    frames = np.arange(48000)
    peak = np.where(frames < 12003, 0.25, np.where(frames < 24007, 0.0625, 0.5))
    carrier = 1 - 2 * ((frames // 137) % 2)
    return np.column_stack(
        [
            carrier * peak * np.where(frames < 24007, 1, 0.5),
            -carrier * peak * np.where(frames < 24007, 0.5, 1),
        ]
    )


@pytest.mark.parametrize("attack,sustain", [(12, -6), (-12, 6), (0, 0)])
def test_transient_contrast_matches_closed_form_steps_and_linked_replay(
    tmp_path: Path, attack: float, sustain: float
) -> None:
    source = stepped_audio()
    source[36003:] = 0
    expected = transient_audio(source, attack, sustain)
    prepared = effects.prepare(shaper_graph(attack, sustain), 48000)
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
        snapshot = effects.EffectSnapshot.model_validate_json(
            first.snapshot().model_dump_json()
        )
        second = effects.OfflineEffects(prepared, target)
        second.restore(snapshot)
        tail = second.advance({"main": source[23928:]}, actions, 23928, 48000)
        assert second.drained and second.stopped_at == 36003
        state = second.snapshot().processors["shape"].transient
        assert state is not None and state.fast == state.slow == 0
        actual = np.vstack([head, tail])
        if attack == sustain == 0:
            np.testing.assert_array_equal(actual, source)
        check_audio(tmp_path / f"{origin}-{target}.wav", actual, expected)
    actual, replay, drained = callback_audio(prepared, {"main": source}, actions)
    assert drained
    check_audio(tmp_path / "callback.wav", actual, expected)
    check_audio(tmp_path / "replay.wav", replay, actual)


@pytest.mark.parametrize("amplitude", [1e-15, 1e-35])
def test_quiet_normalization_and_detector_floor_remain_finite(
    tmp_path: Path, amplitude: float
) -> None:
    source = np.tile([amplitude, -amplitude / 2], (48000, 1))
    expected = transient_audio(source, 24, 24) if amplitude > 1e-30 else source
    prepared = effects.prepare(shaper_graph(24, 24), 48000)
    for backend in BACKENDS:
        actual = effects.process_audio(
            prepared, {"main": source}, backend=backend, block_frames=997
        )
        np.testing.assert_allclose(actual, expected, rtol=1e-12, atol=0)
        check_audio(tmp_path / f"{backend}.wav", actual, expected)


def test_gain_ramps_and_warm_bypass_follow_upstream_audio_and_stop(
    tmp_path: Path,
) -> None:
    graph = effects.serial_graph(
        audio_effects.AttachmentScope.master,
        graph_input(),
        [
            audio_effects.Waveshaper(name="latency", mix=0),
            audio_effects.TransientShaper(
                name="shape", attack_gain_db=6, sustain_gain_db=-3
            ),
        ],
        48000,
    )
    prepared = effects.prepare(graph, 48000)
    frames = np.arange(48000)
    source = stepped_audio()
    dry = np.vstack([np.zeros((48, 2)), source[:-48]])
    attack = 6 + 6 * np.clip((frames - 12051) / 7000, 0, 1)
    sustain = -3 + 9 * np.clip((frames - 16055) / 6000, 0, 1)
    wet = transient_audio(dry, attack, sustain)
    mix = 1 - np.clip((frames - 6051) / 64, 0, 1) + np.clip((frames - 18051) / 64, 0, 1)
    expected = (1 - mix[:, None]) * dry + mix[:, None] * wet
    expected[36003:] = 0
    actions: list[audio_effects.EffectAction] = [
        audio_effects.BypassAction(
            tick=6003, ordinal=0, processor="shape", bypassed=True
        ),
        audio_effects.ParameterAction(
            tick=12003,
            ordinal=0,
            processor="shape",
            parameter="attack_gain_db",
            value=12,
            duration_frames=7000,
        ),
        audio_effects.ParameterAction(
            tick=16007,
            ordinal=0,
            processor="shape",
            parameter="sustain_gain_db",
            value=6,
            duration_frames=6000,
        ),
        audio_effects.BypassAction(
            tick=18003, ordinal=0, processor="shape", bypassed=False
        ),
        audio_effects.StopAction(tick=36003, ordinal=0),
    ]
    for backend in BACKENDS:
        check_audio(
            tmp_path / f"{backend}.wav",
            effects.process_audio(prepared, {"main": source}, actions, backend, 997),
            expected,
        )
    actual, replay, drained = callback_audio(prepared, {"main": source}, actions)
    assert drained
    check_audio(tmp_path / "callback.wav", actual, expected)
    check_audio(tmp_path / "replay.wav", replay, actual)


@pytest.mark.parametrize("failure", ["attack", "sustain", "overflow"])
def test_failed_transient_callbacks_silence_latch_and_restore(
    tmp_path: Path, failure: str
) -> None:
    attack = 24 if failure == "overflow" else 6
    prepared = effects.prepare(shaper_graph(attack), 48000)
    runtime = _native.LiveRuntime.empty(48000, 2, 48000, 64)
    runtime.set_effect_graph(
        *effects.native_live_graph(prepared), 2, input_routes=[[-2]]
    )
    saved = runtime.snapshot()
    source = np.tile([0.25, -0.125], (48000, 1))
    if failure != "overflow":
        runtime.submit_effects(
            np.array(
                [
                    [
                        12003,
                        0,
                        0,
                        0 if failure == "attack" else 3,
                        25 if failure == "attack" else -25,
                        0,
                    ]
                ],
                dtype=np.float64,
            )
        )
    invalid = np.full_like(source, 1e308) if failure == "overflow" else source
    actual = np.ones_like(source)
    with pytest.raises(ValueError, match="[Tt]ransient shaper"):
        runtime.process_into(actual, [invalid])
    assert runtime.failed()
    check_audio(tmp_path / "failed.wav", actual, np.zeros_like(source))
    runtime.restore(saved)
    runtime.process_into(actual, [source])
    check_audio(tmp_path / "restored.wav", actual, transient_audio(source, attack, -3))


def test_captured_envelope_modulates_numpy_transients_but_native_preparation_rejects_it(
    tmp_path: Path,
) -> None:
    raw = modulated_graph("transient")[0].model_dump(mode="json")
    raw["envelfos"]["motion"] = {
        "kind": "envelope",
        "initial": 0,
        "segments": [
            {
                "kind": "exponential_approach",
                "asymptote": 1,
                "time_constant": "20 ms",
                "threshold": 0.8,
                "transition_to": 1,
                "transition_seconds": "1 ms",
            }
        ],
        "release": [{"duration": "50 ms", "to": 0}],
    }
    raw["modulation"]["sources"][0]["minimum"] = 0
    raw["modulation"]["routes"][0]["points"][0]["input"] = 0
    prepared = effects.prepare(audio_effects.EffectGraph.model_validate(raw), 48000)
    source = stepped_audio()
    source[36003:] = 0
    time = np.arange(48000) / 48000
    duration = round(-0.02 * np.log(0.2) * 1e9) / 1e9
    control = np.where(
        time < duration,
        1 - np.exp(np.log(0.2) * time / duration),
        0.8 + 0.2 * np.clip((time - duration) / 0.001, 0, 1),
    )
    expected = transient_audio(source, 3 + 6 * control, -3)
    actions: list[audio_effects.EffectAction] = [
        audio_effects.InputEndAction(tick=36003, ordinal=0, input="main")
    ]
    first = effects.OfflineEffects(prepared, "numpy")
    head = first.advance({"main": source[:997]}, [], 0, 997)
    saved = effects.EffectSnapshot.model_validate_json(
        first.snapshot().model_dump_json()
    )
    second = effects.OfflineEffects(prepared, "numpy")
    second.restore(saved)
    tail = second.advance({"main": source[997:]}, actions, 997, 48000)
    check_audio(tmp_path / "numpy-replay.wav", np.vstack([head, tail]), expected)
    with pytest.raises(effects.EngineError, match="bounded shared storage"):
        effects.native_live_graph(prepared)
    with pytest.raises(effects.EngineError, match="bounded shared storage"):
        effects.OfflineEffects(prepared, "native")


def test_native_routes_reject_unused_transient_profile_fields() -> None:
    graph = effects.serial_graph(
        audio_effects.AttachmentScope.master,
        graph_input(),
        [
            audio_effects.Waveshaper(name="latency"),
            audio_effects.TransientShaper(name="shape"),
        ],
        997,
    )
    plan = effects.native_live_graph(effects.prepare(graph, 48000))
    runtime = _native.LiveRuntime.empty(48000, 2, 997, 64)
    with pytest.raises(ValueError, match="control target"):
        runtime.set_effect_graph(
            *plan[:12], ([(1, 4, -1.0, 1.0)], []), plan[13], 2, input_routes=[[-2]]
        )


def test_native_effect_preparation_rejects_captured_release_inside_patch() -> None:
    body = envelfo.Envelope.model_validate(
        {
            "initial": 0,
            "segments": [{"duration": "100 ms", "to": 1}],
            "release": [
                {
                    "kind": "exponential_approach",
                    "asymptote": 0,
                    "time_constant": "20 ms",
                    "threshold": 0.1,
                    "transition_to": 0,
                    "transition_seconds": "1 ms",
                }
            ],
        }
    )
    graph = shaper_graph().model_copy(
        update={
            "envelfos": {
                "motion": envelfo.Patch(
                    envelfos={"level": body}, outputs={"out": "level"}
                )
            }
        }
    )
    with pytest.raises(effects.EngineError, match="bounded shared storage"):
        effects.native_live_graph(effects.prepare(graph, 48000))


def test_negative_detector_snapshot_is_rejected_before_replacing_state() -> None:
    renderer = effects.OfflineEffects(effects.prepare(shaper_graph(), 48000), "native")
    original = renderer.snapshot()
    invalid = original.model_copy(deep=True)
    state = invalid.processors["shape"].transient
    assert state is not None
    state.fast = -1
    with pytest.raises(ValueError, match="fast"):
        renderer.restore(invalid)
    assert renderer.snapshot() == original


BACKENDS: list[Literal["numpy", "native"]] = ["numpy", "native"]
