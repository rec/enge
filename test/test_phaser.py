from math import pi, tan
from pathlib import Path

import numpy as np
import pytest
from test_effects import graph_input
from test_persistent_effects import callback_audio
from test_synth import check_audio
from ufor import audio_effects

from enge import _native, effects


def phaser_graph(feedback: float = 0, mix: float = 0.5) -> audio_effects.EffectGraph:
    return effects.serial_graph(
        audio_effects.AttachmentScope.master,
        graph_input(),
        [audio_effects.Phaser(name="phase", feedback=feedback, mix=mix)],
        48000,
    )


@pytest.mark.parametrize(
    "feedback,mix,frequency",
    [(0, 0.5, 800), (0, 1, 3200), (0.6, 0.5, 1100), (-0.6, 0.75, 400)],
)
def test_phaser_matches_the_allpass_and_feedback_transfer_function(
    tmp_path: Path, feedback: float, mix: float, frequency: int
) -> None:
    prepared = effects.prepare(phaser_graph(feedback, mix), 48000)
    assert prepared.latency_frames == 0
    time = np.arange(48000) / 48000
    source = np.column_stack(
        [
            a * np.cos(2 * pi * f * time)
            for a, f in [(0.3, frequency), (0.2, 2 * frequency)]
        ]
    )
    expected = effects.OfflineEffects(prepared).advance({"main": source}, [], 0, 48000)
    g = tan(pi * 800 / 48000)
    a = (g - 1) / (g + 1)
    for c, (amplitude, f) in enumerate([(0.3, frequency), (0.2, 2 * frequency)]):
        z = np.exp(-2j * pi * f / 48000)
        transfer = ((a + z) / (1 + a * z)) ** 6
        transfer /= 1 - feedback * z * transfer
        response = amplitude * (1 - mix + mix * transfer)
        measured = 2 * np.mean(
            expected[24000:, c] * np.exp(-2j * pi * f * time[24000:])
        )
        assert abs(measured - response) < 1e-10
    actual = effects.OfflineEffects(prepared, "native").advance(
        {"main": source}, [], 0, 48000
    )
    callback, replay, drained = callback_audio(prepared, {"main": source}, [])
    assert not drained
    check_audio(tmp_path / "response-native.wav", actual, expected)
    check_audio(tmp_path / "response-callback.wav", callback, expected)
    check_audio(tmp_path / "response-replay.wav", replay, callback)


def test_phaser_fast_shared_sweeps_preserve_the_impulse_energy_bound(
    tmp_path: Path,
) -> None:
    raw = phaser_graph(0.95, 1).model_dump(mode="json")
    raw["processors"][0].update(center_hz=10025, depth_cents=0)
    raw.update(
        envelfos={"motion": {"kind": "lfo", "shape": "square", "rate": 24000}},
        bindings=[{"kind": "envelfo", "name": "motion", "reference": "motion"}],
        modulation={
            "sources": [
                {"name": "motion", "scope": "voice", "minimum": -1, "maximum": 1}
            ],
            "parameters": [
                {
                    "target": {"name": "phase", "parameter": "center_hz"},
                    "unit": "hz",
                    "scope": "voice",
                    "minimum": 50,
                    "maximum": 20000,
                    "default": 10025,
                }
            ],
            "routes": [
                {
                    "name": "motion",
                    "source": "motion",
                    "target": {"name": "phase", "parameter": "center_hz"},
                    "unit": "hz",
                    "operation": "add",
                    "points": [
                        {"input": -1, "amount": -9975},
                        {"input": 1, "amount": 9975},
                    ],
                }
            ],
        },
    )
    prepared = effects.prepare(audio_effects.EffectGraph.model_validate(raw), 48000)
    source = np.zeros((48000, 2))
    source[0] = [1, 0.5]
    expected = effects.OfflineEffects(prepared).advance({"main": source}, [], 0, 48000)
    assert np.sum(expected**2) <= (1.25 / (1 - 0.95**2)) * (1 + 1e-10)
    actual = effects.OfflineEffects(prepared, "native").advance(
        {"main": source}, [], 0, 48000
    )
    callback, replay, _ = callback_audio(prepared, {"main": source}, [])
    check_audio(tmp_path / "fast-sweep-native.wav", actual, expected)
    check_audio(tmp_path / "fast-sweep-callback.wav", callback, expected)
    check_audio(tmp_path / "fast-sweep-replay.wav", replay, callback)


def test_phaser_bypass_keeps_the_filter_warm_and_drains_after_input_end(
    tmp_path: Path,
) -> None:
    prepared = effects.prepare(phaser_graph(0.5), 48000)
    source = np.column_stack(
        [0.3 * np.cos(np.arange(48000) * f / 48000) for f in (631, 997)]
    )
    source[32000:] = 0
    actions: list[audio_effects.EffectAction] = [
        audio_effects.ParameterAction(
            tick=10000,
            ordinal=0,
            processor="phase",
            parameter="feedback",
            value=-0.5,
            duration_frames=128,
        ),
        audio_effects.ParameterAction(
            tick=18003,
            ordinal=0,
            processor="phase",
            parameter="center_hz",
            value="1600 Hz",
            duration_frames=512,
        ),
        audio_effects.ParameterAction(
            tick=20000,
            ordinal=0,
            processor="phase",
            parameter="depth_cents",
            value="1 octave",
            duration_frames=128,
        ),
        audio_effects.ParameterAction(
            tick=24000,
            ordinal=0,
            processor="phase",
            parameter="sweep",
            value=0.5,
            duration_frames=512,
        ),
        audio_effects.InputEndAction(tick=32000, ordinal=0, input="main"),
    ]
    wet = effects.OfflineEffects(effects.prepare(phaser_graph(0.5, 1), 48000)).advance(
        {"main": source}, actions, 0, 48000
    )
    time = np.arange(48000)
    weight = 0.5 * (
        1 - np.clip((time - 12000) / 64, 0, 1) + np.clip((time - 24000) / 64, 0, 1)
    )
    expected = (1 - weight[:, None]) * source + weight[:, None] * wet
    actions.extend(
        [
            audio_effects.BypassAction(
                tick=12000, ordinal=0, processor="phase", bypassed=True
            ),
            audio_effects.BypassAction(
                tick=24000, ordinal=1, processor="phase", bypassed=False
            ),
        ]
    )
    actions.sort(key=lambda a: (a.tick, a.ordinal))
    for backend in ("numpy", "native"):
        renderer = effects.OfflineEffects(prepared, backend)
        first = renderer.advance(
            {"main": source[:23928]}, [a for a in actions if a.tick < 23928], 0, 23928
        )
        saved = effects.EffectSnapshot.model_validate_json(
            renderer.snapshot().model_dump_json()
        )
        if backend == "native":
            invalid = saved.model_copy(deep=True)
            state = invalid.processors["phase"].phaser
            assert state is not None
            state.stages.pop()
            with pytest.raises(ValueError, match="snapshot matrix"):
                renderer.restore(invalid)
            assert renderer.snapshot() == saved
        renderer.restore(saved)
        last = renderer.advance(
            {"main": source[23928:]},
            [a for a in actions if a.tick >= 23928],
            23928,
            48000,
        )
        assert renderer.drained
        check_audio(
            tmp_path / f"bypass-{backend}.wav", np.concatenate((first, last)), expected
        )
    callback, replay, drained = callback_audio(prepared, {"main": source}, actions)
    assert drained
    check_audio(tmp_path / "bypass-callback.wav", callback, expected)
    check_audio(tmp_path / "bypass-replay.wav", replay, callback)


@pytest.mark.parametrize(
    "parameter,value", [("center_hz", 24000), ("center_hz", 1e-300), ("feedback", 0.99)]
)
def test_phaser_callback_rejects_invalid_coefficients_and_recovers(
    tmp_path: Path, parameter: str, value: float
) -> None:
    prepared = effects.prepare(phaser_graph(), 48000)
    runtime = _native.LiveRuntime.empty(48000, 2, 48000, 64)
    runtime.set_effect_graph(
        *effects.native_live_graph(prepared), 2, input_routes=[[-2]]
    )
    saved = runtime.snapshot()
    source = np.column_stack(
        [0.2 * np.cos(np.arange(48000) * f / 48000) for f in (631, 997)]
    )
    action = audio_effects.ParameterAction(
        tick=12003,
        ordinal=0,
        processor="phase",
        parameter=parameter,
        value=0.5 if parameter == "feedback" else 800,
    )
    encoded = effects.native_live_actions(prepared, [action], 0, 48000)
    encoded[0, 4] = value
    runtime.submit_effects(encoded)
    actual = np.ones_like(source)
    with pytest.raises(ValueError, match="[Pp]haser"):
        runtime.process_into(actual, [source])
    assert runtime.failed()
    check_audio(tmp_path / "invalid-callback.wav", actual, np.zeros_like(actual))
    runtime.restore(saved)
    assert not runtime.failed()
    runtime.process_into(actual, [source])
    expected = effects.OfflineEffects(prepared).advance({"main": source}, [], 0, 48000)
    check_audio(tmp_path / "recovered-callback.wav", actual, expected)


def test_phaser_after_waveshaping_aligns_controls_and_portable_state(
    tmp_path: Path,
) -> None:
    processor = audio_effects.Phaser(name="phase", feedback=0.5)
    graph = effects.serial_graph(
        audio_effects.AttachmentScope.master,
        graph_input(),
        [audio_effects.Waveshaper(name="latency", mix=0), processor],
        48000,
    )
    reference = effects.prepare(phaser_graph(0.5), 48000)
    prepared = effects.prepare(graph, 48000)
    assert prepared.latency_frames == 48
    source = np.column_stack(
        [0.3 * np.cos(np.arange(48000) * f / 48000) for f in (631, 997)]
    )
    source[36000:] = 0
    delayed = np.concatenate((np.zeros((48, 2)), source[:-48]))
    actions: list[audio_effects.EffectAction] = [
        audio_effects.ParameterAction(
            tick=12003,
            ordinal=0,
            processor="phase",
            parameter="center_hz",
            value=1600,
            duration_frames=512,
        ),
        audio_effects.BypassAction(
            tick=18000, ordinal=0, processor="phase", bypassed=True
        ),
        audio_effects.ParameterAction(
            tick=20003,
            ordinal=0,
            processor="phase",
            parameter="depth_cents",
            value=1200,
            duration_frames=256,
        ),
        audio_effects.BypassAction(
            tick=22000, ordinal=0, processor="phase", bypassed=False
        ),
        audio_effects.ParameterAction(
            tick=23920,
            ordinal=0,
            processor="phase",
            parameter="center_hz",
            value=2400,
            duration_frames=512,
        ),
        audio_effects.ParameterAction(
            tick=25000,
            ordinal=0,
            processor="phase",
            parameter="sweep",
            value=0.5,
            duration_frames=512,
        ),
        audio_effects.InputEndAction(tick=36000, ordinal=0, input="main"),
    ]
    expected = effects.OfflineEffects(reference).advance(
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
    for origin, target in [("numpy", "native"), ("native", "numpy")]:
        first = effects.OfflineEffects(prepared, origin)
        head = first.advance(
            {"main": source[:23928]}, [a for a in actions if a.tick < 23928], 0, 23928
        )
        saved = effects.EffectSnapshot.model_validate_json(
            first.snapshot().model_dump_json()
        )
        second = effects.OfflineEffects(prepared, target)
        second.restore(saved)
        tail = second.advance(
            {"main": source[23928:]},
            [a for a in actions if a.tick >= 23928],
            23928,
            48000,
        )
        assert second.drained
        check_audio(
            tmp_path / f"latent-{origin}.wav", np.concatenate((head, tail)), expected
        )
    actual, replay, drained = callback_audio(prepared, {"main": source}, actions)
    assert drained
    check_audio(tmp_path / "latent-callback.wav", actual, expected)
    check_audio(tmp_path / "latent-replay.wav", replay, actual)
