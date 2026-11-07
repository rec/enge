from pathlib import Path

import numpy as np
import pytest
from test_persistent_effects import callback_audio
from test_synth import check_audio
from ufor import audio_effects

from enge import _native, effects


def dynamics_graph(
    kind: str, attack: float = 0, release: float = 0
) -> audio_effects.EffectGraph:
    processor = {
        "kind": kind,
        "name": "fx",
        "threshold_db": -20,
        "attack_seconds": attack,
        "release_seconds": release,
    }
    if kind == "gate":
        processor.update(hysteresis_db=6, hold_seconds=5.2 / 48000)
    return audio_effects.EffectGraph.model_validate(
        {
            "scope": "master",
            "inputs": [
                {
                    "name": "main",
                    "channels": ["left", "right"],
                    "source": {"kind": "main"},
                },
                {
                    "name": "detector",
                    "channels": ["a", "b", "c"],
                    "source": {"kind": "host", "stream": "sidechain"},
                },
            ],
            "processors": [processor],
            "connections": [
                {"processor": "fx", "port": p, "source": {"kind": "input", "input": n}}
                for p, n in [("input", "main"), ("detector", "detector")]
            ],
            "output": {"kind": "processor", "processor": "fx"},
            "maximum_block_frames": 48000,
        }
    )


def program_audio() -> np.ndarray:
    frames = np.arange(48000)
    return np.column_stack(
        [0.2 * np.sin(2 * np.pi * f * frames / 48000) for f in [220, 330]]
    )


def test_gate_hysteresis_fractional_hold_retiming_warm_bypass_and_replay(
    tmp_path: Path,
) -> None:
    prepared = effects.prepare(dynamics_graph("gate"), 48000)
    frames = np.arange(48000)
    peak = np.full(48000, 0.02)
    for first, last, value in [
        (1000, 2000, 0.1),
        (2000, 3000, 0.075),
        (3003, 3004, 0.075),
        (4000, 5000, 0.2),
        (23900, 23920, 0.2),
        (25000, 25010, 0.2),
        (26000, 26010, 0.2),
        (27100, 27300, 0.2),
        (27300, 28000, 0.075),
    ]:
        peak[first:last] = value
    peak[28000:] = 0
    source = program_audio()
    source[30000:] = 0
    inputs = {"main": source, "detector": np.column_stack([peak / 2, -peak, peak / 4])}
    opened = np.zeros(48000)
    for first, last in [
        (1000, 3010),
        (4000, 5006),
        (23900, 23930),
        (25000, 25061),
        (26000, 26010),
        (27100, 28000),
    ]:
        opened[first:last] = 1
    bypass = (
        1 - np.clip((frames - 27000) / 64, 0, 1) + np.clip((frames - 27800) / 64, 0, 1)
    )
    expected = source * (1 - bypass + bypass * opened)[:, None]
    actions: list[audio_effects.EffectAction] = [
        audio_effects.ParameterAction(
            tick=t, ordinal=0, processor="fx", parameter="hold_seconds", value=v / 48000
        )
        for t, v in [(23800, 100.2), (23930, 9.2), (25015, 50.2), (26000, 0)]
    ]
    actions.extend(
        [
            audio_effects.BypassAction(
                tick=27000, ordinal=0, processor="fx", bypassed=True
            ),
            audio_effects.BypassAction(
                tick=27800, ordinal=0, processor="fx", bypassed=False
            ),
            audio_effects.InputEndAction(tick=28000, ordinal=0, input="detector"),
            audio_effects.InputEndAction(tick=30000, ordinal=0, input="main"),
        ]
    )
    for origin, target in [
        ("numpy", "native"),
        ("native", "numpy"),
        ("native", "native"),
    ]:
        renderer = effects.OfflineEffects(prepared, origin)
        before = renderer.advance(
            {n: a[:23928] for n, a in inputs.items()},
            [a for a in actions if a.tick < 23928],
            0,
            23928,
        )
        saved = effects.EffectSnapshot.model_validate_json(
            renderer.snapshot().model_dump_json()
        )
        assert saved.processors["fx"].gate is not None
        assert saved.processors["fx"].gate.elapsed == 8
        restored = effects.OfflineEffects(prepared, target)
        restored.restore(saved)
        if target == "native":
            before_rejection = restored.snapshot()
            broken = saved.model_copy(deep=True)
            assert broken.processors["fx"].gate is not None
            broken.processors["fx"].gate.gain = 2
            with pytest.raises(ValueError, match="gate snapshot"):
                restored.restore(broken)
            assert restored.snapshot() == before_rejection
        after = restored.advance(
            {n: a[23928:] for n, a in inputs.items()},
            [a for a in actions if a.tick >= 23928],
            23928,
            48000,
        )
        assert restored.drained
        check_audio(
            tmp_path / f"hold-{origin}-{target}.wav",
            np.vstack([before, after]),
            expected,
        )
    actual, replay, drained = callback_audio(prepared, inputs, actions)
    assert drained
    check_audio(tmp_path / "hold-callback.wav", actual, expected)
    check_audio(tmp_path / "hold-replay.wav", replay, actual)


@pytest.mark.parametrize("kind", ["gate", "expander"])
def test_dynamics_startup_and_linear_gain_smoothing_match_exponentials(
    tmp_path: Path, kind: str
) -> None:
    raw = dynamics_graph(
        kind,
        attack=0.5 if kind == "gate" else 0.05,
        release=0.25 if kind == "gate" else 0.2,
    ).model_dump()
    if kind == "gate":
        raw["processors"][0]["hold_seconds"] = 0
    prepared = effects.prepare(audio_effects.EffectGraph.model_validate(raw), 48000)
    first = (
        1 - np.exp(-np.arange(1, 24001) / 24000)
        if kind == "gate"
        else 0.5 + 0.5 * np.exp(-np.arange(1, 24001) / 9600)
    )
    second = (
        first[-1] * np.exp(-np.arange(1, 24001) / 12000)
        if kind == "gate"
        else 1 - (1 - first[-1]) * np.exp(-np.arange(1, 24001) / 2400)
    )
    gain = np.concatenate([first, second])
    peak = np.repeat([0.2, 0] if kind == "gate" else [0.05, 0.2], 24000)
    inputs = {
        "main": program_audio(),
        "detector": np.column_stack([peak / 2, -peak, peak / 4]),
    }
    expected = inputs["main"] * gain[:, None]
    for origin, target in [("numpy", "native"), ("native", "numpy")]:
        renderer = effects.OfflineEffects(prepared, origin)
        before = renderer.advance(
            {n: a[:23928] for n, a in inputs.items()}, [], 0, 23928
        )
        saved = effects.EffectSnapshot.model_validate_json(
            renderer.snapshot().model_dump_json()
        )
        restored = effects.OfflineEffects(prepared, target)
        restored.restore(saved)
        after = restored.advance(
            {n: a[23928:] for n, a in inputs.items()}, [], 23928, 48000
        )
        check_audio(
            tmp_path / f"smooth-{origin}-{target}.wav",
            np.vstack([before, after]),
            expected,
        )
    actual, replay, _ = callback_audio(prepared, inputs, [])
    check_audio(tmp_path / "smooth-callback.wav", actual, expected)
    check_audio(tmp_path / "smooth-replay.wav", replay, actual)


def test_expander_hard_knee_ratio_one_and_sidechain_end(tmp_path: Path) -> None:
    prepared = effects.prepare(dynamics_graph("expander"), 48000)
    peak = np.repeat([0.2, 0.1, 0.05, 0, 1e-310, 0.01, 0.01, 0.01], 6000)
    peak[39000:] = 0
    gain = np.minimum(1, peak / 0.1)
    gain[30000:36000] = 1
    gain[36000:39000] = 0.01
    gain[39000:] = 0
    source = program_audio()
    source[45000:] = 0
    inputs = {"main": source, "detector": np.column_stack([peak / 2, -peak, peak / 4])}
    expected = source * gain[:, None]
    actions: list[audio_effects.EffectAction] = [
        audio_effects.ParameterAction(
            tick=30000, ordinal=0, processor="fx", parameter="ratio", value=1
        ),
        audio_effects.ParameterAction(
            tick=36000, ordinal=0, processor="fx", parameter="ratio", value=3
        ),
        audio_effects.InputEndAction(tick=39000, ordinal=0, input="detector"),
        audio_effects.InputEndAction(tick=45000, ordinal=0, input="main"),
    ]
    for backend in ["numpy", "native"]:
        actual = effects.OfflineEffects(prepared, backend).advance(
            inputs, actions, 0, 48000
        )
        check_audio(tmp_path / f"curve-{backend}.wav", actual, expected)
    actual, replay, drained = callback_audio(prepared, inputs, actions)
    assert drained
    check_audio(tmp_path / "curve-callback.wav", actual, expected)
    check_audio(tmp_path / "curve-replay.wav", replay, actual)


@pytest.mark.parametrize("kind", ["gate", "expander"])
def test_dynamics_invalid_callback_silences_the_block_and_restores(
    tmp_path: Path, kind: str
) -> None:
    prepared = effects.prepare(dynamics_graph(kind), 48000)
    runtime = _native.LiveRuntime.empty(48000, 2, 48000, 64)
    runtime.set_effect_graph(
        *effects.native_live_graph(prepared), 2, input_routes=[[-2], [-2]]
    )
    saved = runtime.snapshot()
    action = audio_effects.ParameterAction(
        tick=12003,
        ordinal=0,
        processor="fx",
        parameter="hold_seconds" if kind == "gate" else "ratio",
        value=0.01 if kind == "gate" else 2,
    )
    encoded = effects.native_live_actions(prepared, [action], 0, 48000)
    encoded[0, 4] = 1e308 if kind == "gate" else 0.5
    runtime.submit_effects(encoded)
    source = program_audio()
    output = np.ones_like(source)
    detector = np.full((48000, 3), 0.2)
    with pytest.raises(ValueError, match="[Gg]ate|[Ee]xpander"):
        runtime.process_into(output, [source, detector])
    assert runtime.failed()
    check_audio(tmp_path / "failed.wav", output, np.zeros_like(source))
    runtime.restore(saved)
    runtime.process_into(output, [source, detector])
    check_audio(tmp_path / "recovered.wav", output, source)


@pytest.mark.parametrize("kind", ["gate", "expander"])
def test_dynamics_align_sidechain_and_same_frame_follower_control(
    tmp_path: Path, kind: str
) -> None:
    raw = dynamics_graph(kind).model_dump()
    if kind == "gate":
        raw["processors"][0].update(hold_seconds=0, hysteresis_db=0)
    raw["processors"].extend(
        [
            {"kind": "waveshaper", "name": "shape", "mix": 0},
            {
                "kind": "follower",
                "name": "level",
                "attack_seconds": 0,
                "release_seconds": 0,
            },
        ]
    )
    raw["connections"][1]["source"] = {"kind": "processor", "processor": "shape"}
    raw["connections"].extend(
        [
            {
                "processor": "shape",
                "port": "input",
                "source": {"kind": "input", "input": "detector"},
            },
            {
                "processor": "level",
                "port": "input",
                "source": {"kind": "input", "input": "main"},
            },
        ]
    )
    raw["bindings"] = [{"kind": "follower", "name": "amplitude", "processor": "level"}]
    raw["modulation"] = {
        "sources": [
            {"name": "amplitude", "scope": "voice", "minimum": 0, "maximum": 1}
        ],
        "parameters": [
            {
                "target": {"name": "fx", "parameter": "threshold_db"},
                "scope": "voice",
                "unit": "db",
                "minimum": -40,
                "maximum": -20,
                "default": -20,
            }
        ],
        "routes": [
            {
                "name": "follow",
                "source": "amplitude",
                "target": {"name": "fx", "parameter": "threshold_db"},
                "operation": "add",
                "unit": "db",
                "points": [{"input": 0, "amount": -20}, {"input": 1, "amount": 0}],
            }
        ],
    }
    prepared = effects.prepare(audio_effects.EffectGraph.model_validate(raw), 48000)
    assert prepared.latency_frames == 48
    source = program_audio()
    peak = 0.03 * (1 + np.sin(2 * np.pi * 5 * np.arange(48000) / 48000))
    threshold = 10 ** ((-40 + 20 * np.max(np.abs(source), axis=1)) / 20)
    gain = (
        (peak >= threshold).astype(float)
        if kind == "gate"
        else np.minimum(1, peak / threshold)
    )
    expected = np.zeros_like(source)
    expected[48:] = source[:-48] * gain[:-48, None]
    inputs = {"main": source, "detector": np.column_stack([peak / 2, -peak, peak / 4])}
    for origin, target in [("numpy", "native"), ("native", "numpy")]:
        renderer = effects.OfflineEffects(prepared, origin)
        before = renderer.advance(
            {n: a[:23928] for n, a in inputs.items()}, [], 0, 23928
        )
        saved = effects.EffectSnapshot.model_validate_json(
            renderer.snapshot().model_dump_json()
        )
        restored = effects.OfflineEffects(prepared, target)
        restored.restore(saved)
        after = restored.advance(
            {n: a[23928:] for n, a in inputs.items()}, [], 23928, 48000
        )
        check_audio(
            tmp_path / f"aligned-{origin}-{target}.wav",
            np.vstack([before, after]),
            expected,
        )
    actual, replay, _ = callback_audio(prepared, inputs, [])
    check_audio(tmp_path / "aligned-callback.wav", actual, expected)
    check_audio(tmp_path / "aligned-replay.wav", replay, actual)


def test_gate_rejects_unrepresentable_prepared_hold() -> None:
    raw = dynamics_graph("gate").model_dump()
    raw["processors"][0]["hold_seconds"] = 1e308
    with pytest.raises(ValueError, match="frame format"):
        effects.prepare(audio_effects.EffectGraph.model_validate(raw), 48000)
