from pathlib import Path

import numpy as np
import pytest
from test_persistent_effects import callback_audio
from test_synth import check_audio
from ufor import audio_effects

from enge import _native, effects


def follower_graph() -> audio_effects.EffectGraph:
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
                    "channels": ["left", "right"],
                    "source": {"kind": "host", "stream": "detector"},
                },
            ],
            "processors": [
                {"kind": "gain", "name": "gain"},
                {
                    "kind": "follower",
                    "name": "level",
                    "attack_seconds": 0.005,
                    "release_seconds": 0.01,
                },
            ],
            "connections": [
                {
                    "processor": "gain",
                    "port": "input",
                    "source": {"kind": "input", "input": "main"},
                },
                {
                    "processor": "level",
                    "port": "input",
                    "source": {"kind": "input", "input": "detector"},
                },
            ],
            "output": {"kind": "processor", "processor": "gain"},
            "maximum_block_frames": 48000,
            "bindings": [
                {"kind": "follower", "name": "amplitude", "processor": "level"}
            ],
            "modulation": {
                "sources": [
                    {"name": "amplitude", "scope": "voice", "minimum": 0, "maximum": 1}
                ],
                "parameters": [
                    {
                        "target": {"name": "gain", "parameter": "gain_db"},
                        "scope": "voice",
                        "unit": "db",
                        "default": 0,
                        "minimum": -63,
                        "maximum": 0,
                    }
                ],
                "routes": [
                    {
                        "name": "follow",
                        "source": "amplitude",
                        "target": {"name": "gain", "parameter": "gain_db"},
                        "operation": "add",
                        "unit": "db",
                        "points": [
                            {"input": 0, "amount": -60},
                            {"input": 1, "amount": 0},
                        ],
                    }
                ],
            },
        }
    )


def test_follower_linked_peak_attack_release_end_and_portable_replay(
    tmp_path: Path,
) -> None:
    prepared = effects.prepare(follower_graph(), 48000)
    frames = np.arange(48000)
    source = np.column_stack(
        [0.2 * np.sin(2 * np.pi * f * frames / 48000) for f in [220, 330]]
    )
    source[24000:] = 0
    peak = np.repeat([0.8, 0.2, 0, 0], 12000)
    detector = np.column_stack([-peak, peak / 2])
    first = 0.8 * (1 - np.exp(-np.arange(1, 12001) / 240))
    second = 0.2 + (first[-1] - 0.2) * np.exp(-np.arange(1, 12001) / 480)
    release = second[-1] * np.exp(-np.arange(1, 24001) / 480)
    level = np.concatenate([first, second, release])
    base = -3 * np.clip((frames - 23900) / 400, 0, 1)
    expected = source * (10 ** ((base - 60 + 60 * level) / 20))[:, None]
    inputs = {"main": source, "detector": detector}
    actions: list[audio_effects.EffectAction] = [
        audio_effects.ParameterAction(
            tick=23900,
            ordinal=0,
            processor="gain",
            parameter="gain_db",
            value=-3,
            duration_frames=400,
        ),
        audio_effects.InputEndAction(tick=24000, ordinal=0, input="main"),
        audio_effects.InputEndAction(tick=24000, ordinal=1, input="detector"),
    ]
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
        assert saved.processors["level"].follower_level == pytest.approx(level[23927])
        restored = effects.OfflineEffects(prepared, target)
        restored.restore(saved)
        if target == "native":
            before_rejection = restored.snapshot()
            broken = saved.model_copy(deep=True)
            broken.processors["level"].follower_level = -1
            with pytest.raises(ValueError, match="follower level"):
                restored.restore(broken)
            assert restored.snapshot() == before_rejection
        after = restored.advance(
            {n: a[23928:] for n, a in inputs.items()},
            [a for a in actions if a.tick >= 23928],
            23928,
            48000,
        )
        assert restored.drained
        assert restored.snapshot().processors["level"].follower_level == 0
        check_audio(
            tmp_path / f"follower-{origin}-{target}.wav",
            np.vstack([before, after]),
            expected,
        )
    actual, replay, drained = callback_audio(prepared, inputs, actions)
    assert drained
    check_audio(tmp_path / "follower-callback.wav", actual, expected)
    check_audio(tmp_path / "follower-replay.wav", replay, actual)


@pytest.mark.parametrize("delayed_path", ["detector", "audio"])
def test_follower_and_generator_contributions_align_with_delayed_audio(
    tmp_path: Path,
    delayed_path: str,
) -> None:
    raw = follower_graph().model_dump(mode="json")
    raw["processors"].append({"kind": "waveshaper", "name": "shape", "mix": 0})
    connection = raw["connections"][1 if delayed_path == "detector" else 0]
    source = connection["source"]
    connection["source"] = {"kind": "processor", "processor": "shape"}
    raw["connections"].append({"processor": "shape", "port": "input", "source": source})
    raw["processors"][1].update(attack_seconds=0, release_seconds=0)
    raw["envelfos"] = {"motion": {"kind": "lfo", "shape": "square", "rate": 2}}
    raw["bindings"].append({"kind": "envelfo", "name": "motion", "reference": "motion"})
    raw["modulation"]["sources"].append(
        {"name": "motion", "scope": "voice", "minimum": -1, "maximum": 1}
    )
    raw["modulation"]["parameters"][0].update(minimum=-66, maximum=6)
    raw["modulation"]["routes"].extend(
        [
            {
                "name": "motion-add",
                "source": "motion",
                "target": {"name": "gain", "parameter": "gain_db"},
                "operation": "add",
                "unit": "db",
                "points": [{"input": -1, "amount": -3}, {"input": 1, "amount": 3}],
            },
            {
                "name": "amplitude-multiply",
                "source": "amplitude",
                "target": {"name": "gain", "parameter": "gain_db"},
                "operation": "multiply",
                "unit": "ratio",
                "points": [{"input": 0, "amount": 0.5}, {"input": 1, "amount": 1}],
            },
        ]
    )
    graph = audio_effects.EffectGraph.model_validate(raw)
    prepared = effects.prepare(graph, 48000)
    assert prepared.latency_frames == 48
    frames = np.arange(48000)
    source = np.repeat(
        (0.1 * np.sin(2 * np.pi * 440 * frames / 48000))[:, None], 2, axis=1
    )
    peak = 0.4 + 0.2 * np.sin(2 * np.pi * 7 * frames / 48000)
    detector = np.column_stack([peak / 2, -peak])
    db = (-60 + 60 * peak + np.where(frames % 24000 < 12000, 3, -3)) * (
        0.5 + 0.5 * peak
    )
    expected = np.zeros_like(source)
    expected[48:] = source[:-48] * (10 ** (db[:-48] / 20))[:, None]
    inputs = {"main": source, "detector": detector}
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
        if target == "native":
            before_rejection = restored.snapshot()
            broken = saved.model_copy(deep=True)
            broken.processors["gain"].route_delays["motion-add"].history.pop()
            with pytest.raises(ValueError, match="modulation delay"):
                restored.restore(broken)
            assert restored.snapshot() == before_rejection
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


def test_follower_domain_failure_zeros_callback_and_restores(tmp_path: Path) -> None:
    prepared = effects.prepare(follower_graph(), 48000)
    runtime = _native.LiveRuntime.empty(48000, 2, 48000, 64)
    runtime.set_effect_graph(
        *effects.native_live_graph(prepared), 4, input_routes=[[-2], [-2]]
    )
    saved = runtime.snapshot()
    source = np.full((48000, 2), 0.1)
    output = np.ones_like(source)
    with pytest.raises(ValueError, match="outside its domain"):
        runtime.process_into(output, [source, np.full_like(source, 2)])
    assert runtime.failed()
    check_audio(tmp_path / "failed.wav", output, np.zeros_like(source))
    runtime.restore(saved)
    runtime.process_into(output, [source, np.full_like(source, 1e-32)])
    check_audio(tmp_path / "recovered.wav", output, source * 0.001)
