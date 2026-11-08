from pathlib import Path

import numpy as np
import pytest
from test_persistent_effects import callback_audio
from test_synth import check_audio
from ufor import audio_effects

from enge import effects


def profile_taps(profile: str, stereo: bool) -> list[audio_effects.ModulatedTap]:
    return [
        audio_effects.ModulatedTap(
            delay=audio_effects.ModulatedDelay(
                name=n,
                base_delay_seconds=b,
                depth_seconds=d,
                maximum_delay_seconds=0.05,
                rate_hz=r,
                channel_phase_offsets=[p, (p + 0.37) % 1] if stereo else [p],
            ),
            gain_db=g,
            pan=s,
        )
        for n, b, d, r, p, g, s in PROFILES[profile]
    ]


def profile_graph(
    profile: str, stereo: bool, taps: list[audio_effects.ModulatedTap]
) -> audio_effects.EffectGraph:
    return audio_effects.modulated_multitap_graph(
        profile,
        audio_effects.AttachmentScope.master,
        audio_effects.GraphInput(
            name="main",
            channels=["left", "right"] if stereo else ["centre"],
            source=audio_effects.MainStream(),
        ),
        taps,
        48000,
        dry_gain_db=0 if profile == "doubling" else -3,
    )


def replay_audio(
    tmp_path: Path,
    graph: audio_effects.EffectGraph,
    source: np.ndarray,
    actions: list[audio_effects.EffectAction],
    expected: np.ndarray,
    drained: bool,
) -> None:
    prepared = effects.prepare(graph, 48000)
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
        assert restored.drained == drained
        check_audio(
            tmp_path / f"{origin}-{target}.wav", np.vstack([before, after]), expected
        )
    actual, replay, ended = callback_audio(prepared, {"main": source}, actions)
    assert ended == drained
    check_audio(tmp_path / "callback.wav", actual, expected)
    check_audio(tmp_path / "replay.wav", replay, actual)


@pytest.mark.parametrize("profile", ["doubling", "ensemble", "adt"])
@pytest.mark.parametrize("stereo", [False, True])
def test_modulated_voices_match_moving_reads_placement_automation_and_endings(
    tmp_path: Path, profile: str, stereo: bool
) -> None:
    taps = profile_taps(profile, stereo)
    graph = profile_graph(profile, stereo, taps)
    frames = np.arange(48000)
    source = np.column_stack(
        [
            0.1 * np.sin(2 * np.pi * f * frames / 48000)
            + 0.03 * np.cos(2 * np.pi * 3 * f * frames / 48000)
            for f in ([220, 330] if stereo else [220])
        ]
    )
    source[30000:] = 0
    expected = (
        source.copy() if stereo else np.repeat(source / np.sqrt(2), 2, axis=1)
    ) * 10 ** ((0 if profile == "doubling" else -3) / 20)
    for index, tap in enumerate(taps):
        delay = tap.delay
        base = delay.base_delay_seconds + (
            0.001 * np.clip((frames - 18000) / 1200, 0, 1) if index == 0 else 0
        )
        wet = np.column_stack(
            [
                np.interp(
                    frames
                    - 48000
                    * (
                        base
                        + delay.depth_seconds
                        * np.sin(2 * np.pi * (delay.rate_hz * frames / 48000 + p))
                    ),
                    np.arange(-1, 48000),
                    np.concatenate([[0], source[:, c]]),
                    left=0,
                    right=0,
                )
                for c, p in enumerate(delay.channel_phase_offsets)
            ]
        )
        placement = (
            [1 - max(0, tap.pan), 1 + min(0, tap.pan)]
            if stereo
            else [np.cos((tap.pan + 1) * np.pi / 4), np.sin((tap.pan + 1) * np.pi / 4)]
        )
        gain = tap.gain_db + (
            (-15 - tap.gain_db) * np.clip((frames - 24000) / 1000, 0, 1)
            if index == 0
            else 0
        )
        expected += wet * placement * np.asarray(10 ** (gain / 20)).reshape(-1, 1)
    actions: list[audio_effects.EffectAction] = [
        audio_effects.ParameterAction(
            tick=18000,
            ordinal=0,
            processor=f"{profile}-tap-one",
            parameter="base_delay_seconds",
            value=taps[0].delay.base_delay_seconds + 0.001,
            duration_frames=1200,
        ),
        audio_effects.ParameterAction(
            tick=24000,
            ordinal=0,
            processor=profile,
            parameter="tap-one-gain_db",
            value=-15,
            duration_frames=1000,
        ),
        audio_effects.InputEndAction(tick=30000, ordinal=0, input="main"),
    ]
    replay_audio(tmp_path, graph, source, actions, expected, True)


def test_independent_shared_envelfo_routes_replace_periodic_voice_motion(
    tmp_path: Path,
) -> None:
    taps = [
        t.model_copy(
            update={"delay": t.delay.model_copy(update={"depth_seconds": 0.0})}
        )
        for t in profile_taps("doubling", True)
    ]
    graph = profile_graph("doubling", True, taps)
    raw = graph.model_dump()
    raw.update(
        envelfos={"motion": {"kind": "lfo", "shape": "square", "rate": 2}},
        bindings=[{"kind": "envelfo", "name": "motion", "reference": "motion"}],
        modulation={
            "sources": [
                {"name": "motion", "scope": "voice", "minimum": -1, "maximum": 1}
            ],
            "parameters": [
                {
                    "target": {
                        "name": f"doubling-tap-{t.delay.name}",
                        "parameter": "base_delay_seconds",
                    },
                    "unit": "seconds",
                    "scope": "voice",
                    "minimum": t.delay.base_delay_seconds - 0.002,
                    "maximum": t.delay.base_delay_seconds + 0.002,
                    "default": t.delay.base_delay_seconds,
                }
                for t in taps
            ],
            "routes": [
                {
                    "name": f"motion-{t.delay.name}",
                    "source": "motion",
                    "target": {
                        "name": f"doubling-tap-{t.delay.name}",
                        "parameter": "base_delay_seconds",
                    },
                    "unit": "seconds",
                    "operation": "add",
                    "points": [{"input": -1, "amount": -d}, {"input": 1, "amount": d}],
                }
                for t, d in zip(taps, [0.001, -0.001], strict=True)
            ],
        },
    )
    graph = audio_effects.EffectGraph.model_validate(raw)
    frames = np.arange(48000)
    source = np.column_stack(
        [0.1 * np.sin(2 * np.pi * f * frames / 48000) for f in [220, 330]]
    )
    expected = source.copy()
    square = np.where(frames % 24000 < 12000, 1, -1)
    for tap, depth in zip(taps, [0.001, -0.001], strict=True):
        position = frames - 48000 * (tap.delay.base_delay_seconds + depth * square)
        wet = np.column_stack(
            [
                np.interp(
                    position,
                    np.arange(-1, 48000),
                    np.concatenate([[0], source[:, c]]),
                    left=0,
                )
                for c in range(2)
            ]
        )
        expected += (
            wet * [1 - max(0, tap.pan), 1 + min(0, tap.pan)] * 10 ** (tap.gain_db / 20)
        )
    replay_audio(tmp_path, graph, source, [], expected, False)


PROFILES = {
    "doubling": [
        ("one", 0.012, 0.0008, 0.41, 0, -9, -0.8),
        ("two", 0.018, 0.0012, 0.67, 0.37, -9, 0.8),
    ],
    "ensemble": [
        ("one", 0.015, 0.002, 0.37, 0.05, -12, -1),
        ("two", 0.019, 0.0025, 0.53, 0.27, -12, -0.35),
        ("three", 0.023, 0.003, 0.71, 0.51, -12, 0.35),
        ("four", 0.027, 0.0018, 0.89, 0.78, -12, 1),
    ],
    "adt": [("one", 0.023, 0.003, 0.31, 0.23, -6, 0.5)],
}
