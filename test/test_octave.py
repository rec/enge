from pathlib import Path
from typing import Literal

import numpy as np
import pytest
from test_effects import graph_input
from test_persistent_effects import callback_audio
from test_synth import check_audio
from ufor import audio_effects

from enge import _native, effects


def divider_graph() -> audio_effects.EffectGraph:
    return effects.serial_graph(
        audio_effects.AttachmentScope.master,
        graph_input(),
        [audio_effects.OctaveDivider(name="octave")],
        96000,
    )


def test_divider_halves_independent_channel_frequencies_and_replays_its_tail(
    tmp_path: Path,
) -> None:
    frames = np.arange(96000)
    end = 12000
    source = np.column_stack([0.5 * (1 - 2 * ((frames // h) % 2)) for h in (150, 100)])
    source[end:] = 0
    level = np.where(
        frames < end,
        0.5 * (1 - np.exp(-(frames + 1) / 48)),
        0.5 * (1 - np.exp(-end / 48)) * np.exp(-(frames - end + 1) / 2400),
    )
    expected = np.column_stack(
        [
            np.where(
                frames < 2 * h,
                0,
                level
                * (1 - 2 * (((np.minimum(frames, end - 1) - 2 * h) // (2 * h)) % 2)),
            )
            for h in (150, 100)
        ]
    )
    definition = audio_effects.OctaveDivider(name="octave")
    retirement = int(
        np.flatnonzero((frames >= end) & (level < definition.tail_threshold))[0]
    )
    expected[retirement:] = 0
    prepared = effects.prepare(divider_graph(), 48000)
    assert prepared.latency_frames == 0
    actions: list[audio_effects.EffectAction] = [
        audio_effects.InputEndAction(tick=end, ordinal=0, input="main")
    ]
    pairs: list[tuple[Literal["numpy", "native"], Literal["numpy", "native"]]] = [
        ("numpy", "native"),
        ("native", "numpy"),
    ]
    for origin, target in pairs:
        first = effects.OfflineEffects(prepared, origin)
        head = first.advance({"main": source[:23928]}, actions, 0, 23928)
        saved = effects.EffectSnapshot.model_validate_json(
            first.snapshot().model_dump_json()
        )
        second = effects.OfflineEffects(prepared, target)
        second.restore(saved)
        tail = second.advance({"main": source[23928:]}, [], 23928, len(source))
        assert second.drained
        check_audio(
            tmp_path / f"{origin}-{target}.wav", np.vstack([head, tail]), expected
        )
    actual, replay, drained = callback_audio(prepared, {"main": source}, actions)
    assert drained
    check_audio(tmp_path / "callback.wav", actual, expected)
    check_audio(tmp_path / "replay.wav", replay, actual)


def test_divider_hysteresis_ignores_small_crossings_until_fully_armed(
    tmp_path: Path,
) -> None:
    source = np.zeros((48000, 2))
    source[:12000] = 0.02
    source[12000:24000] = -0.005
    source[24000:36000] = 0.02
    expected = np.zeros_like(source)
    prepared = effects.prepare(divider_graph(), 48000)
    for backend in BACKENDS:
        check_audio(
            tmp_path / f"{backend}.wav",
            effects.process_audio(
                prepared, {"main": source}, backend=backend, block_frames=997
            ),
            expected,
        )


@pytest.mark.parametrize("parameter,value", [(0, 0), (3, -1), (4, -1), (5, 0.1)])
def test_invalid_divider_controls_silence_the_block_and_allow_restore(
    tmp_path: Path, parameter: int, value: float
) -> None:
    prepared = effects.prepare(divider_graph(), 48000)
    runtime = _native.LiveRuntime.empty(48000, 2, 48000, 64)
    runtime.set_effect_graph(
        *effects.native_live_graph(prepared), 2, input_routes=[[-2]]
    )
    saved = runtime.snapshot()
    runtime.submit_effects(
        np.array([[12003, 0, 0, parameter, value, 0]], dtype=np.float64)
    )
    source = np.full((48000, 2), 0.5)
    actual = np.ones_like(source)
    with pytest.raises(ValueError, match="octave divider"):
        runtime.process_into(actual, [source])
    assert runtime.failed()
    check_audio(tmp_path / "failed.wav", actual, np.zeros_like(source))
    runtime.restore(saved)
    runtime.process_into(actual, [source])
    check_audio(tmp_path / "restored.wav", actual, np.zeros_like(source))


def test_divider_warm_bypass_follows_upstream_latency_and_stop_is_immediate(
    tmp_path: Path,
) -> None:
    frames = np.arange(48000)
    source = np.column_stack([0.5 * (1 - 2 * ((frames // h) % 2)) for h in (150, 100)])
    dry = np.vstack([np.zeros((48, 2)), source[:-48]])
    clock = np.maximum(frames - 48, 0)
    level = np.where(frames < 48, 0, 0.5 * (1 - np.exp(-(clock + 1) / 48)))
    wet = np.column_stack(
        [
            np.where(
                clock < 2 * h, 0, level * (1 - 2 * (((clock - 2 * h) // (2 * h)) % 2))
            )
            for h in (150, 100)
        ]
    )
    mix = (
        1 - np.clip((frames - 12051) / 64, 0, 1) + np.clip((frames - 24051) / 64, 0, 1)
    )
    expected = (1 - mix[:, None]) * dry + mix[:, None] * wet
    expected[36003:] = 0
    graph = effects.serial_graph(
        audio_effects.AttachmentScope.master,
        graph_input(),
        [
            audio_effects.Waveshaper(name="latency", mix=0),
            audio_effects.OctaveDivider(name="octave"),
        ],
        48000,
    )
    prepared = effects.prepare(graph, 48000)
    actions: list[audio_effects.EffectAction] = [
        audio_effects.BypassAction(
            tick=12003, ordinal=0, processor="octave", bypassed=True
        ),
        audio_effects.BypassAction(
            tick=24003, ordinal=0, processor="octave", bypassed=False
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


def test_malformed_divider_state_does_not_replace_live_state() -> None:
    renderer = effects.OfflineEffects(effects.prepare(divider_graph(), 48000), "native")
    original = renderer.snapshot()
    invalid = original.model_copy(deep=True)
    state = invalid.processors["octave"].octave
    assert state is not None
    state.levels.pop()
    with pytest.raises(ValueError, match="snapshot"):
        renderer.restore(invalid)
    assert renderer.snapshot() == original


BACKENDS: list[Literal["numpy", "native"]] = ["numpy", "native"]
