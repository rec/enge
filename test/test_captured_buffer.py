from pathlib import Path
from typing import Literal

import numpy as np
import pytest
from captured_buffer_oracle import capture_gain, region_audio
from test_effect_envelfo import modulated_graph
from test_follower import follower_graph
from test_modulated_multitap import replay_audio
from test_synth import check_audio
from ufor import audio_effects

from enge import _native, effects


def graph(
    mode: Literal["reverse", "scrub", "stutter"],
    latency: bool = False,
    capacity: int = 64,
) -> audio_effects.EffectGraph:
    processors: list[audio_effects.Processor] = (
        [audio_effects.FrequencyShifter(name="align")] if latency else []
    )
    processors.append(
        audio_effects.CapturedBuffer(
            name="capture",
            mode=mode,
            window_seconds=0.025,
            position=0.25 if mode == "scrub" else 0,
            maximum_pending_actions=capacity,
        )
    )
    return effects.serial_graph(
        audio_effects.AttachmentScope.master,
        audio_effects.GraphInput(
            name="main", channels=["left", "right"], source=audio_effects.MainStream()
        ),
        processors,
        48000,
    )


def source_audio(ended: int = 36003) -> np.ndarray:
    frames = np.arange(48000)
    source = np.column_stack(
        [
            0.1 * np.sin(2 * np.pi * f * frames / 48000)
            + 0.03 * np.cos(2 * np.pi * 73 * frames / 48000)
            for f in [223, 337]
        ]
    )
    source[ended:] = 0
    return source


def capture(at: int) -> audio_effects.CaptureAction:
    return audio_effects.CaptureAction(tick=at, ordinal=0, processor="capture")


def release(at: int) -> audio_effects.CaptureReleaseAction:
    return audio_effects.CaptureReleaseAction(tick=at, ordinal=0, processor="capture")


@pytest.mark.parametrize("mode", ["reverse", "scrub", "stutter"])
def test_captured_modes_follow_chronological_window_fades_endings_and_replay(
    tmp_path: Path, mode: Literal["reverse", "scrub", "stutter"]
) -> None:
    source = source_audio(24007)
    frames = np.arange(48000)
    wet = region_audio(source, 6001, mode, np.full(48000, 0.25))
    gain = capture_gain(6001, mode) * (1 - np.clip((frames - 30003) / 64, 0, 1))
    actions: list[audio_effects.EffectAction] = [
        capture(6001),
        audio_effects.InputEndAction(tick=24007, ordinal=0, input="main"),
        release(30003),
    ]
    replay_audio(
        tmp_path,
        graph(mode),
        source,
        actions,
        source + gain[:, None] * (wet - source),
        True,
    )


@pytest.mark.parametrize("mode", ["scrub", "stutter"])
def test_fresh_capture_and_release_retime_current_mixture_while_bypassed(
    tmp_path: Path, mode: Literal["scrub", "stutter"]
) -> None:
    source = source_audio()
    frames = np.arange(48000)
    first = region_audio(source, 6001, mode, np.full(48000, 0.25))
    second = region_audio(source, 23900, mode, np.full(48000, 0.25))
    replacement = np.clip((frames - 23900) / 64, 0, 1)
    first_gain = capture_gain(6001, mode) * (1 - replacement)
    second_gain = replacement.copy()
    after = frames >= 23932
    fade = 1 - np.clip((frames - 23932) / 64, 0, 1)
    first_gain[after] = fade[after] * 0.5
    second_gain[after] = fade[after] * 0.5
    # Re-capture after release verifies recording and free-region reuse under bypass.
    third = region_audio(source, 26007, mode, np.full(48000, 0.25))
    third_gain = capture_gain(26007, mode) * (1 - np.clip((frames - 33003) / 64, 0, 1))
    wet = (
        source
        + first_gain[:, None] * (first - source)
        + second_gain[:, None] * (second - source)
        + third_gain[:, None] * (third - source)
    )
    mix = (
        1 - np.clip((frames - 12003) / 64, 0, 1) + np.clip((frames - 27007) / 64, 0, 1)
    )
    actions: list[audio_effects.EffectAction] = [
        capture(6001),
        audio_effects.BypassAction(
            tick=12003, ordinal=0, processor="capture", bypassed=True
        ),
        capture(23900),
        release(23932),
        capture(26007),
        audio_effects.BypassAction(
            tick=27007, ordinal=0, processor="capture", bypassed=False
        ),
        release(33003),
        audio_effects.InputEndAction(tick=36003, ordinal=0, input="main"),
    ]
    replay_audio(
        tmp_path,
        graph(mode),
        source,
        actions,
        source + mix[:, None] * (wet - source),
        True,
    )


def test_scrub_position_accepts_scoped_motion_and_explicit_ramps(
    tmp_path: Path,
) -> None:
    raw = graph("scrub").model_dump()
    modulation = modulated_graph("gain")[0].model_dump()
    raw.update({n: modulation[n] for n in ("envelfos", "bindings", "modulation")})
    target = {"name": "capture", "parameter": "position"}
    raw["modulation"]["parameters"][0].update(
        target=target, unit="ratio", minimum=0, maximum=1, default=0.25
    )
    raw["modulation"]["routes"][0].update(
        target=target,
        unit="ratio",
        points=[{"input": -1, "amount": -0.05}, {"input": 1, "amount": 0.05}],
    )
    frames = np.arange(48000)
    positions = (
        0.25
        + 0.5 * np.clip((frames - 12003) / 15000, 0, 1)
        + 0.05 * np.where((frames // 12000) % 2 == 0, 1, -1)
    )
    source = source_audio()
    wet = region_audio(source, 6001, "scrub", positions)
    gain = capture_gain(6001, "scrub") * (1 - np.clip((frames - 33003) / 64, 0, 1))
    actions: list[audio_effects.EffectAction] = [
        capture(6001),
        audio_effects.ParameterAction(
            tick=12003,
            ordinal=0,
            processor="capture",
            parameter="position",
            value=0.75,
            duration_frames=15000,
        ),
        release(33003),
        audio_effects.InputEndAction(tick=36003, ordinal=0, input="main"),
    ]
    replay_audio(
        tmp_path,
        audio_effects.EffectGraph.model_validate(raw),
        source,
        actions,
        source + gain[:, None] * (wet - source),
        True,
    )


@pytest.mark.parametrize("mode", ["scrub", "stutter"])
def test_held_capture_survives_input_end_until_explicit_stop(
    tmp_path: Path, mode: Literal["scrub", "stutter"]
) -> None:
    source = source_audio(12003)
    wet = region_audio(source, 6001, mode, np.full(48000, 0.25))
    gain = capture_gain(6001, mode)
    expected = source + gain[:, None] * (wet - source)
    actions: list[audio_effects.EffectAction] = [
        capture(6001),
        audio_effects.InputEndAction(tick=12003, ordinal=0, input="main"),
    ]
    replay_audio(tmp_path, graph(mode), source, actions, expected, False)
    expected[36003:] = 0
    actions.append(audio_effects.StopAction(tick=36003, ordinal=0))
    replay_audio(tmp_path, graph(mode), source, actions, expected, True)


@pytest.mark.parametrize("mode", ["reverse", "scrub", "stutter"])
def test_capture_commands_follow_upstream_latency_and_survive_pending_snapshot(
    tmp_path: Path, mode: Literal["reverse", "scrub", "stutter"]
) -> None:
    source = source_audio()
    delayed = np.vstack([np.zeros((512, 2)), source])[:48000]
    frames = np.arange(48000)
    wet = region_audio(delayed, 23900 + 512, mode, np.full(48000, 0.25))
    gain = capture_gain(23900 + 512, mode) * (
        1 - np.clip((frames - 30003 - 512) / 64, 0, 1)
    )
    actions: list[audio_effects.EffectAction] = [
        capture(23900),
        release(30003),
        audio_effects.InputEndAction(tick=36003, ordinal=0, input="main"),
    ]
    prepared = effects.prepare(graph(mode, latency=True), 48000)
    assert prepared.latency_frames == 512
    replay_audio(
        tmp_path,
        prepared.definition,
        source,
        actions,
        delayed + gain[:, None] * (wet - delayed),
        True,
    )


@pytest.mark.parametrize("case", ["history", "ended", "regions", "capacity"])
def test_capture_failures_silence_latch_and_restore_prepared_storage(
    tmp_path: Path, case: str
) -> None:
    definition = graph(
        "stutter", latency=case == "capacity", capacity=1 if case == "capacity" else 64
    )
    prepared = effects.prepare(definition, 48000)
    runtime = _native.LiveRuntime.empty(48000, 2, 48000, 64)
    runtime.set_effect_graph(
        *effects.native_live_graph(prepared), 2, input_routes=[[-2]]
    )
    saved = runtime.snapshot()
    node = 1 if case == "capacity" else 0
    rows = {
        "history": [[1000, 4, node, 0, 0, 0]],
        "ended": [[12003, 2, 0, 0, 0, 0], [12003, 4, node, 0, 0, 0]],
        "regions": [
            [6001, 4, node, 0, 0, 0],
            [12003, 4, node, 0, 0, 0],
            [12004, 4, node, 0, 0, 0],
        ],
        "capacity": [[6001, 4, node, 0, 0, 0], [6002, 5, node, 0, 0, 0]],
    }[case]
    runtime.submit_effects(np.asarray(rows, dtype=np.float64))
    source = source_audio()
    actual = np.ones_like(source)
    with pytest.raises(
        ValueError,
        match={
            "history": "complete history",
            "ended": "live input",
            "regions": "regions occupied",
            "capacity": "capacity exhausted",
        }[case],
    ):
        runtime.process_into(actual, [source])
    assert runtime.failed()
    check_audio(tmp_path / "failed.wav", actual, np.zeros_like(source))
    runtime.restore(saved)
    # Enqueue after restoring an empty queue to check prepared capacity survives.
    runtime.submit_effects(
        np.array(
            [[6001, 4, node, 0, 0, 0], [30003, 5, node, 0, 0, 0]], dtype=np.float64
        )
    )
    runtime.process_into(actual, [source])
    dry = (
        np.vstack([np.zeros((512, 2)), source])[:48000]
        if case == "capacity"
        else source
    )
    delay = 512 if case == "capacity" else 0
    wet = region_audio(dry, 6001 + delay, "stutter", np.zeros(48000))
    gain = capture_gain(6001 + delay, "stutter") * (
        1 - np.clip((np.arange(48000) - 30003 - delay) / 64, 0, 1)
    )
    check_audio(tmp_path / "restored.wav", actual, dry + gain[:, None] * (wet - dry))


def test_capture_preparation_rejects_windows_too_short_for_the_fixed_fades() -> None:
    raw = graph("reverse").model_dump()
    raw["processors"][0]["window_seconds"] = 128 / 48000
    with pytest.raises(ValueError, match="129 frames"):
        effects.prepare(audio_effects.EffectGraph.model_validate(raw), 48000)


def test_maximum_two_second_capture_uses_the_complete_prepared_window(
    tmp_path: Path,
) -> None:
    raw = graph("stutter").model_dump()
    raw["processors"][0]["window_seconds"] = 2
    prepared = effects.prepare(audio_effects.EffectGraph.model_validate(raw), 48000)
    frames = np.arange(144000)
    source = np.column_stack(
        [0.1 * np.sin(2 * np.pi * f * frames / 48000) for f in [227, 331]]
    )
    source[140003:] = 0
    wet = region_audio(source, 96000, "stutter", np.zeros(144000), length=96000)
    gain = np.clip((frames - 96000) / 64, 0, 1) * (
        1 - np.clip((frames - 138003) / 64, 0, 1)
    )
    actions: list[audio_effects.EffectAction] = [
        capture(96000),
        release(138003),
        audio_effects.InputEndAction(tick=140003, ordinal=0, input="main"),
    ]
    runtime = _native.LiveRuntime.empty(48000, 2, 48000, 64)
    runtime.set_effect_graph(
        *effects.native_live_graph(prepared), 2, input_routes=[[-2]]
    )
    actual = np.empty_like(source)
    for start in range(0, 144000, 48000):
        end = start + 48000
        batch = [a for a in actions if start <= a.tick < end]
        if batch:
            runtime.submit_effects(
                effects.native_live_actions(prepared, batch, start, end)
            )
        runtime.process_into(actual[start:end], [source[start:end]])
    assert runtime.effects_drained
    check_audio(
        tmp_path / "maximum-window.wav", actual, source + gain[:, None] * (wet - source)
    )


@pytest.mark.parametrize("backend", ["numpy", "native", "callback"])
@pytest.mark.parametrize("after_end", [0, 1])
def test_live_follower_control_cannot_keep_ended_capture_audio_live(
    tmp_path: Path, backend: Literal["numpy", "native", "callback"], after_end: int
) -> None:
    raw = follower_graph().model_dump()
    raw["processors"][0] = audio_effects.CapturedBuffer(
        name="gain", mode="scrub", window_seconds=0.025, position=0.25
    ).model_dump()
    target = {"name": "gain", "parameter": "position"}
    raw["modulation"]["parameters"][0].update(
        target=target, unit="ratio", minimum=0, maximum=1, default=0.25
    )
    raw["modulation"]["routes"][0].update(
        target=target,
        unit="ratio",
        points=[{"input": 0, "amount": 0}, {"input": 1, "amount": 0.1}],
    )
    prepared = effects.prepare(audio_effects.EffectGraph.model_validate(raw), 48000)
    source = source_audio(12003)
    detector = np.full_like(source, 0.25)
    actual = np.zeros_like(source)
    if backend == "callback":
        runtime = _native.LiveRuntime.empty(48000, 2, 48000, 64)
        runtime.set_effect_graph(
            *effects.native_live_graph(prepared), 2, input_routes=[[-2], [-2]]
        )
        runtime.submit_effects(
            np.array(
                [
                    [12003, 2, 0, 0, 0, 0],
                    [12003 + after_end, 4, prepared.order.index("gain"), 0, 0, 0],
                ],
                dtype=np.float64,
            )
        )
        with pytest.raises(ValueError, match="live input"):
            runtime.process_into(actual, [source, detector])
        check_audio(tmp_path / "failed.wav", actual, np.zeros_like(source))
    else:
        renderer = effects.OfflineEffects(prepared, backend)
        at = 12003 + after_end
        end_action = audio_effects.InputEndAction(tick=12003, ordinal=0, input="main")
        actual[:at] = renderer.advance(
            {"main": source[:at], "detector": detector[:at]},
            [end_action] if after_end else [],
            0,
            at,
        )
        actions: list[audio_effects.EffectAction] = [] if after_end else [end_action]
        actions.append(
            audio_effects.CaptureAction(tick=at, ordinal=1, processor="gain")
        )
        with pytest.raises(ValueError, match="Effect processing failed") as failure:
            renderer.advance(
                {"main": source[at : at + 1], "detector": detector[at : at + 1]},
                actions,
                at,
                at + 1,
            )
        assert "live input" in str(failure.value.__cause__)
        check_audio(tmp_path / "before-error.wav", actual, source)
