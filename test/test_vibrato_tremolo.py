from pathlib import Path

import numpy as np
import pytest
from test_modulated_multitap import replay_audio
from ufor import audio_effects


def input_stream(stereo: bool) -> audio_effects.GraphInput:
    return audio_effects.GraphInput(
        name="main",
        channels=["left", "right"] if stereo else ["centre"],
        source=audio_effects.MainStream(),
    )


def source_audio(stereo: bool) -> np.ndarray:
    time = np.arange(48000) / 48000
    source = np.column_stack(
        [
            0.1 * np.sin(2 * np.pi * f * time) + 0.03 * np.cos(6 * np.pi * f * time)
            for f in ([220, 330] if stereo else [220])
        ]
    )
    source[36003:] = 0
    return source


def delayed_audio(
    source: np.ndarray, depth: float, rate: float, phase: float
) -> np.ndarray:
    frames = np.arange(48000)
    positions = frames - np.maximum(
        1, 48000 * (0.005 + depth * np.sin(2 * np.pi * (rate * frames / 48000 + phase)))
    )
    return np.column_stack(
        [
            np.interp(
                positions,
                np.arange(-1, 48000),
                np.concatenate([[0], source[:, c]]),
                left=0,
                right=0,
            )
            for c in range(source.shape[1])
        ]
    )


@pytest.mark.parametrize(
    "stereo,depth,phase",
    [(False, 0, 0.25), (True, 0.1, 0.25), (True, 1, 0.25), (False, 0.5, 0.125)],
)
def test_tremolo_matches_linear_amplitude_and_shared_phase_across_replay(
    tmp_path: Path, stereo: bool, depth: float, phase: float
) -> None:
    graph = audio_effects.tremolo_graph(
        "pulse",
        audio_effects.AttachmentScope.master,
        input_stream(stereo),
        48000,
        depth=depth,
        phase=phase,
    )
    source = source_audio(stereo)
    time = np.arange(48000) / 48000
    gain = 1 - depth / 2 + depth / 2 * np.sin(2 * np.pi * (5 * time + phase))
    expected = source * gain[:, None]
    actions: list[audio_effects.EffectAction] = [
        audio_effects.InputEndAction(tick=36003, ordinal=0, input="main")
    ]
    replay_audio(tmp_path, graph, source, actions, expected, True)


@pytest.mark.parametrize(
    "stereo,depth,rate,phase",
    [(False, 0, 5, 0), (True, 0.0005, 5, 0), (False, 0.0005, 3.5, 0.125)],
)
def test_vibrato_matches_fractional_delay_bias_and_shared_motion_across_replay(
    tmp_path: Path, stereo: bool, depth: float, rate: float, phase: float
) -> None:
    graph = audio_effects.vibrato_graph(
        "pitch",
        audio_effects.AttachmentScope.master,
        input_stream(stereo),
        48000,
        depth_seconds=depth,
        rate_hz=rate,
        phase=phase,
    )
    source = source_audio(stereo)
    expected = delayed_audio(source, depth, rate, phase)
    actions: list[audio_effects.EffectAction] = [
        audio_effects.InputEndAction(tick=36003, ordinal=0, input="main")
    ]
    replay_audio(tmp_path, graph, source, actions, expected, True)


def test_vibrato_warm_bypass_keeps_shared_motion_and_history(tmp_path: Path) -> None:
    graph = audio_effects.vibrato_graph(
        "pitch", audio_effects.AttachmentScope.master, input_stream(True), 48000
    )
    source = source_audio(True)
    wet = delayed_audio(source, 0.0005, 5, 0)
    frames = np.arange(48000)
    mix = (
        1 - np.clip((frames - 15003) / 64, 0, 1) + np.clip((frames - 27007) / 64, 0, 1)
    )
    expected = source + mix[:, None] * (wet - source)
    actions: list[audio_effects.EffectAction] = [
        audio_effects.BypassAction(
            tick=15003, ordinal=0, processor="pitch", bypassed=True
        ),
        audio_effects.BypassAction(
            tick=27007, ordinal=0, processor="pitch", bypassed=False
        ),
        audio_effects.InputEndAction(tick=36003, ordinal=0, input="main"),
    ]
    replay_audio(tmp_path, graph, source, actions, expected, True)


def test_tremolo_stop_ends_immediately(tmp_path: Path) -> None:
    graph = audio_effects.tremolo_graph(
        "pulse",
        audio_effects.AttachmentScope.master,
        input_stream(True),
        48000,
        depth=1,
    )
    source = source_audio(True)
    time = np.arange(48000) / 48000
    expected = source * (0.5 + 0.5 * np.sin(2 * np.pi * (5 * time + 0.25)))[:, None]
    expected[24007:] = 0
    actions: list[audio_effects.EffectAction] = [
        audio_effects.StopAction(tick=24007, ordinal=0)
    ]
    replay_audio(tmp_path, graph, source, actions, expected, True)
