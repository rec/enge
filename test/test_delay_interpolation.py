from pathlib import Path

import numpy as np
import pytest
from test_effects import graph_input
from test_live import native_oscillator_runtime
from test_synth import check_audio
from ufor import audio_effects

from enge import _native, effects


def source_audio() -> np.ndarray:
    return native_oscillator_runtime().process_actions(
        48000, 0, 1, 0, np.array([[0, 0, 0, 220, 0.2, 0]], dtype=np.float64)
    )


def delayed(source: np.ndarray, positions: np.ndarray) -> np.ndarray:
    return np.column_stack(
        [
            np.interp(
                positions[:, c], np.arange(-1, 48000), np.r_[0.0, source[:, c]], left=0
            )
            for c in range(2)
        ]
    )


def callback_audio(
    definition: effects.PreparedEffects,
    actions: list[audio_effects.EffectAction],
    split: int,
) -> tuple[np.ndarray, np.ndarray]:
    runtime = _native.LiveRuntime([native_oscillator_runtime()], 997, 128, 4)
    runtime.set_effect_graph(*effects.native_live_graph(definition), 4)
    effects.configure_native_generators(runtime, definition)
    runtime.submit(0, np.array([[0, 0, 0, 220, 0.2, 0]], dtype=np.float64))
    actual = np.empty((48000, 2))
    for first, end in [(0, split), (split, 48000)]:
        if first:
            saved = runtime.snapshot()
        for start in range(first, end, 997):
            stop = min(start + 997, end)
            batch = [a for a in actions if start <= a.tick < stop]
            if batch:
                runtime.submit_effects(
                    effects.native_live_actions(definition, batch, start, stop)
                )
            runtime.process_into(actual[start:stop])
    runtime.restore(saved)
    replay = actual.copy()
    for start in range(split, 48000, 997):
        stop = min(start + 997, 48000)
        batch = [a for a in actions if start <= a.tick < stop]
        if batch:
            runtime.submit_effects(
                effects.native_live_actions(definition, batch, start, stop)
            )
        runtime.process_into(replay[start:stop])
    return actual, replay


@pytest.mark.parametrize("kind", ["tap", "modulated"])
def test_fractional_delay_retains_the_oldest_interpolation_sample(
    tmp_path: Path, kind: str
) -> None:
    processor = (
        audio_effects.TapDelay(
            name="delay", delay_seconds=3.25 / 48000, maximum_delay_seconds=3.25 / 48000
        )
        if kind == "tap"
        else audio_effects.ModulatedDelay(
            name="delay",
            base_delay_seconds=3.25 / 48000,
            maximum_delay_seconds=3.25 / 48000,
            rate_hz=1,
        )
    )
    definition = effects.prepare(
        effects.serial_graph(
            audio_effects.AttachmentScope.master, graph_input(), [processor], 48000
        ),
        48000,
    )
    source = source_audio()
    positions = np.column_stack([np.arange(48000) - 3.25] * 2)
    expected = delayed(source, positions)
    reference = effects.OfflineEffects(definition).advance(
        {"main": source}, [], 0, 48000
    )
    actual, replay = callback_audio(definition, [], 24001)
    check_audio(tmp_path / "fractional-reference.wav", reference, expected)
    check_audio(tmp_path / "fractional-callback.wav", actual, expected)
    check_audio(tmp_path / "fractional-replay.wav", replay, actual)


def test_chorus_moves_continuously_between_samples_with_stereo_phase(
    tmp_path: Path,
) -> None:
    processor = audio_effects.ModulatedDelay(
        name="delay",
        base_delay_seconds=200.25 / 48000,
        depth_seconds=40.5 / 48000,
        maximum_delay_seconds=250.75 / 48000,
        rate_hz=5,
        channel_phase_offsets=[0, 0.25],
    )
    definition = effects.prepare(
        effects.serial_graph(
            audio_effects.AttachmentScope.master, graph_input(), [processor], 48000
        ),
        48000,
    )
    source = source_audio()
    frames = np.arange(48000)
    phases = frames[:, None] * 5 / 48000 + [0, 0.25]
    positions = frames[:, None] - 200.25 - 40.5 * np.sin(2 * np.pi * phases)
    expected = delayed(source, positions)
    reference = effects.OfflineEffects(definition).advance(
        {"main": source}, [], 0, 48000
    )
    actual, replay = callback_audio(definition, [], 17003)
    check_audio(tmp_path / "chorus-reference.wav", reference, expected)
    check_audio(tmp_path / "chorus-callback.wav", actual, expected)
    check_audio(tmp_path / "chorus-replay.wav", replay, actual)


@pytest.mark.parametrize("overlap", [False, True])
def test_tap_retimes_from_the_current_read_head_mixture(
    tmp_path: Path, overlap: bool
) -> None:
    processor = audio_effects.TapDelay(
        name="delay", delay_seconds=31.25 / 48000, maximum_delay_seconds=100 / 48000
    )
    definition = effects.prepare(
        effects.serial_graph(
            audio_effects.AttachmentScope.master, graph_input(), [processor], 48000
        ),
        48000,
    )
    actions = [
        audio_effects.ParameterAction(
            tick=12000,
            ordinal=0,
            processor="delay",
            parameter="delay_seconds",
            value=83.75 / 48000,
        )
    ]
    if overlap:
        actions.append(
            audio_effects.ParameterAction(
                tick=12023,
                ordinal=0,
                processor="delay",
                parameter="delay_seconds",
                value=41.5 / 48000,
            )
        )
    source = source_audio()
    frames = np.arange(48000)
    old = np.clip(1 - (frames - 12000) / 64, 0, 1)
    middle = 1 - old
    if overlap:
        middle[12023:] = 23 / 64 * np.clip(1 - (frames[12023:] - 12023) / 64, 0, 1)
    newest = 1 - old - middle
    expected = sum(
        w[:, None] * delayed(source, np.column_stack([frames - d] * 2))
        for w, d in [(old, 31.25), (middle, 83.75), (newest, 41.5)]
    )
    renderer = effects.OfflineEffects(definition)
    head = renderer.advance({"main": source[:12031]}, actions, 0, 12031)
    saved = effects.EffectSnapshot.model_validate_json(
        renderer.snapshot().model_dump_json()
    )
    renderer.restore(saved)
    reference = np.concatenate(
        [head, renderer.advance({"main": source[12031:]}, [], 12031, 48000)]
    )
    actual, replay = callback_audio(definition, actions, 12031)
    check_audio(tmp_path / "retime-reference.wav", reference, expected)
    check_audio(tmp_path / "retime-callback.wav", actual, expected)
    check_audio(tmp_path / "retime-replay.wav", replay, actual)


def test_samplewise_retiming_reuses_its_prepared_read_heads(tmp_path: Path) -> None:
    processor = audio_effects.TapDelay(
        name="delay",
        delay_seconds=31.25 / 48000,
        maximum_delay_seconds=100 / 48000,
        feedback=0.25,
    )
    definition = effects.prepare(
        effects.serial_graph(
            audio_effects.AttachmentScope.master, graph_input(), [processor], 48000
        ),
        48000,
    )
    actions = [
        audio_effects.ParameterAction(
            tick=24000 + i,
            ordinal=0,
            processor="delay",
            parameter="delay_seconds",
            value=(32.5 + i / 2) / 48000,
        )
        for i in range(130)
    ]
    source = source_audio()
    renderer = effects.OfflineEffects(definition)
    head = renderer.advance(
        {"main": source[:24065]}, [a for a in actions if a.tick < 24065], 0, 24065
    )
    saved = effects.EffectSnapshot.model_validate_json(
        renderer.snapshot().model_dump_json()
    )
    renderer.restore(saved)
    reference = np.concatenate(
        [
            head,
            renderer.advance(
                {"main": source[24065:]},
                [a for a in actions if a.tick >= 24065],
                24065,
                48000,
            ),
        ]
    )
    # Keep blocks short enough that each complete action batch fits the native queue.
    actual, replay = callback_audio(definition, actions, 24065)
    check_audio(tmp_path / "dense-retime.wav", actual, reference)
    check_audio(tmp_path / "dense-replay.wav", replay, actual)
