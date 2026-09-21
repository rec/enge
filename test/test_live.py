from pathlib import Path

import numpy as np
from test_effects import gain_graph
from test_noise import score as noise_score
from test_noise import trigger as noise_trigger
from test_synth import check_audio
from test_synth import score as synth_score
from ufor.synth_trace import prepare as prepare_trace

from enge import _native, effects, live, noise, synth


def test_action_queue_preserves_complete_batches_and_entry_boundary() -> None:
    queue = _native.ActionQueue(2)
    first = np.arange(12, dtype=np.float64).reshape(2, 6)
    second = np.arange(6, dtype=np.float64).reshape(1, 6) + 20
    queue.push(first)
    queue.push(second)
    with np.testing.assert_raises_regex(ValueError, "capacity"):
        queue.push(second)
    short = np.empty((1, 6))
    assert queue.drain_into(short) == 0
    output = np.empty((3, 6))
    assert queue.drain_into(output) == 3
    np.testing.assert_array_equal(output, np.concatenate((first, second)))
    assert queue.queued_batches() == 0


def native_oscillator_runtime() -> _native.SynthRuntime:
    return _native.SynthRuntime(
        48000,
        0,
        0.5,
        np.array([[1.0, 0.5]]),
        1,
        np.array([[0, 1]], dtype=np.float64),
        np.array([[0, 0]], dtype=np.float64),
        0,
        np.empty((0, 8), dtype=np.float64),
        [],
        np.empty((0, 6), dtype=np.float64),
        [],
        np.empty((0, 7), dtype=np.float64),
        [1, 0, 1e300, 0, -120000, 120000],
        1,
    )


def native_noise_runtime() -> _native.SynthRuntime:
    return _native.SynthRuntime.noise(
        48000,
        np.array([[0.25, 1.0]]),
        1,
        np.array([[0, 1]], dtype=np.float64),
        np.array([[0, 0]], dtype=np.float64),
        0,
        np.empty((0, 8), dtype=np.float64),
        [],
        np.empty((0, 6), dtype=np.float64),
        [],
        np.empty((0, 7), dtype=np.float64),
        [1, 0, 1e300, 0, -120000, 120000],
        1,
    )


def test_native_live_runtime_owns_sources_queue_scratch_and_snapshot(
    tmp_path: Path,
) -> None:
    oscillator = native_oscillator_runtime()
    white = native_noise_runtime()
    oscillator_start = np.array([[0, 0, 0, 220, 0.2, 0]], dtype=np.float64)
    key = 0xFEDCBA9876543210
    noise_start = np.array(
        [[0, 0, 0, key & 0xFFFF_FFFF, key >> 32, 0.1]], dtype=np.float64
    )
    expected = (
        oscillator.process_actions(48000, 0, 1, 0, oscillator_start)
        + white.process_actions(48000, 0, 1, 0, noise_start)
    ) * 10 ** (-3 / 20)
    runtime = _native.LiveRuntime(
        [native_oscillator_runtime(), native_noise_runtime()], 997, 64, 4
    )
    runtime.set_effect_graph(
        [0],
        np.array([[0, -1]], dtype=np.int64),
        np.array([[-3, 1]], dtype=np.float64),
        1,
        np.empty((0, 4), dtype=np.float64),
        4,
    )
    runtime.submit(0, oscillator_start)
    runtime.submit(1, noise_start)
    actual = np.empty_like(expected)
    for start in range(0, 48000, 997):
        end = min(start + 997, 48000)
        runtime.process_into(actual[start:end])
        if end == 23928:
            snapshot = runtime.snapshot()
    restored = _native.LiveRuntime(
        [native_oscillator_runtime(), native_noise_runtime()], 997, 64, 4
    )
    restored.set_effect_graph(
        [0],
        np.array([[0, -1]], dtype=np.int64),
        np.array([[-3, 1]], dtype=np.float64),
        1,
        np.empty((0, 4), dtype=np.float64),
        4,
    )
    restored.restore(snapshot)
    replay = np.empty((48000 - 23928, 2))
    for start in range(0, len(replay), 997):
        restored.process_into(replay[start : start + 997])

    check_audio(tmp_path / "native-live-runtime.wav", actual, expected)
    np.testing.assert_allclose(replay, actual[23928:], atol=0)


def test_native_live_runtime_owns_effect_graph_and_parameter_ramps(
    tmp_path: Path,
) -> None:
    start_action = np.array([[0, 0, 0, 220, 0.2, 0]], dtype=np.float64)
    source = native_oscillator_runtime().process_actions(48000, 0, 1, 0, start_action)
    kinds = [0, 1, 2]
    sources = np.array([[0, -1], [0, 1], [2, -1]], dtype=np.int64)
    initial = np.array(
        [[-12, 0.6, 0, 0], [0, 0.4, 0, 0], [0, 0.75, 1500, 0.7]],
        dtype=np.float64,
    )
    parameter_values = np.broadcast_to(initial[:, :2], (48000, 3, 2)).copy()
    duration = 257
    ramp = np.minimum(np.maximum(np.arange(48000) - 32, 0), duration) / duration
    parameter_values[:, 0, 0] = -12 + 12 * ramp
    filter_values = np.broadcast_to(initial[2, 2:], (48000, 1, 2)).copy()
    expected, _ = _native.render_effects(
        source[None, :, :],
        kinds,
        sources,
        parameter_values,
        3,
        48000,
        [
            None,
            None,
            ([0], np.zeros((1, 2, 2)), filter_values),
        ],
        [0, 0, 1e-300],
    )
    runtime = _native.LiveRuntime([native_oscillator_runtime()], 997, 64, 4)
    runtime.set_effect_graph(
        kinds,
        sources,
        initial,
        3,
        np.array([[2, 0, 2, 1e-300]], dtype=np.float64),
        4,
    )
    runtime.submit(0, start_action)
    runtime.submit_effects(np.array([[32, 0, 0, 0, 0, duration]], dtype=np.float64))
    actual = np.empty_like(expected)
    for start in range(0, 48000, 997):
        runtime.process_into(actual[start : start + 997])

    check_audio(tmp_path / "native-live-effects.wav", actual, expected)


def test_native_live_runtime_latches_failure_and_zeros_later_blocks() -> None:
    runtime = _native.LiveRuntime([native_oscillator_runtime()], 64, 64, 1)
    runtime.submit(0, np.array([[64, 0, 0, 220, 0.2, 0]], dtype=np.float64))
    output = np.ones((64, 2))
    with np.testing.assert_raises_regex(ValueError, "runtime block"):
        runtime.process_into(output)
    assert runtime.failed()
    np.testing.assert_array_equal(output, np.zeros_like(output))
    output.fill(1)
    runtime.process_into(output)
    np.testing.assert_array_equal(output, np.zeros_like(output))


def test_native_live_runtime_owns_sample_asset_and_traversal(tmp_path: Path) -> None:
    frames = np.arange(96000)
    audio = np.sin(2 * np.pi * frames * 220 / 48000)[:, None]
    buffer = _native.SampleBuffer(audio)

    def make_runtime() -> tuple[_native.LiveRuntime, int]:
        runtime = _native.LiveRuntime([native_oscillator_runtime()], 997, 64, 4)
        source = runtime.add_sample(
            buffer,
            (0, 96000, False, None),
            (0, 0, 0, 1, False, False, False, None),
            48000,
            np.array([[1.0, 0.5]]),
            1,
            np.array([[0, 1]], dtype=np.float64),
            np.array([[0, 0]], dtype=np.float64),
            0,
            2,
            4,
        )
        return runtime, source

    runtime, source = make_runtime()
    runtime.submit(source, np.array([[0, 0, 0, 48000, 0.2, 0]]))
    actual = np.empty((48000, 2))
    for start in range(0, 48000, 997):
        end = min(start + 997, 48000)
        runtime.process_into(actual[start:end])
        if end == 23928:
            snapshot = runtime.snapshot()
    restored, _ = make_runtime()
    restored.restore(snapshot)
    replay = np.empty((48000 - 23928, 2))
    for start in range(0, len(replay), 997):
        restored.process_into(replay[start : start + 997])

    expected = audio[:48000] * np.array([[0.2, 0.1]])
    check_audio(tmp_path / "native-live-sample.wav", actual, expected)
    np.testing.assert_allclose(replay, actual[23928:], atol=0)


def test_live_engine_mixes_heterogeneous_sources_and_restores(tmp_path: Path) -> None:
    synth_document = synth_score()
    noise_document = noise_score()
    synth_actions = prepare_trace(
        synth_document.body,
        [noise_trigger().model_copy(update={"pitch_hz": 220})],
        seed=0,
    ).actions
    noise_actions = prepare_trace(
        noise_document.body, [noise_trigger()], seed=19
    ).actions
    synth_definition = synth.prepare(synth_document)
    noise_definition = noise.prepare(noise_document)
    dry = synth.OfflineSynth(synth_definition, "native").advance(
        synth_actions, 0, 48000
    ) + noise.OfflineNoise(noise_definition, "native").advance(noise_actions, 0, 48000)
    prepared_effects = effects.prepare(gain_graph(-3), 48000)
    expected = effects.OfflineEffects(prepared_effects, "native").advance(
        {"main": dry}, [], 0, 48000
    )
    engine = live.LiveEngine(
        {
            "tone": synth.PersistentSynth(synth_definition, voices=2),
            "noise": noise.PersistentNoise(noise_definition, voices=2),
        },
        effects.OfflineEffects(prepared_effects, "native"),
    )
    chunks = []
    boundaries = [0, 997, 24001, 48000]
    all_actions = {"tone": synth_actions, "noise": noise_actions}
    for start, end in zip(boundaries, boundaries[1:], strict=False):
        chunks.append(
            engine.advance(
                {
                    n: [a for a in actions if start <= a.tick < end]
                    for n, actions in all_actions.items()
                },
                [],
                start,
                end,
            )
        )
        if end == 24001:
            snapshot = engine.snapshot()
    actual = np.concatenate(chunks)
    restored = live.LiveEngine(
        {
            "tone": synth.PersistentSynth(synth_definition, voices=2),
            "noise": noise.PersistentNoise(noise_definition, voices=2),
        },
        effects.OfflineEffects(prepared_effects, "native"),
    )
    restored.restore(snapshot)
    replay = restored.advance({"tone": [], "noise": []}, [], 24001, 48000)

    check_audio(tmp_path / "live-engine.wav", actual, expected)
    np.testing.assert_allclose(replay, actual[24001:], atol=0)
