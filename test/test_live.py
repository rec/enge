from pathlib import Path

import numpy as np
from test_effects import gain_graph, graph_input
from test_noise import score as noise_score
from test_noise import trigger as noise_trigger
from test_synth import check_audio
from test_synth import score as synth_score
from ufor import audio_effects
from ufor.samples import processing
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
        np.array([[-3, 1, 1]], dtype=np.float64),
        1,
        np.empty((0, 4), dtype=np.float64),
        np.empty((0, 4), dtype=np.int64),
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
        np.array([[-3, 1, 1]], dtype=np.float64),
        1,
        np.empty((0, 4), dtype=np.float64),
        np.empty((0, 4), dtype=np.int64),
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
    graph = audio_effects.EffectGraph(
        scope="host",
        owner="live-test",
        inputs=[graph_input()],
        processors=[
            audio_effects.Gain(name="trim", gain_db=-12, mix=0.6),
            audio_effects.Multiply(name="ring", mix=0.4),
            audio_effects.Filter(
                name="low",
                filters=[
                    processing.ResonantFilter(
                        name="stage", response="lowpass", cutoff_hz=1500, q=0.7
                    )
                ],
                mix=0.75,
                bypass_fade_frames=128,
            ),
        ],
        connections=[
            audio_effects.Connection(
                processor="trim",
                port="input",
                source=audio_effects.InputSource(input="main"),
            ),
            audio_effects.Connection(
                processor="ring",
                port="carrier",
                source=audio_effects.InputSource(input="main"),
            ),
            audio_effects.Connection(
                processor="ring",
                port="modulator",
                source=audio_effects.ProcessorSource(processor="trim"),
            ),
            audio_effects.Connection(
                processor="low",
                port="input",
                source=audio_effects.ProcessorSource(processor="ring"),
            ),
        ],
        output=audio_effects.ProcessorSource(processor="low"),
        maximum_block_frames=997,
    )
    prepared = effects.prepare(graph, 48000)
    (
        kinds,
        sources,
        initial,
        output_source,
        filter_rows,
        granulator_rows,
    ) = effects.native_live_graph(prepared)
    parameter_values = np.broadcast_to(initial[:, :2], (48000, 3, 2)).copy()
    duration = 257
    ramp = np.minimum(np.maximum(np.arange(48000) - 32, 0), duration) / duration
    parameter_values[:, 0, 0] = -12 + 12 * ramp
    bypass = 1 - np.minimum(np.maximum(np.arange(48000) - 16000, 0), 128) / 128
    parameter_values[:, 2, 1] *= bypass
    filter_values = np.broadcast_to(initial[2, 3:], (48000, 1, 2)).copy()
    expected, _ = _native.render_effects(
        source[None, :, :],
        kinds,
        sources,
        parameter_values,
        output_source,
        48000,
        [
            None,
            None,
            ([0], np.zeros((1, 2, 2)), filter_values),
        ],
        [0, 0, filter_rows[0, 3]],
    )
    runtime = _native.LiveRuntime([native_oscillator_runtime()], 997, 64, 4)
    runtime.set_effect_graph(
        kinds,
        sources,
        initial,
        output_source,
        filter_rows,
        granulator_rows,
        4,
    )
    runtime.submit(0, start_action)
    gain_action = audio_effects.ParameterAction(
        tick=32,
        ordinal=0,
        processor="trim",
        parameter="gain_db",
        value=0,
        duration_frames=duration,
    )
    runtime.submit_effects(effects.native_live_actions(prepared, [gain_action], 0, 997))
    actual = np.empty_like(expected)
    for start in range(0, 48000, 997):
        if start == 15952:
            runtime.submit_effects(
                effects.native_live_actions(
                    prepared,
                    [
                        audio_effects.BypassAction(
                            tick=16000,
                            ordinal=0,
                            processor="low",
                            bypassed=True,
                        )
                    ],
                    start,
                    start + 997,
                )
            )
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


def test_native_live_runtime_owns_bounded_granulator_state(tmp_path: Path) -> None:
    start_action = np.array([[0, 0, 0, 220, 0.2, 0]], dtype=np.float64)
    source = native_oscillator_runtime().process_actions(48000, 0, 1, 0, start_action)
    graph = effects.serial_graph(
        audio_effects.AttachmentScope.master,
        graph_input(),
        [
            audio_effects.Granulator(
                name="cloud",
                duration_seconds=0.015,
                density_hz=90,
                lookback_seconds=0.025,
                playback_ratio=1.25,
                position_jitter_seconds=0.003,
                history_seconds=0.06,
                maximum_grains=3,
                mix=0.8,
            )
        ],
        997,
    )
    prepared = effects.prepare(graph, 48000)
    actions: list[audio_effects.EffectAction] = [
        audio_effects.FreezeAction(
            tick=18000, ordinal=0, processor="cloud", frozen=True
        ),
        audio_effects.ParameterAction(
            tick=19000,
            ordinal=0,
            processor="cloud",
            parameter="density_hz",
            value=140,
            duration_frames=7000,
        ),
        audio_effects.FreezeAction(
            tick=30000, ordinal=0, processor="cloud", frozen=False
        ),
    ]
    expected = effects.process_audio(
        prepared,
        {"main": source},
        actions,
        backend="native",
        block_frames=997,
    )

    def make_runtime() -> _native.LiveRuntime:
        runtime = _native.LiveRuntime([native_oscillator_runtime()], 997, 64, 4)
        runtime.set_effect_graph(*effects.native_live_graph(prepared), 4)
        return runtime

    runtime = make_runtime()
    runtime.submit(0, start_action)
    actual = np.empty_like(expected)
    for start in range(0, 48000, 997):
        end = min(start + 997, 48000)
        block_actions = [a for a in actions if start <= a.tick < end]
        if block_actions:
            runtime.submit_effects(
                effects.native_live_actions(prepared, block_actions, start, end)
            )
        runtime.process_into(actual[start:end])
        if end == 23928:
            snapshot = runtime.snapshot()
    restored = make_runtime()
    restored.restore(snapshot)
    replay = np.empty((48000 - 23928, 2))
    for start in range(23928, 48000, 997):
        end = min(start + 997, 48000)
        block_actions = [a for a in actions if start <= a.tick < end]
        if block_actions:
            restored.submit_effects(
                effects.native_live_actions(prepared, block_actions, start, end)
            )
        restored.process_into(replay[start - 23928 : end - 23928])

    check_audio(tmp_path / "native-live-granulator.wav", actual, expected)
    np.testing.assert_allclose(replay, actual[23928:], atol=0)


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

    native = live.NativeLiveEngine(
        {
            "tone": synth.PersistentSynth(synth_definition, voices=2),
            "noise": noise.PersistentNoise(noise_definition, voices=2),
        },
        997,
        prepared_effects,
    )
    native_audio = np.empty_like(expected)
    for start in range(0, 48000, 997):
        end = min(start + 997, 48000)
        native.advance_into(
            {
                n: [a for a in source_actions if start <= a.tick < end]
                for n, source_actions in all_actions.items()
            },
            [],
            start,
            end,
            native_audio[start:end],
        )
        if end == 23928:
            native_snapshot = native.snapshot()
    restored_native = live.NativeLiveEngine(
        {
            "tone": synth.PersistentSynth(synth_definition, voices=2),
            "noise": noise.PersistentNoise(noise_definition, voices=2),
        },
        997,
        prepared_effects,
    )
    restored_native.restore(native_snapshot)
    native_replay = np.empty((48000 - 23928, 2))
    for start in range(23928, 48000, 997):
        end = min(start + 997, 48000)
        restored_native.advance_into(
            {"tone": [], "noise": []},
            [],
            start,
            end,
            native_replay[start - 23928 : end - 23928],
        )

    check_audio(tmp_path / "native-live-engine.wav", native_audio, expected)
    np.testing.assert_allclose(native_replay, native_audio[23928:], atol=0)
