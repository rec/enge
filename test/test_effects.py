from pathlib import Path

import numpy as np
import pytest
from test_synth import check_audio
from ufor import audio_effects
from ufor.samples import processing

from enge import effects


def graph_input(
    name: str = "main", source: audio_effects.StreamSource | None = None
) -> audio_effects.GraphInput:
    return audio_effects.GraphInput(
        name=name,
        channels=["left", "right"],
        source=source or audio_effects.MainStream(),
    )


def gain_graph(
    gain_db: float = -6, maximum_block_frames: int = 48000
) -> audio_effects.EffectGraph:
    return effects.serial_graph(
        audio_effects.AttachmentScope.voice,
        graph_input(),
        [audio_effects.Gain(name="trim", gain_db=gain_db)],
        maximum_block_frames,
    )


def test_gain_automation_is_partition_independent(tmp_path: Path) -> None:
    prepared = effects.prepare(gain_graph(), 48000)
    frames = np.arange(48000)
    source = np.column_stack(
        [np.sin(2 * np.pi * frames / 97), np.cos(2 * np.pi * frames / 131)]
    )
    action = audio_effects.ParameterAction(
        tick=24000,
        ordinal=0,
        processor="trim",
        parameter="gain_db",
        value=0,
        duration_frames=64,
    )
    whole = effects.OfflineEffects(prepared).advance(
        {"main": source}, [action], 0, 48000
    )
    split = effects.OfflineEffects(prepared)
    chunks = [
        split.advance({"main": source[:24000]}, [], 0, 24000),
        split.advance({"main": source[24000:24031]}, [action], 24000, 24031),
        split.advance({"main": source[24031:]}, [], 24031, 48000),
    ]
    actual = np.concatenate(chunks)

    check_audio(tmp_path / "gain-automation.wav", actual, whole)


def test_multiply_uses_two_named_inputs(tmp_path: Path) -> None:
    graph = audio_effects.EffectGraph(
        scope="host",
        owner="ring-input",
        inputs=[
            graph_input(),
            graph_input("modulator", audio_effects.HostStream(stream="modulator")),
        ],
        processors=[audio_effects.Multiply(name="ring")],
        connections=[
            audio_effects.Connection(
                processor="ring",
                port="carrier",
                source=audio_effects.InputSource(input="main"),
            ),
            audio_effects.Connection(
                processor="ring",
                port="modulator",
                source=audio_effects.InputSource(input="modulator"),
            ),
        ],
        output=audio_effects.ProcessorSource(processor="ring"),
        maximum_block_frames=48000,
    )
    renderer = effects.OfflineEffects(effects.prepare(graph, 48000))
    frames = np.arange(48000)
    carrier = np.column_stack(
        [np.sin(2 * np.pi * frames / 101), np.sin(2 * np.pi * frames / 151)]
    )
    modulator = np.column_stack(
        [np.sin(2 * np.pi * frames / 211), np.sin(2 * np.pi * frames / 251)]
    )
    actual = renderer.advance({"main": carrier, "modulator": modulator}, [], 0, 48000)

    check_audio(tmp_path / "ring-modulator.wav", actual, carrier * modulator)


def test_filter_tail_drains_after_input_end(tmp_path: Path) -> None:
    definition = processing.ResonantFilter(
        name="low", response="lowpass", cutoff_hz=2000
    )
    graph = effects.serial_graph(
        audio_effects.AttachmentScope.voice,
        graph_input(),
        [
            audio_effects.Filter(
                name="tone",
                filters=[definition],
            )
        ],
        48000,
    )
    renderer = effects.OfflineEffects(effects.prepare(graph, 48000))
    source = np.zeros((48000, 2))
    source[0] = 1
    actual = renderer.advance(
        {"main": source},
        [audio_effects.InputEndAction(tick=1, ordinal=0, input="main")],
        0,
        48000,
    )
    expected, _ = effects.filters.filter_samples(
        [definition],
        effects.filters.initial_states([definition], 2),
        source,
        effects.filters.parameters([definition], 48000, 48000),
        48000,
    )

    check_audio(tmp_path / "filter-tail.wav", actual, expected)
    assert renderer.drained
    assert np.any(actual[1:])
    assert renderer.stopped_at is not None
    assert not np.any(actual[renderer.stopped_at :])


def test_bypass_retains_authored_mix() -> None:
    renderer = effects.OfflineEffects(effects.prepare(gain_graph(-20), 48000))
    source = np.ones((256, 2))
    actions: list[audio_effects.EffectAction] = [
        audio_effects.BypassAction(tick=32, ordinal=0, processor="trim", bypassed=True),
        audio_effects.ParameterAction(
            tick=64,
            ordinal=0,
            processor="trim",
            parameter="mix",
            value=0.5,
        ),
        audio_effects.BypassAction(
            tick=128, ordinal=0, processor="trim", bypassed=False
        ),
    ]
    actual = renderer.advance({"main": source}, actions, 0, 256)

    assert actual[31, 0] == pytest.approx(0.1)
    assert actual[96, 0] == pytest.approx(1)
    assert actual[192, 0] == pytest.approx(0.55)


def test_snapshot_restores_graph_state() -> None:
    prepared = effects.prepare(gain_graph(), 48000)
    source = np.ones((256, 2))
    first = effects.OfflineEffects(prepared)
    first.advance(
        {"main": source[:64]},
        [
            audio_effects.ParameterAction(
                tick=0,
                ordinal=0,
                processor="trim",
                parameter="gain_db",
                value=0,
                duration_frames=128,
            )
        ],
        0,
        64,
    )
    snapshot = first.snapshot()
    expected = first.advance({"main": source[64:]}, [], 64, 256)
    restored = effects.OfflineEffects(prepared)
    restored.restore(snapshot)

    np.testing.assert_array_equal(
        restored.advance({"main": source[64:]}, [], 64, 256), expected
    )


class ConstantSource:
    def advance(self, actions: list[object], start: int, end: int) -> np.ndarray:
        assert not actions
        return np.full((end - start, 2), 0.25)


def test_offline_attachment_processes_a_source_engine() -> None:
    attachment = effects.OfflineEffectAttachment(
        ConstantSource(), effects.OfflineEffects(effects.prepare(gain_graph(6), 48000))
    )

    actual = attachment.advance([], [], 0, 64)

    np.testing.assert_allclose(actual, 0.25 * 10 ** (6 / 20))


def test_process_audio_uses_prepared_block_limit() -> None:
    prepared = effects.prepare(gain_graph(6, 4096), 48000)
    source = np.full((8192, 2), 0.25)

    actual = effects.process_audio(
        prepared, {"main": source}, backend="native", block_frames=997
    )

    np.testing.assert_allclose(actual, 0.25 * 10 ** (6 / 20))
    with pytest.raises(effects.EngineError, match="prepared maximum"):
        effects.process_audio(prepared, {"main": source}, block_frames=4097)


def test_effect_advance_enforces_prepared_blocks_and_allows_noop() -> None:
    renderer = effects.OfflineEffects(
        effects.prepare(gain_graph(maximum_block_frames=4096), 48000)
    )

    assert renderer.advance({"main": np.empty((0, 2))}, [], 0, 0).shape == (0, 2)
    with pytest.raises(effects.EngineError, match="prepared maximum"):
        renderer.advance({"main": np.zeros((4097, 2))}, [], 0, 4097)


def test_serial_shorthand_rejects_multi_input_processor() -> None:
    with pytest.raises(effects.EngineError, match="one-input"):
        effects.serial_graph(
            audio_effects.AttachmentScope.voice,
            graph_input(),
            [audio_effects.Multiply(name="ring")],
            256,
        )


def test_native_graph_matches_numpy(tmp_path: Path) -> None:
    definition = processing.ResonantFilter(
        name="low", response="lowpass", cutoff_hz=3000, q=0.8
    )
    graph = effects.serial_graph(
        audio_effects.AttachmentScope.instrument,
        graph_input(),
        [
            audio_effects.Filter(name="tone", filters=[definition], mix=0.75),
            audio_effects.Gain(name="trim", gain_db=-3),
        ],
        48000,
        "organ",
    )
    frames = np.arange(48000)
    source = np.column_stack(
        [np.sin(2 * np.pi * frames / 71), np.cos(2 * np.pi * frames / 113)]
    )
    actions: list[audio_effects.EffectAction] = [
        audio_effects.ParameterAction(
            tick=12000,
            ordinal=0,
            processor="tone",
            parameter="low-cutoff_hz",
            value=800,
            duration_frames=16000,
        ),
        audio_effects.BypassAction(
            tick=32000, ordinal=0, processor="trim", bypassed=True
        ),
    ]
    prepared = effects.prepare(graph, 48000)
    expected = effects.OfflineEffects(prepared).advance(
        {"main": source}, actions, 0, 48000
    )
    actual = effects.OfflineEffects(prepared, "native").advance(
        {"main": source}, actions, 0, 48000
    )

    check_audio(tmp_path / "native-effects.wav", actual, expected)


def test_granulator_is_partition_independent_and_snapshots(tmp_path: Path) -> None:
    graph = effects.serial_graph(
        audio_effects.AttachmentScope.master,
        graph_input(),
        [
            audio_effects.Granulator(
                name="cloud",
                duration_seconds=0.02,
                density_hz=120,
                lookback_seconds=0.03,
                playback_ratio=1.5,
                position_jitter_seconds=0.005,
                history_seconds=0.08,
                maximum_grains=2,
            )
        ],
        48000,
    )
    prepared = effects.prepare(graph, 48000)
    frames = np.arange(48000)
    source = np.column_stack(
        [np.sin(2 * np.pi * frames / 97), np.cos(2 * np.pi * frames / 127)]
    )
    actions: list[audio_effects.EffectAction] = [
        audio_effects.FreezeAction(
            tick=18000, ordinal=0, processor="cloud", frozen=True
        ),
        audio_effects.ParameterAction(
            tick=24000,
            ordinal=0,
            processor="cloud",
            parameter="playback_ratio",
            value=0.5,
        ),
        audio_effects.FreezeAction(
            tick=30000, ordinal=0, processor="cloud", frozen=False
        ),
    ]
    whole = effects.OfflineEffects(prepared).advance(
        {"main": source}, actions, 0, 48000
    )
    split = effects.OfflineEffects(prepared)
    first = split.advance({"main": source[:24000]}, actions[:1], 0, 24000)
    snapshot = split.snapshot()
    second = split.advance({"main": source[24000:]}, actions[1:], 24000, 48000)
    restored = effects.OfflineEffects(prepared)
    restored.restore(snapshot)

    np.testing.assert_array_equal(
        restored.advance({"main": source[24000:]}, actions[1:], 24000, 48000),
        second,
    )
    check_audio(tmp_path / "granulator.wav", np.concatenate([first, second]), whole)
    assert len(split.processors["cloud"].granulator.grains) <= 2


def test_native_granulator_matches_reference_across_blocks(tmp_path: Path) -> None:
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
            )
        ],
        48000,
    )
    prepared = effects.prepare(graph, 48000)
    frames = np.arange(48000)
    source = np.column_stack(
        [np.sin(2 * np.pi * frames / 83), np.cos(2 * np.pi * frames / 139)]
    )
    action = audio_effects.ParameterAction(
        tick=19000,
        ordinal=0,
        processor="cloud",
        parameter="density_hz",
        value=140,
        duration_frames=7000,
    )
    expected = effects.OfflineEffects(prepared).advance(
        {"main": source}, [action], 0, 48000
    )
    native = effects.OfflineEffects(prepared, "native")
    parts = [
        native.advance({"main": source[:12345]}, [], 0, 12345),
        native.advance({"main": source[12345:27111]}, [action], 12345, 27111),
    ]
    snapshot = native.snapshot()
    parts.append(native.advance({"main": source[27111:]}, [], 27111, 48000))
    restored = effects.OfflineEffects(prepared, "native")
    restored.restore(snapshot)

    np.testing.assert_allclose(
        restored.advance({"main": source[27111:]}, [], 27111, 48000),
        parts[-1],
        atol=1e-12,
    )
    check_audio(tmp_path / "native-granulator.wav", np.concatenate(parts), expected)
