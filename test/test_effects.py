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


def gain_graph(gain_db: float = -6) -> audio_effects.EffectGraph:
    return effects.serial_graph(
        audio_effects.AttachmentScope.voice,
        graph_input(),
        [audio_effects.Gain(name="trim", gain_db=gain_db)],
        4096,
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
        maximum_block_frames=4096,
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
        4096,
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


def test_serial_shorthand_rejects_multi_input_processor() -> None:
    with pytest.raises(effects.EngineError, match="one-input"):
        effects.serial_graph(
            audio_effects.AttachmentScope.voice,
            graph_input(),
            [audio_effects.Multiply(name="ring")],
            256,
        )
