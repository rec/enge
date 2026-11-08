from pathlib import Path

import numpy as np
import pytest
from test_persistent_effects import callback_audio
from test_synth import check_audio
from ufor import audio_effects

from enge import _native, effects, resonant


def resonant_graph(
    frequency: float = 440, cutoff: float = 6000, decay: float = 0.04
) -> audio_effects.EffectGraph:
    return effects.serial_graph(
        audio_effects.AttachmentScope.master,
        audio_effects.GraphInput(
            name="main", channels=["left", "right"], source=audio_effects.MainStream()
        ),
        [
            audio_effects.ResonantDelay(
                name="string",
                frequency_hz=frequency,
                damping_hz=cutoff,
                decay_seconds=decay,
            )
        ],
        48000,
    )


def replay_resonance(
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


@pytest.mark.parametrize(
    "frequency,cutoff,decay",
    [(55, 500, 0.04), (440, 6000, 20.0), (9973, 20000, 0.04), (12000, 6000, 0.04)],
)
def test_resonance_matches_passive_transfer_and_damping_compensated_tuning(
    tmp_path: Path, frequency: float, cutoff: float, decay: float
) -> None:
    frames, a = resonant.tuning(frequency, cutoff, 48000)
    assert 0.75 * 48000 / frequency - 2 <= frames < 48000 / frequency
    assert abs(a) <= np.sqrt(2) - 1 + 1e-12
    g = np.tan(np.pi * cutoff / 48000)
    low_a = (g - 1) / (g + 1)
    fundamental = np.exp(-2j * np.pi * frequency / 48000)
    phase_loop = (
        fundamental**frames
        * (a + fundamental)
        / (1 + a * fundamental)
        * (1 + low_a)
        / 2
        * (1 + fundamental)
        / (1 + low_a * fundamental)
    )
    assert abs(np.angle(phase_loop)) < 1e-12
    source = np.zeros((48000, 2))
    source[[0, 23923 - frames]] = [0.2, -0.1]
    size = 131072
    z = np.exp(-2j * np.pi * np.arange(size // 2 + 1) / size)
    read = z**frames * (a + z) / (1 + a * z)
    lowpass = (1 + low_a) / 2 * (1 + z) / (1 + low_a * z)
    loss = np.exp(-3 * np.log(10) / (frequency * decay))
    loop = loss * lowpass * read
    # Higher loop powers cannot arrive within this record. Truncation avoids
    # circular FFT contamination from the long-decay case's later response.
    transfer = read * (1 - loop ** (48000 // frames + 1)) / (1 - loop)
    expected = np.fft.irfft(
        np.fft.rfft(source, n=size, axis=0) * transfer[:, None], n=size, axis=0
    )[:48000]
    replay_resonance(
        tmp_path,
        resonant_graph(frequency, cutoff, decay),
        source,
        [audio_effects.InputEndAction(tick=24000, ordinal=0, input="main")],
        expected,
        decay < 1,
    )


def test_resonance_retains_active_decay_ramps_and_filter_memories(
    tmp_path: Path,
) -> None:
    graph = resonant_graph()
    frames = np.arange(48000)
    source = np.column_stack(
        [0.1 * np.sin(2 * np.pi * f * frames / 48000) for f in [220, 330]]
    )
    source[30000:] = 0
    actions: list[audio_effects.EffectAction] = [
        audio_effects.ParameterAction(
            tick=23880,
            ordinal=0,
            processor="string",
            parameter="decay_seconds",
            value=0.08,
            duration_frames=64,
        ),
        audio_effects.ParameterAction(
            tick=24000,
            ordinal=0,
            processor="string",
            parameter="mix",
            value=0.25,
            duration_frames=64,
        ),
        audio_effects.InputEndAction(tick=30000, ordinal=0, input="main"),
        audio_effects.BypassAction(
            tick=25000, ordinal=0, processor="string", bypassed=True
        ),
        audio_effects.BypassAction(
            tick=26000, ordinal=0, processor="string", bypassed=False
        ),
    ]
    actions.sort(key=lambda a: (a.tick, a.ordinal))
    expected = effects.OfflineEffects(effects.prepare(graph, 48000)).advance(
        {"main": source}, actions, 0, 48000
    )
    replay_resonance(tmp_path, graph, source, actions, expected, True)


@pytest.mark.parametrize("field", ["history", "write", "tuning", "damping"])
def test_resonant_snapshot_rejects_malformed_state_without_replacing_audio(
    tmp_path: Path, field: str
) -> None:
    prepared = effects.prepare(resonant_graph(), 48000)
    source = np.zeros((48000, 2))
    source[[0, 23800]] = [0.2, -0.1]
    expected = effects.OfflineEffects(prepared).advance({"main": source}, [], 0, 48000)
    for backend in ["numpy", "native"]:
        renderer = effects.OfflineEffects(prepared, backend)
        head = renderer.advance({"main": source[:23928]}, [], 0, 23928)
        saved = renderer.snapshot()
        broken = saved.model_copy(deep=True)
        state = broken.processors["string"].resonant_delay
        assert state is not None
        if field == "history":
            state.history[0].pop()
        elif field == "write":
            state.write = len(state.history)
        elif field == "tuning":
            state.tuning.memory.pop()
        else:
            state.damping.memory.pop()
        with pytest.raises(ValueError, match="resonant"):
            renderer.restore(broken)
        assert renderer.snapshot() == saved
        tail = renderer.advance({"main": source[23928:]}, [], 23928, 48000)
        check_audio(
            tmp_path / f"preserved-{backend}.wav", np.vstack([head, tail]), expected
        )


@pytest.mark.parametrize(
    "parameter,value,message",
    [
        (0, 220, "fixed"),
        (4, 3000, "fixed"),
        (3, 1e300, "Unrepresentable resonant decay"),
    ],
)
def test_invalid_resonant_actions_silence_the_callback_and_restore(
    tmp_path: Path, parameter: int, value: float, message: str
) -> None:
    prepared = effects.prepare(resonant_graph(), 48000)
    runtime = _native.LiveRuntime.empty(48000, 2, 48000, 64)
    runtime.set_effect_graph(
        *effects.native_live_graph(prepared), 2, input_routes=[[-2]]
    )
    saved = runtime.snapshot()
    runtime.submit_effects(
        np.array([[12003, 0, 0, parameter, value, 0]], dtype=np.float64)
    )
    source = np.zeros((48000, 2))
    source[0] = [0.2, -0.1]
    actual = np.ones_like(source)
    with pytest.raises(ValueError, match=message):
        runtime.process_into(actual, [source])
    assert runtime.failed()
    check_audio(tmp_path / "failed.wav", actual, np.zeros_like(source))
    runtime.restore(saved)
    runtime.process_into(actual, [source])
    expected = effects.OfflineEffects(prepared).advance({"main": source}, [], 0, 48000)
    check_audio(tmp_path / "recovered.wav", actual, expected)


def test_resonant_preparation_rejects_unsupported_or_unrepresentable_coefficients() -> (
    None
):
    for frequency, cutoff, decay in [
        (12001, 6000, 1),
        (440, 24000, 1),
        (440, 6000, 1e300),
        (1e-300, 6000, 1),
        (440, 1e-300, 1),
    ]:
        with pytest.raises(ValueError):
            effects.prepare(resonant_graph(frequency, cutoff, decay), 48000)


def test_zero_feedback_retains_the_tuning_filter_tail_after_history_empties(
    tmp_path: Path,
) -> None:
    graph = resonant_graph(decay=1e-300)
    prepared = effects.prepare(graph, 48000)
    frames, a = resonant.tuning(440, 6000, 48000)
    source = np.zeros((48000, 2))
    source[0] = [0.2, -0.1]
    size = 131072
    z = np.exp(-2j * np.pi * np.arange(size // 2 + 1) / size)
    expected = np.fft.irfft(
        np.fft.rfft(source, n=size, axis=0)
        * (z**frames * (a + z) / (1 + a * z))[:, None],
        n=size,
        axis=0,
    )[:48000]
    stop = frames + 1
    for origin, target in [("numpy", "native"), ("native", "numpy")]:
        renderer = effects.OfflineEffects(prepared, origin)
        head = renderer.advance(
            {"main": source[:stop]},
            [audio_effects.InputEndAction(tick=1, ordinal=0, input="main")],
            0,
            stop,
        )
        assert not renderer.drained
        saved = renderer.snapshot()
        state = saved.processors["string"].resonant_delay
        assert state is not None and not any(v for r in state.history for v in r)
        assert max(abs(v) for v in state.tuning.memory) > 0.01
        restored = effects.OfflineEffects(prepared, target)
        restored.restore(
            effects.EffectSnapshot.model_validate_json(saved.model_dump_json())
        )
        tail = restored.advance({"main": source[stop:]}, [], stop, 48000)
        assert restored.drained
        check_audio(
            tmp_path / f"filter-tail-{origin}-{target}.wav",
            np.vstack([head, tail]),
            expected,
        )


@pytest.mark.parametrize("frequency,cutoff", [(440.01, 6000), (440, 6001)])
def test_native_resonant_snapshot_preserves_fixed_tuning_identity(
    tmp_path: Path, frequency: float, cutoff: float
) -> None:
    prepared = effects.prepare(resonant_graph(), 48000)
    other = effects.prepare(resonant_graph(frequency, cutoff), 48000)
    runtime = _native.LiveRuntime.empty(48000, 2, 48000, 64)
    runtime.set_effect_graph(
        *effects.native_live_graph(prepared), 2, input_routes=[[-2]]
    )
    wrong = _native.LiveRuntime.empty(48000, 2, 48000, 64)
    wrong.set_effect_graph(*effects.native_live_graph(other), 2, input_routes=[[-2]])
    with pytest.raises(ValueError, match="different live effect graph"):
        runtime.restore(wrong.snapshot())
    source = np.zeros((48000, 2))
    source[0] = [0.2, -0.1]
    actual = np.empty_like(source)
    runtime.process_into(actual, [source])
    expected = effects.OfflineEffects(prepared).advance({"main": source}, [], 0, 48000)
    check_audio(tmp_path / "preserved-tuning.wav", actual, expected)


@pytest.mark.parametrize(
    "parameter,value", [(0, 12001), (4, 24000), (3, 1e300), (5, 1e-6)]
)
def test_native_resonant_preparation_enforces_the_profile(
    parameter: int, value: float
) -> None:
    prepared = effects.prepare(resonant_graph(), 48000)
    plan = effects.native_live_graph(prepared)
    plan[2][0, parameter] = value
    runtime = _native.LiveRuntime.empty(48000, 2, 48000, 64)
    with pytest.raises(ValueError):
        runtime.set_effect_graph(*plan, 2, input_routes=[[-2]])


def test_nonfinite_resonant_state_silences_the_block_and_recovers(
    tmp_path: Path,
) -> None:
    prepared = effects.prepare(resonant_graph(), 48000)
    runtime = _native.LiveRuntime.empty(48000, 2, 48000, 64)
    runtime.set_effect_graph(
        *effects.native_live_graph(prepared), 2, input_routes=[[-2]]
    )
    saved = runtime.snapshot()
    actual = np.ones((48000, 2))
    with pytest.raises(ValueError, match="Non-finite"):
        runtime.process_into(actual, [np.full_like(actual, 1e308)])
    assert runtime.failed()
    check_audio(tmp_path / "overflow.wav", actual, np.zeros_like(actual))
    runtime.restore(saved)
    source = np.full_like(actual, 0.01)
    runtime.process_into(actual, [source])
    expected = effects.OfflineEffects(prepared).advance({"main": source}, [], 0, 48000)
    check_audio(tmp_path / "recovered-overflow.wav", actual, expected)
