from math import copysign, floor
from pathlib import Path
from typing import Literal

import numpy as np
import pytest
from test_effects import graph_input
from test_persistent_effects import callback_audio
from test_synth import check_audio
from ufor import audio_effects

from enge import _native, effects


def crusher_graph(bits: int = 8, hold: int = 1) -> audio_effects.EffectGraph:
    return effects.serial_graph(
        audio_effects.AttachmentScope.master,
        graph_input(),
        [audio_effects.Bitcrusher(name="crush", bits=bits, hold_frames=hold)],
        96000,
    )


@pytest.mark.parametrize("bits,hold", [(1, 1), (2, 17), (8, 257), (24, 65536)])
def test_signed_quantization_and_final_hold_replay_across_backends(
    tmp_path: Path, bits: int, hold: int
) -> None:
    frames = 96000 if hold == 65536 else 48000
    time = np.arange(frames) / 48000
    source = np.column_stack(
        [1.2 * np.sin(2 * np.pi * 997 * time), 0.7 * np.cos(2 * np.pi * 631 * time)]
    )
    end = 23003
    source[end:] = 0
    expected = np.zeros_like(source)
    step = 2.0 ** (1 - bits)
    for start in range(0, end, hold):
        values = [
            min(
                1 - step,
                max(
                    -1.0,
                    copysign(floor(abs(min(1.0, max(-1.0, v))) / step + 0.5) * step, v),
                ),
            )
            for v in source[start]
        ]
        expected[start : min(start + hold, frames)] = values
    prepared = effects.prepare(crusher_graph(bits, hold), 48000)
    assert prepared.latency_frames == 0
    actions: list[audio_effects.EffectAction] = [
        audio_effects.InputEndAction(tick=end, ordinal=0, input="main")
    ]
    drained = ((end + hold - 1) // hold) * hold < frames
    pairs: list[tuple[Literal["numpy", "native"], Literal["numpy", "native"]]] = [
        ("numpy", "native"),
        ("native", "numpy"),
    ]
    for origin, target in pairs:
        renderer = effects.OfflineEffects(prepared, origin)
        first = renderer.advance({"main": source[:23928]}, actions, 0, 23928)
        saved = effects.EffectSnapshot.model_validate_json(
            renderer.snapshot().model_dump_json()
        )
        restored = effects.OfflineEffects(prepared, target)
        restored.restore(saved)
        last = restored.advance({"main": source[23928:]}, [], 23928, frames)
        assert restored.drained == drained
        check_audio(
            tmp_path / f"{origin}-{target}.wav", np.vstack([first, last]), expected
        )
    actual, replay, ended = callback_audio(prepared, {"main": source}, actions)
    assert ended == drained
    check_audio(tmp_path / "callback.wav", actual, expected)
    check_audio(tmp_path / "replay.wav", replay, actual)


def test_quantization_ties_and_signed_endpoints_are_explicit(tmp_path: Path) -> None:
    source = np.tile(np.array([-1.25, -1, -0.75, -0.25, 0, 0.25, 0.75, 1, 1.25]), 5334)[
        :48000, None
    ]
    source = np.repeat(source, 2, axis=1)
    expected = np.repeat(
        np.tile(np.array([-1, -1, -1, -0.5, 0, 0.5, 0.5, 0.5, 0.5]), 5334)[
            :48000, None
        ],
        2,
        axis=1,
    )
    prepared = effects.prepare(crusher_graph(2), 48000)
    for backend in BACKENDS:
        check_audio(
            tmp_path / f"{backend}.wav",
            effects.process_audio(
                prepared, {"main": source}, backend=backend, block_frames=17
            ),
            expected,
        )


def test_integer_changes_latch_at_capture_and_final_hold_drains_on_its_boundary(
    tmp_path: Path,
) -> None:
    prepared = effects.prepare(crusher_graph(2, 17), 48000)
    source = np.full((48000, 2), 0.3)
    actions: list[audio_effects.EffectAction] = [
        audio_effects.ParameterAction(
            tick=3, ordinal=0, processor="crush", parameter="bits", value=3
        ),
        audio_effects.ParameterAction(
            tick=5, ordinal=0, processor="crush", parameter="hold_frames", value=7
        ),
        audio_effects.InputEndAction(tick=23, ordinal=0, input="main"),
    ]
    expected = np.zeros_like(source)
    expected[:17] = 0.5
    expected[17:24] = 0.25
    for backend in BACKENDS:
        renderer = effects.OfflineEffects(prepared, backend)
        actual = renderer.advance({"main": source}, actions, 0, 48000)
        assert renderer.stopped_at == 24
        check_audio(tmp_path / f"{backend}.wav", actual, expected)
    callback, replay, drained = callback_audio(prepared, {"main": source}, actions)
    assert drained
    check_audio(tmp_path / "callback.wav", callback, expected)
    check_audio(tmp_path / "replay.wav", replay, callback)


@pytest.mark.parametrize(
    "parameter,value,duration", [(0, 1.5, 0), (3, 65537, 0), (0, 4, 17), (3, 2, 17)]
)
def test_invalid_native_crusher_controls_silence_and_restore(
    tmp_path: Path, parameter: int, value: float, duration: int
) -> None:
    prepared = effects.prepare(crusher_graph(), 48000)
    runtime = _native.LiveRuntime.empty(48000, 2, 48000, 64)
    runtime.set_effect_graph(
        *effects.native_live_graph(prepared), 2, input_routes=[[-2]]
    )
    saved = runtime.snapshot()
    runtime.submit_effects(
        np.array([[12003, 0, 0, parameter, value, duration]], dtype=np.float64)
    )
    source = np.full((48000, 2), 0.3)
    actual = np.ones_like(source)
    with pytest.raises(ValueError, match="[Bb]itcrusher"):
        runtime.process_into(actual, [source])
    assert runtime.failed()
    check_audio(tmp_path / "failed.wav", actual, np.zeros_like(actual))
    runtime.restore(saved)
    runtime.process_into(actual, [source])
    check_audio(tmp_path / "restored.wav", actual, np.full_like(actual, 38 / 128))


def test_portable_crusher_state_is_validated_before_restore() -> None:
    prepared = effects.prepare(crusher_graph(), 48000)
    renderer = effects.OfflineEffects(prepared, "native")
    original = renderer.snapshot()
    invalid = original.model_copy(deep=True)
    state = invalid.processors["crush"].bitcrusher
    assert state is not None
    state.held.pop()
    with pytest.raises(ValueError, match="snapshot"):
        renderer.restore(invalid)
    assert renderer.snapshot() == original


def test_warm_bypass_preserves_the_capture_clock(tmp_path: Path) -> None:
    prepared = effects.prepare(crusher_graph(2, 17), 48000)
    frames = np.arange(48000)
    source = np.repeat((0.3 * (1 - 2 * ((frames // 5) % 2)))[:, None], 2, axis=1)
    wet = np.sign(source[(frames // 17) * 17]) * 0.5
    gain = (
        1 - np.clip((frames - 10003) / 64, 0, 1) + np.clip((frames - 20003) / 64, 0, 1)
    )
    expected = (1 - gain[:, None]) * source + gain[:, None] * wet
    actions: list[audio_effects.EffectAction] = [
        audio_effects.BypassAction(
            tick=10003, ordinal=0, processor="crush", bypassed=True
        ),
        audio_effects.BypassAction(
            tick=20003, ordinal=0, processor="crush", bypassed=False
        ),
    ]
    for backend in BACKENDS:
        check_audio(
            tmp_path / f"{backend}.wav",
            effects.process_audio(prepared, {"main": source}, actions, backend, 997),
            expected,
        )
    actual, replay, _ = callback_audio(prepared, {"main": source}, actions)
    check_audio(tmp_path / "callback.wav", actual, expected)
    check_audio(tmp_path / "replay.wav", replay, actual)


def test_crusher_integer_controls_follow_upstream_audio_latency(tmp_path: Path) -> None:
    graph = effects.serial_graph(
        audio_effects.AttachmentScope.master,
        graph_input(),
        [
            audio_effects.Waveshaper(name="latency", mix=0),
            audio_effects.Bitcrusher(name="crush"),
        ],
        48000,
    )
    prepared = effects.prepare(graph, 48000)
    source = np.full((48000, 2), 0.3)
    source[23003:] = 0
    expected = np.zeros_like(source)
    expected[48:12048] = 38 / 128
    expected[12048:23051] = 0.5
    actions: list[audio_effects.EffectAction] = [
        audio_effects.ParameterAction(
            tick=12000, ordinal=0, processor="crush", parameter="bits", value=2
        ),
        audio_effects.InputEndAction(tick=23003, ordinal=0, input="main"),
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


@pytest.mark.parametrize("node,parameter", [(0, 5), (1, 4)])
def test_native_control_targets_reject_fixed_or_unused_fields(
    node: int, parameter: int
) -> None:
    graph = effects.serial_graph(
        audio_effects.AttachmentScope.master,
        graph_input(),
        [
            audio_effects.Waveshaper(name="shape"),
            audio_effects.Bitcrusher(name="crush"),
        ],
        997,
    )
    plan = effects.native_live_graph(effects.prepare(graph, 48000))
    runtime = _native.LiveRuntime.empty(48000, 2, 997, 64)
    with pytest.raises(ValueError, match="control target"):
        runtime.set_effect_graph(
            *plan[:12],
            ([(node, parameter, 0.0, 2.0)], []),
            plan[13],
            2,
            input_routes=[[-2]],
        )


BACKENDS: list[Literal["numpy", "native"]] = ["numpy", "native"]
