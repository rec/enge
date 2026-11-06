from math import sqrt
from pathlib import Path
from typing import Literal

import numpy as np
import pytest
from test_persistent_effects import callback_audio, graph_audio
from test_synth import check_audio
from ufor import audio_effects

from enge import _native, effects


def reverb_graph(
    channels: int = 2, mix: float = 1.0, predelay: float = 0.0
) -> audio_effects.EffectGraph:
    return effects.serial_graph(
        audio_effects.AttachmentScope.master,
        audio_effects.GraphInput(
            name="main",
            channels=["mono"] if channels == 1 else ["left", "right"],
            source=audio_effects.MainStream(),
        ),
        [
            audio_effects.Reverb(
                name="room",
                predelay_seconds=predelay,
                maximum_predelay_seconds=0.003,
                decay_seconds=0.08,
                damping_hz=6000,
                mix=mix,
                retime_fade_frames=8,
                tail_threshold=1e-9,
            )
        ],
        48000,
    )


@pytest.mark.parametrize("channels", [1, 2])
@pytest.mark.parametrize("predelay_frames", [0.0, 0.5])
def test_reverb_impulse_obeys_injection_and_output_matrices(
    tmp_path: Path,
    channels: int,
    predelay_frames: float,
) -> None:
    prepared = effects.prepare(
        reverb_graph(channels, predelay=predelay_frames / 48000), 48000
    )
    source = np.zeros((48000, channels))
    source[0, 0] = 1
    actions: list[audio_effects.EffectAction] = [
        audio_effects.InputEndAction(tick=1, ordinal=0, input="main")
    ]
    reference = effects.OfflineEffects(prepared)
    expected = reference.advance({"main": source}, actions, 0, 48000)
    first = audio_effects.reverb_delay_frames(48000)[0]
    assert not np.any(expected[:first])
    amplitude = 1 / (4 * sqrt(2)) if channels == 1 else 1 / 8
    np.testing.assert_allclose(
        expected[first], amplitude * (1 - predelay_frames), rtol=1e-14
    )
    np.testing.assert_allclose(
        expected[first + 1], amplitude * predelay_frames, rtol=1e-14
    )
    assert reference.drained
    native = effects.OfflineEffects(prepared, "native")
    actual = native.advance({"main": source}, actions, 0, 48000)
    assert native.drained
    callback, replay, drained = callback_audio(prepared, {"main": source}, actions)
    assert drained
    check_audio(tmp_path / "network-offline.wav", actual, expected)
    check_audio(tmp_path / "network-callback.wav", callback, expected)
    check_audio(tmp_path / "network-replay.wav", replay, callback)


@pytest.mark.parametrize("channels", [1, 2])
def test_reverb_zero_mix_preserves_the_declared_dry_mapping(
    tmp_path: Path, channels: int
) -> None:
    prepared = effects.prepare(reverb_graph(channels, mix=0), 48000)
    inputs = graph_audio(prepared.definition)
    expected = (
        inputs["main"]
        if channels == 2
        else np.repeat(inputs["main"] / sqrt(2), 2, axis=1)
    )
    for backend in ("numpy", "native"):
        actual = effects.OfflineEffects(prepared, backend).advance(inputs, [], 0, 48000)
        check_audio(tmp_path / f"dry-{backend}.wav", actual, expected)


@pytest.mark.parametrize(
    "origin,target", [("numpy", "native"), ("native", "numpy"), ("native", "native")]
)
def test_reverb_retimes_and_replays_portable_state(
    tmp_path: Path,
    origin: Literal["numpy", "native"],
    target: Literal["numpy", "native"],
) -> None:
    prepared = effects.prepare(reverb_graph(), 48000)
    inputs = graph_audio(prepared.definition)
    actions: list[audio_effects.EffectAction] = [
        audio_effects.ParameterAction(
            tick=t,
            ordinal=0,
            processor="room",
            parameter="predelay_seconds",
            value=(t % 113) / 48000,
        )
        for t in range(23920, 23940)
    ]
    actions.extend(
        [
            audio_effects.BypassAction(
                tick=15000, ordinal=0, processor="room", bypassed=True
            ),
            audio_effects.BypassAction(
                tick=17000, ordinal=0, processor="room", bypassed=False
            ),
            audio_effects.ParameterAction(
                tick=20003,
                ordinal=0,
                processor="room",
                parameter="damping_hz",
                value=1200,
                duration_frames=5000,
            ),
            audio_effects.ParameterAction(
                tick=30007,
                ordinal=0,
                processor="room",
                parameter="decay_seconds",
                value=0.2,
                duration_frames=2000,
            ),
            audio_effects.InputEndAction(tick=36000, ordinal=0, input="main"),
            audio_effects.StopAction(tick=45000, ordinal=0),
        ]
    )
    actions.sort(key=lambda a: (a.tick, a.ordinal))
    expected = effects.OfflineEffects(prepared, origin).advance(
        inputs, actions, 0, 48000
    )
    first = effects.OfflineEffects(prepared, origin)
    head = first.advance(
        {n: a[:23928] for n, a in inputs.items()},
        [a for a in actions if a.tick < 23928],
        0,
        23928,
    )
    saved = effects.EffectSnapshot.model_validate_json(
        first.snapshot().model_dump_json()
    )
    second = effects.OfflineEffects(prepared, target)
    if origin == target == "native":
        before = second.snapshot()
        invalid = saved.model_copy(deep=True)
        network = invalid.processors["room"].reverb
        assert network is not None
        network.lines[0].pop()
        with pytest.raises(ValueError, match="delay-line snapshot"):
            second.restore(invalid)
        assert second.snapshot() == before
    second.restore(saved)
    tail = second.advance(
        {n: a[23928:] for n, a in inputs.items()},
        [a for a in actions if a.tick >= 23928],
        23928,
        48000,
    )
    actual = np.concatenate((head, tail))
    assert not np.any(actual[45000:])
    check_audio(tmp_path / "network-portable.wav", actual, expected)
    if origin == target:
        np.testing.assert_array_equal(actual, expected)
    callback, replay, drained = callback_audio(prepared, inputs, actions)
    assert drained
    check_audio(tmp_path / "retime-callback.wav", callback, expected)
    check_audio(tmp_path / "retime-replay.wav", replay, callback)


def test_reverb_shared_envelfo_modulates_the_network(tmp_path: Path) -> None:
    raw = reverb_graph(predelay=0.001).model_dump(mode="json")
    raw.update(
        envelfos={"motion": {"kind": "lfo", "shape": "sine", "rate": 3}},
        bindings=[{"kind": "envelfo", "name": "motion", "reference": "motion"}],
        modulation={
            "sources": [
                {"name": "motion", "scope": "voice", "minimum": -1, "maximum": 1}
            ],
            "parameters": [
                {
                    "target": {"name": "room", "parameter": "predelay_seconds"},
                    "unit": "seconds",
                    "scope": "voice",
                    "minimum": 0,
                    "maximum": 0.003,
                    "default": 0.001,
                }
            ],
            "routes": [
                {
                    "name": "motion",
                    "source": "motion",
                    "target": {"name": "room", "parameter": "predelay_seconds"},
                    "unit": "seconds",
                    "operation": "add",
                    "points": [
                        {"input": -1, "amount": -0.0005},
                        {"input": 1, "amount": 0.0005},
                    ],
                }
            ],
        },
    )
    prepared = effects.prepare(audio_effects.EffectGraph.model_validate(raw), 48000)
    inputs = graph_audio(prepared.definition)
    expected = effects.OfflineEffects(prepared).advance(inputs, [], 0, 48000)
    actual = effects.OfflineEffects(prepared, "native").advance(inputs, [], 0, 48000)
    callback, replay, _ = callback_audio(prepared, inputs, [])
    check_audio(tmp_path / "modulated-offline.wav", actual, expected)
    check_audio(tmp_path / "modulated-callback.wav", callback, expected)
    check_audio(tmp_path / "modulated-replay.wav", replay, callback)


@pytest.mark.parametrize("damping", [0.0, 1e-300])
def test_reverb_callback_coefficient_failure_recovers_without_output(
    tmp_path: Path,
    damping: float,
) -> None:
    prepared = effects.prepare(reverb_graph(), 48000)
    inputs = graph_audio(prepared.definition)
    runtime = _native.LiveRuntime.empty(48000, 2, 48000, 64)
    runtime.set_effect_graph(
        *effects.native_live_graph(prepared), 2, input_routes=[[-2]]
    )
    saved = runtime.snapshot()
    action = audio_effects.ParameterAction(
        tick=100, ordinal=0, processor="room", parameter="damping_hz", value=3000
    )
    encoded = effects.native_live_actions(prepared, [action], 0, 48000)
    encoded[0, 4] = damping
    runtime.submit_effects(encoded)
    actual = np.ones((48000, 2))
    with pytest.raises(ValueError, match="reverb"):
        runtime.process_into(actual, [inputs["main"]])
    check_audio(tmp_path / "coefficient-failure.wav", actual, np.zeros_like(actual))
    runtime.restore(saved)
    assert not runtime.failed()
    runtime.process_into(actual, [inputs["main"]])
    expected = effects.OfflineEffects(prepared).advance(inputs, [], 0, 48000)
    check_audio(tmp_path / "coefficient-recovery.wav", actual, expected)
