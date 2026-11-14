from pathlib import Path
from typing import Literal

import numpy as np
import pytest
from test_effects import graph_input
from test_persistent_effects import callback_audio
from test_synth import check_audio
from ufor import audio_effects

from enge import _native, effects


def limiter_graph(
    release: float = 0.05, lookahead: float = 0.005
) -> audio_effects.EffectGraph:
    return effects.serial_graph(
        audio_effects.AttachmentScope.master,
        graph_input(),
        [
            audio_effects.Limiter(
                name="limit", release_seconds=release, lookahead_seconds=lookahead
            )
        ],
        48000,
    )


@pytest.mark.parametrize("release", [0.0, 0.05])
def test_linked_limiter_matches_analytical_pulse_recovery_and_replays_lookahead(
    tmp_path: Path, release: float
) -> None:
    frames = np.arange(48000)
    source = np.tile([0.25, 0.125], (48000, 1))
    pulses = [10003, 23903, 31007]
    source[pulses] = [[4, 1], [1, 4], [-2, -4]]
    source[36003:] = 0
    latency = 240
    ceiling = 10 ** (-1 / 20)
    deficit = np.zeros(48000)
    for start in pulses:
        hold = np.maximum(frames - start - latency, 0)
        decay = np.exp(-hold / (48000 * release)) if release else (hold == 0)
        deficit = np.maximum(
            deficit, np.where(frames >= start, (1 - ceiling / 4) * decay, 0)
        )
    dry = np.vstack([np.zeros((latency, 2)), source[:-latency]])
    expected = np.clip(dry * (1 - deficit[:, None]), -ceiling, ceiling)
    prepared = effects.prepare(limiter_graph(release), 48000)
    assert prepared.latency_frames == latency
    actions: list[audio_effects.EffectAction] = [
        audio_effects.InputEndAction(tick=36003, ordinal=0, input="main")
    ]
    pairs: list[tuple[Literal["numpy", "native"], Literal["numpy", "native"]]] = [
        ("numpy", "native"),
        ("native", "numpy"),
    ]
    for origin, target in pairs:
        first = effects.OfflineEffects(prepared, origin)
        head = first.advance({"main": source[:23928]}, [], 0, 23928)
        snapshot = effects.EffectSnapshot.model_validate_json(
            first.snapshot().model_dump_json()
        )
        second = effects.OfflineEffects(prepared, target)
        second.restore(snapshot)
        tail = second.advance({"main": source[23928:]}, actions, 23928, 48000)
        actual = np.vstack([head, tail])
        assert np.max(np.abs(actual)) <= ceiling
        assert second.drained
        assert second.stopped_at == 36003 + latency
        check_audio(tmp_path / f"{origin}-{target}.wav", actual, expected)
    actual, replay, drained = callback_audio(prepared, {"main": source}, actions)
    assert drained
    check_audio(tmp_path / "callback.wav", actual, expected)
    check_audio(tmp_path / "replay.wav", replay, actual)


def test_limiter_ceiling_ramp_aligns_with_its_own_and_upstream_audio_latency(
    tmp_path: Path,
) -> None:
    graph = effects.serial_graph(
        audio_effects.AttachmentScope.master,
        graph_input(),
        [
            audio_effects.Waveshaper(name="upstream", mix=0),
            audio_effects.Limiter(name="limit", release_seconds=0),
        ],
        48000,
    )
    prepared = effects.prepare(graph, 48000)
    assert prepared.latency_frames == 288
    source = np.tile([2.0, 1.0], (48000, 1))
    source[24003:] = 0
    frames = np.arange(48000)
    ceiling = 10 ** ((-1 - 11 * np.clip((frames - 12288) / 1000, 0, 1)) / 20)
    expected = ceiling[:, None] * np.array([1, 0.5])
    expected[:288] = expected[24291:] = 0
    actions: list[audio_effects.EffectAction] = [
        audio_effects.ParameterAction(
            tick=12000,
            ordinal=0,
            processor="limit",
            parameter="ceiling_db",
            value=-12,
            duration_frames=1000,
        ),
        audio_effects.InputEndAction(tick=24003, ordinal=0, input="main"),
    ]
    for backend in BACKENDS:
        actual = effects.process_audio(
            prepared, {"main": source}, actions, backend, 997
        )
        # Independent vector/scalar ramp arithmetic can differ by a few ulps.
        assert np.all(
            np.abs(actual) <= ceiling[:, None] * (1 + 8 * np.finfo(float).eps)
        )
        check_audio(tmp_path / f"{backend}.wav", actual, expected)
    actual, replay, drained = callback_audio(prepared, {"main": source}, actions)
    assert drained
    check_audio(tmp_path / "callback.wav", actual, expected)
    check_audio(tmp_path / "replay.wav", replay, actual)


@pytest.mark.parametrize("parameter,value", [(0, 1), (3, -1), (4, 17)])
def test_invalid_limiter_controls_silence_and_restore(
    tmp_path: Path, parameter: int, value: float
) -> None:
    prepared = effects.prepare(limiter_graph(), 48000)
    runtime = _native.LiveRuntime.empty(48000, 2, 48000, 64)
    runtime.set_effect_graph(
        *effects.native_live_graph(prepared), 2, input_routes=[[-2]]
    )
    saved = runtime.snapshot()
    runtime.submit_effects(
        np.array([[12003, 0, 0, parameter, value, 0]], dtype=np.float64)
    )
    source = np.ones((48000, 2))
    actual = np.ones_like(source)
    with pytest.raises(ValueError, match="limiter"):
        runtime.process_into(actual, [source])
    assert runtime.failed()
    check_audio(tmp_path / "failed.wav", actual, np.zeros_like(source))
    runtime.restore(saved)
    runtime.process_into(actual, [source])
    expected = np.full_like(source, 10 ** (-1 / 20))
    expected[:240] = 0
    check_audio(tmp_path / "restored.wav", actual, expected)


def test_limiter_bypass_keeps_delayed_dry_and_detector_warm(tmp_path: Path) -> None:
    prepared = effects.prepare(limiter_graph(0), 48000)
    source = np.tile([2.0, 1.0], (48000, 1))
    frames = np.arange(48000)
    mix = (
        1 - np.clip((frames - 12243) / 64, 0, 1) + np.clip((frames - 24243) / 64, 0, 1)
    )
    dry = np.vstack([np.zeros((240, 2)), source[:-240]])
    wet = np.minimum(dry, np.array([1, 0.5]) * 10 ** (-1 / 20))
    expected = (1 - mix[:, None]) * dry + mix[:, None] * wet
    expected[36003:] = 0
    actions: list[audio_effects.EffectAction] = [
        audio_effects.BypassAction(
            tick=12003, ordinal=0, processor="limit", bypassed=True
        ),
        audio_effects.BypassAction(
            tick=24003, ordinal=0, processor="limit", bypassed=False
        ),
        audio_effects.StopAction(tick=36003, ordinal=0),
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


def test_limiter_rejects_bad_history_without_replacing_state() -> None:
    renderer = effects.OfflineEffects(effects.prepare(limiter_graph(), 48000), "native")
    original = renderer.snapshot()
    invalid = original.model_copy(deep=True)
    state = invalid.processors["limit"].limiter
    assert state is not None
    state.history.pop()
    with pytest.raises(ValueError, match="snapshot"):
        renderer.restore(invalid)
    assert renderer.snapshot() == original


def test_limiter_opaque_snapshot_rejects_different_lookahead_storage() -> None:
    runtimes = []
    for lookahead in (0.005, 0.01):
        runtime = _native.LiveRuntime.empty(48000, 2, 997, 64)
        runtime.set_effect_graph(
            *effects.native_live_graph(
                effects.prepare(limiter_graph(lookahead=lookahead), 48000)
            ),
            2,
            input_routes=[[-2]],
        )
        runtimes.append(runtime)
    with pytest.raises(ValueError, match="[Ss]napshot"):
        runtimes[1].restore(runtimes[0].snapshot())


@pytest.mark.parametrize("rate,latency", [(44100, 221), (48000, 240), (96000, 480)])
def test_limiter_preparation_uses_the_explicit_render_rate(
    rate: int, latency: int
) -> None:
    prepared = effects.prepare(limiter_graph(), rate)
    assert prepared.latency_frames == latency
    effects.OfflineEffects(prepared, "native")


BACKENDS: list[Literal["numpy", "native"]] = ["numpy", "native"]
