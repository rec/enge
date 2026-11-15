from pathlib import Path
from typing import Literal

import numpy as np
import pytest
from multiband_oracle import bands, compressed
from test_effect_envelfo import modulated_graph
from test_effects import graph_input
from test_persistent_effects import callback_audio
from test_synth import check_audio
from ufor import audio_effects

from enge import _native, effects


def graph(compress: bool = False) -> audio_effects.EffectGraph:
    high = (
        audio_effects.CompressorBand(
            threshold_db=-24,
            ratio=4,
            knee_db=6,
            attack_seconds=0.001,
            release_seconds=0.05,
        )
        if compress
        else audio_effects.CompressorBand()
    )
    return effects.serial_graph(
        audio_effects.AttachmentScope.master,
        graph_input(),
        [audio_effects.MultibandCompressor(name="bands", high=high)],
        48000,
    )


def source_audio() -> np.ndarray:
    frames = np.arange(48000)
    time = frames / 48000
    low = 0.1 * np.sin(2 * np.pi * 70 * time)
    high = (
        0.4 * np.sin(2 * np.pi * 7000 * time) * ((frames >= 12003) & (frames < 30007))
    )
    source = np.column_stack([low + high, low * 0.5 + high * 0.1])
    source[36003:] = 0
    return source


@pytest.mark.parametrize("compress", [False, True])
def test_multiband_matches_independent_transfers_and_linked_compression_across_replay(
    tmp_path: Path, compress: bool
) -> None:
    source = source_audio()
    low, mid, high = bands(source)
    expected = (
        low + mid + (compressed(high, -24, 4, 6, 0.001, 0.05) if compress else high)
    )
    prepared = effects.prepare(graph(compress), 48000)
    assert prepared.latency_frames == 0 and prepared.channels == ["left", "right"]
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
        saved = effects.EffectSnapshot.model_validate_json(
            first.snapshot().model_dump_json()
        )
        second = effects.OfflineEffects(prepared, target)
        second.restore(saved)
        tail = second.advance({"main": source[23928:]}, actions, 23928, 48000)
        assert second.drained
        actual = np.vstack([head, tail])
        assert np.any(actual[36003:37000])
        check_audio(tmp_path / f"{origin}-{target}.wav", actual, expected)
    actual, replay, drained = callback_audio(prepared, {"main": source}, actions)
    assert drained
    check_audio(tmp_path / "callback.wav", actual, expected)
    check_audio(tmp_path / "replay.wav", replay, actual)


def test_multiband_neutral_impulse_has_allpass_magnitude(tmp_path: Path) -> None:
    source = np.zeros((48000, 2))
    source[0] = [1, -0.5]
    expected = np.sum(bands(source), axis=0)
    for backend in BACKENDS:
        actual = effects.process_audio(
            effects.prepare(graph(), 48000),
            {"main": source},
            backend=backend,
            block_frames=137,
        )
        np.testing.assert_allclose(
            np.abs(np.fft.rfft(actual[:, 0])), 1, rtol=0, atol=1e-10
        )
        check_audio(tmp_path / f"{backend}.wav", actual, expected)


def test_multiband_rejects_nyquist_crossovers() -> None:
    with pytest.raises(ValueError, match="Nyquist"):
        effects.prepare(graph(), 8000)
    raw = graph().model_dump()
    raw["processors"][0]["lower_crossover_hz"] = 1e-300
    with pytest.raises(ValueError, match="Unrepresentable"):
        effects.prepare(audio_effects.EffectGraph.model_validate(raw), 48000)


def test_multiband_scoped_modulation_and_ramps_keep_crossover_state(
    tmp_path: Path,
) -> None:
    raw = graph(True).model_dump()
    modulation = modulated_graph("gain")[0].model_dump()
    raw.update({n: modulation[n] for n in ("envelfos", "bindings", "modulation")})
    target = {"name": "bands", "parameter": "high-threshold_db"}
    raw["modulation"]["parameters"][0].update(
        target=target, unit="db", minimum=-48, maximum=-6, default=-24
    )
    raw["modulation"]["routes"][0].update(
        target=target,
        unit="db",
        points=[{"input": -1, "amount": -6}, {"input": 1, "amount": 6}],
    )
    prepared = effects.prepare(audio_effects.EffectGraph.model_validate(raw), 48000)
    source = source_audio()
    frames = np.arange(48000)
    threshold = (
        -24
        - 6 * np.clip((frames - 15003) / 6000, 0, 1)
        + 6 * np.where((frames // 12000) % 2 == 0, 1, -1)
    )
    low, mid, high = bands(source)
    expected = low + mid + compressed(high, threshold, 4, 6, 0.001, 0.05)
    actions: list[audio_effects.EffectAction] = [
        audio_effects.ParameterAction(
            tick=15003,
            ordinal=0,
            processor="bands",
            parameter="high-threshold_db",
            value=-30,
            duration_frames=6000,
        ),
        audio_effects.InputEndAction(tick=36003, ordinal=0, input="main"),
    ]
    for backend in BACKENDS:
        actual = effects.process_audio(
            prepared, {"main": source}, actions, backend, 137
        )
        check_audio(tmp_path / f"{backend}.wav", actual, expected)
    actual, replay, drained = callback_audio(prepared, {"main": source}, actions)
    assert drained
    check_audio(tmp_path / "callback.wav", actual, expected)
    check_audio(tmp_path / "replay.wav", replay, actual)


def test_multiband_mono_stop_and_malformed_snapshot(tmp_path: Path) -> None:
    raw = graph().model_dump()
    raw["inputs"][0]["channels"] = ["mono"]
    prepared = effects.prepare(audio_effects.EffectGraph.model_validate(raw), 48000)
    source = source_audio()[:, :1]
    expected = np.sum(bands(source), axis=0)
    expected[24007:] = 0
    actions: list[audio_effects.EffectAction] = [
        audio_effects.StopAction(tick=24007, ordinal=0)
    ]
    for backend in BACKENDS:
        renderer = effects.OfflineEffects(prepared, backend)
        for field in ["integrators", "reductions"]:
            saved = renderer.snapshot()
            assert saved.processors["bands"].multiband is not None
            getattr(saved.processors["bands"].multiband, field).pop()
            with pytest.raises(ValueError, match="multiband|Multiband"):
                renderer.restore(saved)
        actual = renderer.advance({"main": source}, actions, 0, 48000)
        assert renderer.drained
        check_audio(tmp_path / f"{backend}.wav", actual, expected)
    actual, replay, drained = callback_audio(prepared, {"main": source}, actions)
    assert drained
    check_audio(tmp_path / "callback.wav", actual, expected)
    check_audio(tmp_path / "replay.wav", replay, actual)


def test_failed_multiband_callback_silences_latches_and_restores(
    tmp_path: Path,
) -> None:
    prepared = effects.prepare(graph(), 48000)
    runtime = _native.LiveRuntime.empty(48000, 2, 48000, 64)
    runtime.set_effect_graph(
        *effects.native_live_graph(prepared), 2, input_routes=[[-2]]
    )
    saved = runtime.snapshot()
    runtime.submit_effects(np.array([[12003, 0, 0, 7 + 5, 1e9, 0]], dtype=np.float64))
    source = source_audio()
    actual = np.ones_like(source)
    with pytest.raises(ValueError, match="compressor"):
        runtime.process_into(actual, [source])
    assert runtime.failed()
    check_audio(tmp_path / "failed.wav", actual, np.zeros_like(source))
    runtime.restore(saved)
    runtime.process_into(actual, [source])
    check_audio(tmp_path / "restored.wav", actual, np.sum(bands(source), axis=0))


BACKENDS: list[Literal["numpy", "native"]] = ["numpy", "native"]
