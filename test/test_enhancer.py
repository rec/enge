from pathlib import Path
from typing import Literal

import numpy as np
from test_effects import graph_input
from test_persistent_effects import callback_audio
from test_synth import check_audio
from ufor import audio_effects
from waveshaper_oracle import waveshaper_audio

from enge import effects


def test_enhancer_matches_independent_transfers_and_aligned_dry_mix(
    tmp_path: Path,
) -> None:
    graph = audio_effects.enhancer_graph(
        "exciter", audio_effects.AttachmentScope.master, graph_input(), 48000
    )
    prepared = effects.prepare(graph, 48000)
    assert prepared.latency_frames == 48
    time = np.arange(48000) / 48000
    source = np.column_stack(
        [0.4 * np.sin(2 * np.pi * f * time) + 0.1 for f in (997, 7001)]
    )
    source[12003:] = 0
    size = 131072
    z = np.exp(-2j * np.pi * np.arange(size // 2 + 1) / size)
    g = np.tan(np.pi * 2000 / 48000)
    pole = (1 - g) / (1 + g)
    response = (1 + pole) / 2 * (1 - z) / (1 - pole * z)
    high = np.fft.irfft(
        np.fft.rfft(source, n=size, axis=0) * response[:, None], n=size, axis=0
    )[:48000]
    shaped = waveshaper_audio(high, 2, 0.25)
    g = np.tan(np.pi * 20 / 48000)
    pole = (1 - g) / (1 + g)
    response = (1 + pole) / 2 * (1 - z) / (1 - pole * z)
    wet = np.fft.irfft(
        np.fft.rfft(shaped, n=size, axis=0) * response[:, None], n=size, axis=0
    )[:48000]
    dry = np.vstack([np.zeros((48, 2)), source[:-48]])
    expected = dry + 0.25 * wet
    actions: list[audio_effects.EffectAction] = [
        audio_effects.InputEndAction(tick=12003, ordinal=0, input="main")
    ]
    pairs: list[tuple[Literal["numpy", "native"], Literal["numpy", "native"]]] = [
        ("numpy", "native"),
        ("native", "numpy"),
    ]
    for origin, target in pairs:
        first = effects.OfflineEffects(prepared, origin)
        head = first.advance({"main": source[:23928]}, actions, 0, 23928)
        saved = effects.EffectSnapshot.model_validate_json(
            first.snapshot().model_dump_json()
        )
        second = effects.OfflineEffects(prepared, target)
        second.restore(saved)
        tail = second.advance({"main": source[23928:]}, [], 23928, 48000)
        assert second.drained
        check_audio(
            tmp_path / f"{origin}-{target}.wav", np.vstack([head, tail]), expected
        )
    actual, replay, drained = callback_audio(prepared, {"main": source}, actions)
    assert drained
    check_audio(tmp_path / "callback.wav", actual, expected)
    check_audio(tmp_path / "replay.wav", replay, actual)
