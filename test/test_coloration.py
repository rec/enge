from pathlib import Path
from typing import Literal

import numpy as np
import pytest
from test_effects import graph_input
from test_persistent_effects import callback_audio
from test_synth import check_audio
from ufor import audio_effects
from waveshaper_oracle import waveshaper_audio

from enge import effects


@pytest.mark.parametrize(
    "profile,cutoff,bias", [("tape", 8000, 0), ("tube", 12000, 0.25)]
)
def test_coloration_matches_independent_linear_transfers_and_oversampling(
    tmp_path: Path, profile: Literal["tape", "tube"], cutoff: float, bias: float
) -> None:
    graph = audio_effects.coloration_graph(
        "color", audio_effects.AttachmentScope.master, graph_input(), profile, 48000
    )
    prepared = effects.prepare(graph, 48000)
    assert prepared.latency_frames == 48
    source = np.column_stack(
        [
            0.7 * np.cos(2 * np.pi * f * np.arange(48000) / 48000) + 0.2
            for f in (997, 13001)
        ]
    )
    source[23003:] = 0
    size = 131072
    z = np.exp(-2j * np.pi * np.arange(size // 2 + 1) / size)
    g = np.tan(np.pi * 20 / 48000)
    pole = (1 - g) / (1 + g)
    high = (1 + pole) / 2 * (1 - z) / (1 - pole * z)
    first = np.fft.irfft(
        np.fft.rfft(source, n=size, axis=0) * high[:, None], n=size, axis=0
    )[:48000]
    shaped = waveshaper_audio(first, 2, bias)
    k = np.tan(np.pi * cutoff / 48000)
    den = 1 + np.sqrt(2) * k + k * k
    low = (
        k
        * k
        * (1 + z) ** 2
        / (den + 2 * (k * k - 1) * z + (1 - np.sqrt(2) * k + k * k) * z * z)
    )
    expected = np.fft.irfft(
        np.fft.rfft(shaped, n=size, axis=0) * low[:, None], n=size, axis=0
    )[:48000]
    actions: list[audio_effects.EffectAction] = [
        audio_effects.InputEndAction(tick=23003, ordinal=0, input="main")
    ]
    pairs: list[tuple[Literal["numpy", "native"], Literal["numpy", "native"]]] = [
        ("numpy", "native"),
        ("native", "numpy"),
    ]
    for origin, target in pairs:
        renderer = effects.OfflineEffects(prepared, origin)
        before = renderer.advance({"main": source[:23928]}, actions, 0, 23928)
        saved = effects.EffectSnapshot.model_validate_json(
            renderer.snapshot().model_dump_json()
        )
        restored = effects.OfflineEffects(prepared, target)
        restored.restore(saved)
        after = restored.advance({"main": source[23928:]}, [], 23928, 48000)
        assert restored.drained
        check_audio(
            tmp_path / f"{origin}-{target}.wav", np.vstack([before, after]), expected
        )
    actual, replay, drained = callback_audio(prepared, {"main": source}, actions)
    assert drained
    check_audio(tmp_path / "callback.wav", actual, expected)
    check_audio(tmp_path / "replay.wav", replay, actual)
