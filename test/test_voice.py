from fractions import Fraction
from itertools import pairwise
from pathlib import Path
from typing import Literal

import numpy as np
import pytest
from test_synth import check_audio
from ufor.envelope import Envelope, Segment
from ufor.oscillator import Oscillator, Waveform

from enge.synth import PreparedVoice, VoiceRenderer


@pytest.mark.parametrize("block", [64, 997, 1024])
def test_routed_voice_restores_pending_release_and_retires_exactly(
    tmp_path: Path, block: int, backend: Literal["numpy", "native"]
) -> None:
    definition = PreparedVoice(
        sample_rate=48000,
        oscillator=Oscillator(waveform=Waveform.sine),
        envelope=Envelope(
            segments=[Segment(duration=Fraction(1, 10), target=1)],
            release=[Segment(duration=Fraction(12001, 96000), target=0.25)],
        ),
        frequencies=[100, 200],
        routes=[[1, 1, 0], [-0.25, 0.5, 0]],
        gain=2,
        minimum_hold_seconds=Fraction(8001, 32000),
    )
    renderer = VoiceRenderer.start(definition, backend=backend)
    boundaries = sorted({0, 120, 6000, 14000, 18002, 48000, *range(0, 48000, block)})
    chunks: list[np.ndarray] = []
    for start, end in pairwise(boundaries):
        if start == 120:
            assert renderer.release()
            assert not renderer.release()
        if start in (6000, 14000):
            saved = renderer.model_dump_json()
            restored = VoiceRenderer.model_validate_json(saved)
            renderer.render(1)
            assert restored.model_dump_json() == saved
            renderer = restored
        chunks.append(renderer.render(end - start))
        assert renderer.complete == (end >= 18002)

    frames = np.arange(48000)
    envelope = np.where(
        frames < 12001.5,
        np.minimum(frames / 4800, 1),
        1 - 0.75 * (frames - 12001.5) / 6000.5,
    )
    envelope[18002:] = 0
    low = np.sin(2 * np.pi * frames * 100 / 48000)
    high = np.sin(2 * np.pi * frames * 200 / 48000)
    expected = np.column_stack([low - 0.25 * high, low + 0.5 * high, np.zeros(48000)])
    expected *= (envelope * 2)[:, None]
    actual = np.concatenate(chunks)
    check_audio(tmp_path / "routed-release.wav", actual, expected)
    assert actual.flags.c_contiguous
    assert np.max(np.abs(actual)) > 1
    assert np.all(actual[18002:] == 0)
    assert np.all(actual[:, 2] == 0)
    phases = renderer.oscillators.copy()
    assert phases[0].position == 18002 * 100 % 48000
    assert phases[1].position == 18002 * 200 % 48000
    assert np.all(renderer.render(48000) == 0)
    assert renderer.oscillators == phases
    assert not renderer.release()


@pytest.mark.parametrize("routes", [[[1]], [[1], [1, 0]], [[], []]])
def test_voice_rejects_inconsistent_channel_routes(routes: list[list[float]]) -> None:
    with pytest.raises(ValueError, match="route"):
        PreparedVoice(
            sample_rate=48000,
            oscillator=Oscillator(),
            envelope=Envelope(
                segments=[Segment(duration=0, target=1)],
                release=[Segment(duration=0, target=0)],
            ),
            frequencies=[100, 200],
            routes=routes,
        )
