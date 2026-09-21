from pathlib import Path

import numpy as np
from test_synth import check_audio
from ufor.samples.processing import FilterResponse, ResonantFilter

from enge import _native, filters


def test_persistent_runtime_matches_oscillators_and_filter(tmp_path: Path) -> None:
    frequencies = [110.0, 165.0, 220.0]
    gains = [0.2, 0.15, 0.1]
    routes = np.array([[1.0, 0.25], [0.5, 1.0], [0.75, 0.75]])
    runtime = _native.OscillatorFilterRuntime(48000, frequencies, gains, routes)
    actual = np.concatenate(
        [runtime.process(n, 2400, 0.8, -3) for n in (997, 4096, 42907)]
    )
    frames = np.arange(48000)
    dry = sum(
        np.sin(2 * np.pi * frames * f / 48000)[:, None] * g * r
        for f, g, r in zip(frequencies, gains, routes, strict=True)
    )
    definition = ResonantFilter(
        name="low", response=FilterResponse.lowpass, cutoff_hz=2400, q=0.8
    )
    expected, _ = filters.filter_samples(
        [definition],
        filters.initial_states([definition], 2),
        dry,
        filters.parameters([definition], 48000, 48000),
        48000,
    )
    expected *= 10 ** (-3 / 20)

    check_audio(tmp_path / "persistent-runtime.wav", actual, expected)


def test_persistent_runtime_applies_voice_actions_at_exact_frames() -> None:
    runtime = _native.OscillatorFilterRuntime(
        48000, [100.0, 200.0], [0.0, 0.0], np.eye(2)
    )
    actions = np.array(
        [
            [10, 0, 0, 100, 0.5, 4],
            [20, 0, 1, 200, 0.25, 0],
            [30, 3, 0, 200, 0.25, 5],
            [40, 1, 1, 0, 0, 4],
            [50, 2, 0, 0, 0, 0],
        ],
        dtype=np.float64,
    )

    actual = runtime.process_actions(64, 10000, 0.7, 0, actions)

    assert np.all(actual[:11] == 0)
    assert np.any(actual[11:20, 0])
    assert np.all(actual[:21, 1] == 0)
    assert np.any(actual[21:44, 1])
    assert abs(actual[-1, 1]) < abs(actual[43, 1])
    assert abs(actual[-1, 0]) < abs(actual[49, 0])
