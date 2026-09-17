from pathlib import Path
from typing import Literal

import numpy as np
import pytest
from test_synth import check_audio
from ufor.samples import processing

from enge import _native, filters


def matrix_filter(
    definitions: list[processing.ResonantFilter],
    samples: np.ndarray,
    values: np.ndarray,
) -> np.ndarray:
    """Independent oracle: explicitly solve the coupled trapezoidal equations."""
    audio = samples.copy()
    for j, definition in enumerate(definitions):
        for _ in range(definition.stages):
            state = np.zeros((2, audio.shape[1]))
            for i, (cutoff, q) in enumerate(values[:, j]):
                g = np.tan(np.pi * cutoff / 48000)
                rhs = state.copy()
                rhs[0] += g * audio[i]
                # Invert [[1 + g/q, g], [-g, 1]] directly. This remains
                # independent from the engine's state-variable recurrence and
                # avoids 48,000 tiny general-purpose LAPACK calls per vector.
                determinant = 1 + g / q + g * g
                band = (rhs[0] - g * rhs[1]) / determinant
                low = (g * rhs[0] + (1 + g / q) * rhs[1]) / determinant
                state = 2 * np.vstack([band, low]) - state
                if definition.response == "lowpass":
                    audio[i] = low
                elif definition.response == "highpass":
                    audio[i] -= band / q + low
                elif definition.response == "bandpass":
                    audio[i] = band / q
                else:
                    audio[i] -= band / q
    return audio


@pytest.mark.parametrize("response", list(processing.FilterResponse))
@pytest.mark.parametrize("stages", [1, 2])
def test_static_filters_match_rbj_transfer_function(
    response: processing.FilterResponse,
    stages: Literal[1, 2],
    backend: Literal["numpy", "native"],
    tmp_path: Path,
) -> None:
    definition = processing.ResonantFilter(
        name="tone", response=response, cutoff_hz=1700, q=1.7, stages=stages
    )
    coefficients = processing.biquad_coefficients(definition, 48000)
    z = np.exp(-2j * np.pi * np.arange(24001) / 48000)
    transfer = (
        (coefficients.b0 + coefficients.b1 * z + coefficients.b2 * z**2)
        / (1 + coefficients.a1 * z + coefficients.a2 * z**2)
    ) ** stages
    expected = np.column_stack(
        [
            np.fft.irfft(transfer, n=48000),
            np.fft.irfft(-0.3 * transfer * z**37, n=48000),
        ]
    )
    impulse = np.zeros((48000, 2))
    impulse[0, 0], impulse[37, 1] = 1, -0.3
    actual, _ = filters.filter_samples(
        [definition],
        filters.initial_states([definition], 2),
        impulse,
        filters.parameters([definition], 48000, 48000),
        48000,
        backend,
    )
    check_audio(tmp_path / "static-filter.wav", actual, expected)
    usable = np.abs(transfer) > 1e-5
    db_error = 20 * np.log10(
        np.abs(np.fft.rfft(actual[:, 0])[usable] / transfer[usable])
    )
    assert np.max(np.abs(db_error)) < definition.tolerance.response_db


@pytest.fixture(scope="module")
def changing_filters() -> tuple[
    list[processing.ResonantFilter], np.ndarray, np.ndarray, np.ndarray
]:
    definitions = [
        processing.ResonantFilter(
            name="low", response="lowpass", cutoff_hz=1000, stages=2
        ),
        processing.ResonantFilter(name="high", response="highpass", cutoff_hz=100),
    ]
    frames = np.arange(48000)
    values = filters.parameters(definitions, 48000, 48000)
    values[:, 0, 0] = np.where(frames % 2, 20000, 1000)
    values[:, 0, 1] = np.where(frames % 3, 0.15, 12)
    values[:, 1, 0] = 200 + 150 * np.sin(2 * np.pi * frames / 17000)
    values[:, 1, 1] = 0.8
    source = np.random.default_rng(731).uniform(-0.2, 0.2, (48000, 2))
    source[24000:] = 0
    return definitions, source, values, matrix_filter(definitions, source, values)


@pytest.mark.parametrize("block", [64, 128, 256, 997, 1024])
def test_audio_rate_changes_survive_partitions_and_serialized_restores(
    changing_filters: tuple[
        list[processing.ResonantFilter], np.ndarray, np.ndarray, np.ndarray
    ],
    block: int,
    backend: Literal["numpy", "native"],
    tmp_path: Path,
) -> None:
    definitions, source, values, expected = changing_filters
    states = filters.initial_states(definitions, 2)
    chunks: list[np.ndarray] = []
    for start in range(0, 48000, block):
        audio, states = filters.filter_samples(
            definitions,
            states,
            source[start : start + block],
            values[start : start + block],
            48000,
            backend,
        )
        chunks.append(audio)
        states = [
            filters.FilterState.model_validate_json(s.model_dump_json()) for s in states
        ]
    actual = np.concatenate(chunks)
    check_audio(tmp_path / "changing-filters.wav", actual, expected)
    assert np.max(np.abs(actual)) < 1
    assert not np.any(source[24000:])
    assert all(not np.shares_memory(c, source) for c in chunks)


def test_switching_cutoff_cannot_amplify_zero_input_state(
    backend: Literal["numpy", "native"],
    tmp_path: Path,
) -> None:
    definitions = [
        processing.ResonantFilter(name="tone", response="lowpass", cutoff_hz=1000)
    ]
    states = [filters.FilterState(integrators=[[1, -0.3]])]
    energy = 1.09
    actual = np.zeros((48000, 1))
    for i in range(256):
        values = np.array([[[20000 if i % 2 else 1000, 1 / np.sqrt(2)]]])
        audio, states = filters.filter_samples(
            definitions, states, np.zeros((1, 1)), values, 48000, backend
        )
        actual[i] = audio[0]
        next_energy = float(np.sum(np.array(states[0].integrators) ** 2))
        assert next_energy <= energy * (1 + 1e-13)
        energy = next_energy
    assert energy < 1e-30
    # Reconstruct the expected zero-input response from the matrix state equations.
    state = np.array([1.0, -0.3])
    expected = np.zeros_like(actual)
    for i in range(256):
        g = np.tan(np.pi * (20000 if i % 2 else 1000) / 48000)
        solution = np.linalg.solve([[1 + g * np.sqrt(2), g], [-g, 1]], state)
        expected[i, 0] = solution[1]
        state = 2 * solution - state
    check_audio(tmp_path / "zero-input-decay.wav", actual, expected)


def test_filter_boundary_policy_is_applied_after_modulation() -> None:
    definition = processing.ResonantFilter(
        name="tone", response="lowpass", cutoff_hz=1000
    )
    values = np.array([[[0.5, 0.7]], [[24000, 2]]])
    with pytest.raises(ValueError, match="tone: invalid cutoff/Q at frame 123"):
        filters.parameters([definition], 48000, 2, values, start=123)
    clipped = definition.model_copy(
        update={"boundary": processing.FilterBoundary.clamp}
    )
    actual = filters.parameters([clipped], 48000, 2, values)
    np.testing.assert_array_equal(actual[:, 0, 0], [1, 23976])
    assert values[0, 0, 0] == 0.5
    for invalid in (0, -1, np.inf, np.nan):
        values[0, 0, 1] = invalid
        with pytest.raises(ValueError, match="cutoff/Q"):
            filters.parameters([clipped], 48000, 2, values)


@pytest.mark.parametrize("q", [1e-310, 1e300])
def test_extreme_positive_q_remains_finite(
    q: float,
    backend: Literal["numpy", "native"],
    tmp_path: Path,
) -> None:
    definition = processing.ResonantFilter(
        name="tone", response="bandpass", cutoff_hz=12000, q=q
    )
    source = np.zeros((48000, 1))
    source[0] = 1
    actual, states = filters.filter_samples(
        [definition],
        filters.initial_states([definition], 1),
        source,
        filters.parameters([definition], 48000, 48000),
        48000,
        backend,
    )
    assert np.all(np.isfinite(actual))
    assert np.all(np.isfinite(states[0].integrators))
    expected = np.zeros_like(source)
    expected[0] = 1 if q < 1 else 0
    check_audio(tmp_path / "extreme-q.wav", actual, expected)


@pytest.mark.parametrize("invalid", ["response", "state", "parameters", "cutoff", "q"])
def test_native_filter_rejects_invalid_buffers_without_mutating_inputs(
    invalid: str,
) -> None:
    source = np.ones((1, 2))
    states = np.zeros((1, 1 if invalid == "state" else 2, 2))
    values = np.array([[[1000.0, 0.7]]])
    if invalid == "parameters":
        values = values[:, :, :1]
    elif invalid == "cutoff":
        values[0, 0, 0] = 24000
    elif invalid == "q":
        values[0, 0, 1] = np.nan
    with pytest.raises(ValueError, match="native filter"):
        _native.render_filters(
            source, 48000, ([4 if invalid == "response" else 0], states, values)
        )
    assert np.all(source == 1)
    assert not np.any(states)
