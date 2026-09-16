from fractions import Fraction
from itertools import pairwise
from pathlib import Path
from typing import Literal

import numpy as np
import pytest
from test_synth import check_audio
from ufor import lfo
from ufor.oscillator import Waveform

from enge.lfo import lfo_samples


@pytest.mark.parametrize(
    "waveform,duty",
    [("sine", Fraction(1, 2))]
    + [
        (w, d)
        for w in ("square", "triangle")
        for d in (Fraction(0), Fraction(1, 3), Fraction(1))
    ],
)
def test_lfo_shapes_delay_and_fade_survive_partitions_and_restores(
    tmp_path: Path,
    backend: Literal["numpy", "native"],
    waveform: Waveform,
    duty: Fraction,
) -> None:
    definition = lfo.LFO(
        rate=4,
        phase=Fraction(1, 7),
        waveform=waveform,
        duty_cycle=duty,
        delay=Fraction(3, 96000),
        fade_in=Fraction(24001, 96000),
    )
    state = lfo.initial_lfo(definition, Fraction(0))
    saved = state.model_dump_json()
    actual = lfo_samples(definition, state, 0, 48000, 48000, backend)
    phase = (1 / 7 + np.arange(48000) / 12000) % 1
    if waveform == "sine":
        values = np.sin(2 * np.pi * phase)
    elif waveform == "square":
        values = np.where(phase < float(duty), 1, -1)
    elif duty == 0:
        values = 1 - 2 * phase
    elif duty == 1:
        values = 2 * phase - 1
    else:
        values = np.where(
            phase < float(duty),
            2 * phase / float(duty) - 1,
            (1 + float(duty) - 2 * phase) / (1 - float(duty)),
        )
    expected = np.column_stack(
        [values, np.clip((np.arange(48000) - 1.5) / 12000.5, 0, 1)]
    )
    check_audio(tmp_path / "lfo-shapes.wav", actual, expected)
    chunks = [
        lfo_samples(
            definition,
            lfo.LFOState.model_validate_json(saved),
            a,
            b - a,
            48000,
            backend,
        )
        for a, b in pairwise(sorted({0, 1, 2, 12002, 48000, *range(0, 48000, 997)}))
    ]
    np.testing.assert_allclose(np.concatenate(chunks), actual, atol=1e-10, rtol=1e-9)
    assert state.model_dump_json() == saved


@pytest.mark.parametrize("reset", list(lfo.Reset))
def test_lfo_rate_changes_freeze_and_resets_preserve_exact_event_state(
    tmp_path: Path, backend: Literal["numpy", "native"], reset: lfo.Reset
) -> None:
    definition = lfo.LFO(
        rate=4,
        phase=Fraction(1, 4),
        delay=Fraction(1, 8),
        fade_in=Fraction(1, 8),
        reset=reset,
    )
    state = lfo.initial_lfo(definition, Fraction(0))
    events = [
        lfo.LFOEvent(at=Fraction(f, 48000), ordinal=0, action=a, rate=r)
        for f, a, r in [
            (12000, "rate", Fraction(0)),
            (24000, "rate", Fraction(3)),
            (30000, "trigger", None),
            (36000, "transport", None),
            (42000, "reset", None),
        ]
    ]
    starts = [0, 12000, 24000, 30000, 36000, 42000, 48000]
    phases = [
        0.25,
        0.25,
        0.25,
        0.25 if reset == "trigger" else 0.625,
        0.25 if reset == "transport" else 0.625 if reset == "trigger" else 0,
        0.25,
    ]
    activations = [
        0,
        0,
        0,
        30000 if reset == "trigger" else 0,
        36000 if reset == "transport" else 30000 if reset == "trigger" else 0,
        42000,
    ]
    rates = [4, 0, 3, 3, 3, 3]
    chunks: list[np.ndarray] = []
    expected: list[np.ndarray] = []
    for i, (start, end) in enumerate(pairwise(starts)):
        if i:
            state = lfo.lfo_event(definition, state, events[i - 1])
        state = lfo.LFOState.model_validate_json(state.model_dump_json())
        assert state.rate == rates[i]
        assert state.started_at == Fraction(activations[i], 48000)
        assert state.phase == Fraction(phases[i])
        chunks.append(
            lfo_samples(definition, state, start, end - start, 48000, backend)
        )
        phase = (phases[i] + rates[i] * np.arange(end - start) / 48000) % 1
        weight = np.clip((np.arange(start, end) - activations[i] - 6000) / 6000, 0, 1)
        expected.append(np.column_stack([np.sin(2 * np.pi * phase), weight]))
    check_audio(
        tmp_path / "lfo-rate-reset.wav",
        np.concatenate(chunks),
        np.concatenate(expected),
    )


@pytest.mark.parametrize(
    "waveform,phase,duty,value",
    [
        ("square", Fraction(1, 3) - Fraction(1, 10**30), Fraction(1, 3), 1),
        ("square", Fraction(1, 3), Fraction(1, 3), -1),
        ("triangle", 1 - Fraction(1, 2 * 10**40), 1 - Fraction(1, 10**40), 0),
    ],
)
def test_lfo_edges_are_resolved_before_float_conversion(
    tmp_path: Path,
    backend: Literal["numpy", "native"],
    waveform: Waveform,
    phase: Fraction,
    duty: Fraction,
    value: float,
) -> None:
    definition = lfo.LFO(rate=0, phase=phase, duty_cycle=duty, waveform=waveform)
    state = lfo.initial_lfo(definition, Fraction(0))
    actual = lfo_samples(definition, state, 2**53 + 37, 48000, 48000, backend)
    check_audio(
        tmp_path / "exact-lfo-edge.wav", actual, np.tile([value, 1], (48000, 1))
    )


def test_native_lfo_does_not_call_reference_dsp(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    definition = lfo.LFO(
        rate=Fraction(144001, 3),
        waveform="square",
        duty_cycle=Fraction(1, 3),
        delay=Fraction(1, 7),
        fade_in=Fraction(1, 9),
    )
    state = lfo.initial_lfo(definition, Fraction(0))
    expected = lfo_samples(definition, state, 2**53 + 37, 48000, 48000)

    def forbidden(*args: object, **kwargs: object) -> None:
        pytest.fail("Native LFO called the scalar reference")

    monkeypatch.setattr(lfo, "lfo_at", forbidden)
    actual = lfo_samples(definition, state, 2**53 + 37, 48000, 48000, "native")
    check_audio(tmp_path / "native-lfo.wav", actual, expected)
    assert actual.flags.c_contiguous
    assert not np.shares_memory(actual, expected)


def test_fractional_rate_event_keeps_phase_and_activation_age(
    tmp_path: Path, backend: Literal["numpy", "native"]
) -> None:
    definition = lfo.LFO(rate=4, phase=Fraction(1, 4))
    initial = lfo.initial_lfo(definition, Fraction(0))
    event = lfo.LFOEvent(
        at=Fraction(3, 96000), ordinal=0, action="rate", rate=Fraction(7, 3)
    )
    state = lfo.lfo_event(definition, initial, event)
    actual = np.concatenate(
        [
            lfo_samples(definition, initial, 0, 2, 48000, backend),
            lfo_samples(definition, state, 2, 47998, 48000, backend),
        ]
    )
    frames = np.arange(48000)
    phase = (
        0.25
        + (4 * np.minimum(frames, 1.5) + 7 / 3 * np.maximum(frames - 1.5, 0)) / 48000
    ) % 1
    expected = np.column_stack([np.sin(2 * np.pi * phase), np.ones(48000)])
    check_audio(tmp_path / "fractional-lfo-rate.wav", actual, expected)
    assert state.phase == Fraction(2001, 8000)
    assert state.started_at == 0


@pytest.mark.parametrize("invalid", ["gap", "columns", "nan", "weight"])
def test_native_lfo_rejects_invalid_spans(invalid: str) -> None:
    from enge import _native

    phases = np.array([[0, 1, 0, 0, 1]], dtype=np.float64)
    if invalid == "gap":
        phases[0, 0] = 1
    elif invalid == "columns":
        phases = phases[:, :4]
    elif invalid == "nan":
        phases[0, 2] = np.nan
    weights = np.array(
        [[0, 2 if invalid == "weight" else 1, 1, 0, 0, 1, 0]], dtype=np.float64
    )
    with pytest.raises(ValueError):
        _native.render_lfo(0, phases, weights, 1)
