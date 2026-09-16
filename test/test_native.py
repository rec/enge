from fractions import Fraction
from itertools import pairwise
from pathlib import Path

import numpy as np
import pytest
from test_dynamic_synth import dynamic_score, onset
from test_synth import check_audio
from ufor.envelope import Envelope, Segment
from ufor.oscillator import Oscillator, Waveform
from ufor.samples.processing import ResonantFilter
from ufor.synth_trace import prepare

from enge import filters, native, synth


@pytest.mark.parametrize(
    "waveform,duty",
    [(Waveform.sine, Fraction(1, 2))]
    + [
        (w, d)
        for w in (Waveform.square, Waveform.triangle)
        for d in (Fraction(0), Fraction(1, 3), Fraction(1))
    ],
)
def test_native_matches_reference_without_python_dsp(
    waveform: Waveform, duty: Fraction, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    definition = synth.PreparedVoice(
        sample_rate=48000,
        oscillator=Oscillator(waveform=waveform, duty_cycle=duty),
        envelope=Envelope(
            initial=0.2,
            segments=[
                Segment(duration=0, target=0.4),
                Segment(duration=Fraction(96001, 192000), target=1),
            ],
            release=[
                Segment(duration=Fraction(12001, 96000), target=0.7),
                Segment(duration=0, target=0.4),
                Segment(duration=Fraction(1, 8), target=0.25),
            ],
        ),
        frequencies=[100.25, 200.5],
        routes=[[1, -0.5, 0], [0.25, 0.75, 0]],
        gain=1.3,
        filters=[
            ResonantFilter(
                name="tone", response="lowpass", cutoff_hz=700, q=2, stages=2
            )
        ],
        minimum_hold_seconds=Fraction(8001, 32000),
    )
    reference = synth.VoiceRenderer.start(definition, phase_origin=2**53 + 125)
    compiled = synth.VoiceRenderer.start(
        definition, phase_origin=2**53 + 125, backend="native"
    )
    reference.release()
    compiled.release()
    # Read-only, non-contiguous parameters must not be modified or misread.
    frequencies = (np.arange(48000 * 4).reshape(48000, 4) / 100000 + 100.25)[:, ::2]
    gains = np.linspace(0.1, 0.9, 96000)[::2]
    frequencies.flags.writeable = gains.flags.writeable = False
    expected = reference.render(48000, frequencies, gains)

    def forbidden(*args: object, **kwargs: object) -> None:
        pytest.fail("Native rendering delegated DSP to the NumPy reference")

    for name in ("oscillator_samples", "envelope_samples", "route_samples"):
        monkeypatch.setattr(synth, name, forbidden)
    monkeypatch.setattr(filters, "filter_samples", forbidden)
    chunks: list[np.ndarray] = []
    for start, end in pairwise([0, 120, 12001, 12002, 18002, 24002, 48000]):
        chunks.append(
            compiled.render(end - start, frequencies[start:end], gains[start:end])
        )
    actual = np.concatenate(chunks)
    check_audio(tmp_path / "native-conformance.wav", actual, expected)
    assert np.all(actual[24002:] == 0)
    assert np.all(actual[:, 2] == 0)
    assert compiled.complete and reference.complete
    for actual_state, expected_state in zip(
        compiled.oscillators, reference.oscillators, strict=True
    ):
        assert actual_state.position == pytest.approx(
            expected_state.position, abs=1e-10, rel=1e-9
        )
        assert actual_state.error == pytest.approx(
            expected_state.error, abs=1e-10, rel=1e-9
        )
    assert all(not np.shares_memory(a, b) for a, b in pairwise(chunks))


@pytest.mark.parametrize("invalid", ["sources", "gains", "span", "dtype", "layout"])
def test_native_rejects_invalid_buffers_before_changing_state(invalid: str) -> None:
    states = np.zeros((1, 2))
    frequencies = np.ones((1, 2 if invalid == "sources" else 1))
    gains = np.ones(2 if invalid == "gains" else 1)
    spans = np.array(
        [[0, 2 if invalid == "span" else 1, 1, 0, 0, 1, 0]], dtype=np.float64
    )
    if invalid == "dtype":
        states = states.astype(np.float32)
    elif invalid == "layout":
        spans = np.asfortranarray(np.tile(spans, (2, 1)))
    with pytest.raises((ValueError, TypeError)):
        native.render(
            Oscillator(), states, frequencies, 48000, 1, gains, spans, [[1]], 1
        )
    assert np.all(states == 0)


def test_native_returns_owned_audio_and_state_from_readonly_inputs(
    tmp_path: Path,
) -> None:
    states = np.zeros((1, 2))
    states.flags.writeable = False
    frequencies = np.full((48000, 1), 100.25)
    gains = np.ones(48000)
    spans = np.array([[0, 48000, 1, 0, 0, 1, 0]], dtype=np.float64)
    actual, next_states, _ = native.render(
        Oscillator(waveform=Waveform.sine),
        states,
        frequencies,
        48000,
        1,
        gains,
        spans,
        [[1, 0]],
        48000,
    )
    expected = np.zeros((48000, 2))
    expected[:, 0] = np.sin(2 * np.pi * np.arange(48000) * 100.25 / 48000)
    check_audio(tmp_path / "owned-native-buffers.wav", actual, expected)
    assert next_states[0, 0] == 12000
    assert next_states[0, 1] == 0
    assert np.all(states == 0)
    assert actual.flags.c_contiguous
    assert not np.shares_memory(states, next_states)
    assert not np.shares_memory(actual, next_states)
    next_states[:] = 0
    frequencies[:] = 0
    gains[:] = 0
    spans[:] = 0
    np.testing.assert_allclose(actual, expected, atol=1e-10, rtol=1e-9)


def test_restore_rejects_active_voices_from_another_backend(tmp_path: Path) -> None:
    document = dynamic_score()
    reference = synth.OfflineSynth(synth.prepare(document))
    actions = prepare(document.body, [onset()], seed=0).actions
    actual = reference.advance(actions, 0, 48000)
    expected = np.zeros_like(actual)
    expected[:, 0] = 0.5 * np.sin(2 * np.pi * np.arange(48000) * 100 / 48000)
    check_audio(tmp_path / "snapshot-backend.wav", actual, expected)
    compiled = synth.OfflineSynth(synth.prepare(document), backend="native")
    with pytest.raises(synth.EngineError, match="different synth backend"):
        compiled.restore(reference.snapshot())
