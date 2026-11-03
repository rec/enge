import wave
from pathlib import Path

import numpy as np
import pytest
from test_synth import check_audio
from ufor import tx81z
from ufor.dx7 import DX7Voice

from enge.synth import EngineError
from enge.yamaha import (
    DX7Envelope,
    DX7Renderer,
    DX7Snapshot,
    TX81ZEnvelope,
    tx81z_base_frequency,
    tx81z_wave_samples,
)


@pytest.mark.parametrize(
    "fixed,coarse,fine,frequency_range,note,hz",
    [
        (False, 4, 8, 0, 69, 660),
        (False, 0, 7, 0, 69, 412.5),
        (False, 1, 0, 0, 69, 220 * np.sqrt(2)),
        (False, 13, 0, 0, 69, 1760),
        (False, 43, 0, 0, 69, 4400 * np.sqrt(2)),
        (False, 44, 0, 0, 69, 3960 * 2 ** (781 / 1200)),
        (True, 0, 0, 0, 69, 8),
        (True, 4, 0, 0, 57, 16),
        (True, 63, 15, 7, 127, 32640),
        (True, 32, 5, 3, 0, 1064),
    ],
)
def test_tx81z_nominal_frequency_preserves_ratio_and_fixed_controls(
    tmp_path: Path,
    fixed: bool,
    coarse: int,
    fine: int,
    frequency_range: int,
    note: int,
    hz: float,
) -> None:
    raw = bytearray(93)
    raw[50] = coarse
    extra = bytearray(23)
    extra[15:18] = bytes([fixed, frequency_range, fine])
    frequency = tx81z_base_frequency(
        tx81z.TX81ZVoice(data=bytes(raw)).operator(1),
        tx81z.TX81ZAdditional(data=bytes(extra)).operator(1),
        note,
    )
    assert frequency == pytest.approx(hz, abs=1e-10)
    phase = 2 * np.pi * np.arange(48000) / 48000
    check_audio(
        tmp_path / "tx81z-frequency.wav",
        np.sin(phase * frequency)[:, None],
        np.sin(phase * hz)[:, None],
    )


@pytest.mark.parametrize("coarse", [0, 1, 2, 3])
def test_tx81z_rejects_unavailable_half_multiple_fine_values(coarse: int) -> None:
    raw = bytearray(93)
    raw[50] = coarse
    extra = bytearray(23)
    extra[17] = 8
    with pytest.raises(EngineError, match="half-multiple"):
        tx81z_base_frequency(
            tx81z.TX81ZVoice(data=bytes(raw)).operator(1),
            tx81z.TX81ZAdditional(data=bytes(extra)).operator(1),
            69,
        )


@pytest.mark.parametrize("number,shift", [(1, 3), (2, 0), (2, 1), (2, 2), (2, 3)])
def test_tx81z_envelope_holds_maximum_and_releases_with_shift(
    tmp_path: Path, number: int, shift: int
) -> None:
    raw = bytearray(93)
    start = (4 - number) * 13
    raw[start] = raw[start + 1] = 31
    raw[start + 3] = raw[start + 4] = 15
    extra = bytearray(23)
    extra[(4 - number) * 5 + 4] = shift
    envelope = TX81ZEnvelope.start(
        tx81z.TX81ZVoice(data=bytes(raw)).operator(number),
        tx81z.TX81ZAdditional(data=bytes(extra)).operator(number),
        0,
    )
    assert envelope.sustain == 0
    attenuation = []
    for i in range(48000):
        if i == 24000:
            envelope.release()
            envelope = TX81ZEnvelope.model_validate_json(envelope.model_dump_json())
        attenuation.append(envelope.clock(i))
    expected = np.minimum(np.maximum(np.arange(48000) - 23999, 0) * 8, 1023)
    expected >>= 0 if number == 1 else shift
    np.testing.assert_array_equal(attenuation, expected)
    carrier = np.sin(2 * np.pi * np.arange(48000) * 440 / 48000)
    actual = carrier * 2 ** (-np.array(attenuation) / 64)
    check_audio(
        tmp_path / "tx81z-envelope.wav",
        actual[:, None],
        (carrier * 2 ** (-expected / 64))[:, None],
    )


@pytest.mark.parametrize("waveform", range(8))
def test_tx81z_waveforms_preserve_signed_and_gated_shapes(
    tmp_path: Path, waveform: int
) -> None:
    root = np.sqrt(0.5)
    low = (2 - np.sqrt(2)) / 4
    high = (2 + np.sqrt(2)) / 4
    a, b = np.sqrt(low), np.sqrt(high)
    cycles = [
        [0, a, root, b, 1, b, root, a, 0, -a, -root, -b, -1, -b, -root, -a],
        [
            0,
            low,
            0.5,
            high,
            1,
            high,
            0.5,
            low,
            0,
            -low,
            -0.5,
            -high,
            -1,
            -high,
            -0.5,
            -low,
        ],
        [0, a, root, b, 1, b, root, a] + [0] * 8,
        [0, low, 0.5, high, 1, high, 0.5, low] + [0] * 8,
        [0, root, 1, root, 0, -root, -1, -root] + [0] * 8,
        [0, 0.5, 1, 0.5, 0, -0.5, -1, -0.5] + [0] * 8,
        [0, root, 1, root, 0, root, 1, root] + [0] * 8,
        [0, 0.5, 1, 0.5, 0, 0.5, 1, 0.5] + [0] * 8,
    ]
    phases = np.arange(48000) / 16 - 1500
    phases.flags.writeable = False
    actual = tx81z_wave_samples(waveform, phases)
    expected = np.tile(cycles[waveform], 3000)
    check_audio(
        tmp_path / f"tx81z-wave-{waveform}.wav", actual[:, None], expected[:, None]
    )


@pytest.mark.parametrize("waveform", [-1, 8])
def test_tx81z_rejects_unknown_waveforms(waveform: int) -> None:
    with pytest.raises(EngineError, match="waveform"):
        tx81z_wave_samples(waveform, np.zeros(48000))


@pytest.mark.parametrize("feedback", [0, 1, 7])
@pytest.mark.parametrize("algorithm", [31, 32])
def test_dx7_feedback_uses_two_delayed_cycle_samples(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, feedback: int, algorithm: int
) -> None:
    def gain(self: DX7Envelope) -> float:
        return 0.01

    monkeypatch.setattr(DX7Envelope, "sample", gain)
    data = bytearray(voice().data)
    data[135] = feedback
    data[134] = algorithm - 1
    renderer = DX7Renderer(DX7Voice(data=bytes(data)), note=69, velocity=100)
    expected = np.empty((48000, 1))
    sine = np.sin(2 * np.pi * np.arange(48000) * 440 / 48000) * 0.01
    previous = older = 0.0
    for i in range(48000):
        offset = (older + previous) / 2 ** (9 - feedback) if feedback else 0.0
        value = 0.01 * np.sin(2 * np.pi * (i * 440 / 48000 + offset))
        expected[i, 0] = (
            5 * sine[i] + value
            if algorithm == 32
            else 4 * sine[i] + 0.01 * np.sin(2 * np.pi * (i * 440 / 48000 + value))
        )
        older, previous = previous, value
    actual = renderer.render(48000)
    check_audio(tmp_path / "feedback.wav", actual, expected)


@pytest.mark.parametrize("released", [False, True])
def test_dx7_snapshot_replays_feedback_and_envelopes(
    tmp_path: Path, released: bool
) -> None:
    data = bytearray(voice().data)
    data[134] = 0
    data[135] = 7
    data[126:130] = bytes([50] * 4)
    data[130:134] = bytes([82, 40, 70, 50])
    renderer = DX7Renderer(DX7Voice(data=bytes(data)), note=69, velocity=100)
    renderer.render(12345)
    if released:
        renderer.release()
    snapshot = DX7Snapshot.model_validate_json(renderer.snapshot().model_dump_json())
    saved = snapshot.model_dump_json()
    expected = renderer.render(48000)
    assert snapshot.model_dump_json() == saved
    renderer.restore(snapshot)
    actual = np.concatenate([renderer.render(997), renderer.render(47003)])
    check_audio(tmp_path / "dx7-snapshot.wav", actual, expected)
    renderer.restore(snapshot)
    check_audio(tmp_path / "dx7-snapshot-reused.wav", renderer.render(48000), expected)
    assert snapshot.model_dump_json() == saved


@pytest.mark.parametrize("difference", ["voice", "note", "velocity", "sample_rate"])
def test_dx7_snapshot_rejects_different_note_definition(difference: str) -> None:
    renderer = DX7Renderer(voice(), note=69, velocity=100)
    raw = renderer.snapshot().model_dump()
    if difference == "voice":
        data = bytearray(voice().data)
        data[135] = 1
        raw["voice"] = DX7Voice(data=bytes(data))
    else:
        raw[difference] += 1
    with pytest.raises(EngineError, match="different note definition"):
        renderer.restore(DX7Snapshot.model_validate(raw))


def test_pitch_envelope_octave_calibration(tmp_path: Path) -> None:
    data = bytearray(voice().data)
    data[130:134] = bytes([82] * 4)
    actual = DX7Renderer(DX7Voice(data=bytes(data)), note=69, velocity=100).render(
        48000
    )
    expected = DX7Renderer(voice(), note=81, velocity=100).render(48000)
    check_audio(tmp_path / "pitch-octave.wav", actual, expected)


def test_pitch_envelope_release_and_partitioned_continuation(tmp_path: Path) -> None:
    data = bytearray(voice().data)
    data[126:130] = bytes([99] * 4)
    data[130:134] = bytes([82, 82, 82, 50])
    patch = DX7Voice(data=bytes(data))
    whole = DX7Renderer(patch, note=69, velocity=100)
    first = whole.render(24000)
    assert whole.pitch_envelope.level == 2**24
    whole.release()
    tail = whole.render(24000)
    assert whole.pitch_envelope.level == 0
    split = DX7Renderer(patch, note=69, velocity=100)
    chunks = [split.render(1001), split.render(22999)]
    split.release()
    # Envelope state is portable independently of the enclosing note renderer.
    split.pitch_envelope = type(split.pitch_envelope).model_validate_json(
        split.pitch_envelope.model_dump_json()
    )
    chunks.extend([split.render(321), split.render(23679)])
    check_audio(
        tmp_path / "pitch-release.wav",
        np.concatenate(chunks),
        np.concatenate([first, tail]),
    )


@pytest.mark.parametrize("coarse", [4, 31])
@pytest.mark.parametrize("detune", [0, 7, 14])
def test_fixed_operators_repeat_decades_and_ignore_played_key(
    tmp_path: Path, coarse: int, detune: int
) -> None:
    data = bytearray(voice().data)
    for i in range(0, 126, 21):
        data[i + 17] = 1
        data[i + 18] = coarse
        data[i + 19] = 37
        data[i + 20] = detune
    actual = DX7Renderer(DX7Voice(data=bytes(data)), note=57, velocity=100).render(
        48000
    )
    for i in range(0, 126, 21):
        data[i + 18] = coarse & 3
    expected = DX7Renderer(DX7Voice(data=bytes(data)), note=81, velocity=100).render(
        48000
    )
    check_audio(tmp_path / "fixed-frequency.wav", actual, expected)
    if detune <= 7:
        peak_hz = int(np.argmax(np.abs(np.fft.rfft(actual[:, 0]))))
        assert abs(peak_hz - 10 ** ((coarse & 3) + 0.37)) < 1
    if detune == 0:
        for i in range(0, 126, 21):
            data[i + 20] = 7
        neutral = DX7Renderer(DX7Voice(data=bytes(data)), note=69, velocity=100).render(
            48000
        )
        check_audio(tmp_path / "fixed-negative-detune.wav", actual, neutral)


def voice() -> DX7Voice:
    data = bytearray(155)
    for start in range(0, 126, 21):
        data[start : start + 4] = bytes([99, 99, 99, 99])
        data[start + 4 : start + 8] = bytes([99, 99, 99, 99])
        data[start + 16] = 99
        data[start + 18] = 1
        data[start + 20] = 7
    data[134] = 31
    data[130:134] = bytes([50] * 4)
    return DX7Voice(data=bytes(data))


@pytest.mark.parametrize(
    "packed,field",
    [
        (False, f)
        for f in (
            "algorithm",
            "feedback",
            "output",
            "rate",
            "level",
            "pitch_rate",
            "pitch_level",
        )
    ]
    + [
        (True, f)
        for f in ("algorithm", "output", "rate", "level", "pitch_rate", "pitch_level")
    ],
)
def test_malformed_parameters_remain_inspectable_but_cannot_render(
    packed: bool, field: str
) -> None:
    data = bytearray(128 if packed else 155)
    positions = {
        "algorithm": (110 if packed else 134, 32),
        "feedback": (111 if packed else 135, 8),
        "output": (13 if packed else 16, 100),
        "rate": (0, 100),
        "level": (4, 100),
        "pitch_rate": (102 if packed else 126, 100),
        "pitch_level": (106 if packed else 130, 100),
    }
    index, value = positions[field]
    data[index] = value
    malformed = DX7Voice(data=bytes(data))
    assert malformed.data == bytes(data)
    assert len(malformed.title) == 10
    with pytest.raises(EngineError):
        DX7Renderer(malformed, note=69, velocity=100)


def test_dx7_renderer_has_deterministic_one_second_wav_regression(
    tmp_path: Path,
) -> None:
    whole = DX7Renderer(voice(), note=69, velocity=100).render(48000)
    split = DX7Renderer(voice(), note=69, velocity=100)
    actual = np.vstack((split.render(17321), split.render(30679)))

    assert whole.shape == (48000, 1)
    assert whole.dtype == np.float64
    assert np.all(np.isfinite(whole))
    assert np.max(np.abs(whole)) > 0
    with wave.open(str(tmp_path / "dx7-renderer.wav"), "wb") as output:
        output.setnchannels(1)
        output.setsampwidth(4)
        output.setframerate(48000)
        output.writeframes(
            (whole / np.max(np.abs(whole)) * (2**31 - 1)).astype("<i4").tobytes()
        )
    np.testing.assert_allclose(actual, whole, atol=1e-10, rtol=1e-9)


def test_dx7_renderer_releases_all_operators() -> None:
    data = bytearray(voice().data)
    for start in range(0, 126, 21):
        data[start + 7] = 0
    renderer = DX7Renderer(DX7Voice(data=bytes(data)), note=69, velocity=100)

    renderer.render(1)
    renderer.release()
    released = renderer.render(48000)

    assert np.max(np.abs(released[-1000:])) < 5e-4
