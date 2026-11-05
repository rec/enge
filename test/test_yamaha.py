import wave
from pathlib import Path

import numpy as np
import pytest
from test_synth import check_audio
from test_tx81z_frequency import opz_frequency
from test_tx81z_integer import opz_carrier, reference_tables
from ufor import tx81z
from ufor.dx7 import DX7Voice

from enge.synth import EngineError
from enge.yamaha import (
    DX7Envelope,
    DX7Renderer,
    DX7Snapshot,
    TX81ZEnvelope,
    TX81ZEnvelopeBank,
    TX81ZRenderer,
    TX81ZSnapshot,
    TX81ZState,
    dx7_operator_sample,
    tx81z_base_frequency,
    tx81z_detune_step,
    tx81z_operator_samples,
    tx81z_output_gain,
    tx81z_wave_samples,
)


@pytest.mark.parametrize(
    "source,index,value",
    [
        ("voice", 5, 100),
        ("voice", 9, 8),
        ("voice", 56, 100),
        ("additional", 20, 8),
        ("voice", 10, 100),
        ("additional", 3, 8),
        ("voice", 54, 100),
        ("voice", 58, 2),
        ("voice", 59, 4),
        ("voice", 60, 8),
        ("voice", 61, 4),
        ("voice", 62, 49),
        ("voice", 63, 2),
        ("voice", 64, 13),
        ("voice", 65, 2),
        ("voice", 66, 100),
        ("voice", 68, 2),
        ("voice", 70, 2),
        ("voice", 75, 100),
        ("additional", 21, 100),
        ("additional", 22, 100),
    ],
)
def test_tx81z_renderer_rejects_unimplemented_or_invalid_parameters(
    source: str, index: int, value: int
) -> None:
    raw = bytearray(93)
    raw[62] = 24
    extra = bytearray(23)
    (raw if source == "voice" else extra)[index] = value
    with pytest.raises(EngineError):
        TX81ZRenderer(
            tx81z.TX81ZVoice(data=bytes(raw)),
            tx81z.TX81ZAdditional(data=bytes(extra)),
            69,
            100,
        )


@pytest.mark.parametrize(
    "source,index,value",
    [
        ("voice", 54, 99),
        ("voice", 58, 1),
        ("voice", 59, 3),
        ("voice", 60, 7),
        ("voice", 61, 3),
        ("voice", 62, 48),
        ("voice", 64, 12),
        ("additional", 21, 99),
        ("additional", 22, 99),
    ],
)
def test_tx81z_renderer_preserves_valid_global_boundaries(
    source: str, index: int, value: int
) -> None:
    raw = bytearray(93)
    raw[62] = 24
    extra = bytearray(23)
    (raw if source == "voice" else extra)[index] = value
    voice = tx81z.TX81ZVoice(data=bytes(raw))
    additional = tx81z.TX81ZAdditional(data=bytes(extra))
    renderer = TX81ZRenderer(voice, additional, 60, 100)
    saved = renderer.snapshot()
    assert saved.voice.data == voice.data
    assert saved.additional.data == additional.data


@pytest.mark.parametrize("algorithm", range(1, 9))
def test_tx81z_note_renderer_pairs_buffers_and_restores_release_tails(
    tmp_path: Path, algorithm: int
) -> None:
    raw = bytearray(93)
    extra = bytearray(23)
    for i in range(1, 5):
        start = (4 - i) * 13
        raw[start] = 31
        raw[start + 1] = 10
        raw[start + 3] = 15
        raw[start + 4] = 11
        raw[start + 5] = 10 * i
        raw[start + 9] = i
        raw[start + 10] = 80 - i
        raw[start + 11] = [4, 8, 10, 13][i - 1]
        raw[start + 12] = i
        extra[(4 - i) * 5 + 3] = [0, 2, 4, 7][i - 1]
    raw[52], raw[53], raw[62] = algorithm - 1, 5, 24
    extra[0], extra[1] = 1, 1
    extra[20] = 7 if algorithm == 8 else 0
    voice = tx81z.TX81ZVoice(data=bytes(raw))
    additional = tx81z.TX81ZAdditional(data=bytes(extra))
    whole = TX81ZRenderer(voice, additional, 69, 100)
    first = whole.render(24000)
    held = TX81ZSnapshot.model_validate_json(whole.snapshot().model_dump_json())
    whole.release()
    saved = TX81ZSnapshot.model_validate_json(whole.snapshot().model_dump_json())
    tail = whole.render(24000)
    replay = TX81ZRenderer(voice, additional, 69, 100)
    replay.restore(held)
    replay.release()
    restored = np.concatenate([replay.render(997), replay.render(23003)])
    check_audio(
        tmp_path / "tx81z-note.wav",
        np.concatenate([first, restored]),
        np.concatenate([first, tail]),
    )
    replay.restore(saved)
    np.testing.assert_array_equal(replay.render(24000), tail)
    assert replay.snapshot() == whole.snapshot()
    assert np.max(np.abs(tail[-1000:])) < 0.001


def test_tx81z_reverb_cannot_accelerate_a_slow_release(tmp_path: Path) -> None:
    raw = bytearray(93)
    raw[39] = 31
    envelope = TX81ZEnvelope.start(
        tx81z.TX81ZVoice(data=bytes(raw)).operator(1),
        tx81z.TX81ZAdditional(data=bytes(23)).operator(1),
        0,
        7,
    )
    envelope.attenuation = 200
    plain = envelope.model_copy(deep=True, update={"reverb_rate": 0})
    bank = TX81ZEnvelopeBank(
        envelopes=[
            plain,
            envelope,
            plain.model_copy(deep=True),
            envelope.model_copy(deep=True),
        ],
        clock_hz=9216000,
    )
    bank.release()
    actual = bank.attenuation_samples(48000)
    expected = np.repeat(actual[:, :1], 4, axis=1)
    np.testing.assert_array_equal(actual, expected)
    carrier = np.sin(2 * np.pi * np.arange(48000) * 440 / 48000)[:, None]
    check_audio(
        tmp_path / "tx81z-slow-release.wav",
        carrier * 2 ** (-actual / 64),
        carrier * 2 ** (-expected / 64),
    )


def test_tx81z_reverb_changes_only_the_post_release_decay(tmp_path: Path) -> None:
    envelope = TX81ZEnvelope(
        rates=[62, 62, 0, 62], sustain=0, shift=0, attenuation=0, reverb_rate=30
    )
    bank = TX81ZEnvelopeBank(
        envelopes=[envelope.model_copy(deep=True) for _ in range(4)], clock_hz=9216000
    )
    bank.release()
    first = bank.attenuation_samples(25)
    assert all(e.stage == 4 for e in bank.envelopes)
    assert first[-1, 0] == 192
    bank.release()
    assert all(e.stage == 4 for e in bank.envelopes)
    saved = TX81ZEnvelopeBank.model_validate_json(bank.model_dump_json())
    tail = bank.attenuation_samples(47975)
    actual = np.concatenate([first, tail])
    frames = np.arange(48000)
    counts = frames // 16
    after_threshold = np.maximum(counts - counts // 4 - 1, 0)
    expected = np.where(
        frames < 24, frames * 8, np.minimum(192 + after_threshold, 1023)
    )
    np.testing.assert_array_equal(actual[:, 0], expected)
    np.testing.assert_array_equal(saved.attenuation_samples(47975), tail)
    carrier = np.sin(2 * np.pi * frames * 440 / 48000)[:, None]
    check_audio(
        tmp_path / "tx81z-reverb-envelope.wav",
        carrier * 2 ** (-actual / 64),
        carrier * 2 ** (-expected[:, None] / 64) * np.ones((1, 4)),
    )


@pytest.mark.parametrize(
    "difference", ["voice", "additional", "note", "velocity", "sample_rate"]
)
def test_tx81z_note_snapshot_rejects_different_definition(difference: str) -> None:
    raw = bytearray(93)
    raw[62] = 24
    renderer = TX81ZRenderer(
        tx81z.TX81ZVoice(data=bytes(raw)),
        tx81z.TX81ZAdditional(data=bytes(23)),
        69,
        100,
    )
    saved = renderer.snapshot().model_dump()
    if difference == "voice":
        raw[53] = 1
        saved["voice"] = tx81z.TX81ZVoice(data=bytes(raw))
    elif difference == "additional":
        extra = bytearray(23)
        extra[3] = 1
        saved["additional"] = tx81z.TX81ZAdditional(data=bytes(extra))
    else:
        saved[difference] += 1
    with pytest.raises(EngineError, match="different note definition"):
        renderer.restore(TX81ZSnapshot.model_validate(saved))


@pytest.mark.parametrize(
    "detune,key_code,expected",
    [
        (0, 0, -2),
        (1, 0, -1),
        (2, 0, 0),
        (3, 31, 0),
        (4, 31, 8),
        (5, 31, 16),
        (6, 31, 22),
        (6, 18, 9),
        (0, 18, -9),
    ],
)
def test_tx81z_detune_preserves_sign_and_key_dependent_chip_steps(
    tmp_path: Path, detune: int, key_code: int, expected: int
) -> None:
    step = tx81z_detune_step(detune, key_code)
    assert step == expected
    frequency = 440 + step * (3579545 / 64) / 2**20
    frequencies = np.full((48000, 4), frequency)
    gains = np.zeros((48000, 4))
    gains[:, 0] = 0.05
    actual, _ = tx81z_operator_samples(8, [0] * 4, TX81ZState(), frequencies, gains)
    expected_hz = 440 + expected * (3579545 / 64) / 2**20
    audio = opz_carrier(expected_hz, 0.05)
    check_audio(tmp_path / "tx81z-detune.wav", actual, audio[:, None])


@pytest.mark.parametrize(
    "level,algorithm,operator,expected",
    [
        (99, 1, 1, 1),
        (91, 1, 1, 0.5),
        (20, 1, 1, 2 ** (-79 / 8)),
        (19, 1, 1, 2 ** (-81 / 8)),
        (1, 1, 1, 2 ** (-122 / 8)),
        (0, 1, 1, 2 ** (-127 / 8)),
        (99, 5, 1, 0.5),
        (99, 5, 3, 0.5),
        (99, 5, 2, 1),
        (99, 5, 4, 1),
        (99, 6, 3, 2 ** (-13 / 8)),
        (99, 7, 3, 2 ** (-13 / 8)),
        (99, 6, 4, 1),
        (99, 8, 4, 0.25),
    ],
)
def test_tx81z_output_level_curve_and_carrier_headroom(
    tmp_path: Path, level: int, algorithm: int, operator: int, expected: float
) -> None:
    raw = bytearray(93)
    raw[(4 - operator) * 13 + 10] = level
    gain = tx81z_output_gain(
        tx81z.TX81ZVoice(data=bytes(raw)).operator(operator), algorithm, 69, 127
    )
    assert gain == expected
    carrier = np.sin(2 * np.pi * np.arange(48000) * 440 / 48000)[:, None]
    check_audio(tmp_path / "tx81z-output-level.wav", carrier * gain, carrier * expected)


@pytest.mark.parametrize(
    "scaling,note,sensitivity,velocity,level,algorithm,attenuation",
    [
        (99, 0, 0, 127, 99, 1, 0),
        (99, 30, 0, 127, 99, 1, 0),
        (99, 31, 0, 127, 99, 1, 1),
        (99, 36, 0, 127, 99, 1, 1),
        (99, 37, 0, 127, 99, 1, 2),
        (50, 69, 0, 127, 99, 1, 7),
        (66, 84, 0, 127, 99, 1, 22),
        (1, 108, 0, 127, 99, 1, 0),
        (50, 108, 0, 127, 99, 1, 63),
        (99, 108, 0, 127, 99, 1, 127),
        (99, 109, 0, 127, 99, 1, 79),
        (99, 120, 0, 127, 99, 1, 127),
        (99, 121, 0, 127, 99, 1, 79),
        (99, 127, 0, 127, 99, 1, 111),
        (0, 69, 0, 1, 99, 1, 0),
        (0, 69, 0, 0, 99, 1, 0),
        (0, 69, 1, 0, 99, 1, 22),
        (0, 69, 2, 0, 99, 1, 37),
        (0, 69, 3, 0, 99, 1, 52),
        (0, 69, 4, 0, 99, 1, 67),
        (0, 69, 5, 0, 99, 1, 81),
        (0, 69, 6, 0, 99, 1, 96),
        (0, 69, 7, 0, 99, 1, 111),
        (99, 108, 7, 0, 0, 8, 127),
        (0, 69, 1, 1, 99, 1, 18),
        (0, 69, 7, 1, 99, 1, 84),
        (0, 69, 4, 2, 99, 1, 48),
        (0, 69, 7, 50, 99, 1, 28),
        (0, 69, 7, 100, 99, 1, 7),
        (0, 69, 1, 127, 99, 1, 6),
        (0, 69, 7, 127, 99, 1, 0),
        (50, 69, 7, 1, 91, 8, 115),
        (99, 108, 7, 1, 0, 8, 127),
    ],
)
def test_tx81z_total_level_combines_measured_key_and_velocity_attenuation(
    tmp_path: Path,
    scaling: int,
    note: int,
    sensitivity: int,
    velocity: int,
    level: int,
    algorithm: int,
    attenuation: int,
) -> None:
    raw = bytearray(93)
    raw[44], raw[48], raw[49] = scaling, sensitivity, level
    operator = tx81z.TX81ZVoice(data=bytes(raw)).operator(1)
    gain = tx81z_output_gain(operator, algorithm, note, velocity)
    expected = 2 ** (-attenuation / 8)
    assert gain == expected
    carrier = np.sin(2 * np.pi * np.arange(48000) * 440 / 48000)[:, None]
    check_audio(tmp_path / "tx81z-key-level.wav", carrier * gain, carrier * expected)


@pytest.mark.parametrize("velocity", [-1, 128])
def test_tx81z_velocity_scaling_rejects_out_of_range_velocities(velocity: int) -> None:
    raw = bytearray(93)
    raw[48] = 7
    operator = tx81z.TX81ZVoice(data=bytes(raw)).operator(1)
    with pytest.raises(EngineError, match="note-on velocity"):
        tx81z_output_gain(operator, 1, 69, velocity)


def test_tx81z_prepared_patch_components_render_and_release(tmp_path: Path) -> None:
    raw = bytearray(93)
    for i in range(0, 52, 13):
        raw[i] = 31
        raw[i + 3] = raw[i + 4] = 15
        raw[i + 10] = 99
        raw[i + 11] = 4
        raw[i + 12] = 3
    raw[52] = 7
    voice = tx81z.TX81ZVoice(data=bytes(raw))
    additional = tx81z.TX81ZAdditional(data=bytes(23))
    bank = TX81ZEnvelopeBank(
        envelopes=[
            TX81ZEnvelope.start(voice.operator(i), additional.operator(i), 0)
            for i in range(1, 5)
        ]
    )
    frequencies = np.tile(
        [
            tx81z_base_frequency(voice.operator(i), additional.operator(i), 69)
            for i in range(1, 5)
        ],
        (24000, 1),
    )
    levels = np.array(
        [
            tx81z_output_gain(voice.operator(i), voice.algorithm, 69, 127)
            for i in range(1, 5)
        ]
    )
    attack = 2 ** (-bank.attenuation_samples(24000) / 64) * levels
    first, state = tx81z_operator_samples(
        voice.algorithm, [0] * 4, TX81ZState(), frequencies, attack
    )
    bank.release()
    release = 2 ** (-bank.attenuation_samples(24000) / 64) * levels
    tail, _ = tx81z_operator_samples(
        voice.algorithm, [0] * 4, state, frequencies, release
    )
    ticks = np.arange(48000, dtype=np.int64) * bank.clock_hz // (192 * 48000)
    attenuation = np.minimum(np.maximum(ticks - ticks[24000], 0) * 8, 1023)
    expected = 4 * opz_carrier(opz_frequency(), 0.25 * 2 ** (-attenuation / 64))
    check_audio(
        tmp_path / "tx81z-paired-components.wav",
        np.concatenate([first, tail]),
        expected[:, None],
    )


@pytest.mark.parametrize("clock_hz", [3579545, 9216000])
def test_tx81z_envelope_clock_survives_partitioned_json_continuation(
    tmp_path: Path, clock_hz: int
) -> None:
    bank = TX81ZEnvelopeBank(
        envelopes=[
            TX81ZEnvelope(rates=[62, 62, 0, 62], sustain=0, shift=i, attenuation=0)
            for i in range(4)
        ],
        clock_hz=clock_hz,
    )
    bank.release()
    initial = bank.model_dump_json()
    actual = bank.attenuation_samples(48000)
    ticks = np.arange(48000, dtype=np.int64) * clock_hz // (192 * 48000)
    expected = np.broadcast_to(np.minimum(ticks[:, None] * 8, 1023), (48000, 4))
    np.testing.assert_array_equal(actual, expected)
    assert bank.counter == 48000 * clock_hz // (192 * 48000)
    split = TX81ZEnvelopeBank.model_validate_json(initial)
    first = split.attenuation_samples(97)
    split = TX81ZEnvelopeBank.model_validate_json(split.model_dump_json())
    tail = split.attenuation_samples(47903)
    np.testing.assert_array_equal(np.concatenate([first, tail]), actual)
    assert split == bank
    carrier = np.sin(2 * np.pi * np.arange(48000) * 440 / 48000)[:, None]
    check_audio(
        tmp_path / "tx81z-envelope-clock.wav",
        carrier * 2 ** (-actual / 64),
        carrier * 2 ** (-expected / 64),
    )


@pytest.mark.parametrize("feedback", [1, 7])
def test_tx81z_operator_bank_retains_two_feedback_samples(
    tmp_path: Path, feedback: int
) -> None:
    frequencies = np.full((48000, 4), 440.0)
    gains = np.zeros((48000, 4))
    gains[:, 3] = 0.05
    state = TX81ZState(feedback=[164, -82])
    expected = np.empty((48000, 1))
    older, newer = state.feedback
    waves, power = reference_tables()
    for i in range(48000):
        phase = (i * round(440 * 2**20 / 48000) >> 10) + (
            (older + newer) >> (10 - feedback)
        )
        entry = int(waves[0, phase & 1023])
        level = (entry & 32767) + 4 * round(-64 * np.log2(0.05))
        value = int(power[level & 255]) >> (level >> 8)
        if entry & 32768:
            value = -value
        expected[i, 0] = value / 8192
        older, newer = newer, value
    actual, final = tx81z_operator_samples(
        8, [0] * 4, state, frequencies, gains, feedback=feedback
    )
    check_audio(tmp_path / "tx81z-feedback.wav", actual, expected)
    first, saved = tx81z_operator_samples(
        8, [0] * 4, state, frequencies[:1001], gains[:1001], feedback=feedback
    )
    saved = TX81ZState.model_validate_json(saved.model_dump_json())
    tail, resumed = tx81z_operator_samples(
        8, [0] * 4, saved, frequencies[1001:], gains[1001:], feedback=feedback
    )
    np.testing.assert_array_equal(np.concatenate([first, tail]), actual)
    assert resumed == final
    assert state.feedback == [164, -82]


def test_tx81z_bank_uses_each_operators_selected_waveform(tmp_path: Path) -> None:
    waveforms = [0, 2, 4, 7]
    state = TX81ZState(phases=[104858, 209715, 314573, 419430])
    frequencies = np.tile([220, 330, 550, 770], (48000, 1)).astype(float)
    gains = np.tile([0.1, 0.2, 0.3, 0.4], (48000, 1))
    expected = sum(
        opz_carrier(
            frequencies[:, i], gains[:, i], waveform=waveforms[i], phase=state.phases[i]
        )
        for i in range(4)
    )
    actual, _ = tx81z_operator_samples(8, waveforms, state, frequencies, gains)
    check_audio(tmp_path / "tx81z-bank-waves.wav", actual, expected[:, None])


@pytest.mark.parametrize(
    "fixed,coarse,fine,frequency_range,note,hz",
    [
        (False, 4, 8, 0, 69, 659.9147728085518),
        (False, 0, 7, 0, 69, 412.42006331682205),
        (False, 1, 0, 0, 69, 311.0752469301224),
        (False, 13, 0, 0, 69, 1759.7727274894714),
        (False, 43, 0, 0, 69, 6221.5049386024475),
        (False, 44, 0, 0, 69, 6213.824068307877),
        (True, 0, 0, 0, 69, 8),
        (True, 3, 7, 0, 69, 15),
        (True, 3, 7, 7, 127, 1920),
        (True, 4, 8, 0, 69, 24),
        (True, 4, 0, 0, 57, 16),
        (True, 63, 15, 7, 127, 32640),
        (True, 32, 5, 3, 0, 1064),
    ],
)
def test_tx81z_base_frequency_preserves_ratio_and_fixed_controls(
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
@pytest.mark.parametrize("fixed", [False, True])
def test_tx81z_rejects_unavailable_low_coarse_fine_values(
    coarse: int, fixed: bool
) -> None:
    raw = bytearray(93)
    raw[50] = coarse
    extra = bytearray(23)
    extra[15] = int(fixed)
    extra[17] = 8
    with pytest.raises(EngineError, match="coarse values below four"):
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


@pytest.mark.parametrize("waveform", [-1, 8])
def test_tx81z_rejects_unknown_waveforms(waveform: int) -> None:
    with pytest.raises(EngineError, match="waveform"):
        tx81z_wave_samples(waveform, np.zeros(48000))


@pytest.mark.parametrize("feedback", [0, 1, 7])
@pytest.mark.parametrize("algorithm", [31, 32])
@pytest.mark.parametrize("amplitude_modulation", [False, True])
def test_dx7_feedback_uses_two_delayed_cycle_samples(
    tmp_path: Path,
    feedback: int,
    algorithm: int,
    amplitude_modulation: bool,
) -> None:
    data = bytearray(voice().data)
    data[135] = feedback
    data[134] = algorithm - 1
    data[14], data[140] = 3, 99 if amplitude_modulation else 0
    renderer = DX7Renderer(DX7Voice(data=bytes(data)), note=69, velocity=100)
    saved = renderer.snapshot()
    for envelope in saved.envelopes:
        envelope.level, envelope.stage, envelope.held = 9 * 2**24, 3, True
    renderer.restore(saved)
    expected = np.empty((48000, 1))
    previous = older = 0
    rows = np.column_stack([np.linspace(0, 1, 48000), np.ones(48000)])
    for i in range(48000):
        offset = (
            (older + previous) >> (9 - feedback + (2 if algorithm == 32 else 0))
            if feedback
            else 0
        )
        attenuation = int(52.75 * rows[i, 0] / 8 * 1024) if amplitude_modulation else 0
        phase = i * round(440 * 2**24 / 48000)
        sine = dx7_operator_sample(phase, 7168)
        value = dx7_operator_sample(phase + offset, 7168 + attenuation)
        expected[i, 0] = (
            5 * sine + value
            if algorithm == 32
            else 4 * sine + dx7_operator_sample(phase + value, 7168)
        ) / 2**24
        older, previous = previous, value
    actual = renderer.render(48000, rows if amplitude_modulation else None)
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


@pytest.mark.parametrize(
    "note,positive_hz",
    [(21, 27.788284414879204), (69, 441.73227187499987), (108, 4192.235613457401)],
)
@pytest.mark.parametrize("detune", [0, 7, 14])
@pytest.mark.parametrize("coarse,fine", [(0, 0), (3, 37)])
def test_ratio_detune_uses_key_dependent_log_frequency_fit(
    tmp_path: Path, note: int, positive_hz: float, detune: int, coarse: int, fine: int
) -> None:
    # Positive anchors were evaluated with 50-digit decimal arithmetic from
    # Dexed's continuous fit. Opposite detune has the reciprocal frequency ratio.
    neutral = 440 * 2 ** ((note - 69) / 12)
    frequency = {0: neutral**2 / positive_hz, 7: neutral, 14: positive_hz}[detune]
    frequency *= (0.5 if coarse == 0 else coarse) * (1 + fine / 100)
    data = bytearray(voice().data)
    for i in range(0, 126, 21):
        data[i + 16] = 67
        data[i + 18 : i + 21] = bytes([coarse, fine, detune])
    patch = DX7Voice(data=bytes(data))
    renderer = DX7Renderer(patch, note=note, velocity=100)
    first = renderer.render(1001)
    saved = DX7Snapshot.model_validate_json(renderer.snapshot().model_dump_json())
    tail = renderer.render(46999)
    envelope = DX7Envelope.start(patch.operator(1), note, 100, 48000)
    expected = np.empty(48000)
    for i in range(48000):
        envelope.sample()
        expected[i] = (
            6
            * dx7_operator_sample(
                i * round(frequency * 2**24 / 48000), 16384 - (envelope.level >> 14)
            )
            / 2**24
        )
    check_audio(
        tmp_path / "ratio-detune.wav", np.concatenate([first, tail]), expected[:, None]
    )
    renderer.restore(saved)
    np.testing.assert_array_equal(renderer.render(46999), tail)


def voice() -> DX7Voice:
    data = bytearray(155)
    for start in range(0, 126, 21):
        data[start : start + 4] = bytes([99, 99, 99, 99])
        data[start + 4 : start + 8] = bytes([99, 99, 99, 99])
        data[start + 16] = 99
        data[start + 18] = 1
        data[start + 20] = 7
    data[134] = 31
    data[144] = 24
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
        "output": (14 if packed else 16, 100),
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


@pytest.mark.parametrize("transpose", [0, 12])
def test_dx7_packed_bank_and_edit_buffer_render_the_same_note(
    tmp_path: Path, transpose: int
) -> None:
    unpacked = bytearray(voice().data)
    packed = bytearray(128)
    for i in range(6):
        start = i * 21
        unpacked[start : start + 21] = bytes(
            [99, 71, 33, 62, 99, 75, 61, 0, 18 + i, 7 + i, 14 + i]
            + [1, 2, i, 1, i + 1, 85 - 3 * i, i % 2, i + 1, 17 + i, 4 + i]
        )
        target = i * 17
        packed[target : target + 11] = unpacked[start : start + 11]
        packed[target + 11 : target + 17] = bytes(
            [
                1 | (2 << 2),
                i | ((4 + i) << 3),
                1 | ((i + 1) << 2),
                85 - 3 * i,
                (i % 2) | ((i + 1) << 1),
                17 + i,
            ]
        )
    unpacked[134:137] = bytes([15, 4, 1])
    unpacked[144] = 24 + transpose
    packed[102:111] = unpacked[126:135]
    packed[111] = 4 | (1 << 3)
    packed[117] = 24 + transpose
    edit_voice = DX7Voice(data=bytes(unpacked))
    bank_voice = DX7Voice(data=bytes(packed))
    assert bank_voice.operators == edit_voice.operators
    actual = DX7Renderer(bank_voice, 69, 93)
    expected = DX7Renderer(edit_voice, 69, 93)
    first = actual.render(24000)
    held = expected.render(24000)
    assert np.max(np.abs(first)) > 0
    actual.release()
    expected.release()
    check_audio(
        tmp_path / "dx7-packed-bank.wav",
        np.concatenate([first, actual.render(24000)]),
        np.concatenate([held, expected.render(24000)]),
    )
    assert bank_voice.data == bytes(packed)
    assert edit_voice.data == bytes(unpacked)


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


@pytest.mark.parametrize("synchronize", [False, True])
def test_dx7_initial_phases_obey_sync_and_remain_owned(
    tmp_path: Path, synchronize: bool
) -> None:
    raw = bytearray(voice().data)
    raw[136] = int(synchronize)
    patch = DX7Voice(data=bytes(raw))
    phases = [0.125, 0.25, 0.375, 0.5, 0.625, 0.75]
    renderer = DX7Renderer(patch, 69, 100, initial_phases=phases)
    reference = DX7Renderer(patch, 69, 100)
    if not synchronize:
        reference.restore(
            reference.snapshot().model_copy(
                update={"phases": [int(p * 2**24) for p in phases]}
            )
        )
    phases[:] = [0.0] * 6
    head = renderer.render(12001)
    saved = DX7Snapshot.model_validate_json(renderer.snapshot().model_dump_json())
    renderer.restore(saved)
    actual = np.concatenate([head, renderer.render(35999)])
    check_audio(tmp_path / "initial-phases.wav", actual, reference.render(48000))


@pytest.mark.parametrize(
    "phases",
    [
        [0.0] * 5,
        [0.0] * 7,
        [float("nan")] * 6,
        [float("inf")] * 6,
        [-0.1] * 6,
        [1.0] * 6,
    ],
)
def test_dx7_rejects_invalid_initial_phases(phases: list[float]) -> None:
    with pytest.raises(EngineError, match="initial phases"):
        DX7Renderer(voice(), 69, 100, initial_phases=phases)


@pytest.mark.parametrize("algorithm", [4, 6])
@pytest.mark.parametrize("feedback", [1, 7])
def test_dx7_feedback_returns_from_end_of_operator_chain(
    tmp_path: Path, algorithm: int, feedback: int
) -> None:
    data = bytearray(voice().data)
    data[134], data[135] = algorithm - 1, feedback
    renderer = DX7Renderer(DX7Voice(data=bytes(data)), 69, 100)
    saved = renderer.snapshot()
    for envelope in saved.envelopes:
        envelope.level, envelope.stage, envelope.held = 9 * 2**24, 3, True
    renderer.restore(saved)
    expected = np.empty((48000, 1))
    older = previous = 0
    for i in range(48000):
        phase = i * round(440 * 2**24 / 48000)
        unmodulated = dx7_operator_sample(phase, 7168)
        pair = dx7_operator_sample(phase + unmodulated, 7168)
        returned = dx7_operator_sample(
            phase + ((older + previous) >> (11 - feedback)), 7168
        )
        returned = dx7_operator_sample(phase + returned, 7168)
        if algorithm == 4:
            returned = dx7_operator_sample(phase + returned, 7168)
            other = dx7_operator_sample(phase + pair, 7168)
        else:
            other = 2 * pair
        expected[i, 0] = (returned + other) / 2**24
        older, previous = previous, returned
    head = renderer.render(12001)
    renderer.restore(
        DX7Snapshot.model_validate_json(renderer.snapshot().model_dump_json())
    )
    actual = np.concatenate([head, renderer.render(35999)])
    check_audio(tmp_path / "multi-operator-feedback.wav", actual, expected)
