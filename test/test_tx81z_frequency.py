from pathlib import Path

import numpy as np
import pytest
from test_synth import check_audio
from test_tx81z_integer import native_carrier
from ufor import tx81z

from enge.yamaha import TX81ZRenderer, tx81z_base_frequency


def opz_frequency(
    note: int = 69, coarse: int = 4, fine: int = 0, detune: int = 3
) -> float:
    """Unmodulated Hz from the pinned register export, including DT1 rounding."""
    return float(_BASE[note - 13, coarse, fine, detune]) * (3579545 / (64 * 2**20))


def opz_pm_frequency(delta: np.ndarray, note: int = 69, detune: int = 3) -> np.ndarray:
    """Multiplier-one Hz from the pinned native key-code helper export."""
    return _PM[note - 13, detune, delta.astype(np.int64) + 512] * (
        3579545 / (64 * 2**20)
    )


@pytest.mark.parametrize("coarse", range(64))
def test_ratio_base_frequencies_match_all_legal_reference_controls(
    tmp_path: Path, coarse: int
) -> None:
    raw = bytearray(93)
    raw[50] = coarse
    operator = tx81z.TX81ZVoice(data=bytes(raw)).operator(1)
    actual, expected = [], []
    for fine in range(8 if coarse < 4 else 16):
        extra = bytearray(23)
        extra[17] = fine
        additional = tx81z.TX81ZAdditional(data=bytes(extra)).operator(1)
        for note in range(13, 109):
            actual.append(tx81z_base_frequency(operator, additional, note))
            expected.append(opz_frequency(note, coarse, fine))
    np.testing.assert_array_equal(actual, expected)
    repeats = (48000 + len(actual) - 1) // len(actual)
    actual_phase = np.cumsum(np.repeat(actual, repeats)) / 48000
    expected_phase = np.cumsum(np.repeat(expected, repeats)) / 48000
    check_audio(
        tmp_path / "native-ratios.wav",
        np.sin(2 * np.pi * actual_phase)[:, None],
        np.sin(2 * np.pi * expected_phase)[:, None],
    )


@pytest.mark.parametrize(
    "note,coarse,fine,detune",
    [
        (13, 0, 3, 0),
        (13, 0, 7, 6),
        (69, 44, 15, 0),
        (69, 43, 8, 6),
        (108, 3, 7, 0),
        (108, 63, 15, 6),
    ],
)
def test_ratio_renderer_adds_detune_before_multiplier_rounding(
    tmp_path: Path, note: int, coarse: int, fine: int, detune: int
) -> None:
    raw, extra = bytearray(93), bytearray(23)
    for i in range(4):
        raw[i * 13], raw[i * 13 + 4] = 31, 15
        raw[i * 13 + 10 : i * 13 + 13] = bytes([99, coarse, detune])
        extra[i * 5 + 2] = fine
    raw[52], raw[62] = 7, 24
    voice = tx81z.TX81ZVoice(data=bytes(raw))
    additional = tx81z.TX81ZAdditional(data=bytes(extra))
    actual = TX81ZRenderer(voice, additional, note, 127).render(48000)
    expected = (4 * native_carrier(opz_frequency(note, coarse, fine, detune)))[:, None]
    check_audio(tmp_path / "native-detune.wav", actual, expected)


@pytest.mark.parametrize("note", [13, 108])
@pytest.mark.parametrize("detune", [0, 6])
def test_pitch_modulation_uses_native_table_and_saturates_before_detune(
    tmp_path: Path, note: int, detune: int
) -> None:
    raw = bytearray(93)
    for i in range(4):
        raw[i * 13], raw[i * 13 + 4] = 31, 15
        raw[i * 13 + 10 : i * 13 + 13] = bytes([99, 4, detune])
    raw[52], raw[56], raw[60], raw[62] = 7, 99, 7, 24
    voice = tx81z.TX81ZVoice(data=bytes(raw))
    additional = tx81z.TX81ZAdditional(data=bytes(23))
    wave = np.resize(np.arange(-128, 128, dtype=np.int64), 48000)
    rows = np.column_stack([wave, np.zeros(48000), np.ones((48000, 2))])
    renderer = TX81ZRenderer(voice, additional, note, 127)
    first = renderer.render(17003, rows[:17003])
    saved = renderer.snapshot()
    tail = renderer.render(30997, rows[17003:])
    expected = 4 * native_carrier(opz_pm_frequency((wave * 127 >> 7) * 4, note, detune))
    check_audio(
        tmp_path / "native-pm.wav", np.concatenate([first, tail]), expected[:, None]
    )
    renderer.restore(saved)
    check_audio(
        tmp_path / "native-pm-replay.wav",
        np.concatenate([first, renderer.render(30997, rows[17003:])]),
        expected[:, None],
    )


with np.load(Path(__file__).with_name("data") / "tx81z-frequency.npz") as _reference:
    _BASE = _reference["base"]
    _PM = _reference["pm"]
