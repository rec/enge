from pathlib import Path

import numpy as np
import pytest
from test_synth import check_audio
from test_tx81z_integer import opz_carrier
from test_tx81z_lfo import lfo_voice
from ufor import tx81z

from enge.yamaha import TX81ZRenderer, TX81ZSnapshot


@pytest.mark.parametrize("attenuation", [896, 897, 1023])
def test_quiet_gate_uses_raw_attenuation_before_envelope_shift(
    tmp_path: Path, attenuation: int
) -> None:
    raw = bytearray(lfo_voice().data)
    raw[56:58] = bytes(2)
    note = TX81ZRenderer(
        tx81z.TX81ZVoice(data=bytes(raw)),
        tx81z.TX81ZAdditional(data=bytes(23)),
        69,
        127,
    )
    for envelope in note.envelopes.envelopes:
        envelope.attenuation = 1023
        envelope.stage = 2
        envelope.rates = [0] * 4
    operator = note.envelopes.envelopes[1]
    operator.attenuation = attenuation
    operator.shift = 2
    gain = 0.25 * 2 ** (-(attenuation >> 2) / 64) if attenuation <= 896 else 0
    check_audio(
        tmp_path / "raw-quiet-gate.wav",
        note.render(48000),
        opz_carrier(440, gain)[:, None],
    )


@pytest.mark.parametrize("case", range(32))
def test_note_envelope_and_integer_core_match_pinned_opz(
    tmp_path: Path, case: int
) -> None:
    with np.load(Path(__file__).parent / "data/tx81z-opz-dynamic.npz") as reference:
        algorithm, feedback, profile = map(int, reference["parameters"][case])
        expected = reference[f"case{case}"].astype(float)[:, None] / 8192
    raw, extra = bytearray(93), bytearray(23)
    for i in range(4):
        raw[i * 13 : i * 13 + 5] = bytes(
            [20, 16, 12, 15, 0] if profile else [31, 12, 4, 9, 7]
        )
        raw[i * 13 + 10 : i * 13 + 13] = bytes([99, 4, 3])
        extra[i * 5 + 3] = [0, 1, 4, 7][i] if profile else 0
        extra[i * 5 + 4] = 3 - i if profile and i != 0 else 0
    raw[52], raw[53], raw[62] = algorithm - 1, feedback, 24
    note = TX81ZRenderer(
        tx81z.TX81ZVoice(data=bytes(raw)),
        tx81z.TX81ZAdditional(data=bytes(extra)),
        69,
        127,
        pitch_hz=9600 * 48000 / 2**20,
    )
    first = note.render(17003)
    saved = TX81ZSnapshot.model_validate_json(note.snapshot().model_dump_json())
    middle = note.render(6997)
    note.release()
    tail = note.render(24000)
    actual = np.concatenate([first, middle, tail])
    check_audio(tmp_path / "dynamic-opz.wav", actual, expected)
    np.testing.assert_array_equal(actual, expected)
    note.restore(saved)
    replay = [first, note.render(6997)]
    note.release()
    replay.extend([note.render(137), note.render(23863)])
    np.testing.assert_array_equal(np.concatenate(replay), expected)
