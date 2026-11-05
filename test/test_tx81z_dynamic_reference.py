from pathlib import Path

import numpy as np
import pytest
from test_synth import check_audio
from test_tx81z_frequency import opz_frequency
from test_tx81z_integer import native_carrier
from test_tx81z_lfo import lfo_voice
from ufor import tx81z

from enge.synth import EngineError
from enge.yamaha import TX81ZRenderer, TX81ZSnapshot, tx81z_dac
from enge.yamaha_lfo import TX81ZLFOContext


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
        native_carrier(opz_frequency(), gain)[:, None],
    )


@pytest.mark.parametrize(
    "algorithm,feedback,profile",
    [
        (a, f, p)
        for a in range(1, 9)
        for f in range(8)
        for p in range(6)
        if p < 2 or f in (0, 7)
    ],
)
def test_note_envelope_and_integer_core_match_pinned_opz(
    tmp_path: Path, algorithm: int, feedback: int, profile: int
) -> None:
    with np.load(Path(__file__).parent / "data/tx81z-ym2414-notes.npz") as reference:
        expected = (
            reference[f"a{algorithm}f{feedback}p{profile}"].astype(float)[:, None]
            / 32768
        )
    raw, extra = bytearray(93), bytearray(23)
    for i in range(4):
        raw[i * 13 : i * 13 + 5] = bytes(
            [20, 16, 0, 15, 10] if profile else [31, 12, 4, 9, 7]
        )
        raw[i * 13 + 10 : i * 13 + 13] = bytes(
            [99, 4 + i if profile else 4, [0, 2, 4, 6][i] if profile else 3]
        )
        if profile:
            extra[i * 5 + 2 : i * 5 + 5] = bytes(
                [[0, 3, 8, 15][i], [0, 1, 4, 7][i], [0, 2, 1, 0][i]]
            )
    raw[52], raw[53], raw[62] = algorithm - 1, feedback, 24
    if profile >= 2:
        waveform = profile - 2
        full = waveform >= 2
        raw[54:62] = bytes(
            [
                75,
                0,
                99 if full else 50,
                99 if full else 50,
                1,
                waveform,
                7 if full else 4,
                3 if full else 2,
            ]
        )
        for i in range(4):
            raw[i * 13 + 8] = int(i % 2 == 0)
    note = TX81ZRenderer(
        tx81z.TX81ZVoice(data=bytes(raw)),
        tx81z.TX81ZAdditional(data=bytes(extra)),
        69,
        127,
    )
    clock = TX81ZLFOContext(note.voice) if profile >= 2 else None
    if clock:
        clock.keydown()
    first = note.render(17003, clock.advance_ticks(17003) if clock else None)
    saved_clock = clock.snapshot() if clock else None
    saved = TX81ZSnapshot.model_validate_json(note.snapshot().model_dump_json())
    middle = note.render(6997, clock.advance_ticks(6997) if clock else None)
    note.release()
    tail = note.render(24000, clock.advance_ticks(24000) if clock else None)
    actual = tx81z_dac(np.concatenate([first, middle, tail]))
    check_audio(tmp_path / "dynamic-opz.wav", actual, expected)
    np.testing.assert_array_equal(actual, expected)
    note.restore(saved)
    if clock and saved_clock:
        clock.restore(saved_clock)
    replay = [first, note.render(6997, clock.advance_ticks(6997) if clock else None)]
    note.release()
    replay.extend(
        [
            note.render(n, clock.advance_ticks(n) if clock else None)
            for n in (137, 23863)
        ]
    )
    np.testing.assert_array_equal(tx81z_dac(np.concatenate(replay)), expected)


def test_note_holds_output_and_restores_between_chip_ticks(tmp_path: Path) -> None:
    raw = bytearray(lfo_voice().data)
    raw[56:58] = bytes(2)
    voice, extra = (
        tx81z.TX81ZVoice(data=bytes(raw)),
        tx81z.TX81ZAdditional(data=bytes(23)),
    )
    note = TX81ZRenderer(voice, extra, 69, 127, 96000)
    first = note.render(2)
    assert first[0, 0] == 0 and first[1, 0] != 0
    saved = TX81ZSnapshot.model_validate_json(note.snapshot().model_dump_json())
    with pytest.raises(EngineError, match="tick grid"):
        note.render(1, (np.array([1]), np.zeros((2, 4))))
    with pytest.raises(EngineError, match="envelope clock"):
        note.restore(saved.model_copy(update={"frame": 0}))
    assert note.snapshot() == saved
    held = note.render(1)
    assert held[0, 0] == first[1, 0] and note.operators == saved.operators
    note.restore(saved)
    tail = note.render(95998)
    expected = TX81ZRenderer(voice, extra, 69, 127).render(48000)
    check_audio(
        tmp_path / "held-native-output.wav",
        np.concatenate([first, tail])[1::2],
        expected,
    )
