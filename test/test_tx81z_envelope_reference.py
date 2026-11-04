from pathlib import Path

import numpy as np
import pytest
from test_synth import check_audio
from ufor import tx81z

from enge.yamaha import TX81ZEnvelope, TX81ZEnvelopeBank


@pytest.mark.parametrize("case", range(116))
def test_tx81z_envelope_matches_pinned_ymfm_through_release(
    tmp_path: Path, case: int
) -> None:
    with np.load(Path(__file__).parent / "data" / "tx81z-envelope.npz") as reference:
        attack, decay, sustain, release, level, scaling, key, reverb = map(
            int, reference["parameters"][case]
        )
        expected = reference[f"case{case}"].copy()
    raw = bytearray(93)
    extra = bytearray(23)
    for i in range(4):
        start = (3 - i) * 13
        raw[start : start + 5] = bytes([attack, decay, sustain, release, level])
        raw[start + 6] = scaling
        extra[(3 - i) * 5 + 4] = i
    voice = tx81z.TX81ZVoice(data=bytes(raw))
    additional = tx81z.TX81ZAdditional(data=bytes(extra))
    bank = TX81ZEnvelopeBank(
        envelopes=[
            TX81ZEnvelope.start(voice.operator(i), additional.operator(i), key, reverb)
            for i in range(1, 5)
        ]
    )
    first = bank.attenuation_samples(24000)
    bank.release()
    saved = TX81ZEnvelopeBank.model_validate_json(bank.model_dump_json())
    tail = np.concatenate(
        [bank.attenuation_samples(137), bank.attenuation_samples(23863)]
    )
    actual = np.concatenate([first, tail]) >> np.arange(4)
    np.testing.assert_array_equal(actual, expected)
    np.testing.assert_array_equal(saved.attenuation_samples(24000), tail)
    carrier = np.sin(2 * np.pi * np.arange(48000) * 440 / 48000)[:, None]
    check_audio(
        tmp_path / "tx81z-reference-envelope.wav",
        carrier * 2 ** (-actual / 64),
        carrier * 2 ** (-expected / 64),
    )
