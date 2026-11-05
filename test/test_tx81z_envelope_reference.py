from pathlib import Path

import numpy as np
import pytest
from test_synth import check_audio
from ufor import tx81z

from enge.yamaha import TX81ZEnvelope, TX81ZEnvelopeBank


@pytest.mark.parametrize("case", range(116))
@pytest.mark.parametrize("native_ticks", [False, True])
def test_tx81z_envelope_matches_pinned_ymfm_through_release(
    tmp_path: Path, case: int, native_ticks: bool
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
    indices = np.arange(48001) * 3579545 // (64 * 48000)
    release_at = int(indices[24000]) if native_ticks else 24000
    total = int(indices[-1]) if native_ticks else 48000
    observe = bank.attenuation_ticks if native_ticks else bank.attenuation_samples
    initial = [e.attenuation for e in bank.envelopes]
    first = observe(release_at)
    bank.release()
    saved = TX81ZEnvelopeBank.model_validate_json(bank.model_dump_json())
    length = total - release_at
    tail = np.concatenate([observe(137), observe(length - 137)])
    raw_rows = np.concatenate([first, tail])
    if native_ticks:
        raw_rows = np.vstack([initial, raw_rows])[indices[:-1]]
    actual = raw_rows >> np.arange(4)
    np.testing.assert_array_equal(actual, expected)
    restored = saved.attenuation_ticks if native_ticks else saved.attenuation_samples
    np.testing.assert_array_equal(restored(length), tail)
    assert saved == bank
    carrier = np.sin(2 * np.pi * np.arange(48000) * 440 / 48000)[:, None]
    check_audio(
        tmp_path / "tx81z-reference-envelope.wav",
        carrier * 2 ** (-actual / 64),
        carrier * 2 ** (-expected / 64),
    )
