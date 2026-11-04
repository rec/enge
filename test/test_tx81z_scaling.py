from pathlib import Path

import numpy as np
import pytest
from test_synth import check_audio
from ufor import tx81z

from enge.synth import EngineError
from enge.yamaha import TX81ZRenderer


@pytest.mark.parametrize(
    "note,scaling,sensitivity,velocity,attenuation",
    [
        (69, 0, 1, 127, 6),
        (69, 0, 7, 1, 84),
        (69, 0, 7, 100, 7),
        (69, 50, 7, 1, 91),
        (108, 99, 7, 1, 211),
        (31, 99, 0, 100, 1),
    ],
)
def test_tx81z_note_applies_key_and_velocity_attenuation_before_saturation(
    tmp_path: Path,
    note: int,
    scaling: int,
    sensitivity: int,
    velocity: int,
    attenuation: int,
) -> None:
    raw = bytearray(93)
    for i in range(0, 52, 13):
        raw[i], raw[i + 3], raw[i + 4] = 31, 15, 15
        raw[i + 11], raw[i + 12] = 4, 3
    raw[44], raw[48], raw[49] = scaling, sensitivity, 99
    raw[52], raw[62] = 7, 24
    voice = tx81z.TX81ZVoice(data=bytes(raw))
    additional = tx81z.TX81ZAdditional(data=bytes(23))
    actual = TX81ZRenderer(voice, additional, note, velocity).render(48000)
    frequency = 440 * 2 ** ((note - 69) / 12)
    # Algorithm 8 adds 16 TL to each carrier. The other three operators have OUT 0.
    gain = 2 ** (-min(127, 16 + attenuation) / 8) + 3 * 2 ** (-127 / 8)
    expected = gain * np.sin(2 * np.pi * frequency * np.arange(48000) / 48000)
    check_audio(tmp_path / "tx81z-note-scaling.wav", actual, expected[:, None])
    assert voice.data == bytes(raw)


@pytest.mark.parametrize("transpose", [23, 25])
def test_tx81z_key_level_scaling_rejects_non_neutral_transpose(
    transpose: int,
) -> None:
    raw = bytearray(93)
    raw[44], raw[62] = 50, transpose
    with pytest.raises(EngineError, match="key-level scaling with non-neutral"):
        TX81ZRenderer(
            tx81z.TX81ZVoice(data=bytes(raw)),
            tx81z.TX81ZAdditional(data=bytes(23)),
            69,
            100,
        )


def test_tx81z_velocity_scaling_requires_positive_note_on_velocity() -> None:
    raw = bytearray(93)
    raw[48], raw[62] = 7, 24
    with pytest.raises(EngineError, match="note-on velocity"):
        TX81ZRenderer(
            tx81z.TX81ZVoice(data=bytes(raw)),
            tx81z.TX81ZAdditional(data=bytes(23)),
            69,
            0,
        )
