from hashlib import sha256
from pathlib import Path

import numpy as np
import pytest
from test_synth import check_audio
from ufor import tx81z

from enge import yamaha
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
@pytest.mark.parametrize("transpose", [-12, 0, 12])
def test_tx81z_note_applies_key_and_velocity_attenuation_before_saturation(
    tmp_path: Path,
    note: int,
    scaling: int,
    sensitivity: int,
    velocity: int,
    attenuation: int,
    transpose: int,
) -> None:
    raw = bytearray(93)
    for i in range(0, 52, 13):
        raw[i], raw[i + 3], raw[i + 4] = 31, 15, 15
        raw[i + 11], raw[i + 12] = 4, 3
    raw[44], raw[48], raw[49] = scaling, sensitivity, 99
    raw[52], raw[62] = 7, 24 + transpose
    voice = tx81z.TX81ZVoice(data=bytes(raw))
    additional = tx81z.TX81ZAdditional(data=bytes(23))
    actual = TX81ZRenderer(voice, additional, note - transpose, velocity).render(48000)
    frequency = 440 * 2 ** ((note - 69) / 12)
    # Algorithm 8 adds 16 TL to each carrier. The other three operators have OUT 0.
    gain = 2 ** (-min(127, 16 + attenuation) / 8) + 3 * 2 ** (-127 / 8)
    expected = gain * np.sin(2 * np.pi * frequency * np.arange(48000) / 48000)
    check_audio(tmp_path / "tx81z-note-scaling.wav", actual, expected[:, None])
    assert voice.data == bytes(raw)


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


@pytest.mark.parametrize("sensitivity,output", [(7, 99), (0, 99), (7, 0)])
@pytest.mark.parametrize("foot", [False, True])
def test_tx81z_breath_uses_measured_attenuation_before_saturation(
    tmp_path: Path, sensitivity: int, output: int, foot: bool
) -> None:
    raw = bytearray(93)
    for i in range(0, 52, 13):
        raw[i], raw[i + 3], raw[i + 4] = 31, 15, 15
        raw[i + 7], raw[i + 11], raw[i + 12] = sensitivity, 4, 3
    raw[49], raw[52], raw[62], raw[76] = output, 7, 24, 99
    raw[67] = 99 if foot else 0
    voice = tx81z.TX81ZVoice(data=bytes(raw))
    additional = tx81z.TX81ZAdditional(data=bytes(23))
    # Measured row BC EG Bias 99, EBS 7, CC2 0..7 and 127.
    cc = np.repeat([0, 1, 2, 3, 4, 5, 6, 7, 127], [5000] * 8 + [8000])
    bias = np.repeat([127, 112, 96, 86, 80, 74, 70, 67, 0], [5000] * 8 + [8000])
    if not sensitivity:
        bias = np.zeros(48000)
    gain = 2 ** (
        -np.minimum(127, (16 if output else 127) + bias + (15 if foot else 0)) / 8
    )
    gain += 3 * 2 ** (-127 / 8)
    expected = gain * np.sin(2 * np.pi * 440 * np.arange(48000) / 48000)
    note = TX81ZRenderer(voice, additional, 69, 127)
    actual = note.render(
        48000, breath=(cc + 0.25 * (cc < 127)) / 127, foot=np.full(48000, 64 / 127)
    )
    check_audio(tmp_path / "breath.wav", actual, expected[:, None])
    default = TX81ZRenderer(voice, additional, 69, 127).render(48000)
    explicit = TX81ZRenderer(voice, additional, 69, 127).render(
        48000, breath=np.zeros(48000)
    )
    check_audio(tmp_path / "breath-default.wav", default, explicit)


@pytest.mark.parametrize("algorithm", [1, 8])
@pytest.mark.parametrize(
    "amount,attenuation",
    [(0, [0, 0, 0, 0, 0]), (50, [16, 15, 15, 6, 0]), (99, [127, 102, 90, 15, 0])],
)
def test_tx81z_foot_range_attenuates_only_carriers(
    tmp_path: Path, algorithm: int, amount: int, attenuation: list[int]
) -> None:
    raw = bytearray(93)
    for i in range(0, 52, 13):
        raw[i], raw[i + 3], raw[i + 4] = 31, 15, 15
        raw[i + 10], raw[i + 11], raw[i + 12] = 99, 4, 3
    raw[52], raw[62] = algorithm - 1, 24
    additional = tx81z.TX81ZAdditional(data=bytes(23))
    baseline = TX81ZRenderer(
        tx81z.TX81ZVoice(data=bytes(raw)), additional, 69, 127
    ).render(48000)
    raw[67] = amount
    voice = tx81z.TX81ZVoice(data=bytes(raw))
    foot = np.repeat([0, 1.25 / 127, 2.25 / 127, 64.25 / 127, 1], 9600)
    base_tl = 16 if algorithm == 8 else 0
    change = np.minimum(127, base_tl + np.repeat(attenuation, 9600)) - base_tl
    expected = baseline * 2 ** (-change[:, None] / 8)
    note = TX81ZRenderer(voice, additional, 69, 127)
    before = note.render(17003, foot=foot[:17003])
    saved = note.snapshot()
    after = note.render(30997, foot=foot[17003:])
    check_audio(tmp_path / "foot-range.wav", np.concatenate([before, after]), expected)
    note.restore(saved)
    replay = note.render(30997, foot=foot[17003:])
    check_audio(
        tmp_path / "foot-replay.wav", np.concatenate([before, replay]), expected
    )
    default = TX81ZRenderer(voice, additional, 69, 127).render(48000)
    explicit = TX81ZRenderer(voice, additional, 69, 127).render(
        48000, foot=np.zeros(48000)
    )
    check_audio(tmp_path / "foot-default.wav", default, explicit)


@pytest.mark.parametrize("control", ["breath", "foot"])
@pytest.mark.parametrize(
    "values", [np.zeros((1, 1)), np.array([-0.1]), np.array([1.1]), np.array([np.nan])]
)
def test_tx81z_invalid_controller_does_not_advance_note(
    control: str, values: np.ndarray
) -> None:
    raw = bytearray(93)
    raw[62] = 24
    note = TX81ZRenderer(
        tx81z.TX81ZVoice(data=bytes(raw)),
        tx81z.TX81ZAdditional(data=bytes(23)),
        69,
        127,
    )
    before = note.snapshot()
    with pytest.raises(EngineError, match=f"{control} rows"):
        if control == "breath":
            note.render(1, breath=values)
        else:
            note.render(1, foot=values)
    assert note.snapshot() == before


def test_tx81z_breath_asset_preserves_complete_measured_table() -> None:
    table = np.load(
        Path(yamaha.__file__).with_name("yamaha_breath.npy"), allow_pickle=False
    )
    assert table.shape == (100, 8, 128)
    assert table.dtype == np.uint8
    assert sha256(table.tobytes()).hexdigest() == (
        "80910f5f9803def7904843145433f9428407cd2395cc72ff2da3c8779f3da3ee"
    )
    assert not np.any(table[0])
    assert not np.any(table[:, 0])
    assert not np.any(table[:, :, 127])
    assert np.all(np.diff(table.astype(int), axis=2) <= 0)
