from pathlib import Path

import numpy as np
import pytest
from test_synth import check_audio
from ufor import tx81z

from enge.synth import EngineError
from enge.yamaha import TX81ZRenderer, TX81ZSnapshot, tx81z_lfo_registers
from enge.yamaha_lfo import TX81ZLFOContext, TX81ZLFOSnapshot


def lfo_voice(waveform: int = 0) -> tx81z.TX81ZVoice:
    raw = bytearray(93)
    for i in range(0, 52, 13):
        raw[i], raw[i + 3], raw[i + 4] = 31, 15, 15
        raw[i + 8], raw[i + 10], raw[i + 11], raw[i + 12] = 1, 99, 4, 3
    raw[52], raw[54], raw[56], raw[57] = 7, 75, 99, 99
    raw[58], raw[59], raw[60], raw[61], raw[62] = 1, waveform, 7, 3, 24
    return tx81z.TX81ZVoice(data=bytes(raw))


@pytest.mark.parametrize("waveform", range(4))
def test_tx81z_lfo_matches_pinned_ymfm_rows_and_restores_noise(
    tmp_path: Path, waveform: int
) -> None:
    voice = lfo_voice(waveform)
    with np.load(Path(__file__).with_name("data") / "tx81z-lfo.npz") as reference:
        expected = reference[f"wave{waveform}"].astype(float)
    context = TX81ZLFOContext(voice)
    first = context.advance(17003)
    saved = TX81ZLFOSnapshot.model_validate_json(context.snapshot().model_dump_json())
    saved_json = saved.model_dump_json()
    restored = TX81ZLFOContext(voice)
    restored.restore(saved)
    actual = np.concatenate([first, restored.advance(30997)])
    check_audio(tmp_path / "tx81z-ymfm-lfo.wav", actual.astype(float), expected)
    assert saved.model_dump_json() == saved_json
    context.advance(30997)
    assert context.snapshot() == restored.snapshot()
    assert context.snapshot().frame == 48000


@pytest.mark.parametrize(
    "speed,pitch,amplitude,waveform,expected",
    [
        (0, 0, 0, 0, (0, 0, 0)),
        (1, 1, 4, 0, (31, 0, 0)),
        (50, 50, 50, 0, (223, 63, 16)),
        (99, 99, 99, 0, (255, 127, 127)),
        (75, 50, 98, 3, (193, 63, 96)),
    ],
)
def test_tx81z_measured_lfo_register_mapping(
    speed: int,
    pitch: int,
    amplitude: int,
    waveform: int,
    expected: tuple[int, int, int],
) -> None:
    raw = bytearray(lfo_voice(waveform).data)
    raw[54], raw[56], raw[57] = speed, pitch, amplitude
    assert tx81z_lfo_registers(tx81z.TX81ZVoice(data=bytes(raw))) == expected


def test_tx81z_pitch_modulation_bypasses_fixed_operators(tmp_path: Path) -> None:
    voice = lfo_voice()
    extra = bytearray(23)
    extra[0], extra[1] = 1, 2
    additional = tx81z.TX81ZAdditional(data=bytes(extra))
    rows = np.tile([96, 0], (48000, 1))
    actual = TX81ZRenderer(voice, additional, 69, 100).render(48000, rows)
    raw = bytearray(voice.data)
    raw[56], raw[57] = 0, 0
    expected = TX81ZRenderer(
        tx81z.TX81ZVoice(data=bytes(raw)), additional, 75, 100
    ).render(48000)
    check_audio(tmp_path / "tx81z-lfo-fixed.wav", actual, expected)


def test_tx81z_pitch_modulation_keeps_fine_detune_outside_the_ratio(
    tmp_path: Path,
) -> None:
    raw = bytearray(lfo_voice().data)
    for i in range(0, 52, 13):
        raw[i + 12] = 6
    voice = tx81z.TX81ZVoice(data=bytes(raw))
    rows = np.tile([96, 0], (48000, 1))
    actual = TX81ZRenderer(
        voice, tx81z.TX81ZAdditional(data=bytes(23)), 69, 100
    ).render(48000, rows)
    frequency = 440 * 2**0.5 + 9 * 3579545 / (64 * 2**20)
    expected = np.sin(2 * np.pi * frequency * np.arange(48000) / 48000)[:, None]
    check_audio(tmp_path / "tx81z-lfo-detune.wav", actual, expected)


@pytest.mark.parametrize("sensitivity", range(4))
@pytest.mark.parametrize("enabled", [False, True])
def test_tx81z_amplitude_modulation_obeys_channel_sensitivity_and_operator_enable(
    tmp_path: Path, sensitivity: int, enabled: bool
) -> None:
    raw = bytearray(lfo_voice().data)
    raw[56], raw[61] = 0, sensitivity
    for i in range(0, 52, 13):
        raw[i + 8] = int(enabled)
    voice = tx81z.TX81ZVoice(data=bytes(raw))
    additional = tx81z.TX81ZAdditional(data=bytes(23))
    rows = np.tile([0, 64], (48000, 1))
    actual = TX81ZRenderer(voice, additional, 69, 100).render(48000, rows)
    raw[57] = 0
    baseline = TX81ZRenderer(
        tx81z.TX81ZVoice(data=bytes(raw)), additional, 69, 100
    ).render(48000)
    attenuation = 64 << (sensitivity - 1) if sensitivity and enabled else 0
    check_audio(
        tmp_path / "tx81z-lfo-ame.wav", actual, baseline * 2 ** (-attenuation / 64)
    )


def test_tx81z_shared_lfo_and_notes_restore_together(tmp_path: Path) -> None:
    voice = lfo_voice(3)
    additional = tx81z.TX81ZAdditional(data=bytes(23))
    context = TX81ZLFOContext(voice)
    notes = [TX81ZRenderer(voice, additional, n, 100) for n in (69, 76)]
    rows = context.advance(24000)
    first = notes[0].render(24000, rows) + notes[1].render(24000, rows)
    clock = TX81ZLFOSnapshot.model_validate_json(context.snapshot().model_dump_json())
    snapshots = [
        TX81ZSnapshot.model_validate_json(n.snapshot().model_dump_json()) for n in notes
    ]
    notes[0].release()
    rows = context.advance(24000)
    tail = notes[0].render(24000, rows) + notes[1].render(24000, rows)
    replay_clock = TX81ZLFOContext(voice)
    replay_clock.restore(clock)
    replay = [TX81ZRenderer(voice, additional, n, 100) for n in (69, 76)]
    for renderer, snapshot in zip(replay, snapshots, strict=True):
        renderer.restore(snapshot)
    replay[0].release()
    blocks = [first]
    for length in (997, 23003):
        rows = replay_clock.advance(length)
        blocks.append(replay[1].render(length, rows) + replay[0].render(length, rows))
    check_audio(
        tmp_path / "tx81z-shared-lfo.wav",
        np.concatenate(blocks),
        np.concatenate([first, tail]),
    )
    assert replay_clock.snapshot() == context.snapshot()


@pytest.mark.parametrize("sync", [False, True])
def test_tx81z_keydown_only_resets_synchronized_lfo_counter(sync: bool) -> None:
    raw = bytearray(lfo_voice(3).data)
    raw[58] = int(sync)
    context = TX81ZLFOContext(tx81z.TX81ZVoice(data=bytes(raw)))
    context.advance(100)
    before = context.snapshot()
    context.keydown()
    after = context.snapshot()
    assert after.counter == (0 if sync else before.counter)
    assert after.noise_lfsr == before.noise_lfsr
    assert after.secondary_counter == before.secondary_counter
    assert after.noise_table == before.noise_table
    assert after.frame == before.frame


@pytest.mark.parametrize("delay", [1, 99])
def test_tx81z_shared_context_rejects_unimplemented_delay(delay: int) -> None:
    raw = bytearray(lfo_voice().data)
    raw[55] = delay
    with pytest.raises(EngineError, match="LFO delay is not implemented"):
        TX81ZLFOContext(tx81z.TX81ZVoice(data=bytes(raw)))


@pytest.mark.parametrize(
    "field,value",
    [
        ("rate", 0),
        ("pitch_depth", 0),
        ("amplitude_depth", 0),
        ("sync", False),
        ("waveform", 1),
        ("sample_rate", 44100),
        ("clock_hz", 3072000),
    ],
)
def test_tx81z_context_rejects_a_different_clock_definition(
    field: str, value: int | bool
) -> None:
    context = TX81ZLFOContext(lfo_voice())
    values = context.snapshot().model_dump()
    values[field] = value
    with pytest.raises(EngineError, match="different clock definition"):
        context.restore(TX81ZLFOSnapshot.model_validate(values))


@pytest.mark.parametrize(
    "invalid", ["missing", "shape", "fraction", "pitch", "amplitude", "nan"]
)
def test_tx81z_rejects_invalid_shared_rows_before_advancing(invalid: str) -> None:
    renderer = TX81ZRenderer(
        lfo_voice(), tx81z.TX81ZAdditional(data=bytes(23)), 69, 100
    )
    saved = renderer.snapshot()
    rows = np.zeros((2, 2))
    if invalid == "shape":
        rows = np.zeros((2, 3))
    elif invalid == "fraction":
        rows[0, 0] = 0.5
    elif invalid == "pitch":
        rows[0, 0] = 128
    elif invalid == "amplitude":
        rows[0, 1] = -1
    elif invalid == "nan":
        rows[0, 0] = np.nan
    with pytest.raises(EngineError, match="shared LFO rows"):
        renderer.render(2, None if invalid == "missing" else rows)
    assert renderer.snapshot() == saved
