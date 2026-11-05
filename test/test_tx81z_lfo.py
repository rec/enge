from pathlib import Path

import numpy as np
import pytest
from test_synth import check_audio
from test_tx81z_frequency import opz_frequency, opz_pm_frequency
from test_tx81z_integer import native_carrier
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


@pytest.mark.parametrize("first", range(4))
@pytest.mark.parametrize("second", range(4))
def test_tx81z_dual_lfos_match_pinned_shared_noise_and_snapshot(
    tmp_path: Path, first: int, second: int
) -> None:
    primary = lfo_voice(first)
    raw = bytearray(lfo_voice(second).data)
    raw[54] = 50
    secondary = tx81z.TX81ZVoice(data=bytes(raw))
    context = TX81ZLFOContext(primary, secondary=secondary)
    rows = context.advance(17003)
    saved = TX81ZLFOSnapshot.model_validate_json(context.snapshot().model_dump_json())
    restored = TX81ZLFOContext(primary, secondary=secondary)
    restored.restore(saved)
    actual = np.concatenate([rows, restored.advance(30997)])
    with np.load(Path(__file__).with_name("data") / "tx81z-dual-lfo.npz") as reference:
        expected = reference[f"wave{first}{second}"]
    scaled = ((actual[:, :, :2] * 127) >> 7).reshape(48000, 4)
    check_audio(
        tmp_path / "dual-lfos.wav", scaled.astype(float), expected.astype(float)
    )
    np.testing.assert_array_equal(actual[:, :, 2:], 1)
    context.advance(30997)
    assert context.snapshot() == restored.snapshot()


@pytest.mark.parametrize("rate", [24000, 96000])
def test_shared_tick_rows_match_native_capture_across_host_partitions(
    tmp_path: Path, rate: int
) -> None:
    primary = lfo_voice(3)
    raw = bytearray(lfo_voice(2).data)
    raw[54] = 50
    secondary = tx81z.TX81ZVoice(data=bytes(raw))
    context = TX81ZLFOContext(primary, rate, secondary=secondary)
    first_ends, first = context.advance_ticks(17)
    saved = TX81ZLFOSnapshot.model_validate_json(context.snapshot().model_dump_json())
    replay = TX81ZLFOContext(primary, rate, secondary=secondary)
    replay.restore(saved)
    empty, held = replay.advance_ticks(0)
    assert not len(empty) and replay.snapshot() == saved
    np.testing.assert_array_equal(held, first[-1:])
    ends, rows = replay.advance_ticks(rate - 17)
    np.testing.assert_array_equal(rows[0], first[-1])
    all_rows = np.concatenate([first, rows[1:]])
    all_ends = np.concatenate([first_ends, ends + first_ends[-1]])
    expected_ends = np.arange(1, rate + 1) * 3579545 // (64 * rate)
    np.testing.assert_array_equal(all_ends, expected_ends)
    with np.load(Path(__file__).with_name("data") / "tx81z-lfo-ticks.npz") as reference:
        expected = reference["rows"]
    scaled = ((all_rows[1:, :, :2] * 127) >> 7).reshape(-1, 4)
    check_audio(
        tmp_path / "native-lfo-ticks.wav", scaled.astype(float), expected.astype(float)
    )
    host = TX81ZLFOContext(primary, rate, secondary=secondary)
    np.testing.assert_array_equal(host.advance(rate), all_rows[all_ends])
    assert host.snapshot() == replay.snapshot()


@pytest.mark.parametrize("sync", [False, True])
def test_tx81z_dual_lfos_reset_delays_and_only_synced_phases(sync: bool) -> None:
    first = bytearray(lfo_voice().data)
    first[55], first[58] = 1, int(sync)
    second = bytearray(lfo_voice().data)
    second[55], second[58] = 50, int(not sync)
    context = TX81ZLFOContext(
        tx81z.TX81ZVoice(data=bytes(first)),
        secondary=tx81z.TX81ZVoice(data=bytes(second)),
    )
    rows = context.advance(20000)
    assert rows[-1, 0, 2] == rows[-1, 0, 3]
    assert rows[-1, 1, 2] == 0
    before = context.snapshot()
    context.keydown()
    after = context.snapshot()
    assert after.counter == (0 if sync else before.counter)
    assert after.secondary_counter == (before.secondary_counter if sync else 0)
    assert after.delay_elapsed == after.secondary_delay_elapsed == 0
    assert after.delay_progress == after.secondary_delay_progress == 0
    assert after.noise_table == before.noise_table
    assert after.noise_lfsr == before.noise_lfsr


def test_tx81z_dual_lfo_rejects_foreign_secondary_before_restore() -> None:
    context = TX81ZLFOContext(lfo_voice(), secondary=lfo_voice(3))
    snapshot = context.snapshot()
    other = TX81ZLFOContext(lfo_voice(), secondary=lfo_voice(2))
    with pytest.raises(EngineError, match="different clock definition"):
        context.restore(other.snapshot())
    assert context.snapshot() == snapshot


def test_shared_lfo_restore_rejects_a_remainder_containing_a_whole_tick() -> None:
    context = TX81ZLFOContext(lfo_voice())
    context.advance_ticks(17)
    saved = context.snapshot()
    with pytest.raises(EngineError, match="clock remainder"):
        context.restore(
            saved.model_copy(update={"clock_remainder": 64 * saved.sample_rate})
        )
    assert context.snapshot() == saved


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
    scaled = (actual[:, :2] * 127) >> 7
    check_audio(tmp_path / "tx81z-ymfm-lfo.wav", scaled.astype(float), expected)
    assert saved.model_dump_json() == saved_json
    context.advance(30997)
    assert context.snapshot() == restored.snapshot()
    assert context.snapshot().frame == 48000
    raw = bytearray(voice.data)
    raw[56:58] = bytes(2)
    independent = TX81ZLFOContext(tx81z.TX81ZVoice(data=bytes(raw)))
    np.testing.assert_array_equal(independent.advance(48000), actual)


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
    rows = np.tile([97, 0, 1, 1], (48000, 1))
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
    rows = np.tile([97, 0, 1, 1], (48000, 1))
    actual = TX81ZRenderer(
        voice, tx81z.TX81ZAdditional(data=bytes(23)), 69, 100
    ).render(48000, rows)
    frequency = opz_pm_frequency(np.array([384]), detune=6)[0]
    expected = (4 * native_carrier(frequency))[:, None]
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
    rows = np.tile([0, 65, 1, 1], (48000, 1))
    actual = TX81ZRenderer(voice, additional, 69, 100).render(48000, rows)
    attenuation = 64 << (sensitivity - 1) if sensitivity and enabled else 0
    check_audio(
        tmp_path / "tx81z-lfo-ame.wav",
        actual,
        (4 * native_carrier(opz_frequency(), 0.25 * 2 ** (-attenuation / 64)))[:, None],
    )


@pytest.mark.parametrize("delay", [0, 1])
def test_tx81z_shared_lfo_and_notes_restore_together(
    tmp_path: Path, delay: int
) -> None:
    raw = bytearray(lfo_voice(3).data)
    raw[55] = delay
    voice = tx81z.TX81ZVoice(data=bytes(raw))
    additional = tx81z.TX81ZAdditional(data=bytes(23))
    context = TX81ZLFOContext(voice)
    notes = [TX81ZRenderer(voice, additional, n, 100) for n in (69, 76)]
    rows = context.advance(11000)
    first = notes[0].render(11000, rows) + notes[1].render(11000, rows)
    clock = TX81ZLFOSnapshot.model_validate_json(context.snapshot().model_dump_json())
    snapshots = [
        TX81ZSnapshot.model_validate_json(n.snapshot().model_dump_json()) for n in notes
    ]
    notes[0].release()
    rows = context.advance(37000)
    tail = notes[0].render(37000, rows) + notes[1].render(37000, rows)
    replay_clock = TX81ZLFOContext(voice)
    replay_clock.restore(clock)
    replay = [TX81ZRenderer(voice, additional, n, 100) for n in (69, 76)]
    for renderer, snapshot in zip(replay, snapshots, strict=True):
        renderer.restore(snapshot)
    replay[0].release()
    blocks = [first]
    for length in (997, 36003):
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


@pytest.mark.parametrize(
    "depth,controller,registers",
    [
        (50, 0, [0, 2, 6, 10, 16]),
        (99, 0, [0, 6, 15, 31, 127]),
        (50, 127, [16, 22, 31, 47, 127]),
        (99, 127, [16, 31, 112, 127, 127]),
    ],
)
def test_tx81z_delayed_amplitude_uses_curved_depth_and_undelayed_controllers(
    tmp_path: Path, depth: int, registers: list[int], controller: int
) -> None:
    raw = bytearray(lfo_voice().data)
    raw[55], raw[56], raw[57] = 1, 0, depth
    voice = tx81z.TX81ZVoice(data=bytes(raw))
    extra = tx81z.TX81ZAdditional(data=bytes(23))
    # Exact quarter-progress observations, including a large rational denominator.
    rows = np.tile([0, 255, 0, 2**56], (48000, 1))
    rows[:, 2] = np.repeat(np.arange(5, dtype=np.int64) * 2**54, 9600)
    controls = np.tile([0, controller], (48000, 1))
    note = TX81ZRenderer(voice, extra, 69, 100)
    first = note.render(17003, rows[:17003], controller_depths=controls[:17003])
    saved = TX81ZSnapshot.model_validate_json(note.snapshot().model_dump_json())
    note.restore(saved)
    actual = np.concatenate(
        [first, note.render(30997, rows[17003:], controller_depths=controls[17003:])]
    )
    # Firmware curve after patch-delay multiplication and controller addition.
    depths = np.repeat(registers, 9600)
    attenuation = (255 * depths >> 7) << 2
    expected = 4 * native_carrier(opz_frequency(), 0.25 * 2 ** (-attenuation / 64))
    check_audio(tmp_path / "delayed-amplitude.wav", actual, expected[:, None])


@pytest.mark.parametrize(
    "field,value",
    [
        ("rate", 0),
        ("sync", False),
        ("waveform", 1),
        ("sample_rate", 44100),
        ("clock_hz", 3072000),
        ("delay", 1),
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
    "invalid",
    [
        "missing",
        "shape",
        "fraction",
        "pitch",
        "amplitude",
        "nan",
        "progress",
        "denominator",
    ],
)
def test_tx81z_rejects_invalid_shared_rows_before_advancing(invalid: str) -> None:
    renderer = TX81ZRenderer(
        lfo_voice(), tx81z.TX81ZAdditional(data=bytes(23)), 69, 100
    )
    saved = renderer.snapshot()
    rows = np.tile([0.0, 0, 1, 1], (2, 1))
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
    elif invalid == "progress":
        rows[0, 2] = 2
    elif invalid == "denominator":
        rows[0, 3] = 0
    with pytest.raises(EngineError, match="shared LFO rows"):
        renderer.render(2, None if invalid == "missing" else rows)
    assert renderer.snapshot() == saved


@pytest.mark.parametrize(
    "delay,ticks,clock_hz",
    [(1, 15, 3579545), (50, 121, 3579545), (99, 1022, 3579545), (1, 15, 1536000)],
)
def test_tx81z_delayed_pitch_uses_measured_half_wait_and_byte_depth_ramp(
    tmp_path: Path, delay: int, ticks: int, clock_hz: int
) -> None:
    raw = bytearray(lfo_voice(1).data)
    raw[54], raw[55], raw[57], raw[58] = 0, delay, 0, 0
    voice = tx81z.TX81ZVoice(data=bytes(raw))
    context = TX81ZLFOContext(voice, clock_hz=clock_hz)
    frames = max(48000, ticks * 1000 + 1000)
    expected = np.zeros((frames, 2))
    # At speed zero the square remains at +127 PM through this interval.
    elapsed = np.arange(frames)
    if clock_hz == 1536000:
        elapsed = np.maximum(0, (elapsed - 1) // 2 * 2 + 1)
    delay_byte = np.clip(255 * (2 * elapsed - ticks * 1000) // (ticks * 1000), 0, 255)
    depth = np.floor(np.floor(255 * delay_byte / 256) / 2)
    expected[:, 0] = np.floor(127 * depth / 128)
    first = context.advance(12345)
    saved = TX81ZLFOSnapshot.model_validate_json(context.snapshot().model_dump_json())
    restored = TX81ZLFOContext(voice, clock_hz=clock_hz)
    restored.restore(saved)
    actual = np.concatenate([first, restored.advance(frames - len(first))])
    delayed_depth = (255 * (255 * actual[:, 2] // actual[:, 3]) >> 8) >> 1
    scaled = np.column_stack([actual[:, 0] * delayed_depth >> 7, np.zeros(frames)])
    check_audio(tmp_path / "delayed-pitch.wav", scaled, expected)
    restored.keydown()
    assert restored.snapshot().delay_elapsed == 0
    assert restored.snapshot().counter != 0
    assert restored.advance(1)[0, 2] == 0
    assert saved.delay_elapsed == len(first)


@pytest.mark.parametrize("patch_depth", [0, 99])
@pytest.mark.parametrize("waveform", range(4))
def test_tx81z_controller_depths_add_to_patch_and_saturate(
    tmp_path: Path, patch_depth: int, waveform: int
) -> None:
    raw = bytearray(lfo_voice(waveform).data)
    raw[56:58] = bytes([patch_depth] * 2)
    voice = tx81z.TX81ZVoice(data=bytes(raw))
    extra = tx81z.TX81ZAdditional(data=bytes(23))
    rows = TX81ZLFOContext(voice).advance(48000)
    controls = np.tile([127, 204], (48000, 1))
    note = TX81ZRenderer(voice, extra, 69, 100)
    first = note.render(12345, rows[:12345], controller_depths=controls[:12345])
    saved = TX81ZSnapshot.model_validate_json(note.snapshot().model_dump_json())
    note.restore(saved)
    actual = np.concatenate(
        [first, note.render(35655, rows[12345:], controller_depths=controls[12345:])]
    )
    pitch, amplitude = (127, 127) if patch_depth else (63, 37)
    pm = rows[:, 0] * pitch >> 7
    attenuation = (rows[:, 1] * amplitude >> 7) << 2
    expected = (
        4 * native_carrier(opz_pm_frequency(pm * 4), 0.25 * 2 ** (-attenuation / 64))
    )[:, None]
    check_audio(tmp_path / "tx81z-controller-depths.wav", actual, expected)
    for invalid in (
        np.full((1, 2), np.nan),
        np.full((1, 2), 256),
        np.full((1, 2), 0.5),
        np.zeros((1, 3)),
    ):
        before = note.snapshot()
        with pytest.raises(EngineError, match="controller depths"):
            note.render(1, rows[:1], controller_depths=invalid)
        assert note.snapshot() == before


def test_tx81z_controller_modulation_bypasses_patch_delay(tmp_path: Path) -> None:
    raw = bytearray(lfo_voice().data)
    extra = tx81z.TX81ZAdditional(data=bytes(23))
    reference = tx81z.TX81ZVoice(data=bytes(raw))
    expected = TX81ZRenderer(reference, extra, 69, 100).render(
        48000, TX81ZLFOContext(reference).advance(48000)
    )
    raw[55:58] = bytes([99, 0, 0])
    delayed = tx81z.TX81ZVoice(data=bytes(raw))
    actual = TX81ZRenderer(delayed, extra, 69, 100).render(
        48000,
        TX81ZLFOContext(delayed).advance(48000),
        controller_depths=np.full((48000, 2), 255),
    )
    check_audio(tmp_path / "tx81z-controller-delay.wav", actual, expected)
