from pathlib import Path

import numpy as np
import pytest
from test_synth import check_audio
from test_yamaha import voice
from ufor.dx7 import DX7Voice

from enge.synth import EngineError
from enge.yamaha import DX7Envelope, DX7Renderer, DX7Snapshot, dx7_operator_sample
from enge.yamaha_lfo import DX7LFOContext, DX7LFOSnapshot


@pytest.mark.parametrize("waveform", range(6))
def test_shared_lfo_retains_waveform_delay_and_json_continuation(
    tmp_path: Path, waveform: int
) -> None:
    data = bytearray(voice().data)
    data[137:143] = bytes([75, 42, 0, 0, 1, waveform])
    patch = DX7Voice(data=bytes(data))
    whole = DX7LFOContext(patch)
    whole.keydown()
    expected = whole.advance(48000)
    split = DX7LFOContext(patch)
    split.keydown()
    first = split.advance(17003)
    saved = DX7LFOSnapshot.model_validate_json(split.snapshot().model_dump_json())
    restored = DX7LFOContext(patch)
    restored.restore(saved)
    actual = np.concatenate([first, restored.advance(30997)])
    check_audio(tmp_path / "dx7-shared-lfo.wav", actual, expected)
    assert saved.frame == 17003
    assert restored.snapshot() == whole.snapshot()
    assert np.all((actual >= 0) & (actual <= 1))
    assert actual[0, 1] == 0 and actual[-1, 1] == 1
    assert np.all(np.diff(actual[:, 1]) >= 0)
    assert np.ptp(actual[:, 0]) > 0
    phases = (
        2**31 - 1 + np.arange(1, 48001, dtype=np.int64) * (33587 * 2509) // 64
    ) % 2**32
    position = phases / 2**32
    if waveform == 5:
        changes = np.concatenate([[0], np.flatnonzero(np.diff(actual[:, 0])) + 1])
        assert actual[changes[:3], 0].tolist() == [129 / 256, 146 / 256, 117 / 256]
    else:
        shapes = [
            1 - np.abs(2 * position - 1),
            (0.5 - position) % 1,
            (position + 0.5) % 1,
            (position < 0.5).astype(float),
            0.5 + 0.5 * np.sin(2 * np.pi * position),
        ]
        np.testing.assert_allclose(actual[:, 0], shapes[waveform], atol=2**-24, rtol=0)


def test_shared_lfo_rate_zero_and_no_delay_match_reference_calibration(
    tmp_path: Path,
) -> None:
    data = bytearray(voice().data)
    data[142] = 2
    context = DX7LFOContext(DX7Voice(data=bytes(data)))
    actual = context.advance(48000)
    # Pinned MSFA: rate zero uses multiplier 11 and a 33587-unit 48 kHz clock.
    phases = (np.arange(1, 48001, dtype=np.int64) * 369457) // 64
    expected = np.column_stack([((phases ^ 2**31) >> 8) / 2**24, np.ones(48000)])
    check_audio(tmp_path / "dx7-lfo-calibration.wav", actual, expected)
    assert context.snapshot().phase == 277092750
    assert context.snapshot().phase_remainder == 0


@pytest.mark.parametrize("sync", [False, True])
def test_shared_keydown_resets_delay_and_only_synchronized_phase(
    tmp_path: Path, sync: bool
) -> None:
    data = bytearray(voice().data)
    data[137:143] = bytes([99, 42, 0, 0, int(sync), 5])
    patch = DX7Voice(data=bytes(data))
    context = DX7LFOContext(patch)
    first = context.advance(24000)
    before = context.snapshot()
    context.keydown()
    after = context.snapshot()
    assert after.phase == (2**31 - 1 if sync else before.phase)
    assert after.random_state == before.random_state
    assert after.frame == before.frame
    assert after.delay_position == 0
    actual = np.concatenate([first, context.advance(24000)])
    replay = DX7LFOContext(patch)
    replay.restore(after)
    expected = np.concatenate([first, replay.advance(997), replay.advance(23003)])
    check_audio(tmp_path / "dx7-lfo-keydown.wav", actual, expected)


def test_note_pitch_modulation_uses_shared_rows_in_octaves(tmp_path: Path) -> None:
    data = bytearray(voice().data)
    data[139], data[143] = 99, 7
    patch = DX7Voice(data=bytes(data))
    rows = np.tile([0.5 + 16384 / (255 * 255), 1], (48000, 1))
    actual = DX7Renderer(patch, 69, 100).render(48000, rows)
    expected = DX7Renderer(voice(), 75, 100).render(48000)
    check_audio(tmp_path / "dx7-lfo-pitch.wav", actual, expected)


def test_two_notes_share_one_clock_and_restore_an_aggregate_snapshot(
    tmp_path: Path,
) -> None:
    data = bytearray(voice().data)
    data[137:144] = bytes([80, 17, 65, 70, 1, 5, 5])
    for i in range(6):
        data[i * 21 + 14] = i % 4
        data[i * 21 + 7] = 0
    patch = DX7Voice(data=bytes(data))
    context = DX7LFOContext(patch)
    context.keydown()
    notes = [DX7Renderer(patch, n, 100) for n in (69, 76)]
    rows = context.advance(24000)
    first = notes[0].render(24000, rows) + notes[1].render(24000, rows)
    assert context.snapshot().frame == 24000
    saved_clock = DX7LFOSnapshot.model_validate_json(
        context.snapshot().model_dump_json()
    )
    saved_notes = [
        DX7Snapshot.model_validate_json(n.snapshot().model_dump_json()) for n in notes
    ]
    notes[0].release()
    rows = context.advance(24000)
    tail = notes[0].render(24000, rows) + notes[1].render(24000, rows)
    replay_clock = DX7LFOContext(patch)
    replay_clock.restore(saved_clock)
    replay = [DX7Renderer(patch, n, 100) for n in (69, 76)]
    for renderer, snapshot in zip(replay, saved_notes, strict=True):
        renderer.restore(snapshot)
    replay[0].release()
    restored = []
    for length in (997, 23003):
        rows = replay_clock.advance(length)
        restored.append(replay[1].render(length, rows) + replay[0].render(length, rows))
    check_audio(
        tmp_path / "dx7-shared-notes.wav",
        np.concatenate([first, *restored]),
        np.concatenate([first, tail]),
    )
    assert replay_clock.snapshot() == context.snapshot()
    assert saved_clock.frame == 24000


@pytest.mark.parametrize("field", ["speed", "delay", "sync", "waveform", "sample_rate"])
def test_shared_clock_rejects_a_different_definition(field: str) -> None:
    context = DX7LFOContext(voice())
    values = context.snapshot().model_dump()
    values[field] = True if field == "sync" else 1
    with pytest.raises(EngineError, match="different clock definition"):
        context.restore(DX7LFOSnapshot.model_validate(values))


@pytest.mark.parametrize("invalid", ["missing", "shape", "range", "nan"])
def test_note_rejects_invalid_shared_rows_before_advancing(invalid: str) -> None:
    data = bytearray(voice().data)
    data[139], data[143] = 99, 7
    renderer = DX7Renderer(DX7Voice(data=bytes(data)), 69, 100)
    before = renderer.snapshot()
    rows = np.tile([0.5, 1], (2, 1))
    if invalid == "shape":
        rows = np.zeros((2, 3))
    elif invalid == "range":
        rows[0, 0] = 1.1
    elif invalid == "nan":
        rows[0, 0] = np.nan
    with pytest.raises(EngineError, match="shared LFO rows"):
        renderer.render(2, None if invalid == "missing" else rows)
    assert renderer.snapshot() == before


def test_amplitude_modulation_requires_shared_rows_before_advancing() -> None:
    data = bytearray(voice().data)
    data[14], data[140] = 1, 1
    renderer = DX7Renderer(DX7Voice(data=bytes(data)), 69, 100)
    before = renderer.snapshot()
    with pytest.raises(EngineError, match="shared LFO rows"):
        renderer.render(1)
    assert renderer.snapshot() == before


def test_modulation_depths_have_no_effect_with_zero_sensitivity(tmp_path: Path) -> None:
    data = bytearray(voice().data)
    data[139:141] = bytes([99, 99])
    actual = DX7Renderer(DX7Voice(data=bytes(data)), 69, 100).render(48000)
    expected = DX7Renderer(voice(), 69, 100).render(48000)
    check_audio(tmp_path / "dx7-zero-sensitivity.wav", actual, expected)


@pytest.mark.parametrize(
    "depth,level", [(0, 0), (1, 0.784829), (50, 6.422298), (98, 50.347169), (99, 52.75)]
)
@pytest.mark.parametrize(
    "sensitivity,ratio", [(0, 0), (1, 0.238058), (2, 0.460510), (3, 1)]
)
def test_operator_amplitude_follows_the_provisional_hexter_calibration(
    tmp_path: Path, depth: int, level: float, sensitivity: int, ratio: float
) -> None:
    data = bytearray(voice().data)
    data[140] = depth
    for i in range(0, 126, 21):
        data[i + 14] = sensitivity
    rows = np.ones((48000, 2))
    actual = DX7Renderer(DX7Voice(data=bytes(data)), 69, 100).render(48000, rows)
    envelope = DX7Envelope.start(voice().operator(1), 69, 100, 48000)
    expected = np.empty((48000, 1))
    for i in range(48000):
        envelope.sample()
        expected[i, 0] = (
            6
            * dx7_operator_sample(
                i * round(440 * 2**24 / 48000),
                16384 - (envelope.level >> 14) + int(level * ratio / 8 * 1024),
            )
            / 2**24
        )
    check_audio(tmp_path / "dx7-amplitude-calibration.wav", actual, expected)


@pytest.mark.parametrize("value,delay", [(0, 1), (1, 0)])
def test_zero_lfo_level_or_delay_preserves_operator_gain(
    tmp_path: Path, value: float, delay: float
) -> None:
    data = bytearray(voice().data)
    data[140] = 99
    for i in range(0, 126, 21):
        data[i + 14] = 3
    rows = np.tile([value, delay], (48000, 1))
    actual = DX7Renderer(DX7Voice(data=bytes(data)), 69, 100).render(48000, rows)
    expected = DX7Renderer(voice(), 69, 100).render(48000)
    check_audio(tmp_path / "dx7-amplitude-zero.wav", actual, expected)
