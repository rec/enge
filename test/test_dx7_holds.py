from pathlib import Path

import numpy as np
import pytest
from test_synth import check_audio
from test_yamaha import voice
from ufor.dx7 import DX7Voice

from enge.yamaha import DX7Envelope, DX7Renderer, DX7Snapshot


@pytest.mark.parametrize(
    "sample_rate,step", [(44100, 4096), (48000, 3763), (96000, 1882)]
)
def test_release_rate_uses_per_frame_reference_timing(
    tmp_path: Path, sample_rate: int, step: int
) -> None:
    raw = bytearray(voice().data)
    raw[3], raw[7] = 50, 0
    envelope = DX7Envelope.start(
        DX7Voice(data=bytes(raw)).operator(6), 69, 100, sample_rate
    )
    for _ in range(1000):
        envelope.sample()
    assert envelope.stage == 3
    initial = envelope.level
    envelope.release()
    # Pinned Dexed Env drops 262144 units per 64 frames at 44.1 kHz.
    # Convert its measured slope to the nearest integer per output frame.
    gains = np.array([envelope.sample() for _ in range(sample_rate)])
    expected = 2 ** (
        ((initial - step * np.arange(1, sample_rate + 1)) / 65536 - 14 * 256) / 256
    )
    np.testing.assert_array_equal(gains, expected)
    if sample_rate == 48000:
        check_audio(tmp_path / "release-rate.wav", gains[:, None], expected[:, None])


@pytest.mark.parametrize(
    "rate,scaling,frames",
    [(50, 0, 7200), (49, 1, 7200), (76, 0, 480), (77, 0, 479), (98, 0, 22)],
)
def test_equal_levels_hold_for_measured_duration(
    rate: int, scaling: int, frames: int
) -> None:
    raw = bytearray(voice().data)
    raw[1], raw[6] = rate, 50
    envelope = DX7Envelope.start(DX7Voice(data=bytes(raw)).operator(6), 69, 100, 48000)
    # Key rate scaling is already prepared by admission; isolate its contribution.
    envelope.rate_scaling = scaling
    for _ in range(1000):
        envelope.sample()
        if envelope.stage == 1:
            break
    assert envelope.stage == 1
    assert envelope.hold_frames == frames
    level = envelope.level
    for _ in range(frames):
        envelope.sample()
        assert envelope.level == level
    assert envelope.stage == 2
    envelope.sample()
    assert envelope.level < level


@pytest.mark.parametrize(
    "sample_rate,frames", [(22050, 3308), (44100, 6615), (48000, 7200), (96000, 14400)]
)
def test_hold_duration_stays_constant_across_sample_rates(
    sample_rate: int, frames: int
) -> None:
    raw = bytearray(voice().data)
    raw[1] = 50
    envelope = DX7Envelope.start(
        DX7Voice(data=bytes(raw)).operator(6), 69, 100, sample_rate
    )
    for _ in range(1000):
        envelope.sample()
        if envelope.stage == 1:
            break
    assert envelope.stage == 1
    assert envelope.hold_frames == frames


def test_zero_attack_level_waits_before_rising() -> None:
    raw = bytearray(voice().data)
    raw[0], raw[4] = 50, 0
    envelope = DX7Envelope.start(DX7Voice(data=bytes(raw)).operator(6), 69, 100, 48000)
    assert envelope.hold_frames == 360
    for _ in range(360):
        envelope.sample()
        assert envelope.level == 0
    assert envelope.stage == 1
    envelope.sample()
    assert envelope.level > 0


@pytest.mark.parametrize("release_frame", [1001, 10000])
def test_hold_snapshot_replays_and_release_interrupts_wait(
    tmp_path: Path, release_frame: int
) -> None:
    raw = bytearray(voice().data)
    for i in range(0, 126, 21):
        raw[i + 1], raw[i + 6], raw[i + 7] = 50, 50, 0
    note = DX7Renderer(DX7Voice(data=bytes(raw)), 69, 100)
    head = note.render(997)
    saved = DX7Snapshot.model_validate_json(note.snapshot().model_dump_json())
    middle = note.render(release_frame - 997)
    note.release()
    assert all(e.hold_frames == 0 for e in note.envelopes)
    expected = np.concatenate([head, middle, note.render(48000 - release_frame)])
    note.restore(saved)
    replay = [head, note.render(release_frame - 997)]
    note.release()
    replay.extend([note.render(137), note.render(48000 - release_frame - 137)])
    check_audio(tmp_path / "equal-level-hold.wav", np.concatenate(replay), expected)
