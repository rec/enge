from pathlib import Path

import pytest
from meedee import messages, smf
from ufor.events import ControlChange, Release, Trigger

from enge.midi import read_midi


def test_midi_tempo_map_rounds_absolute_time_and_matches_overlapping_notes(
    tmp_path: Path,
) -> None:
    path = tmp_path / "tempo.mid"
    writer = smf.Writer(path, header=smf.Header(format=0, tracks=1, division=7))
    records = [
        (0, smf.Meta(kind=0x51, data=(500003).to_bytes(3, "big"))),
        (0, messages.NoteOn(channel=1, note=60, velocity=80)),
        (1, messages.NoteOn(channel=1, note=60, velocity=100)),
        (7, smf.Meta(kind=0x51, data=(1000000).to_bytes(3, "big"))),
        (7, messages.NoteOff(channel=1, note=60, velocity=64)),
        (14, messages.NoteOn(channel=1, note=60, velocity=0)),
        (14, smf.Meta(kind=0x2F, data=b"")),
    ]
    for t, r in records:
        writer.write(smf.Event(track=0, tick=t, record=r))
    writer.finish()
    performance = read_midi(path)
    events = performance.parts[1]
    assert [e.tick for e in events] == [0, 3429, 24000, 72000]
    first, second, first_release, second_release = events
    assert isinstance(first, Trigger)
    assert isinstance(second, Trigger)
    assert isinstance(first_release, Release)
    assert isinstance(second_release, Release)
    assert first.trigger_id == first_release.trigger_id
    assert second.trigger_id == second_release.trigger_id
    assert second.controls["velocity"] == 100 / 127


def test_midi_controls_keep_same_frame_order_and_channel_scope(tmp_path: Path) -> None:
    path = tmp_path / "controls.mid"
    writer = smf.Writer(path, header=smf.Header(format=0, tracks=1, division=480))
    records = [
        (0, messages.ControlChange(channel=4, controller=11, value=100)),
        (0, messages.PitchBend(channel=4, value=-4096)),
        (0, messages.NoteOn(channel=4, note=70, velocity=90)),
        (480, messages.NoteOff(channel=4, note=70, velocity=64)),
        (480, smf.Meta(kind=0x2F, data=b"")),
    ]
    for t, r in records:
        writer.write(smf.Event(track=0, tick=t, record=r))
    writer.finish()
    events = read_midi(path).parts[4]
    assert isinstance(events[0], ControlChange)
    assert events[0].scope == "part"
    assert events[0].part == "channel-4"
    assert events[0].value == 100 / 127
    assert isinstance(events[1], ControlChange)
    assert events[1].value == -0.5
    assert events[0].ordinal < events[1].ordinal < events[2].ordinal


@pytest.mark.parametrize(
    "message",
    [
        messages.NoteOff(channel=1, note=60, velocity=64),
        messages.NoteOn(channel=1, note=60, velocity=100),
        messages.ControlChange(channel=1, controller=64, value=127),
        messages.ChannelPressure(channel=1, pressure=80),
        smf.SysExFragment(status=0xF0, data=b"\x7d\x01\xf7"),
    ],
)
def test_midi_rejects_unmatched_notes_and_unsupported_messages(
    tmp_path: Path, message: messages.Message | smf.SysExFragment
) -> None:
    path = tmp_path / "invalid.mid"
    writer = smf.Writer(path, header=smf.Header(format=0, tracks=1, division=480))
    writer.write(smf.Event(track=0, tick=0, record=message))
    writer.write(smf.Event(track=0, tick=0, record=smf.Meta(kind=0x2F, data=b"")))
    writer.finish()
    with pytest.raises(ValueError):
        read_midi(path)


def test_midi_merges_tracks_stably_and_preserves_trailing_silence(
    tmp_path: Path,
) -> None:
    path = tmp_path / "tracks.mid"
    writer = smf.Writer(path, header=smf.Header(format=1, tracks=3, division=480))
    tracks = [
        [
            (0, smf.Meta(kind=0x51, data=(500000).to_bytes(3, "big"))),
            (240, smf.Meta(kind=0x2F, data=b"")),
        ],
        [
            (0, messages.NoteOn(channel=1, note=60, velocity=80)),
            (0, messages.ProgramChange(channel=1, program=128)),
            (0, smf.Meta(kind=0x7F, data=b"unknown")),
            (480, messages.NoteOff(channel=1, note=60, velocity=64)),
            (960, smf.Meta(kind=0x2F, data=b"")),
        ],
        [
            (0, messages.NoteOn(channel=16, note=64, velocity=80)),
            (480, messages.NoteOff(channel=16, note=64, velocity=64)),
            (480, smf.Meta(kind=0x2F, data=b"")),
        ],
    ]
    for i, v in enumerate(tracks):
        for t, r in v:
            writer.write(smf.Event(track=i, tick=t, record=r))
    writer.finish()
    performance = read_midi(path)
    assert performance.frames == 48001
    assert list(performance.parts) == [1, 16]
    assert [e.ordinal for e in performance.parts[1]] == [1, 5]
    assert [e.ordinal for e in performance.parts[16]] == [4, 6]
    assert [e.tick for e in performance.parts[16]] == [0, 24000]
    assert all(
        isinstance(e, Trigger | Release) and e.part == "channel-16"
        for e in performance.parts[16]
    )


@pytest.mark.parametrize("format,division", [(2, 480), (0, 0xE728)])
def test_midi_rejects_asynchronous_or_smpte_timing(
    tmp_path: Path, format: int, division: int
) -> None:
    path = tmp_path / "timing.mid"
    writer = smf.Writer(
        path, header=smf.Header(format=format, tracks=1, division=division)
    )
    writer.write(smf.Event(track=0, tick=0, record=smf.Meta(kind=0x2F, data=b"")))
    writer.finish()
    with pytest.raises((ValueError, OSError)):
        read_midi(path)


def test_midi_rejects_nonpositive_sample_rate(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="positive rate"):
        read_midi(tmp_path / "unused.mid", sample_rate=0)


def test_midi_rejects_truncated_track(tmp_path: Path) -> None:
    path = tmp_path / "truncated.mid"
    path.write_bytes(
        bytes.fromhex("4d546864000000060000000101e04d54726b0000000400ff2f")
    )
    with pytest.raises(OSError):
        read_midi(path)


def test_bach_fixture_preserves_four_complete_parts_and_subtle_modulations() -> None:
    path = Path(__file__).parents[1] / "scripts/bwv-578.mid"
    performance = read_midi(path)
    counts = {
        c: sum(isinstance(e, Trigger) for e in v) for c, v in performance.parts.items()
    }
    assert counts == {1: 179, 2: 511, 3: 268, 4: 807}
    assert performance.frames == 11519998
    for events in performance.parts.values():
        controls = [e for e in events if isinstance(e, ControlChange)]
        assert {e.control for e in controls} == {"expression", "modulation", "bend"}
        assert all(abs(e.value * 200) <= 3.01 for e in controls if e.control == "bend")
        assert all(
            103 / 127 <= e.value <= 113 / 127
            for e in controls
            if e.control == "expression"
        )
