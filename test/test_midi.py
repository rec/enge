from pathlib import Path

import mido
import pytest
from ufor.events import ControlChange, Release, Trigger

from enge.midi import read_midi


def test_midi_tempo_map_rounds_absolute_time_and_matches_overlapping_notes(
    tmp_path: Path,
) -> None:
    source = mido.MidiFile(ticks_per_beat=7)
    source.tracks.append(
        mido.MidiTrack(
            [
                mido.MetaMessage("set_tempo", tempo=500003),
                mido.Message("note_on", note=60, velocity=80),
                mido.Message("note_on", note=60, velocity=100, time=1),
                mido.MetaMessage("set_tempo", tempo=1000000, time=6),
                mido.Message("note_off", note=60),
                mido.Message("note_on", note=60, velocity=0, time=7),
            ]
        )
    )
    path = tmp_path / "tempo.mid"
    source.save(path)
    performance = read_midi(path)
    events = performance.parts[0]
    assert [e.tick for e in events] == [0, 3429, 24000, 72000]
    assert [type(e) for e in events] == [Trigger, Trigger, Release, Release]
    assert events[0].trigger_id == events[2].trigger_id
    assert events[1].trigger_id == events[3].trigger_id
    assert events[1].controls["velocity"] == 100 / 127


def test_midi_controls_keep_same_frame_order_and_channel_scope(tmp_path: Path) -> None:
    source = mido.MidiFile()
    source.tracks.append(
        mido.MidiTrack(
            [
                mido.Message("control_change", channel=3, control=11, value=100),
                mido.Message("pitchwheel", channel=3, pitch=-4096),
                mido.Message("note_on", channel=3, note=70, velocity=90),
                mido.Message("note_off", channel=3, note=70, time=480),
            ]
        )
    )
    path = tmp_path / "controls.mid"
    source.save(path)
    events = read_midi(path).parts[3]
    assert isinstance(events[0], ControlChange)
    assert events[0].scope == "part"
    assert events[0].part == "channel-3"
    assert events[0].value == 100 / 127
    assert events[1].value == -0.5
    assert events[0].ordinal < events[1].ordinal < events[2].ordinal


@pytest.mark.parametrize(
    "messages",
    [
        [mido.Message("note_off", note=60)],
        [mido.Message("note_on", note=60, velocity=100)],
        [mido.Message("control_change", control=64, value=127)],
    ],
)
def test_midi_rejects_unmatched_notes_and_unsupported_controls(
    tmp_path: Path, messages: list[mido.Message]
) -> None:
    source = mido.MidiFile()
    source.tracks.append(mido.MidiTrack(messages))
    path = tmp_path / "invalid.mid"
    source.save(path)
    with pytest.raises(ValueError):
        read_midi(path)


def test_bach_fixture_preserves_four_complete_parts_and_subtle_modulations() -> None:
    path = Path(__file__).parents[1] / "scripts/bwv-578.mid"
    performance = read_midi(path)
    counts = {
        c: sum(isinstance(e, Trigger) for e in v) for c, v in performance.parts.items()
    }
    assert counts == {0: 179, 1: 511, 2: 268, 3: 807}
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
