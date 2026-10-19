from pathlib import Path

import pytest
from meedee import messages, smf
from ufor import performance_binding
from ufor.events import ControlChange, Release, Trigger
from ufor.tuning import Computed, Tuning

from enge.midi import read_midi
from enge.performance import MidiConverter
from enge.presets import Patch


@pytest.fixture
def converters() -> list[MidiConverter]:
    entries = [
        {
            "channels": [c],
            "part": f"channel-{c}",
            "repeated_key_release": "oldest",
            "controllers": [
                {"number": n, "control": v}
                for n, v in [(1, "modulation"), (11, "expression")]
            ],
            "pitch_bend": {"control": "bend"},
        }
        for c in range(1, 17)
    ]
    instrument, _ = Patch(engine="synth").prepare_score(48000)
    return [
        MidiConverter(
            source="file",
            binding=performance_binding.PerformanceBindingScore.model_validate(
                {
                    "name": "file",
                    "title": "File input",
                    "body": {
                        "instrument": {"selector": "test:instrument"},
                        "midi": entries,
                    },
                }
            ),
            instrument=instrument,
        )
    ]


@pytest.fixture
def other_converter() -> MidiConverter:
    score, _ = Patch(engine="synth").prepare_score(48000)
    binding = performance_binding.PerformanceBindingScore.model_validate(
        {
            "name": "other",
            "title": "Other input",
            "body": {
                "instrument": {"selector": "test:instrument"},
                "midi": [
                    {
                        "channels": [1],
                        "part": "other",
                        "repeated_key_release": "newest",
                        "tuning": {"path": "tuning.toml"},
                    }
                ],
            },
        }
    )
    return MidiConverter(
        source="other",
        binding=binding,
        instrument=score,
        tunings={"tuning.toml": Tuning(source=Computed(), root_frequency="442")},
    )


def test_midi_tempo_map_rounds_absolute_time_and_matches_overlapping_notes(
    tmp_path: Path,
    converters: list[MidiConverter],
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
    performance = read_midi(path, converters=converters)
    events = performance.parts["channel-1"]
    assert [e.tick for e in events] == [0, 3429, 24000, 72000]
    first, second, first_release, second_release = events
    assert isinstance(first, Trigger)
    assert isinstance(second, Trigger)
    assert isinstance(first_release, Release)
    assert isinstance(second_release, Release)
    assert first.trigger_id == first_release.trigger_id
    assert second.trigger_id == second_release.trigger_id
    assert second.velocity == 100 / 127


def test_midi_controls_keep_same_frame_order_and_channel_scope(
    tmp_path: Path, converters: list[MidiConverter]
) -> None:
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
    events = read_midi(path, converters=converters).parts["channel-4"]
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
        smf.SysExFragment(status=0xF0, data=b"\x7d\x01\xf7"),
    ],
)
def test_midi_rejects_unmatched_notes_and_unsupported_messages(
    tmp_path: Path,
    converters: list[MidiConverter],
    message: messages.Message | smf.SysExFragment,
) -> None:
    path = tmp_path / "invalid.mid"
    writer = smf.Writer(path, header=smf.Header(format=0, tracks=1, division=480))
    writer.write(smf.Event(track=0, tick=0, record=message))
    writer.write(smf.Event(track=0, tick=0, record=smf.Meta(kind=0x2F, data=b"")))
    writer.finish()
    with pytest.raises(ValueError):
        read_midi(path, converters=converters)


def test_midi_merges_tracks_stably_and_preserves_trailing_silence(
    tmp_path: Path,
    converters: list[MidiConverter],
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
    performance = read_midi(path, converters=converters)
    assert performance.frames == 48001
    assert [p for p, v in performance.parts.items() if v] == ["channel-1", "channel-16"]
    assert [e.ordinal for e in performance.parts["channel-1"]] == [1, 5]
    assert [e.ordinal for e in performance.parts["channel-16"]] == [4, 6]
    assert [e.tick for e in performance.parts["channel-16"]] == [0, 24000]
    assert all(
        isinstance(e, Trigger | Release) and e.part == "channel-16"
        for e in performance.parts["channel-16"]
    )


@pytest.mark.parametrize("format,division", [(2, 480), (0, 0xE728)])
def test_midi_rejects_asynchronous_or_smpte_timing(
    tmp_path: Path, converters: list[MidiConverter], format: int, division: int
) -> None:
    path = tmp_path / "timing.mid"
    writer = smf.Writer(
        path, header=smf.Header(format=format, tracks=1, division=division)
    )
    writer.write(smf.Event(track=0, tick=0, record=smf.Meta(kind=0x2F, data=b"")))
    writer.finish()
    with pytest.raises((ValueError, OSError)):
        read_midi(path, converters=converters)


def test_midi_rejects_nonpositive_sample_rate(
    tmp_path: Path, converters: list[MidiConverter]
) -> None:
    with pytest.raises(ValueError, match="positive integer sample rate"):
        read_midi(tmp_path / "unused.mid", converters=converters, sample_rate=0)
    assert converters[0].ready


def test_midi_rejects_truncated_track(
    tmp_path: Path, converters: list[MidiConverter]
) -> None:
    path = tmp_path / "truncated.mid"
    path.write_bytes(
        bytes.fromhex("4d546864000000060000000101e04d54726b0000000400ff2f")
    )
    with pytest.raises(OSError):
        read_midi(path, converters=converters)
    assert not converters[0].ready


def test_bach_fixture_preserves_four_complete_parts_and_subtle_modulations(
    converters: list[MidiConverter],
) -> None:
    path = Path(__file__).parents[1] / "scripts/bwv-578.mid"
    performance = read_midi(path, converters=converters)
    counts = {
        c: sum(isinstance(e, Trigger) for e in v) for c, v in performance.parts.items()
    }
    assert {c: v for c, v in counts.items() if v} == {
        "channel-1": 179,
        "channel-2": 511,
        "channel-3": 268,
        "channel-4": 807,
    }
    assert performance.frames == 11519998
    for events in [v for v in performance.parts.values() if v]:
        controls = [e for e in events if isinstance(e, ControlChange)]
        assert {e.control for e in controls} == {"expression", "modulation", "bend"}
        assert all(abs(e.value * 200) <= 3.01 for e in controls if e.control == "bend")
        assert all(
            103 / 127 <= e.value <= 113 / 127
            for e in controls
            if e.control == "expression"
        )


def test_file_dispatch_uses_each_binding_and_preserves_rounded_frame_order(
    tmp_path: Path, converters: list[MidiConverter], other_converter: MidiConverter
) -> None:
    path = tmp_path / "bound.mid"
    writer = smf.Writer(path, header=smf.Header(format=0, tracks=1, division=480))
    records = [
        messages.NoteOn(channel=1, note=69, velocity=80),
        messages.NoteOn(channel=1, note=69, velocity=100),
        messages.NoteOff(channel=1, note=69, velocity=0),
        messages.NoteOn(channel=1, note=69, velocity=0),
    ]
    for i, m in enumerate(records):
        writer.write(smf.Event(track=0, tick=i, record=m))
    writer.write(smf.Event(track=0, tick=3, record=smf.Meta(kind=0x2F, data=b"")))
    writer.finish()
    performance = read_midi(
        path, converters=[*converters, other_converter], sample_rate=1
    )
    oldest = performance.parts["channel-1"]
    newest = performance.parts["other"]
    assert [e.tick for e in newest] == [0, 0, 0, 0]
    assert [e.ordinal for e in newest] == [0, 1, 2, 3]
    assert isinstance(newest[0], Trigger)
    assert newest[0].pitch_hz == 442
    assert isinstance(newest[1], Trigger)
    assert isinstance(newest[2], Release)
    assert newest[2].trigger_id == newest[1].trigger_id
    assert isinstance(oldest[0], Trigger)
    assert isinstance(oldest[2], Release)
    assert oldest[2].trigger_id == oldest[0].trigger_id
    assert oldest[0].trigger_id != newest[0].trigger_id
    assert performance.parts["channel-16"] == []
    assert not other_converter.ready
    with pytest.raises(ValueError, match="fresh"):
        read_midi(path, converters=[other_converter])
    other_converter.restart()
    assert read_midi(path, converters=[other_converter]).parts["other"]


def test_file_failure_invalidates_all_converters_without_partial_results(
    tmp_path: Path, converters: list[MidiConverter], other_converter: MidiConverter
) -> None:
    path = tmp_path / "sysex.mid"
    writer = smf.Writer(path, header=smf.Header(format=0, tracks=1, division=480))
    writer.write(
        smf.Event(
            track=0, tick=0, record=messages.NoteOn(channel=1, note=69, velocity=80)
        )
    )
    writer.write(
        smf.Event(
            track=0, tick=1, record=smf.SysExFragment(status=0xF7, data=b"\x90\x3c\x50")
        )
    )
    writer.write(smf.Event(track=0, tick=1, record=smf.Meta(kind=0x2F, data=b"")))
    writer.finish()
    with pytest.raises(ValueError, match="SysEx/escape"):
        read_midi(path, converters=[*converters, other_converter])
    for c in [*converters, other_converter]:
        with pytest.raises(RuntimeError, match="restart"):
            c.finish()


def test_file_setup_rejects_empty_or_duplicate_routes_before_consumption(
    tmp_path: Path, converters: list[MidiConverter]
) -> None:
    with pytest.raises(ValueError, match="at least one"):
        read_midi(tmp_path / "unused.mid", converters=[])
    with pytest.raises(ValueError, match="sources must be distinct"):
        read_midi(tmp_path / "unused.mid", converters=converters * 2)
    assert converters[0].ready


def test_unmapped_file_controls_follow_binding_policy(
    tmp_path: Path, converters: list[MidiConverter]
) -> None:
    path = tmp_path / "ignored.mid"
    writer = smf.Writer(path, header=smf.Header(format=0, tracks=1, division=480))
    for m in [
        messages.ControlChange(channel=1, controller=64, value=127),
        messages.ChannelPressure(channel=1, pressure=80),
    ]:
        writer.write(smf.Event(track=0, tick=0, record=m))
    writer.write(smf.Event(track=0, tick=0, record=smf.Meta(kind=0x2F, data=b"")))
    writer.finish()
    assert not any(read_midi(path, converters=converters).parts.values())


def test_file_setup_rejects_part_collisions_without_consuming_converters(
    tmp_path: Path, converters: list[MidiConverter]
) -> None:
    score, _ = Patch(engine="synth").prepare_score(48000)
    binding = performance_binding.PerformanceBindingScore.model_validate(
        {
            "name": "collision",
            "title": "Collision",
            "body": {
                "instrument": {"selector": "test:instrument"},
                "midi": [
                    {
                        "channels": [1],
                        "part": "channel-1",
                        "repeated_key_release": "oldest",
                    }
                ],
            },
        }
    )
    other = MidiConverter(source="distinct", binding=binding, instrument=score)
    with pytest.raises(ValueError, match="parts must be distinct"):
        read_midi(tmp_path / "unused.mid", converters=[*converters, other])
    assert other.ready and converters[0].ready
