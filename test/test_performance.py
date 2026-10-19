from typing import Literal

import pytest
from meedee import messages, ump
from test_sample_instrument import sample_score
from test_synth import score
from ufor import events, performance_binding
from ufor.interface import ScoreReference
from ufor.samples.instrument import SampleInstrumentScore
from ufor.synth import SynthInstrumentScore
from ufor.tuning import FrequencyTable, Tuning

from enge.performance import MidiConverter


@pytest.fixture
def instrument() -> SynthInstrumentScore:
    raw = score().model_dump(mode="json")
    raw["body"]["controls"] = {
        "expression": {"default": 0.8},
        "bend": {"polarity": "bipolar"},
    }
    return SynthInstrumentScore.model_validate(raw)


@pytest.fixture
def binding() -> performance_binding.PerformanceBindingScore:
    return performance_binding.PerformanceBindingScore(
        name="input",
        title="MIDI input",
        body=performance_binding.PerformanceInputBinding(
            instrument=ScoreReference(path="instrument.toml"),
            midi=[
                performance_binding.MidiPerformanceInput(
                    channels=[1, 2],
                    part="main",
                    repeated_key_release="oldest",
                    controllers=[
                        performance_binding.MidiController(
                            number=11, control="expression"
                        )
                    ],
                    pitch_bend=performance_binding.MidiPitchBend(control="bend"),
                )
            ],
        ),
    )


@pytest.fixture
def converter(
    binding: performance_binding.PerformanceBindingScore,
    instrument: SynthInstrumentScore,
) -> MidiConverter:
    return MidiConverter(source="keyboard", binding=binding, instrument=instrument)


def test_notes_preserve_exact_host_time_velocity_and_source_identity(
    converter: MidiConverter,
    binding: performance_binding.PerformanceBindingScore,
    instrument: SynthInstrumentScore,
) -> None:
    tick = 2**70
    first = converter.feed(
        messages.NoteOn(channel=1, note=69, velocity=80), tick=tick, ordinal=7
    )[0]
    second = converter.feed(
        messages.NoteOn(channel=2, note=69, velocity=100), tick=tick, ordinal=8
    )[0]
    assert isinstance(first, events.Trigger)
    assert isinstance(second, events.Trigger)
    assert first.tick == tick
    assert first.ordinal == 7
    assert first.part == "main"
    assert first.pitch_hz == 440
    assert first.velocity == 80 / 127
    assert first.controls == {}
    assert first.trigger_id != second.trigger_id
    other = MidiConverter(source="other", binding=binding, instrument=instrument)
    third = other.feed(
        messages.NoteOn(channel=1, note=69, velocity=80), tick=tick, ordinal=7
    )[0]
    assert isinstance(third, events.Trigger)
    assert third.trigger_id != first.trigger_id
    release = converter.feed(
        messages.NoteOff(channel=2, note=69, velocity=64), tick=tick + 1, ordinal=9
    )[0]
    assert isinstance(release, events.Release)
    assert release.trigger_id == second.trigger_id
    release = converter.feed(
        messages.NoteOn(channel=1, note=69, velocity=0), tick=tick + 1, ordinal=10
    )[0]
    assert isinstance(release, events.Release)
    assert release.trigger_id == first.trigger_id
    converter.finish()


@pytest.mark.parametrize("policy", ["oldest", "newest"])
def test_repeated_keys_follow_declared_pairing(
    binding: performance_binding.PerformanceBindingScore,
    instrument: SynthInstrumentScore,
    policy: Literal["oldest", "newest"],
) -> None:
    raw = binding.model_dump()
    raw["body"]["midi"][0]["repeated_key_release"] = policy
    converter = MidiConverter(
        source="keys",
        binding=performance_binding.PerformanceBindingScore.model_validate(raw),
        instrument=instrument,
    )
    starts = [
        converter.feed(
            messages.NoteOn(channel=1, note=60, velocity=80), tick=0, ordinal=i
        )[0]
        for i in range(2)
    ]
    releases = [
        converter.feed(
            messages.NoteOff(channel=1, note=60, velocity=64), tick=1, ordinal=i
        )[0]
        for i in range(2)
    ]
    assert all(
        isinstance(e, events.Trigger | events.Release) for e in starts + releases
    )
    expected = starts if policy == "oldest" else list(reversed(starts))
    assert [e.trigger_id for e in releases if isinstance(e, events.Release)] == [
        e.trigger_id for e in expected if isinstance(e, events.Trigger)
    ]
    converter.finish()


def test_controls_use_declared_ranges_and_explicit_bend_center(
    binding: performance_binding.PerformanceBindingScore,
    instrument: SynthInstrumentScore,
) -> None:
    raw = binding.model_dump()
    raw["body"]["midi"][0]["pitch_bend"].update(minimum=-0.3, center=0.7, maximum=0.9)
    converter = MidiConverter(
        source="keys",
        binding=performance_binding.PerformanceBindingScore.model_validate(raw),
        instrument=instrument,
    )
    for i, (v, expected) in enumerate(
        [(-8192, -0.3), (-4096, 0.2), (0, 0.7), (8191, 0.9)]
    ):
        event = converter.feed(
            messages.PitchBend(channel=1, value=v), tick=0, ordinal=i
        )[0]
        assert isinstance(event, events.ControlChange)
        assert event.control == "bend"
        assert event.part == "main"
        assert event.scope == "part"
        if v == -4096:
            assert event.value == pytest.approx(expected)
        else:
            assert event.value == expected
    for i, v in enumerate([0, 64, 127], start=4):
        event = converter.feed(
            messages.ControlChange(channel=1, controller=11, value=v), tick=0, ordinal=i
        )[0]
        assert isinstance(event, events.ControlChange)
        assert event.control == "expression"
        assert event.value == v / 127


def test_unmapped_messages_have_no_implicit_musical_effect(
    converter: MidiConverter, binding: performance_binding.PerformanceBindingScore
) -> None:
    ignored = [
        messages.ControlChange(channel=1, controller=64, value=127),
        messages.NoteOff(channel=16, note=60, velocity=0),
        messages.ProgramChange(channel=1, program=2),
        messages.ChannelPressure(channel=1, pressure=127),
        messages.TimingClock(),
        messages.SysEx(data=b"\x7d\x01"),
    ]
    for i, m in enumerate(ignored):
        assert converter.feed(m, tick=0, ordinal=i) == []
    binding.body.midi[0].channels.clear()
    event = converter.feed(
        messages.NoteOn(channel=1, note=60, velocity=80), tick=1, ordinal=0
    )[0]
    assert isinstance(event, events.Trigger)


def test_absent_bend_mapping_does_not_invent_a_target(
    binding: performance_binding.PerformanceBindingScore,
    instrument: SynthInstrumentScore,
) -> None:
    raw = binding.model_dump()
    raw["body"]["midi"][0]["pitch_bend"] = None
    converter = MidiConverter(
        source="keys",
        binding=performance_binding.PerformanceBindingScore.model_validate(raw),
        instrument=instrument,
    )
    assert (
        converter.feed(messages.PitchBend(channel=1, value=8191), tick=0, ordinal=0)
        == []
    )


def test_named_parts_remain_distinct_and_releases_free_admission(
    binding: performance_binding.PerformanceBindingScore,
    instrument: SynthInstrumentScore,
) -> None:
    raw = binding.model_dump()
    entry = raw["body"]["midi"][0]
    entry["channels"] = [1]
    raw["body"]["midi"].append(entry | {"channels": [2], "part": "other"})
    converter = MidiConverter(
        source="keys",
        binding=performance_binding.PerformanceBindingScore.model_validate(raw),
        instrument=instrument,
        max_held_notes=2,
    )
    for i, c in enumerate([1, 2]):
        event = converter.feed(
            messages.NoteOn(channel=c, note=60, velocity=80), tick=0, ordinal=i
        )[0]
        assert isinstance(event, events.Trigger)
        assert event.part == ("main" if c == 1 else "other")
    converter.feed(messages.NoteOff(channel=1, note=60, velocity=0), tick=1, ordinal=0)
    converter.feed(messages.NoteOn(channel=2, note=62, velocity=80), tick=2, ordinal=0)
    control = converter.feed(
        messages.ControlChange(channel=2, controller=11, value=127), tick=3, ordinal=0
    )[0]
    assert isinstance(control, events.ControlChange)
    assert control.part == "other"
    for i, n in enumerate([60, 62]):
        converter.feed(
            messages.NoteOff(channel=2, note=n, velocity=0), tick=4, ordinal=i
        )
    converter.finish()


@pytest.mark.parametrize("change", ["unknown", "range"])
def test_mappings_must_match_instrument_control_declarations(
    binding: performance_binding.PerformanceBindingScore,
    instrument: SynthInstrumentScore,
    change: str,
) -> None:
    raw = binding.model_dump()
    mapping = raw["body"]["midi"][0]["controllers"][0]
    if change == "unknown":
        mapping["control"] = "unknown"
    else:
        mapping["minimum"] = -1.0
    with pytest.raises(ValueError):
        MidiConverter(
            source="keys",
            binding=performance_binding.PerformanceBindingScore.model_validate(raw),
            instrument=instrument,
        )


def test_sample_instrument_uses_its_shared_control_declarations(
    binding: performance_binding.PerformanceBindingScore,
    instrument: SynthInstrumentScore,
) -> None:
    raw = sample_score().model_dump(mode="json")
    raw["body"]["settings"]["controls"] = instrument.body.model_dump()["controls"]
    converter = MidiConverter(
        source="keys",
        binding=binding,
        instrument=SampleInstrumentScore.model_validate(raw),
    )
    event = converter.feed(
        messages.ControlChange(channel=1, controller=11, value=127), tick=0, ordinal=0
    )[0]
    assert isinstance(event, events.ControlChange)
    assert event.value == 1


def test_tuning_references_are_required_snapshotted_and_never_wrapped(
    binding: performance_binding.PerformanceBindingScore,
    instrument: SynthInstrumentScore,
) -> None:
    raw = binding.model_dump()
    raw["body"]["midi"][0]["tuning"] = {"path": "tuning.toml"}
    binding = performance_binding.PerformanceBindingScore.model_validate(raw)
    with pytest.raises(ValueError, match="Unresolved tuning"):
        MidiConverter(source="keys", binding=binding, instrument=instrument)
    tuning = Tuning(source=FrequencyTable(first_note=69, values=["432"]))
    converter = MidiConverter(
        source="keys",
        binding=binding,
        instrument=instrument,
        tunings={"tuning.toml": tuning},
    )
    assert isinstance(tuning.source, FrequencyTable)
    tuning.source.values.clear()
    event = converter.feed(
        messages.NoteOn(channel=1, note=69, velocity=80), tick=0, ordinal=0
    )[0]
    assert isinstance(event, events.Trigger)
    assert event.pitch_hz == 432
    with pytest.raises(ValueError, match="Cannot convert MIDI note"):
        converter.feed(
            messages.NoteOn(channel=1, note=70, velocity=80), tick=1, ordinal=0
        )
    with pytest.raises(RuntimeError, match="restart"):
        converter.finish()


@pytest.mark.parametrize("limit", [2, 4096])
def test_note_admission_is_bounded_and_restart_never_reuses_ids(
    binding: performance_binding.PerformanceBindingScore,
    instrument: SynthInstrumentScore,
    limit: int,
) -> None:
    converter = (
        MidiConverter(source="keys", binding=binding, instrument=instrument)
        if limit == 4096
        else MidiConverter(
            source="keys", binding=binding, instrument=instrument, max_held_notes=limit
        )
    )
    for i in range(limit):
        last = converter.feed(
            messages.NoteOn(channel=1, note=60, velocity=80), tick=0, ordinal=i
        )[0]
    assert isinstance(last, events.Trigger)
    with pytest.raises(ValueError, match="held-note limit"):
        converter.feed(
            messages.NoteOn(channel=1, note=61, velocity=80), tick=1, ordinal=0
        )
    with pytest.raises(RuntimeError, match="restart"):
        converter.feed(
            messages.NoteOff(channel=1, note=60, velocity=0), tick=2, ordinal=0
        )
    converter.restart()
    fresh = converter.feed(
        messages.NoteOn(channel=1, note=60, velocity=80), tick=0, ordinal=0
    )[0]
    assert isinstance(fresh, events.Trigger)
    assert fresh.trigger_id != last.trigger_id
    converter.feed(messages.NoteOff(channel=1, note=60, velocity=0), tick=1, ordinal=0)
    converter.finish()


@pytest.mark.parametrize("reset", [False, True])
def test_reported_loss_or_system_reset_requires_explicit_restart(
    converter: MidiConverter, reset: bool
) -> None:
    converter.feed(messages.NoteOn(channel=1, note=60, velocity=80), tick=0, ordinal=0)
    if reset:
        with pytest.raises(ValueError, match="System Reset"):
            converter.feed(messages.SystemReset(), tick=1, ordinal=0)
    else:
        converter.report_loss()
    with pytest.raises(RuntimeError, match="restart"):
        converter.feed(
            messages.NoteOff(channel=1, note=60, velocity=0), tick=2, ordinal=0
        )
    converter.restart()
    with pytest.raises(ValueError, match="Unmatched"):
        converter.feed(
            messages.NoteOff(channel=1, note=60, velocity=0), tick=0, ordinal=0
        )
    with pytest.raises(RuntimeError, match="restart"):
        converter.finish()


def test_end_of_source_reports_unreleased_notes(converter: MidiConverter) -> None:
    converter.feed(messages.NoteOn(channel=1, note=60, velocity=80), tick=0, ordinal=0)
    with pytest.raises(ValueError, match="unreleased notes"):
        converter.finish()
    converter.restart()
    converter.finish()
    with pytest.raises(RuntimeError, match="restart"):
        converter.feed(messages.TimingClock(), tick=1, ordinal=0)


def test_out_of_order_input_invalidates_pairing(converter: MidiConverter) -> None:
    converter.feed(messages.NoteOn(channel=1, note=60, velocity=80), tick=10, ordinal=1)
    with pytest.raises(ValueError, match="out of order"):
        converter.feed(
            messages.NoteOff(channel=1, note=60, velocity=0), tick=10, ordinal=1
        )
    with pytest.raises(RuntimeError, match="restart"):
        converter.finish()


@pytest.mark.parametrize(
    "message",
    [ump.Noop(), messages.ControlChange14(channel=1, controller=1, value=100)],
)
def test_nonprimitive_or_ump_messages_fail_explicitly(
    converter: MidiConverter, message: messages.Message
) -> None:
    with pytest.raises(ValueError, match="primitive MIDI 1"):
        converter.feed(message, tick=0, ordinal=0)
    with pytest.raises(RuntimeError, match="restart"):
        converter.finish()
