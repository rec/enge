from collections import deque
from typing import cast

import pytest
from meedee import events, messages, midi
from ufor import performance_binding
from ufor.events import ControlChange, Release, Trigger

from enge.midi import LiveInput
from enge.performance import MidiConverter
from enge.presets import Patch


class InputQueue:
    def __init__(self, items: list[events.Event | events.StreamDiagnostic]) -> None:
        self.items = deque(items)
        self.dropped_packets = 0
        self.closed = False
        self.failure: Exception | None = None
        self.fail_after_read = False

    def receive(
        self, *, timeout: float | None = None
    ) -> events.Event | events.StreamDiagnostic | None:
        assert timeout == 0
        item = self.items.popleft() if self.items else None
        if self.fail_after_read:
            self.failure = RuntimeError("worker failed during receive")
        return item


@pytest.fixture
def converters() -> list[MidiConverter]:
    score, _ = Patch(engine="synth").prepare_score(48000)
    binding = performance_binding.PerformanceBindingScore.model_validate(
        {
            "name": "input",
            "title": "Live input",
            "body": {
                "instrument": {"selector": "test:instrument"},
                "midi": [
                    {
                        "channels": [1, 2],
                        "part": "main",
                        "repeated_key_release": "oldest",
                        "controllers": [{"number": 11, "control": "expression"}],
                        "pitch_bend": {"control": "bend"},
                    }
                ],
            },
        }
    )
    return [MidiConverter(source="keyboard", binding=binding, instrument=score)]


def packet(message: messages.Message) -> events.Event:
    return events.Event(
        stream="test",
        message=message,
        timestamp=events.Timestamp(
            ticks=-1000000, clock=events.Clock(identity="unrelated-clock")
        ),
    )


def test_live_drain_is_bounded_and_uses_host_frames_instead_of_native_time(
    converters: list[MidiConverter],
) -> None:
    queue = InputQueue(
        [
            packet(messages.NoteOn(channel=1, note=69, velocity=80)),
            packet(messages.ControlChange(channel=1, controller=11, value=100)),
            packet(messages.NoteOff(channel=1, note=69, velocity=0)),
        ]
    )
    adapter = LiveInput(
        cast(midi.TypedInputConnection, queue), converters=converters, message_limit=2
    )
    first = adapter.drain(frame=480)
    assert [e.tick for e in first["main"]] == [480, 480]
    assert [e.ordinal for e in first["main"]] == [0, 1]
    start, control = first["main"]
    assert isinstance(start, Trigger)
    assert start.velocity == 80 / 127
    assert isinstance(control, ControlChange)
    assert control.value == 100 / 127
    assert len(queue.items) == 1
    second = adapter.drain(frame=960)["main"]
    assert len(second) == 1 and isinstance(second[0], Release)
    assert second[0].tick == 960
    assert second[0].ordinal == 2
    assert second[0].trigger_id == start.trigger_id
    assert adapter.drain(frame=960) == {"main": []}
    converters[0].finish()
    adapter.stop()
    assert not queue.closed


def test_live_ignored_messages_still_count_toward_drain_budget(
    converters: list[MidiConverter],
) -> None:
    queue = InputQueue(
        [
            packet(messages.ControlChange(channel=1, controller=64, value=127)),
            packet(messages.NoteOn(channel=1, note=69, velocity=80)),
        ]
    )
    adapter = LiveInput(
        cast(midi.TypedInputConnection, queue), converters=converters, message_limit=1
    )
    assert adapter.drain(frame=0) == {"main": []}
    start = adapter.drain(frame=0)["main"][0]
    assert start.ordinal == 1
    adapter.stop()


def test_bad_host_frame_does_not_consume_or_stop_input(
    converters: list[MidiConverter],
) -> None:
    queue = InputQueue([])
    adapter = LiveInput(cast(midi.TypedInputConnection, queue), converters=converters)
    assert adapter.drain(frame=100) == {"main": []}
    queue.items.append(packet(messages.NoteOn(channel=1, note=69, velocity=80)))
    with pytest.raises(ValueError, match="backward"):
        adapter.drain(frame=99)
    assert len(queue.items) == 1
    assert adapter.drain(frame=100)["main"]
    with pytest.raises(TypeError, match="integer"):
        adapter.drain(frame=True)
    adapter.stop()


@pytest.mark.parametrize("failure", ["closed", "worker", "during_receive"])
def test_connection_failure_rejects_queued_prefix_and_requires_recovery(
    converters: list[MidiConverter], failure: str
) -> None:
    queue = InputQueue([packet(messages.NoteOn(channel=1, note=69, velocity=80))])
    adapter = LiveInput(cast(midi.TypedInputConnection, queue), converters=converters)
    if failure == "closed":
        queue.closed = True
    elif failure == "worker":
        queue.failure = RuntimeError("worker failed")
    else:
        queue.fail_after_read = True
    with pytest.raises(RuntimeError, match="connection"):
        adapter.drain(frame=0)
    with pytest.raises(RuntimeError, match="stopped"):
        adapter.drain(frame=1)
    with pytest.raises(RuntimeError, match="restart"):
        converters[0].finish()
    assert queue.closed == (failure == "closed")


def test_quiet_receive_loss_stops_adapter_without_a_following_packet(
    converters: list[MidiConverter],
) -> None:
    queue = InputQueue([])
    adapter = LiveInput(cast(midi.TypedInputConnection, queue), converters=converters)
    assert adapter.drain(frame=0) == {"main": []}
    queue.dropped_packets = 1
    with pytest.raises(RuntimeError, match="receive loss"):
        adapter.drain(frame=1)
    with pytest.raises(RuntimeError, match="stopped"):
        adapter.drain(frame=2)
    assert not queue.closed
    assert not converters[0].ready


def test_input_diagnostic_discards_current_batch_and_invalidates_conversion(
    converters: list[MidiConverter],
) -> None:
    diagnostic = events.PacketDecoder(stream="test", protocol=midi.Protocol.Midi1).loss(
        lost=1, reason="test loss"
    )
    queue = InputQueue(
        [packet(messages.NoteOn(channel=1, note=69, velocity=80)), diagnostic]
    )
    adapter = LiveInput(cast(midi.TypedInputConnection, queue), converters=converters)
    with pytest.raises(ValueError, match="test loss"):
        adapter.drain(frame=0)
    with pytest.raises(RuntimeError, match="restart"):
        converters[0].feed(
            messages.NoteOff(channel=1, note=69, velocity=0), tick=1, ordinal=1
        )
    assert not queue.closed


def test_conversion_error_stops_adapter_without_closing_caller_connection(
    converters: list[MidiConverter],
) -> None:
    queue = InputQueue([packet(messages.NoteOff(channel=1, note=69, velocity=0))])
    adapter = LiveInput(cast(midi.TypedInputConnection, queue), converters=converters)
    with pytest.raises(ValueError, match="Unmatched"):
        adapter.drain(frame=0)
    assert not queue.closed
    with pytest.raises(RuntimeError, match="stopped"):
        adapter.drain(frame=1)


def test_stop_is_final_and_never_releases_owned_notes(
    converters: list[MidiConverter],
) -> None:
    queue = InputQueue([packet(messages.NoteOn(channel=1, note=69, velocity=80))])
    adapter = LiveInput(cast(midi.TypedInputConnection, queue), converters=converters)
    assert adapter.drain(frame=0)["main"]
    adapter.stop()
    adapter.stop()
    assert not queue.closed
    with pytest.raises(RuntimeError, match="stopped"):
        adapter.drain(frame=1)
    with pytest.raises(RuntimeError, match="restart"):
        converters[0].finish()


@pytest.mark.parametrize("limit", [0, -1, True])
def test_invalid_drain_limit_does_not_consume_converters(
    converters: list[MidiConverter], limit: int
) -> None:
    queue = InputQueue([])
    with pytest.raises(ValueError, match="positive integer"):
        LiveInput(
            cast(midi.TypedInputConnection, queue),
            converters=converters,
            message_limit=limit,
        )
    assert converters[0].ready
