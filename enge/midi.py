"""File and caller-polled live MIDI dispatch to uFor performance converters."""

from contextlib import closing
from fractions import Fraction
from pathlib import Path

from meedee import events, messages, midi, smf
from pydantic import BaseModel
from ufor.events import ControlChange, PerformanceEvent, Release, Trigger

from .performance import MidiConverter, converter_parts


class MidiPerformance(BaseModel, frozen=True):
    sample_rate: int
    frames: int
    parts: dict[str, list[PerformanceEvent]]


class LiveInput:
    """Drain a caller-owned queue-mode connection on one host control thread.

    Native timestamps are not mapped to frames. Each drain assigns the supplied
    frame to its messages and retains order within each named destination part.
    """

    def __init__(
        self,
        connection: midi.TypedInputConnection,
        *,
        converters: list[MidiConverter],
        message_limit: int = 128,
    ) -> None:
        if type(message_limit) is not int or message_limit < 1:
            raise ValueError("message_limit must be a positive integer")
        self._parts = converter_parts(converters)
        self._connection = connection
        self._converters = converters.copy()
        self._limit = message_limit
        self._ordinal = 0
        self._frame: int | None = None
        self._stopped = False

    def drain(self, *, frame: int) -> dict[str, list[PerformanceEvent]]:
        """Poll a bounded prefix without waiting for incoming MIDI."""
        if self._stopped:
            raise RuntimeError("Live MIDI input is stopped; create a new adapter")
        if type(frame) is not int:
            raise TypeError("frame must be a built-in integer")
        if self._frame is not None and frame < self._frame:
            raise ValueError("Live MIDI frames must not move backward")
        parts: dict[str, list[PerformanceEvent]] = {n: [] for n in self._parts}
        completed = False
        try:
            self._check_connection()
            for _ in range(self._limit):
                item = self._connection.receive(timeout=0)
                self._check_connection()
                if item is None:
                    break
                if isinstance(item, events.StreamDiagnostic):
                    raise ValueError(f"Live MIDI {item.kind}: {item.reason}")
                for converter in self._converters:
                    for event in converter.feed(
                        item.message, tick=frame, ordinal=self._ordinal
                    ):
                        assert isinstance(event, Trigger | Release | ControlChange)
                        assert event.part is not None
                        parts[event.part].append(event)
                self._ordinal += 1
            self._check_connection()
            self._frame = frame
            completed = True
            return parts
        finally:
            if not completed:
                self.stop()

    def stop(self) -> None:
        """Stop observation; leave the connection and musical cleanup to its host."""
        self._stopped = True
        for converter in self._converters:
            converter.report_loss()

    def _check_connection(self) -> None:
        if self._connection.dropped_packets:
            raise RuntimeError(
                "Live MIDI receive loss; restart with a fresh connection"
            )
        if (failure := self._connection.failure) is not None:
            raise RuntimeError("Live MIDI connection failed") from failure
        if self._connection.closed:
            raise RuntimeError("Live MIDI connection is closed")


def read_midi(
    path: Path, *, converters: list[MidiConverter], sample_rate: int = 48000
) -> MidiPerformance:
    """Read synchronous PPQN MIDI through fresh, explicitly configured converters.

    Absolute rational file times round to output frames, ties to even. Binding
    names determine parts. All participants stop on success or become invalid
    on a failed read; the caller must explicitly restart them before reuse.
    """
    if type(sample_rate) is not int or sample_rate <= 0:
        raise ValueError("MIDI requires a positive integer sample rate")
    parts: dict[str, list[PerformanceEvent]] = {
        n: [] for n in converter_parts(converters)
    }
    tempo = 500000
    elapsed = Fraction(0)
    frame = 0
    ordinal = -1
    tick = 0
    completed = False
    try:
        with closing(smf.MergedReader(path)) as midi:
            if midi.header.division & 0x8000:
                raise ValueError("MIDI requires synchronous PPQN timing")
            for event in midi:
                elapsed += Fraction(
                    (event.tick - tick) * tempo, midi.header.division * 1000000
                )
                tick = event.tick
                frame = round(elapsed * sample_rate)
                message = event.record
                # Track ends contribute silence, not merged performance ordinals.
                if isinstance(message, smf.Meta) and message.kind == 0x2F:
                    continue
                ordinal += 1
                if isinstance(message, smf.Meta):
                    if message.kind == 0x51:
                        tempo = int.from_bytes(message.data[:3], "big")
                    continue
                if not isinstance(message, messages.Message):
                    raise ValueError("SMF SysEx/escape records are unsupported")
                for converter in converters:
                    for converted in converter.feed(
                        message, tick=frame, ordinal=ordinal
                    ):
                        assert isinstance(converted, Trigger | Release | ControlChange)
                        assert converted.part is not None
                        parts[converted.part].append(converted)
        for converter in converters:
            converter.finish()
        performance = MidiPerformance(
            sample_rate=sample_rate, frames=frame + 1, parts=parts
        )
        completed = True
        return performance
    finally:
        if not completed:
            for converter in converters:
                converter.report_loss()
