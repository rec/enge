"""Caller-driven MIDI 1 conversion using resolved uFor performance bindings."""

from collections import deque
from typing import NoReturn

from meedee import messages
from ufor import events, performance_binding
from ufor.base import identifier
from ufor.samples.instrument import SampleInstrumentScore
from ufor.synth import SynthInstrumentScore
from ufor.tuning import Tuning


class MidiConverter:
    """Pair notes from one source; own no transport, clock, files or audio.

    The host resolves references and supplies destination ticks and ordinals in
    strictly increasing (tick, ordinal) order. Source names must be unique among
    converters whose output will be merged. Restart preserves the ID counter.
    The host must serialize all calls on an instance.
    """

    def __init__(
        self,
        *,
        source: str,
        binding: performance_binding.PerformanceBindingScore,
        instrument: SynthInstrumentScore | SampleInstrumentScore,
        tunings: dict[str, Tuning] | None = None,
        max_held_notes: int = 4096,
    ) -> None:
        if type(max_held_notes) is not int or max_held_notes < 1:
            raise ValueError("max_held_notes must be a positive integer")
        self._source = identifier(source)
        self._limit = max_held_notes
        self._inputs: dict[int, performance_binding.MidiPerformanceInput] = {}
        self._controllers: dict[
            tuple[int, int], performance_binding.MidiController
        ] = {}
        self._tunings: dict[int, Tuning] = {}
        declarations = (
            instrument.body.settings
            if isinstance(instrument, SampleInstrumentScore)
            else instrument.body
        )
        for entry in binding.model_copy(deep=True).body.midi:
            for mapping in [
                *entry.controllers,
                *([entry.pitch_bend] if entry.pitch_bend else []),
            ]:
                declared = declarations.require_control(mapping.control)
                declared.validate_value(mapping.minimum)
                declared.validate_value(mapping.maximum)
            tuning = None
            if entry.tuning is not None:
                key = entry.tuning.key
                if tunings is None or key not in tunings:
                    raise ValueError(f"Unresolved tuning: {key}")
                tuning = tunings[key].model_copy(deep=True)
            for channel in entry.channels:
                self._inputs[channel] = entry
                if tuning is not None:
                    self._tunings[channel] = tuning
                for mapping in entry.controllers:
                    self._controllers[channel, mapping.number] = mapping
        self._held: dict[tuple[int, int], deque[str]] = {}
        self._count = 0
        self._next_id = 0
        self._failed = False
        self._last: tuple[int, int] | None = None

    @property
    def source(self) -> str:
        """The host-selected identity used to namespace emitted triggers."""
        return self._source

    @property
    def parts(self) -> list[str]:
        """Configured destination names, including parts with no observations."""
        return list(dict.fromkeys(e.part for e in self._inputs.values()))

    @property
    def ready(self) -> bool:
        """Whether a new observation sequence can start without implicit reset."""
        return not self._failed and self._last is None

    def feed(
        self, message: messages.Message, *, tick: int, ordinal: int
    ) -> list[events.PerformanceEvent]:
        """Convert one decoded message; never infer missing notes or controls."""
        if self._failed:
            raise RuntimeError("MIDI converter requires explicit restart")
        if not isinstance(message, messages.Message):
            raise TypeError("message must be a meedee Message")
        time = events.Event(tick=tick, ordinal=ordinal)
        order = time.tick, time.ordinal
        if self._last is not None and order <= self._last:
            self._fail("MIDI input is out of order")
        self._last = order
        if isinstance(message, messages.SystemReset):
            self._fail("MIDI System Reset invalidates performance state")
        if not isinstance(
            message,
            messages.NoteOn
            | messages.NoteOff
            | messages.ControlChange
            | messages.PitchBend,
        ):
            if isinstance(message, _IGNORED_MESSAGES):
                return []
            self._fail("MIDI converter requires a primitive MIDI 1 message")
        entry = self._inputs.get(message.channel)
        if entry is None:
            return []
        if isinstance(message, messages.NoteOn) and message.velocity:
            if self._count >= self._limit:
                self._fail("MIDI held-note limit exceeded")
            tuning = self._tunings.get(message.channel)
            try:
                pitch = (
                    float(tuning(message.note))
                    if tuning is not None
                    else 440 * 2 ** ((message.note - 69) / 12)
                )
                event = events.Trigger(
                    tick=tick,
                    ordinal=ordinal,
                    part=entry.part,
                    trigger_id=f"{self._source}-note-{self._next_id}",
                    key=message.note,
                    velocity=message.velocity / 127,
                    pitch_hz=pitch,
                )
            except (ValueError, OverflowError) as error:
                self.report_loss()
                raise ValueError(
                    f"Cannot convert MIDI note {message.note}: {error}"
                ) from error
            self._held.setdefault((message.channel, message.note), deque()).append(
                event.trigger_id
            )
            self._next_id += 1
            self._count += 1
            return [event]
        if isinstance(message, messages.NoteOn | messages.NoteOff):
            key = message.channel, message.note
            queue = self._held.get(key)
            if not queue:
                self._fail("Unmatched MIDI note-off")
            oldest = entry.repeated_key_release == "oldest"
            release = events.Release(
                tick=tick,
                ordinal=ordinal,
                part=entry.part,
                trigger_id=queue[0] if oldest else queue[-1],
            )
            if oldest:
                queue.popleft()
            else:
                queue.pop()
            if not queue:
                del self._held[key]
            self._count -= 1
            return [release]
        if isinstance(message, messages.ControlChange):
            mapping = self._controllers.get((message.channel, message.controller))
            if mapping is None:
                return []
            control = mapping.control
            value = (
                mapping.maximum
                if message.value == 127
                else (
                    mapping.minimum
                    + (mapping.maximum - mapping.minimum) * message.value / 127
                )
            )
        else:
            bend = entry.pitch_bend
            if bend is None:
                return []
            control = bend.control
            if message.value == -8192:
                value = bend.minimum
            elif message.value == 8191:
                value = bend.maximum
            else:
                value = (
                    bend.center + (bend.center - bend.minimum) * message.value / 8192
                    if message.value < 0
                    else bend.center
                    + (bend.maximum - bend.center) * message.value / 8191
                )
        return [
            events.ControlChange(
                tick=tick,
                ordinal=ordinal,
                part=entry.part,
                scope="part",
                control=control,
                value=value,
            )
        ]

    def report_loss(self) -> None:
        """Invalidate known pairing without emitting speculative releases."""
        self._held.clear()
        self._count = 0
        self._failed = True

    def restart(self) -> None:
        """Start fresh observation state; the host owns musical recovery."""
        self._held.clear()
        self._count = 0
        self._failed = False
        self._last = None

    def finish(self) -> None:
        """Require all observed notes released, then stop until restart."""
        if self._failed:
            raise RuntimeError("MIDI converter requires explicit restart")
        if self._held:
            self._fail("MIDI source ends with unreleased notes")
        self._failed = True

    def _fail(self, message: str) -> NoReturn:
        self.report_loss()
        raise ValueError(message)


_IGNORED_MESSAGES = (
    messages.PolyPressure,
    messages.ChannelPressure,
    messages.ProgramChange,
    messages.SysEx,
    messages.QuarterFrame,
    messages.SongPosition,
    messages.SongSelect,
    messages.TuneRequest,
    messages.TimingClock,
    messages.Start,
    messages.Continue,
    messages.Stop,
    messages.ActiveSensing,
)
