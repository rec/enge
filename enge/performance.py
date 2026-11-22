"""Caller-driven MIDI 1 conversion using resolved uFor performance bindings."""

from collections.abc import Callable
from json import dumps
from typing import NoReturn

from middy import messages
from ufor import events, performance_binding
from ufor.base import identifier
from ufor.samples.controls import ControlDeclaration
from ufor.samples.instrument import SampleInstrumentScore
from ufor.synth import SynthInstrumentScore
from ufor.tuning import Tuning

from ._native import MidiHeldNotes, PreparedEventValidator, PreparedMidiBindings


def prepare_event_validator(
    controls: dict[str, ControlDeclaration],
) -> Callable[[events.PerformanceEvent], None]:
    validator = PreparedEventValidator(
        dumps({n: c.model_dump(mode="json") for n, c in controls.items()})
    )

    def validate(event: events.PerformanceEvent) -> None:
        validator.validate(event.model_dump_json(exclude={"tick", "ordinal"}))

    return validate


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
        self._notes = MidiHeldNotes(identifier(source), max_held_notes)
        entries = binding.body.midi
        declarations = (
            instrument.body.settings
            if isinstance(instrument, SampleInstrumentScore)
            else instrument.body
        )
        tuning_keys = {e.tuning.key for e in entries if e.tuning is not None}
        sources = tunings if tunings is not None else {}
        self._bindings = PreparedMidiBindings(
            binding.model_dump_json(),
            dumps(
                {
                    "controls": {
                        n: c.model_dump(mode="json")
                        for n, c in declarations.controls.items()
                    },
                    "tunings": {
                        k: sources[k].model_dump(mode="json")
                        for k in sorted(tuning_keys)
                        if k in sources
                    },
                }
            ),
        )
        self._last: tuple[int, int] | None = None

    @property
    def source(self) -> str:
        """The host-selected identity used to namespace emitted triggers."""
        return self._notes.source()

    @property
    def parts(self) -> list[str]:
        """Configured destination names, including parts with no observations."""
        return self._bindings.parts()

    @property
    def ready(self) -> bool:
        """Whether a new observation sequence can start without implicit reset."""
        return not self._notes.stopped() and self._last is None

    def feed(
        self, message: messages.Message, *, tick: int, ordinal: int
    ) -> list[events.PerformanceEvent]:
        """Convert one decoded message; never infer missing notes or controls."""
        if self._notes.stopped():
            raise RuntimeError("MIDI converter requires explicit restart")
        if not isinstance(message, messages.Message):
            raise TypeError("message must be a middy Message")
        time = events.Event(tick=tick, ordinal=ordinal)
        order = time.tick, time.ordinal
        if self._last is not None and order <= self._last:
            self._fail("MIDI input is out of order")
        self._last = order
        if isinstance(message, messages.SystemReset):
            self._fail("MIDI System Reset invalidates performance state")
        if isinstance(message, _CHANNEL_MESSAGES) and (
            message.version != 1
            or message.group is not None
            or isinstance(message, messages.ControlChange)
            and message.resolution != 7
        ):
            self._fail("MIDI converter requires a primitive MIDI 1 message")
        if not isinstance(message, _PERFORMANCE_MESSAGES):
            if isinstance(message, _IGNORED_MESSAGES):
                return []
            self._fail("MIDI converter requires a primitive MIDI 1 message")
        channel = message.channel
        if channel is None:
            self._fail("MIDI converter requires a channel on channel messages")
        if isinstance(message, messages.ControlChange | messages.PitchBend):
            mapped = (
                self._bindings.control_change(
                    channel, message.controller, message.value
                )
                if isinstance(message, messages.ControlChange)
                else self._bindings.pitch_bend(channel, message.value)
            )
            if mapped is None:
                return []
            part, control, value = mapped
            return [
                events.ControlChange(
                    tick=tick,
                    ordinal=ordinal,
                    part=part,
                    scope="part",
                    control=control,
                    value=value,
                )
            ]
        route = self._bindings.note_route(channel)
        if route is None:
            return []
        part, oldest = route
        if isinstance(message, messages.NoteOn) and message.velocity:
            trigger_id, velocity, pitch = self._notes.note_on(
                channel,
                message.note,
                message.velocity,
                self._bindings,
            )
            return [
                events.Trigger(
                    tick=time.tick,
                    ordinal=time.ordinal,
                    part=part,
                    trigger_id=trigger_id,
                    key=message.note,
                    velocity=velocity,
                    pitch_hz=pitch,
                )
            ]
        return [
            events.Release(
                tick=time.tick,
                ordinal=time.ordinal,
                part=part,
                trigger_id=self._notes.note_off(channel, message.note, oldest),
            )
        ]

    def report_loss(self) -> None:
        """Invalidate known pairing without emitting speculative releases."""
        self._notes.report_loss()

    def restart(self) -> None:
        """Start fresh observation state; the host owns musical recovery."""
        self._notes.restart()
        self._last = None

    def finish(self) -> None:
        """Require all observed notes released, then stop until restart."""
        if self._notes.stopped():
            raise RuntimeError("MIDI converter requires explicit restart")
        self._notes.finish()

    def _fail(self, message: str) -> NoReturn:
        self.report_loss()
        raise ValueError(message)


def converter_parts(converters: list[MidiConverter]) -> list[str]:
    """Validate a fresh converter group and return its distinct named routes."""
    if not converters:
        raise ValueError("MIDI requires at least one converter")
    sources: set[str] = set()
    parts: list[str] = []
    for converter in converters:
        if not converter.ready:
            raise ValueError("MIDI converters must be fresh or explicitly restarted")
        if converter.source in sources:
            raise ValueError("MIDI converter sources must be distinct")
        sources.add(converter.source)
        for part in converter.parts:
            if part in parts:
                raise ValueError("MIDI parts must be distinct across converters")
            parts.append(part)
    return parts


_IGNORED_CHANNEL_MESSAGES = (
    messages.PolyPressure,
    messages.ChannelPressure,
    messages.ProgramChange,
)

_PERFORMANCE_MESSAGES = (
    messages.NoteOn,
    messages.NoteOff,
    messages.ControlChange,
    messages.PitchBend,
)

_CHANNEL_MESSAGES = (*_PERFORMANCE_MESSAGES, *_IGNORED_CHANNEL_MESSAGES)

_IGNORED_MESSAGES = (
    *_IGNORED_CHANNEL_MESSAGES,
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
