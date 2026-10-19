"""Offline MIDI timing and dispatch to configured uFor performance converters."""

from contextlib import closing
from fractions import Fraction
from pathlib import Path

from meedee import messages, smf
from pydantic import BaseModel
from ufor.events import ControlChange, PerformanceEvent, Release, Trigger

from .performance import MidiConverter


class MidiPerformance(BaseModel, frozen=True):
    sample_rate: int
    frames: int
    parts: dict[str, list[PerformanceEvent]]


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
    if not converters:
        raise ValueError("MIDI requires at least one converter")
    sources: set[str] = set()
    parts: dict[str, list[PerformanceEvent]] = {}
    for converter in converters:
        if not converter.ready:
            raise ValueError("MIDI converters must be fresh or explicitly restarted")
        if converter.source in sources:
            raise ValueError("MIDI converter sources must be distinct")
        sources.add(converter.source)
        for part in converter.parts:
            if part in parts:
                raise ValueError("MIDI parts must be distinct across converters")
            parts[part] = []
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
