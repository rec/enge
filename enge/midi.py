"""Offline MIDI adaptation to uFor events; no device or transport ownership."""

from collections import defaultdict, deque
from contextlib import closing
from fractions import Fraction
from pathlib import Path

from meedee import messages, smf
from pydantic import BaseModel
from ufor.events import ControlChange, PerformanceEvent, Release, Trigger


class MidiPerformance(BaseModel, frozen=True):
    sample_rate: int
    frames: int
    parts: dict[int, list[PerformanceEvent]]


def read_midi(path: Path, sample_rate: int = 48000) -> MidiPerformance:
    """Read synchronous PPQN MIDI, rounding absolute rational times to frames.

    MIDI channels 1–16 become parts, with FIFO matching for repeated same-key notes.
    CC1/CC11 map to modulation/expression, pitch wheel to bend (±2 semitones in
    the supplied patches). Velocity initializes a trigger-scoped control. Patch
    selection belongs to the caller, so program changes are ignored. Other
    controllers are rejected rather than silently changing the performance.
    """
    if sample_rate <= 0:
        raise ValueError("MIDI requires positive rate and synchronous PPQN timing")
    parts: dict[int, list[PerformanceEvent]] = defaultdict(list)
    held: dict[tuple[int, int], deque[str]] = defaultdict(deque)
    tempo = 500000
    elapsed = Fraction(0)
    frame = 0
    ordinal = -1
    tick = 0
    with closing(smf.MergedReader(path)) as midi:
        if midi.header.division & 0x8000:
            raise ValueError("MIDI requires positive rate and synchronous PPQN timing")
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
            if isinstance(message, messages.ProgramChange):
                continue
            elif isinstance(message, messages.NoteOn) and message.velocity:
                name = f"note-{ordinal}"
                held[message.channel, message.note].append(name)
                parts[message.channel].append(
                    Trigger(
                        tick=frame,
                        ordinal=ordinal,
                        part=f"channel-{message.channel}",
                        trigger_id=name,
                        key=message.note,
                        velocity=message.velocity / 127,
                        pitch_hz=440 * 2 ** ((message.note - 69) / 12),
                        controls={"velocity": message.velocity / 127},
                    )
                )
            elif isinstance(message, messages.NoteOn | messages.NoteOff):
                queue = held[message.channel, message.note]
                if not queue:
                    raise ValueError(f"Unmatched MIDI note-off at frame {frame}")
                parts[message.channel].append(
                    Release(
                        tick=frame,
                        ordinal=ordinal,
                        part=f"channel-{message.channel}",
                        trigger_id=queue.popleft(),
                    )
                )
            elif isinstance(message, messages.ControlChange | messages.PitchBend):
                if isinstance(message, messages.PitchBend):
                    control, value = "bend", message.value / 8192
                elif message.controller in (1, 11):
                    control = "modulation" if message.controller == 1 else "expression"
                    value = message.value / 127
                else:
                    raise ValueError(
                        f"Unsupported MIDI controller {message.controller}"
                    )
                parts[message.channel].append(
                    ControlChange(
                        tick=frame,
                        ordinal=ordinal,
                        part=f"channel-{message.channel}",
                        scope="part",
                        control=control,
                        value=value,
                    )
                )
            else:
                raise ValueError(f"Unsupported MIDI message: {type(message).__name__}")
    if any(held.values()):
        raise ValueError("MIDI ends with unreleased notes")
    return MidiPerformance(sample_rate=sample_rate, frames=frame + 1, parts=dict(parts))
