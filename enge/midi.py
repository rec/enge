"""Offline MIDI adaptation to uFor events; no device or transport ownership."""

from collections import defaultdict, deque
from fractions import Fraction
from pathlib import Path

import mido
from pydantic import BaseModel
from ufor.events import ControlChange, PerformanceEvent, Release, Trigger


class MidiPerformance(BaseModel, frozen=True):
    sample_rate: int
    frames: int
    parts: dict[int, list[PerformanceEvent]]


def read_midi(path: Path, sample_rate: int = 48000) -> MidiPerformance:
    """Read synchronous PPQN MIDI, rounding absolute rational times to frames.

    MIDI channels become parts, with FIFO matching for repeated same-key notes.
    CC1/CC11 map to modulation/expression, pitch wheel to bend (±2 semitones in
    the supplied patches). Velocity initializes a trigger-scoped control. Patch
    selection belongs to the caller, so program changes are ignored. Other
    controllers are rejected rather than silently changing the performance.
    """
    midi = mido.MidiFile(path)
    if sample_rate <= 0 or midi.type == 2 or midi.ticks_per_beat <= 0:
        raise ValueError("MIDI requires positive rate and synchronous PPQN timing")
    parts: dict[int, list[PerformanceEvent]] = defaultdict(list)
    held: dict[tuple[int, int], deque[str]] = defaultdict(deque)
    tempo = 500000
    elapsed = Fraction(0)
    frame = 0
    for ordinal, message in enumerate(mido.merge_tracks(midi.tracks)):
        elapsed += Fraction(message.time * tempo, midi.ticks_per_beat * 1000000)
        frame = round(elapsed * sample_rate)
        if message.type == "set_tempo":
            tempo = message.tempo
        elif message.is_meta or message.type == "program_change":
            continue
        elif message.type == "note_on" and message.velocity:
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
        elif message.type in ("note_on", "note_off"):
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
        elif message.type in ("control_change", "pitchwheel"):
            if message.type == "pitchwheel":
                control, value = "bend", message.pitch / 8192
            elif message.control in (1, 11):
                control = "modulation" if message.control == 1 else "expression"
                value = message.value / 127
            else:
                raise ValueError(f"Unsupported MIDI controller {message.control}")
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
            raise ValueError(f"Unsupported MIDI message: {message.type}")
    if any(held.values()):
        raise ValueError("MIDI ends with unreleased notes")
    return MidiPerformance(sample_rate=sample_rate, frames=frame + 1, parts=dict(parts))
