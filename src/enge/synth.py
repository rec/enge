"""Offline rendering for Ufor's canonical oscillator synth profile."""

from collections.abc import Mapping
from dataclasses import dataclass
from fractions import Fraction

import numpy as np
from ufor.envelope import Envelope, Segment
from ufor.instrument_trace import TraceAction, VoiceRetirement
from ufor.lfo import LFO
from ufor.oscillator import Oscillator, Waveform
from ufor.samples.processing import Processing
from ufor.streams import AudioType
from ufor.synth import SynthInstrumentScore
from ufor.synth_trace import VoiceStart
from ufor.time import Timebase


class EngineError(ValueError):
    """The requested Ufor definition is outside Enge's implemented profile."""


@dataclass(frozen=True)
class PreparedSynth:
    sample_rate: int
    channels: list[str]


@dataclass(frozen=True)
class VoiceSnapshot:
    voice_id: str
    oscillator: Oscillator
    frequency_hz: float
    phase: float
    routes: list[tuple[int, float]]
    gain: float


@dataclass(frozen=True)
class SynthSnapshot:
    frame: int
    voices: list[VoiceSnapshot]


@dataclass
class _Voice:
    oscillator: Oscillator
    frequency_hz: float
    phase: float
    routes: list[tuple[int, float]]
    gain: float


def prepare(score: SynthInstrumentScore) -> PreparedSynth:
    """Validate the first rendering profile and return immutable engine inputs."""
    output = score.outputs[0].stream
    if not isinstance(output, AudioType):
        raise EngineError("Synth output must be sampled audio")
    timebase = next(t for t in score.timebases if t.name == output.timebase)
    sample_rate = _sample_rate(timebase)
    for voice in score.body.voices:
        _validate_voice(voice.envelope, voice.processing, voice.envelopes, voice.lfos)
        if (
            voice.modulation.sources
            or voice.modulation.parameters
            or voice.modulation.routes
        ):
            raise EngineError("Synth modulation is not implemented")
        if voice.bindings:
            raise EngineError("Synth bindings are not implemented")
    return PreparedSynth(sample_rate=sample_rate, channels=output.channels)


class OfflineSynth:
    """Render one prepared synth instance with exact frame-boundary actions."""

    def __init__(self, definition: PreparedSynth) -> None:
        self.definition = definition
        self.frame = 0
        self.voices: dict[str, _Voice] = {}

    def advance(self, actions: list[TraceAction], start: int, end: int) -> np.ndarray:
        """Render [start, end), applying actions at their exact frame boundaries."""
        if start != self.frame or end <= start:
            raise EngineError(
                "advance must continue from the current nonempty interval"
            )
        ordered = sorted(actions, key=lambda a: (a.tick, a.ordinal))
        if any(a.tick < start or a.tick >= end for a in ordered):
            raise EngineError("actions must belong to the rendered interval")
        output = np.zeros((end - start, len(self.definition.channels)), dtype=float)
        cursor = start
        for action in ordered:
            output[cursor - start : action.tick - start] = self._render(
                action.tick - cursor
            )
            self._apply(action)
            cursor = action.tick
        output[cursor - start :] = self._render(end - cursor)
        self.frame = end
        return output

    def snapshot(self) -> SynthSnapshot:
        return SynthSnapshot(
            frame=self.frame,
            voices=[
                VoiceSnapshot(
                    voice_id=voice_id,
                    oscillator=voice.oscillator,
                    frequency_hz=voice.frequency_hz,
                    phase=voice.phase,
                    routes=voice.routes,
                    gain=voice.gain,
                )
                for voice_id, voice in self.voices.items()
            ],
        )

    def restore(self, snapshot: SynthSnapshot) -> None:
        self.frame = snapshot.frame
        self.voices = {
            voice.voice_id: _Voice(
                oscillator=voice.oscillator,
                frequency_hz=voice.frequency_hz,
                phase=voice.phase,
                routes=voice.routes,
                gain=voice.gain,
            )
            for voice in snapshot.voices
        }

    def _apply(self, action: TraceAction) -> None:
        if isinstance(action, VoiceStart):
            if action.pitch_hz is None:
                raise EngineError("Pitch-tracked synth voice requires Trigger.pitch_hz")
            if action.voice_id in self.voices:
                raise EngineError(f"Duplicate active voice: {action.voice_id}")
            self.voices[action.voice_id] = _Voice(
                oscillator=action.oscillator,
                frequency_hz=action.pitch_hz,
                phase=0,
                routes=[
                    (self.definition.channels.index(route.output), route.gain)
                    for route in action.channels
                ],
                gain=action.oscillator.gain(action.key),
            )
        elif isinstance(action, VoiceRetirement):
            self.voices.pop(action.voice_id, None)

    def _render(self, frames: int) -> np.ndarray:
        output = np.zeros((frames, len(self.definition.channels)), dtype=float)
        for voice in self.voices.values():
            phases = (
                voice.phase
                + np.arange(frames, dtype=float)
                * voice.frequency_hz
                / self.definition.sample_rate
            ) % 1
            wave = _waveform(voice.oscillator, phases) * voice.gain
            for channel, gain in voice.routes:
                output[:, channel] += wave * gain
            voice.phase = (
                voice.phase + frames * voice.frequency_hz / self.definition.sample_rate
            ) % 1
        return output


def _sample_rate(timebase: Timebase) -> int:
    if timebase.rate.denominator != 1:
        raise EngineError(
            "Synth output rate must be an integer number of frames per second"
        )
    return timebase.rate.numerator


def _validate_voice(
    envelope: Envelope,
    processing: Processing,
    envelopes: dict[str, Envelope],
    lfos: Mapping[str, LFO],
) -> None:
    gate = Envelope(
        segments=[Segment(duration=Fraction(0), target=1)],
        release=[Segment(duration=Fraction(0), target=0)],
    )
    if envelope != gate:
        raise EngineError(
            "Only the canonical instantaneous gate envelope is implemented"
        )
    if processing != Processing():
        raise EngineError("Synth processing is not implemented")
    if envelopes or lfos:
        raise EngineError("Named synth generators are not implemented")


def _waveform(oscillator: Oscillator, phase: np.ndarray) -> np.ndarray:
    if oscillator.waveform == Waveform.sine:
        return np.sin(2 * np.pi * phase)
    if oscillator.waveform == Waveform.square:
        return np.where(phase < float(oscillator.duty_cycle), 1.0, -1.0)
    duty = float(oscillator.duty_cycle)
    if duty == 0:
        return 1 - 2 * phase
    if duty == 1:
        return 2 * phase - 1
    return np.where(
        phase < duty,
        2 * phase / duty - 1,
        (1 + duty - 2 * phase) / (1 - duty),
    )
