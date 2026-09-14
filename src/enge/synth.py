"""Offline rendering for Ufor's canonical oscillator synth profile."""

from dataclasses import dataclass
from fractions import Fraction

import numpy as np
from ufor.envelope import Envelope, Segment
from ufor.instrument_trace import TraceAction, VoiceRetirement
from ufor.oscillator import Oscillator, Waveform
from ufor.samples.processing import Processing
from ufor.streams import AudioType
from ufor.synth import SynthInstrumentScore, SynthVoice
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
    started_frame: int
    envelope: Envelope
    minimum_hold_seconds: Fraction
    release_frame: float | None
    release_gain: float


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
    started_frame: int
    envelope: Envelope
    minimum_hold_seconds: Fraction
    release_frame: float | None = None
    release_gain: float = 1


def prepare(score: SynthInstrumentScore) -> PreparedSynth:
    """Validate the first rendering profile and return immutable engine inputs."""
    output = score.outputs[0].stream
    if not isinstance(output, AudioType):
        raise EngineError("Synth output must be sampled audio")
    timebase = next(t for t in score.timebases if t.name == output.timebase)
    sample_rate = _sample_rate(timebase)
    for voice in score.body.voices:
        _validate_voice(voice)
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
                cursor, action.tick - cursor
            )
            self._apply(action)
            cursor = action.tick
        output[cursor - start :] = self._render(cursor, end - cursor)
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
                    started_frame=voice.started_frame,
                    envelope=voice.envelope,
                    minimum_hold_seconds=voice.minimum_hold_seconds,
                    release_frame=voice.release_frame,
                    release_gain=voice.release_gain,
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
                started_frame=voice.started_frame,
                envelope=voice.envelope,
                minimum_hold_seconds=voice.minimum_hold_seconds,
                release_frame=voice.release_frame,
                release_gain=voice.release_gain,
            )
            for voice in snapshot.voices
        }

    def _apply(self, action: TraceAction) -> None:
        if isinstance(action, VoiceStart):
            if action.pitch_hz is None:
                raise EngineError("Pitch-tracked synth voice requires Trigger.pitch_hz")
            if action.voice_id in self.voices:
                raise EngineError(f"Duplicate active voice: {action.voice_id}")
            if not isinstance(action.settings, SynthVoice):
                raise EngineError("Synth voice start must carry a synth voice template")
            phase = (
                action.tick * action.pitch_hz / self.definition.sample_rate % 1
                if action.settings.synchronize_oscillator
                else 0
            )
            self.voices[action.voice_id] = _Voice(
                oscillator=action.oscillator,
                frequency_hz=action.pitch_hz,
                phase=phase,
                routes=[
                    (self.definition.channels.index(route.output), route.gain)
                    for route in action.channels
                ],
                gain=action.oscillator.gain(action.key),
                started_frame=action.tick,
                envelope=action.settings.envelope,
                minimum_hold_seconds=action.settings.minimum_hold_seconds,
            )
        elif isinstance(action, VoiceRetirement):
            if action.action == "stop":
                self.voices.pop(action.voice_id, None)
            elif voice := self.voices.get(action.voice_id):
                if voice.release_frame is None:
                    voice.release_frame = max(
                        float(action.tick),
                        voice.started_frame
                        + float(voice.minimum_hold_seconds)
                        * self.definition.sample_rate,
                    )
                    voice.release_gain = _envelope_value(
                        voice.envelope,
                        (voice.release_frame - voice.started_frame)
                        / self.definition.sample_rate,
                    )

    def _render(self, start: int, frames: int) -> np.ndarray:
        output = np.zeros((frames, len(self.definition.channels)), dtype=float)
        complete: list[str] = []
        for voice_id, voice in self.voices.items():
            phases = (
                voice.phase
                + np.arange(frames, dtype=float)
                * voice.frequency_hz
                / self.definition.sample_rate
            ) % 1
            wave = _waveform(voice.oscillator, phases)
            wave *= _voice_envelope(voice, start, frames, self.definition.sample_rate)
            wave *= voice.gain
            for channel, gain in voice.routes:
                output[:, channel] += wave * gain
            voice.phase = (
                voice.phase + frames * voice.frequency_hz / self.definition.sample_rate
            ) % 1
            if voice.release_frame is not None and start + frames >= (
                voice.release_frame
                + _duration(voice.envelope.release) * self.definition.sample_rate
            ):
                complete.append(voice_id)
        for voice_id in complete:
            self.voices.pop(voice_id)
        return output


def _sample_rate(timebase: Timebase) -> int:
    if timebase.rate.denominator != 1:
        raise EngineError(
            "Synth output rate must be an integer number of frames per second"
        )
    return timebase.rate.numerator


def _validate_voice(voice: SynthVoice) -> None:
    envelope = voice.envelope
    if envelope.clock != "seconds" or envelope.scope != "voice":
        raise EngineError("Synth envelopes must use the voice seconds clock")
    if not envelope.hold or any(
        s.curve != 0 for s in [*envelope.segments, *envelope.release]
    ):
        raise EngineError("Only held linear synth envelopes are implemented")
    if voice.processing != Processing():
        raise EngineError("Synth processing is not implemented")
    if voice.envelopes or voice.lfos:
        raise EngineError("Named synth generators are not implemented")


def _voice_envelope(
    voice: _Voice, start: int, frames: int, sample_rate: int
) -> np.ndarray:
    elapsed = (
        np.arange(start, start + frames, dtype=float) - voice.started_frame
    ) / sample_rate
    values = _envelope_values(voice.envelope.initial, voice.envelope.segments, elapsed)
    if voice.release_frame is None:
        return values
    released = (
        np.arange(start, start + frames, dtype=float) - voice.release_frame
    ) / sample_rate
    mask = released >= 0
    values[mask] = _envelope_values(
        voice.release_gain,
        voice.envelope.release,
        released[mask],
    )
    return values


def _envelope_value(envelope: Envelope, elapsed: float) -> float:
    return float(
        _envelope_values(
            envelope.initial,
            envelope.segments,
            np.array([elapsed], dtype=float),
        )[0]
    )


def _envelope_values(
    initial: float, segments: list[Segment], elapsed: np.ndarray
) -> np.ndarray:
    values = np.full(elapsed.shape, initial, dtype=float)
    remaining = elapsed.copy()
    active = np.ones(elapsed.shape, dtype=bool)
    for segment in segments:
        duration = float(segment.duration)
        current = active & (remaining < duration)
        if duration:
            values[current] += (segment.target - values[current]) * (
                remaining[current] / duration
            )
        completed = active & ~current
        values[completed] = segment.target
        remaining[completed] -= duration
        active = current
    return values


def _duration(segments: list[Segment]) -> float:
    return float(sum(segment.duration for segment in segments))


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
