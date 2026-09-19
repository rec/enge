"""Two-operator phase modulation with independent NumPy and Rust numerical kernels."""

from math import ceil, isfinite
from typing import Literal, Self

import numpy as np
from pydantic import Field, model_validator
from ufor import instrument_trace
from ufor.base import Model
from ufor.fm import FM
from ufor.samples.processing import Processing, ResonantFilter
from ufor.streams import AudioType
from ufor.synth import FMVoice, SynthInstrumentScore
from ufor.synth_trace import VoiceStart

from . import filters, native, synth


class PreparedVoice(synth.PreparedEnvelope, frozen=True):
    fm: FM
    pitch_hz: float = Field(gt=0)
    routes: list[list[float]]
    filters: list[ResonantFilter] = Field(default_factory=list)

    @model_validator(mode="after")
    def voice_profile(self) -> Self:
        carrier = next(
            o for o in self.fm.operators if o.name == self.fm.connection.destination
        )
        if self.envelope != carrier.envelope:
            raise synth.EngineError("FM lifetime envelope must match the carrier")
        if len(self.routes) != 1 or not self.routes[0]:
            raise synth.EngineError(
                "FM routes must map one mono source to output channels"
            )
        filters.parameters(self.filters, self.sample_rate, 1)
        return self


class VoiceRenderer(synth.EnvelopeRenderer):
    definition: PreparedVoice
    backend: Literal["numpy", "native"] = "numpy"
    phases: list[list[float]]
    previous_modulator: float = 0
    filter_states: list[filters.FilterState] = Field(default_factory=list)

    @classmethod
    def start(
        cls, definition: PreparedVoice, backend: Literal["numpy", "native"] = "numpy"
    ) -> "VoiceRenderer":
        operators = {o.name: o for o in definition.fm.operators}
        connection = definition.fm.connection
        return cls(
            definition=definition.model_copy(deep=True),
            backend=backend,
            phases=[
                [operators[n].phase_cycles * definition.sample_rate, 0]
                for n in (connection.source, connection.destination)
            ],
            filter_states=filters.initial_states(definition.filters, 1),
        )

    def render(
        self,
        frames: int,
        parameters: np.ndarray,
        gains: np.ndarray,
        filter_values: np.ndarray,
    ) -> np.ndarray:
        """Render active-frame arrays, returning silence after carrier completion.

        Parameter columns are modulator Hz, carrier Hz, index radians, feedback
        radians, and carrier level. Python resolves validation and envelope boundaries.
        """
        if frames < 0:
            raise synth.EngineError("FM frame count must be nonnegative")
        count = self.active_frames(frames)
        if parameters.shape != (count, 5) or gains.shape != (count,):
            raise synth.EngineError("FM parameters must match the active frame count")
        if (
            not np.all(np.isfinite(parameters))
            or np.any(parameters[:, :2] <= 0)
            or np.any(parameters[:, 2:] < 0)
            or not np.all(np.isfinite(gains))
            or np.any(gains < 0)
        ):
            raise synth.EngineError(
                "FM requires positive frequencies and nonnegative levels/indices"
            )
        output = np.zeros((frames, len(self.definition.routes[0])))
        if count == 0:
            return output
        rate = self.definition.sample_rate
        operators = {o.name: o for o in self.definition.fm.operators}
        connection = self.definition.fm.connection
        values = filters.parameters(
            self.definition.filters, rate, count, filter_values, self.frame_count
        )
        if self.backend == "native":
            from . import _native

            spans = [
                native.envelope_spans(
                    operators[n].envelope,
                    self.frame_count,
                    count,
                    rate,
                    self.release_frame,
                )
                for n in (connection.source, connection.destination)
            ]
            modulator_frames = count
            if self.release_frame is not None:
                end = (
                    self.release_frame
                    + sum(
                        s.duration
                        for s in operators[connection.source].envelope.release
                    )
                    * rate
                )
                modulator_frames = max(0, min(count, ceil(end) - self.frame_count))
            audio, phases, history, memory = _native.render_fm(
                rate,
                parameters,
                np.array(self.phases),
                self.previous_modulator,
                gains,
                spans[0],
                spans[1],
                modulator_frames,
                np.array(self.definition.routes[0]),
                filters.native_inputs(
                    self.definition.filters, self.filter_states, values
                ),
            )
            output[:count] = audio
            self.phases = phases.tolist()
            self.previous_modulator = history
            self.filter_states = filters.restored_states(memory)
            self.frame_count += count
            return output
        envelopes = np.column_stack(
            [
                synth.envelope_samples(
                    operators[n].envelope,
                    self.frame_count,
                    count,
                    rate,
                    self.release_frame,
                )
                for n in (connection.source, connection.destination)
            ]
        )
        modulator = operators[connection.source]
        if self.release_frame is not None:
            end = (
                self.release_frame
                + sum(s.duration for s in modulator.envelope.release) * rate
            )
            envelopes[max(0, ceil(end) - self.frame_count) :, 0] = 0
        audio, phases, history = fm_samples(
            parameters[:, :2],
            envelopes,
            parameters[:, 2],
            parameters[:, 3],
            parameters[:, 4],
            np.array(self.phases),
            np.array([self.previous_modulator]),
            rate,
        )
        if not np.all(np.isfinite(audio)):
            raise synth.EngineError("FM produced non-finite output")
        audio, filter_states = filters.filter_samples(
            self.definition.filters,
            self.filter_states,
            audio,
            values,
            rate,
        )
        output[:count] = synth.route_samples(
            audio * gains[:, None], self.definition.routes
        )
        self.phases = phases.tolist()
        self.previous_modulator = float(history[0])
        self.filter_states = filter_states
        self.frame_count += count
        return output


class PreparedFM(Model, frozen=True):
    sample_rate: int
    channels: list[str]
    score: SynthInstrumentScore


class VoiceSnapshot(Model, frozen=True):
    voice_id: str
    template: str
    renderer: VoiceRenderer
    sources: dict[str, int]


class FMSnapshot(Model, frozen=True):
    control_interval: int = Field(strict=True, gt=0)
    backend: Literal["numpy", "native"]
    definition: PreparedFM
    frame: int
    voices: list[VoiceSnapshot]
    contexts: list[synth.ControlContext]
    lfos: list[synth.LFOSource]


class OfflineFM:
    """Consume uFor prepared synth actions for an FM-only instrument instance."""

    def __init__(
        self,
        definition: PreparedFM,
        backend: Literal["numpy", "native"] = "numpy",
        control_interval: int = 1,
    ) -> None:
        if backend not in ("numpy", "native"):
            raise synth.EngineError(f"Unknown FM backend: {backend}")
        self.backend: Literal["numpy", "native"] = backend
        self.definition = definition.model_copy(deep=True)
        self.frame = 0
        self.voices: dict[str, VoiceSnapshot] = {}
        self.templates = {
            v.name: v
            for v in self.definition.score.body.voices
            if isinstance(v, FMVoice)
        }
        self.controls = synth.ControlRenderer(
            definition.sample_rate,
            definition.score.body.controls,
            list(self.templates.values()),
            backend,
            control_interval,
        )

    def advance(
        self,
        actions: list[instrument_trace.TraceAction],
        start: int,
        end: int,
    ) -> np.ndarray:
        output = synth.render_actions(
            actions,
            start,
            end,
            self.frame,
            len(self.definition.channels),
            self._render,
            self._apply,
        )
        self.frame = end
        return output

    def snapshot(self) -> FMSnapshot:
        return FMSnapshot(
            control_interval=self.controls.control_interval,
            backend=self.backend,
            definition=self.definition,
            frame=self.frame,
            voices=list(self.voices.values()),
            contexts=self.controls.contexts,
            lfos=self.controls.lfos,
        ).model_copy(deep=True)

    def restore(self, snapshot: FMSnapshot) -> None:
        if snapshot.control_interval != self.controls.control_interval:
            raise synth.EngineError("Snapshot belongs to a different control interval")
        if snapshot.definition != self.definition:
            raise synth.EngineError("Snapshot belongs to a different prepared FM synth")
        if snapshot.backend != self.backend or any(
            v.renderer.backend != self.backend for v in snapshot.voices
        ):
            raise synth.EngineError("Snapshot belongs to a different FM backend")
        snapshot = snapshot.model_copy(deep=True)
        self.frame = snapshot.frame
        self.voices = {v.voice_id: v for v in snapshot.voices}
        self.controls.clear_cache()
        self.controls.contexts = snapshot.contexts
        self.controls.lfos = snapshot.lfos

    def _apply(self, action: instrument_trace.TraceAction) -> None:
        if self.controls.apply(action):
            return
        if isinstance(action, instrument_trace.VoiceRetirement):
            if action.action == "fade":
                raise synth.EngineError("Fade retirement is not implemented")
            if action.action == "stop":
                self.voices.pop(action.voice_id, None)
            elif voice := self.voices.get(action.voice_id):
                voice.renderer.release()
            return
        if not isinstance(action, VoiceStart):
            raise synth.EngineError(f"Unsupported FM action at frame {action.tick}")
        template = self.templates.get(action.template)
        if (
            template is None
            or action.settings != template
            or action.oscillator is not None
            or action.channels != template.channels
        ):
            raise synth.EngineError("Voice start must match its prepared FM template")
        if (
            action.pitch_hz is None
            or not isfinite(action.pitch_hz)
            or action.pitch_hz <= 0
        ):
            raise synth.EngineError("FM voice requires positive resolved pitch_hz")
        if action.voice_id in self.voices:
            raise synth.EngineError(f"Duplicate active voice: {action.voice_id}")
        carrier = next(
            o
            for o in template.fm.operators
            if o.name == template.fm.connection.destination
        )
        self.voices[action.voice_id] = VoiceSnapshot(
            voice_id=action.voice_id,
            template=template.name,
            sources=self.controls.sources(template, action),
            renderer=VoiceRenderer.start(
                PreparedVoice(
                    sample_rate=self.definition.sample_rate,
                    fm=template.fm,
                    pitch_hz=action.pitch_hz,
                    envelope=carrier.envelope,
                    minimum_hold_seconds=template.minimum_hold_seconds,
                    filters=template.processing.filters,
                    routes=[
                        [
                            sum(r.gain for r in template.channels if r.output == c)
                            for c in self.definition.channels
                        ]
                    ],
                ),
                backend=self.backend,
            ),
        )

    def _render(self, start: int, frames: int) -> np.ndarray:
        output = np.zeros((frames, len(self.definition.channels)))
        for voice in list(self.voices.values()):
            renderer = voice.renderer
            count = renderer.active_frames(frames)
            template = self.templates[voice.template]
            values = self.controls.values(template, voice.sources, start, count)
            tuning, gains, filter_values = synth.processing_parameters(
                template,
                values,
                self.definition.sample_rate,
                start,
                count,
            )
            operators = {o.name: o for o in template.fm.operators}
            connection = template.fm.connection
            parameters = np.empty((count, 5))
            for i, name in enumerate((connection.source, connection.destination)):
                operator = operators[name]
                ratio = values.get((f"operator-{name}", "ratio"), operator.ratio)
                cents = values.get(
                    (f"operator-{name}", "tuning_cents"), operator.tuning_cents
                )
                parameters[:, i] = (
                    renderer.definition.pitch_hz
                    * ratio
                    * np.exp2((tuning + cents) / 1200)
                )
            for i, (name, default) in enumerate(
                (
                    ("index", connection.index),
                    ("feedback", template.fm.feedback),
                    ("carrier_level", template.fm.carrier_level),
                ),
                start=2,
            ):
                parameters[:, i] = values.get(("fm", name), default)
            output += renderer.render(
                frames,
                parameters,
                gains * 10 ** (template.processing.volume_db / 20),
                filter_values,
            )
            if renderer.complete:
                del self.voices[voice.voice_id]
        return output


def prepare(score: SynthInstrumentScore) -> PreparedFM:
    """Validate the supported source profile without silently dropping settings."""
    score = SynthInstrumentScore.model_validate(score.model_dump())
    output = score.outputs[0].stream
    if not isinstance(output, AudioType):
        raise synth.EngineError("FM output must be sampled audio")
    timebase = next(t for t in score.timebases if t.name == output.timebase)
    if timebase.rate.denominator != 1:
        raise synth.EngineError("FM output rate must be an integer")
    rate = timebase.rate.numerator
    for voice in score.body.voices:
        if not isinstance(voice, FMVoice):
            raise synth.EngineError("FM engine requires FM voice templates")
        synth.validate_generators(voice)
        if voice.processing != Processing(
            tuning_cents=voice.processing.tuning_cents,
            volume_db=voice.processing.volume_db,
            filters=voice.processing.filters,
        ):
            raise synth.EngineError(
                "Only FM tuning, volume, and filter processing are implemented"
            )
        if any(c.mode == "fade" for c in voice.chokes):
            raise synth.EngineError("Fade retirement is not implemented")
        if any(
            p.target.name == "processing"
            and p.target.parameter not in ("amplitude", "tuning_cents")
            or p.target.name.startswith(("eq-", "env-"))
            for p in voice.modulation.parameters
        ):
            raise synth.EngineError("Unsupported FM modulation target")
        filters.parameters(voice.processing.filters, rate, 1)
    return PreparedFM(sample_rate=rate, channels=output.channels, score=score)


def fm_samples(
    frequencies: np.ndarray,
    envelopes: np.ndarray,
    indices: np.ndarray,
    feedback: np.ndarray,
    levels: np.ndarray,
    phases: np.ndarray,
    history: np.ndarray,
    sample_rate: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Pure array kernel: audio plus independent next phase/feedback state.

    Inputs are validated by the caller. No models, scalar extraction, event
    dispatch, or data-dependent Python branches belong here. The scalar time
    recurrence is the readable reference; a future tensor port needs a supported
    scan/loop rather than compiling a large unrolled Python loop.
    """
    position = phases[:, 0].copy()
    error = phases[:, 1].copy()
    previous = history.copy()
    audio = np.empty((len(frequencies), 1))
    for i in range(len(frequencies)):
        angle = 2 * np.pi * position / sample_rate
        modulator = envelopes[i, 0] * np.sin(angle[0] + feedback[i] * previous)
        audio[i] = (
            levels[i] * envelopes[i, 1] * np.sin(angle[1] + indices[i] * modulator)
        )
        increment = frequencies[i] - error
        total = position + increment
        error = (total - position) - increment
        position = np.remainder(total, sample_rate)
        previous = modulator
    return audio, np.column_stack((position, error)), previous.copy()
