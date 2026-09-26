"""Deterministic white noise with shared envelope, filter, and control processing."""

from typing import Literal, Self, cast

import numpy as np
from pydantic import ConfigDict, Field, model_validator
from ufor import instrument_trace, modulation
from ufor.base import Model
from ufor.samples.processing import Processing, ResonantFilter
from ufor.streams import AudioType
from ufor.synth import NoiseVoice, SynthInstrumentScore
from ufor.synth_trace import VoiceStart

from . import _native, filters, native, synth


class PreparedVoice(synth.PreparedEnvelope, frozen=True):
    routes: list[list[float]]
    filters: list[ResonantFilter] = Field(default_factory=list)

    @model_validator(mode="after")
    def noise_profile(self) -> Self:
        if len(self.routes) != 1 or not self.routes[0]:
            raise synth.EngineError("Noise routes require one mono source")
        filters.parameters(self.filters, self.sample_rate, 1)
        return self


class VoiceRenderer(synth.EnvelopeRenderer):
    definition: PreparedVoice
    backend: Literal["numpy", "native"] = "numpy"
    stream_key: int = Field(strict=True, ge=0, lt=2**64)
    frame_count: int = Field(default=0, strict=True, ge=0, le=2**64)
    filter_states: list[filters.FilterState] = Field(default_factory=list)

    @classmethod
    def start(
        cls,
        definition: PreparedVoice,
        stream_key: int,
        backend: Literal["numpy", "native"] = "numpy",
    ) -> "VoiceRenderer":
        return cls(
            definition=definition.model_copy(deep=True),
            stream_key=stream_key,
            backend=backend,
            filter_states=filters.initial_states(definition.filters, 1),
        )

    def render(
        self, frames: int, gains: np.ndarray, filter_values: np.ndarray | None = None
    ) -> np.ndarray:
        if frames < 0:
            raise synth.EngineError("Noise frame count must be nonnegative")
        count = self.active_frames(frames)
        if (
            gains.shape != (count,)
            or not np.all(np.isfinite(gains))
            or np.any(gains < 0)
        ):
            raise synth.EngineError(
                "Noise gains must be finite, nonnegative, and match active frames"
            )
        if self.frame_count + count > 2**64:
            raise synth.EngineError("Noise counter exhausted")
        output = np.zeros((frames, len(self.definition.routes[0])))
        if count == 0:
            return output
        definition = self.definition
        values = filters.parameters(
            definition.filters,
            definition.sample_rate,
            count,
            filter_values,
            self.frame_count,
        )
        if self.backend == "native":
            from . import _native

            audio, memory = _native.render_noise(
                self.stream_key,
                self.frame_count,
                definition.sample_rate,
                gains,
                native.envelope_spans(
                    definition.envelope,
                    self.frame_count,
                    count,
                    definition.sample_rate,
                    self.release_frame,
                ),
                np.array(definition.routes[0]),
                filters.native_inputs(definition.filters, self.filter_states, values),
            )
            states = filters.restored_states(memory)
        else:
            audio = noise_samples(self.stream_key, self.frame_count, count)
            audio, states = filters.filter_samples(
                definition.filters,
                self.filter_states,
                audio,
                values,
                definition.sample_rate,
            )
            amplitudes = synth.envelope_samples(
                definition.envelope,
                self.frame_count,
                count,
                definition.sample_rate,
                self.release_frame,
            )
            audio = synth.route_samples(
                audio * (amplitudes * gains)[:, None], definition.routes
            )
        if not np.all(np.isfinite(audio)):
            raise synth.EngineError("Non-finite noise output")
        output[:count] = audio
        self.filter_states = states
        self.frame_count += count
        return output


class PreparedNoise(Model, frozen=True):
    sample_rate: int
    channels: list[str]
    score: SynthInstrumentScore


class VoiceSnapshot(Model, frozen=True):
    voice_id: str
    template: str
    renderer: VoiceRenderer
    sources: dict[str, int]


class NoiseSnapshot(Model, frozen=True):
    control_interval: int = Field(strict=True, gt=0)
    backend: Literal["numpy", "native"]
    definition: PreparedNoise
    frame: int
    voices: list[VoiceSnapshot]
    contexts: list[synth.ControlContext]
    lfos: list[synth.LFOSource]


class PersistentNoiseSnapshot(Model, frozen=True):
    definition: PreparedNoise
    action_capacity: int
    context_capacity: int
    frame: int
    voices: dict[str, int]
    part_contexts: dict[str, int]
    trigger_contexts: dict[tuple[str, str], int]
    state: _native.SynthRuntimeSnapshot

    model_config = ConfigDict(arbitrary_types_allowed=True)


class OfflineNoise:
    """Consume uFor prepared synth actions for a noise-only instrument instance."""

    def __init__(
        self,
        definition: PreparedNoise,
        backend: Literal["numpy", "native"] = "numpy",
        control_interval: int = 1,
    ) -> None:
        if backend not in ("numpy", "native"):
            raise synth.EngineError(f"Unknown noise backend: {backend}")
        self.backend: Literal["numpy", "native"] = backend
        self.definition = definition.model_copy(deep=True)
        self.frame = 0
        self.voices: dict[str, VoiceSnapshot] = {}
        self.templates = {
            v.name: v
            for v in self.definition.score.body.voices
            if isinstance(v, NoiseVoice)
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

    def snapshot(self) -> NoiseSnapshot:
        return NoiseSnapshot(
            control_interval=self.controls.control_interval,
            backend=self.backend,
            definition=self.definition,
            frame=self.frame,
            voices=list(self.voices.values()),
            contexts=self.controls.contexts,
            lfos=self.controls.lfos,
        ).model_copy(deep=True)

    def restore(self, snapshot: NoiseSnapshot) -> None:
        if snapshot.control_interval != self.controls.control_interval:
            raise synth.EngineError("Snapshot belongs to a different control interval")
        if snapshot.definition != self.definition:
            raise synth.EngineError(
                "Snapshot belongs to a different prepared noise synth"
            )
        if snapshot.backend != self.backend or any(
            v.renderer.backend != self.backend for v in snapshot.voices
        ):
            raise synth.EngineError("Snapshot belongs to a different noise backend")
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
            raise synth.EngineError(f"Unsupported noise action at frame {action.tick}")
        template = self.templates.get(action.template)
        if (
            template is None
            or action.settings != template
            or action.oscillator is not None
            or action.channels != template.channels
        ):
            raise synth.EngineError(
                "Voice start must match its prepared noise template"
            )
        if action.noise_key is None:
            raise synth.EngineError("Noise voice requires a resolved stream key")
        if action.voice_id in self.voices:
            raise synth.EngineError(f"Duplicate active voice: {action.voice_id}")
        self.voices[action.voice_id] = VoiceSnapshot(
            voice_id=action.voice_id,
            template=template.name,
            sources=self.controls.sources(template, action),
            renderer=VoiceRenderer.start(
                PreparedVoice(
                    sample_rate=self.definition.sample_rate,
                    envelope=template.envelope,
                    minimum_hold_seconds=template.minimum_hold_seconds,
                    filters=template.processing.filters,
                    routes=[
                        [
                            sum(r.gain for r in template.channels if r.output == c)
                            for c in self.definition.channels
                        ]
                    ],
                ),
                stream_key=action.noise_key,
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
            _, gains, filter_values = synth.processing_parameters(
                template,
                values,
                self.definition.sample_rate,
                start,
                count,
            )
            output += renderer.render(
                frames,
                gains * 10 ** (template.processing.volume_db / 20),
                filter_values,
            )
            if renderer.complete:
                del self.voices[voice.voice_id]
        return output


class PersistentNoise(synth.PersistentSynth):
    """Bounded native noise-v1 runtime for one static voice template."""

    def __init__(
        self,
        definition: PreparedNoise,
        voices: int = 16,
        action_capacity: int = 64,
        context_capacity: int = 64,
    ) -> None:
        if type(voices) is not int or voices <= 0:
            raise synth.EngineError("Persistent noise voice capacity must be positive")
        if type(action_capacity) is not int or action_capacity <= 0:
            raise synth.EngineError("Persistent noise action capacity must be positive")
        if type(context_capacity) is not int or context_capacity <= 0:
            raise synth.EngineError(
                "Persistent noise context capacity must be positive"
            )
        templates = definition.score.body.voices
        if len(templates) != 1 or not isinstance(templates[0], NoiseVoice):
            raise synth.EngineError(
                "Persistent noise requires one noise voice template"
            )
        template = templates[0]
        if template.envelopes:
            raise synth.EngineError(
                "Persistent noise named envelopes are not implemented"
            )
        if template.processing != Processing(
            volume_db=template.processing.volume_db,
            filters=template.processing.filters,
        ):
            raise synth.EngineError(
                "Persistent noise spatial processing is not implemented"
            )
        route = [
            sum(r.gain for r in template.channels if r.output == c)
            for c in definition.channels
        ]
        rate = definition.sample_rate
        shared = synth.PreparedSynth(
            sample_rate=rate,
            channels=definition.channels,
            instrument=definition.score.body,
        )
        source_parameters = [
            (
                ("processing", "amplitude"),
                0,
                modulation.Operation.multiply,
                1.0,
                0.0,
                1e300,
            )
        ]
        self.definition = definition.model_copy(deep=True)
        self.instrument = self.definition.score.body
        self.template = template
        self.frame = 0
        self.voices: dict[str, int] = {}
        self.action_capacity = action_capacity
        self.context_capacity = context_capacity
        self._action_buffer = np.empty((action_capacity, 6), dtype=np.float64)
        (
            controls,
            smoothing,
            parameters,
            self.control_sources,
            self.control_scopes,
            self.source_controls,
            lfos,
            lfo_rationals,
            self.source_scopes,
            runtime_filters,
        ) = synth._persistent_modulation(shared, template, source_parameters)
        self.part_contexts: dict[str, int] = {}
        self.trigger_contexts: dict[tuple[str, str], int] = {}
        self.runtime = _native.SynthRuntime.noise(
            rate,
            np.tile(np.asarray(route, dtype=np.float64), (voices, 1)),
            template.envelope.initial,
            synth._runtime_segments(template.envelope.segments, rate),
            synth._runtime_segments(template.envelope.release, rate),
            float(template.minimum_hold_seconds * rate),
            controls,
            smoothing,
            lfos,
            lfo_rationals,
            runtime_filters,
            parameters,
            context_capacity,
        )

    def snapshot(self) -> PersistentNoiseSnapshot:  # ty: ignore[invalid-method-override]
        return PersistentNoiseSnapshot(
            definition=cast(PreparedNoise, self.definition).model_copy(deep=True),
            action_capacity=self.action_capacity,
            context_capacity=self.context_capacity,
            frame=self.frame,
            voices=self.voices.copy(),
            part_contexts=self.part_contexts.copy(),
            trigger_contexts=self.trigger_contexts.copy(),
            state=self.runtime.snapshot(),
        )

    def restore(  # ty: ignore[invalid-method-override]
        self, snapshot: PersistentNoiseSnapshot
    ) -> None:
        if snapshot.definition != self.definition:
            raise synth.EngineError(
                "Snapshot belongs to a different prepared noise synth"
            )
        if snapshot.action_capacity != self.action_capacity:
            raise synth.EngineError("Snapshot belongs to a different action capacity")
        if snapshot.context_capacity != self.context_capacity:
            raise synth.EngineError("Snapshot belongs to a different context capacity")
        self.runtime.restore(snapshot.state)
        self.frame = snapshot.frame
        self.voices = snapshot.voices.copy()
        self.part_contexts = snapshot.part_contexts.copy()
        self.trigger_contexts = snapshot.trigger_contexts.copy()

    def _start(
        self, action: VoiceStart, active: list[bool]
    ) -> tuple[int, float, float, float]:
        if action.voice_id in self.voices:
            raise synth.EngineError(f"Duplicate active voice: {action.voice_id}")
        if (
            action.template != self.template.name
            or action.settings != self.template
            or action.oscillator is not None
            or action.channels != self.template.channels
        ):
            raise synth.EngineError(
                "Voice start must match its prepared noise template"
            )
        if action.noise_key is None:
            raise synth.EngineError("Noise voice requires a resolved stream key")
        slot = next((i for i, value in enumerate(active) if not value), None)
        if slot is None:
            raise synth.EngineError("Persistent noise voice capacity exceeded")
        active[slot] = True
        self.voices[action.voice_id] = slot
        return (
            slot,
            action.noise_key & 0xFFFF_FFFF,
            action.noise_key >> 32,
            10 ** (self.template.processing.volume_db / 20),
        )


def prepare(score: SynthInstrumentScore) -> PreparedNoise:
    """Validate the supported source profile without silently dropping settings."""
    score = SynthInstrumentScore.model_validate(score.model_dump())
    output = score.outputs[0].stream
    if not isinstance(output, AudioType):
        raise synth.EngineError("Noise output must be sampled audio")
    timebase = next(t for t in score.timebases if t.name == output.timebase)
    if timebase.rate.denominator != 1:
        raise synth.EngineError("Noise output rate must be an integer")
    rate = timebase.rate.numerator
    for voice in score.body.voices:
        if not isinstance(voice, NoiseVoice):
            raise synth.EngineError("Noise engine requires noise voice templates")
        synth.validate_generators(voice)
        if voice.processing != Processing(
            volume_db=voice.processing.volume_db,
            filters=voice.processing.filters,
        ):
            raise synth.EngineError(
                "Only noise volume and filter processing are implemented"
            )
        if any(c.mode == "fade" for c in voice.chokes):
            raise synth.EngineError("Fade retirement is not implemented")
        synth.validate_envelope(voice.envelope)
        synth.validate_modulation(voice)
        filters.parameters(voice.processing.filters, rate, 1)
    return PreparedNoise(sample_rate=rate, channels=output.channels, score=score)


def noise_samples(stream_key: int, start: int, frames: int) -> np.ndarray:
    """Vectorized noise-v1 samples; integer arithmetic wraps modulo 2**64."""
    if not 0 <= stream_key < 2**64 or start < 0 or frames < 0 or start + frames > 2**64:
        raise synth.EngineError("Invalid noise key or exhausted counter")
    if frames == 0:
        return np.empty((0, 1))
    indices = np.arange(frames, dtype=np.uint64) + np.uint64(start)
    words = np.uint64(stream_key) + (indices + np.uint64(1)) * np.uint64(
        0x9E3779B97F4A7C15
    )
    words = (words ^ (words >> 30)) * np.uint64(0xBF58476D1CE4E5B9)
    words = (words ^ (words >> 27)) * np.uint64(0x94D049BB133111EB)
    words ^= words >> 31
    return (2 * ((words >> 11).astype(np.float64) / 2**53) - 1)[:, None]
