"""Offline NumPy sample instruments driven by prepared uFor actions."""

from hashlib import sha256
from math import isfinite

import numpy as np
from ufor import instrument_trace
from ufor.base import Model
from ufor.samples import instrument, playback, processing, trace
from ufor.streams import AudioType

from . import sampler, synth


class PreparedSampler(Model, frozen=True):
    document: instrument.SampleInstrumentScore
    sample_rate: int
    channels: list[str]
    samples: dict[str, sampler.PreparedSample]
    settings: dict[str, processing.SoundSettings]
    audio_digests: dict[str, str]


class SampleVoiceSnapshot(Model, frozen=True):
    voice_id: str
    template: str
    renderer: sampler.SampleVoiceRenderer
    instrument_sources: dict[str, int]
    slot_sources: dict[str, int]


class SamplerSnapshot(Model, frozen=True):
    definition: instrument.SampleInstrumentScore
    audio_digests: dict[str, str]
    frame: int
    voices: list[SampleVoiceSnapshot]
    contexts: list[synth.ControlContext]


def prepare(
    score: instrument.SampleInstrumentScore, audio: dict[str, np.ndarray]
) -> PreparedSampler:
    """Prepare already-decoded assets; file discovery and decoding stay outside.

    Asset arrays must match the declared native frame/channel layout. Decoded
    content fingerprints bind snapshots to the actual audio, independently of
    the document's encoded-file hashes.
    """
    score = instrument.SampleInstrumentScore.model_validate(score.model_dump())
    output = score.outputs[0].stream
    assert isinstance(output, AudioType)
    rates = {t.name: t.rate for t in score.timebases}
    if rates[output.timebase].denominator != 1:
        raise synth.EngineError("Sampler output rate must be an integer")
    sample_rate = rates[output.timebase].numerator
    assets = {a.name: a for a in score.assets}
    slices = {s.name: s for s in score.body.slices}
    groups = {g.name: g for g in score.body.groups}
    required = {s.asset for s in score.body.slices}
    if audio.keys() != required:
        raise synth.EngineError("Decoded assets must match the referenced asset IDs")
    decoded: dict[str, np.ndarray] = {}
    digests: dict[str, str] = {}
    for name in required:
        metadata = assets[name].audio
        if audio[name].shape != (metadata.frames, len(metadata.channels)):
            raise synth.EngineError(
                f"Decoded asset {name} does not match its frame/channel layout"
            )
        if rates[metadata.timebase].denominator != 1:
            raise synth.EngineError("Sampler native rate must be an integer")
        decoded[name] = sampler.PreparedSample.immutable_audio(audio[name])
        digests[name] = sha256(decoded[name].tobytes()).hexdigest()

    _validate_settings(score.body.settings)
    samples: dict[str, sampler.PreparedSample] = {}
    settings: dict[str, processing.SoundSettings] = {}
    for slot in score.body.slots:
        settings[slot.name] = instrument.effective_settings(
            slot, groups.get(slot.group) if slot.group is not None else None
        )
        _validate_settings(settings[slot.name])
        if slot.crossfades:
            raise synth.EngineError("Sample layer crossfades are not implemented")
        if (
            slot.alignment_frames
            or slot.variation.delay_seconds
            or slot.variation.offset_frames
        ):
            raise synth.EngineError(
                "Delayed or offset sample starts are not implemented"
            )
        if any(c.mode == "fade" for c in slot.chokes):
            raise synth.EngineError("Fade retirement is not implemented")
        selection = slices[slot.slice]
        samples[slot.name] = sampler.PreparedSample(
            samples=decoded[selection.asset],
            native_rate=rates[assets[selection.asset].audio.timebase].numerator,
            sample_rate=sample_rate,
            slice=selection,
            playback=playback.Playback(
                direction=slot.playback.direction
                or score.body.settings.playback.direction,
                mode=slot.playback.mode or score.body.settings.playback.mode,
            ),
        )
    return PreparedSampler(
        document=score,
        sample_rate=sample_rate,
        channels=output.channels,
        samples=samples,
        settings=settings,
        audio_digests=digests,
    )


class OfflineSampler:
    """Render one sample instrument with the synth's scheduling and controls."""

    def __init__(self, definition: PreparedSampler) -> None:
        self.definition = definition
        self.frame = 0
        self.voices: dict[str, SampleVoiceSnapshot] = {}
        self.slots = {s.name: s for s in definition.document.body.slots}
        self.controls = synth.ControlRenderer(
            definition.sample_rate,
            definition.document.body.settings.controls,
            [definition.document.body.settings, *definition.settings.values()],
        )

    def advance(
        self, actions: list[instrument_trace.TraceAction], start: int, end: int
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

    def snapshot(self) -> SamplerSnapshot:
        return SamplerSnapshot(
            definition=self.definition.document,
            audio_digests=self.definition.audio_digests,
            frame=self.frame,
            voices=list(self.voices.values()),
            contexts=self.controls.contexts,
        ).model_copy(deep=True)

    def restore(self, snapshot: SamplerSnapshot) -> None:
        if (
            snapshot.definition != self.definition.document
            or snapshot.audio_digests != self.definition.audio_digests
        ):
            raise synth.EngineError(
                "Snapshot belongs to a different prepared sampler or decoded audio"
            )
        snapshot = snapshot.model_copy(deep=True)
        self.frame = snapshot.frame
        self.voices = {v.voice_id: v for v in snapshot.voices}
        self.controls.contexts = snapshot.contexts

    def _apply(self, action: instrument_trace.TraceAction) -> None:
        if self.controls.apply(action) or isinstance(
            action, instrument_trace.Diagnostic
        ):
            return
        if isinstance(action, trace.VoiceStart):
            self._start_voice(action)
        elif isinstance(action, instrument_trace.VoiceRetirement):
            if action.action == "fade":
                raise synth.EngineError("Fade retirement is not implemented")
            if action.action == "stop":
                self.voices.pop(action.voice_id, None)
            elif voice := self.voices.get(action.voice_id):
                voice.renderer.release()
        else:
            raise synth.EngineError(
                f"Unsupported sampler action at frame {action.tick}: "
                f"{type(action).__name__}"
            )

    def _start_voice(self, action: trace.VoiceStart) -> None:
        if action.voice_id in self.voices:
            raise synth.EngineError(f"Duplicate active voice: {action.voice_id}")
        slot = self.slots.get(action.template)
        if slot is None:
            raise synth.EngineError("Voice start has no prepared sample slot")
        sample = self.definition.samples[slot.name]
        settings = self.definition.settings[slot.name]
        if (
            action.slice != slot.slice
            or action.settings != settings
            or action.channels != slot.channels
            or action.start_frame != sample.slice.start_frame
        ):
            raise synth.EngineError("Voice start must match its prepared sample slot")
        if (
            action.alignment_frames
            or action.variation.delay_seconds
            or action.variation.offset_frames
        ):
            raise synth.EngineError(
                "Delayed or offset sample starts are not implemented"
            )
        if action.parameters:
            raise synth.EngineError(
                "Latched sample parameter overrides are not implemented"
            )
        if slot.mapping.pitch_tracking and (
            action.pitch_hz is None
            or not isfinite(action.pitch_hz)
            or action.pitch_hz <= 0
        ):
            raise synth.EngineError(
                "Pitch-tracked sample requires positive resolved pitch_hz"
            )
        common = self.definition.document.body.settings
        envelope = settings.envelope or common.envelope
        assert envelope is not None
        source_channels = next(
            a.audio.channels
            for a in self.definition.document.assets
            if a.name == sample.slice.asset
        )
        self.voices[action.voice_id] = SampleVoiceSnapshot(
            voice_id=action.voice_id,
            template=slot.name,
            renderer=sampler.SampleVoiceRenderer.start(
                sampler.PreparedSampleVoice(
                    sample_rate=self.definition.sample_rate,
                    slice=sample.slice,
                    envelope=envelope,
                    pitch_ratio=playback.pitch_ratio(
                        slot.mapping, action.pitch_hz, 0, action.variation.pitch_cents
                    ),
                    gain=10
                    ** (
                        (
                            common.processing.volume_db
                            + settings.processing.volume_db
                            + action.variation.gain_db
                        )
                        / 20
                    ),
                    routes=[
                        [
                            sum(
                                r.gain
                                for r in slot.channels
                                if r.input == s and r.output == c
                            )
                            for c in self.definition.channels
                        ]
                        for s in source_channels
                    ],
                ),
                sample,
            ),
            instrument_sources=self.controls.sources(common, action),
            slot_sources=self.controls.sources(settings, action),
        )

    def _render(self, start: int, frames: int) -> np.ndarray:
        output = np.zeros((frames, len(self.definition.channels)))
        for voice in list(self.voices.values()):
            count = voice.renderer.active_frames(frames)
            common = self.definition.document.body.settings
            settings = self.definition.settings[voice.template]
            # Pitch determines natural exhaustion. Resolve live values only
            # while this voice exists, so caller block size cannot cause a
            # domain error from a ramp after the source has already ended.
            span = (
                1
                if common.modulation.parameters or settings.modulation.parameters
                else max(1, count)
            )
            for offset in range(0, count, span):
                if voice.renderer.complete:
                    break
                size = min(span, count - offset)
                tuning, gains = self.controls.parameters(
                    common, voice.instrument_sources, start + offset, size
                )
                slot_tuning, slot_gains = self.controls.parameters(
                    settings, voice.slot_sources, start + offset, size
                )
                ratios = voice.renderer.definition.pitch_ratio * np.exp2(
                    (tuning + slot_tuning) / 1200
                )
                output[offset : offset + size] += voice.renderer.render(
                    self.definition.samples[voice.template],
                    size,
                    ratios,
                    gains * slot_gains,
                )
            if voice.renderer.complete:
                del self.voices[voice.voice_id]
        return output


def _validate_settings(settings: processing.SoundSettings) -> None:
    if settings.envelope is not None:
        synth.validate_envelope(settings.envelope)
    if settings.processing != processing.Processing(
        volume_db=settings.processing.volume_db,
        tuning_cents=settings.processing.tuning_cents,
    ):
        raise synth.EngineError(
            "Only sample volume and tuning processing are implemented"
        )
    if settings.envelopes or settings.lfos:
        raise synth.EngineError("Named sample generators are not implemented")
    if any(not isinstance(b, processing.ControlBinding) for b in settings.bindings):
        raise synth.EngineError("Only control bindings are implemented")
    if any(
        p.target.name != "processing"
        or p.target.parameter not in ("amplitude", "tuning_cents")
        for p in settings.modulation.parameters
    ):
        raise synth.EngineError("Only amplitude and tuning modulation are implemented")
