"""Offline sample instruments driven by prepared uFor actions."""

from fractions import Fraction
from hashlib import sha256
from math import ceil, isfinite, pi
from typing import Literal, cast

import numpy as np
from envelfo import patch
from pydantic import Field
from ufor import instrument_trace
from ufor.base import Model
from ufor.samples import crossfade, enums, instrument, playback, processing, trace
from ufor.samples.enums import Direction, LoopMode
from ufor.streams import AudioType

from . import control, filters, sampler, synth


class PreparedSampler(Model, frozen=True):
    document: instrument.SampleInstrumentScore
    sample_rate: int
    channels: list[str]
    samples: dict[str, sampler.PreparedSample]
    settings: dict[str, processing.SoundSettings]
    routes: dict[str, list[list[float]]]
    audio_digests: dict[str, str]


class SampleVoiceSnapshot(Model, frozen=True):
    voice_id: str
    part: str
    trigger_id: str | None
    template: str
    renderer: sampler.SampleVoiceRenderer
    instrument_sources: dict[str, int]
    slot_sources: dict[str, int]
    crossfade_sources: list[int]


class SamplerSnapshot(Model, frozen=True):
    control_interval: int = Field(strict=True, gt=0)
    definition: instrument.SampleInstrumentScore
    audio_digests: dict[str, str]
    frame: int
    voices: list[SampleVoiceSnapshot]
    pending: list[trace.VoiceStart]
    contexts: list[synth.ControlContext]
    lfos: list[synth.LFOSource]
    envelopes: list[synth.EnvelopeSource]
    patches: list[synth.PatchState]
    graphs: dict[str, patch.PatchState]
    backend: Literal["numpy", "native"]


class PersistentSamplerSnapshot(Model, frozen=True):
    state: SamplerSnapshot
    voice_capacity: int
    action_capacity: int
    native_voices: dict[str, "NativeSampleVoice"]


class NativeSampleVoice(Model, frozen=True):
    part: str
    trigger_id: str | None
    source: int
    slot: int
    started_frame: int
    trigger: enums.TriggerKind


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

    score = score.model_copy(
        update={
            "body": score.body.model_copy(
                update={
                    "settings": synth.expand_patch_outputs(score.body.settings),
                }
            )
        }
    )
    _validate_settings(score.body.settings)
    filters.parameters(score.body.settings.processing.filters, sample_rate, 1)
    samples: dict[str, sampler.PreparedSample] = {}
    settings: dict[str, processing.SoundSettings] = {}
    routes: dict[str, list[list[float]]] = {}
    for slot in score.body.slots:
        settings[slot.name] = synth.expand_patch_outputs(
            instrument.effective_settings(
                slot, groups.get(slot.group) if slot.group is not None else None
            )
        )
        _validate_settings(settings[slot.name])
        filters.parameters(settings[slot.name].processing.filters, sample_rate, 1)
        selection = slices[slot.slice]
        source_channels = assets[selection.asset].audio.channels
        gains = [1.0] * len(output.channels)
        if output.channels == ["left", "right"] and source_channels in (
            ["mono"],
            ["left", "right"],
        ):
            stereo = source_channels == ["left", "right"]
            parameter = "stereo_balance" if stereo else "pan"
            position = getattr(score.body.settings.processing, parameter) + getattr(
                settings[slot.name].processing, parameter
            )
            gains = processing.pan_gains(position, score.body.pan_law, stereo=stereo)
            if not stereo:
                gains = [g / 2**-0.5 for g in gains]
        routes[slot.name] = [
            [
                sum(r.gain for r in slot.channels if r.input == s and r.output == c)
                * gains[i]
                for i, c in enumerate(output.channels)
            ]
            for s in source_channels
        ]
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
        routes=routes,
        audio_digests=digests,
    )


class OfflineSampler:
    """Render one sample instrument with the synth's scheduling and controls."""

    def __init__(
        self,
        definition: PreparedSampler,
        backend: Literal["numpy", "native"] = "numpy",
        control_interval: int = 1,
    ) -> None:
        if backend not in ("numpy", "native"):
            raise synth.EngineError(f"Unknown sampler backend: {backend}")
        self.definition = definition
        self.backend: Literal["numpy", "native"] = backend
        self.frame = 0
        self.voices: dict[str, SampleVoiceSnapshot] = {}
        self.pending: dict[str, trace.VoiceStart] = {}
        self.slots = {s.name: s for s in definition.document.body.slots}
        self.controls = synth.ControlRenderer(
            definition.sample_rate,
            definition.document.body.settings.controls,
            [definition.document.body.settings, *definition.settings.values()],
            backend,
            control_interval,
            control_sources=[
                (f.scope, f.control, Fraction(str(f.smoothing_seconds)))
                for s in self.slots.values()
                for f in s.crossfades
                if isinstance(f, crossfade.ControlCrossfade)
            ],
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
            control_interval=self.controls.control_interval,
            definition=self.definition.document,
            audio_digests=self.definition.audio_digests,
            frame=self.frame,
            voices=list(self.voices.values()),
            pending=list(self.pending.values()),
            contexts=self.controls.contexts,
            lfos=self.controls.lfos,
            envelopes=self.controls.envelopes,
            patches=self.controls.patches,
            graphs=self.controls.graphs,
            backend=self.backend,
        ).model_copy(deep=True)

    def restore(self, snapshot: SamplerSnapshot) -> None:
        if snapshot.control_interval != self.controls.control_interval:
            raise synth.EngineError("Snapshot belongs to a different control interval")
        if snapshot.backend != self.backend or any(
            v.renderer.backend != self.backend for v in snapshot.voices
        ):
            raise synth.EngineError("Snapshot belongs to a different sampler backend")
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
        self.pending = {a.voice_id: a for a in snapshot.pending}
        self.controls.clear_cache()
        self.controls.contexts = snapshot.contexts
        self.controls.lfos = snapshot.lfos
        self.controls.envelopes = snapshot.envelopes
        self.controls.patches = snapshot.patches
        self.controls.graphs = snapshot.graphs

    def _apply(self, action: instrument_trace.TraceAction) -> None:
        self._activate_pending(action.tick)
        if isinstance(action, instrument_trace.TriggerRelease):
            self.pending = {
                k: v
                for k, v in self.pending.items()
                if (v.part, v.trigger_id) != (action.part, action.trigger_id)
                or self.slots[v.template].trigger != enums.TriggerKind.start
            }
            return
        if synth.apply_amplitude_event(
            action, ((v.renderer, v.part, v.trigger_id) for v in self.voices.values())
        ):
            return
        if self.controls.apply(action, self.voices) or isinstance(
            action, instrument_trace.Diagnostic
        ):
            return
        if isinstance(action, trace.VoiceStart):
            slot, _, _ = self._voice_definition(action)
            delay = _delay_frames(slot, action, self.definition.sample_rate)
            if delay:
                self.pending[action.voice_id] = action.model_copy(
                    update={"tick": action.tick + delay}
                )
            else:
                self._start_voice(action)
        elif isinstance(action, instrument_trace.VoiceRetirement):
            if action.voice_id in self.pending:
                del self.pending[action.voice_id]
                return
            if action.action == "fade":
                if voice := self.voices.get(action.voice_id):
                    assert action.fade_seconds is not None
                    voice.renderer.fade(action.fade_seconds)
            elif action.action == "stop":
                self.voices.pop(action.voice_id, None)
                self.controls.stop(action.voice_id)
            elif voice := self.voices.get(action.voice_id):
                if voice.renderer.release():
                    self.controls.release(
                        self.definition.document.body.settings,
                        voice.instrument_sources,
                        action,
                        voice.renderer,
                    )
                    self.controls.release(
                        self.definition.settings[voice.template],
                        voice.slot_sources,
                        action,
                        voice.renderer,
                    )
        else:
            raise synth.EngineError(
                f"Unsupported sampler action at frame {action.tick}: "
                f"{type(action).__name__}"
            )

    def _start_voice(self, action: trace.VoiceStart) -> None:
        slot, sample, definition = self._voice_definition(action)
        settings = self.definition.settings[slot.name]
        common = self.definition.document.body.settings
        crossfade_sources: list[int] = []
        for fade in slot.crossfades:
            if not isinstance(fade, crossfade.ControlCrossfade):
                continue
            part = None if fade.scope == "instrument" else action.part
            trigger = action.trigger_id if fade.scope == "trigger" else None
            index = self.controls.context(fade.scope, part, trigger)
            if index is None:
                index = self.controls.new_context(fade.scope, part, trigger, 0, {})
            crossfade_sources.append(index)
        self.voices[action.voice_id] = SampleVoiceSnapshot(
            voice_id=action.voice_id,
            part=action.part,
            trigger_id=action.trigger_id,
            template=slot.name,
            renderer=sampler.SampleVoiceRenderer.start(
                definition,
                sample,
                backend=self.backend,
            ),
            instrument_sources=self.controls.sources(common, action, action.velocity),
            slot_sources=self.controls.sources(settings, action, action.velocity),
            crossfade_sources=crossfade_sources,
        )
        renderer = self.voices[action.voice_id].renderer
        index = (
            renderer.source.index
            + action.alignment_frames
            + renderer.source.direction * action.variation.offset_frames
        )
        renderer.source = renderer.source.model_copy(
            update={
                "index": index,
                "exhaustion_frame": None
                if sample.slice.start_frame <= index < sample.slice.end_frame
                else 0,
            }
        )

    def _voice_definition(
        self, action: trace.VoiceStart
    ) -> tuple[
        instrument.SampleSlot, sampler.PreparedSample, sampler.PreparedSampleVoice
    ]:
        if action.voice_id in self.voices or action.voice_id in self.pending:
            raise synth.EngineError(f"Duplicate active voice: {action.voice_id}")
        slot = self.slots.get(action.template)
        if slot is None:
            raise synth.EngineError("Voice start has no prepared sample slot")
        sample = self.definition.samples[slot.name]
        settings = self.definition.settings[slot.name]
        if (
            action.slice != slot.slice
            or synth.expand_patch_outputs(action.settings) != settings
            or action.channels != slot.channels
            or action.start_frame != sample.slice.start_frame + slot.alignment_frames
            or action.alignment_frames != slot.alignment_frames
        ):
            raise synth.EngineError("Voice start must match its prepared sample slot")
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
        return (
            slot,
            sample,
            sampler.PreparedSampleVoice(
                sample_rate=self.definition.sample_rate,
                slice=sample.slice,
                envelope=envelope,
                filters=[*settings.processing.filters, *common.processing.filters],
                filter_order=settings.processing.filter_order
                if settings.processing.filters
                else common.processing.filter_order,
                pitch_ratio=playback.pitch_ratio(
                    slot.mapping, action.pitch_hz, 0, action.variation.pitch_cents
                ),
                gain=_layer_gain(slot, action)
                * 10
                ** (
                    (
                        common.processing.volume_db
                        + settings.processing.volume_db
                        + action.variation.gain_db
                    )
                    / 20
                ),
                routes=self.definition.routes[slot.name],
            ),
        )

    def _render(self, start: int, frames: int) -> np.ndarray:
        output = np.zeros((frames, len(self.definition.channels)))
        cursor = start
        for at in sorted(
            {a.tick for a in self.pending.values() if a.tick < start + frames}
        ):
            output[cursor - start : at - start] = self._render_voices(
                cursor, at - cursor
            )
            self._activate_pending(at)
            cursor = at
        output[cursor - start :] = self._render_voices(cursor, start + frames - cursor)
        return output

    def _activate_pending(self, frame: int) -> None:
        for action in list(self.pending.values()):
            if action.tick <= frame:
                del self.pending[action.voice_id]
                self._start_voice(action)

    def _render_voices(self, start: int, frames: int) -> np.ndarray:
        output = np.zeros((frames, len(self.definition.channels)))
        for voice in list(self.voices.values()):
            count = voice.renderer.active_frames(frames)
            common = self.definition.document.body.settings
            settings = self.definition.settings[voice.template]
            offset = 0
            while offset < count:
                if voice.renderer.complete:
                    break
                # Parameters may become invalid after natural exhaustion. Use
                # their declared maximum tuning to find a prefix guaranteed to
                # remain within the source, then resolve/render that prefix in
                # one call. The last uncertain frame remains a one-frame call.
                size = _safe_render_frames(
                    voice.renderer,
                    self.definition.samples[voice.template],
                    common,
                    settings,
                    count - offset,
                )
                tuning, gains, common_filters = self.controls.parameters(
                    common, voice.instrument_sources, start + offset, size
                )
                slot_tuning, slot_gains, slot_filters = self.controls.parameters(
                    settings, voice.slot_sources, start + offset, size
                )
                ratios = voice.renderer.definition.pitch_ratio * np.exp2(
                    (tuning + slot_tuning) / 1200
                )
                output[offset : offset + size] += voice.renderer.render(
                    self.definition.samples[voice.template],
                    size,
                    ratios,
                    gains
                    * slot_gains
                    * self._crossfade_gain(voice, start + offset, size),
                    np.concatenate((slot_filters, common_filters), axis=1),
                )
                offset += size
            if voice.renderer.complete:
                del self.voices[voice.voice_id]
                self.controls.stop(voice.voice_id)
        return output

    def _crossfade_gain(
        self, voice: SampleVoiceSnapshot, start: int, frames: int
    ) -> np.ndarray:
        gain = np.ones(frames)
        fades = [
            f
            for f in self.slots[voice.template].crossfades
            if isinstance(f, crossfade.ControlCrossfade)
        ]
        for fade, index in zip(fades, voice.crossfade_sources, strict=True):
            state = next(
                r.state
                for r in self.controls.contexts[index].ramps
                if r.control == fade.control
                and r.state.smoothing == Fraction(str(fade.smoothing_seconds))
            )
            values = control.control_samples(
                state, start, frames, self.definition.sample_rate
            )
            gain *= _crossfade_weight(fade, values)
        return gain


class PersistentSampler(OfflineSampler):
    """Bounded block adapter for the native persistent sample renderer."""

    def __init__(
        self,
        definition: PreparedSampler,
        voices: int = 16,
        action_capacity: int = 64,
    ) -> None:
        if type(voices) is not int or voices <= 0:
            raise synth.EngineError(
                "Persistent sampler voice capacity must be positive"
            )
        if type(action_capacity) is not int or action_capacity <= 0:
            raise synth.EngineError(
                "Persistent sampler action capacity must be positive"
            )
        super().__init__(definition, backend="native")
        self.voice_capacity = voices
        self.action_capacity = action_capacity
        self.native_sources: dict[str, int] = {}
        self.native_voices: dict[str, NativeSampleVoice] = {}

    def advance(
        self, actions: list[instrument_trace.TraceAction], start: int, end: int
    ) -> np.ndarray:
        if len(actions) > self.action_capacity:
            raise synth.EngineError("Persistent sampler action capacity exceeded")
        return super().advance(actions, start, end)

    def advance_into(
        self,
        actions: list[instrument_trace.TraceAction],
        start: int,
        end: int,
        output: np.ndarray,
    ) -> None:
        if (
            output.shape != (end - start, len(self.definition.channels))
            or output.dtype != np.float64
            or not output.flags.c_contiguous
            or not output.flags.writeable
        ):
            raise synth.EngineError(
                "Persistent sampler output must be writable C-contiguous float64 "
                "with shape (frames, channels)"
            )
        output[:] = self.advance(actions, start, end)

    def snapshot(self) -> PersistentSamplerSnapshot:  # ty: ignore[invalid-method-override]
        return PersistentSamplerSnapshot(
            state=super().snapshot(),
            voice_capacity=self.voice_capacity,
            action_capacity=self.action_capacity,
            native_voices=self.native_voices.copy(),
        )

    def restore(  # ty: ignore[invalid-method-override]
        self, snapshot: PersistentSamplerSnapshot
    ) -> None:
        if snapshot.voice_capacity != self.voice_capacity:
            raise synth.EngineError("Snapshot belongs to a different voice capacity")
        if snapshot.action_capacity != self.action_capacity:
            raise synth.EngineError("Snapshot belongs to a different action capacity")
        super().restore(snapshot.state)
        self.native_voices = snapshot.native_voices.copy()

    def add_to_native_live(self, runtime: object, batch_capacity: int) -> None:
        from . import _native

        if not isinstance(runtime, _native.LiveRuntime):
            raise synth.EngineError("Invalid native live runtime")
        common = self.definition.document.body.settings
        if any(
            isinstance(f, crossfade.ControlCrossfade)
            for s in self.slots.values()
            for f in s.crossfades
        ):
            raise synth.EngineError(
                "Native live sampler control crossfades are not implemented"
            )
        all_settings = [common, *self.definition.settings.values()]
        if any(
            s.envelfos
            or s.bindings
            or s.modulation.sources
            or s.modulation.parameters
            or s.modulation.routes
            or s.processing.filters
            for s in all_settings
        ):
            raise synth.EngineError(
                "Native live sampler controls, generators, and filters are not "
                "implemented"
            )
        for slot in self.definition.document.body.slots:
            sample = self.definition.samples[slot.name]
            settings = self.definition.settings[slot.name]
            envelope = settings.envelope or common.envelope
            assert envelope is not None
            loop = sample.slice.loop
            selection = (
                sample.slice.start_frame,
                sample.slice.end_frame,
                sample.playback.direction == Direction.mirror,
                None
                if loop is None
                else (
                    loop.start_frame,
                    loop.end_frame,
                    loop.crossfade_frames,
                    loop.mode == LoopMode.until_release,
                ),
            )
            backward = sample.playback.direction == Direction.backward
            state = (
                sample.slice.end_frame - 1 if backward else sample.slice.start_frame,
                0,
                0,
                -1 if backward else 1,
                loop is not None,
                False,
                False,
                None,
            )
            routes = np.array(
                self.definition.routes[slot.name],
                dtype=np.float64,
            )
            self.native_sources[slot.name] = runtime.add_sample(
                sample.native_buffer,
                selection,
                state,
                self.definition.sample_rate,
                routes,
                cast(float, envelope.initial),
                synth._runtime_segments(envelope.segments, self.definition.sample_rate),
                synth._runtime_segments(envelope.release, self.definition.sample_rate),
                0,
                self.voice_capacity,
                batch_capacity,
            )

    def native_live_actions(
        self,
        actions: list[instrument_trace.TraceAction],
        start: int,
        end: int,
        runtime: object,
    ) -> dict[int, np.ndarray]:
        from . import _native

        if not isinstance(runtime, _native.LiveRuntime):
            raise synth.EngineError("Invalid native live runtime")
        if start != self.frame or end <= start:
            raise synth.EngineError(
                "advance must continue from the current nonempty interval"
            )
        ordered = sorted(actions, key=lambda a: (a.tick, a.ordinal))
        if len(ordered) > self.action_capacity:
            raise synth.EngineError("Persistent sampler action capacity exceeded")
        if any(a.tick < start or a.tick >= end for a in ordered):
            raise synth.EngineError("actions must belong to the rendered interval")
        active = {
            source: runtime.active_slots(source)
            for source in self.native_sources.values()
        }
        self.native_voices = {
            name: voice
            for name, voice in self.native_voices.items()
            if active[voice.source][voice.slot]
        }
        encoded: dict[int, list[list[float]]] = {
            source: [] for source in self.native_sources.values()
        }
        for action in ordered:
            offset = action.tick - start
            if isinstance(action, instrument_trace.TriggerContext) or isinstance(
                action, instrument_trace.Diagnostic
            ):
                continue
            if isinstance(action, instrument_trace.TriggerRelease):
                for name, voice in list(self.native_voices.items()):
                    if (
                        (voice.part, voice.trigger_id)
                        == (action.part, action.trigger_id)
                        and voice.trigger == enums.TriggerKind.start
                        and action.tick < voice.started_frame
                    ):
                        encoded[voice.source].append([offset, 2, voice.slot, 0, 0, 0])
                        active[voice.source][voice.slot] = False
                        del self.native_voices[name]
                continue
            if isinstance(action, trace.VoiceStart):
                if action.voice_id in self.native_voices:
                    raise synth.EngineError(
                        f"Duplicate active voice: {action.voice_id}"
                    )
                slot, sample, definition = self._voice_definition(action)
                source = self.native_sources[slot.name]
                voice = next(
                    (i for i, value in enumerate(active[source]) if not value), None
                )
                if voice is None:
                    raise synth.EngineError(
                        "Persistent sampler voice capacity exceeded"
                    )
                active[source][voice] = True
                delay = _delay_frames(slot, action, self.definition.sample_rate)
                if delay > 2**53:
                    raise synth.EngineError(
                        "Native sample delay is not exactly representable"
                    )
                self.native_voices[action.voice_id] = NativeSampleVoice(
                    source=source,
                    slot=voice,
                    part=action.part,
                    trigger_id=action.trigger_id,
                    started_frame=action.tick + delay,
                    trigger=slot.trigger,
                )
                tuning = (
                    self.definition.document.body.settings.processing.tuning_cents
                    + self.definition.settings[slot.name].processing.tuning_cents
                )
                position_offset = (
                    action.alignment_frames
                    + (-1 if sample.playback.direction == Direction.backward else 1)
                    * action.variation.offset_frames
                )
                if abs(position_offset) > 2**53:
                    raise synth.EngineError(
                        "Native sample offset is not exactly representable"
                    )
                encoded[source].append(
                    [
                        offset,
                        0,
                        voice,
                        definition.pitch_ratio
                        * np.exp2(tuning / 1200)
                        * sample.native_rate,
                        definition.gain,
                        position_offset,
                    ]
                )
                if delay:
                    encoded[source].append([offset, 6, voice, delay, 0, 0])
            elif isinstance(action, instrument_trace.VoiceRetirement):
                if (voice := self.native_voices.get(action.voice_id)) is not None:
                    kind = {"release": 1, "stop": 2, "fade": 5}[action.action]
                    if action.tick < voice.started_frame:
                        kind = 2
                    duration = (
                        0.0
                        if action.fade_seconds is None
                        else action.fade_seconds * self.definition.sample_rate
                    )
                    encoded[voice.source].append(
                        [offset, kind, voice.slot, duration, 0, 0]
                    )
                    if kind == 2:
                        active[voice.source][voice.slot] = False
                        del self.native_voices[action.voice_id]
            elif (
                isinstance(action, instrument_trace.EnvelfoObservation)
                and action.target == "envelope"
            ):
                if action.op is not None:
                    raise synth.EngineError("op requires an FM voice")
                command = (
                    "pause",
                    "resume",
                    "reverse",
                    "seek",
                    "shift",
                    "advance",
                ).index(action.action)
                for voice in self.native_voices.values():
                    if voice.started_frame <= action.tick and (
                        voice.part,
                        voice.trigger_id,
                    ) == (
                        action.part,
                        action.trigger_id,
                    ):
                        encoded[voice.source].append(
                            [
                                offset,
                                4,
                                voice.slot,
                                command,
                                action.position
                                if action.position is not None
                                else action.offset
                                if action.offset is not None
                                else 0,
                                0,
                            ]
                        )
            else:
                raise synth.EngineError(
                    f"Unsupported native live sampler action at frame {action.tick}: "
                    f"{type(action).__name__}"
                )
        return {
            source: np.asarray(rows, dtype=np.float64).reshape(-1, 6)
            for source, rows in encoded.items()
        }

    def finish_native_live_block(self, end: int, runtime: object) -> None:
        from . import _native

        if not isinstance(runtime, _native.LiveRuntime):
            raise synth.EngineError("Invalid native live runtime")
        active = {
            source: runtime.active_slots(source)
            for source in self.native_sources.values()
        }
        self.native_voices = {
            name: voice
            for name, voice in self.native_voices.items()
            if active[voice.source][voice.slot]
        }
        self.frame = end

    def _apply(self, action: instrument_trace.TraceAction) -> None:
        if (
            isinstance(action, trace.VoiceStart)
            and len(self.voices) + len(self.pending) >= self.voice_capacity
        ):
            raise synth.EngineError("Persistent sampler voice capacity exceeded")
        super()._apply(action)


def _delay_frames(
    slot: instrument.SampleSlot, action: trace.VoiceStart, sample_rate: int
) -> int:
    return ceil(
        (
            Fraction(str(slot.playback.start_delay_seconds))
            + Fraction(str(action.variation.delay_seconds))
        )
        * sample_rate
    )


def _validate_settings(settings: processing.SoundSettings) -> None:
    if settings.envelope is not None:
        synth.validate_envelope(settings.envelope)
    if settings.processing != processing.Processing(
        volume_db=settings.processing.volume_db,
        tuning_cents=settings.processing.tuning_cents,
        pan=settings.processing.pan,
        stereo_balance=settings.processing.stereo_balance,
        filters=settings.processing.filters,
        filter_order=settings.processing.filter_order,
    ):
        raise synth.EngineError(
            "Only sample volume, tuning, pan, balance, and filter processing "
            "are implemented"
        )
    synth.validate_generators(
        settings, allow_velocity_bindings=True, allow_key_bindings=True
    )
    synth.validate_modulation(settings)


def _safe_render_frames(
    renderer: sampler.SampleVoiceRenderer,
    source: sampler.PreparedSample,
    common: processing.SoundSettings,
    settings: processing.SoundSettings,
    frames: int,
) -> int:
    """Return a prefix known not to cross natural sample exhaustion.

    Filter/control values may be invalid after a finite source ends, so a batch
    must stop before that boundary. A currently looping source cannot exhaust.
    For a finite traversal, the maximum declared tuning bounds its furthest
    possible advance. The remaining uncertain frame is rendered alone.
    """
    state = renderer.source
    loop = source.slice.loop
    ending_loop = (
        loop is not None
        and loop.mode == LoopMode.until_release
        and renderer.release_frame is not None
        and renderer.release_frame <= renderer.frame_count
    )
    if state.looping and not ending_loop:
        return frames
    maximum = renderer.definition.pitch_ratio * np.exp2(
        (_maximum_tuning(common) + _maximum_tuning(settings)) / 1200
    )
    step = maximum * source.native_rate
    distance = (
        source.slice.end_frame - state.index
        if state.direction == 1
        else state.index - source.slice.start_frame + 1
    )
    remaining = distance * source.sample_rate - state.position
    known = int(np.floor(np.nextafter(remaining / step, -np.inf)))
    return max(1, min(frames, known))


def _layer_gain(slot: instrument.SampleSlot, action: trace.VoiceStart) -> float:
    gain = 1.0
    for fade in slot.crossfades:
        if isinstance(fade, crossfade.ControlCrossfade):
            continue
        value = (
            action.key if fade.input == enums.CrossfadeInput.key else action.velocity
        )
        if value is None:
            raise synth.EngineError("Key crossfade requires resolved onset key")
        gain *= float(_crossfade_weight(fade, np.asarray(value)))
    return gain


def _crossfade_weight(fade: crossfade.Crossfade, values: np.ndarray) -> np.ndarray:
    position = np.clip((values - fade.start) / (fade.end - fade.start), 0, 1)
    inward = fade.direction == enums.FadeDirection.fade_in
    weight = position if inward else 1 - position
    if fade.curve == enums.FadeCurve.equal_power:
        curved = np.sin(pi * position / 2) if inward else np.cos(pi * position / 2)
        weight = np.where((position == 0) | (position == 1), weight, curved)
    return weight


def _maximum_tuning(settings: processing.SoundSettings) -> float:
    return next(
        (
            parameter.maximum
            for parameter in settings.modulation.parameters
            if parameter.target.name == "processing"
            and parameter.target.parameter == "tuning_cents"
        ),
        settings.processing.tuning_cents,
    )
