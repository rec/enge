"""Offline sample instruments driven by prepared uFor actions."""

from fractions import Fraction
from hashlib import sha256
from json import dumps
from math import isfinite, pi
from typing import Literal

import numpy as np
from envelfo import patch
from pydantic import ConfigDict, Field
from ufor import instrument_trace
from ufor.base import Model
from ufor.envelfo import Patch
from ufor.events import PerformanceEvent
from ufor.samples import (
    crossfade,
    enums,
    instrument,
    playback,
    processing,
    selection,
    trace,
    variation,
)
from ufor.streams import AudioType

from . import (
    _native,
    control,
    filters,
    native,
    sampler,
    synth,
    voice_admission,
    voice_release,
)
from .performance import prepare_event_validator


class PreparedSampler(Model, frozen=True):
    document: instrument.SampleInstrumentScore
    sample_rate: int
    channels: list[str]
    samples: dict[str, sampler.PreparedSample]
    settings: dict[str, processing.SoundSettings]
    routes: dict[str, list[list[float]]]
    equalizer: dict[str, list[processing.BiquadCoefficients]]
    audio_digests: dict[str, str]
    slot_playback: dict[str, _native.PreparedSlotPlayback] = Field(exclude=True)

    model_config = ConfigDict(arbitrary_types_allowed=True)


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
    context_capacity: int
    native_contexts: list["NativeSampleContext"]


class NativeSampleContext(Model, frozen=True):
    source: int
    index: int
    part: str
    trigger_id: str | None


class NativeSampleControls(Model, frozen=True):
    names: list[str]
    scopes: list[str]
    events: dict[int, str]


class NativeSampleGenerators(Model, frozen=True):
    lfos: dict[str, list[int]]
    voice_lfos: dict[str, list[int]]
    envelopes: dict[str, list[int]]
    stages: dict[str, list[int]]
    scopes: list[str]
    patch_targets: list[str]


class NativeSampleVoice(Model, frozen=True):
    part: str
    trigger_id: str | None
    source: int
    slot: int
    started_frame: int
    trigger: enums.TriggerKind


def prepare_trace(
    definition: instrument.SampleInstrument,
    events: list[PerformanceEvent],
    seed: int,
    *,
    max_records: int = 10_000,
) -> trace.SampleTrace:
    """Prepare sampler actions with Rust uFor selection and variation."""
    slots = _native.PreparedSampleSlots(
        definition.model_dump_json(),
        max_records=max_records,
    )

    def matching_slots(context: trace.EligibilityContext) -> list[str]:
        return slots.matching_slots(dumps(context))

    def keyswitches(
        declarations: selection.Articulations,
    ) -> dict[int, tuple[str | None, bool]]:
        return _native.prepare_sample_keyswitches(
            declarations.model_dump_json(), max_records=max_records
        )

    return trace.prepare(
        definition,
        events,
        seed,
        eligible_slots=matching_slots,
        advance_selection=_advance_selection,
        admit_voices=voice_admission.native_retirements,
        release_plan=voice_release.native_release_plan,
        prepare_start=_native.prepare_voice_start,
        slot_settings=lambda _: _resolve_slots(slots),
        draw_random_range=_native.draw_sample_range,
        resolve_variation=_resolve_variation,
        validate_event=prepare_event_validator(definition.settings.controls),
        prepare_keyswitches=keyswitches,
        prepare_sustain=voice_release.native_initially_pressed,
    )


def _resolve_slots(
    slots: _native.PreparedSampleSlots,
) -> dict[str, instrument.ResolvedSlotSettings]:
    return {
        n: instrument.ResolvedSlotSettings.model_validate_json(t)
        for n, t in slots.resolved_settings().items()
    }


def _advance_selection(
    definition: selection.Selection, seed: int, sequence: selection.SelectionSequence
) -> tuple[str, selection.SelectionSequence]:
    choice, state = _native.advance_sample_selection(
        definition.mode, seed, sequence.model_dump_json()
    )
    return choice, selection.SelectionSequence.model_validate_json(state)


def _resolve_variation(
    definition: variation.Variation,
    seed: int,
    part: str,
    trigger_id: str | None,
    tick: int,
    ordinal: int,
    identity: str,
) -> variation.ResolvedVariation:
    return variation.ResolvedVariation.model_validate_json(
        _native.resolve_sample_variation(
            definition.model_dump_json(),
            seed,
            part,
            trigger_id,
            tick,
            ordinal,
            identity,
        )
    )


def prepare(
    score: instrument.SampleInstrumentScore,
    audio: dict[str, np.ndarray],
    *,
    max_records: int = 10_000,
) -> PreparedSampler:
    """Prepare already-decoded assets; file discovery and decoding stay outside.

    Asset arrays must match the declared native frame/channel layout. Decoded
    content fingerprints bind snapshots to the actual audio, independently of
    the document's encoded-file hashes. `max_records` bounds preparation of
    the asset inventory and the slots.
    """
    score = instrument.SampleInstrumentScore.model_validate(score.model_dump())
    output = score.outputs[0].stream
    assert isinstance(output, AudioType)
    try:
        sample_rate, native_rates = _native.prepare_sample_assets(
            score.model_dump_json(),
            {n: a.shape for n, a in audio.items()},
            max_records=max_records,
        )
    except ValueError as error:
        raise synth.EngineError(str(error)) from error
    assets = {a.name: a for a in score.assets}
    slices = {s.name: s for s in score.body.slices}
    decoded: dict[str, np.ndarray] = {}
    digests: dict[str, str] = {}
    for name in native_rates:
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
    slot_playback: dict[str, _native.PreparedSlotPlayback] = {}
    default_playback = score.body.settings.playback.model_dump_json()
    settings: dict[str, processing.SoundSettings] = {}
    routes: dict[str, list[list[float]]] = {}
    equalizer: dict[str, list[processing.BiquadCoefficients]] = {}
    resolved_slots = _resolve_slots(
        _native.PreparedSampleSlots(
            score.body.model_dump_json(), max_records=max_records
        )
    )
    instrument_processing = score.body.settings.processing
    common_bands = [
        b.biquad_coefficients(sample_rate) for b in instrument_processing.equalizer
    ]
    for slot in score.body.slots:
        settings[slot.name] = synth.expand_patch_outputs(
            resolved_slots[slot.name].settings
        )
        _validate_settings(settings[slot.name])
        filters.parameters(settings[slot.name].processing.filters, sample_rate, 1)
        selection = slices[slot.slice]
        source_channels = assets[selection.asset].audio.channels
        slot_processing = settings[slot.name].processing
        equalizer[slot.name] = [
            *(b.biquad_coefficients(sample_rate) for b in slot_processing.equalizer),
            *common_bands,
        ]
        routes[slot.name] = native.channel_routes(
            source_channels,
            output.channels,
            slot.channels,
            (
                score.body.pan_law,
                instrument_processing.pan + slot_processing.pan,
                instrument_processing.stereo_balance + slot_processing.stereo_balance,
            ),
        )
        traversal = _native.PreparedSlotPlayback(
            slot.playback.model_dump_json(), default_playback
        )
        slot_playback[slot.name] = traversal
        resolved, additional_plays, end_fade_seconds = traversal.resolved()
        samples[slot.name] = sampler.PreparedSample(
            samples=decoded[selection.asset],
            native_rate=native_rates[selection.asset],
            sample_rate=sample_rate,
            slice=selection,
            play_count=additional_plays + 1,
            end_fade_seconds=end_fade_seconds,
            playback=playback.Playback.model_validate_json(resolved),
        )
    return PreparedSampler(
        document=score,
        sample_rate=sample_rate,
        channels=output.channels,
        samples=samples,
        settings=settings,
        routes=routes,
        equalizer=equalizer,
        audio_digests=digests,
        slot_playback=slot_playback,
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
        self.native_crossfades = (
            {
                s.name: _native.SampleCrossfades(
                    [f.model_dump_json() for f in s.crossfades]
                )
                for s in self.slots.values()
            }
            if backend == "native"
            else {}
        )
        declarations = definition.document.body.settings.controls
        if backend == "native":
            encoded = dumps(
                {n: c.model_dump(mode="json") for n, c in declarations.items()}
            )
            try:
                self.native_fade_routes = {
                    name: fades.control_routes(encoded, definition.sample_rate)
                    for name, fades in self.native_crossfades.items()
                }
            except ValueError as error:
                raise synth.EngineError(str(error)) from error
            self.reference_fades = {}
            self.crossfade_controls = {
                name: [
                    (scope, control, Fraction(*duration) / definition.sample_rate)
                    for control, scope, _, duration, _, _ in routes
                ]
                for name, routes in self.native_fade_routes.items()
            }
        else:
            self.native_fade_routes = {}
            self.reference_fades = {
                s.name: [
                    f for f in s.crossfades if isinstance(f, crossfade.ControlCrossfade)
                ]
                for s in self.slots.values()
            }
            self.crossfade_controls = {
                name: [
                    (f.scope, f.control, Fraction(str(f.smoothing_seconds)))
                    for f in fades
                ]
                for name, fades in self.reference_fades.items()
            }
        self.native_mappings = (
            {
                s.name: _native.SamplePitchMapping(s.mapping.model_dump_json())
                for s in self.slots.values()
            }
            if backend == "native"
            else {}
        )
        self.native_tuning_bounds: dict[str, tuple[float, float]] = {}
        if backend == "native":
            common, *slots = _native.prepare_sample_tuning_bounds(
                [
                    (
                        s.processing.tuning_cents,
                        s.modulation.model_dump_json(include={"parameters"}),
                    )
                    for s in (
                        definition.document.body.settings,
                        *definition.settings.values(),
                    )
                ]
            )
            self.native_tuning_bounds = {
                name: (common, slot)
                for name, slot in zip(definition.settings, slots, strict=True)
            }
        self.controls = synth.ControlRenderer(
            definition.sample_rate,
            declarations,
            [definition.document.body.settings, *definition.settings.values()],
            backend,
            control_interval,
            control_sources=[
                source
                for sources in self.crossfade_controls.values()
                for source in sources
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
            delay = self.definition.slot_playback[slot.name].delay_frames(
                str(action.variation.delay_seconds), self.definition.sample_rate
            )
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
                    assert voice.renderer.release_frame is not None
                    release_at = (
                        Fraction(action.tick, self.definition.sample_rate)
                        + (voice.renderer.release_frame - voice.renderer.frame_count)
                        / self.definition.sample_rate
                    )
                    self.controls.release(
                        self.definition.document.body.settings,
                        voice.instrument_sources,
                        action,
                        release_at,
                    )
                    self.controls.release(
                        self.definition.settings[voice.template],
                        voice.slot_sources,
                        action,
                        release_at,
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
        for scope, _, _ in self.crossfade_controls[slot.name]:
            part = None if scope == "instrument" else action.part
            trigger = action.trigger_id if scope == "trigger" else None
            index = self.controls.context(scope, part, trigger)
            if index is None:
                index = self.controls.new_context(scope, part, trigger, 0, {})
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
                alignment_frames=action.alignment_frames,
                variation_frames=action.variation.offset_frames,
            ),
            instrument_sources=self.controls.sources(common, action, action.velocity),
            slot_sources=self.controls.sources(settings, action, action.velocity),
            crossfade_sources=crossfade_sources,
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
        common = self.definition.document.body.settings
        volume_db = (
            common.processing.volume_db,
            settings.processing.volume_db,
            action.variation.gain_db,
        )
        if self.backend == "native":
            try:
                ratio = self.native_mappings[slot.name].pitch_ratio(
                    action.pitch_hz, 0, action.variation.pitch_cents
                )
                gain = self.native_crossfades[slot.name].onset_gain(
                    action.key, action.velocity, volume_db
                )
            except ValueError as error:
                raise synth.EngineError(str(error)) from error
        else:
            if slot.mapping.pitch_tracking and (
                action.pitch_hz is None
                or not isfinite(action.pitch_hz)
                or action.pitch_hz <= 0
            ):
                raise synth.EngineError(
                    "Pitch-tracked sample requires positive resolved pitch_hz"
                )
            ratio = playback.pitch_ratio(
                slot.mapping, action.pitch_hz, 0, action.variation.pitch_cents
            )
            instrument_db, slot_db, variation_db = volume_db
            gain = _layer_gain(slot, action) * 10 ** (
                (instrument_db + slot_db + variation_db) / 20
            )
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
                equalizer=self.definition.equalizer[slot.name],
                filter_order=settings.processing.filter_order
                if settings.processing.filters or settings.processing.equalizer
                else common.processing.filter_order,
                pitch_ratio=ratio,
                gain=gain,
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
            sample = self.definition.samples[voice.template]
            pitch_ratio = voice.renderer.definition.pitch_ratio
            if self.backend == "native":
                common_maximum, slot_maximum = self.native_tuning_bounds[voice.template]
                maximum_step = _native.maximum_sample_step(
                    pitch_ratio, sample.native_rate, common_maximum, slot_maximum
                )
            else:
                common_maximum = _maximum_tuning(common)
                slot_maximum = _maximum_tuning(settings)
                with np.errstate(over="ignore", invalid="ignore"):
                    maximum_step = float(
                        pitch_ratio
                        * np.exp2((common_maximum + slot_maximum) / 1200)
                        * sample.native_rate
                    )
            offset = 0
            exhausted = False
            while offset < count:
                if voice.renderer.complete:
                    break
                # Parameters may become invalid after natural exhaustion. Use
                # their declared maximum tuning to find a prefix guaranteed to
                # remain within the source, then resolve/render that prefix in
                # one call. The last uncertain frame remains a one-frame call.
                size = voice.renderer.active_source_prefix(
                    sample, maximum_step, count - offset
                )
                if size == 0:
                    exhausted = True
                    break
                tuning, gains, common_filters = self.controls.parameters(
                    common, voice.instrument_sources, start + offset, size
                )
                slot_tuning, slot_gains, slot_filters = self.controls.parameters(
                    settings, voice.slot_sources, start + offset, size
                )
                if self.backend == "native":
                    try:
                        ratios = _native.tuned_sample_ratios(
                            pitch_ratio, tuning, slot_tuning
                        )
                    except ValueError as error:
                        raise synth.EngineError(
                            "Sample pitch ratios must be a positive finite vector"
                        ) from error
                else:
                    ratios = pitch_ratio * np.exp2((tuning + slot_tuning) / 1200)
                output[offset : offset + size] += voice.renderer.render(
                    sample,
                    size,
                    ratios,
                    gains
                    * slot_gains
                    * self._crossfade_gain(voice, start + offset, size),
                    np.concatenate((slot_filters, common_filters), axis=1),
                )
                offset += size
            if exhausted or voice.renderer.complete:
                del self.voices[voice.voice_id]
                self.controls.stop(voice.voice_id)
        return output

    def _crossfade_gain(
        self, voice: SampleVoiceSnapshot, start: int, frames: int
    ) -> np.ndarray:
        gain = np.ones(frames)
        for route, ((_, name, smoothing), index) in enumerate(
            zip(
                self.crossfade_controls[voice.template],
                voice.crossfade_sources,
                strict=True,
            )
        ):
            state = next(
                r.state
                for r in self.controls.contexts[index].ramps
                if r.control == name and r.state.smoothing == smoothing
            )
            values = control.control_samples(
                state, start, frames, self.definition.sample_rate
            )
            gain *= (
                self.native_crossfades[voice.template].control_weights(route, values)
                if self.backend == "native"
                else _crossfade_weight(
                    self.reference_fades[voice.template][route], values
                )
            )
        return gain


class PersistentSampler(OfflineSampler):
    """Bounded block adapter for the native persistent sample renderer."""

    def __init__(
        self,
        definition: PreparedSampler,
        voices: int = 16,
        action_capacity: int = 64,
        context_capacity: int = 64,
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
        if type(context_capacity) is not int or context_capacity <= 0:
            raise synth.EngineError(
                "Persistent sampler context capacity must be positive"
            )
        self.context_capacity = context_capacity
        self.voice_capacity = voices
        self.action_capacity = action_capacity
        self.native_sources: dict[str, int] = {}
        self.native_voices: dict[str, NativeSampleVoice] = {}
        self.native_controls: dict[int, NativeSampleControls] = {}
        self.native_generators: dict[int, NativeSampleGenerators] = {}
        self.native_contexts: list[NativeSampleContext] = []

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
            context_capacity=self.context_capacity,
            native_contexts=self.native_contexts.copy(),
        )

    def restore(  # ty: ignore[invalid-method-override]
        self, snapshot: PersistentSamplerSnapshot
    ) -> None:
        if snapshot.voice_capacity != self.voice_capacity:
            raise synth.EngineError("Snapshot belongs to a different voice capacity")
        if snapshot.action_capacity != self.action_capacity:
            raise synth.EngineError("Snapshot belongs to a different action capacity")
        if snapshot.context_capacity != self.context_capacity:
            raise synth.EngineError("Snapshot belongs to a different context capacity")
        super().restore(snapshot.state)
        self.native_voices = snapshot.native_voices.copy()
        self.native_contexts = snapshot.native_contexts.copy()

    def add_to_native_live(self, runtime: object, batch_capacity: int) -> None:
        if not isinstance(runtime, _native.LiveRuntime):
            raise synth.EngineError("Invalid native live runtime")
        common = self.definition.document.body.settings
        for slot in self.definition.document.body.slots:
            sample = self.definition.samples[slot.name]
            settings = self.definition.settings[slot.name]
            envelope = settings.envelope or common.envelope
            assert envelope is not None
            routes = np.array(
                self.definition.routes[slot.name],
                dtype=np.float64,
            )
            filter_definitions = [
                *settings.processing.filters,
                *common.processing.filters,
            ]
            filter_values = filters.parameters(
                filter_definitions, self.definition.sample_rate, 1
            )
            filter_states = filters.initial_states(
                filter_definitions,
                sample.samples.shape[1],
                len(self.definition.equalizer[slot.name]),
            )
            filter_order = (
                settings.processing.filter_order
                if settings.processing.filters or settings.processing.equalizer
                else common.processing.filter_order
            )
            self.native_sources[slot.name] = runtime.add_sample(
                sample.native_traversal,
                routes,
                envelope.model_dump_json(),
                0,
                self.voice_capacity,
                batch_capacity,
                filters.native_inputs(
                    filter_definitions,
                    filter_states,
                    filter_values,
                    self.definition.equalizer[slot.name],
                ),
                filter_order == processing.FilterOrder.after_amplitude,
            )
            rows = []
            smoothing = []
            parameters = [0.0] * 12
            filter_groups: list[list[tuple[int, int, int, bool, float, float]]] = []
            maps = []
            names = []
            scopes = []
            events: dict[int, str] = {}
            generator_groups: list[
                tuple[processing.SoundSettings, synth.PersistentModulation, list[int]]
            ] = []
            for group, template in enumerate((common, settings)):
                prepared = synth._persistent_modulation(
                    self.definition.sample_rate,
                    common.controls,
                    template,
                )
                controls = prepared[0]
                filter_offset = len(parameters) // 3
                if template.envelfos:
                    generator_groups.append(
                        (
                            template,
                            prepared,
                            [
                                2 * group,
                                2 * group + 1,
                                *range(
                                    filter_offset,
                                    filter_offset + len(prepared[2]) // 3 - 2,
                                ),
                            ],
                        )
                    )
                controls[:, 2] = np.where(
                    controls[:, 2] < 2,
                    controls[:, 2] + 2 * group,
                    controls[:, 2] + filter_offset - 2,
                )
                rows.append(controls)
                events.update({len(names) + i: k for i, k in prepared[19].items()})
                smoothing.extend(prepared[1])
                parameters[group * 6 : group * 6 + 6] = prepared[2][:6]
                parameters.extend(prepared[2][6:])
                filter_groups.append(
                    [
                        (
                            filter_offset + 2 * i,
                            filter_offset + 2 * i + 1,
                            int(stages),
                            bool(boundary),
                            float(minimum),
                            float(maximum),
                        )
                        for i, (
                            _,
                            stages,
                            boundary,
                            minimum,
                            maximum,
                            _,
                            _,
                        ) in enumerate(prepared[14])
                    ]
                )
                scopes.extend(prepared[4])
                names.extend(prepared[5])
                maps.extend(prepared[18][0])
            source = self.native_sources[slot.name]
            fades = []
            for (
                name,
                scope,
                facts,
                duration,
                interval,
                shape,
            ) in self.native_fade_routes[slot.name]:
                default, scope_index, minimum = facts
                start, end = interval
                rows.append(
                    np.array(
                        [
                            [
                                default,
                                scope_index,
                                len(parameters) // 3,
                                0,
                                minimum,
                                1,
                            ]
                        ],
                        dtype=np.float64,
                    )
                )
                smoothing.append(duration)
                parameters.extend([0, 0, 1])
                scopes.append(scope)
                names.append(name)
                maps.append((0, [(start, 0.0), (end, 1.0)]))
                fades.append(shape)
            runtime.set_sample_controls(
                source,
                np.concatenate(rows),
                smoothing,
                parameters,
                maps,
                self.context_capacity,
                fades,
                [f for g in reversed(filter_groups) for f in g],
            )
            self.native_controls[source] = NativeSampleControls(
                names=names, scopes=scopes, events=events
            )
            if generator_groups:
                bank = _native.GeneratorBank(
                    self.definition.sample_rate,
                    self.voice_capacity,
                    len(parameters) // 3,
                    np.empty((0, 4)),
                    [],
                    [],
                    [],
                )
                generator_maps: list[list[tuple[int, list[tuple[float, float]]]]] = [
                    [],
                    [],
                    [],
                    [],
                ]
                metadata = NativeSampleGenerators(
                    lfos={},
                    voice_lfos={},
                    envelopes={},
                    stages={},
                    scopes=[],
                    patch_targets=[],
                )
                for template, prepared, parameter_map in generator_groups:
                    generators, _, transforms = synth._prepare_native_generators(
                        self.definition.sample_rate,
                        self.voice_capacity,
                        template,
                        prepared,
                    )
                    lfo, envelope, stage, _ = bank.append(generators, parameter_map)
                    for target, sources, base in (
                        (
                            metadata.lfos,
                            {
                                n: v
                                for n, v in prepared[8].items()
                                if template.envelfos[n].scope == "instrument"
                            },
                            lfo,
                        ),
                        (metadata.voice_lfos, prepared[9], lfo),
                        (metadata.envelopes, prepared[12], envelope),
                        (metadata.stages, prepared[16], stage),
                    ):
                        for name, indices in sources.items():
                            target.setdefault(name, []).extend(
                                base + i for i in indices
                            )
                    metadata.scopes.extend(sorted(prepared[13]))
                    metadata.patch_targets.extend(
                        sorted(synth._patch_child_names(template))
                    )
                    metadata.patch_targets.extend(
                        n for n, v in template.envelfos.items() if isinstance(v, Patch)
                    )
                    for target, mappings in zip(
                        generator_maps, [*prepared[18][1:], transforms], strict=True
                    ):
                        target.extend(mappings)
                runtime.set_sample_generators(source, bank, generator_maps)
                self.native_generators[source] = metadata

    def native_live_actions(
        self,
        actions: list[instrument_trace.TraceAction],
        start: int,
        end: int,
        runtime: object,
    ) -> dict[int, np.ndarray]:
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
        contexts = {s: runtime.active_contexts(s) for s in self.native_sources.values()}
        self.native_contexts = [
            c for c in self.native_contexts if contexts[c.source][c.index]
        ]
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
            if isinstance(action, instrument_trace.Diagnostic):
                continue
            if isinstance(action, instrument_trace.TriggerContext):
                declarations = self.definition.document.body.settings.controls
                if action.controls.keys() != declarations.keys():
                    raise synth.EngineError(
                        "Trigger context must contain all declared controls"
                    )
                for name, value in action.controls.items():
                    declarations[name].validate_value(value)
                for source, controls in self.native_controls.items():
                    if "trigger" not in controls.scopes:
                        continue
                    context = self._native_context(
                        source,
                        action.part,
                        action.trigger_id,
                        contexts[source],
                        encoded[source],
                        offset,
                    )
                    for index, (name, scope) in enumerate(
                        zip(controls.names, controls.scopes, strict=True)
                    ):
                        if scope == "trigger":
                            encoded[source].append(
                                [offset, 10, context, index, action.controls[name], 0]
                            )
                continue
            if isinstance(action, instrument_trace.ControlObservation):
                declaration = self.definition.document.body.settings.controls.get(
                    action.control
                )
                if declaration is None:
                    raise synth.EngineError(f"Unknown control: {action.control}")
                declaration.validate_value(action.value)
                for source, controls in self.native_controls.items():
                    indices = [
                        i
                        for i, (n, s) in enumerate(
                            zip(controls.names, controls.scopes, strict=True)
                        )
                        if (n, s) == (action.control, action.scope)
                    ]
                    if not indices:
                        continue
                    context = -1
                    if action.scope == "part":
                        assert action.part is not None
                        context = self._native_context(
                            source,
                            action.part,
                            None,
                            contexts[source],
                            encoded[source],
                            offset,
                        )
                    elif action.scope == "trigger":
                        context = next(
                            (
                                c.index
                                for c in self.native_contexts
                                if (c.source, c.part, c.trigger_id)
                                == (source, action.part, action.trigger_id)
                            ),
                            -1,
                        )
                        if context == -1:
                            continue
                    for index in indices:
                        encoded[source].append(
                            [offset, 7, index, action.value, context, 0]
                        )
                continue
            if isinstance(action, instrument_trace.LFOObservation):
                found = False
                for source, generators in self.native_generators.items():
                    for index in generators.lfos.get(action.name, []):
                        found = True
                        encoded[source].append(
                            [
                                offset,
                                12,
                                index,
                                (
                                    "reset",
                                    "rate",
                                    "pause",
                                    "resume",
                                    "reverse",
                                    "seek",
                                    "shift",
                                ).index(action.action),
                                action.rate
                                if action.rate is not None
                                else action.position
                                if action.position is not None
                                else action.offset
                                if action.offset is not None
                                else 0,
                                0,
                            ]
                        )
                if not found:
                    raise synth.EngineError(f"Unknown instrument LFO: {action.name}")
                continue
            if (
                isinstance(action, instrument_trace.EnvelfoObservation)
                and action.target == "envelfo"
            ):
                assert action.name is not None
                if any(
                    action.name in g.patch_targets
                    for g in self.native_generators.values()
                ):
                    raise synth.EngineError(
                        "Patch-level Envelfo commands are not implemented"
                    )
                command = (
                    "pause",
                    "resume",
                    "reverse",
                    "seek",
                    "shift",
                    "advance",
                ).index(action.action)
                for voice in self.native_voices.values():
                    if voice.started_frame > action.tick or (
                        voice.part,
                        voice.trigger_id,
                    ) != (action.part, action.trigger_id):
                        continue
                    if (generators := self.native_generators.get(voice.source)) is None:
                        continue
                    for kind, sources in (
                        (13, generators.envelopes),
                        (14, generators.stages),
                        (15, generators.voice_lfos),
                    ):
                        for index in sources.get(action.name, []):
                            encoded[voice.source].append(
                                [
                                    offset,
                                    kind,
                                    voice.slot,
                                    index,
                                    command,
                                    action.position
                                    if action.position is not None
                                    else action.offset
                                    if action.offset is not None
                                    else 0,
                                ]
                            )
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
                delay = self.definition.slot_playback[slot.name].delay_frames(
                    str(action.variation.delay_seconds), self.definition.sample_rate
                )
                if delay > 2**53:
                    raise synth.EngineError(
                        "Native sample delay is not exactly representable"
                    )
                position_offset = sample.native_traversal.start_offset(
                    action.alignment_frames, action.variation.offset_frames
                )
                if abs(position_offset) > 2**53:
                    raise synth.EngineError(
                        "Native sample offset is not exactly representable"
                    )
                self.native_voices[action.voice_id] = NativeSampleVoice(
                    source=source,
                    slot=voice,
                    part=action.part,
                    trigger_id=action.trigger_id,
                    started_frame=action.tick + delay,
                    trigger=slot.trigger,
                )
                if source in self.native_generators:
                    encoded[source].append(
                        [
                            offset,
                            16,
                            voice,
                            action.envelfo_key & 0xFFFF_FFFF,
                            action.envelfo_key >> 32,
                            0,
                        ]
                    )
                encoded[source].append(
                    [
                        offset,
                        0,
                        voice,
                        definition.pitch_ratio * sample.native_rate,
                        definition.gain,
                        position_offset,
                    ]
                )
                scopes = set(self.native_controls[source].scopes)
                if source in self.native_generators:
                    scopes.update(self.native_generators[source].scopes)
                for index, kind in self.native_controls[source].events.items():
                    value = action.key if kind == "key" else action.velocity
                    if value is None:
                        raise synth.EngineError(
                            f"{kind.title()} binding requires resolved onset {kind}"
                        )
                    encoded[source].append([offset, 11, voice, index, value, 0])
                part_context = -1
                if "part" in scopes:
                    part_context = self._native_context(
                        source,
                        action.part,
                        None,
                        contexts[source],
                        encoded[source],
                        offset,
                    )
                trigger_context = next(
                    (
                        c.index
                        for c in self.native_contexts
                        if "trigger" in scopes
                        and (c.source, c.part, c.trigger_id)
                        == (source, action.part, action.trigger_id)
                    ),
                    -1,
                )
                if "trigger" in scopes and trigger_context == -1:
                    raise synth.EngineError(
                        "Voice is missing its trigger control context"
                    )
                if part_context != -1 or trigger_context != -1:
                    encoded[source].append(
                        [offset, 8, voice, part_context, trigger_context, 0]
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
        if any(len(r) > self.action_capacity for r in encoded.values()):
            raise synth.EngineError("Persistent sampler action capacity exceeded")
        return {
            source: np.asarray(rows, dtype=np.float64).reshape(-1, 6)
            for source, rows in encoded.items()
        }

    def finish_native_live_block(self, end: int, runtime: object) -> None:
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

    def _native_context(
        self,
        source: int,
        part: str,
        trigger_id: str | None,
        used: list[bool],
        rows: list[list[float]],
        offset: int,
    ) -> int:
        previous = next(
            (
                c
                for c in self.native_contexts
                if (c.source, c.part, c.trigger_id) == (source, part, trigger_id)
            ),
            None,
        )
        if previous is not None and trigger_id is None:
            return previous.index
        index = next((i for i, u in enumerate(used) if not u), None)
        if index is None:
            raise synth.EngineError("Persistent sampler context capacity exceeded")
        used[index] = True
        if previous is not None:
            self.native_contexts.remove(previous)
        self.native_contexts.append(
            NativeSampleContext(
                source=source, index=index, part=part, trigger_id=trigger_id
            )
        )
        rows.append([offset, 9, index, 1 if trigger_id is None else 2, 0, 0])
        return index

    def _apply(self, action: instrument_trace.TraceAction) -> None:
        if (
            isinstance(action, trace.VoiceStart)
            and len(self.voices) + len(self.pending) >= self.voice_capacity
        ):
            raise synth.EngineError("Persistent sampler voice capacity exceeded")
        super()._apply(action)


def _validate_settings(settings: processing.SoundSettings) -> None:
    if settings.envelope is not None:
        synth.validate_envelope(settings.envelope)
    if settings.processing != processing.Processing(
        volume_db=settings.processing.volume_db,
        tuning_cents=settings.processing.tuning_cents,
        pan=settings.processing.pan,
        stereo_balance=settings.processing.stereo_balance,
        equalizer=settings.processing.equalizer,
        filters=settings.processing.filters,
        filter_order=settings.processing.filter_order,
    ):
        raise synth.EngineError(
            "Only sample volume, tuning, pan, balance, static EQ and filter processing "
            "are implemented"
        )
    synth.validate_generators(
        settings, allow_velocity_bindings=True, allow_key_bindings=True
    )
    synth.validate_modulation(settings)


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
