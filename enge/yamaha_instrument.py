"""Offline Yamaha instruments consuming uFor's shared lifecycle actions."""

from copy import deepcopy
from math import ceil, floor

import numpy as np
from pydantic import BaseModel, ConfigDict, Field, FiniteFloat
from ufor import instrument_trace, synth_trace, tx81z
from ufor.dx7 import DX7Performance
from ufor.samples.processing import FilterOrder, Processing
from ufor.streams import AudioType
from ufor.synth import SynthInstrumentScore, YamahaDX7Voice, YamahaTX81ZVoice

from . import filters, synth, yamaha, yamaha_lfo


class PreparedYamaha(BaseModel, frozen=True):
    sample_rate: int
    channels: list[str]
    score: SynthInstrumentScore


class VoiceState(BaseModel):
    voice_id: str
    template: str
    part: str
    start: int
    oscillator_slot: int | None = None
    sources: dict[str, int] = Field(default_factory=dict)
    release_at: int | None = None
    released: bool = False
    fade_at: int | None = None
    fade_frames: float = 1
    fade_gain: float = 1
    filter_states: list[filters.FilterState] = Field(default_factory=list)


class Voice(VoiceState):
    renderer: yamaha.DX7Renderer | yamaha.TX81ZRenderer = Field(exclude=True)

    model_config = ConfigDict(arbitrary_types_allowed=True)


class VoiceSnapshot(VoiceState):
    note: yamaha.DX7Snapshot | yamaha.TX81ZSnapshot


class OscillatorSlot(BaseModel):
    """Retained DX7 Q24 phase and increment state for sync-off allocation."""

    phases: list[int] = Field(min_length=6, max_length=6)
    phase_steps: list[int] = Field(min_length=6, max_length=6)
    idle_since: int | None = None


class PitchHistory(BaseModel):
    """Previous TX81Z sounding fundamental, after transpose and octave wrapping."""

    voice_id: str
    pitch_hz: float


class MonoNote(BaseModel, frozen=True):
    action: synth_trace.VoiceStart
    pitch_hz: float
    sources: dict[str, int]


class YamahaSnapshot(BaseModel, frozen=True):
    definition: PreparedYamaha
    frame: int
    voices: list[VoiceSnapshot]
    clocks: dict[str, yamaha_lfo.DX7LFOSnapshot | yamaha_lfo.TX81ZLFOSnapshot]
    vibrato_clocks: dict[str, yamaha_lfo.TX81ZLFOSnapshot]
    oscillator_slots: dict[str, list[OscillatorSlot]]
    pitch_history: dict[str, dict[str, PitchHistory]]
    mono_notes: list[MonoNote]
    contexts: list[synth.ControlContext]
    output_filters: dict[str, list[filters.FilterState]]
    held_dac: dict[str, FiniteFloat]


class OfflineYamaha:
    """Shared Yamaha clocks with explicit TX81Z performance ownership.

    Python offline execution only. Internal operator envelopes precede the
    external processing chain; filter_order controls placement of outer gain.
    """

    def __init__(self, definition: PreparedYamaha) -> None:
        self.definition = definition.model_copy(deep=True)
        self.frame = 0
        self.templates = {
            v.name: v
            for v in self.definition.score.body.voices
            if isinstance(v, (YamahaDX7Voice, YamahaTX81ZVoice))
        }
        self.authored_templates = self.templates.copy()
        self.lfo_groups = {
            n: g for g in self.definition.score.body.tx81z_lfos for n in g.routes
        }
        self.clocks = {
            n: yamaha_lfo.DX7LFOContext(v.patch, definition.sample_rate)
            if isinstance(v, YamahaDX7Voice)
            else yamaha_lfo.TX81ZLFOContext(v.patch, definition.sample_rate)
            for n, v in self.templates.items()
            if n not in self.lfo_groups
        }
        for group in self.definition.score.body.tx81z_lfos:
            first = self.templates[group.first]
            second = self.templates[group.second] if group.second else None
            assert isinstance(first, YamahaTX81ZVoice)
            assert second is None or isinstance(second, YamahaTX81ZVoice)
            self.clocks[group.first] = yamaha_lfo.TX81ZLFOContext(
                first.patch,
                definition.sample_rate,
                secondary=second.patch if second else None,
            )
            for name, selector in group.routes.items():
                if selector not in ("first", "second"):
                    continue
                owner = first if selector == "first" else second
                template = self.templates[name]
                assert owner is not None and isinstance(template, YamahaTX81ZVoice)
                patch = bytearray(template.patch.data)
                patch[56:58] = owner.patch.data[56:58]
                self.templates[name] = template.model_copy(
                    update={"patch": tx81z.TX81ZVoice(data=bytes(patch))}
                )
        # Provisional software-vibrato model: independent patch clocks, PM only.
        self.vibrato_clocks = {
            n: yamaha_lfo.TX81ZLFOContext(v.patch, definition.sample_rate)
            for n, v in self.templates.items()
            if isinstance(v, YamahaTX81ZVoice)
            and n in self.lfo_groups
            and self.lfo_groups[n].routes[n] == "vibrato"
        }
        self.voices: dict[str, Voice] = {}
        self.output_groups = {
            v: o.name
            for o in self.definition.score.body.tx81z_outputs
            for v in o.voices
        }
        self.output_filters = {
            o.name: filters.initial_states(o.processing.filters, 1)
            for o in self.definition.score.body.tx81z_outputs
        }
        self.held_dac = dict.fromkeys(self.output_filters, 0.0)
        self.mono_notes: list[MonoNote] = []
        self.controls = synth.ControlRenderer(
            definition.sample_rate,
            self.definition.score.body.controls,
            list(self.templates.values()),
        )
        self.oscillator_slots: dict[str, list[OscillatorSlot]] = {
            n: []
            for n, v in self.templates.items()
            if isinstance(v, YamahaDX7Voice)
            and not yamaha.dx7_global_parameters(v.patch)[0]
        }
        self.pitch_history: dict[str, dict[str, PitchHistory]] = {
            n: {}
            for n, v in self.templates.items()
            if isinstance(v, YamahaTX81ZVoice) and v.patch.data[66]
        }

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

    def snapshot(self) -> YamahaSnapshot:
        return YamahaSnapshot(
            definition=self.definition,
            frame=self.frame,
            voices=[
                VoiceSnapshot(**v.model_dump(), note=v.renderer.snapshot())
                for v in self.voices.values()
            ],
            clocks={n: c.snapshot() for n, c in self.clocks.items()},
            vibrato_clocks={n: c.snapshot() for n, c in self.vibrato_clocks.items()},
            oscillator_slots=self.oscillator_slots,
            pitch_history=self.pitch_history,
            mono_notes=self.mono_notes,
            contexts=self.controls.contexts,
            output_filters=self.output_filters,
            held_dac=self.held_dac,
        ).model_copy(deep=True)

    def restore(self, snapshot: YamahaSnapshot) -> None:
        if (
            snapshot.definition != self.definition
            or snapshot.clocks.keys() != self.clocks.keys()
            or snapshot.vibrato_clocks.keys() != self.vibrato_clocks.keys()
            or snapshot.oscillator_slots.keys() != self.oscillator_slots.keys()
            or snapshot.pitch_history.keys() != self.pitch_history.keys()
            or snapshot.output_filters.keys() != self.output_filters.keys()
            or snapshot.held_dac.keys() != self.held_dac.keys()
        ):
            raise synth.EngineError("Snapshot belongs to a different Yamaha instrument")
        snapshot = snapshot.model_copy(deep=True)
        voices = {}
        for saved in snapshot.voices:
            renderer = _note(
                self.templates[saved.template],
                saved.note.note,
                saved.note.velocity,
                self.definition.sample_rate,
                saved.note.pitch_hz,
                start_pitch=saved.note.start_pitch_hz
                if isinstance(saved.note, yamaha.TX81ZSnapshot)
                else None,
            )
            if isinstance(renderer, yamaha.DX7Renderer) and isinstance(
                saved.note, yamaha.DX7Snapshot
            ):
                renderer.restore(saved.note)
            elif isinstance(renderer, yamaha.TX81ZRenderer) and isinstance(
                saved.note, yamaha.TX81ZSnapshot
            ):
                if saved.note.frame != snapshot.frame:
                    raise synth.EngineError(
                        "TX81Z voice clock disagrees with instrument frame"
                    )
                renderer.restore(saved.note)
            else:
                raise synth.EngineError(
                    "Yamaha snapshot note type does not match template"
                )
            voices[saved.voice_id] = Voice(
                **saved.model_dump(exclude={"note"}), renderer=renderer
            )
        clocks = deepcopy(self.clocks)
        for name, clock in clocks.items():
            saved_clock = snapshot.clocks[name]
            if isinstance(clock, yamaha_lfo.DX7LFOContext) and isinstance(
                saved_clock, yamaha_lfo.DX7LFOSnapshot
            ):
                clock.restore(saved_clock)
            elif isinstance(clock, yamaha_lfo.TX81ZLFOContext) and isinstance(
                saved_clock, yamaha_lfo.TX81ZLFOSnapshot
            ):
                if saved_clock.frame != snapshot.frame:
                    raise synth.EngineError(
                        "TX81Z LFO clock disagrees with instrument frame"
                    )
                clock.restore(saved_clock)
            else:
                raise synth.EngineError(
                    "Yamaha snapshot clock type does not match template"
                )
        vibrato_clocks = deepcopy(self.vibrato_clocks)
        for name, clock in vibrato_clocks.items():
            if snapshot.vibrato_clocks[name].frame != snapshot.frame:
                raise synth.EngineError(
                    "TX81Z vibrato clock disagrees with instrument frame"
                )
            clock.restore(snapshot.vibrato_clocks[name])
        self.voices = voices
        self.clocks = clocks
        self.vibrato_clocks = vibrato_clocks
        self.oscillator_slots = snapshot.oscillator_slots
        self.pitch_history = snapshot.pitch_history
        self.mono_notes = snapshot.mono_notes
        self.output_filters = snapshot.output_filters
        self.held_dac = snapshot.held_dac
        self.controls.clear_cache()
        self.controls.contexts = snapshot.contexts
        self.frame = snapshot.frame

    def _apply(self, action: instrument_trace.TraceAction) -> None:
        if isinstance(action, instrument_trace.Diagnostic):
            return
        if isinstance(action, instrument_trace.TriggerRelease):
            released = [
                n
                for n in self.mono_notes
                if (n.action.part, n.action.trigger_id)
                == (action.part, action.trigger_id)
            ]
            self.mono_notes = [n for n in self.mono_notes if n not in released]
            for note in released:
                if (voice := self.voices.get(note.action.voice_id)) is not None:
                    previous = [
                        n
                        for n in self.mono_notes
                        if (n.action.template, n.action.part)
                        == (voice.template, voice.part)
                    ]
                    if previous and voice.release_at is None and voice.fade_at is None:
                        self._legato(voice, previous[-1])
            return
        if self.controls.apply(action, self.voices):
            return
        if isinstance(action, instrument_trace.VoiceRetirement):
            self.mono_notes = [
                n for n in self.mono_notes if n.action.voice_id != action.voice_id
            ]
            if (voice := self.voices.get(action.voice_id)) is None:
                return
            if action.action == "stop":
                self._retire(voice, action.tick)
            elif action.action == "fade":
                assert action.fade_seconds is not None
                if voice.fade_at is not None:
                    voice.fade_gain *= max(
                        0, 1 - (action.tick - voice.fade_at) / voice.fade_frames
                    )
                voice.fade_at = action.tick
                voice.fade_frames = action.fade_seconds * self.definition.sample_rate
            elif voice.release_at is None:
                voice.release_at = max(
                    action.tick,
                    voice.start
                    + ceil(
                        self.templates[voice.template].minimum_hold_seconds
                        * self.definition.sample_rate
                    ),
                )
            return
        if not isinstance(action, synth_trace.VoiceStart):
            raise synth.EngineError("Unsupported Yamaha action")
        template = self.templates.get(action.template)
        if (
            template is None
            or action.settings != self.authored_templates[action.template]
            or action.channels != template.channels
            or action.key is None
            or action.velocity is None
        ):
            raise synth.EngineError(
                "Yamaha start must match its prepared template and carry key/velocity"
            )
        if action.voice_id in self.voices or any(
            n.action.voice_id == action.voice_id for n in self.mono_notes
        ):
            raise synth.EngineError("Duplicate active Yamaha voice")
        velocity = (
            0 if action.velocity == 0 else max(1, floor(action.velocity * 127 + 0.5))
        )
        pitch = action.pitch_hz
        if not template.mapping.pitch_tracking:
            reference = template.mapping.reference_pitch_hz
            assert reference is not None
            pitch = reference + template.frequency_offset_hz
        elif pitch is None:
            pitch = 440 * 2 ** ((action.key - 69) / 12) + template.frequency_offset_hz
        pitch *= 2 ** (template.processing.tuning_cents / 1200)
        mono = isinstance(template, YamahaTX81ZVoice) and bool(template.patch.data[63])
        sources = self.controls.sources(template, action, action.velocity)
        mono_voice = None
        note = None
        if mono:
            held = [
                n
                for n in self.mono_notes
                if (n.action.template, n.action.part) == (template.name, action.part)
            ]
            note = MonoNote(action=action, pitch_hz=pitch, sources=sources)
            mono_voice = next(
                (
                    v
                    for v in self.voices.values()
                    if (v.template, v.part) == (template.name, action.part)
                ),
                None,
            )
            if mono_voice is not None:
                if (
                    held
                    and mono_voice.release_at is None
                    and mono_voice.fade_at is None
                ):
                    self._legato(mono_voice, note)
                    self._remember(note)
                    return
        slot_index = None
        phases = None
        if template.name in self.oscillator_slots:
            slots = self.oscillator_slots[template.name]
            idle = [
                (s.idle_since, i)
                for i, s in enumerate(slots)
                if s.idle_since is not None
            ]
            if idle:
                idle_since, slot_index = min(idle)
                slot = slots[slot_index]
                elapsed = action.tick - idle_since
                phases = [
                    ((p + elapsed * s) % 2**24) / 2**24
                    for p, s in zip(slot.phases, slot.phase_steps, strict=True)
                ]
            else:
                slot_index = len(slots)
        start_pitch = None
        if template.name in self.pitch_history:
            if previous := self.pitch_history[template.name].get(action.part):
                start_pitch = previous.pitch_hz
                if active := self.voices.get(previous.voice_id):
                    assert isinstance(active.renderer, yamaha.TX81ZRenderer)
                    if active.renderer.portamento is not None:
                        start_pitch = active.renderer.portamento.current_hz
        if mono and template.patch.data[65]:
            start_pitch = None
        renderer = _note(
            template,
            action.key,
            velocity,
            self.definition.sample_rate,
            pitch,
            phases,
            start_pitch,
            start_frame=action.tick,
        )
        if mono_voice is not None:
            self._retire(mono_voice, action.tick)
        if note is not None:
            self._remember(note)
        if slot_index is not None:
            assert isinstance(renderer, yamaha.DX7Renderer)
            slot = OscillatorSlot(
                phases=renderer.phases.tolist(),
                phase_steps=renderer.phase_steps.tolist(),
            )
            slots = self.oscillator_slots[template.name]
            if slot_index == len(slots):
                slots.append(slot)
            else:
                slots[slot_index] = slot
        group = self.lfo_groups.get(template.name)
        self.clocks[group.first if group else template.name].keydown()
        if template.name in self.vibrato_clocks:
            self.vibrato_clocks[template.name].keydown()
        if template.name in self.pitch_history:
            assert isinstance(renderer, yamaha.TX81ZRenderer)
            self.pitch_history[template.name][action.part] = PitchHistory(
                voice_id=action.voice_id, pitch_hz=renderer.sounding_pitch_hz
            )
        self.voices[action.voice_id] = Voice(
            voice_id=action.voice_id,
            template=template.name,
            part=action.part,
            start=action.tick,
            oscillator_slot=slot_index,
            sources=sources,
            renderer=renderer,
            filter_states=filters.initial_states(template.processing.filters, 1),
        )

    def _remember(self, note: MonoNote) -> None:
        # Only onset templates represent a physically held key. Single mode
        # remembers eight previous keys in addition to the selected key.
        if note.action.settings.trigger != "start":
            return
        held = [
            n
            for n in self.mono_notes
            if (n.action.template, n.action.part)
            == (note.action.template, note.action.part)
        ]
        if len(held) == 9:
            self.mono_notes.remove(held[0])
        self.mono_notes.append(note)

    def _legato(self, voice: Voice, note: MonoNote) -> None:
        assert isinstance(voice.renderer, yamaha.TX81ZRenderer)
        action = note.action
        assert action.key is not None and action.velocity is not None
        velocity = (
            0 if action.velocity == 0 else max(1, floor(action.velocity * 127 + 0.5))
        )
        voice.renderer.retarget(action.key, velocity, note.pitch_hz)
        del self.voices[voice.voice_id]
        self.controls.stop(voice.voice_id)
        voice.voice_id = action.voice_id
        voice.sources = note.sources
        voice.release_at = None
        self.voices[voice.voice_id] = voice
        if voice.template in self.pitch_history:
            self.pitch_history[voice.template][voice.part] = PitchHistory(
                voice_id=voice.voice_id, pitch_hz=voice.renderer.sounding_pitch_hz
            )

    def _render(self, start: int, frames: int) -> np.ndarray:
        output = np.zeros((frames, len(self.definition.channels)))
        groups = {n: np.zeros((frames, 1)) for n in self.output_filters}
        rows = {}
        tick_ends = {}
        for name, clock in self.clocks.items():
            if isinstance(clock, yamaha_lfo.TX81ZLFOContext):
                tick_ends[name], rows[name] = clock.advance_ticks(frames)
            else:
                rows[name] = clock.advance(frames)
        vibrato_rows = {
            n: c.advance_ticks(frames)[1] for n, c in self.vibrato_clocks.items()
        }
        for group in self.definition.score.body.tx81z_lfos:
            shared = rows[group.first]
            for name, selector in group.routes.items():
                tick_ends[name] = tick_ends[group.first]
                if selector == "off":
                    rows[name] = np.zeros((len(shared), 4), dtype=np.int64)
                    rows[name][:, 3] = 1
                elif selector == "vibrato":
                    rows[name] = vibrato_rows[name]
                    rows[name][:, 1] = 0
                else:
                    rows[name] = (
                        shared[:, int(selector == "second")] if group.second else shared
                    )
        for voice in list(self.voices.values()):
            template = self.templates[voice.template]
            audio = np.zeros((frames, 1))
            count = 0
            dynamic_tuning = any(
                p.target.parameter == "tuning_cents"
                for p in template.modulation.parameters
            )
            pitch_bend = any(
                p.target.name == "yamaha" and p.target.parameter == "pitch_bend"
                for p in template.modulation.parameters
            )
            portamento_switch = any(
                p.target.name == "yamaha" and p.target.parameter == "portamento"
                for p in template.modulation.parameters
            )
            assignments = isinstance(template, YamahaDX7Voice) and any(
                a.pitch or a.amplitude or a.eg_bias
                for a in (
                    template.performance.modulation_wheel,
                    template.performance.breath,
                    template.performance.foot,
                )
            )
            if isinstance(template, YamahaTX81ZVoice):
                assignments = any(
                    template.patch.data[71:75] + template.additional.data[21:23]
                ) or bool(template.patch.data[67] or template.patch.data[76])
            for i in range(frames):
                at = start + i
                if (
                    voice.fade_at is not None
                    and at - voice.fade_at >= voice.fade_frames
                ):
                    break
                if (
                    not voice.released
                    and voice.release_at is not None
                    and at >= voice.release_at
                ):
                    voice.renderer.release()
                    voice.released = True
                if _complete(voice):
                    break
                tuning = None
                bend = None
                controllers = None
                breath = None
                foot = None
                portamento_enabled = None
                if dynamic_tuning or pitch_bend or assignments or portamento_switch:
                    values = self.controls.values(template, voice.sources, at, 1)
                    if portamento_switch:
                        portamento_enabled = values[("yamaha", "portamento")]
                    if dynamic_tuning:
                        tuning = (
                            values[("processing", "tuning_cents")]
                            - template.processing.tuning_cents
                        )
                    if pitch_bend:
                        position = values[("yamaha", "pitch_bend")]
                        if isinstance(template, YamahaDX7Voice):
                            bend = (
                                position * template.performance.pitch_bend_range * 100
                            )
                        else:
                            cents = position * template.patch.data[64] * 100
                            tuning = cents if tuning is None else tuning + cents
                    if assignments:
                        controllers = (
                            _dx7_controller_row(template.performance, values)
                            if isinstance(template, YamahaDX7Voice)
                            else _tx81z_controller_row(template, values)
                        )
                        if isinstance(template, YamahaTX81ZVoice):
                            breath = values.get(("yamaha", "breath"))
                            foot = values.get(("yamaha", "foot"))
                if isinstance(voice.renderer, yamaha.DX7Renderer):
                    audio[i] = voice.renderer.render(
                        1, rows[voice.template][i : i + 1], tuning, bend, controllers
                    )[0]
                else:
                    ends = tick_ends[voice.template]
                    begin = int(ends[i - 1]) if i else 0
                    end = int(ends[i])
                    audio[i] = voice.renderer.render(
                        1,
                        (
                            ends[i : i + 1] - begin,
                            rows[voice.template][begin : end + 1],
                        ),
                        tuning,
                        controllers,
                        breath,
                        foot,
                        portamento_enabled,
                    )[0]
                count += 1
            gain = 10 ** (template.processing.volume_db / 20)
            if template.processing.invert_polarity:
                gain = -gain
            values = self.controls.values(template, voice.sources, start, count)
            _, amplitudes, parameters = synth.processing_parameters(
                template, values, self.definition.sample_rate, start, count
            )
            if template.processing.filter_order == FilterOrder.after_amplitude:
                audio[:count] *= (gain * amplitudes)[:, None]
            if count and template.processing.filters:
                audio[:count], voice.filter_states = filters.filter_samples(
                    template.processing.filters,
                    voice.filter_states,
                    audio[:count],
                    parameters,
                    self.definition.sample_rate,
                )
            if template.processing.filter_order == FilterOrder.before_amplitude:
                audio[:count] *= (gain * amplitudes)[:, None]
            if voice.fade_at is not None:
                audio *= (
                    voice.fade_gain
                    * np.maximum(
                        0,
                        1
                        - (start + np.arange(frames) - voice.fade_at)
                        / voice.fade_frames,
                    )
                )[:, None]
            routes = [
                [
                    sum(r.gain for r in template.channels if r.output == c)
                    for c in self.definition.channels
                ]
            ]
            if (group := self.output_groups.get(voice.template)) is not None:
                groups[group] += audio
            else:
                output += synth.route_samples(audio, routes)
            if count < frames or _complete(voice):
                self._retire(voice, start + count)
        for group in self.definition.score.body.tx81z_outputs:
            audio = yamaha.tx81z_dac(groups[group.name])
            if frames:
                ends = tick_ends[group.voices[0]]
                clocked = np.diff(ends, prepend=0) > 0
                if not np.all(clocked):
                    latest = np.maximum.accumulate(
                        np.where(clocked, np.arange(1, frames + 1), 0)
                    )
                    audio = np.concatenate([[[self.held_dac[group.name]]], audio])[
                        latest
                    ]
                self.held_dac[group.name] = float(audio[-1, 0])
            processing = group.processing
            gain = 10 ** (processing.volume_db / 20)
            if processing.invert_polarity:
                gain = -gain
            if processing.filter_order == FilterOrder.after_amplitude:
                audio *= gain
            if frames and processing.filters:
                audio, self.output_filters[group.name] = filters.filter_samples(
                    processing.filters,
                    self.output_filters[group.name],
                    audio,
                    filters.parameters(
                        processing.filters, self.definition.sample_rate, frames
                    ),
                    self.definition.sample_rate,
                )
            if processing.filter_order == FilterOrder.before_amplitude:
                audio *= gain
            routes = [
                [
                    sum(r.gain for r in group.channels if r.output == c)
                    for c in self.definition.channels
                ]
            ]
            output += synth.route_samples(audio, routes)
        return output

    def _retire(self, voice: Voice, frame: int) -> None:
        self.mono_notes = [
            n for n in self.mono_notes if n.action.voice_id != voice.voice_id
        ]
        if voice.template in self.pitch_history:
            previous = self.pitch_history[voice.template].get(voice.part)
            if previous is not None and previous.voice_id == voice.voice_id:
                assert isinstance(voice.renderer, yamaha.TX81ZRenderer)
                if voice.renderer.portamento is not None:
                    previous.pitch_hz = voice.renderer.portamento.current_hz
        if voice.oscillator_slot is not None:
            assert isinstance(voice.renderer, yamaha.DX7Renderer)
            self.oscillator_slots[voice.template][voice.oscillator_slot] = (
                OscillatorSlot(
                    phases=voice.renderer.phases.tolist(),
                    phase_steps=voice.renderer.phase_steps.tolist(),
                    idle_since=frame,
                )
            )
        del self.voices[voice.voice_id]
        self.controls.stop(voice.voice_id)


def prepare(score: SynthInstrumentScore) -> PreparedYamaha:
    score = SynthInstrumentScore.model_validate(score.model_dump())
    output = score.outputs[0].stream
    if not isinstance(output, AudioType):
        raise synth.EngineError("Yamaha output must be sampled audio")
    timebase = next(t for t in score.timebases if t.name == output.timebase)
    if timebase.rate.denominator != 1:
        raise synth.EngineError("Yamaha output rate must be an integer")
    for voice in score.body.voices:
        if not isinstance(voice, (YamahaDX7Voice, YamahaTX81ZVoice)):
            raise synth.EngineError(
                "Yamaha instruments require dedicated Yamaha templates"
            )
        if (
            not voice.mapping.pitch_tracking
            and voice.mapping.reference_pitch_hz is None
        ):
            raise synth.EngineError(
                "Untracked Yamaha pitch requires reference_pitch_hz"
            )
        if voice.envelfos or voice.event_connections:
            raise synth.EngineError("Yamaha external generators are not implemented")
        synth.validate_generators(
            voice, allow_velocity_bindings=True, allow_key_bindings=True
        )
        synth.validate_modulation(
            voice,
            tuple(
                ("yamaha", p)
                for p in (
                    "pitch_bend",
                    "modulation_wheel",
                    "breath",
                    "foot",
                    "portamento",
                )
            ),
        )
        if voice.processing != Processing(
            volume_db=voice.processing.volume_db,
            invert_polarity=voice.processing.invert_polarity,
            tuning_cents=voice.processing.tuning_cents,
            filters=voice.processing.filters,
            filter_order=voice.processing.filter_order,
        ):
            raise synth.EngineError("Unsupported Yamaha processing")
        if isinstance(voice, YamahaDX7Voice):
            yamaha_lfo.DX7LFOContext(voice.patch, timebase.rate.numerator)
        else:
            targets = {
                p.target.parameter
                for p in voice.modulation.parameters
                if p.target.name == "yamaha"
            }
            if "breath" in targets and voice.patch.data[75] != 50:
                raise synth.EngineError("TX81Z breath pitch bias is not implemented")
            yamaha_lfo.TX81ZLFOContext(voice.patch, timebase.rate.numerator)
        _note(voice, 69, 127, timebase.rate.numerator, None)
        filters.parameters(voice.processing.filters, timebase.rate.numerator, 1)
    for group in score.body.tx81z_outputs:
        filters.parameters(group.processing.filters, timebase.rate.numerator, 1)
    return PreparedYamaha(
        sample_rate=timebase.rate.numerator, channels=output.channels, score=score
    )


def _dx7_controller_row(
    performance: DX7Performance, values: dict[tuple[str, str], np.ndarray]
) -> np.ndarray:
    """Dexed's quantized range scaling and strongest assigned contribution."""
    row = np.zeros((1, 3))
    eg_assigned = False
    for name, assignment in (
        ("modulation_wheel", performance.modulation_wheel),
        ("breath", performance.breath),
        ("foot", performance.foot),
    ):
        position = values.get(("yamaha", name))
        cc = 0 if position is None else int(127 * position[0])
        amount = int(cc * (0.01 * assignment.range))
        for i, enabled in enumerate(
            (assignment.pitch, assignment.amplitude, assignment.eg_bias)
        ):
            if enabled:
                row[0, i] = max(row[0, i], amount)
        eg_assigned |= assignment.eg_bias
    row[0, :2] /= 128
    row[0, 2] = (127 - row[0, 2]) / 128 if eg_assigned else 0
    return row


def _tx81z_controller_row(
    template: YamahaTX81ZVoice, values: dict[tuple[str, str], np.ndarray]
) -> np.ndarray:
    """Sum firmware byte-depth contributions before patch addition/conversion."""
    row = np.zeros((1, 2), dtype=np.int64)
    data, extra = template.patch.data, template.additional.data
    for name, pitch, amplitude in (
        ("modulation_wheel", data[71], data[72]),
        ("breath", data[73], data[74]),
        ("foot", extra[21], extra[22]),
    ):
        if (position := values.get(("yamaha", name))) is not None:
            cc = floor(position[0] * 127)
            row[0, 0] += ((pitch * 165 >> 6) * cc * 2) >> 8
            row[0, 1] += ((amplitude * 165 >> 6) * cc * 2) >> 8
    return np.minimum(row, 255)


def _note(
    template: YamahaDX7Voice | YamahaTX81ZVoice,
    key: int,
    velocity: int,
    rate: int,
    pitch: float | None,
    phases: list[float] | None = None,
    start_pitch: float | None = None,
    *,
    start_frame: int = 0,
) -> yamaha.DX7Renderer | yamaha.TX81ZRenderer:
    if isinstance(template, YamahaDX7Voice):
        return yamaha.DX7Renderer(template.patch, key, velocity, rate, pitch, phases)
    return yamaha.TX81ZRenderer(
        template.patch,
        template.additional,
        key,
        velocity,
        rate,
        pitch,
        start_pitch,
        start_frame=start_frame,
    )


def _complete(voice: Voice) -> bool:
    if not voice.released:
        return False
    renderer = voice.renderer
    if isinstance(renderer, yamaha.DX7Renderer):
        return all(
            renderer.envelopes[i - 1].stage == 4
            and renderer.envelopes[i - 1].levels[3] == 0
            for i in renderer.voice.topology.carriers
        )
    return all(
        e.stage >= 3 and e.attenuation == 1023 for e in renderer.envelopes.envelopes
    )
