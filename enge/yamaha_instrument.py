"""Offline Yamaha instruments consuming uFor's shared lifecycle actions."""

from math import ceil, floor

import numpy as np
from pydantic import BaseModel, ConfigDict, Field
from ufor import instrument_trace, synth_trace
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
    start: int
    oscillator_slot: int | None = None
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
    phases: list[float] = Field(min_length=6, max_length=6)
    phase_steps: list[float] = Field(min_length=6, max_length=6)
    idle_since: int | None = None


class YamahaSnapshot(BaseModel, frozen=True):
    definition: PreparedYamaha
    frame: int
    voices: list[VoiceSnapshot]
    clocks: dict[str, yamaha_lfo.DX7LFOSnapshot | yamaha_lfo.TX81ZLFOSnapshot]
    oscillator_slots: dict[str, list[OscillatorSlot]]


class OfflineYamaha:
    """One shared hardware-model LFO clock per authored voice template.

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
        self.clocks = {
            n: yamaha_lfo.DX7LFOContext(v.patch, definition.sample_rate)
            if isinstance(v, YamahaDX7Voice)
            else yamaha_lfo.TX81ZLFOContext(v.patch, definition.sample_rate)
            for n, v in self.templates.items()
        }
        self.voices: dict[str, Voice] = {}
        self.oscillator_slots: dict[str, list[OscillatorSlot]] = {
            n: []
            for n, v in self.templates.items()
            if isinstance(v, YamahaDX7Voice)
            and not yamaha.dx7_global_parameters(v.patch)[0]
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
            oscillator_slots=self.oscillator_slots,
        ).model_copy(deep=True)

    def restore(self, snapshot: YamahaSnapshot) -> None:
        if (
            snapshot.definition != self.definition
            or snapshot.clocks.keys() != self.clocks.keys()
            or snapshot.oscillator_slots.keys() != self.oscillator_slots.keys()
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
            )
            if isinstance(renderer, yamaha.DX7Renderer) and isinstance(
                saved.note, yamaha.DX7Snapshot
            ):
                renderer.restore(saved.note)
            elif isinstance(renderer, yamaha.TX81ZRenderer) and isinstance(
                saved.note, yamaha.TX81ZSnapshot
            ):
                renderer.restore(saved.note)
            else:
                raise synth.EngineError(
                    "Yamaha snapshot note type does not match template"
                )
            voices[saved.voice_id] = Voice(
                **saved.model_dump(exclude={"note"}), renderer=renderer
            )
        for name, clock in self.clocks.items():
            saved_clock = snapshot.clocks[name]
            if isinstance(clock, yamaha_lfo.DX7LFOContext) and isinstance(
                saved_clock, yamaha_lfo.DX7LFOSnapshot
            ):
                clock.restore(saved_clock)
            elif isinstance(clock, yamaha_lfo.TX81ZLFOContext) and isinstance(
                saved_clock, yamaha_lfo.TX81ZLFOSnapshot
            ):
                clock.restore(saved_clock)
            else:
                raise synth.EngineError(
                    "Yamaha snapshot clock type does not match template"
                )
        self.voices = voices
        self.oscillator_slots = snapshot.oscillator_slots
        self.frame = snapshot.frame

    def _apply(self, action: instrument_trace.TraceAction) -> None:
        if isinstance(
            action, (instrument_trace.TriggerContext, instrument_trace.Diagnostic)
        ):
            return
        if isinstance(action, instrument_trace.ControlObservation):
            sustain = self.definition.score.body.sustain
            if sustain is None or action.control != sustain.control:
                raise synth.EngineError(
                    "Yamaha external controller modulation is not implemented"
                )
            return  # Sustain retirement has already been resolved by uFor.
        if isinstance(action, instrument_trace.VoiceRetirement):
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
            or action.settings != template
            or action.channels != template.channels
            or action.key is None
            or action.velocity is None
        ):
            raise synth.EngineError(
                "Yamaha start must match its prepared template and carry key/velocity"
            )
        if action.voice_id in self.voices:
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
                    (p + elapsed * s) % 1
                    for p, s in zip(slot.phases, slot.phase_steps, strict=True)
                ]
            else:
                slot_index = len(slots)
        renderer = _note(
            template, action.key, velocity, self.definition.sample_rate, pitch, phases
        )
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
        self.clocks[template.name].keydown()
        self.voices[action.voice_id] = Voice(
            voice_id=action.voice_id,
            template=template.name,
            start=action.tick,
            oscillator_slot=slot_index,
            renderer=renderer,
            filter_states=filters.initial_states(template.processing.filters, 1),
        )

    def _render(self, start: int, frames: int) -> np.ndarray:
        output = np.zeros((frames, len(self.definition.channels)))
        rows = {n: c.advance(frames) for n, c in self.clocks.items()}
        for voice in list(self.voices.values()):
            template = self.templates[voice.template]
            audio = np.zeros((frames, 1))
            count = 0
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
                audio[i] = voice.renderer.render(1, rows[voice.template][i : i + 1])[0]
                count += 1
            gain = 10 ** (template.processing.volume_db / 20)
            if template.processing.invert_polarity:
                gain = -gain
            if template.processing.filter_order == FilterOrder.after_amplitude:
                audio *= gain
            if count and template.processing.filters:
                parameters = filters.parameters(
                    template.processing.filters, self.definition.sample_rate, count
                )
                audio[:count], voice.filter_states = filters.filter_samples(
                    template.processing.filters,
                    voice.filter_states,
                    audio[:count],
                    parameters,
                    self.definition.sample_rate,
                )
            if template.processing.filter_order == FilterOrder.before_amplitude:
                audio *= gain
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
            output += synth.route_samples(audio, routes)
            if count < frames or _complete(voice):
                self._retire(voice, start + count)
        return output

    def _retire(self, voice: Voice, frame: int) -> None:
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


def prepare(score: SynthInstrumentScore) -> PreparedYamaha:
    score = SynthInstrumentScore.model_validate(score.model_dump())
    output = score.outputs[0].stream
    if not isinstance(output, AudioType):
        raise synth.EngineError("Yamaha output must be sampled audio")
    timebase = next(t for t in score.timebases if t.name == output.timebase)
    if timebase.rate.denominator != 1:
        raise synth.EngineError("Yamaha output rate must be an integer")
    sustain = score.body.sustain
    if any(n != (sustain.control if sustain else None) for n in score.body.controls):
        raise synth.EngineError(
            "Yamaha external controller modulation is not implemented"
        )
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
        if (
            voice.envelfos
            or voice.bindings
            or voice.modulation.parameters
            or voice.event_connections
        ):
            raise synth.EngineError("Yamaha external modulation is not implemented")
        if voice.processing != Processing(
            volume_db=voice.processing.volume_db,
            invert_polarity=voice.processing.invert_polarity,
            tuning_cents=voice.processing.tuning_cents,
            filters=voice.processing.filters,
            filter_order=voice.processing.filter_order,
        ):
            raise synth.EngineError("Unsupported Yamaha processing")
        if isinstance(voice, YamahaDX7Voice):
            parameters = yamaha.dx7_global_parameters(voice.patch)
            if parameters[8] != 24:
                raise synth.EngineError("DX7 integration requires neutral transpose")
            if voice.patch.algorithm in (4, 6) and voice.patch.feedback:
                raise synth.EngineError(
                    "DX7 multi-operator feedback is not implemented"
                )
            yamaha_lfo.DX7LFOContext(voice.patch, timebase.rate.numerator)
        else:
            yamaha_lfo.TX81ZLFOContext(voice.patch, timebase.rate.numerator)
        _note(voice, 69, 127, timebase.rate.numerator, None)
        filters.parameters(voice.processing.filters, timebase.rate.numerator, 1)
    return PreparedYamaha(
        sample_rate=timebase.rate.numerator, channels=output.channels, score=score
    )


def _note(
    template: YamahaDX7Voice | YamahaTX81ZVoice,
    key: int,
    velocity: int,
    rate: int,
    pitch: float | None,
    phases: list[float] | None = None,
) -> yamaha.DX7Renderer | yamaha.TX81ZRenderer:
    if isinstance(template, YamahaDX7Voice):
        return yamaha.DX7Renderer(template.patch, key, velocity, rate, pitch, phases)
    return yamaha.TX81ZRenderer(
        template.patch, template.additional, key, velocity, rate, pitch
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
