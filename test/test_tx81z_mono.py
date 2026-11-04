from pathlib import Path

import numpy as np
import pytest
from test_synth import check_audio
from test_tx81z_lfo import lfo_voice
from test_yamaha_instrument import instrument
from ufor import synth_trace, tx81z
from ufor.events import ControlChange, Release, Trigger
from ufor.instrument_trace import RetirementCause, VoiceRetirement
from ufor.synth import SynthInstrumentScore, YamahaTX81ZVoice

from enge import yamaha, yamaha_instrument
from enge.synth import EngineError


def mono_score(fingered: bool = False, time: int = 0) -> SynthInstrumentScore:
    raw = instrument("tx81z").model_dump(mode="json")
    template = raw["body"]["voices"][0]
    patch = bytearray(lfo_voice().data)
    patch[56:58] = bytes(2)
    patch[63] = 1
    patch[65] = int(fingered)
    patch[66] = time
    template["patch"] = bytes(patch).hex()
    raw["body"]["controls"] = {"pedal": {"default": 0}, "glide": {"default": 0}}
    raw["body"]["sustain"] = {"control": "pedal"}
    return SynthInstrumentScore.model_validate(raw)


def test_legato_retarget_keeps_envelope_progress_and_validates_before_mutating(
    tmp_path: Path,
) -> None:
    template = mono_score().body.voices[0]
    assert isinstance(template, YamahaTX81ZVoice)
    note = yamaha.TX81ZRenderer(template.patch, template.additional, 69, 127)
    first = note.render(1001)
    saved = note.snapshot()
    with pytest.raises(EngineError, match="pitch_hz"):
        note.retarget(81, 127, float("nan"))
    assert note.snapshot() == saved
    note.retarget(81, 127, 880)
    changed = note.snapshot()
    assert changed.operators == saved.operators
    assert changed.envelopes == saved.envelopes
    tail = note.render(46999)
    frequencies = np.concatenate([np.full(1001, 440), np.full(46999, 880)])
    phase = np.concatenate([[0], np.cumsum(frequencies[:-1])]) / 48000
    check_audio(
        tmp_path / "legato-phase.wav",
        np.concatenate([first, tail]),
        np.sin(2 * np.pi * phase)[:, None],
    )
    restored = yamaha.TX81ZRenderer(
        template.patch, template.additional, 81, 127, pitch_hz=880, start_pitch_hz=440
    )
    restored.restore(
        yamaha.TX81ZSnapshot.model_validate_json(changed.model_dump_json())
    )
    check_audio(
        tmp_path / "legato-note-replay.wav",
        np.concatenate([first, restored.render(46999)]),
        np.concatenate([first, tail]),
    )


def test_legato_keeps_a_developing_envelope_when_microtuning_holds_pitch(
    tmp_path: Path,
) -> None:
    template = mono_score().body.voices[0]
    assert isinstance(template, YamahaTX81ZVoice)
    raw = bytearray(template.patch.data)
    for i in range(0, 52, 13):
        raw[i : i + 5] = bytes([18, 14, 8, 15, 7])
    patch = tx81z.TX81ZVoice(data=bytes(raw))
    note = yamaha.TX81ZRenderer(patch, template.additional, 69, 127, pitch_hz=440)
    first = note.render(7001)
    note.retarget(81, 127, 440)
    tail = note.render(40999)
    reference = yamaha.TX81ZRenderer(patch, template.additional, 69, 127, pitch_hz=440)
    check_audio(
        tmp_path / "legato-envelope.wav",
        np.concatenate([first, tail]),
        reference.render(48000),
    )


@pytest.mark.parametrize("mono", [False, True])
def test_explicit_zero_velocity_triggers_render_and_restore_with_velocity_sensitivity(
    tmp_path: Path, mono: bool
) -> None:
    raw = mono_score().model_dump(mode="json")
    template = raw["body"]["voices"][0]
    patch = bytearray.fromhex(template["patch"])
    patch[63] = int(mono)
    for i in range(4):
        patch[13 * i + 9] = i + 1
    template["patch"] = bytes(patch).hex()
    document = SynthInstrumentScore.model_validate(raw)
    actions = synth_trace.prepare(
        document.body,
        [
            Trigger(tick=0, ordinal=0, part="main", trigger_id="a", key=69, velocity=0),
            Trigger(
                tick=12001, ordinal=0, part="main", trigger_id="b", key=81, velocity=0
            ),
        ],
        seed=0,
    ).actions
    engine = yamaha_instrument.OfflineYamaha(yamaha_instrument.prepare(document))
    first = engine.advance(actions, 0, 17003)
    saved = engine.snapshot()
    assert all(v.note.velocity == 0 for v in saved.voices)
    assert len(saved.voices) == (1 if mono else 2)
    tail = engine.advance([], 17003, 48000)
    engine.restore(
        yamaha_instrument.YamahaSnapshot.model_validate_json(saved.model_dump_json())
    )
    check_audio(
        tmp_path / "zero-velocity-replay.wav",
        np.concatenate([first, tail]),
        np.concatenate([first, engine.advance([], 17003, 48000)]),
    )
    # Algorithm 8 sums four identical sine carriers, each with TL 16 plus KVS.
    gain = sum(2 ** (-(16 + a) / 8) for a in [22, 37, 52, 67])
    if mono:
        frequencies = np.concatenate([np.full(12001, 440), np.full(35999, 880)])
        phase = np.concatenate([[0], np.cumsum(frequencies[:-1])]) / 48000
        expected = np.sin(2 * np.pi * phase)
    else:
        expected = np.sin(2 * np.pi * 440 * np.arange(48000) / 48000)
        expected[12001:] += np.sin(2 * np.pi * 880 * np.arange(35999) / 48000)
    check_audio(
        tmp_path / "zero-velocity.wav",
        np.concatenate([first, tail]),
        expected[:, None] * gain * [[0.3, 0.15]],
    )


@pytest.mark.parametrize("sustain", [False, True])
def test_mono_last_held_key_returns_without_retriggering_under_sustain(
    tmp_path: Path, sustain: bool
) -> None:
    document = mono_score()
    events = [
        Trigger(tick=0, ordinal=0, part="main", trigger_id="a", key=69),
        Trigger(tick=1001, ordinal=0, part="main", trigger_id="b", key=81),
        Release(tick=2002, ordinal=0, part="main", trigger_id="b"),
        Release(tick=12000, ordinal=0, part="main", trigger_id="a"),
    ]
    if sustain:
        events.insert(
            0,
            ControlChange(
                tick=0, ordinal=0, part="main", scope="part", control="pedal", value=1
            ),
        )
        events[1] = events[1].model_copy(update={"ordinal": 1})
        events.append(
            ControlChange(
                tick=24000,
                ordinal=0,
                part="main",
                scope="part",
                control="pedal",
                value=0,
            )
        )
    actions = synth_trace.prepare(document.body, events, seed=0).actions
    engine = yamaha_instrument.OfflineYamaha(yamaha_instrument.prepare(document))
    first = engine.advance([a for a in actions if a.tick < 1500], 0, 1500)
    saved = engine.snapshot()
    assert len(saved.voices) == 1
    assert saved.voices[0].note.note == 81
    assert len(saved.mono_notes) == 2
    middle = engine.advance([a for a in actions if 1500 <= a.tick < 12001], 1500, 12001)
    assert len(engine.voices) == 1
    assert next(iter(engine.voices.values())).renderer.note == 69
    assert not engine.mono_notes
    assert next(iter(engine.voices.values())).released is not sustain
    tail = engine.advance([a for a in actions if a.tick >= 12001], 12001, 48000)
    actual = np.concatenate([first, middle, tail])
    engine.restore(
        yamaha_instrument.YamahaSnapshot.model_validate_json(saved.model_dump_json())
    )
    replay = engine.advance([a for a in actions if a.tick >= 1500], 1500, 48000)
    check_audio(tmp_path / "mono-snapshot.wav", np.concatenate([first, replay]), actual)
    template = document.body.voices[0]
    assert isinstance(template, YamahaTX81ZVoice)
    reference = yamaha.TX81ZRenderer(template.patch, template.additional, 69, 127)
    pieces = [reference.render(1001)]
    reference.retarget(81, 127)
    pieces.append(reference.render(1001))
    reference.retarget(69, 127)
    release_at = 24000 if sustain else 12000
    pieces.append(reference.render(release_at - 2002))
    reference.release()
    released = np.zeros((48000 - release_at, 1))
    for i in range(len(released)):
        if all(
            e.stage >= 3 and e.attenuation == 1023
            for e in reference.envelopes.envelopes
        ):
            break
        released[i] = reference.render(1)[0]
    pieces.append(released)
    check_audio(
        tmp_path / "mono-priority.wav", actual, np.concatenate(pieces) * [[0.3, 0.15]]
    )


@pytest.mark.parametrize("fingered", [False, True])
@pytest.mark.parametrize("switch", [0, 1])
def test_fingered_glides_only_between_held_keys_and_ignores_switch(
    tmp_path: Path, fingered: bool, switch: int
) -> None:
    raw = mono_score(fingered, 18).model_dump(mode="json")
    raw["body"]["controls"]["pedal"]["default"] = 1
    raw["body"]["controls"]["glide"]["default"] = switch
    template = raw["body"]["voices"][0]
    template["bindings"] = [{"name": "switch", "kind": "control", "control": "glide"}]
    template["modulation"] = {
        "sources": [{"name": "switch", "scope": "part", "minimum": 0, "maximum": 1}],
        "parameters": [
            {
                "target": {"name": "yamaha", "parameter": "portamento"},
                "unit": "ratio",
                "scope": "voice",
                "minimum": 0,
                "maximum": 1,
                "default": 1,
            }
        ],
        "routes": [
            {
                "name": "switch",
                "source": "switch",
                "target": {"name": "yamaha", "parameter": "portamento"},
                "unit": "ratio",
                "operation": "multiply",
                "points": [{"input": 0, "amount": 0}, {"input": 1, "amount": 1}],
            }
        ],
    }
    document = SynthInstrumentScore.model_validate(raw)
    events = [
        Trigger(tick=0, ordinal=0, part="main", trigger_id="a", key=69),
        Trigger(tick=1001, ordinal=0, part="main", trigger_id="b", key=81),
        Release(tick=2002, ordinal=0, part="main", trigger_id="a"),
        Release(tick=3003, ordinal=0, part="main", trigger_id="b"),
        Trigger(tick=4004, ordinal=0, part="main", trigger_id="c", key=57),
    ]
    actions = synth_trace.prepare(document.body, events, seed=0).actions
    engine = yamaha_instrument.OfflineYamaha(yamaha_instrument.prepare(document))
    first = engine.advance([a for a in actions if a.tick < 2002], 0, 2002)
    saved = engine.snapshot()
    glide = saved.voices[0].note.portamento
    assert glide is not None
    assert (glide.current_hz < 880) is (fingered or bool(switch))
    tail = engine.advance([a for a in actions if a.tick >= 2002], 2002, 48000)
    # Sustain keeps the old voice sounding, but released keys are not legato.
    np.testing.assert_array_equal(tail[4004 - 2002], [0, 0])
    assert len(engine.voices) == 1
    current = next(iter(engine.voices.values())).renderer
    assert isinstance(current, yamaha.TX81ZRenderer)
    if fingered:
        assert current.portamento is None
    else:
        assert current.portamento is not None
        assert current.portamento.current_hz == 220
    engine.restore(saved)
    replay = engine.advance([a for a in actions if a.tick >= 2002], 2002, 48000)
    check_audio(
        tmp_path / "fingered-switch.wav",
        np.concatenate([first, tail]),
        np.concatenate([first, replay]),
    )


def test_mono_parts_keep_separate_priority_and_remember_eight_previous_keys(
    tmp_path: Path,
) -> None:
    document = mono_score()
    events = [
        Trigger(tick=i * 100, ordinal=0, part="main", trigger_id=f"n{i}", key=60 + i)
        for i in range(10)
    ]
    events.append(Trigger(tick=1000, ordinal=0, part="other", trigger_id="n9", key=81))
    events.extend(
        Release(tick=1100 + (9 - i) * 100, ordinal=0, part="main", trigger_id=f"n{i}")
        for i in range(9, 0, -1)
    )
    actions = synth_trace.prepare(document.body, events, seed=0).actions
    definition = yamaha_instrument.prepare(document)
    engine = yamaha_instrument.OfflineYamaha(definition)
    first = engine.advance([a for a in actions if a.tick < 1100], 0, 1100)
    assert len(engine.voices) == 2
    assert len(engine.mono_notes) == 10
    assert not any(n.action.trigger_id == "n0" for n in engine.mono_notes)
    tail = engine.advance([a for a in actions if a.tick >= 1100], 1100, 48000)
    assert all(n.action.part == "other" for n in engine.mono_notes)
    assert not any(v.renderer.note == 60 for v in engine.voices.values())
    expected = yamaha_instrument.OfflineYamaha(definition).advance(actions, 0, 48000)
    check_audio(
        tmp_path / "mono-memory-parts.wav", np.concatenate([first, tail]), expected
    )


@pytest.mark.parametrize("retirement", ["stop", "fade", "release"])
def test_mono_explicit_retirement_does_not_revive_an_older_held_key(
    tmp_path: Path, retirement: str
) -> None:
    document = mono_score()
    actions = synth_trace.prepare(
        document.body,
        [
            Trigger(tick=0, ordinal=0, part="main", trigger_id="a", key=69),
            Trigger(tick=1001, ordinal=0, part="main", trigger_id="b", key=81),
            Release(tick=2002, ordinal=0, part="main", trigger_id="b"),
        ],
        seed=0,
    ).actions
    second = next(
        a for a in actions if isinstance(a, synth_trace.VoiceStart) and a.key == 81
    )
    actions.append(
        VoiceRetirement(
            tick=1500,
            ordinal=0,
            voice_id=second.voice_id,
            cause=RetirementCause.choke,
            action=retirement,
            fade_seconds=0.1 if retirement == "fade" else None,
        )
    )
    actions.sort(key=lambda a: (a.tick, a.ordinal))
    definition = yamaha_instrument.prepare(document)
    engine = yamaha_instrument.OfflineYamaha(definition)
    first = engine.advance([a for a in actions if a.tick < 2003], 0, 2003)
    assert all(v.renderer.note != 69 for v in engine.voices.values())
    saved = engine.snapshot()
    tail = engine.advance([], 2003, 48000)
    engine.restore(saved)
    check_audio(
        tmp_path / "mono-choke.wav",
        np.concatenate([first, tail]),
        np.concatenate([first, engine.advance([], 2003, 48000)]),
    )
