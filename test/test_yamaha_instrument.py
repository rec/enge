from pathlib import Path

import numpy as np
import pytest
from test_render import decode
from test_synth import check_audio, score
from test_yamaha import voice
from ufor import synth_trace, tx81z
from ufor.dx7 import DX7Voice
from ufor.events import Release, Trigger
from ufor.instrument_trace import RetirementCause, VoiceRetirement
from ufor.synth import SynthInstrumentScore, YamahaDX7Voice, YamahaTX81ZVoice

from enge import yamaha, yamaha_instrument, yamaha_lfo
from enge.midi import MidiPerformance
from enge.render import render_midi
from enge.synth import EngineError


def instrument(kind: str) -> SynthInstrumentScore:
    raw = score().model_dump(mode="json")
    template = {
        "name": "yamaha",
        "yamaha": kind,
        "mapping": {"lowest_key": 0, "highest_key": 127, "reference_pitch_hz": 440},
        "channels": [
            {"input": "mono", "output": "left", "gain": 0.3},
            {"input": "mono", "output": "right", "gain": 0.15},
        ],
    }
    if kind == "dx7":
        data = bytearray(voice().data)
        for i in range(0, 126, 21):
            data[i + 7] = 1
            data[i + 14] = 1
            data[i + 15] = 3
            data[i + 16] = 60
        data[136] = 1
        data[137:145] = bytes([75, 0, 50, 30, 1, 2, 4, 24])
        template["patch"] = bytes(data).hex()
    else:
        data = bytearray(93)
        for i in range(0, 52, 13):
            data[i : i + 5] = bytes([31, 0, 0, 0, 15])
            data[i + 8 : i + 13] = bytes([1, 3, 60, 4, 3])
        data[52] = 7
        data[54:63] = bytes([75, 0, 99, 99, 1, 2, 4, 2, 24])
        template["patch"] = bytes(data).hex()
        template["additional"] = bytes(23).hex()
    raw["body"] = {"voices": [template]}
    return SynthInstrumentScore.model_validate(raw)


@pytest.mark.parametrize("kind", ["dx7", "tx81z"])
def test_yamaha_instrument_shares_clock_and_restores_overlapping_microtuned_notes(
    tmp_path: Path, kind: str
) -> None:
    document = instrument(kind)
    definition = yamaha_instrument.prepare(document)
    actions = synth_trace.prepare(
        document.body,
        [
            Trigger(
                tick=0,
                ordinal=0,
                part="main",
                trigger_id="first",
                key=69,
                pitch_hz=442,
                velocity=0.25,
            ),
            Trigger(
                tick=1001,
                ordinal=0,
                part="main",
                trigger_id="second",
                key=69,
                pitch_hz=444,
                velocity=1e-9,
            ),
            Release(tick=2001, ordinal=0, part="main", trigger_id="first"),
            Release(tick=4001, ordinal=0, part="main", trigger_id="second"),
        ],
        seed=0,
    ).actions
    actions.extend(
        VoiceRetirement(
            tick=6001,
            ordinal=i,
            voice_id=f"voice-{i}",
            cause=RetirementCause.transport_stop,
            action="stop",
        )
        for i in range(2)
    )
    template = document.body.voices[0]
    if isinstance(template, YamahaDX7Voice):
        clock = yamaha_lfo.DX7LFOContext(template.patch)
        first = yamaha.DX7Renderer(template.patch, 69, 32, pitch_hz=442)
        second = yamaha.DX7Renderer(template.patch, 69, 1, pitch_hz=444)
    else:
        assert isinstance(template, YamahaTX81ZVoice)
        clock = yamaha_lfo.TX81ZLFOContext(template.patch)
        first = yamaha.TX81ZRenderer(
            template.patch, template.additional, 69, 32, pitch_hz=442
        )
        second = yamaha.TX81ZRenderer(
            template.patch, template.additional, 69, 1, pitch_hz=444
        )
    expected = np.zeros((48000, 2))
    clock.keydown()
    expected[:1001] = first.render(1001, clock.advance(1001)) * [0.3, 0.15]
    clock.keydown()
    rows = clock.advance(1000)
    expected[1001:2001] = (first.render(1000, rows) + second.render(1000, rows)) * [
        0.3,
        0.15,
    ]
    first.release()
    rows = clock.advance(2000)
    expected[2001:4001] = (first.render(2000, rows) + second.render(2000, rows)) * [
        0.3,
        0.15,
    ]
    second.release()
    rows = clock.advance(2000)
    expected[4001:6001] = (first.render(2000, rows) + second.render(2000, rows)) * [
        0.3,
        0.15,
    ]
    renderer = yamaha_instrument.OfflineYamaha(definition)
    actual = renderer.advance(actions, 0, 48000)
    check_audio(tmp_path / "yamaha-instrument.wav", actual, expected)
    assert renderer.snapshot().voices == []
    split = yamaha_instrument.OfflineYamaha(definition)
    head = split.advance([a for a in actions if a.tick < 3001], 0, 3001)
    saved = yamaha_instrument.YamahaSnapshot.model_validate_json(
        split.snapshot().model_dump_json()
    )
    split.restore(saved)
    tail = split.advance([a for a in actions if a.tick >= 3001], 3001, 48000)
    np.testing.assert_array_equal(np.concatenate([head, tail]), actual)


@pytest.mark.parametrize("kind", ["dx7", "tx81z"])
def test_microtuning_preserves_fixed_operators_and_snapshot_identity(
    tmp_path: Path, kind: str
) -> None:
    template = instrument(kind).body.voices[0]
    if isinstance(template, YamahaDX7Voice):
        raw = bytearray(template.patch.data)
        raw[139:141] = bytes(2)
        for i in range(0, 126, 21):
            raw[i + 17] = 1
        patch = DX7Voice(data=bytes(raw))
        plain = yamaha.DX7Renderer(patch, 69, 100)
        tuned = yamaha.DX7Renderer(patch, 69, 100, pitch_hz=445)
    else:
        assert isinstance(template, YamahaTX81ZVoice)
        raw = bytearray(template.patch.data)
        raw[56:58] = bytes(2)
        additional = bytearray(23)
        for i in range(0, 20, 5):
            additional[i] = 1
        patch = tx81z.TX81ZVoice(data=bytes(raw))
        extra = tx81z.TX81ZAdditional(data=bytes(additional))
        plain = yamaha.TX81ZRenderer(patch, extra, 69, 100)
        tuned = yamaha.TX81ZRenderer(patch, extra, 69, 100, pitch_hz=445)
    check_audio(tmp_path / "yamaha-fixed.wav", tuned.render(48000), plain.render(48000))
    with pytest.raises(EngineError, match="different note definition"):
        plain.restore(tuned.snapshot())


@pytest.mark.parametrize("kind", ["dx7", "tx81z"])
@pytest.mark.parametrize(
    "velocity, expected", [(0.0, 0), (1e-9, 1), (0.5, 64), (1.0, 127)]
)
def test_trigger_velocity_rounds_ties_up_and_preserves_positive_notes(
    tmp_path: Path, kind: str, velocity: float, expected: int
) -> None:
    document = instrument(kind)
    # TX81Z's nonzero velocity sensitivity still excludes velocity zero.
    template = document.body.voices[0]
    if isinstance(template, YamahaTX81ZVoice):
        raw = bytearray(template.patch.data)
        for i in range(0, 52, 13):
            raw[i + 9] = 0
        template = template.model_copy(
            update={"patch": tx81z.TX81ZVoice(data=bytes(raw))}
        )
        document = document.model_copy(
            update={"body": document.body.model_copy(update={"voices": [template]})}
        )
    actions = synth_trace.prepare(
        document.body,
        [
            Trigger(
                tick=0,
                ordinal=0,
                part="main",
                trigger_id="note",
                key=69,
                velocity=velocity,
            )
        ],
        seed=0,
    ).actions
    renderer = yamaha_instrument.OfflineYamaha(yamaha_instrument.prepare(document))
    actual = renderer.advance(actions, 0, 48000)
    if isinstance(template, YamahaDX7Voice):
        note = yamaha.DX7Renderer(template.patch, 69, expected, pitch_hz=440)
        clock = yamaha_lfo.DX7LFOContext(template.patch)
    else:
        assert isinstance(template, YamahaTX81ZVoice)
        note = yamaha.TX81ZRenderer(
            template.patch, template.additional, 69, expected, pitch_hz=440
        )
        clock = yamaha_lfo.TX81ZLFOContext(template.patch)
    clock.keydown()
    check_audio(
        tmp_path / "velocity.wav",
        actual,
        note.render(48000, clock.advance(48000)) * [0.3, 0.15],
    )
    assert renderer.snapshot().voices[0].note.velocity == expected


@pytest.mark.parametrize("kind", ["dx7", "tx81z"])
def test_minimum_hold_and_retimed_fade_survive_snapshot(
    tmp_path: Path, kind: str
) -> None:
    document = instrument(kind)
    raw = document.model_dump(mode="json")
    raw["body"]["voices"][0]["minimum_hold_seconds"] = "1/100"
    document = SynthInstrumentScore.model_validate(raw)
    events = [
        Trigger(tick=0, ordinal=0, part="main", trigger_id="note", key=69),
        Release(tick=100, ordinal=0, part="main", trigger_id="note"),
    ]
    actions = synth_trace.prepare(document.body, events, seed=0).actions
    definition = yamaha_instrument.prepare(document)
    baseline = yamaha_instrument.OfflineYamaha(definition).advance(actions, 0, 48000)
    delayed = synth_trace.prepare(
        document.body, [events[0], events[1].model_copy(update={"tick": 480})], seed=0
    ).actions
    expected = yamaha_instrument.OfflineYamaha(definition).advance(delayed, 0, 48000)
    check_audio(tmp_path / "minimum-hold.wav", baseline, expected)
    for tick in (1000, 1240):
        actions.append(
            VoiceRetirement(
                tick=tick,
                ordinal=0,
                voice_id="voice-0",
                cause=RetirementCause.transport_stop,
                action="fade",
                fade_seconds=0.01,
            )
        )
    gains = np.ones(48000)
    gains[1000:1240] = 1 - np.arange(240) / 480
    gains[1240:1720] = 0.5 * (1 - np.arange(480) / 480)
    gains[1720:] = 0
    renderer = yamaha_instrument.OfflineYamaha(definition)
    head = renderer.advance([a for a in actions if a.tick < 1200], 0, 1200)
    renderer.restore(
        yamaha_instrument.YamahaSnapshot.model_validate_json(
            renderer.snapshot().model_dump_json()
        )
    )
    tail = renderer.advance([a for a in actions if a.tick >= 1200], 1200, 48000)
    check_audio(
        tmp_path / "retimed-fade.wav",
        np.concatenate([head, tail]),
        baseline * gains[:, None],
    )
    assert renderer.snapshot().voices == []


@pytest.mark.parametrize(
    "kind,synchronize", [("dx7", False), ("dx7", True), ("tx81z", True)]
)
def test_midi_render_dispatches_yamaha_instruments(
    tmp_path: Path, kind: str, synchronize: bool
) -> None:
    document = instrument(kind)
    raw = document.model_dump(mode="json")
    patch = bytearray.fromhex(raw["body"]["voices"][0]["patch"])
    if kind == "dx7":
        patch[136] = int(synchronize)
        for i in range(0, 126, 21):
            patch[i + 7] = 0
    else:
        for i in range(0, 52, 13):
            patch[i + 3] = 15
    raw["body"]["voices"][0]["patch"] = bytes(patch).hex()
    document = SynthInstrumentScore.model_validate(raw)
    events = [
        Trigger(tick=0, ordinal=0, part="main", trigger_id="note", key=69),
        Release(tick=1000, ordinal=0, part="main", trigger_id="note"),
    ]
    performance = MidiPerformance(
        sample_rate=48000, frames=36000, parts={"main": events}
    )
    path = tmp_path / "yamaha.flac"
    report = render_midi(performance, {"main": (document, {})}, path, block_size=1001)
    actions = synth_trace.prepare(document.body, events, seed=0).actions
    expected = yamaha_instrument.OfflineYamaha(
        yamaha_instrument.prepare(document)
    ).advance(actions, 0, report.frames)
    assert report.frames == 48000
    assert report.notes == 1
    check_audio(
        tmp_path / "dispatch.wav",
        decode(path),
        np.rint(expected * (2**23 - 1)) / (2**23 - 1),
    )


@pytest.mark.parametrize("retirement", ["stop", "fade"])
def test_dx7_reuses_oldest_idle_slots_across_overlaps_and_snapshots(
    tmp_path: Path, retirement: str
) -> None:
    raw = instrument("dx7").model_dump(mode="json")
    data = bytearray.fromhex(raw["body"]["voices"][0]["patch"])
    data[136] = 0
    data[139:141] = bytes(2)
    raw["body"]["voices"][0]["patch"] = bytes(data).hex()
    document = SynthInstrumentScore.model_validate(raw)
    patch = DX7Voice(data=bytes(data))
    events = [
        Trigger(tick=t, ordinal=0, part="main", trigger_id=f"note-{i}", key=k)
        for i, (t, k) in enumerate([(0, 69), (100, 72), (500, 65), (600, 67)])
    ]
    actions = synth_trace.prepare(document.body, events, seed=0).actions
    for i, end in enumerate([200, 300, 1000, 1100]):
        actions.append(
            VoiceRetirement(
                tick=end if retirement == "stop" else end - 100,
                ordinal=1,
                voice_id=f"voice-{i}",
                cause=RetirementCause.transport_stop,
                action=retirement,
                fade_seconds=None if retirement == "stop" else 100 / 48000,
            )
        )
    actions.sort(key=lambda a: (a.tick, a.ordinal))
    expected = np.zeros((48000, 2))
    first = yamaha.DX7Renderer(patch, 69, 127)
    second = yamaha.DX7Renderer(patch, 72, 127)
    first_audio, second_audio = first.render(200), second.render(200)
    third = yamaha.DX7Renderer(
        patch,
        65,
        127,
        initial_phases=((first.phases + 300 * first.phase_steps) % 1).tolist(),
    )
    fourth = yamaha.DX7Renderer(
        patch,
        67,
        127,
        initial_phases=((second.phases + 300 * second.phase_steps) % 1).tolist(),
    )
    for start, audio in [
        (0, first_audio),
        (100, second_audio),
        (500, third.render(500)),
        (600, fourth.render(500)),
    ]:
        if retirement == "fade":
            audio[-100:] *= (1 - np.arange(100) / 100)[:, None]
        expected[start : start + len(audio)] += audio * [0.3, 0.15]
    definition = yamaha_instrument.prepare(document)
    renderer = yamaha_instrument.OfflineYamaha(definition)
    chunks = []
    for start, end in [(0, 400), (400, 550), (550, 48000)]:
        chunks.append(
            renderer.advance([a for a in actions if start <= a.tick < end], start, end)
        )
        snapshot = yamaha_instrument.YamahaSnapshot.model_validate_json(
            renderer.snapshot().model_dump_json()
        )
        renderer = yamaha_instrument.OfflineYamaha(definition)
        renderer.restore(snapshot)
    actual = np.concatenate(chunks)
    check_audio(tmp_path / "oscillator-slots.wav", actual, expected)
    whole = yamaha_instrument.OfflineYamaha(definition).advance(actions, 0, 48000)
    np.testing.assert_array_equal(actual, whole)
    saved = renderer.snapshot()
    assert saved.voices == []
    assert len(saved.oscillator_slots["yamaha"]) == 2
    assert [s.idle_since for s in saved.oscillator_slots["yamaha"]] == [1000, 1100]
