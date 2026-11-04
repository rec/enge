from pathlib import Path

import numpy as np
import pytest
from test_render import decode
from test_synth import check_audio, score
from test_yamaha import voice
from ufor import synth_trace, tx81z
from ufor.dx7 import DX7Voice
from ufor.events import ControlChange, Release, Trigger
from ufor.instrument_trace import RetirementCause, VoiceRetirement
from ufor.synth import SynthInstrumentScore, YamahaDX7Voice, YamahaTX81ZVoice

from enge import filters, yamaha, yamaha_instrument, yamaha_lfo
from enge.midi import MidiPerformance
from enge.render import render_midi
from enge.synth import EngineError


def instrument(kind: str, transpose: int = 0) -> SynthInstrumentScore:
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
        data[144] += transpose
        template["patch"] = bytes(data).hex()
    else:
        data = bytearray(93)
        for i in range(0, 52, 13):
            data[i : i + 5] = bytes([31, 0, 0, 0, 15])
            data[i + 8 : i + 13] = bytes([1, 3, 60, 4, 3])
        data[52] = 7
        data[54:63] = bytes([75, 0, 99, 99, 1, 2, 4, 2, 24])
        data[62] += transpose
        template["patch"] = bytes(data).hex()
        template["additional"] = bytes(23).hex()
    raw["body"] = {"voices": [template]}
    return SynthInstrumentScore.model_validate(raw)


@pytest.mark.parametrize("kind", ["dx7", "tx81z"])
@pytest.mark.parametrize("transpose", [0, 12])
def test_yamaha_instrument_shares_clock_and_restores_overlapping_microtuned_notes(
    tmp_path: Path, kind: str, transpose: int
) -> None:
    document = instrument(kind, transpose)
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
    "kind,synchronize,algorithm",
    [
        ("dx7", False, 32),
        ("dx7", True, 32),
        ("tx81z", True, 0),
        ("dx7", True, 4),
        ("dx7", True, 6),
    ],
)
def test_midi_render_dispatches_yamaha_instruments(
    tmp_path: Path, kind: str, synchronize: bool, algorithm: int
) -> None:
    document = instrument(kind)
    raw = document.model_dump(mode="json")
    patch = bytearray.fromhex(raw["body"]["voices"][0]["patch"])
    if kind == "dx7":
        patch[134] = algorithm - 1
        patch[135] = 7 if algorithm in (4, 6) else 0
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


@pytest.mark.parametrize("kind", ["dx7", "tx81z"])
@pytest.mark.parametrize("scope", ["instrument", "part", "trigger"])
@pytest.mark.parametrize("order", ["before_amplitude", "after_amplitude"])
def test_scoped_yamaha_controls_smooth_amplitude_and_filters_across_restore(
    tmp_path: Path, kind: str, scope: str, order: str
) -> None:
    raw = instrument(kind).model_dump(mode="json")
    raw["body"]["controls"] = {"expression": {"default": 0.0}}
    voice = raw["body"]["voices"][0]
    patch = bytearray.fromhex(voice["patch"])
    offset = 139 if kind == "dx7" else 56
    patch[offset : offset + 2] = bytes(2)
    voice["patch"] = bytes(patch).hex()
    voice["processing"] = {
        "filter_order": order,
        "filters": [
            {"name": "tone", "response": "lowpass", "cutoff_hz": 500, "q": 0.5}
        ],
    }
    voice["bindings"] = [
        {
            "name": "expression",
            "kind": "control",
            "control": "expression",
            "smoothing": "1/200",
        }
    ]
    parameters, routes = [], []
    for name, target_name, parameter, unit, default, amount in [
        ("gain", "processing", "amplitude", "ratio", 1, -0.8),
        ("cutoff", "filter-tone", "cutoff_hz", "hz", 500, 1500),
        ("resonance", "filter-tone", "q", "ratio", 0.5, 1),
    ]:
        target = {"name": target_name, "parameter": parameter}
        parameters.append(
            {
                "target": target,
                "unit": unit,
                "scope": "voice",
                "minimum": min(default, default + amount),
                "maximum": max(default, default + amount),
                "default": default,
            }
        )
        routes.append(
            {
                "name": name,
                "source": "expression",
                "target": target,
                "unit": unit,
                "operation": "add",
                "points": [{"input": 0, "amount": 0}, {"input": 1, "amount": amount}],
            }
        )
    voice["modulation"] = {
        "sources": [{"name": "expression", "scope": scope, "minimum": 0, "maximum": 1}],
        "parameters": parameters,
        "routes": routes,
    }
    document = SynthInstrumentScore.model_validate(raw)
    events = [
        Trigger(tick=0, ordinal=0, part="main", trigger_id="first", key=69),
        Trigger(tick=1001, ordinal=0, part="other", trigger_id="second", key=69),
    ]
    changes = [
        ControlChange(
            tick=t,
            ordinal=0,
            control="expression",
            value=v,
            scope=scope,
            part=None if scope == "instrument" else "main",
            trigger_id="first" if scope == "trigger" else None,
        )
        for t, v in [(2000, 1.0), (2120, 0.4)]
    ]
    actions = synth_trace.prepare(document.body, [*events, *changes], seed=0).actions
    template = document.body.voices[0]
    ramp = np.zeros(48000)
    ramp[2000:2120] = np.arange(120) / 240
    ramp[2120:2360] = 0.5 - 0.1 * np.arange(240) / 240
    ramp[2360:] = 0.4
    expected = np.zeros((48000, 2))
    for i, start in enumerate([0, 1001]):
        if isinstance(template, YamahaDX7Voice):
            note = yamaha.DX7Renderer(template.patch, 69, 127)
        else:
            assert isinstance(template, YamahaTX81ZVoice)
            note = yamaha.TX81ZRenderer(template.patch, template.additional, 69, 127)
        audio = note.render(48000 - start)
        signal = (
            ramp[start:] if i == 0 or scope == "instrument" else np.zeros(48000 - start)
        )
        amplitude = 1 - 0.8 * signal
        values = np.column_stack([500 + 1500 * signal, 0.5 + signal])[:, None, :]
        if order == "after_amplitude":
            audio *= amplitude[:, None]
        audio, _ = filters.filter_samples(
            template.processing.filters,
            filters.initial_states(template.processing.filters, 1),
            audio,
            filters.parameters(template.processing.filters, 48000, len(audio), values),
            48000,
        )
        if order == "before_amplitude":
            audio *= amplitude[:, None]
        expected[start:] += audio * [0.3, 0.15]
    definition = yamaha_instrument.prepare(document)
    renderer = yamaha_instrument.OfflineYamaha(definition)
    head = renderer.advance([a for a in actions if a.tick < 2111], 0, 2111)
    saved = yamaha_instrument.YamahaSnapshot.model_validate_json(
        renderer.snapshot().model_dump_json()
    )
    renderer = yamaha_instrument.OfflineYamaha(definition)
    renderer.restore(saved)
    actual = np.concatenate(
        [head, renderer.advance([a for a in actions if a.tick >= 2111], 2111, 48000)]
    )
    check_audio(tmp_path / "scoped-controls.wav", actual, expected)
    whole = yamaha_instrument.OfflineYamaha(definition).advance(actions, 0, 48000)
    np.testing.assert_allclose(actual, whole, rtol=0, atol=1e-14)


@pytest.mark.parametrize("kind", ["dx7", "tx81z"])
@pytest.mark.parametrize("binding", ["key", "velocity"])
@pytest.mark.parametrize("transpose", [0, 12])
def test_yamaha_event_gain_uses_onset_key_or_unquantized_velocity(
    tmp_path: Path, kind: str, binding: str, transpose: int
) -> None:
    plain = instrument(kind, transpose)
    raw = plain.model_dump(mode="json")
    voice = raw["body"]["voices"][0]
    maximum = 127 if binding == "key" else 1
    target = {"name": "processing", "parameter": "amplitude"}
    voice["bindings"] = [{"name": "onset", "kind": binding}]
    voice["modulation"] = {
        "sources": [
            {"name": "onset", "scope": "voice", "minimum": 0, "maximum": maximum}
        ],
        "parameters": [
            {
                "target": target,
                "unit": "ratio",
                "scope": "voice",
                "minimum": 0,
                "maximum": 1,
                "default": 1,
            }
        ],
        "routes": [
            {
                "name": "gain",
                "source": "onset",
                "target": target,
                "unit": "ratio",
                "operation": "multiply",
                "points": [{"input": 0, "amount": 0}, {"input": maximum, "amount": 1}],
            }
        ],
    }
    document = SynthInstrumentScore.model_validate(raw)
    event = Trigger(
        tick=0, ordinal=0, part="main", trigger_id="note", key=64, velocity=0.25
    )
    actual = yamaha_instrument.OfflineYamaha(
        yamaha_instrument.prepare(document)
    ).advance(synth_trace.prepare(document.body, [event], seed=0).actions, 0, 48000)
    expected = yamaha_instrument.OfflineYamaha(
        yamaha_instrument.prepare(plain)
    ).advance(synth_trace.prepare(plain.body, [event], seed=0).actions, 0, 48000)
    expected *= 64 / 127 if binding == "key" else 0.25
    check_audio(tmp_path / "event-gain.wav", actual, expected)


def test_yamaha_dynamic_tuning_is_rejected_before_rendering() -> None:
    raw = instrument("dx7").model_dump(mode="json")
    raw["body"]["voices"][0]["modulation"] = {
        "parameters": [
            {
                "target": {"name": "processing", "parameter": "tuning_cents"},
                "unit": "cents",
                "scope": "voice",
                "minimum": -1200,
                "maximum": 1200,
                "default": 0,
            }
        ]
    }
    with pytest.raises(EngineError, match="dynamic tuning"):
        yamaha_instrument.prepare(SynthInstrumentScore.model_validate(raw))


@pytest.mark.parametrize("kind", ["dx7", "tx81z"])
@pytest.mark.parametrize("transpose", [-24, 12, 24])
def test_yamaha_transpose_matches_shifted_key_with_microtuning_and_snapshot(
    tmp_path: Path, kind: str, transpose: int
) -> None:
    template = instrument(kind).body.voices[0]
    raw = bytearray(template.patch.data)
    if isinstance(template, YamahaDX7Voice):
        raw[139:141] = bytes(2)
        for i in range(0, 126, 21):
            raw[i + 8 : i + 14] = bytes([35, 17, 29, 0, 0, 4])
            raw[i + 17] = (i // 21) % 2
        neutral = DX7Voice(data=bytes(raw))
        raw[144] = 24 + transpose
        actual = yamaha.DX7Renderer(DX7Voice(data=bytes(raw)), 69, 100, pitch_hz=442)
        expected = yamaha.DX7Renderer(
            neutral, 69 + transpose, 100, pitch_hz=442 * 2 ** (transpose / 12)
        )
    else:
        assert isinstance(template, YamahaTX81ZVoice)
        raw[56:58] = bytes(2)
        extra = bytearray(template.additional.data)
        for i in range(4):
            raw[i * 13 + 5] = 50
            raw[i * 13 + 6] = 2
            extra[i * 5] = i % 2
        additional = tx81z.TX81ZAdditional(data=bytes(extra))
        neutral = tx81z.TX81ZVoice(data=bytes(raw))
        raw[62] = 24 + transpose
        actual = yamaha.TX81ZRenderer(
            tx81z.TX81ZVoice(data=bytes(raw)), additional, 69, 100, pitch_hz=442
        )
        expected = yamaha.TX81ZRenderer(
            neutral,
            additional,
            69 + transpose,
            100,
            pitch_hz=442 * 2 ** (transpose / 12),
        )
    head = actual.render(24000)
    expected_head = expected.render(24000)
    actual.release()
    expected.release()
    saved = type(actual.snapshot()).model_validate_json(
        actual.snapshot().model_dump_json()
    )
    assert saved.note == 69
    tail = actual.render(24000)
    check_audio(
        tmp_path / "yamaha-transpose.wav",
        np.concatenate([head, tail]),
        np.concatenate([expected_head, expected.render(24000)]),
    )
    actual.restore(saved)
    np.testing.assert_array_equal(actual.render(24000), tail)


@pytest.mark.parametrize("note,transpose", [(0, 23), (127, 25)])
def test_dx7_rejects_transposed_keys_outside_supported_range(
    note: int, transpose: int
) -> None:
    raw = bytearray(voice().data)
    raw[144] = transpose
    with pytest.raises(EngineError, match="transposed note"):
        yamaha.DX7Renderer(DX7Voice(data=bytes(raw)), note, 100)


@pytest.mark.parametrize("first", ["dx7", "tx81z"])
def test_rejected_yamaha_clock_restore_preserves_audio_continuation(
    tmp_path: Path, first: str
) -> None:
    second = "tx81z" if first == "dx7" else "dx7"
    raw = instrument(first).model_dump(mode="json")
    raw["body"]["voices"][0]["name"] = first
    other = instrument(second).model_dump(mode="json")["body"]["voices"][0]
    other["name"] = second
    raw["body"]["voices"].append(other)
    document = SynthInstrumentScore.model_validate(raw)
    definition = yamaha_instrument.prepare(document)
    renderer = yamaha_instrument.OfflineYamaha(definition)
    rejected = renderer.snapshot()
    rejected.clocks[second] = rejected.clocks[second].model_copy(
        update={"sample_rate": 44100}
    )
    actions = synth_trace.prepare(
        document.body,
        [
            Trigger(
                tick=0, ordinal=0, part="main", trigger_id="note", key=69, velocity=0.5
            )
        ],
        seed=0,
    ).actions
    renderer.advance(actions, 0, 24000)
    before = renderer.snapshot()
    reference = yamaha_instrument.OfflineYamaha(definition)
    reference.restore(before)
    with pytest.raises(EngineError, match="different clock definition"):
        renderer.restore(rejected)
    assert renderer.snapshot() == before
    check_audio(
        tmp_path / "rejected-clock-restore.wav",
        renderer.advance([], 24000, 72000),
        reference.advance([], 24000, 72000),
    )
