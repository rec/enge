from math import ceil
from pathlib import Path

import numpy as np
import pytest
from meedee import messages
from test_synth import check_audio
from test_tx81z_integer import opz_carrier
from test_tx81z_lfo import lfo_voice
from test_yamaha_instrument import instrument
from ufor import performance_binding, synth_trace, tx81z
from ufor.events import Trigger
from ufor.instrument_trace import RetirementCause, VoiceRetirement
from ufor.synth import SynthInstrumentScore

from enge import yamaha, yamaha_instrument
from enge.performance import MidiConverter
from enge.synth import EngineError
from enge.yamaha_portamento import TX81ZPortamento


@pytest.mark.parametrize(
    "first,last,source,target", [(108, 109, 108, 97), (13, 12, 13, 24)]
)
def test_legato_glide_starts_at_previous_sounding_pitch_across_wrap(
    tmp_path: Path, first: int, last: int, source: int, target: int
) -> None:
    raw = bytearray(lfo_voice().data)
    raw[56:58] = bytes(2)
    raw[66] = 18
    patch = tx81z.TX81ZVoice(data=bytes(raw))
    extra = tx81z.TX81ZAdditional(data=bytes(23))
    note = yamaha.TX81ZRenderer(
        patch, extra, first, 127, pitch_hz=440 * 2 ** ((first - 69) / 12) * 1.01
    )
    head = note.render(1001)
    note.retarget(last, 127, pitch_hz=440 * 2 ** ((last - 69) / 12) * 1.01)
    start_hz = 440 * 2 ** ((source - 69) / 12) * 1.01
    target_hz = 440 * 2 ** ((target - 69) / 12) * 1.01
    assert note.portamento is not None
    assert note.portamento.current_hz == pytest.approx(start_hz)
    glide = TX81ZPortamento.start(18, 48000, start_hz, target_hz)
    frequencies = np.concatenate(
        [np.full(1001, start_hz), target_hz * glide.ratios(46999)]
    )
    expected = (4 * opz_carrier(frequencies))[:, None]
    saved = note.snapshot()
    actual = np.concatenate([head, note.render(46999)])
    check_audio(tmp_path / "legato-wrap.wav", actual, expected)
    note.restore(yamaha.TX81ZSnapshot.model_validate_json(saved.model_dump_json()))
    check_audio(
        tmp_path / "legato-wrap-replay.wav",
        np.concatenate([head, note.render(46999)]),
        actual,
    )


@pytest.mark.parametrize(
    "time,ascending,trace",
    [
        (1, True, [0, 444, 762, 1016, 1206, 1333, 1397, 1460, 1524, 1536]),
        (1, False, [1536, 1091, 774, 520, 329, 202, 75, 12, 0]),
        (
            18,
            True,
            [
                0,
                210,
                390,
                540,
                660,
                780,
                870,
                960,
                1050,
                1110,
                1170,
                1230,
                1290,
                1320,
                1350,
                1380,
                1410,
                1440,
                1470,
                1500,
                1530,
                1536,
            ],
        ),
        (
            18,
            False,
            [
                1536,
                1326,
                1146,
                996,
                846,
                726,
                606,
                516,
                426,
                366,
                306,
                246,
                186,
                156,
                126,
                96,
                66,
                36,
                6,
                0,
            ],
        ),
        (99, True, [0] + [3 * i // 2 for i in range(1, 96)]),
    ],
)
def test_portamento_matches_measured_two_octave_register_steps(
    tmp_path: Path, time: int, ascending: bool, trace: list[int]
) -> None:
    start, target = (440, 1760) if ascending else (1760, 440)
    glide = TX81ZPortamento.start(time, 48000, start, target)
    actual = glide.ratios(48000)
    steps = np.repeat(trace + [trace[-1]] * (96 - len(trace)), 500)
    expected = 2 ** ((steps - (1536 if ascending else 0)) / 768)
    check_audio(
        tmp_path / "measured-portamento.wav", actual[:, None], expected[:, None]
    )


@pytest.mark.parametrize("rate", [44100, 48000, 96000])
def test_portamento_clock_waits_one_firmware_tick_before_first_update(
    rate: int,
) -> None:
    glide = TX81ZPortamento.start(18, rate, 440, 1760)
    boundary = ceil(rate / 96)
    glide.ratios(boundary - 1)
    assert glide.current_hz == 440
    assert glide.ratios(1)[0] == 0.25
    assert glide.current_hz == pytest.approx(440 * 2 ** (210 / 768))
    before = glide.model_copy(deep=True)
    with pytest.raises(EngineError, match="nonnegative"):
        glide.ratios(-1)
    assert glide == before


@pytest.mark.parametrize("time", [0, 1, 99])
def test_note_portamento_preserves_fixed_operators_detune_and_snapshot(
    tmp_path: Path, time: int
) -> None:
    raw = bytearray(lfo_voice().data)
    raw[56:58] = bytes(2)
    raw[66] = time
    for i in range(0, 52, 13):
        raw[i + 12] = 6
    extra = bytearray(23)
    extra[0] = 1
    voice = tx81z.TX81ZVoice(data=bytes(raw))
    additional = tx81z.TX81ZAdditional(data=bytes(extra))
    tuning = np.linspace(0, 1200, 48000)
    note = yamaha.TX81ZRenderer(
        voice, additional, 69, 127, pitch_hz=442, start_pitch_hz=221
    )
    before = note.snapshot()
    with pytest.raises(EngineError, match="tuning"):
        note.render(1, tuning_cents=np.array([np.nan]))
    assert note.snapshot() == before
    first = note.render(17003, tuning_cents=tuning[:17003])
    saved = note.snapshot()
    saved_json = saved.model_dump_json()
    tail = note.render(30997, tuning_cents=tuning[17003:])
    assert saved.model_dump_json() == saved_json
    note.restore(yamaha.TX81ZSnapshot.model_validate_json(saved_json))
    replay = note.render(30997, tuning_cents=tuning[17003:])
    check_audio(
        tmp_path / "portamento-replay.wav",
        np.concatenate([first, tail]),
        np.concatenate([first, replay]),
    )
    glide = (
        TX81ZPortamento.start(time, 48000, 221, 442).ratios(48000)
        if time
        else np.ones(48000)
    )
    ratio_frequency = 442 * glide * 2 ** (tuning / 1200) + 9 * 3579545 / (64 * 2**20)
    fixed_frequency = yamaha.tx81z_base_frequency(
        voice.operator(4), additional.operator(4), 69
    )
    expected = 3 * opz_carrier(ratio_frequency) + opz_carrier(fixed_frequency)
    check_audio(
        tmp_path / "portamento-tuning.wav",
        np.concatenate([first, tail]),
        expected[:, None],
    )


@pytest.mark.parametrize(
    "keys,transpose,sounding",
    [
        ([69, 81, 57], 0, [69, 81, 57]),
        ([108, 109, 12], 0, [108, 97, 24]),
        ([84, 85, 0], 24, [108, 97, 24]),
        ([37, 36, 127], -24, [13, 24, 103]),
    ],
)
def test_polyphonic_portamento_uses_current_pitch_and_retains_part_history(
    tmp_path: Path,
    keys: list[int],
    transpose: int,
    sounding: list[int],
) -> None:
    raw = instrument("tx81z").model_dump(mode="json")
    template = raw["body"]["voices"][0]
    patch = bytearray.fromhex(template["patch"])
    patch[56:58] = bytes(2)
    patch[66] = 18
    patch[62] = 24 + transpose
    template["patch"] = bytes(patch).hex()
    document = SynthInstrumentScore.model_validate(raw)
    events = [
        Trigger(tick=t, ordinal=0, part=p, trigger_id=f"note-{i}", key=k, velocity=1)
        for i, (t, p, k) in enumerate(
            [
                (0, "main", keys[0]),
                (1001, "main", keys[1]),
                (2002, "main", keys[2]),
                (3003, "other", 76),
                (5005, "main", keys[0]),
            ]
        )
    ]
    actions = synth_trace.prepare(document.body, events, seed=0).actions
    actions.append(
        VoiceRetirement(
            tick=4004,
            ordinal=0,
            voice_id="voice-2",
            cause=RetirementCause.transport_stop,
            action="stop",
        )
    )
    actions.sort(key=lambda a: (a.tick, a.ordinal))
    renderer = yamaha_instrument.OfflineYamaha(yamaha_instrument.prepare(document))
    first = renderer.advance([a for a in actions if a.tick < 4005], 0, 4005)
    saved = renderer.snapshot()
    saved_json = saved.model_dump_json()
    tail = renderer.advance([a for a in actions if a.tick >= 4005], 4005, 48000)
    assert saved.model_dump_json() == saved_json
    renderer.restore(yamaha_instrument.YamahaSnapshot.model_validate_json(saved_json))
    replay = renderer.advance([a for a in actions if a.tick >= 4005], 4005, 48000)
    check_audio(
        tmp_path / "instrument-glide-replay.wav",
        np.concatenate([first, tail]),
        np.concatenate([first, replay]),
    )
    pitches = [440 * 2 ** ((k - 69) / 12) for k in sounding]
    second_glide = TX81ZPortamento.start(18, 48000, pitches[0], pitches[1])
    second_glide.ratios(1001)
    third_start = second_glide.current_hz
    third_glide = TX81ZPortamento.start(18, 48000, third_start, pitches[2])
    third_glide.ratios(2002)
    starts = [None, pitches[0], third_start, None, third_glide.current_hz]
    expected = np.zeros((48000, 1))
    for i, event in enumerate(events):
        end = 4004 if i == 2 else 48000
        note = yamaha.TX81ZRenderer(
            tx81z.TX81ZVoice(data=bytes(patch)),
            tx81z.TX81ZAdditional(data=bytes(23)),
            event.key,
            127,
            start_pitch_hz=starts[i],
        )
        expected[event.tick : end] += note.render(end - event.tick)
    check_audio(
        tmp_path / "instrument-portamento.wav",
        np.concatenate([first, tail]),
        expected * np.array([[0.3, 0.15]]),
    )


@pytest.mark.parametrize("disabled_at", [None, 0, 1001])
def test_portamento_switch_cancels_without_restarting_the_glide(
    tmp_path: Path, disabled_at: int | None
) -> None:
    raw = bytearray(lfo_voice().data)
    raw[56:58] = bytes(2)
    raw[66] = 99
    voice = tx81z.TX81ZVoice(data=bytes(raw))
    extra = tx81z.TX81ZAdditional(data=bytes(23))
    note = yamaha.TX81ZRenderer(voice, extra, 69, 127, pitch_hz=442, start_pitch_hz=221)
    for invalid in (
        np.array([-0.1]),
        np.array([1.1]),
        np.array([np.nan]),
        np.ones((1, 1)),
    ):
        before = note.snapshot()
        with pytest.raises(EngineError, match="portamento rows"):
            note.render(1, portamento_enabled=invalid)
        assert note.snapshot() == before
    switch = np.full(48000, 0.5)
    if disabled_at is not None:
        switch[disabled_at:2002] = 63 / 127
        switch[2002:] = 64 / 127
    first = note.render(1000, portamento_enabled=switch[:1000])
    saved = note.snapshot()
    tail = note.render(47000, portamento_enabled=switch[1000:])
    note.restore(saved)
    replay = note.render(47000, portamento_enabled=switch[1000:])
    actual = np.concatenate([first, tail])
    check_audio(tmp_path / "switch-replay.wav", actual, np.concatenate([first, replay]))
    ratios = TX81ZPortamento.start(99, 48000, 221, 442).ratios(48000)
    if disabled_at is not None:
        ratios[disabled_at:] = 1
    expected = yamaha.TX81ZRenderer(voice, extra, 69, 127, pitch_hz=442).render(
        48000, tuning_cents=1200 * np.log2(ratios)
    )
    check_audio(tmp_path / "switch-cancellation.wav", actual, expected)


def test_cc65_switch_respects_part_scope_smoothing_and_pitch_history(
    tmp_path: Path,
) -> None:
    raw = instrument("tx81z").model_dump(mode="json")
    raw["body"]["controls"] = {"glide": {"default": 1}}
    template = raw["body"]["voices"][0]
    patch = bytearray.fromhex(template["patch"])
    patch[56:58] = bytes(2)
    patch[66] = 18
    template["patch"] = bytes(patch).hex()
    template["bindings"] = [
        {"name": "glide", "kind": "control", "control": "glide", "smoothing": "1/200"}
    ]
    target = {"name": "yamaha", "parameter": "portamento"}
    template["modulation"] = {
        "sources": [{"name": "glide", "scope": "part", "minimum": 0, "maximum": 1}],
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
                "name": "glide",
                "source": "glide",
                "target": target,
                "unit": "ratio",
                "operation": "multiply",
                "points": [{"input": 0, "amount": 0}, {"input": 1, "amount": 1}],
            }
        ],
    }
    document = SynthInstrumentScore.model_validate(raw)
    binding = performance_binding.PerformanceBindingScore.model_validate(
        {
            "name": "pedal",
            "title": "Portamento pedal",
            "body": {
                "instrument": {"selector": "test:instrument"},
                "midi": [
                    {
                        "channels": [c],
                        "part": p,
                        "repeated_key_release": "oldest",
                        "controllers": [{"number": 65, "control": "glide"}],
                    }
                    for c, p in [(1, "main"), (2, "other")]
                ],
            },
        }
    )
    converter = MidiConverter(source="pedal", binding=binding, instrument=document)
    inputs = [
        (0, messages.NoteOn(channel=1, note=69, velocity=127)),
        (0, messages.NoteOn(channel=2, note=57, velocity=127)),
        (1001, messages.NoteOn(channel=1, note=81, velocity=127)),
        (1001, messages.NoteOn(channel=2, note=69, velocity=127)),
        (2002, messages.ControlChange(channel=1, controller=65, value=63)),
        (3003, messages.ControlChange(channel=1, controller=65, value=64)),
        (4004, messages.NoteOn(channel=1, note=69, velocity=127)),
    ]
    events = [
        e
        for i, (t, m) in enumerate(inputs)
        for e in converter.feed(m, tick=t, ordinal=i)
    ]
    actions = synth_trace.prepare(document.body, events, seed=0).actions
    renderer = yamaha_instrument.OfflineYamaha(yamaha_instrument.prepare(document))
    first = renderer.advance([a for a in actions if a.tick < 2200], 0, 2200)
    saved = renderer.snapshot()
    tail = renderer.advance([a for a in actions if a.tick >= 2200], 2200, 48000)
    renderer.restore(
        yamaha_instrument.YamahaSnapshot.model_validate_json(saved.model_dump_json())
    )
    replay = renderer.advance([a for a in actions if a.tick >= 2200], 2200, 48000)
    actual = np.concatenate([first, tail])
    check_audio(tmp_path / "cc65-replay.wav", actual, np.concatenate([first, replay]))
    switch = np.ones(48000)
    switch[2002:2242] = 1 + (63 / 127 - 1) * np.arange(240) / 240
    switch[2242:3003] = 63 / 127
    switch[3003:3243] = (63 + np.arange(240) / 240) / 127
    switch[3243:] = 64 / 127
    starts = [None, None, 440, 220, 880]
    expected = np.zeros((48000, 1))
    notes = [e for e in events if isinstance(e, Trigger)]
    for event, start_pitch in zip(notes, starts, strict=True):
        assert event.key is not None
        note = yamaha.TX81ZRenderer(
            tx81z.TX81ZVoice(data=bytes(patch)),
            tx81z.TX81ZAdditional(data=bytes(23)),
            event.key,
            127,
            start_pitch_hz=start_pitch,
        )
        expected[event.tick :] += note.render(
            48000 - event.tick,
            portamento_enabled=switch[event.tick :] if event.part == "main" else None,
        )
    check_audio(
        tmp_path / "cc65-portamento.wav", actual, expected * np.array([[0.3, 0.15]])
    )
