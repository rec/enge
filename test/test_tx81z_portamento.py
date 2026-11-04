from math import ceil
from pathlib import Path

import numpy as np
import pytest
from test_synth import check_audio
from test_tx81z_lfo import lfo_voice
from test_yamaha_instrument import instrument
from ufor import synth_trace, tx81z
from ufor.events import Trigger
from ufor.instrument_trace import RetirementCause, VoiceRetirement
from ufor.synth import SynthInstrumentScore

from enge import yamaha, yamaha_instrument
from enge.synth import EngineError
from enge.yamaha_portamento import TX81ZPortamento


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
    phase = np.concatenate([[0], np.cumsum(ratio_frequency[:-1])]) / 48000
    expected = 0.75 * np.sin(2 * np.pi * phase)
    expected += 0.25 * np.sin(2 * np.pi * fixed_frequency * np.arange(48000) / 48000)
    check_audio(
        tmp_path / "portamento-tuning.wav",
        np.concatenate([first, tail]),
        expected[:, None],
    )


def test_polyphonic_portamento_uses_current_pitch_and_retains_part_history(
    tmp_path: Path,
) -> None:
    raw = instrument("tx81z").model_dump(mode="json")
    template = raw["body"]["voices"][0]
    patch = bytearray.fromhex(template["patch"])
    patch[56:58] = bytes(2)
    patch[66] = 18
    template["patch"] = bytes(patch).hex()
    document = SynthInstrumentScore.model_validate(raw)
    events = [
        Trigger(tick=t, ordinal=0, part=p, trigger_id=f"note-{i}", key=k, velocity=1)
        for i, (t, p, k) in enumerate(
            [
                (0, "main", 69),
                (1001, "main", 81),
                (2002, "main", 57),
                (3003, "other", 76),
                (5005, "main", 69),
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
    second_glide = TX81ZPortamento.start(18, 48000, 440, 880)
    second_glide.ratios(1001)
    third_start = second_glide.current_hz
    third_glide = TX81ZPortamento.start(18, 48000, third_start, 220)
    third_glide.ratios(2002)
    starts = [None, 440, third_start, None, third_glide.current_hz]
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


def test_instrument_rejects_mono_until_legato_lifecycle_is_implemented() -> None:
    raw = instrument("tx81z").model_dump(mode="json")
    template = raw["body"]["voices"][0]
    patch = bytearray.fromhex(template["patch"])
    patch[63] = 1
    template["patch"] = bytes(patch).hex()
    with pytest.raises(EngineError, match="mono mode"):
        yamaha_instrument.prepare(SynthInstrumentScore.model_validate(raw))
