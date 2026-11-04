from copy import deepcopy
from pathlib import Path

import numpy as np
import pytest
from test_synth import check_audio
from test_yamaha_instrument import instrument
from ufor import synth_trace, tx81z
from ufor.events import Trigger
from ufor.synth import SynthInstrumentScore, YamahaTX81ZVoice

from enge import yamaha, yamaha_instrument, yamaha_lfo
from enge.synth import EngineError


def performance(paired: bool = True) -> SynthInstrumentScore:
    raw = instrument("tx81z").model_dump(mode="json")
    original = raw["body"]["voices"][0]
    voices = []
    for name in ("a", "b", "first", "second", "off"):
        voice = deepcopy(original)
        voice["name"] = name
        key = 70 if name in ("a", "b") else 69
        voice["mapping"]["lowest_key"] = voice["mapping"]["highest_key"] = key
        patch = bytearray.fromhex(voice["patch"])
        patch[56:58] = bytes(2)
        if name == "a":
            patch[54:60] = bytes([20, 0, 30, 40, 1, 0])
        elif name == "b":
            patch[54:60] = bytes([75, 1, 70, 80, 0, 3])
        elif name == "off":
            patch[56:58] = bytes([99, 99])
        voice["patch"] = bytes(patch).hex()
        voices.append(voice)
    raw["body"]["voices"] = voices
    raw["body"]["tx81z_lfos"] = [
        {
            "name": "chip",
            "first": "a",
            "second": "b" if paired else None,
            "routes": {
                "a": "off",
                "b": "first",
                "first": "first",
                "second": "second" if paired else "first",
                "off": "off",
            },
        }
    ]
    return SynthInstrumentScore.model_validate(raw)


@pytest.mark.parametrize("paired", [False, True])
def test_performance_routes_silent_owners_and_restores_shared_clocks(
    tmp_path: Path, paired: bool
) -> None:
    document = performance(paired)
    templates = {v.name: v for v in document.body.voices}
    first, second = templates["a"], templates["b"]
    assert isinstance(first, YamahaTX81ZVoice)
    assert isinstance(second, YamahaTX81ZVoice)
    clock = yamaha_lfo.TX81ZLFOContext(
        first.patch, secondary=second.patch if paired else None
    )
    clock.keydown()
    head = clock.advance(1001)
    clock.keydown()
    rows = np.concatenate([head, clock.advance(46999)])
    if not paired:
        rows = rows[:, None, :]
    expected = np.zeros((48000, 2))
    for name, index, owner in [
        ("first", 0, first),
        ("second", int(paired), second if paired else first),
        ("off", 0, first),
    ]:
        template = templates[name]
        assert isinstance(template, YamahaTX81ZVoice)
        patch = bytearray(template.patch.data)
        patch[56:58] = owner.patch.data[56:58]
        modulation = rows[:, index].copy()
        if name == "off":
            modulation[:, :2] = 0
        for start in (0, 1001):
            note = yamaha.TX81ZRenderer(
                tx81z.TX81ZVoice(data=bytes(patch)), template.additional, 69, 127
            )
            expected[start:] += note.render(48000 - start, modulation[start:]) * [
                0.3,
                0.15,
            ]
    actions = synth_trace.prepare(
        document.body,
        [
            Trigger(
                tick=t,
                ordinal=0,
                part="main",
                trigger_id=f"note-{t}",
                key=69,
                velocity=1,
            )
            for t in (0, 1001)
        ],
        seed=0,
    ).actions
    definition = yamaha_instrument.prepare(document)
    engine = yamaha_instrument.OfflineYamaha(definition)
    audio = engine.advance(actions, 0, 17003)
    saved = yamaha_instrument.YamahaSnapshot.model_validate_json(
        engine.snapshot().model_dump_json()
    )
    replay = yamaha_instrument.OfflineYamaha(definition)
    replay.restore(saved)
    actual = np.concatenate([audio, replay.advance([], 17003, 48000)])
    check_audio(tmp_path / "performance-lfos.wav", actual, expected)
    assert set(saved.clocks) == {"a"}
    assert saved.clocks["a"].frame == 17003
    assert all(v.template not in ("a", "b") for v in saved.voices)
    engine.advance([], 17003, 48000)
    assert engine.snapshot() == replay.snapshot()


def test_performance_rejects_unimplemented_software_vibrato() -> None:
    raw = performance().model_dump(mode="json")
    raw["body"]["tx81z_lfos"][0]["routes"]["first"] = "vibrato"
    with pytest.raises(EngineError, match="software vibrato"):
        yamaha_instrument.prepare(SynthInstrumentScore.model_validate(raw))
