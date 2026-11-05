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
    _, head = clock.advance_ticks(1001)
    clock.keydown()
    rows = np.concatenate([head, clock.advance_ticks(46999)[1][1:]])
    edges = np.arange(48001) * 3579545 // (64 * 48000)
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
                tx81z.TX81ZVoice(data=bytes(patch)),
                template.additional,
                69,
                127,
                start_frame=start,
            )
            expected[start:] += note.render(
                48000 - start,
                (edges[start + 1 :] - edges[start], modulation[edges[start] :]),
            ) * [
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


def vibrato_performance() -> SynthInstrumentScore:
    raw = performance().model_dump(mode="json")
    raw["body"]["controls"] = {"wheel": {"default": 1.0}}
    raw["body"]["tx81z_lfos"][0]["routes"].update(a="vibrato", b="vibrato")
    for voice in raw["body"]["voices"]:
        key = {"a": 69, "b": 70, "first": 69, "second": 71, "off": 72}[voice["name"]]
        voice["mapping"]["lowest_key"] = voice["mapping"]["highest_key"] = key
        if voice["name"] == "a":
            patch = bytearray.fromhex(voice["patch"])
            patch[71:73] = bytes([99, 99])
            voice["patch"] = bytes(patch).hex()
            voice["bindings"] = [
                {"name": "wheel", "kind": "control", "control": "wheel"}
            ]
            voice["modulation"] = {
                "sources": [
                    {"name": "wheel", "scope": "instrument", "minimum": 0, "maximum": 1}
                ],
                "parameters": [
                    {
                        "target": {"name": "yamaha", "parameter": "modulation_wheel"},
                        "unit": "ratio",
                        "scope": "voice",
                        "minimum": 0,
                        "maximum": 1,
                        "default": 0,
                    }
                ],
                "routes": [
                    {
                        "name": "wheel",
                        "source": "wheel",
                        "target": {"name": "yamaha", "parameter": "modulation_wheel"},
                        "unit": "ratio",
                        "operation": "add",
                        "points": [
                            {"input": 0, "amount": 0},
                            {"input": 1, "amount": 1},
                        ],
                    }
                ],
            }
    return SynthInstrumentScore.model_validate(raw)


def test_vibrato_uses_own_pitch_clock_without_amplitude_or_foreign_resets(
    tmp_path: Path,
) -> None:
    document = vibrato_performance()
    templates = {v.name: v for v in document.body.voices}
    first, second = templates["a"], templates["b"]
    assert isinstance(first, YamahaTX81ZVoice)
    assert isinstance(second, YamahaTX81ZVoice)
    hardware = yamaha_lfo.TX81ZLFOContext(first.patch, secondary=second.patch)
    clocks = {
        "a": yamaha_lfo.TX81ZLFOContext(first.patch),
        "b": yamaha_lfo.TX81ZLFOContext(second.patch),
    }
    segments: dict[str, list[np.ndarray]] = {"a": [], "b": [], "first": []}
    for start, end, name in [(0, 1001, "a"), (1001, 2003, "b"), (2003, 48000, "a")]:
        hardware.keydown()
        clocks[name].keydown()
        first_row = 1 if start else 0
        segments["first"].append(hardware.advance_ticks(end - start)[1][first_row:, 0])
        for owner, clock in clocks.items():
            rows = clock.advance_ticks(end - start)[1][first_row:]
            rows[:, 1] = 0
            segments[owner].append(rows)
    expected = np.zeros((48000, 2))
    for name, key, starts in [
        ("a", 69, [0, 2003]),
        ("b", 70, [1001]),
        ("first", 69, [0, 2003]),
    ]:
        template = templates[name]
        assert isinstance(template, YamahaTX81ZVoice)
        patch = bytearray(template.patch.data)
        if name == "first":
            patch[56:58] = first.patch.data[56:58]
        rows = np.concatenate(segments[name])
        for start in starts:
            note = yamaha.TX81ZRenderer(
                tx81z.TX81ZVoice(data=bytes(patch)),
                template.additional,
                key,
                127,
                start_frame=start,
            )
            controllers = np.full((48000 - start, 2), 253) if name == "a" else None
            edges = np.arange(start, 48001) * 3579545 // (64 * 48000)
            expected[start:] += note.render(
                48000 - start,
                (edges[1:] - edges[0], rows[edges[0] :]),
                controller_depths=controllers,
            ) * [0.3, 0.15]
    actions = synth_trace.prepare(
        document.body,
        [
            Trigger(
                tick=t,
                ordinal=0,
                part="main",
                trigger_id=f"note-{t}",
                key=k,
                velocity=1,
            )
            for t, k in [(0, 69), (1001, 70), (2003, 69)]
        ],
        seed=0,
    ).actions
    definition = yamaha_instrument.prepare(document)
    engine = yamaha_instrument.OfflineYamaha(definition)
    head = engine.advance(actions, 0, 17003)
    saved = yamaha_instrument.YamahaSnapshot.model_validate_json(
        engine.snapshot().model_dump_json()
    )
    restored = yamaha_instrument.OfflineYamaha(definition)
    restored.restore(saved)
    actual = np.concatenate([head, restored.advance([], 17003, 48000)])
    check_audio(tmp_path / "software-vibrato.wav", actual, expected)
    final = restored.snapshot()
    assert final.vibrato_clocks == {n: c.snapshot() for n, c in clocks.items()}
    assert final.clocks["a"] == hardware.snapshot()


def test_vibrato_restore_rejects_foreign_clock_without_changing_state() -> None:
    engine = yamaha_instrument.OfflineYamaha(
        yamaha_instrument.prepare(vibrato_performance())
    )
    saved = engine.snapshot()
    invalid = saved.model_copy(deep=True)
    invalid.vibrato_clocks["a"] = invalid.vibrato_clocks["a"].model_copy(
        update={"rate": 1}
    )
    with pytest.raises(EngineError, match="different clock definition"):
        engine.restore(invalid)
    assert engine.snapshot() == saved
