from pathlib import Path

import numpy as np
import pytest
from test_synth import check_audio
from test_yamaha_instrument import instrument
from ufor import synth_trace, wt11
from ufor.events import ControlChange, Trigger
from ufor.synth import SynthInstrumentScore, YamahaWT11Voice

from enge import yamaha, yamaha_instrument


@pytest.mark.parametrize("scope", ["instrument", "part", "trigger"])
def test_wt11_eg_only_assignment_activates_scoped_pressure_and_restores(
    tmp_path: Path, scope: str
) -> None:
    raw = instrument("wt11").model_dump(mode="json")
    template = raw["body"]["voices"][0]
    data = bytearray.fromhex(template["patch"])
    data[56:58] = bytes(2)
    data[67] = 0
    data[71:77] = bytes(6)
    data[75] = 50
    for index in range(0, 52, 13):
        data[index + 7] = 7
    template["patch"] = bytes(data).hex()
    template["additional2"] = bytes([0, 0, 50, 99, 1, 2, 3, 4, 5, 6]).hex()
    raw["body"]["controls"] = {"pressure": {"default": 0}}
    template["bindings"] = [
        {
            "name": "pressure",
            "kind": "control",
            "control": "pressure",
            "smoothing": "1/200",
        }
    ]
    target = {"name": "yamaha", "parameter": "aftertouch"}
    template["modulation"] = {
        "sources": [{"name": "pressure", "scope": scope, "minimum": 0, "maximum": 1}],
        "parameters": [
            {
                "target": target,
                "unit": "ratio",
                "scope": "voice",
                "minimum": 0,
                "maximum": 1,
                "default": 0,
            }
        ],
        "routes": [
            {
                "name": "pressure",
                "source": "pressure",
                "target": target,
                "unit": "ratio",
                "operation": "add",
                "points": [{"input": 0, "amount": 0}, {"input": 1, "amount": 1}],
            }
        ],
    }
    document = SynthInstrumentScore.model_validate(raw)
    events = [
        Trigger(tick=0, ordinal=0, part="main", trigger_id="first", key=69),
        Trigger(tick=1001, ordinal=0, part="other", trigger_id="second", key=69),
    ]
    events.extend(
        ControlChange(
            tick=t,
            ordinal=0,
            control="pressure",
            value=v,
            scope=scope,
            part=None if scope == "instrument" else "main",
            trigger_id="first" if scope == "trigger" else None,
        )
        for t, v in [(12000, 1), (12120, 0.4)]
    )
    actions = synth_trace.prepare(document.body, events, seed=0).actions
    definition = yamaha_instrument.prepare(document)
    engine = yamaha_instrument.OfflineYamaha(definition)
    head = engine.advance([a for a in actions if a.tick < 12111], 0, 12111)
    saved = yamaha_instrument.YamahaSnapshot.model_validate_json(
        engine.snapshot().model_dump_json()
    )
    replay = yamaha_instrument.OfflineYamaha(definition)
    replay.restore(saved)
    tail_actions = [a for a in actions if a.tick >= 12111]
    actual = np.concatenate([head, replay.advance(tail_actions, 12111, 48000)])
    whole = engine.advance(tail_actions, 12111, 48000)
    check_audio(
        tmp_path / "eg-instrument-replay.wav", actual, np.concatenate([head, whole])
    )
    pressure = np.zeros(48000)
    pressure[12000:12120] = np.arange(120) / 240
    pressure[12120:12360] = 0.5 - 0.1 * np.arange(240) / 240
    pressure[12360:] = 0.4
    expected = np.zeros((48000, 2))
    voice = document.body.voices[0]
    assert isinstance(voice, YamahaWT11Voice)
    data[76] = 99
    for index, start in enumerate((0, 1001)):
        reference = yamaha.WT11Renderer(
            wt11.WT11Voice(data=bytes(data)),
            voice.additional,
            wt11.WT11Additional2(data=bytes([0, 0, 50, 0, 1, 2, 3, 4, 5, 6])),
            69,
            127,
            start_frame=start,
        )
        positions = (
            pressure[start:]
            if index == 0 or scope == "instrument"
            else np.zeros(48000 - start)
        )
        expected[start:] += reference.render(48000 - start, breath=positions) * [
            0.3,
            0.15,
        ]
    check_audio(tmp_path / "eg-instrument-pressure.wav", actual, expected)
