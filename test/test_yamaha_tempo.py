from pathlib import Path

import numpy as np
import pytest
from test_synth import check_audio
from test_yamaha_generators import modulation
from test_yamaha_instrument import instrument
from ufor import synth_trace
from ufor.control import TempoMap
from ufor.events import Trigger
from ufor.synth import SynthInstrumentScore

from enge import yamaha_instrument
from enge.synth import EngineError


@pytest.mark.parametrize("kind", ["dx7", "tx81z"])
@pytest.mark.parametrize("generator", ["envelope", "lfo"])
def test_yamaha_beat_sources_follow_host_tempo_and_restore(
    tmp_path: Path, kind: str, generator: str
) -> None:
    raw = instrument(kind).model_dump(mode="json")
    template = raw["body"]["voices"][0]
    body = (
        {
            "kind": "envelope",
            "segments": [{"duration": "2 beats", "to": 1}],
            "clock": "beats",
        }
        if generator == "envelope"
        else {
            "kind": "lfo",
            "scope": "voice",
            "clock": "beats",
            "position_driver": "transport",
            "reset": "transport",
            "rate": "1",
        }
    )
    template["envelfos"] = {"level": body}
    template["bindings"] = [{"name": "level", "kind": "envelfo", "reference": "level"}]
    template["modulation"] = modulation("voice")
    if generator == "lfo":
        template["modulation"]["sources"][0]["minimum"] = -1
        template["modulation"]["routes"][0]["points"][0] = {"input": -1, "amount": 0}
    document = SynthInstrumentScore.model_validate(raw)
    definition = yamaha_instrument.prepare(document)
    clock = TempoMap.model_validate(
        {
            "points": [
                {"at_seconds": "0", "beat": "0", "bpm": "120"},
                {"at_seconds": "1/4", "beat": "1/2", "bpm": "60"},
                {"at_seconds": "1/2", "beat": "3/4", "bpm": "60", "running": False},
                {"at_seconds": "3/4", "beat": "8", "bpm": "120"},
            ]
        }
    )
    with pytest.raises(EngineError, match="requires a host tempo map"):
        yamaha_instrument.OfflineYamaha(definition)
    events = [
        Trigger(tick=0, ordinal=0, part="main", trigger_id="note", key=69, velocity=1)
    ]
    actions = synth_trace.prepare(document.body, events, seed=0).actions
    engine = yamaha_instrument.OfflineYamaha(definition, tempo_map=clock)
    first = engine.advance(actions, 0, 24000)
    saved = yamaha_instrument.YamahaSnapshot.model_validate_json(
        engine.snapshot().model_dump_json()
    )
    tail = engine.advance([], 24000, 48000)
    replay = yamaha_instrument.OfflineYamaha(definition, tempo_map=clock)
    replay.restore(saved)
    check_audio(
        tmp_path / "beat-replay.wav",
        np.concatenate([first, replay.advance([], 24000, 48000)]),
        np.concatenate([first, tail]),
    )
    foreign = yamaha_instrument.OfflineYamaha(
        definition,
        tempo_map=TempoMap.model_validate(
            {"points": [{"at_seconds": "0", "beat": "0", "bpm": "120"}]}
        ),
    )
    before = foreign.snapshot()
    with pytest.raises(EngineError, match="different host tempo map"):
        foreign.restore(saved)
    assert foreign.snapshot() == before
    plain = document.model_dump(mode="json")
    plain["body"]["voices"][0].update(envelfos={}, bindings=[], modulation={})
    baseline = SynthInstrumentScore.model_validate(plain)
    expected = yamaha_instrument.OfflineYamaha(
        yamaha_instrument.prepare(baseline)
    ).advance(synth_trace.prepare(baseline.body, events, seed=0).actions, 0, 48000)
    seconds = np.arange(48000) / 48000
    beats = (
        2 * np.minimum(seconds, 0.25)
        + np.clip(seconds - 0.25, 0, 0.25)
        + 2 * np.maximum(seconds - 0.75, 0)
    )
    if generator == "lfo":
        beats[36000:] += 7.25
        signal = 0.5 + 0.5 * np.sin(2 * np.pi * beats)
    else:
        signal = beats / 2
    check_audio(
        tmp_path / "beat-source.wav",
        np.concatenate([first, tail]),
        expected * signal[:, None],
    )
