from pathlib import Path

import numpy as np
import pytest
from test_dynamic_synth import onset
from test_lfo_instrument import lfo_score
from test_synth import check_audio
from ufor import synth_trace
from ufor.events import Release
from ufor.synth import SynthInstrumentScore

from enge import synth


@pytest.mark.parametrize("interpolation", ["linear", "step", "exponential"])
@pytest.mark.parametrize("source", ["lfo", "envelope", "stages", "patch"])
def test_persistent_mapping_preserves_knots_fades_and_restore(
    tmp_path: Path, interpolation: str, source: str
) -> None:
    raw = lfo_score("synth").model_dump(mode="json")
    voice = raw["body"]["voices"][0]
    envelope = {
        "kind": "envelope",
        "segments": [{"duration": "1/2 s", "to": 1, "curve": 3}],
        "release": [{"duration": "1/4 s", "to": 0, "curve": -3}],
    }
    if source == "envelope":
        voice["envelfos"]["envelfo"] = envelope
    elif source == "stages":
        envelope["release"] = []
        voice["envelfos"]["envelfo"] = {
            "kind": "stages",
            "initial_stage": "sweep",
            "stages": [{"name": "sweep", "envelfo": envelope}],
        }
    elif source == "patch":
        voice["envelfos"]["envelfo"] = {
            "kind": "patch",
            "envelfos": {
                "sweep": envelope,
                "scaled": {"kind": "affine", "input": "sweep", "scale": 0.75},
            },
            "outputs": {"value": "scaled"},
        }
        voice["bindings"][0]["output"] = "value"
    else:
        voice["envelfos"]["envelfo"].update(center=0.1, depth=0.8)
    if source == "envelope":
        voice["modulation"]["sources"][0]["minimum"] = 0
    route = voice["modulation"]["routes"][0]
    route.update(
        interpolation=interpolation,
        points=[
            {"input": 0.125, "amount": 0.125},
            {"input": 0.25, "amount": 0.75},
            {"input": 0.5, "amount": 0.5},
        ],
    )
    document = SynthInstrumentScore.model_validate(raw)
    actions = synth_trace.prepare(
        document.body,
        [
            onset(pitch=220).model_copy(update={"controls": {}}),
            Release(tick=24000, ordinal=0, part="main", trigger_id="note"),
        ],
        seed=0,
    ).actions
    definition = synth.prepare(document)
    expected = synth.OfflineSynth(definition).advance(actions, 0, 48000)
    renderer = synth.PersistentSynth(definition)
    first = renderer.advance([a for a in actions if a.tick < 17003], 0, 17003)
    restored = synth.PersistentSynth(definition)
    restored.restore(renderer.snapshot())
    remaining = [a for a in actions if a.tick >= 17003]
    tail = renderer.advance(remaining, 17003, 48000)
    check_audio(tmp_path / "mapped.wav", np.concatenate([first, tail]), expected)
    np.testing.assert_array_equal(restored.advance(remaining, 17003, 48000), tail)
