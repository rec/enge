from pathlib import Path

import numpy as np
import pytest
from test_dynamic_synth import onset
from test_envelfo_latch import latch_score
from test_lfo_instrument import lfo_score
from test_synth import check_audio
from ufor import synth_trace
from ufor.control import TempoMap
from ufor.synth import SynthInstrumentScore

from enge import synth


def test_binding_names_do_not_alias_generator_names(tmp_path: Path) -> None:
    baseline = lfo_score("synth")
    raw = baseline.model_dump(mode="json")
    voice = raw["body"]["voices"][0]
    voice["envelfos"]["steady"] = {
        "kind": "envelope",
        "initial": 1,
        "segments": [{"duration": None}],
    }
    voice["bindings"] = [
        {"kind": "envelfo", "name": "wave", "reference": "envelfo"},
        {"kind": "envelfo", "name": "envelfo", "reference": "steady"},
    ]
    voice["modulation"]["sources"] = [
        {"name": "wave", "scope": "voice", "minimum": -1, "maximum": 1},
        {"name": "envelfo", "scope": "voice", "minimum": 0, "maximum": 1},
    ]
    voice["modulation"]["routes"][0]["source"] = "wave"
    document = SynthInstrumentScore.model_validate(raw)
    events = [onset(pitch=0.1).model_copy(update={"controls": {}})]
    actions = synth_trace.prepare(document.body, events, seed=0).actions
    definition = synth.prepare(document)
    expected = synth.PersistentSynth(synth.prepare(baseline), voices=1).advance(
        synth_trace.prepare(baseline.body, events, seed=0).actions, 0, 48000
    )
    assert np.ptp(expected[:, 0]) > 0.5
    actual = synth.OfflineSynth(definition).advance(actions, 0, 48000)
    check_audio(tmp_path / "independent-binding-names.wav", actual, expected)


@pytest.mark.parametrize("clock", ["seconds", "beats"])
def test_shared_patch_state_restores_pending_captures_from_json(
    tmp_path: Path, clock: str
) -> None:
    raw = latch_score("threshold", 1).model_dump(mode="json")
    raw["body"]["voices"][0]["envelfos"]["gesture"]["clock"] = clock
    document = SynthInstrumentScore.model_validate(raw)
    tempo = TempoMap.model_validate(
        {"points": [{"at_seconds": 0, "beat": 0, "bpm": 120}]}
    )
    actions = synth_trace.prepare(
        document.body,
        [onset(pitch=0.1).model_copy(update={"controls": {}})],
        seed=81,
    ).actions
    definition = synth.prepare(document)
    expected = synth.PersistentSynth(definition, voices=1, tempo_map=tempo).advance(
        actions, 0, 48000
    )
    renderer = synth.OfflineSynth(definition, tempo_map=tempo)
    boundary = 6000 if clock == "beats" else 10000
    first = renderer.advance(actions, 0, boundary)
    snapshot = renderer.snapshot()
    assert len(snapshot.graphs) == 1
    assert next(iter(snapshot.graphs.values())).pending
    serialized = snapshot.model_dump_json()
    tail = renderer.advance([], boundary, 48000)
    restored = synth.OfflineSynth(definition, tempo_map=tempo)
    restored.restore(synth.SynthSnapshot.model_validate_json(serialized))
    np.testing.assert_array_equal(restored.advance([], boundary, 48000), tail)
    assert snapshot.model_dump_json() == serialized
    check_audio(
        tmp_path / f"python-patch-{clock}.wav", np.vstack((first, tail)), expected
    )
