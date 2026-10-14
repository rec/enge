from pathlib import Path

import numpy as np
import pytest
from test_dynamic_synth import onset
from test_lfo_instrument import lfo_score
from test_synth import check_audio
from ufor import synth_trace
from ufor.control import TempoMap
from ufor.events import EnvelfoChange
from ufor.synth import SynthInstrumentScore

from enge import synth


@pytest.mark.parametrize("staged", [False, True])
@pytest.mark.parametrize("clock", ["seconds", "beats"])
def test_indefinite_envelope_audio_matches_native_across_advances_and_restore(
    tmp_path: Path, staged: bool, clock: str
) -> None:
    raw = lfo_score("synth", "voice").model_dump(mode="json")
    body = {
        "kind": "envelope",
        "polarity": "bipolar",
        "segments": [
            {"duration": f"1/8 {'s' if clock == 'seconds' else 'beat'}", "to": 1},
            {"duration": None},
            {"duration": f"1/8 {'s' if clock == 'seconds' else 'beat'}", "to": 0},
            {"duration": None},
            {"duration": f"1/8 {'s' if clock == 'seconds' else 'beat'}", "to": 0.5},
        ],
    }
    raw["body"]["voices"][0]["envelfos"]["envelfo"] = (
        {
            "kind": "stages",
            "clock": clock,
            "initial_stage": "sustain",
            "stages": [{"name": "sustain", "envelfo": body}],
        }
        if staged
        else body | {"clock": clock}
    )
    document = SynthInstrumentScore.model_validate(raw)
    tempo = TempoMap.model_validate(
        {
            "points": [
                {"at_seconds": "0", "beat": "0", "bpm": "120"},
                {"at_seconds": "1/4", "beat": "1/2", "bpm": "120", "running": False},
                {"at_seconds": "1/2", "beat": "1/2", "bpm": "120"},
            ]
        }
    )
    actions = synth_trace.prepare(
        document.body,
        [
            onset(0, pitch=0.1).model_copy(update={"controls": {}}),
            EnvelfoChange(
                tick=12000,
                ordinal=0,
                name="envelfo",
                part="main",
                trigger_id="note",
                action="advance",
            ),
            EnvelfoChange(
                tick=30000,
                ordinal=0,
                name="envelfo",
                part="main",
                trigger_id="note",
                action="advance",
            ),
        ],
        seed=0,
    ).actions
    definition = synth.prepare(document)
    expected = synth.OfflineSynth(definition, "numpy", tempo_map=tempo).advance(
        actions, 0, 48000
    )
    renderer = synth.PersistentSynth(definition, voices=2, tempo_map=tempo)
    first = renderer.advance([a for a in actions if a.tick < 16001], 0, 16001)
    snapshot = renderer.snapshot()
    last = renderer.advance([a for a in actions if a.tick >= 16001], 16001, 48000)
    actual = np.concatenate((first, last))
    check_audio(tmp_path / f"indefinite-{clock}-{staged}.wav", actual, expected)
    restored = synth.PersistentSynth(definition, voices=2, tempo_map=tempo)
    restored.restore(snapshot)
    replay = restored.advance([a for a in actions if a.tick >= 16001], 16001, 48000)
    np.testing.assert_allclose(replay, last, atol=0)


@pytest.mark.parametrize("paused", [False, True])
def test_native_final_wait_completion_enters_the_next_stage(
    tmp_path: Path, paused: bool
) -> None:
    raw = lfo_score("synth", "voice").model_dump(mode="json")
    raw["body"]["voices"][0]["envelfos"]["envelfo"] = {
        "kind": "stages",
        "initial_stage": "quiet",
        "stages": [
            {
                "name": "quiet",
                "envelfo": {
                    "kind": "envelope",
                    "initial": 0.2,
                    "segments": [{"duration": None}],
                },
            },
            {
                "name": "loud",
                "envelfo": {
                    "kind": "envelope",
                    "initial": 1,
                    "segments": [{"duration": None}],
                },
            },
        ],
        "transitions": [
            {
                "from": ["quiet"],
                "event": "stage.done",
                "action": {"kind": "enter", "stage": "loud"},
            }
        ],
    }
    document = SynthInstrumentScore.model_validate(raw)
    actions = synth_trace.prepare(
        document.body,
        [
            onset(0, pitch=0.1).model_copy(update={"controls": {}}),
            *(
                [
                    EnvelfoChange(
                        tick=6000,
                        ordinal=0,
                        name="envelfo",
                        part="main",
                        trigger_id="note",
                        action="pause",
                    )
                ]
                if paused
                else []
            ),
            EnvelfoChange(
                tick=12000,
                ordinal=0,
                name="envelfo",
                part="main",
                trigger_id="note",
                action="advance",
            ),
        ],
        seed=0,
    ).actions
    definition = synth.prepare(document)
    expected = synth.OfflineSynth(definition, "numpy").advance(actions, 0, 48000)
    actual = synth.PersistentSynth(definition, voices=2).advance(actions, 0, 48000)
    check_audio(tmp_path / f"final-wait-stage-{paused}.wav", actual, expected)
    assert not np.isclose(actual[11999, 0], actual[12000, 0])
