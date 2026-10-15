from pathlib import Path

import numpy as np
import pytest
from test_dynamic_synth import onset
from test_envelfo_threshold import threshold_score
from test_synth import check_audio
from ufor import synth_trace
from ufor.control import TempoMap
from ufor.synth import SynthInstrumentScore

from enge import synth


@pytest.mark.parametrize("source_clock", ["seconds", "beats"])
def test_cross_clock_patch_deliveries_and_snapshot_continuation(
    tmp_path: Path, source_clock: str
) -> None:
    raw = threshold_score("export").model_dump(mode="json")
    voice = raw["body"]["voices"][0]
    voice["envelfos"]["gesture"]["clock"] = source_clock
    receiver = voice["envelfos"]["receiver"]
    receiver["clock"] = "beats" if source_clock == "seconds" else "seconds"
    receiver["stages"][1]["envelfo"].update(
        initial=0.5,
        segments=[
            {
                "duration": "1/10 beat" if source_clock == "seconds" else "1/10 s",
                "to": 1,
            }
        ],
    )
    for connection in voice["event_connections"]:
        connection.update(delay="12001/96000", every=2, offset=0, probability=0.5)
    document = SynthInstrumentScore.model_validate(raw)
    tempo = TempoMap.model_validate(
        {
            "points": [
                {"at_seconds": "0", "beat": "0", "bpm": "137"},
                {
                    "at_seconds": "1/4",
                    "beat": "137/240",
                    "bpm": "137",
                    "running": False,
                },
                {"at_seconds": "1/2", "beat": "-2", "bpm": "91"},
            ]
        }
    )
    actions = synth_trace.prepare(
        document.body,
        [onset(pitch=0.1).model_copy(update={"controls": {}})],
        seed=81,
    ).actions
    definition = synth.prepare(document)
    expected = synth.OfflineSynth(definition, tempo_map=tempo).advance(
        actions, 0, 48000
    )
    assert np.ptp(expected[:, 0]) > 0.05
    renderer = synth.PersistentSynth(definition, voices=1, tempo_map=tempo)
    first = renderer.advance(actions, 0, 9000)
    snapshot = renderer.snapshot()
    middle = renderer.advance([], 9000, 17003)
    restored = synth.PersistentSynth(definition, voices=1, tempo_map=tempo)
    restored.restore(snapshot)
    np.testing.assert_array_equal(restored.advance([], 9000, 17003), middle)
    tail = renderer.advance([], 17003, 48000)
    check_audio(
        tmp_path / f"native-patch-{source_clock}.wav",
        np.vstack((first, middle, tail)),
        expected,
    )
