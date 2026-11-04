from pathlib import Path
from typing import Literal

import numpy as np
import pytest
import test_fm
import test_noise
import test_synth
import test_yamaha_instrument
from ufor import synth_trace
from ufor.events import ControlChange, Release, Trigger
from ufor.instrument_trace import TriggerRelease
from ufor.synth import SynthInstrumentScore

from enge import fm, noise, synth, yamaha_instrument


@pytest.mark.parametrize(
    "kind,backend",
    [
        (k, b)
        for k in ("synth", "fm", "noise")
        for b in ("numpy", "native", "persistent")
    ]
    + [(k, "numpy") for k in ("dx7", "tx81z")],
)
def test_physical_release_does_not_retire_sustained_synth_voices(
    tmp_path: Path, kind: str, backend: Literal["numpy", "native", "persistent"]
) -> None:
    if kind == "synth":
        document = test_synth.score()
    elif kind == "fm":
        document = test_fm.score()
    elif kind == "noise":
        document = test_noise.score()
    else:
        document = test_yamaha_instrument.instrument(kind)
    raw = document.model_dump(mode="json")
    raw["body"].setdefault("controls", {})["hold"] = {"default": 1}
    raw["body"]["sustain"] = {"control": "hold"}
    document = SynthInstrumentScore.model_validate(raw)
    trace = synth_trace.prepare(
        document.body,
        [
            Trigger(
                tick=0, ordinal=0, part="main", trigger_id="note", key=69, pitch_hz=440
            ),
            Release(tick=12000, ordinal=0, part="main", trigger_id="note"),
            ControlChange(
                tick=36000,
                ordinal=0,
                part="main",
                scope="part",
                control="hold",
                value=0,
            ),
        ],
        seed=0,
    )
    actions = [a for a in trace.actions if not isinstance(a, TriggerRelease)]
    observed = sorted(
        [
            *actions,
            TriggerRelease(tick=12000, ordinal=0, part="main", trigger_id="note"),
        ],
        key=lambda a: (a.tick, a.ordinal),
    )
    expected = renderer(document, kind, backend).advance(actions, 0, 48000)
    engine = renderer(document, kind, backend)
    first = engine.advance([a for a in observed if a.tick < 20000], 0, 20000)
    saved = engine.snapshot()
    assert saved.voices
    tail = engine.advance([a for a in observed if a.tick >= 20000], 20000, 48000)
    actual = np.concatenate([first, tail])
    test_synth.check_audio(tmp_path / "physical-release.wav", actual, expected)
    engine.restore(saved)
    replay = engine.advance([a for a in observed if a.tick >= 20000], 20000, 48000)
    test_synth.check_audio(
        tmp_path / "physical-release-replay.wav",
        np.concatenate([first, replay]),
        expected,
    )


def renderer(
    document: SynthInstrumentScore,
    kind: str,
    backend: Literal["numpy", "native", "persistent"],
) -> (
    synth.OfflineSynth
    | synth.PersistentSynth
    | fm.OfflineFM
    | noise.OfflineNoise
    | yamaha_instrument.OfflineYamaha
):
    if kind == "synth":
        definition = synth.prepare(document)
        return (
            synth.PersistentSynth(definition)
            if backend == "persistent"
            else synth.OfflineSynth(definition, backend)
        )
    if kind == "fm":
        definition = fm.prepare(document)
        return (
            fm.PersistentFM(definition)
            if backend == "persistent"
            else fm.OfflineFM(definition, backend)
        )
    if kind == "noise":
        definition = noise.prepare(document)
        return (
            noise.PersistentNoise(definition)
            if backend == "persistent"
            else noise.OfflineNoise(definition, backend)
        )
    return yamaha_instrument.OfflineYamaha(yamaha_instrument.prepare(document))
