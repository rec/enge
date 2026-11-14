from itertools import pairwise
from math import log
from pathlib import Path

import numpy as np
import pytest
from test_dynamic_synth import onset
from test_lfo_instrument import lfo_score
from test_synth import check_audio
from ufor import synth_trace
from ufor.events import Release
from ufor.samples import trace

from enge import sample_instrument, synth


@pytest.mark.parametrize("kind", ["synth", "sampler"])
@pytest.mark.parametrize("release_frame", [2400, 12000, 36000])
def test_captured_modulation_matches_equations_and_snapshot_continuation(
    tmp_path: Path, kind: str, release_frame: int
) -> None:
    prototype = lfo_score(kind)
    raw = prototype.model_dump(mode="json")
    voice = raw["body"]["voices" if kind == "synth" else "slots"][0]
    voice["envelope"]["release"] = [{"duration": "2 s", "to": 1}]
    voice["envelfos"]["envelfo"] = {
        "kind": "envelope",
        "initial": 0.2,
        "segments": [
            {"duration": "1/10 s", "to": 0.2},
            {"duration": "2/5 s", "to": 1.0},
            {
                "kind": "exponential_approach",
                "asymptote": 0.0,
                "time_constant": 0.1,
                "threshold": 0.3001,
                "transition_to": 0.3,
                "transition_seconds": 0.05,
            },
        ],
        "release": [
            {
                "kind": "exponential_approach",
                "asymptote": 0.0,
                "time_constant": 0.05,
                "threshold": 0.0001,
                "transition_to": 0.0,
                "transition_seconds": 0.05,
                "minimum_transition_distance": 0.0001,
            }
        ],
    }
    voice["modulation"]["sources"][0]["minimum"] = 0
    voice["modulation"]["routes"][0]["points"] = [
        {"input": 0, "amount": 0},
        {"input": 1, "amount": 1},
    ]
    document = type(prototype).model_validate(raw)
    voice["envelfos"] = {}
    voice["bindings"] = []
    voice["modulation"] = {}
    plain = type(prototype).model_validate(raw)
    events = [
        onset(pitch=440).model_copy(update={"controls": {}}),
        Release(tick=release_frame, ordinal=1, part="main", trigger_id="note"),
    ]
    if kind == "synth":
        prepared = synth.prepare(document)
        baseline = synth.prepare(plain)
        actions = synth_trace.prepare(document.body, events, seed=0).actions
        base_actions = synth_trace.prepare(plain.body, events, seed=0).actions
        source = synth.OfflineSynth(baseline).advance(base_actions, 0, 96000)
    else:
        samples = {"asset": np.tile([0.125, 0.0625], (96000, 1))}
        prepared = sample_instrument.prepare(document, samples)
        baseline = sample_instrument.prepare(plain, samples)
        actions = trace.prepare(document.body, events, seed=0).actions
        base_actions = trace.prepare(plain.body, events, seed=0).actions
        source = sample_instrument.OfflineSampler(baseline).advance(
            base_actions, 0, 96000
        )
    expected = source * levels(np.arange(96000) / 48000, release_frame / 48000)[:, None]
    for backend in ["numpy", "native"]:
        if kind == "synth":
            renderer = synth.OfflineSynth(prepared, backend=backend)
        else:
            renderer = sample_instrument.OfflineSampler(prepared, backend=backend)
        chunks = []
        boundaries = sorted({0, release_frame, release_frame + 7, 24001, 48000, 96000})
        for start, end in pairwise(boundaries):
            chunks.append(
                renderer.advance(
                    [a for a in actions if start <= a.tick < end], start, end
                )
            )
            snapshot = renderer.snapshot()
            restored = type(snapshot).model_validate_json(snapshot.model_dump_json())
            renderer = type(renderer)(prepared, backend=backend)
            renderer.restore(restored)
        check_audio(
            tmp_path / f"{kind}-{backend}.wav", np.concatenate(chunks), expected
        )
    persistent = (
        synth.PersistentSynth(prepared, voices=2)
        if kind == "synth"
        else sample_instrument.PersistentSampler(prepared, voices=2)
    )
    split = release_frame + 7
    head = persistent.advance([a for a in actions if a.tick < split], 0, split)
    snapshot = persistent.snapshot()
    tail = persistent.advance([], split, 96000)
    persistent.restore(snapshot)
    replay = np.concatenate(
        [persistent.advance([], split, 48001), persistent.advance([], 48001, 96000)]
    )
    check_audio(tmp_path / f"{kind}-replay.wav", replay, tail)
    check_audio(
        tmp_path / f"{kind}-persistent.wav", np.concatenate([head, tail]), expected
    )


def levels(at: np.ndarray, release: float) -> np.ndarray:
    def onset_level(age: np.ndarray) -> np.ndarray:
        curve = log(0.3001)
        duration = round(-0.1 * curve * 1e9) / 1e9
        decay_age = np.clip(age - 0.5, 0, duration)
        decay = np.exp(curve * decay_age / duration)
        settling = 0.3001 + (0.3 - 0.3001) * np.clip(
            (age - 0.5 - duration) / 0.05, 0, 1
        )
        return np.where(
            age < 0.1,
            0.2,
            np.where(
                age < 0.5,
                0.2 + 2 * (age - 0.1),
                np.where(age < 0.5 + duration, decay, settling),
            ),
        )

    initial = float(onset_level(np.asarray(release)))
    curve = log(0.0001 / initial)
    duration = round(-0.05 * curve * 1e9) / 1e9
    age = np.maximum(0, at - release)
    releasing = np.where(
        age < duration,
        initial * np.exp(curve * np.minimum(age / duration, 1)),
        0.0001 * np.maximum(0, 1 - (age - duration) / 0.05),
    )
    return np.where(at < release, onset_level(at), releasing)
