from math import fsum
from pathlib import Path

import numpy as np
import pytest
from test_dynamic_synth import change, onset
from test_lfo_instrument import lfo_score
from test_synth import check_audio
from ufor import synth_trace
from ufor.events import Release
from ufor.synth import SynthInstrumentScore

from enge import live, synth


@pytest.mark.parametrize("interpolation", ["linear", "step", "exponential"])
@pytest.mark.parametrize("source", ["control", "lfo", "envelope", "stages", "patch"])
@pytest.mark.parametrize("compose", [False, True])
def test_persistent_mapping_preserves_knots_fades_and_restore(
    tmp_path: Path, interpolation: str, source: str, compose: bool
) -> None:
    raw = lfo_score("synth").model_dump(mode="json")
    voice = raw["body"]["voices"][0]
    envelope = {
        "kind": "envelope",
        "segments": [{"duration": "1/2 s", "to": 1, "curve": 3}],
        "release": [{"duration": "1/4 s", "to": 0, "curve": -3}],
    }
    if source == "control":
        raw["body"]["controls"] = {"gain": {}}
        voice["envelfos"] = {}
        voice["bindings"] = [
            {
                "kind": "control",
                "name": "envelfo",
                "control": "gain",
                "smoothing": "1/100",
            }
        ]
        voice["modulation"]["sources"][0].update(scope="trigger", minimum=0)
    elif source == "envelope":
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
    if compose:
        voice["modulation"]["parameters"][0]["maximum"] = 2
        voice["modulation"]["parameters"].append(
            {
                "target": {"name": "processing", "parameter": "tuning_cents"},
                "unit": "cents",
                "scope": "voice",
                "minimum": -1200,
                "maximum": 1200,
                "default": 0,
            }
        )
        voice["modulation"]["routes"].extend(
            [
                {**route, "name": "second-multiplier"},
                {
                    **route,
                    "name": "pitch",
                    "target": {"name": "processing", "parameter": "tuning_cents"},
                    "unit": "cents",
                    "operation": "add",
                },
                {
                    **route,
                    "name": "subtract",
                    "operation": "add",
                    "interpolation": "linear",
                    "points": [{"input": 0, "amount": -1e16}],
                },
                {
                    **route,
                    "name": "restore",
                    "operation": "add",
                    "interpolation": "linear",
                    "points": [{"input": 0, "amount": 1e16}],
                },
                {
                    **route,
                    "name": "retain",
                    "operation": "add",
                    "interpolation": "linear",
                    "points": [{"input": 0, "amount": 1}],
                },
            ]
        )
    document = SynthInstrumentScore.model_validate(raw)
    actions = synth_trace.prepare(
        document.body,
        [
            onset(pitch=220).model_copy(
                update={"controls": {"gain": 0.125} if source == "control" else {}}
            ),
            *([change(12000, 0.75)] if source == "control" else []),
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


@pytest.mark.parametrize(
    "amounts",
    [
        [1e16, 1, -1e16],
        [1e16, -1e16, 1],
        [1, 1e100, 1, -1e100],
        [2, 2**-52, 2**-1074],
        [-0.5, -(2**-54), -(2**-1074)],
    ],
)
def test_route_sums_preserve_small_terms_under_cancellation(
    tmp_path: Path, amounts: list[float]
) -> None:
    raw = lfo_score("synth").model_dump(mode="json")
    voice = raw["body"]["voices"][0]
    voice["envelfos"]["envelfo"].update(delay=0, fade_in=0)
    voice["modulation"]["parameters"][0]["maximum"] = 4
    route = voice["modulation"]["routes"][0]
    voice["modulation"]["routes"] = [
        {
            **route,
            "name": f"route-{i}",
            "operation": "add",
            "points": [{"input": 0, "amount": v}],
        }
        for i, v in enumerate(amounts)
    ]
    document = SynthInstrumentScore.model_validate(raw)
    actions = synth_trace.prepare(
        document.body, [onset(pitch=0.1).model_copy(update={"controls": {}})], seed=0
    ).actions
    definition = synth.prepare(document)
    expected = np.tile([1 + fsum(amounts), 0], (48000, 1))
    for renderer in (synth.OfflineSynth(definition), synth.PersistentSynth(definition)):
        first = np.concatenate(
            [renderer.advance(actions, 0, 1), renderer.advance([], 1, 17003)]
        )
        saved = renderer.snapshot()
        tail = renderer.advance([], 17003, 48000)
        check_audio(
            tmp_path / f"sum-{type(renderer).__name__}.wav",
            np.concatenate([first, tail]),
            expected,
        )
        np.testing.assert_array_equal(np.concatenate([first, tail]), expected)
        renderer.restore(saved)
        np.testing.assert_array_equal(renderer.advance([], 17003, 48000), tail)
    engine = live.NativeLiveEngine({"synth": synth.PersistentSynth(definition)}, 48000)
    actual = np.empty((48000, 2))
    engine.advance_into({"synth": actions}, [], 0, 48000, actual)
    check_audio(tmp_path / "sum-live.wav", actual, expected)
    np.testing.assert_array_equal(actual, expected)
