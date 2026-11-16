"""Portable definitions crossing the native preparation boundary."""

import json
from math import log
from pathlib import Path

import numpy as np
import pytest
import test_fm
import test_noise
from test_dynamic_synth import onset
from test_lfo_instrument import lfo_score, lfo_settings
from test_synth import check_audio
from ufor import envelfo, synth_trace
from ufor.segments import Segment

from enge import _native, fm, noise, synth


@pytest.mark.parametrize(
    "definition",
    [
        {"kind": "lfo", "rate": "1/0"},
        {"kind": "lfo", "rate": {"parameter": "speed"}},
        {"kind": "lfo", "rate": 1, "phase": True},
        {"kind": "envelope", "segments": [{"duration": "1/10 s", "to": 1}]},
    ],
)
def test_native_lfo_slots_validate_definitions(definition: dict[str, object]) -> None:
    with pytest.raises(ValueError):
        _native.GeneratorBank(
            48000,
            1,
            1,
            np.array([[0, 1, 0, 1.0]]),
            [json.dumps(definition)],
            [],
            [],
        )


def test_preparation_budget_covers_both_kinds_and_host_routes() -> None:
    lfo = envelfo.LFO(rate=1).model_dump_json()
    envelope = envelfo.Envelope(segments=[Segment(duration=0, to=1)]).model_dump_json()
    # Root, two route records, two definitions and one segment.
    _native.GeneratorBank(
        48000,
        1,
        1,
        np.array([[0, 1, 0, 1.0]]),
        [lfo],
        [envelope],
        [(0, 1, False)],
        max_records=6,
    )
    with pytest.raises(ValueError):
        _native.GeneratorBank(
            48000,
            1,
            1,
            np.array([[0, 1, 0, 1.0]]),
            [lfo],
            [envelope],
            [(0, 1, False)],
            max_records=5,
        )
    large = envelfo.Envelope(
        segments=[Segment(duration=0, to=1)] * 10000
    ).model_dump_json()
    with pytest.raises(ValueError):
        _native.GeneratorBank(
            48000, 1, 1, np.empty((0, 4)), [], [large], [(0, 1, False)]
        )
    _native.GeneratorBank(
        48000, 1, 1, np.empty((0, 4)), [], [large], [(0, 1, False)], max_records=10003
    )


def test_frame_conversion_rejects_overflow_and_mismatched_routes() -> None:
    envelope = json.dumps(
        {
            "kind": "envelope",
            "segments": [{"duration": "9223372036854775807 s", "to": 1}],
        }
    )
    with pytest.raises(ValueError):
        _native.GeneratorBank(
            48000, 1, 1, np.empty((0, 4)), [], [envelope], [(0, 1, False)]
        )
    with pytest.raises(ValueError):
        _native.GeneratorBank(
            48000,
            1,
            1,
            np.empty((0, 4)),
            [envelfo.LFO(rate=1).model_dump_json()],
            [],
            [],
        )


def test_repeat_count_uses_the_portable_signed_integer_boundary() -> None:
    definition = {
        "kind": "envelope",
        "playback": "loop",
        "segments": [{"duration": "1 s", "to": 1}],
        "repeat_count": 2**63 - 1,
    }
    _native.GeneratorBank(
        48000, 1, 1, np.empty((0, 4)), [], [json.dumps(definition)], [(0, 1, False)]
    )
    definition["repeat_count"] = 2**63
    with pytest.raises(ValueError):
        _native.GeneratorBank(
            48000, 1, 1, np.empty((0, 4)), [], [json.dumps(definition)], [(0, 1, False)]
        )


@pytest.mark.parametrize("kind", ["synth", "fm", "noise"])
@pytest.mark.parametrize("profile", ["reset", "approach"])
def test_named_envelope_profiles_preserve_audio_and_continuation(
    tmp_path: Path,
    kind: str,
    profile: str,
) -> None:
    prototype = (
        test_fm.score()
        if kind == "fm"
        else test_noise.score()
        if kind == "noise"
        else lfo_score("synth")
    )
    raw = prototype.model_dump(mode="json")
    voice = raw["body"]["voices"][0]
    settings = lfo_settings().model_dump(mode="json")
    voice.update({k: settings[k] for k in ("envelfos", "bindings", "modulation")})
    voice["modulation"]["sources"][0]["minimum"] = 0
    voice["modulation"]["routes"][0]["points"] = [
        {"input": 0, "amount": 0},
        {"input": 1, "amount": 1},
    ]
    voice["envelfos"]["envelfo"] = (
        {
            "kind": "envelope",
            "initial": 0.25,
            "retrigger": "reset",
            "segments": [{"duration": "1/2 s", "to": 0.75}],
        }
        if profile == "reset"
        else {
            "kind": "envelope",
            "initial": 1,
            "segments": [
                {
                    "kind": "exponential_approach",
                    "asymptote": 0,
                    "time_constant": 0.2,
                    "threshold": 0.25,
                    "transition_to": 0.2,
                    "transition_seconds": 0.05,
                }
            ],
        }
    )
    document = type(prototype).model_validate(raw)
    voice.update(envelfos={}, bindings=[], modulation={})
    plain = type(prototype).model_validate(raw)
    events = [onset(pitch=220).model_copy(update={"controls": {}})]
    actions = synth_trace.prepare(document.body, events, seed=0).actions
    base_actions = synth_trace.prepare(plain.body, events, seed=0).actions
    module = fm if kind == "fm" else noise if kind == "noise" else synth
    prepared = module.prepare(document)
    baseline = module.prepare(plain)
    reference = (
        fm.OfflineFM(baseline, backend="native")
        if kind == "fm"
        else noise.OfflineNoise(baseline, backend="native")
        if kind == "noise"
        else synth.OfflineSynth(baseline, backend="native")
    )
    source = reference.advance(base_actions, 0, 96000)
    age = np.arange(96000) / 48000
    if profile == "reset":
        gain = 0.25 + 0.5 * np.clip(age / 0.5, 0, 1)
    else:
        duration = round(-0.2 * log(0.25) * 1e9) / 1e9
        gain = np.where(
            age < duration,
            np.exp(log(0.25) * age / duration),
            0.25 - 0.05 * np.clip((age - duration) / 0.05, 0, 1),
        )
    renderer = (
        fm.PersistentFM(prepared, voices=2)
        if kind == "fm"
        else noise.PersistentNoise(prepared, voices=2)
        if kind == "noise"
        else synth.PersistentSynth(prepared, voices=2)
    )
    head = renderer.advance(actions, 0, 4003)
    snapshot = renderer.snapshot()
    tail = renderer.advance([], 4003, 96000)
    renderer.restore(snapshot)
    replay = np.concatenate(
        [renderer.advance([], 4003, 12001), renderer.advance([], 12001, 96000)]
    )
    check_audio(
        tmp_path / f"{kind}-{profile}.wav",
        np.concatenate([head, tail]),
        source * gain[:, None],
    )
    check_audio(tmp_path / f"{kind}-{profile}-continuation.wav", replay, tail)
