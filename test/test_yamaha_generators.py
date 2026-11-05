from pathlib import Path

import numpy as np
import pytest
from test_synth import check_audio
from test_yamaha_instrument import instrument
from ufor import synth_trace
from ufor.events import EnvelfoChange, Release, Trigger
from ufor.synth import SynthInstrumentScore

from enge import yamaha_instrument


@pytest.mark.parametrize("kind", ["dx7", "tx81z", "tx81z-mono"])
@pytest.mark.parametrize("scope", ["instrument", "part"])
def test_yamaha_patch_sources_follow_shared_time_and_restore(
    tmp_path: Path, kind: str, scope: str
) -> None:
    raw = instrument("tx81z" if kind == "tx81z-mono" else kind).model_dump(mode="json")
    template = raw["body"]["voices"][0]
    if kind == "tx81z-mono":
        patch = bytearray.fromhex(template["patch"])
        patch[63] = 1
        template["patch"] = bytes(patch).hex()
    template["envelfos"] = {
        "gesture": {
            "kind": "patch",
            "scope": scope,
            "envelfos": {
                "lfo": {"kind": "lfo", "rate": "4", "phase": "1/4"},
                "output": {
                    "kind": "affine",
                    "input": "lfo",
                    "scale": 0.25,
                    "offset": 0.75,
                },
            },
            "outputs": {"level": "output"},
        }
    }
    template["bindings"] = [
        {"name": "level", "kind": "envelfo", "reference": "gesture", "output": "level"}
    ]
    template["modulation"] = modulation(scope)
    document = SynthInstrumentScore.model_validate(raw)
    events = [
        Trigger(
            tick=t, ordinal=0, part="main", trigger_id=f"note-{t}", key=69, velocity=1
        )
        for t in (6000, 13001)
    ]
    events.append(Release(tick=16001, ordinal=0, part="main", trigger_id="note-13001"))
    plain = document.model_dump(mode="json")
    plain["body"]["voices"][0].update(envelfos={}, bindings=[], modulation={})
    expected = render(SynthInstrumentScore.model_validate(plain), events)
    signal = 0.75 + 0.25 * np.sin(2 * np.pi * (0.25 + 4 * np.arange(48000) / 48000))
    engine = yamaha_instrument.OfflineYamaha(yamaha_instrument.prepare(document))
    first = engine.advance(
        synth_trace.prepare(document.body, events, seed=0).actions, 0, 17003
    )
    saved = yamaha_instrument.YamahaSnapshot.model_validate_json(
        engine.snapshot().model_dump_json()
    )
    tail = engine.advance([], 17003, 48000)
    engine.restore(saved)
    check_audio(
        tmp_path / "patch-replay.wav",
        np.concatenate([first, engine.advance([], 17003, 48000)]),
        np.concatenate([first, tail]),
    )
    check_audio(
        tmp_path / "patch-level.wav",
        np.concatenate([first, tail]),
        expected * signal[:, None],
    )


@pytest.mark.parametrize("kind", ["dx7", "tx81z"])
@pytest.mark.parametrize("timing", ["event", "voice"])
def test_yamaha_envelope_release_obeys_minimum_hold_and_restores(
    tmp_path: Path, kind: str, timing: str
) -> None:
    raw = instrument(kind).model_dump(mode="json")
    template = raw["body"]["voices"][0]
    template["minimum_hold_seconds"] = "3/4"
    template["envelfos"] = {
        "level": {
            "kind": "envelope",
            "segments": [{"duration": "1/4 s", "to": 1}],
            "release": [{"duration": "1/4 s", "to": 0}],
        }
    }
    template["bindings"] = [
        {
            "name": "level",
            "kind": "envelfo",
            "reference": "level",
            "release_timing": timing,
        }
    ]
    template["modulation"] = modulation("voice")
    document = SynthInstrumentScore.model_validate(raw)
    events = [
        Trigger(tick=0, ordinal=0, part="main", trigger_id="note", key=69, velocity=1),
        Release(tick=12000, ordinal=0, part="main", trigger_id="note"),
    ]
    changes = [
        EnvelfoChange(
            tick=t, ordinal=0, part="main", trigger_id="note", name="level", action=a
        )
        for t, a in [(6000, "pause"), (9000, "resume")]
    ]
    plain = document.model_dump(mode="json")
    plain["body"]["voices"][0].update(envelfos={}, bindings=[], modulation={})
    expected = render(SynthInstrumentScore.model_validate(plain), events)
    release_at = 12000 if timing == "event" else 36000
    elapsed = np.arange(48000) - np.clip(np.arange(48000) - 6000, 0, 3000)
    signal = np.minimum(elapsed / 12000, 1)
    at_release = signal[release_at]
    signal[release_at:] = at_release * np.maximum(
        0, 1 - np.arange(48000 - release_at) / 12000
    )
    engine = yamaha_instrument.OfflineYamaha(yamaha_instrument.prepare(document))
    actions = synth_trace.prepare(
        document.body, sorted([*events, *changes], key=lambda e: e.tick), seed=0
    ).actions
    first = engine.advance([a for a in actions if a.tick < 7500], 0, 7500)
    saved = yamaha_instrument.YamahaSnapshot.model_validate_json(
        engine.snapshot().model_dump_json()
    )
    later = [a for a in actions if a.tick >= 7500]
    tail = engine.advance(later, 7500, 48000)
    engine.restore(saved)
    check_audio(
        tmp_path / "release-replay.wav",
        np.concatenate([first, engine.advance(later, 7500, 48000)]),
        np.concatenate([first, tail]),
    )
    check_audio(
        tmp_path / "release-level.wav",
        np.concatenate([first, tail]),
        expected * signal[:, None],
    )


def modulation(scope: str) -> dict[str, object]:
    return {
        "sources": [{"name": "level", "scope": scope, "minimum": 0, "maximum": 1}],
        "parameters": [
            {
                "target": {"name": "processing", "parameter": "amplitude"},
                "unit": "ratio",
                "scope": "voice",
                "minimum": 0,
                "maximum": 1,
                "default": 1,
            }
        ],
        "routes": [
            {
                "name": "level",
                "source": "level",
                "target": {"name": "processing", "parameter": "amplitude"},
                "unit": "ratio",
                "operation": "multiply",
                "points": [{"input": 0, "amount": 0}, {"input": 1, "amount": 1}],
            }
        ],
    }


def render(
    document: SynthInstrumentScore, events: list[Trigger | Release]
) -> np.ndarray:
    return yamaha_instrument.OfflineYamaha(yamaha_instrument.prepare(document)).advance(
        synth_trace.prepare(document.body, events, seed=0).actions, 0, 48000
    )


@pytest.mark.parametrize("kind", ["dx7", "tx81z"])
def test_yamaha_generator_events_cue_stages_across_restore(
    tmp_path: Path, kind: str
) -> None:
    raw = instrument(kind).model_dump(mode="json")
    template = raw["body"]["voices"][0]
    template["envelfos"] = {
        "clock": {
            "kind": "lfo",
            "scope": "voice",
            "rate": "1",
            "markers": [{"name": "mark", "position": "1/4"}],
        },
        "level": {
            "kind": "stages",
            "initial_stage": "waiting",
            "stages": [
                {
                    "name": n,
                    "envelfo": {
                        "kind": "envelope",
                        "initial": v,
                        "segments": [{"duration": None}],
                    },
                }
                for n, v in [("waiting", 0.5), ("bright", 1)]
            ],
            "transitions": [
                {
                    "from": ["waiting"],
                    "event": "cue.brighten",
                    "action": {"kind": "enter", "stage": "bright"},
                }
            ],
        },
    }
    template["bindings"] = [
        {"name": n, "kind": "envelfo", "reference": n} for n in ("clock", "level")
    ]
    template["modulation"] = modulation("voice")
    template["modulation"]["sources"][0]["minimum"] = -1
    template["modulation"]["sources"].append(
        {"name": "clock", "scope": "voice", "minimum": -1, "maximum": 1}
    )
    template["event_connections"] = [
        {"source": "clock", "port": "mark", "destination": "level", "cue": "brighten"}
    ]
    document = SynthInstrumentScore.model_validate(raw)
    events = [
        Trigger(tick=0, ordinal=0, part="main", trigger_id="note", key=69, velocity=1)
    ]
    expected = render(instrument(kind), events)
    signal = np.ones(48000)
    signal[:12000] = 0.5
    engine = yamaha_instrument.OfflineYamaha(yamaha_instrument.prepare(document))
    first = engine.advance(
        synth_trace.prepare(document.body, events, seed=0).actions, 0, 12000
    )
    saved = yamaha_instrument.YamahaSnapshot.model_validate_json(
        engine.snapshot().model_dump_json()
    )
    tail = engine.advance([], 12000, 48000)
    check_audio(
        tmp_path / "generator-cue.wav",
        np.concatenate([first, tail]),
        expected * signal[:, None],
    )
    engine.restore(saved)
    check_audio(
        tmp_path / "generator-cue-replay.wav",
        np.concatenate([first, engine.advance([], 12000, 48000)]),
        np.concatenate([first, tail]),
    )
