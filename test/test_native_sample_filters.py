from pathlib import Path

import numpy as np
import pytest
from test_dynamic_synth import onset
from test_sample_instrument import sample_score
from test_synth import check_audio
from ufor.events import ControlChange, Release
from ufor.instrument_trace import RetirementCause, VoiceRetirement
from ufor.samples import instrument, trace

from enge import live, sample_instrument


def test_native_sample_filter_rejects_cutoff_outside_authored_bounds(
    tmp_path: Path,
) -> None:
    raw = sample_score().model_dump(mode="json")
    raw["body"]["settings"] = {
        "controls": {"cutoff": {"default": 0}},
        "processing": {
            "filters": [
                {
                    "name": "tone",
                    "response": "lowpass",
                    "cutoff_hz": 500,
                    "boundary": "error",
                }
            ]
        },
        "bindings": [
            {"name": "cutoff", "kind": "control", "control": "cutoff", "smoothing": "0"}
        ],
        "modulation": {
            "sources": [
                {"name": "cutoff", "scope": "instrument", "minimum": 0, "maximum": 1}
            ],
            "parameters": [
                {
                    "target": {"name": "filter-tone", "parameter": "cutoff_hz"},
                    "unit": "hz",
                    "scope": "voice",
                    "minimum": 1,
                    "maximum": 50000,
                    "default": 500,
                }
            ],
            "routes": [
                {
                    "name": "cutoff",
                    "source": "cutoff",
                    "target": {"name": "filter-tone", "parameter": "cutoff_hz"},
                    "unit": "hz",
                    "operation": "add",
                    "points": [
                        {"input": 0, "amount": 0},
                        {"input": 1, "amount": 48000},
                    ],
                }
            ],
        },
    }
    document = instrument.SampleInstrumentScore.model_validate(raw)
    definition = sample_instrument.prepare(
        document, {"asset": np.full((96000, 2), 0.1)}
    )
    actions = trace.prepare(
        document.body,
        [
            onset(pitch=440).model_copy(update={"controls": {}}),
            ControlChange(
                tick=48000, ordinal=0, control="cutoff", value=1, scope="instrument"
            ),
        ],
        seed=0,
    ).actions
    reference = sample_instrument.OfflineSampler(definition)
    before = [a for a in actions if a.tick < 48000]
    expected = reference.advance(before, 0, 48000)
    engine = live.NativeLiveEngine(
        {"sample": sample_instrument.PersistentSampler(definition)}, 48000
    )
    actual = np.empty_like(expected)
    engine.advance_into({"sample": before}, [], 0, 48000, actual)
    check_audio(tmp_path / "sample-filter-boundary.wav", actual, expected)
    after = [a for a in actions if a.tick >= 48000]
    with pytest.raises(ValueError, match="cutoff/Q"):
        reference.advance(after, 48000, 48001)
    with pytest.raises(ValueError, match="cutoff/Q"):
        engine.advance_into({"sample": after}, [], 48000, 48001, np.empty((1, 2)))


@pytest.mark.parametrize("order", ["before_amplitude", "after_amplitude"])
@pytest.mark.parametrize("response", ["lowpass", "highpass", "bandpass", "notch"])
@pytest.mark.parametrize("dynamic", [False, True])
def test_native_sample_filters_preserve_channels_voice_reuse_and_snapshots(
    tmp_path: Path, order: str, response: str, dynamic: bool
) -> None:
    raw = sample_score(frames=30000).model_dump(mode="json")
    raw["body"]["settings"] = {
        "controls": {"expression": {"default": 0.25}},
        "processing": {
            "volume_db": -3,
            "filter_order": order,
            "filters": [{"name": "common", "response": "highpass", "cutoff_hz": 80}],
        },
        "envelope": {
            "segments": [{"duration": "0.04 s", "to": 1}],
            "release": [{"duration": "0.125 s", "to": 0}],
        },
    }
    raw["body"]["slots"][0]["processing"] = {
        "volume_db": -2,
        "filter_order": order,
        "filters": [
            {
                "name": "slot",
                "response": response,
                "cutoff_hz": 1320,
                "q": 0.8,
                "stages": 2,
            }
        ],
    }
    raw["body"]["slots"][0]["crossfades"] = [
        {
            "input": "control",
            "control": "expression",
            "scope": "instrument",
            "start": 0.2,
            "end": 0.8,
            "direction": "in",
            "curve": "equal_power",
            "smoothing_seconds": 0.025,
        }
    ]
    if dynamic:
        for settings, name, scope in [
            (raw["body"]["settings"], "common", "instrument"),
            (raw["body"]["slots"][0], "slot", "trigger"),
        ]:
            settings["processing"]["filters"][0]["boundary"] = "clamp"
            settings["processing"]["filters"][0]["minimum_hz"] = 20
            settings["processing"]["filters"][0]["cutoff_hz"] = 1
            settings["processing"]["filters"][0]["q"] = 1
            settings["bindings"] = [
                {"name": n, "kind": "control", "control": "expression", "smoothing": s}
                for n, s in [("cutoff", "0.03"), ("resonance", "0.007")]
            ]
            settings["modulation"] = {
                "sources": [
                    {"name": n, "scope": scope, "minimum": 0, "maximum": 1}
                    for n in ("cutoff", "resonance")
                ],
                "parameters": [
                    {
                        "target": {"name": f"filter-{name}", "parameter": p},
                        "unit": u,
                        "scope": "voice",
                        "minimum": lo,
                        "maximum": hi,
                        "default": d,
                    }
                    for p, u, lo, hi, d in [
                        ("cutoff_hz", "hz", 0.1, 60000, 1),
                        ("q", "ratio", 0.01, 10, 1),
                    ]
                ],
                "routes": [
                    {
                        "name": n,
                        "source": n,
                        "target": {"name": f"filter-{name}", "parameter": p},
                        "unit": u,
                        "operation": op,
                        "points": [
                            {"input": 0, "amount": lo},
                            {"input": 1, "amount": hi},
                        ],
                    }
                    for n, p, u, op, lo, hi in [
                        ("cutoff", "cutoff_hz", "hz", "add", -0.5, 30000),
                        ("resonance", "q", "ratio", "multiply", 0.1, 2.5),
                    ]
                ],
            }
            settings["bindings"].append({"name": "note-key", "kind": "key"})
            settings["modulation"]["sources"].append(
                {"name": "note-key", "scope": "voice", "minimum": 0, "maximum": 127}
            )
            settings["modulation"]["routes"].append(
                {
                    "name": "key-q",
                    "source": "note-key",
                    "target": {"name": f"filter-{name}", "parameter": "q"},
                    "unit": "ratio",
                    "operation": "multiply",
                    "points": [
                        {"input": 0, "amount": 0.5},
                        {"input": 127, "amount": 2},
                    ],
                }
            )
    document = instrument.SampleInstrumentScore.model_validate(raw)
    time = np.arange(30000) / 48000
    definition = sample_instrument.prepare(
        document,
        {
            "asset": np.column_stack(
                [
                    0.1 * np.sin(2 * np.pi * 440 * time),
                    0.2 * np.cos(2 * np.pi * 1130 * time),
                ]
            )
        },
    )
    events = [
        onset(t, n, pitch=440).model_copy(update={"controls": {}})
        for t, n in [(0, "first"), (4000, "second"), (12000, "third")]
    ]
    events.extend(
        [
            ControlChange(
                tick=12500, ordinal=0, control="expression", value=1, scope="instrument"
            ),
            *[
                ControlChange(
                    tick=t,
                    ordinal=0,
                    control="expression",
                    value=v,
                    scope="trigger",
                    part="main",
                    trigger_id="second",
                )
                for t, v in [(5500, 0), (16000, 1)]
            ],
            Release(tick=6000, ordinal=0, part="main", trigger_id="first"),
            Release(tick=18000, ordinal=0, part="main", trigger_id="second"),
        ]
    )
    actions = trace.prepare(document.body, events, seed=0).actions
    actions.append(
        VoiceRetirement(
            tick=8000,
            ordinal=0,
            voice_id=next(
                a.voice_id
                for a in actions
                if isinstance(a, trace.VoiceStart) and a.trigger_id == "first"
            ),
            cause=RetirementCause.choke,
            action="stop",
        )
    )
    expected = sample_instrument.OfflineSampler(definition).advance(actions, 0, 48000)
    source = sample_instrument.PersistentSampler(definition, voices=2)
    engine = live.NativeLiveEngine({"sample": source}, 48000)
    actual = np.empty((48000, 2))
    engine.advance_into(
        {"sample": [a for a in actions if a.tick < 13001]}, [], 0, 13001, actual[:13001]
    )
    saved = engine.snapshot()
    later = [a for a in actions if a.tick >= 13001]
    engine.advance_into({"sample": later}, [], 13001, 48000, actual[13001:])
    check_audio(tmp_path / "native-sample-filters.wav", actual, expected)
    assert not np.any(actual[42000:])
    assert not source.native_voices
    engine.restore(saved)
    replay = np.empty((34999, 2))
    for start in range(13001, 48000, 997):
        end = min(48000, start + 997)
        engine.advance_into(
            {"sample": [a for a in later if start <= a.tick < end]},
            [],
            start,
            end,
            replay[start - 13001 : end - 13001],
        )
    np.testing.assert_array_equal(replay, actual[13001:])
