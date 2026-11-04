from copy import deepcopy
from pathlib import Path

import numpy as np
import pytest
from test_dynamic_synth import onset
from test_envelfo_sample_hold import sample_hold_score
from test_lfo_instrument import lfo_settings
from test_sample_instrument import sample_score
from test_synth import check_audio
from ufor.events import EnvelfoChange, LFOChange, PerformanceEvent, Release
from ufor.samples import instrument, trace

from enge import live, sample_instrument


@pytest.mark.parametrize(
    "profile",
    [
        "voice",
        "part",
        "instrument",
        "envelope",
        "stages",
        "filters",
        "patch_lfo",
        "patch_stages",
    ],
)
def test_common_and_slot_generators_preserve_commands_seeds_and_continuation(
    tmp_path: Path, profile: str
) -> None:
    raw = sample_score().model_dump(mode="json")
    if profile.startswith("patch_"):
        settings = (
            sample_hold_score(profile.removeprefix("patch_"), 0.5)
            .body.voices[0]
            .model_dump(mode="json")
        )
    else:
        settings = lfo_settings(
            profile if profile in ("voice", "part", "instrument") else "voice"
        ).model_dump(mode="json")
        if profile in ("envelope", "stages"):
            envelope = {
                "kind": "envelope",
                "segments": [{"duration": "1/4 s", "to": 1}],
                "release": [{"duration": "1/4 s", "to": 0}],
            }
            if profile == "stages":
                envelope.pop("release")
            else:
                settings["modulation"]["sources"][0]["minimum"] = 0
                settings["modulation"]["routes"][0]["points"][0]["input"] = 0
            settings["envelfos"]["envelfo"] = (
                envelope
                if profile == "envelope"
                else {
                    "kind": "stages",
                    "initial_stage": "rise",
                    "stages": [{"name": "rise", "envelfo": envelope}],
                }
            )
    common = raw["body"]["settings"]
    slot = raw["body"]["slots"][0]
    for target in (common, slot):
        target.update(
            {n: deepcopy(settings[n]) for n in ("envelfos", "bindings", "modulation")}
        )
    if profile in ("voice", "part", "instrument"):
        slot["envelfos"]["envelfo"]["rate"] = "7"
        slot["envelfos"]["envelfo"]["scope"] = "voice"
        slot["modulation"]["sources"][0]["scope"] = "voice"
    if profile == "filters":
        for target in (common, slot):
            target["processing"] = {
                "filters": [{"name": "tone", "response": "lowpass", "cutoff_hz": 500}]
            }
            target["modulation"]["parameters"].append(
                {
                    "target": {"name": "filter-tone", "parameter": "cutoff_hz"},
                    "unit": "hz",
                    "scope": "voice",
                    "minimum": 20,
                    "maximum": 10000,
                    "default": 500,
                }
            )
            target["modulation"]["routes"].append(
                {
                    "name": "cutoff",
                    "source": "envelfo",
                    "target": {"name": "filter-tone", "parameter": "cutoff_hz"},
                    "unit": "hz",
                    "operation": "add",
                    "points": [
                        {"input": -1, "amount": 0},
                        {"input": 1, "amount": 2000},
                    ],
                }
            )
    common["envelope"] = {
        "segments": [{"duration": 0, "to": 1}],
        "release": [{"duration": "1/4 s", "to": 0}],
    }
    slot["playback"]["start_delay_seconds"] = 0.01001
    document = instrument.SampleInstrumentScore.model_validate(raw)
    audio = np.full((96000, 2), 0.125)
    if profile == "filters":
        audio *= np.sin(2 * np.pi * np.arange(96000)[:, None] * [440, 880] / 48000)
    definition = sample_instrument.prepare(document, {"asset": audio})
    events: list[PerformanceEvent] = [
        onset(0, pitch=440).model_copy(update={"controls": {}}),
        onset(7000, "other", "other", pitch=440).model_copy(update={"controls": {}}),
        Release(tick=18000, ordinal=0, part="main", trigger_id="note"),
        Release(tick=26000, ordinal=0, part="other", trigger_id="other"),
        onset(32000, pitch=440).model_copy(update={"controls": {}}),
    ]
    if profile in ("voice", "envelope", "stages"):
        events.extend(
            EnvelfoChange(
                tick=t,
                ordinal=0,
                name="envelfo",
                part="main",
                trigger_id="note",
                action=a,
            )
            for t, a in ((9000, "pause"), (11000, "resume"))
        )
    elif profile == "instrument":
        events.extend(
            [
                LFOChange(tick=9000, ordinal=0, name="envelfo", action="pause"),
                LFOChange(tick=11000, ordinal=0, name="envelfo", action="resume"),
            ]
        )
    actions = trace.prepare(document.body, events, seed=81).actions
    expected = sample_instrument.OfflineSampler(definition).advance(actions, 0, 48000)
    engine = live.NativeLiveEngine(
        {"sample": sample_instrument.PersistentSampler(definition, voices=2)}, 48000
    )
    actual = np.empty_like(expected)
    snapshots = []
    for start, end in (
        (0, 200),
        (200, 12000),
        (12000, 24000),
        (24000, 31000),
        (31000, 48000),
    ):
        engine.advance_into(
            {"sample": [a for a in actions if start <= a.tick < end]},
            [],
            start,
            end,
            actual[start:end],
        )
        if end in (200, 24000):
            snapshots.append((end, engine.snapshot()))
    check_audio(tmp_path / "native-sample-generators.wav", actual, expected)
    for frame, snapshot in snapshots:
        engine.restore(snapshot)
        replay = np.empty((48000 - frame, 2))
        for start in range(frame, 48000, 997):
            end = min(start + 997, 48000)
            engine.advance_into(
                {"sample": [a for a in actions if start <= a.tick < end]},
                [],
                start,
                end,
                replay[start - frame : end - frame],
            )
        np.testing.assert_array_equal(replay, actual[frame:])
