from pathlib import Path

import numpy as np
import pytest
from test_dynamic_synth import onset
from test_lfo_instrument import lfo_score
from test_synth import check_audio
from ufor import instrument_trace, motion_random, synth_trace
from ufor.synth import SynthInstrumentScore

from enge import synth


def threshold_score(target: str, probability: float = 1.0) -> SynthInstrumentScore:
    raw = lfo_score("synth").model_dump(mode="json")
    voice = raw["body"]["voices"][0]
    motions: dict[str, object] = {
        "clock": {"kind": "cycle", "shape": "square", "rate": "4", "phase": "1/2"},
        "edge": {"kind": "threshold", "input": "clock", "lower": -0.25, "upper": 0.25},
        "level": {"kind": "affine", "input": "edge", "scale": 0.5, "offset": 0.5},
    }
    events: list[dict[str, object]] = []
    output = "level"
    if target == "sample":
        motions["held"] = {"kind": "sample_hold", "minimum": 0.5, "maximum": 1}
        events = [
            {
                "source": "edge.rising",
                "target": "held",
                "action": "sample",
                "every": 2,
                "offset": 0,
                "probability": probability,
                "delay": "1/100",
            }
        ]
        motions["level"] = {
            "kind": "affine",
            "input": "held",
            "scale": 0.5,
            "offset": 0.5,
        }
    if target in ("cue", "export"):
        motions["gesture"] = {
            "kind": "stages",
            "initial_stage": "low",
            "stages": [
                {"name": "low", "motion": {"kind": "hold", "value": 0.5}},
                {"name": "high", "motion": {"kind": "hold", "value": 1}},
            ],
            "transitions": [
                {
                    "from": ["low", "high"],
                    "event": "cue.up",
                    "action": {"kind": "enter", "stage": "high"},
                },
                {
                    "from": ["low", "high"],
                    "event": "cue.down",
                    "action": {"kind": "enter", "stage": "low"},
                },
            ],
        }
        events = [
            {
                "source": "edge.rising",
                "target": "gesture",
                "action": "cue",
                "cue": "up",
            },
            {
                "source": "edge.falling",
                "target": "gesture",
                "action": "cue",
                "cue": "down",
            },
        ]
        motions["level"] = {
            "kind": "affine",
            "input": "gesture",
            "scale": 0.5,
            "offset": 0.5,
        }
    if target == "start":
        motions["attack"] = {
            "kind": "contour",
            "initial": 0,
            "start": "event",
            "segments": [{"duration": "1/16 s", "to": 1}],
        }
        motions["level"] = {
            "kind": "affine",
            "input": "attack",
            "scale": 0.5,
            "offset": 0.5,
        }
        events = [{"source": "edge.rising", "target": "attack"}]
    voice["motions"] = {
        "gesture": {
            "body": {
                "kind": "patch",
                "motions": motions,
                "outputs": {"value": output},
                "events": events,
            }
        }
    }
    voice["bindings"][0].update(reference="gesture", output="value")
    voice["modulation"]["sources"][0]["minimum"] = 0
    voice["modulation"]["routes"][0]["points"] = [
        {"input": 0, "amount": 0},
        {"input": 1, "amount": 1},
    ]
    if target == "export":
        receiver = motions.pop("gesture")
        motions["level"] = {
            "kind": "affine",
            "input": "edge",
            "scale": 0.5,
            "offset": 0.5,
        }
        patch = voice["motions"]["gesture"]["body"]
        patch.update(
            events=[], event_outputs={"up": "edge.rising", "down": "edge.falling"}
        )
        voice["motions"]["receiver"] = {"body": receiver}
        voice["bindings"].append(
            {"name": "receiver", "kind": "motion", "reference": "receiver"}
        )
        voice["modulation"]["sources"].append(
            {"name": "receiver", "scope": "voice", "minimum": -1, "maximum": 1}
        )
        voice["modulation"]["routes"][0].update(
            source="receiver",
            points=[{"input": -1, "amount": 0}, {"input": 1, "amount": 1}],
        )
        voice["event_connections"] = [
            {
                "source": voice["bindings"][0]["name"],
                "port": p,
                "destination": "receiver",
                "cue": p,
            }
            for p in ("up", "down")
        ]
    return SynthInstrumentScore.model_validate(raw)


@pytest.mark.parametrize(
    "target,probability,initial_high",
    [
        ("gate", 1.0, False),
        ("sample", 1.0, False),
        ("cue", 1.0, False),
        ("start", 1.0, False),
        ("export", 1.0, False),
        ("sample", 0.0, False),
        ("sample", 0.5, False),
        ("sample", 1.0, True),
    ],
)
def test_threshold_gate_and_next_frame_commands_survive_partitions_and_snapshots(
    tmp_path: Path, target: str, probability: float, initial_high: bool
) -> None:
    document = threshold_score(target, probability)
    if initial_high:
        raw = document.model_dump(mode="json")
        raw["body"]["voices"][0]["motions"]["gesture"]["body"]["motions"]["clock"][
            "phase"
        ] = "0"
        document = SynthInstrumentScore.model_validate(raw)
    events = [onset(0, pitch=0.1).model_copy(update={"controls": {}})]
    actions = synth_trace.prepare(document.body, events, seed=19).actions
    gate = np.tile(np.repeat([0.5, 1.0], 6000), 4)
    if target in ("cue", "export"):
        gate = np.concatenate(([0.75], (0.5 + 0.5 * gate)[:-1]))
    if target == "sample":
        start = next(a for a in actions if isinstance(a, instrument_trace.VoiceStart))
        state = start.motion_key ^ motion_random.stream_key(
            0, "patch-gesture-held:sample-hold"
        )
        state, word = motion_random.random_word(state)
        gate = np.full(48000, 0.75 + 0.25 * (word >> 11) / 2**53)
        gate_state = start.motion_key ^ motion_random.stream_key(
            0, "patch-gesture-held:start-0"
        )
        for at in range(12000 if initial_high else 6000, 48000, 24000):
            if probability == 0:
                continue
            if probability != 1:
                gate_state, gate_word = motion_random.random_word(gate_state)
                if gate_word >> 11 >= int(probability * 2**53):
                    continue
            state, word = motion_random.random_word(state)
            gate[at + 481 :] = 0.75 + 0.25 * (word >> 11) / 2**53
    if target == "start":
        gate = np.full(48000, 0.5)
        gate[6001:9001] = 0.5 + 0.5 * np.arange(3000) / 3000
        gate[9001:] = 1.0
    raw = document.model_dump(mode="json")
    raw["body"]["voices"][0].update(
        motions={}, bindings=[], modulation={}, event_connections=[]
    )
    baseline = SynthInstrumentScore.model_validate(raw)
    expected = (
        synth.OfflineSynth(synth.prepare(baseline)).advance(
            synth_trace.prepare(baseline.body, events, seed=19).actions, 0, 48000
        )
        * gate[:, None]
    )
    definition = synth.prepare(document)
    for backend in ("numpy", "native", "persistent"):
        renderer = (
            synth.PersistentSynth(definition, voices=1)
            if backend == "persistent"
            else synth.OfflineSynth(definition, backend)
        )
        parts = [renderer.advance(actions, 0, 6001)]
        snapshot = renderer.snapshot()
        tail = renderer.advance([], 6001, 6500)
        renderer.restore(snapshot)
        np.testing.assert_array_equal(renderer.advance([], 6001, 6500), tail)
        parts.append(tail)
        for start, end in ((6500, 12345), (12345, 48000)):
            parts.append(renderer.advance([], start, end))
        check_audio(
            tmp_path / f"threshold-{target}-{backend}.wav", np.vstack(parts), expected
        )


def test_threshold_hysteresis_resets_on_voice_slot_reuse(tmp_path: Path) -> None:
    raw = threshold_score("gate").model_dump(mode="json")
    children = raw["body"]["voices"][0]["motions"]["gesture"]["body"]["motions"]
    children["clock"].update(shape="sine", rate="2", phase="0")
    children["edge"].update(lower=-0.35, upper=0.45)
    document = SynthInstrumentScore.model_validate(raw)
    events = [
        onset(frame=t, trigger_id=n, pitch=0.1).model_copy(update={"controls": {}})
        for t, n in ((0, "first"), (30000, "second"))
    ]
    actions = synth_trace.prepare(document.body, events, seed=5).actions
    first = next(a for a in actions if isinstance(a, instrument_trace.VoiceStart))
    stop = instrument_trace.VoiceRetirement(
        tick=6000,
        ordinal=0,
        voice_id=first.voice_id,
        action="stop",
        cause="transport_stop",
    )
    actions.insert(next(i for i, a in enumerate(actions) if a.tick >= 30000), stop)
    raw["body"]["voices"][0].update(motions={}, bindings=[], modulation={})
    baseline = SynthInstrumentScore.model_validate(raw)
    base_actions = synth_trace.prepare(baseline.body, events, seed=5).actions
    base_actions.insert(
        next(i for i, a in enumerate(base_actions) if a.tick >= 30000), stop
    )
    expected = synth.OfflineSynth(synth.prepare(baseline)).advance(
        base_actions, 0, 48000
    )
    for start, end in ((0, 6000), (30000, 48000)):
        high = False
        values = np.empty(end - start)
        for i, value in enumerate(
            np.sin(2 * np.pi * 2 * np.arange(end - start) / 48000)
        ):
            if not high and value >= 0.45:
                high = True
            elif high and value <= -0.35:
                high = False
            values[i] = 1.0 if high else 0.5
        expected[start:end] *= values[:, None]
    definition = synth.prepare(document)
    for backend in ("numpy", "native", "persistent"):
        renderer = (
            synth.PersistentSynth(definition, voices=1)
            if backend == "persistent"
            else synth.OfflineSynth(definition, backend)
        )
        before = renderer.advance([a for a in actions if a.tick < 6000], 0, 6000)
        snapshot = renderer.snapshot()
        tail = renderer.advance([a for a in actions if a.tick >= 6000], 6000, 48000)
        renderer.restore(snapshot)
        np.testing.assert_array_equal(
            renderer.advance([a for a in actions if a.tick >= 6000], 6000, 48000), tail
        )
        check_audio(
            tmp_path / f"threshold-reuse-{backend}.wav",
            np.vstack((before, tail)),
            expected,
        )
