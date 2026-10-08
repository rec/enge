from pathlib import Path

import numpy as np
import pytest
import test_fm
import test_noise
from test_dynamic_synth import onset
from test_lfo_instrument import lfo_score
from test_synth import check_audio
from ufor import instrument_trace, motion_random, synth_trace
from ufor.control import TempoMap
from ufor.events import Release
from ufor.synth import SynthInstrumentScore

from enge import fm, noise, synth


def sample_hold_score(source_kind: str, probability: float) -> SynthInstrumentScore:
    raw = lfo_score("synth").model_dump(mode="json")
    voice = raw["body"]["voices"][0]
    clock = {
        "kind": "cycle",
        "rate": "8",
        "markers": [{"name": "pulse", "position": "1/4"}],
    }
    if source_kind == "stages":
        clock = {
            "kind": "stages",
            "initial_stage": "clock",
            "stages": [{"name": "clock", "motion": clock}],
        }
    voice["motions"] = {
        "gesture": {
            "body": {
                "kind": "patch",
                "motions": {
                    "clock": clock,
                    "left": {"kind": "sample_hold", "minimum": 0, "maximum": 1},
                    "right": {"kind": "sample_hold", "minimum": 0, "maximum": 1},
                    "product": {"kind": "product", "inputs": ["left", "right"]},
                    "level": {
                        "kind": "affine",
                        "input": "product",
                        "scale": 0.5,
                        "offset": 0.5,
                    },
                },
                "outputs": {"level": "level"},
                "events": [
                    {
                        "source": "clock.pulse",
                        "target": "left",
                        "action": "sample",
                        "every": 2,
                        "offset": 1,
                        "probability": probability,
                        "delay": "1/16",
                    },
                    {
                        "source": "clock.pulse",
                        "target": "right",
                        "action": "sample",
                        "every": 3,
                        "delay": "1/32",
                    },
                ],
            }
        }
    }
    voice["bindings"][0].update(reference="gesture", output="level")
    voice["modulation"]["sources"][0]["minimum"] = 0
    voice["modulation"]["routes"][0]["points"] = [
        {"input": 0, "amount": 0},
        {"input": 1, "amount": 1},
    ]
    return SynthInstrumentScore.model_validate(raw)


@pytest.mark.parametrize("source_kind", ["cycle", "stages"])
@pytest.mark.parametrize("probability", [0.0, 0.5, 1.0])
def test_sample_hold_events_are_seeded_independent_delayed_and_snapshot_safe(
    tmp_path: Path, source_kind: str, probability: float
) -> None:
    document = sample_hold_score(source_kind, probability)
    events = [onset(0, pitch=0.1).model_copy(update={"controls": {}})]
    actions = synth_trace.prepare(document.body, events, seed=81).actions
    start = next(a for a in actions if isinstance(a, instrument_trace.VoiceStart))
    values: list[np.ndarray] = []
    for order, child in enumerate(("left", "right")):
        name = f"patch-gesture-{child}"
        random_state = start.motion_key ^ motion_random.stream_key(
            0, name + ":sample-hold"
        )
        random_state, word = motion_random.random_word(random_state)
        held = np.full(48000, (word >> 11) / 2**53)
        gate_state = start.motion_key ^ motion_random.stream_key(
            0, name + f":start-{order}"
        )
        every, offset, delay = (2, 1, 3000) if order == 0 else (3, 0, 1500)
        cutoff = int((probability if order == 0 else 1) * 2**53)
        for i, at in enumerate(range(1500, 48000, 6000)):
            if i < offset or (i - offset) % every:
                continue
            if cutoff == 0:
                continue
            if cutoff != 2**53:
                gate_state, gate_word = motion_random.random_word(gate_state)
                if gate_word >> 11 >= cutoff:
                    continue
            random_state, word = motion_random.random_word(random_state)
            held[at + delay :] = (word >> 11) / 2**53
        values.append(held)
    assert values[0][0] != values[1][0]
    signal = 0.5 + 0.5 * values[0] * values[1]
    baseline_raw = document.model_dump(mode="json")
    baseline_raw["body"]["voices"][0].update(motions={}, bindings=[], modulation={})
    baseline = SynthInstrumentScore.model_validate(baseline_raw)
    baseline_actions = synth_trace.prepare(baseline.body, events, seed=81).actions
    expected = (
        synth.OfflineSynth(synth.prepare(baseline)).advance(baseline_actions, 0, 48000)
        * signal[:, None]
    )
    definition = synth.prepare(document)
    # Stop after emission but before delivery of a queued update.
    split = 9000
    for backend in ("numpy", "native", "persistent"):
        renderer = (
            synth.PersistentSynth(definition, voices=2)
            if backend == "persistent"
            else synth.OfflineSynth(definition, backend)
        )
        first = renderer.advance(actions, 0, split)
        snapshot = renderer.snapshot()
        second = renderer.advance([], split, 48000)
        renderer.restore(snapshot)
        np.testing.assert_array_equal(renderer.advance([], split, 48000), second)
        check_audio(
            tmp_path / f"sample-hold-{source_kind}-{probability}-{backend}.wav",
            np.vstack((first, second)),
            expected,
        )


@pytest.mark.parametrize("clock", ["seconds", "beats"])
def test_sample_hold_activation_is_seeded_without_any_connections(
    tmp_path: Path,
    clock: str,
) -> None:
    raw = lfo_score("synth").model_dump(mode="json")
    voice = raw["body"]["voices"][0]
    voice["motions"] = {"held": {"clock": clock, "body": {"kind": "sample_hold"}}}
    voice["bindings"][0]["reference"] = "held"
    voice["modulation"]["routes"][0]["points"] = [
        {"input": -1, "amount": 0},
        {"input": 1, "amount": 1},
    ]
    document = SynthInstrumentScore.model_validate(raw)
    tempo = (
        TempoMap.model_validate(
            {
                "points": [
                    {"at_seconds": "0", "beat": "0", "bpm": "120", "running": False}
                ]
            }
        )
        if clock == "beats"
        else None
    )
    events = [onset(0, pitch=0.1).model_copy(update={"controls": {}})]
    baseline_raw = document.model_dump(mode="json")
    baseline_raw["body"]["voices"][0].update(motions={}, bindings=[], modulation={})
    baseline = SynthInstrumentScore.model_validate(baseline_raw)
    baseline_actions = synth_trace.prepare(baseline.body, events, seed=1).actions
    base = synth.OfflineSynth(synth.prepare(baseline)).advance(
        baseline_actions, 0, 48000
    )
    levels: list[float] = []
    for seed in (1, 2):
        actions = synth_trace.prepare(document.body, events, seed=seed).actions
        start = next(a for a in actions if isinstance(a, instrument_trace.VoiceStart))
        _, word = motion_random.random_word(
            start.motion_key ^ motion_random.stream_key(0, "held:sample-hold")
        )
        level = (word >> 11) / 2**53
        levels.append(level)
        commands = [
            instrument_trace.MotionObservation(
                tick=12000,
                ordinal=0,
                name="held",
                part="main",
                trigger_id="note",
                action="seek",
                position=0.25,
            ),
            instrument_trace.MotionObservation(
                tick=24000,
                ordinal=0,
                name="held",
                part="main",
                trigger_id="note",
                action="reverse",
            ),
        ]
        for backend in ("numpy", "native", "persistent"):
            renderer = (
                synth.PersistentSynth(
                    synth.prepare(document), voices=1, tempo_map=tempo
                )
                if backend == "persistent"
                else synth.OfflineSynth(
                    synth.prepare(document), backend, tempo_map=tempo
                )
            )
            actual = renderer.advance([*actions, *commands], 0, 48000)
            check_audio(
                tmp_path / f"held-activation-{seed}-{backend}.wav", actual, base * level
            )
    assert levels[0] != levels[1]


def test_sample_hold_reinitializes_when_a_native_voice_slot_is_reused(
    tmp_path: Path,
) -> None:
    raw = lfo_score("synth").model_dump(mode="json")
    voice = raw["body"]["voices"][0]
    voice["motions"] = {"held": {"body": {"kind": "sample_hold"}}}
    voice["bindings"][0]["reference"] = "held"
    voice["modulation"]["routes"][0]["points"] = [
        {"input": -1, "amount": 0},
        {"input": 1, "amount": 1},
    ]
    document = SynthInstrumentScore.model_validate(raw)
    events = [
        onset(0, trigger_id="first", pitch=0.1).model_copy(update={"controls": {}}),
        Release(tick=12000, ordinal=0, part="main", trigger_id="first"),
        onset(30000, trigger_id="second", pitch=0.1).model_copy(
            update={"controls": {}}
        ),
    ]
    actions = synth_trace.prepare(document.body, events, seed=9).actions
    starts = [a for a in actions if isinstance(a, instrument_trace.VoiceStart)]
    levels = []
    for start in starts:
        _, word = motion_random.random_word(
            start.motion_key ^ motion_random.stream_key(0, "held:sample-hold")
        )
        levels.append((word >> 11) / 2**53)
    assert levels[0] != levels[1]
    baseline_raw = document.model_dump(mode="json")
    baseline_raw["body"]["voices"][0].update(motions={}, bindings=[], modulation={})
    baseline = SynthInstrumentScore.model_validate(baseline_raw)
    base_actions = synth_trace.prepare(baseline.body, events, seed=9).actions
    expected = synth.OfflineSynth(synth.prepare(baseline)).advance(
        base_actions, 0, 48000
    )
    expected[:30000] *= levels[0]
    expected[30000:] *= levels[1]
    definition = synth.prepare(document)
    for backend in ("numpy", "native", "persistent"):
        renderer = (
            synth.PersistentSynth(definition, voices=1)
            if backend == "persistent"
            else synth.OfflineSynth(definition, backend)
        )
        first = renderer.advance([a for a in actions if a.tick < 30000], 0, 30000)
        snapshot = renderer.snapshot()
        second = renderer.advance([a for a in actions if a.tick >= 30000], 30000, 48000)
        renderer.restore(snapshot)
        np.testing.assert_array_equal(
            renderer.advance([a for a in actions if a.tick >= 30000], 30000, 48000),
            second,
        )
        check_audio(
            tmp_path / f"held-slot-reuse-{backend}.wav",
            np.vstack((first, second)),
            expected,
        )


def test_unused_sample_hold_does_not_consume_a_native_action_row(
    tmp_path: Path,
) -> None:
    raw = lfo_score("synth").model_dump(mode="json")
    voice = raw["body"]["voices"][0]
    voice.update(
        motions={"unused": {"body": {"kind": "sample_hold"}}},
        bindings=[],
        modulation={},
    )
    document = SynthInstrumentScore.model_validate(raw)
    actions = synth_trace.prepare(
        document.body, [onset(0, pitch=0.1).model_copy(update={"controls": {}})], seed=1
    ).actions
    definition = synth.prepare(document)
    expected = synth.OfflineSynth(definition).advance(actions, 0, 48000)
    actual = synth.PersistentSynth(definition, voices=1, action_capacity=1).advance(
        actions, 0, 48000
    )
    check_audio(tmp_path / "unused-held.wav", actual, expected)


@pytest.mark.parametrize("kind", ["fm", "noise"])
def test_sample_hold_is_shared_by_reference_and_persistent_fm_and_noise(
    tmp_path: Path, kind: str
) -> None:
    raw = (test_fm.score() if kind == "fm" else test_noise.score()).model_dump(
        mode="json"
    )
    voice = raw["body"]["voices"][0]
    voice.update(
        motions={"held": {"body": {"kind": "sample_hold"}}},
        bindings=[{"name": "held", "kind": "motion", "reference": "held"}],
        modulation={
            "sources": [
                {"name": "held", "scope": "voice", "minimum": -1, "maximum": 1}
            ],
            "parameters": [
                {
                    "target": {"name": "processing", "parameter": "amplitude"},
                    "unit": "ratio",
                    "scope": "voice",
                    "default": 1,
                    "minimum": 0,
                    "maximum": 1,
                }
            ],
            "routes": [
                {
                    "name": "held",
                    "source": "held",
                    "target": {"name": "processing", "parameter": "amplitude"},
                    "unit": "ratio",
                    "operation": "multiply",
                    "points": [{"input": -1, "amount": 0}, {"input": 1, "amount": 1}],
                }
            ],
        },
    )
    document = SynthInstrumentScore.model_validate(raw)
    events = [test_fm.trigger() if kind == "fm" else test_noise.trigger()]
    actions = synth_trace.prepare(document.body, events, seed=31).actions
    start = next(a for a in actions if isinstance(a, instrument_trace.VoiceStart))
    _, word = motion_random.random_word(
        start.motion_key ^ motion_random.stream_key(0, "held:sample-hold")
    )
    level = (word >> 11) / 2**53
    baseline_raw = document.model_dump(mode="json")
    baseline_raw["body"]["voices"][0].update(motions={}, bindings=[], modulation={})
    baseline = SynthInstrumentScore.model_validate(baseline_raw)
    base_actions = synth_trace.prepare(baseline.body, events, seed=31).actions
    if kind == "fm":
        expected = (
            fm.OfflineFM(fm.prepare(baseline)).advance(base_actions, 0, 48000) * level
        )
        reference = fm.OfflineFM(fm.prepare(document)).advance(actions, 0, 48000)
        native = fm.PersistentFM(fm.prepare(document), voices=1).advance(
            actions, 0, 48000
        )
    else:
        expected = (
            noise.OfflineNoise(noise.prepare(baseline)).advance(base_actions, 0, 48000)
            * level
        )
        reference = noise.OfflineNoise(noise.prepare(document)).advance(
            actions, 0, 48000
        )
        native = noise.PersistentNoise(noise.prepare(document), voices=1).advance(
            actions, 0, 48000
        )
    check_audio(tmp_path / f"held-{kind}-reference.wav", reference, expected)
    check_audio(tmp_path / f"held-{kind}-persistent.wav", native, expected)
