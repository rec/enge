from math import ceil
from pathlib import Path

import numpy as np
import pytest
from test_dynamic_synth import onset
from test_motion_sample_hold import sample_hold_score
from test_synth import check_audio
from ufor import instrument_trace, motion_random, synth_trace
from ufor.synth import SynthInstrumentScore

from enge import synth


def latch_score(source_kind: str, probability: float) -> SynthInstrumentScore:
    raw = sample_hold_score(
        "stages" if source_kind == "stages" else "cycle", probability
    ).model_dump(mode="json")
    patch = raw["body"]["voices"][0]["motions"]["gesture"]["body"]
    patch["motions"].update(
        {
            "wave": {"kind": "cycle", "shape": "triangle", "rate": "2"},
            "pressure": {
                "kind": "affine",
                "input": "wave",
                "scale": 0.5,
                "offset": 0.5,
            },
            "left": {"kind": "latch", "input": "pressure"},
            "right": {"kind": "latch", "input": "pressure"},
        }
    )
    patch["motions"].pop("product")
    patch["motions"]["sum"] = {"kind": "sum", "inputs": ["left", "right"]}
    patch["motions"]["level"].update(input="sum", scale=0.25)
    # A fractional-frame delay distinguishes emission, delivery, and observation.
    patch["events"][0]["delay"] = "12001/192000"
    for connection in patch["events"]:
        connection["action"] = "capture"
    if source_kind == "threshold":
        patch["motions"]["clock"].update(shape="square", phase="1/2")
        patch["motions"]["edge"] = {
            "kind": "threshold",
            "input": "clock",
            "lower": -0.25,
            "upper": 0.25,
        }
        for connection in patch["events"]:
            connection["source"] = "edge.rising"
    return SynthInstrumentScore.model_validate(raw)


@pytest.mark.parametrize("backend", ["numpy", "native", "persistent"])
@pytest.mark.parametrize("source_kind", ["cycle", "stages", "threshold"])
@pytest.mark.parametrize("probability", [0.0, 0.5, 1.0])
def test_latch_captures_delivery_frame_with_independent_gates_and_snapshot(
    tmp_path: Path, backend: str, source_kind: str, probability: float
) -> None:
    document = latch_score(source_kind, probability)
    events = [onset(0, pitch=0.1).model_copy(update={"controls": {}})]
    actions = synth_trace.prepare(document.body, events, seed=81).actions
    start = next(a for a in actions if isinstance(a, instrument_trace.VoiceStart))
    phase = (np.arange(48000) % 24000) / 24000
    input_values = np.where(phase < 0.5, 2 * phase, 2 - 2 * phase)
    held_values: list[np.ndarray] = []
    for order, child in enumerate(("left", "right")):
        held = np.full(48000, input_values[0])
        state = start.motion_key ^ motion_random.stream_key(
            0, f"gesture:capture-{child}-{order}"
        )
        every, offset, delay = (2, 1, 3000.25) if order == 0 else (3, 0, 1500)
        cutoff = int((probability if order == 0 else 1) * 2**53)
        first = 3000 if source_kind == "threshold" else 1500
        for i, at in enumerate(range(first, 48000, 6000)):
            if i < offset or (i - offset) % every or cutoff == 0:
                continue
            if cutoff != 2**53:
                state, word = motion_random.random_word(state)
                if word >> 11 >= cutoff:
                    continue
            frame = ceil(at + delay + (1 if source_kind == "threshold" else 0))
            if frame < 48000:
                held[frame:] = input_values[frame]
        held_values.append(held)
    signal = 0.5 + 0.25 * (held_values[0] + held_values[1])
    raw = document.model_dump(mode="json")
    raw["body"]["voices"][0].update(motions={}, bindings=[], modulation={})
    baseline = SynthInstrumentScore.model_validate(raw)
    expected = (
        synth.OfflineSynth(synth.prepare(baseline)).advance(
            synth_trace.prepare(baseline.body, events, seed=81).actions, 0, 48000
        )
        * signal[:, None]
    )
    definition = synth.prepare(document)
    renderer = (
        synth.PersistentSynth(definition, voices=1)
        if backend == "persistent"
        else synth.OfflineSynth(definition, backend)
    )
    # Snapshot after emission, with delayed captures still queued.
    first_audio = renderer.advance(actions, 0, 9000)
    snapshot = renderer.snapshot()
    middle = renderer.advance([], 9000, 13003)
    renderer.restore(snapshot)
    np.testing.assert_array_equal(renderer.advance([], 9000, 13003), middle)
    tail = renderer.advance([], 13003, 48000)
    check_audio(
        tmp_path / f"latch-{source_kind}-{probability}-{backend}.wav",
        np.vstack((first_audio, middle, tail)),
        expected,
    )


@pytest.mark.parametrize("backend", ["numpy", "native", "persistent"])
@pytest.mark.parametrize("mode", ["unconnected", "connected", "slew"])
def test_latch_activation_and_same_frame_captures(
    tmp_path: Path, backend: str, mode: str
) -> None:
    raw = latch_score("cycle", 1).model_dump(mode="json")
    patch = raw["body"]["voices"][0]["motions"]["gesture"]["body"]
    if mode != "unconnected":
        for connection in patch["events"]:
            connection.update(every=1, offset=0, delay="0")
        # Repeated commands at one timestamp must not advance the input twice.
        patch["events"].append(dict(patch["events"][0]))
    else:
        patch["events"] = []
    if mode == "slew":
        patch["motions"]["smooth"] = {
            "kind": "slew",
            "input": "wave",
            "rise": 2,
            "fall": 4,
        }
        patch["motions"]["pressure"]["input"] = "smooth"
    document = SynthInstrumentScore.model_validate(raw)
    events = [onset(0, pitch=0.1).model_copy(update={"controls": {}})]
    actions = synth_trace.prepare(document.body, events, seed=0).actions
    phase = (np.arange(48000) % 24000) / 24000
    inputs = np.where(phase < 0.5, 2 * phase, 2 - 2 * phase)
    if mode == "slew":
        wave = 2 * inputs - 1
        smooth = np.empty(48000)
        smooth[0] = wave[0]
        for i in range(1, 48000):
            smooth[i] = (
                min(wave[i], smooth[i - 1] + 2 / 48000)
                if wave[i] >= smooth[i - 1]
                else max(wave[i], smooth[i - 1] - 4 / 48000)
            )
        inputs = 0.5 + 0.5 * smooth
    held = np.full(48000, inputs[0])
    if mode != "unconnected":
        for frame in range(1500, 48000, 6000):
            held[frame:] = inputs[frame]
    raw["body"]["voices"][0].update(motions={}, bindings=[], modulation={})
    baseline = SynthInstrumentScore.model_validate(raw)
    expected = (
        synth.OfflineSynth(synth.prepare(baseline)).advance(
            synth_trace.prepare(baseline.body, events, seed=0).actions, 0, 48000
        )
        * (0.5 + 0.5 * held)[:, None]
    )
    definition = synth.prepare(document)
    renderer = (
        synth.PersistentSynth(definition, voices=1)
        if backend == "persistent"
        else synth.OfflineSynth(definition, backend)
    )
    check_audio(
        tmp_path / f"latch-activation-{mode}-{backend}.wav",
        renderer.advance(actions, 0, 48000),
        expected,
    )


def test_latch_discards_pending_captures_and_initializes_reused_voice_slot(
    tmp_path: Path,
) -> None:
    raw = latch_score("cycle", 1).model_dump(mode="json")
    patch = raw["body"]["voices"][0]["motions"]["gesture"]["body"]
    patch["motions"]["wave"]["phase"] = "1/4"
    for connection in patch["events"]:
        connection.update(every=1, offset=0, delay="1/8")
    document = SynthInstrumentScore.model_validate(raw)
    events = [
        onset(t, trigger_id=n, pitch=0.1).model_copy(update={"controls": {}})
        for t, n in ((0, "first"), (16000, "second"))
    ]
    actions = synth_trace.prepare(document.body, events, seed=0).actions
    first = next(a for a in actions if isinstance(a, instrument_trace.VoiceStart))
    stop = instrument_trace.VoiceRetirement(
        tick=16000,
        ordinal=0,
        voice_id=first.voice_id,
        action="stop",
        cause="transport_stop",
    )
    actions.insert(next(i for i, a in enumerate(actions) if a.tick >= 16000), stop)
    definition = synth.prepare(document)
    reference = synth.OfflineSynth(definition).advance(actions, 0, 48000)
    assert reference[16000, 0] == pytest.approx(0.75)
    assert reference[19500, 0] == pytest.approx(0.75)
    native = synth.PersistentSynth(definition, voices=1)
    before = native.advance([a for a in actions if a.tick < 16000], 0, 16000)
    snapshot = native.snapshot()
    tail_actions = [a for a in actions if a.tick >= 16000]
    tail = native.advance(tail_actions, 16000, 48000)
    native.restore(snapshot)
    np.testing.assert_array_equal(native.advance(tail_actions, 16000, 48000), tail)
    check_audio(
        tmp_path / "latch-voice-reuse.wav", np.vstack((before, tail)), reference
    )
