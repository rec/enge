from pathlib import Path

import numpy as np
import pytest
from test_dynamic_synth import onset
from test_lfo_instrument import lfo_score
from test_motion_threshold import threshold_score
from test_synth import check_audio
from ufor import instrument_trace, synth_trace
from ufor.control import TempoMap
from ufor.synth import SynthInstrumentScore

from enge import synth


def slew_score(
    rise: float, fall: float, initial: float | None, clock: str = "seconds"
) -> SynthInstrumentScore:
    raw = lfo_score("synth").model_dump(mode="json")
    voice = raw["body"]["voices"][0]
    voice["motions"] = {
        "gesture": {
            "clock": clock,
            "body": {
                "kind": "patch",
                "motions": {
                    "clock": {
                        "kind": "cycle",
                        "shape": "square",
                        "rate": "2",
                        "phase": "1/2",
                    },
                    "smooth": {
                        "kind": "slew",
                        "input": "clock",
                        "rise": rise,
                        "fall": fall,
                        "initial": initial,
                    },
                    "shared": {"kind": "sum", "inputs": ["smooth", "smooth"]},
                    "level": {
                        "kind": "affine",
                        "input": "shared",
                        "scale": 0.25,
                        "offset": 0.5,
                    },
                },
                "outputs": {"value": "level"},
            },
        }
    }
    voice["bindings"][0].update(reference="gesture", output="value")
    maximum = 0.5 + 0.5 * max(1.0, 1.0 if initial is None else initial)
    voice["modulation"]["sources"][0].update(minimum=0, maximum=maximum)
    voice["modulation"]["parameters"][0]["maximum"] = maximum
    voice["modulation"]["routes"][0]["points"] = [
        {"input": 0, "amount": 0},
        {"input": maximum, "amount": maximum},
    ]
    return SynthInstrumentScore.model_validate(raw)


@pytest.mark.parametrize(
    "rise,fall,initial,clock",
    [
        (8.0, 4.0, None, "seconds"),
        (8.0, 4.0, 0.5, "seconds"),
        (0.0, 0.0, 0.5, "seconds"),
        (0.0, 4.0, 0.5, "seconds"),
        (8.0, 0.0, None, "seconds"),
        (100000.0, 100000.0, None, "seconds"),
        (8.0, 4.0, 1.5, "seconds"),
        (2.0, 4.0, None, "beats"),
    ],
)
def test_slew_shared_state_limits_rates_and_restores_mid_ramp(
    tmp_path: Path, rise: float, fall: float, initial: float | None, clock: str
) -> None:
    document = slew_score(rise, fall, initial, clock)
    events = [onset(0, pitch=0.1).model_copy(update={"controls": {}})]
    actions = synth_trace.prepare(document.body, events, seed=5).actions
    tempo = (
        TempoMap.model_validate(
            {
                "points": [
                    {"at_seconds": "0", "beat": "0", "bpm": "120"},
                    {
                        "at_seconds": "1/8",
                        "beat": "1/4",
                        "bpm": "120",
                        "running": False,
                    },
                ]
            }
        )
        if clock == "beats"
        else None
    )
    inputs = (
        np.where(np.arange(48000) >= 6000, 1.0, -1.0)
        if clock == "beats"
        else np.tile(np.repeat([-1.0, 1.0], 12000), 2)
    )
    values = np.empty(48000)
    values[0] = inputs[0] if initial is None else initial
    for i in range(1, 48000):
        previous = values[i - 1]
        values[i] = (
            min(inputs[i], previous + rise / 48000)
            if inputs[i] >= previous
            else max(inputs[i], previous - fall / 48000)
        )
    if clock == "beats":
        assert values[24000] > values[12000]
    raw = document.model_dump(mode="json")
    raw["body"]["voices"][0].update(motions={}, bindings=[], modulation={})
    baseline = SynthInstrumentScore.model_validate(raw)
    expected = (
        synth.OfflineSynth(synth.prepare(baseline)).advance(
            synth_trace.prepare(baseline.body, events, seed=5).actions, 0, 48000
        )
        * (0.5 + 0.5 * values)[:, None]
    )
    definition = synth.prepare(document)
    for backend in ("numpy", "native", "persistent"):
        renderer = (
            synth.PersistentSynth(definition, voices=1, tempo_map=tempo)
            if backend == "persistent"
            else synth.OfflineSynth(definition, backend, tempo_map=tempo)
        )
        first = renderer.advance(actions, 0, 14003)
        snapshot = renderer.snapshot()
        middle = renderer.advance([], 14003, 17317)
        renderer.restore(snapshot)
        np.testing.assert_array_equal(renderer.advance([], 14003, 17317), middle)
        tail = renderer.advance([], 17317, 48000)
        check_audio(
            tmp_path / f"slew-{clock}-{rise}-{fall}-{initial}-{backend}.wav",
            np.vstack((first, middle, tail)),
            expected,
        )


def test_slew_initializes_fresh_when_a_voice_slot_is_reused(tmp_path: Path) -> None:
    document = slew_score(8, 4, 0.5)
    events = [
        onset(t, trigger_id=n, pitch=0.1).model_copy(update={"controls": {}})
        for t, n in ((0, "first"), (30000, "second"))
    ]
    actions = synth_trace.prepare(document.body, events, seed=5).actions
    first = next(a for a in actions if isinstance(a, instrument_trace.VoiceStart))
    stop = instrument_trace.VoiceRetirement(
        tick=16000,
        ordinal=0,
        voice_id=first.voice_id,
        action="stop",
        cause="transport_stop",
    )
    actions.insert(next(i for i, a in enumerate(actions) if a.tick >= 30000), stop)
    definition = synth.prepare(document)
    reference = synth.OfflineSynth(definition).advance(actions, 0, 48000)
    assert reference[30000, 0] == pytest.approx(0.75)
    native = synth.PersistentSynth(definition, voices=1)
    before = native.advance([a for a in actions if a.tick < 16000], 0, 16000)
    snapshot = native.snapshot()
    tail = native.advance([a for a in actions if a.tick >= 16000], 16000, 48000)
    native.restore(snapshot)
    np.testing.assert_array_equal(
        native.advance([a for a in actions if a.tick >= 16000], 16000, 48000), tail
    )
    check_audio(tmp_path / "slew-reuse.wav", np.vstack((before, tail)), reference)


def test_slew_drives_threshold_cues_on_the_following_frame(tmp_path: Path) -> None:
    raw = threshold_score("cue").model_dump(mode="json")
    children = raw["body"]["voices"][0]["motions"]["gesture"]["body"]["motions"]
    children["smooth"] = {"kind": "slew", "input": "clock", "rise": 16, "fall": 8}
    children["edge"].update(input="smooth", lower=-0.3113, upper=0.2313)
    document = SynthInstrumentScore.model_validate(raw)
    events = [onset(0, pitch=0.1).model_copy(update={"controls": {}})]
    actions = synth_trace.prepare(document.body, events, seed=5).actions
    levels = np.empty(48000)
    smooth = -1.0
    high = False
    for i, value in enumerate(np.tile(np.repeat([-1.0, 1.0], 6000), 4)):
        levels[i] = 1.0 if high else 0.75
        if i:
            smooth = (
                min(value, smooth + 16 / 48000)
                if value >= smooth
                else max(value, smooth - 8 / 48000)
            )
        if not high and smooth >= 0.2313:
            high = True
        elif high and smooth <= -0.3113:
            high = False
    raw["body"]["voices"][0].update(motions={}, bindings=[], modulation={})
    baseline = SynthInstrumentScore.model_validate(raw)
    expected = (
        synth.OfflineSynth(synth.prepare(baseline)).advance(
            synth_trace.prepare(baseline.body, events, seed=5).actions, 0, 48000
        )
        * levels[:, None]
    )
    # Stop after detection, before the following frame delivers its stage cue.
    split = int(np.flatnonzero(np.diff(levels))[0] + 1)
    definition = synth.prepare(document)
    for backend in ("numpy", "native", "persistent"):
        renderer = (
            synth.PersistentSynth(definition, voices=1)
            if backend == "persistent"
            else synth.OfflineSynth(definition, backend)
        )
        before = renderer.advance(actions, 0, split)
        snapshot = renderer.snapshot()
        tail = renderer.advance([], split, 48000)
        renderer.restore(snapshot)
        np.testing.assert_array_equal(renderer.advance([], split, 48000), tail)
        check_audio(
            tmp_path / f"slew-threshold-{backend}.wav",
            np.vstack((before, tail)),
            expected,
        )
