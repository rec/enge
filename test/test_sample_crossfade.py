from math import cos, pi, sin
from pathlib import Path

import numpy as np
import pytest
from test_dynamic_synth import onset
from test_sample_instrument import sample_score
from test_synth import check_audio
from ufor.samples import instrument, trace

from enge import live, sample_instrument


@pytest.mark.parametrize("curve", ["linear", "equal_power"])
@pytest.mark.parametrize("position", [0.0, 0.5, 1.0])
def test_latched_layer_weights_preserve_silent_voices_and_sum_without_normalizing(
    tmp_path: Path, curve: str, position: float
) -> None:
    raw = sample_score().model_dump(mode="json")
    slot = raw["body"]["slots"][0]
    raw["body"]["slots"] = [
        slot
        | {
            "name": f"layer-{i}",
            "crossfades": [
                {
                    "input": "key",
                    "start": 0,
                    "end": 100,
                    "direction": direction,
                    "curve": curve,
                },
                {
                    "input": "velocity",
                    "start": 0,
                    "end": 1,
                    "direction": direction,
                    "curve": curve,
                },
            ],
        }
        for i, direction in enumerate(["in", "out", "in"])
    ]
    document = instrument.SampleInstrumentScore.model_validate(raw)
    definition = sample_instrument.prepare(
        document, {"asset": np.tile([0.1, 0.2], (96000, 1))}
    )
    actions = trace.prepare(
        document.body,
        [
            onset(pitch=440).model_copy(
                update={
                    "controls": {},
                    "key": round(position * 100),
                    "velocity": position,
                }
            ),
            onset(24000, "second", pitch=440).model_copy(
                update={
                    "controls": {},
                    "key": round((1 - position) * 100),
                    "velocity": 1 - position,
                }
            ),
        ],
        seed=0,
    ).actions
    if curve == "linear" or position in (0.0, 1.0):
        incoming, outgoing = position, 1 - position
    else:
        incoming, outgoing = sin(pi * position / 2), cos(pi * position / 2)
    gain = 2 * incoming**2 + outgoing**2
    expected = np.tile([0.1 * gain, 0.225 * gain], (48000, 1))
    later_gain = 2 * outgoing**2 + incoming**2
    expected[24000:] += [0.1 * later_gain, 0.225 * later_gain]
    early = [a for a in actions if a.tick < 17003]
    later = [a for a in actions if a.tick >= 17003]
    for backend in ("numpy", "native"):
        renderer = sample_instrument.OfflineSampler(definition, backend)
        first = renderer.advance(early, 0, 17003)
        saved = renderer.snapshot()
        assert len(saved.voices) == 3
        restored = sample_instrument.OfflineSampler(definition, backend)
        restored.restore(saved)
        tail = renderer.advance(later, 17003, 48000)
        check_audio(
            tmp_path / f"layers-{backend}.wav", np.concatenate([first, tail]), expected
        )
        np.testing.assert_array_equal(restored.advance(later, 17003, 48000), tail)
    source = sample_instrument.PersistentSampler(definition)
    engine = live.NativeLiveEngine({"sample": source}, 31000)
    actual = np.empty((48000, 2))
    engine.advance_into({"sample": early}, [], 0, 17003, actual[:17003])
    saved_live = engine.snapshot()
    engine.advance_into({"sample": later}, [], 17003, 48000, actual[17003:])
    check_audio(tmp_path / "layers-live.wav", actual, expected)
    engine.restore(saved_live)
    replay = np.empty((30997, 2))
    engine.advance_into({"sample": later}, [], 17003, 48000, replay)
    np.testing.assert_array_equal(replay, actual[17003:])
