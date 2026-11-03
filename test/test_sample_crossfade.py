from math import cos, pi, sin
from pathlib import Path

import numpy as np
import pytest
from test_dynamic_synth import onset
from test_sample_instrument import sample_score
from test_synth import check_audio
from ufor.events import ControlChange
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


@pytest.mark.parametrize("curve", ["linear", "equal_power"])
@pytest.mark.parametrize("scope", ["instrument", "part", "trigger"])
def test_live_layer_weights_smooth_before_mapping_and_restore(
    tmp_path: Path, curve: str, scope: str
) -> None:
    raw = sample_score().model_dump(mode="json")
    raw["body"]["settings"]["controls"] = {"expression": {"default": 0.25}}
    slot = raw["body"]["slots"][0]
    raw["body"]["slots"] = [
        slot
        | {
            "name": f"layer-{i}",
            "crossfades": [
                {
                    "input": "control",
                    "control": "expression",
                    "scope": scope,
                    "start": 0.25,
                    "end": 0.75,
                    "direction": d,
                    "curve": curve,
                    "smoothing_seconds": 0.125,
                },
                {
                    "input": "control",
                    "control": "expression",
                    "scope": scope,
                    "start": 0,
                    "end": 1,
                    "direction": "out" if d == "in" else "in",
                    "curve": curve,
                    "smoothing_seconds": 0.0625,
                },
            ],
        }
        for i, d in enumerate(["in", "out", "in"])
    ]
    document = instrument.SampleInstrumentScore.model_validate(raw)
    definition = sample_instrument.prepare(
        document, {"asset": np.tile([0.1, 0.2], (96000, 1))}
    )
    events = [
        onset(2000, pitch=440).model_copy(update={"controls": {"expression": 0.25}}),
        *[
            ControlChange(
                tick=t,
                ordinal=0,
                control="expression",
                value=v,
                scope=scope,
                part=None if scope == "instrument" else "main",
                trigger_id="note" if scope == "trigger" else None,
            )
            for t, v in [(1000, 1), (4000, 0), (14000, 1)]
        ],
    ]
    actions = trace.prepare(document.body, events, seed=0).actions
    weights = []
    for duration, low, high in [(6000, 0.25, 0.75), (3000, 0, 1)]:
        values = np.full(48000, 0.25)
        for at, target in [(1000, 1), (4000, 0), (14000, 1)]:
            if at < 2000 and scope == "trigger":
                continue
            previous = values[at]
            values[at:] = previous + (target - previous) * np.minimum(
                1, np.arange(48000 - at) / duration
            )
        position = np.clip((values - low) / (high - low), 0, 1)
        incoming, outgoing = position, 1 - position
        if curve == "equal_power":
            incoming = np.where(position == 1, 1, np.sin(pi * position / 2))
            outgoing = np.where(position == 1, 0, np.cos(pi * position / 2))
        weights.append((incoming, outgoing))
    gain = 2 * weights[0][0] * weights[1][1] + weights[0][1] * weights[1][0]
    gain[:2000] = 0
    expected = gain[:, None] * [0.1, 0.225]
    for backend in ("numpy", "native", "persistent"):
        renderer = (
            sample_instrument.PersistentSampler(definition)
            if backend == "persistent"
            else sample_instrument.OfflineSampler(definition, backend)
        )
        first = renderer.advance([a for a in actions if a.tick < 5501], 0, 5501)
        saved = type(renderer.snapshot()).model_validate_json(
            renderer.snapshot().model_dump_json()
        )
        tail = renderer.advance([a for a in actions if a.tick >= 5501], 5501, 48000)
        check_audio(
            tmp_path / f"control-layers-{backend}.wav",
            np.concatenate([first, tail]),
            expected,
        )
        renderer.restore(saved)
        pieces = []
        for start in range(5501, 48000, 997):
            end = min(48000, start + 997)
            pieces.append(
                renderer.advance(
                    [a for a in actions if start <= a.tick < end], start, end
                )
            )
        np.testing.assert_allclose(np.concatenate(pieces), tail, atol=1e-14, rtol=0)
