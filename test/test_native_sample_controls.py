from pathlib import Path

import numpy as np
import pytest
from test_dynamic_synth import change, dynamic_score, onset
from test_sample_instrument import sample_score
from test_synth import check_audio
from ufor.events import Release
from ufor.samples import instrument, trace

from enge import _native, live, sample_instrument
from enge.synth import EngineError


@pytest.mark.parametrize("scope", ["instrument", "part", "trigger"])
@pytest.mark.parametrize("interpolation", ["linear", "step", "exponential"])
def test_authored_sample_controls_preserve_contexts_and_inherited_routes(
    tmp_path: Path, scope: str, interpolation: str
) -> None:
    raw = sample_score(native_rate=24000).model_dump(mode="json")
    dynamic = dynamic_score(scope, smoothing="4801/96000").model_dump(mode="json")[
        "body"
    ]
    common = raw["body"]["settings"]
    common["controls"] = dynamic["controls"]
    common["envelope"] = {
        "segments": [{"duration": 0, "to": 1}],
        "release": [{"duration": "0.5 s", "to": 0}],
    }
    settings = dynamic["voices"][0]
    settings["modulation"]["routes"][0].update(
        interpolation=interpolation,
        points=[{"input": x, "amount": y} for x, y in [(0, 0.1), (0.5, 0.75), (1, 1)]],
    )
    common.update({n: settings[n] for n in ("modulation", "bindings")})
    raw["body"]["groups"] = [
        {"name": "group", **{n: settings[n] for n in ("modulation", "bindings")}}
    ]
    raw["body"]["groups"][0]["bindings"] = dynamic_score(scope).model_dump(mode="json")[
        "body"
    ]["voices"][0]["bindings"]
    slot = raw["body"]["slots"][0]
    for name in ("bindings", "modulation"):
        slot.pop(name, None)
    slot["group"] = "group"
    slot["playback"]["start_delay_seconds"] = 0.01
    document = instrument.SampleInstrumentScore.model_validate(raw)
    ramp = np.arange(96000) / 96000
    definition = sample_instrument.prepare(
        document, {"asset": np.column_stack([ramp, 1 - ramp])}
    )
    actions = trace.prepare(
        document.body,
        [
            change(0, 1, scope=scope),
            onset(1000, pitch=440),
            onset(3000, "other", "other", pitch=330),
            change(1500, 0.1, scope=scope),
            change(1701, 0.8, scope=scope),
            change(4000, 0.5, control="bend", scope=scope),
            Release(tick=6000, ordinal=0, part="main", trigger_id="note"),
            onset(12000, pitch=220, gain=0.2),
            change(13000, 0.4, scope=scope),
            Release(tick=18000, ordinal=0, part="main", trigger_id="note"),
            Release(tick=19000, ordinal=0, part="other", trigger_id="other"),
        ],
        seed=0,
    ).actions
    expected = sample_instrument.OfflineSampler(definition).advance(actions, 0, 48000)
    source = sample_instrument.PersistentSampler(definition, voices=3)
    engine = live.NativeLiveEngine({"sample": source}, 48000)
    actual = np.empty((48000, 2))
    engine.advance_into(
        {"sample": [a for a in actions if a.tick < 500]}, [], 0, 500, actual[:500]
    )
    engine.advance_into(
        {"sample": [a for a in actions if 500 <= a.tick < 14003]},
        [],
        500,
        14003,
        actual[500:14003],
    )
    saved = engine.snapshot()
    engine.advance_into(
        {"sample": [a for a in actions if a.tick >= 14003]},
        [],
        14003,
        48000,
        actual[14003:],
    )
    check_audio(tmp_path / "authored-native-controls.wav", actual, expected)
    engine.restore(saved)
    replay = np.empty((33997, 2))
    for start in range(14003, 48000, 997):
        end = min(start + 997, 48000)
        engine.advance_into(
            {"sample": [a for a in actions if start <= a.tick < end]},
            [],
            start,
            end,
            replay[start - 14003 : end - 14003],
        )
    np.testing.assert_array_equal(replay, actual[14003:])
    if scope != "instrument":
        limited = live.NativeLiveEngine(
            {
                "sample": sample_instrument.PersistentSampler(
                    definition, context_capacity=1
                )
            },
            48000,
        )
        with pytest.raises(EngineError, match="context capacity"):
            limited.advance_into(
                {"sample": actions}, [], 0, 48000, np.empty_like(actual)
            )


@pytest.mark.parametrize("scope", [0, 1, 2])
@pytest.mark.parametrize("interpolation", [0, 1, 2])
def test_sample_controls_follow_scopes_delays_and_snapshot_continuation(
    tmp_path: Path, scope: int, interpolation: int
) -> None:
    audio = (np.arange(192000, dtype=np.float64) / 192000)[:, None]
    runtime = _native.LiveRuntime.empty(48000, 1, 48000, 64)
    source = runtime.add_sample(
        _native.SampleBuffer(audio),
        (0, len(audio), False, None),
        (0, 0, 0, 1, False, False, False, None, None, 0),
        48000,
        np.ones((1, 1)),
        1,
        [((0, 1), 1.0, 0.0)],
        [((0, 1), 0.0, 0.0)],
        0,
        1,
        4,
        ([], np.empty((0, 1, 2)), np.empty((1, 0, 2))),
        False,
    )
    runtime.set_sample_controls(
        source,
        np.array([[0.25, scope, 0, 1, 0, 1], [0.25, scope, 3, 0, 0, 1]]),
        [(4801, 2), (4801, 2)],
        [1, 0, 2, 0, -120000, 120000, 0.5, 0, 2, 0, -120000, 120000],
        [(interpolation, [(0, 0.1), (0.5, 0.75), (1, 1)]), (0, [(0, 0), (1, 1200)])],
        1,
        [],
        [],
    )
    context = -1 if scope == 0 else 0
    actions = [] if scope == 0 else [[0, 9, 0, scope, 0, 0]]
    if scope:
        actions.extend([[0, 10, 0, 0, 0.1, 0], [0, 10, 0, 1, 0.1, 0]])
    actions.extend(
        [
            [0, 0, 0, 48000, 0.2, 0],
            [0, 6, 0, 1200, 0, 0],
            [0, 8, 0, 0 if scope == 1 else -1, 0 if scope == 2 else -1, 0],
            [0, 7, 0, 1, context, 0],
            [0, 7, 1, 1, context, 0],
            [1700, 7, 0, 0.4, context, 0],
            [1700, 7, 1, 0.4, context, 0],
        ]
    )
    runtime.submit(source, np.asarray(actions, dtype=np.float64))
    actual = np.empty((48000, 1))
    runtime.process_into(actual[:1901])
    saved = runtime.snapshot()
    runtime.submit(source, np.array([[36000 - 1901, 2, 0, 0, 0, 0]], dtype=float))
    runtime.process_into(actual[1901:])

    frames = np.arange(48000)
    initial = 0.1 if scope else 0.25
    control = initial + (1 - initial) * np.minimum(frames / 2400.5, 1)
    control[1700:] = control[1700] + (0.4 - control[1700]) * np.minimum(
        np.arange(48000 - 1700) / 2400.5, 1
    )
    if interpolation == 0:
        gain = np.interp(control, [0, 0.5, 1], [0.1, 0.75, 1])
    elif interpolation == 1:
        gain = np.where(control < 0.5, 0.1, 0.75)
    else:
        gain = np.exp(np.interp(control, [0, 0.5, 1], np.log([0.1, 0.75, 1])))
    position = np.zeros(48000)
    position[1201:] = np.cumsum(np.exp2(control[1200:-1]))
    expected = (position / 192000 * 0.2 * 0.5 * gain)[:, None]
    expected[36000:] = 0
    check_audio(tmp_path / "native-sample-controls.wav", actual, expected)
    assert runtime.active_slots(source) == [False]
    assert runtime.active_contexts(source) == [scope == 1]

    runtime.restore(saved)
    replay = np.empty((48000 - 1901, 1))
    for start in range(1901, 48000, 997):
        end = min(start + 997, 48000)
        if start <= 36000 < end:
            runtime.submit(
                source, np.array([[36000 - start, 2, 0, 0, 0, 0]], dtype=float)
            )
        runtime.process_into(replay[start - 1901 : end - 1901])
    np.testing.assert_array_equal(replay, actual[1901:])
