from pathlib import Path

import numpy as np
import pytest
from test_synth import check_audio

from enge import _native


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
        (0, 0, 0, 1, False, False, False, None),
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
