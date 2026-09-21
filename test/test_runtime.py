from fractions import Fraction
from pathlib import Path

import numpy as np
import pytest
from test_dynamic_synth import change, dynamic_score, onset
from test_synth import check_audio, score
from ufor.envelope import Envelope, Segment
from ufor.events import Release, Trigger
from ufor.samples.processing import FilterResponse, ResonantFilter
from ufor.synth_trace import prepare as prepare_trace

from enge import _native, filters
from enge.synth import EngineError, OfflineSynth, PersistentSynth, prepare


def test_persistent_runtime_matches_oscillators_and_filter(tmp_path: Path) -> None:
    frequencies = [110.0, 165.0, 220.0]
    gains = [0.2, 0.15, 0.1]
    routes = np.array([[1.0, 0.25], [0.5, 1.0], [0.75, 0.75]])
    runtime = _native.SynthRuntime(
        48000,
        0,
        0.5,
        routes,
        1,
        np.array([[0, 1]], dtype=np.float64),
        np.array([[0, 0]], dtype=np.float64),
        0,
        np.empty((0, 7), dtype=np.float64),
        [],
        [1, -1e300, 1e300, 0, -120000, 120000],
    )
    starts = np.array(
        [
            [0, 0, i, f, g, 0]
            for i, (f, g) in enumerate(zip(frequencies, gains, strict=True))
        ],
        dtype=np.float64,
    )
    actual = np.concatenate(
        [
            runtime.process_actions(997, 2400, 0.8, -3, starts),
            runtime.process(4096, 2400, 0.8, -3),
            runtime.process(42907, 2400, 0.8, -3),
        ]
    )
    frames = np.arange(48000)
    dry = sum(
        np.sin(2 * np.pi * frames * f / 48000)[:, None] * g * r
        for f, g, r in zip(frequencies, gains, routes, strict=True)
    )
    definition = ResonantFilter(
        name="low", response=FilterResponse.lowpass, cutoff_hz=2400, q=0.8
    )
    expected, _ = filters.filter_samples(
        [definition],
        filters.initial_states([definition], 2),
        dry,
        filters.parameters([definition], 48000, 48000),
        48000,
    )
    expected *= 10 ** (-3 / 20)

    check_audio(tmp_path / "persistent-runtime.wav", actual, expected)


def test_persistent_runtime_applies_voice_actions_at_exact_frames() -> None:
    runtime = _native.SynthRuntime(
        48000,
        0,
        0.5,
        np.eye(2),
        0,
        np.array([[4, 1]], dtype=np.float64),
        np.array([[4, 0]], dtype=np.float64),
        0,
        np.empty((0, 7), dtype=np.float64),
        [],
        [1, -1e300, 1e300, 0, -120000, 120000],
    )
    actions = np.array(
        [
            [10, 0, 0, 100, 0.5, 0],
            [20, 0, 1, 200, 0.25, 0],
            [30, 3, 0, 200, 0.25, 5],
            [40, 1, 1, 0, 0, 0],
            [50, 2, 0, 0, 0, 0],
        ],
        dtype=np.float64,
    )

    actual = runtime.process_actions(64, 10000, 0.7, 0, actions)

    assert np.all(actual[:11] == 0)
    assert np.any(actual[11:20, 0])
    assert np.all(actual[:21, 1] == 0)
    assert np.any(actual[21:44, 1])
    assert abs(actual[-1, 1]) < abs(actual[43, 1])
    assert abs(actual[-1, 0]) < abs(actual[49, 0])


def test_persistent_synth_matches_offline_synth_across_blocks_and_restore(
    tmp_path: Path,
) -> None:
    document = score()
    voice = document.body.voices[0].model_copy(
        update={
            "minimum_hold_seconds": Fraction(1, 10),
            "envelope": Envelope(
                initial=0.2,
                segments=[
                    Segment(duration="1/20", target=1),
                    Segment(duration="1/10", target=0.7),
                ],
                release=[
                    Segment(duration="7/100", target=0.3),
                    Segment(duration="13/100", target=0),
                ],
            ),
        }
    )
    instrument = document.body.model_copy(update={"voices": [voice]})
    document = document.model_copy(update={"body": instrument})
    trace = prepare_trace(
        instrument,
        [
            Trigger(
                tick=101,
                ordinal=0,
                part="main",
                trigger_id="a",
                key=69,
                pitch_hz=440,
            ),
            Trigger(
                tick=4000,
                ordinal=0,
                part="main",
                trigger_id="b",
                key=64,
                pitch_hz=330,
            ),
            Release(tick=3000, ordinal=0, part="main", trigger_id="a"),
            Release(tick=19000, ordinal=0, part="main", trigger_id="b"),
        ],
        seed=42,
    )
    definition = prepare(document)
    with pytest.raises(EngineError, match="action capacity"):
        PersistentSynth(definition, voices=4, action_capacity=1).advance(
            trace.actions, 0, 48000
        )
    expected = OfflineSynth(definition, "native").advance(trace.actions, 0, 48000)
    renderer = PersistentSynth(definition, voices=4)
    pieces = []
    boundaries = [0, 997, 4096, 17003, 25001, 48000]
    for start, end in zip(boundaries, boundaries[1:], strict=False):
        actions = [a for a in trace.actions if start <= a.tick < end]
        output = np.full((end - start, 2), np.nan)
        renderer.advance_into(actions, start, end, output)
        pieces.append(output)
        if end == 17003:
            snapshot = renderer.snapshot()
    actual = np.concatenate(pieces)
    restored = PersistentSynth(definition, voices=4)
    restored.restore(snapshot)
    replay = restored.advance(
        [a for a in trace.actions if 17003 <= a.tick < 48000], 17003, 48000
    )
    owned = PersistentSynth(definition, voices=4)
    first = owned.advance([a for a in trace.actions if a.tick < 24000], 0, 24000)
    saved = first.copy()
    owned.advance([a for a in trace.actions if 24000 <= a.tick < 48000], 24000, 48000)

    check_audio(tmp_path / "persistent-synth.wav", actual, expected)
    np.testing.assert_allclose(replay, actual[17003:], atol=0)
    np.testing.assert_array_equal(first, saved)


def test_persistent_synth_evolves_instrument_controls_in_rust(tmp_path: Path) -> None:
    document = dynamic_score(scope="instrument", smoothing="4801/96000")
    events = [
        change(0, 1, scope="instrument"),
        onset(120, "a", gain=0, pitch=100),
        onset(180, "b", gain=0, pitch=150),
        change(240, 0, scope="instrument"),
        change(120, 1, control="bend", scope="instrument"),
        change(301, 0.25, scope="instrument"),
        Release(tick=20000, ordinal=0, part="main", trigger_id="a"),
        Release(tick=21000, ordinal=0, part="main", trigger_id="b"),
    ]
    actions = prepare_trace(document.body, events, seed=0).actions
    definition = prepare(document)
    with pytest.raises(EngineError, match="instrument-scoped"):
        PersistentSynth(prepare(dynamic_score(scope="part")))
    expected = OfflineSynth(definition, "native").advance(actions, 0, 48000)
    renderer = PersistentSynth(definition, voices=4)
    chunks = []
    boundaries = [0, 64, 121, 181, 241, 301, 997, 48000]
    for start, end in zip(boundaries, boundaries[1:], strict=False):
        block_actions = [a for a in actions if start <= a.tick < end]
        chunks.append(renderer.advance(block_actions, start, end))
        if end == 181:
            snapshot = renderer.snapshot()
    actual = np.concatenate(chunks)
    restored = PersistentSynth(definition, voices=4)
    restored.restore(snapshot)
    replay = restored.advance([a for a in actions if 181 <= a.tick < 48000], 181, 48000)

    check_audio(tmp_path / "persistent-controls.wav", actual, expected)
    np.testing.assert_allclose(replay, actual[181:], atol=0)
