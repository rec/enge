from fractions import Fraction
from pathlib import Path

import numpy as np
import pytest
import test_fm
import test_noise
from test_dynamic_synth import change, dynamic_score, onset
from test_filter_instrument import filter_score
from test_lfo_instrument import lfo_score
from test_synth import check_audio, score
from ufor.control import TempoMap
from ufor.envelope import Envelope
from ufor.events import LFOChange, MotionChange, Release, Trigger
from ufor.instrument_trace import VoiceRetirement
from ufor.samples.processing import FilterResponse, ResonantFilter
from ufor.segments import Segment
from ufor.synth import SynthInstrumentScore
from ufor.synth_trace import prepare as prepare_trace

from enge import _native, filters, fm, noise
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
        np.empty((0, 8), dtype=np.float64),
        [],
        np.empty((0, 6), dtype=np.float64),
        [],
        np.empty((0, 7), dtype=np.float64),
        [1, -1e300, 1e300, 0, -120000, 120000],
        1,
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
        np.empty((0, 8), dtype=np.float64),
        [],
        np.empty((0, 6), dtype=np.float64),
        [],
        np.empty((0, 7), dtype=np.float64),
        [1, -1e300, 1e300, 0, -120000, 120000],
        1,
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
                    Segment(duration="1/20 s", to=1),
                    Segment(duration="1/10 s", to=0.7),
                ],
                release=[
                    Segment(duration="7/100 s", to=0.3),
                    Segment(duration="13/100 s", to=0),
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


def test_persistent_synth_keeps_part_controls_while_silent(tmp_path: Path) -> None:
    document = dynamic_score(scope="part", waveform="square")
    events = [
        change(0, 1, scope="part"),
        onset(120, "a", gain=0, pitch=0.1),
        onset(180, "b", gain=0, pitch=0.1),
        onset(180, "c", "other", pitch=0.1).model_copy(update={"ordinal": 1}),
    ]
    actions = prepare_trace(document.body, events, seed=0).actions
    definition = prepare(document)
    with pytest.raises(EngineError, match="context capacity"):
        PersistentSynth(definition, voices=4, context_capacity=1).advance(
            actions, 0, 48000
        )
    expected = OfflineSynth(definition, "native").advance(actions, 0, 48000)
    renderer = PersistentSynth(definition, voices=4)
    chunks = []
    boundaries = [0, 64, 121, 181, 997, 48000]
    for start, end in zip(boundaries, boundaries[1:], strict=False):
        chunks.append(
            renderer.advance([a for a in actions if start <= a.tick < end], start, end)
        )
        if end == 121:
            snapshot = renderer.snapshot()
    actual = np.concatenate(chunks)
    restored = PersistentSynth(definition, voices=4)
    restored.restore(snapshot)
    replay = restored.advance([a for a in actions if 121 <= a.tick < 48000], 121, 48000)

    check_audio(tmp_path / "persistent-part-controls.wav", actual, expected)
    np.testing.assert_allclose(replay, actual[121:], atol=0)


def test_persistent_synth_keeps_reused_trigger_tails_separate(tmp_path: Path) -> None:
    document = dynamic_score(waveform="square")
    raw = document.model_dump(mode="json")
    raw["body"]["voices"][0]["envelope"]["release"] = [{"duration": "1 s", "to": 0}]
    document = type(document).model_validate(raw)
    events = [
        onset(pitch=0.1, gain=0.2),
        change(60, 1),
        Release(tick=120, ordinal=0, part="main", trigger_id="note"),
        onset(240, pitch=0.1, gain=0.8),
        change(240, 0.4),
    ]
    actions = prepare_trace(document.body, events, seed=0).actions
    definition = prepare(document)
    expected = OfflineSynth(definition, "native").advance(actions, 0, 48000)
    renderer = PersistentSynth(definition, voices=4)
    chunks = []
    boundaries = [0, 64, 128, 256, 997, 48000]
    for start, end in zip(boundaries, boundaries[1:], strict=False):
        chunks.append(
            renderer.advance([a for a in actions if start <= a.tick < end], start, end)
        )
        if end == 256:
            snapshot = renderer.snapshot()
    actual = np.concatenate(chunks)
    restored = PersistentSynth(definition, voices=4)
    restored.restore(snapshot)
    replay = restored.advance([a for a in actions if 256 <= a.tick < 48000], 256, 48000)

    check_audio(tmp_path / "persistent-trigger-controls.wav", actual, expected)
    np.testing.assert_allclose(replay, actual[256:], atol=0)
    assert len(snapshot.trigger_contexts) == 1
    assert len(snapshot.voices) == 2


@pytest.mark.parametrize(
    "scope,waveform,duty",
    [
        ("instrument", "sine", "1/2"),
        ("part", "triangle", "1/3"),
        ("voice", "square", "1/3"),
    ],
)
def test_persistent_synth_evolves_scoped_lfos_in_rust(
    tmp_path: Path, scope: str, waveform: str, duty: str
) -> None:
    raw = lfo_score("synth", scope).model_dump(mode="json")
    raw["body"]["voices"][0]["motions"]["motion"]["body"].update(
        shape=waveform, duty_cycle=duty, center=0.2, depth=0.7
    )
    document = SynthInstrumentScore.model_validate(raw)
    events = [
        onset(6000, pitch=0.1).model_copy(update={"controls": {}}),
        onset(12000, "second", pitch=0.1).model_copy(update={"controls": {}}),
        onset(12000, "third", part="other", pitch=0.1).model_copy(
            update={"controls": {}, "ordinal": 1}
        ),
        Release(tick=18000, ordinal=0, part="main", trigger_id="note"),
        onset(24000, pitch=0.1).model_copy(update={"controls": {}}),
    ]
    actions = prepare_trace(document.body, events, seed=0).actions
    definition = prepare(document)
    expected = OfflineSynth(definition, "native").advance(actions, 0, 48000)
    renderer = PersistentSynth(definition, voices=8)
    chunks = []
    boundaries = [0, 997, 6001, 12001, 18001, 24001, 30000, 48000]
    for start, end in zip(boundaries, boundaries[1:], strict=False):
        chunks.append(
            renderer.advance([a for a in actions if start <= a.tick < end], start, end)
        )
        if end == 24001:
            snapshot = renderer.snapshot()
    actual = np.concatenate(chunks)
    restored = PersistentSynth(definition, voices=8)
    restored.restore(snapshot)
    replay = restored.advance(
        [a for a in actions if 24001 <= a.tick < 48000], 24001, 48000
    )

    check_audio(tmp_path / f"persistent-{scope}-lfo.wav", actual, expected)
    np.testing.assert_allclose(replay, actual[24001:], atol=0)


def test_persistent_synth_applies_instrument_lfo_events(tmp_path: Path) -> None:
    document = lfo_score("synth", "instrument")
    actions = prepare_trace(
        document.body,
        [
            onset(0, pitch=0.1).model_copy(update={"controls": {}}),
            LFOChange(tick=12000, ordinal=0, name="motion", action="rate", rate=5),
            LFOChange(tick=24000, ordinal=0, name="motion", action="reset"),
        ],
        seed=0,
    ).actions
    definition = prepare(document)
    expected = OfflineSynth(definition, "native").advance(actions, 0, 48000)
    renderer = PersistentSynth(definition, voices=4)
    actual = np.empty_like(expected)
    for start in range(0, 48000, 997):
        end = min(48000, start + 997)
        actual[start:end] = renderer.advance(
            [action for action in actions if start <= action.tick < end], start, end
        )
        if end == 24925:
            snapshot = renderer.snapshot()
    restored = PersistentSynth(definition, voices=4)
    restored.restore(snapshot)
    replay = restored.advance(
        [action for action in actions if 24925 <= action.tick < 48000], 24925, 48000
    )

    check_audio(tmp_path / "persistent-instrument-lfo-events.wav", actual, expected)
    np.testing.assert_allclose(replay, actual[24925:], atol=0)


def test_instrument_lfo_playback_commands_match_offline_and_persistent(
    tmp_path: Path,
) -> None:
    document = lfo_score("synth", "instrument")
    events = [
        onset(0, pitch=0.1).model_copy(update={"controls": {}}),
        LFOChange(tick=12000, ordinal=0, name="motion", action="pause"),
        LFOChange(tick=18000, ordinal=0, name="motion", action="resume"),
        LFOChange(tick=24000, ordinal=0, name="motion", action="reverse"),
        LFOChange(tick=30000, ordinal=0, name="motion", action="seek", position=0.75),
        LFOChange(tick=36000, ordinal=0, name="motion", action="rate", rate=2),
        LFOChange(tick=39000, ordinal=0, name="motion", action="shift", offset=-0.5),
    ]
    actions = prepare_trace(document.body, events, seed=0).actions
    definition = prepare(document)
    expected = OfflineSynth(definition, "numpy").advance(actions, 0, 48000)
    native = OfflineSynth(definition, "native").advance(actions, 0, 48000)
    check_audio(tmp_path / "lfo-playback-native.wav", native, expected)
    renderer = PersistentSynth(definition, voices=4)
    actual = np.empty_like(expected)
    for start in range(0, 48000, 997):
        end = min(48000, start + 997)
        actual[start:end] = renderer.advance(
            [action for action in actions if start <= action.tick < end], start, end
        )
        if end == 30907:
            snapshot = renderer.snapshot()
    check_audio(tmp_path / "lfo-playback-persistent.wav", actual, expected)
    restored = PersistentSynth(definition, voices=4)
    restored.restore(snapshot)
    replay = restored.advance(
        [action for action in actions if 30907 <= action.tick < 48000],
        30907,
        48000,
    )
    np.testing.assert_allclose(replay, actual[30907:], atol=0)
    baseline_actions = prepare_trace(document.body, events[:1], seed=0).actions
    baseline = OfflineSynth(definition, "numpy").advance(baseline_actions, 0, 48000)
    assert not np.allclose(expected[18000:24000], baseline[18000:24000])


@pytest.mark.parametrize("body_kind", ["cycle", "contour", "stages"])
def test_trigger_motion_playback_matches_persistent_native(
    tmp_path: Path, body_kind: str
) -> None:
    raw = lfo_score("synth", "voice").model_dump(mode="json")
    contour = {"kind": "contour", "segments": [{"duration": "1 s", "to": 1}]}
    if body_kind == "contour":
        body = contour
        modulation = raw["body"]["voices"][0]["modulation"]
        modulation["sources"][0]["minimum"] = 0
        modulation["routes"][0]["points"][0]["input"] = 0
    elif body_kind == "stages":
        body = {
            "kind": "stages",
            "initial_stage": "rise",
            "stages": [{"name": "rise", "motion": contour}],
        }
    else:
        body = raw["body"]["voices"][0]["motions"]["motion"]["body"]
    raw["body"]["voices"][0]["motions"]["motion"]["body"] = body
    document = SynthInstrumentScore.model_validate(raw)
    events = [
        onset(0, pitch=0.1).model_copy(update={"controls": {}}),
        onset(0, "second", pitch=0.2).model_copy(update={"controls": {}, "ordinal": 1}),
        MotionChange(
            tick=12000,
            ordinal=0,
            name="motion",
            part="main",
            trigger_id="note",
            action="pause",
        ),
        MotionChange(
            tick=18000,
            ordinal=0,
            name="motion",
            part="main",
            trigger_id="note",
            action="seek",
            position=0.75,
        ),
        MotionChange(
            tick=24000,
            ordinal=0,
            name="motion",
            part="main",
            trigger_id="note",
            action="resume",
        ),
        MotionChange(
            tick=30000,
            ordinal=0,
            name="motion",
            part="main",
            trigger_id="note",
            action="reverse",
        ),
        MotionChange(
            tick=36000,
            ordinal=0,
            name="motion",
            part="main",
            trigger_id="note",
            action="seek",
            position=0.5,
        ),
        MotionChange(
            tick=39000,
            ordinal=0,
            name="motion",
            part="main",
            trigger_id="note",
            action="shift",
            offset=0.75,
        ),
        MotionChange(
            tick=42000,
            ordinal=0,
            name="motion",
            part="main",
            trigger_id="note",
            action="shift",
            offset=-0.5,
        ),
    ]
    actions = prepare_trace(document.body, events, seed=0).actions
    definition = prepare(document)
    expected = OfflineSynth(definition, "numpy").advance(actions, 0, 48000)
    renderer = PersistentSynth(definition, voices=4)
    actual = np.empty_like(expected)
    for start in range(0, 48000, 997):
        end = min(48000, start + 997)
        actual[start:end] = renderer.advance(
            [action for action in actions if start <= action.tick < end], start, end
        )
        if end == 37886:
            snapshot = renderer.snapshot()
    check_audio(tmp_path / f"trigger-motion-{body_kind}.wav", actual, expected)
    restored = PersistentSynth(definition, voices=4)
    restored.restore(snapshot)
    replay = restored.advance(
        [a for a in actions if 37886 <= a.tick < 48000], 37886, 48000
    )
    np.testing.assert_allclose(replay, actual[37886:], atol=0)


@pytest.mark.parametrize("playback", ["loop", "ping_pong"])
@pytest.mark.parametrize("named", [False, True])
def test_contour_playback_and_immediate_release_match_persistent_native(
    tmp_path: Path, playback: str, named: bool
) -> None:
    raw = lfo_score("synth", "voice").model_dump(mode="json")
    voice = raw["body"]["voices"][0]
    voice["motions"]["motion"]["body"] = {
        "kind": "contour",
        "playback": playback,
        "segments": [{"duration": "1/2 s", "to": 1}],
        "release": [{"duration": "1/4 s", "to": 0}],
    }
    if named:
        voice["motions"]["motion"]["body"].update(
            {
                "markers": [
                    {"name": "start", "position": "1/4"},
                    {"name": "end", "position": "3/4"},
                ],
                "loop_start": "start",
                "loop_end": "end",
            }
        )
    voice["modulation"]["sources"][0]["minimum"] = 0
    voice["modulation"]["routes"][0]["points"][0]["input"] = 0
    document = SynthInstrumentScore.model_validate(raw)
    events = [
        onset(0, pitch=0.1).model_copy(update={"controls": {}}),
        MotionChange(
            tick=26000,
            ordinal=0,
            name="motion",
            part="main",
            trigger_id="note",
            action="shift",
            offset=1.5,
        ),
        MotionChange(
            tick=28000,
            ordinal=0,
            name="motion",
            part="main",
            trigger_id="note",
            action="reverse",
        ),
        Release(tick=36000, ordinal=0, part="main", trigger_id="note"),
    ]
    actions = prepare_trace(document.body, events, seed=0).actions
    definition = prepare(document)
    expected = OfflineSynth(definition, "numpy").advance(actions, 0, 48000)
    renderer = PersistentSynth(definition, voices=2)
    actual = np.empty_like(expected)
    for start in range(0, 48000, 997):
        end = min(48000, start + 997)
        actual[start:end] = renderer.advance(
            [a for a in actions if start <= a.tick < end], start, end
        )
        if end == 24925:
            snapshot = renderer.snapshot()
    check_audio(tmp_path / f"contour-{playback}-{named}.wav", actual, expected)
    restored = PersistentSynth(definition, voices=2)
    restored.restore(snapshot)
    replay = restored.advance(
        [a for a in actions if 24925 <= a.tick < 48000], 24925, 48000
    )
    np.testing.assert_allclose(replay, actual[24925:], atol=0)


@pytest.mark.parametrize("playback", ["loop", "ping_pong"])
@pytest.mark.parametrize("named", [False, True])
def test_staged_contour_playback_and_release_match_persistent_native(
    tmp_path: Path, playback: str, named: bool
) -> None:
    raw = lfo_score("synth", "voice").model_dump(mode="json")
    raw["body"]["voices"][0]["motions"]["motion"]["body"] = {
        "kind": "stages",
        "initial_stage": "sweep",
        "stages": [
            {
                "name": "sweep",
                "motion": {
                    "kind": "contour",
                    "playback": playback,
                    "segments": [{"duration": "1/4 s", "to": 1}],
                    "markers": [{"name": "quarter", "position": "1/4"}],
                },
            },
            {
                "name": "release",
                "motion": {
                    "kind": "contour",
                    "initial": "current",
                    "segments": [{"duration": "1/4 s", "to": 0}],
                },
            },
        ],
        "transitions": [
            {
                "from": ["sweep"],
                "event": "note_off",
                "action": {"kind": "enter", "stage": "release"},
            }
        ],
    }
    if named:
        raw["body"]["voices"][0]["motions"]["motion"]["body"]["stages"][0][
            "motion"
        ].update(
            {
                "markers": [
                    {"name": "start", "position": "1/4"},
                    {"name": "end", "position": "3/4"},
                ],
                "loop_start": "start",
                "loop_end": "end",
            }
        )
    document = SynthInstrumentScore.model_validate(raw)
    events = [
        onset(0, pitch=0.1).model_copy(update={"controls": {}}),
        MotionChange(
            tick=13000,
            ordinal=0,
            name="motion",
            part="main",
            trigger_id="note",
            action="shift",
            offset=0.5,
        ),
        Release(tick=19000, ordinal=0, part="main", trigger_id="note"),
    ]
    actions = prepare_trace(document.body, events, seed=0).actions
    definition = prepare(document)
    expected = OfflineSynth(definition, "numpy").advance(actions, 0, 48000)
    renderer = PersistentSynth(definition, voices=2)
    actual = np.empty_like(expected)
    for start in range(0, 48000, 997):
        end = min(48000, start + 997)
        actual[start:end] = renderer.advance(
            [a for a in actions if start <= a.tick < end], start, end
        )
        if end == 9970:
            snapshot = renderer.snapshot()
    check_audio(tmp_path / f"staged-contour-{playback}-{named}.wav", actual, expected)
    restored = PersistentSynth(definition, voices=2)
    restored.restore(snapshot)
    replay = restored.advance(
        [a for a in actions if 9970 <= a.tick < 48000], 9970, 48000
    )
    np.testing.assert_allclose(replay, actual[9970:], atol=0)


@pytest.mark.parametrize(
    ("playback", "boundary"), [("loop", 12000), ("ping_pong", 24000)]
)
def test_staged_contour_cycle_enters_next_stage_in_native_runtime(
    tmp_path: Path, playback: str, boundary: int
) -> None:
    raw = lfo_score("synth", "voice").model_dump(mode="json")
    raw["body"]["voices"][0]["motions"]["motion"]["body"] = {
        "kind": "stages",
        "initial_stage": "sweep",
        "stages": [
            {
                "name": "sweep",
                "motion": {
                    "kind": "contour",
                    "playback": playback,
                    "segments": [{"duration": "1/4 s", "to": 1}],
                },
            },
            {"name": "held", "motion": {"kind": "hold", "value": 0.5}},
        ],
        "transitions": [
            {
                "from": ["sweep"],
                "event": "stage.cycle",
                "action": {"kind": "enter", "stage": "held"},
            }
        ],
    }
    document = SynthInstrumentScore.model_validate(raw)
    actions = prepare_trace(
        document.body,
        [onset(0, pitch=0.1).model_copy(update={"controls": {}})],
        seed=0,
    ).actions
    definition = prepare(document)
    expected = OfflineSynth(definition, "numpy").advance(actions, 0, 48000)
    actual = PersistentSynth(definition, voices=2).advance(actions, 0, 48000)
    check_audio(tmp_path / f"staged-cycle-{playback}.wav", actual, expected)
    assert not np.isclose(expected[boundary - 1, 0], expected[boundary + 1, 0])


@pytest.mark.parametrize("kind", ["fm", "noise"])
def test_trigger_motion_playback_matches_persistent_fm_and_noise(
    tmp_path: Path, kind: str
) -> None:
    raw = (test_fm.score() if kind == "fm" else test_noise.score()).model_dump(
        mode="json"
    )
    voice = raw["body"]["voices"][0]
    voice["motions"] = {
        "motion": {"body": {"kind": "cycle", "rate": "2", "phase": "1/4"}}
    }
    voice["bindings"] = [{"name": "motion", "kind": "motion", "reference": "motion"}]
    target = (
        {"name": "fm", "parameter": "carrier_level"}
        if kind == "fm"
        else {"name": "processing", "parameter": "amplitude"}
    )
    level = 0.2 if kind == "fm" else 1
    voice["modulation"] = {
        "sources": [{"name": "motion", "scope": "voice", "minimum": -1, "maximum": 1}],
        "parameters": [
            {
                "target": target,
                "unit": "ratio",
                "scope": "voice",
                "minimum": 0,
                "maximum": level,
                "default": level,
            }
        ],
        "routes": [
            {
                "name": "motion",
                "source": "motion",
                "target": target,
                "operation": "multiply",
                "unit": "ratio",
                "points": [{"input": -1, "amount": 0.25}, {"input": 1, "amount": 1}],
            }
        ],
    }
    document = SynthInstrumentScore.model_validate(raw)
    first = test_fm.trigger(pitch=0.1) if kind == "fm" else test_noise.trigger()
    second = (
        test_fm.trigger(name="second", pitch=0.2)
        if kind == "fm"
        else test_noise.trigger(name="second")
    ).model_copy(update={"ordinal": 1})
    events = [
        first,
        second,
        MotionChange(
            tick=6000,
            ordinal=0,
            name="motion",
            part="main",
            trigger_id="note",
            action="pause",
        ),
        MotionChange(
            tick=10000,
            ordinal=0,
            name="motion",
            part="main",
            trigger_id="note",
            action="seek",
            position=0.75,
        ),
        MotionChange(
            tick=12000,
            ordinal=0,
            name="motion",
            part="main",
            trigger_id="note",
            action="resume",
        ),
        MotionChange(
            tick=15000,
            ordinal=0,
            name="motion",
            part="main",
            trigger_id="note",
            action="reverse",
        ),
        Release(tick=18000, ordinal=0, part="main", trigger_id="note"),
        first.model_copy(update={"tick": 24000}),
        MotionChange(
            tick=30000,
            ordinal=0,
            name="motion",
            part="main",
            trigger_id="note",
            action="seek",
            position=0.25,
        ),
        MotionChange(
            tick=34000,
            ordinal=0,
            name="motion",
            part="main",
            trigger_id="note",
            action="shift",
            offset=-0.5,
        ),
    ]
    actions = prepare_trace(document.body, events, seed=0).actions
    assert any(isinstance(a, VoiceRetirement) for a in actions)
    baseline_actions = prepare_trace(
        document.body, [e for e in events if not isinstance(e, MotionChange)], seed=0
    ).actions
    definition = fm.prepare(document) if kind == "fm" else noise.prepare(document)
    if kind == "fm":
        expected = fm.OfflineFM(definition).advance(actions, 0, 48000)
        baseline = fm.OfflineFM(definition).advance(baseline_actions, 0, 48000)
        renderer = fm.PersistentFM(definition, voices=4)
    else:
        expected = noise.OfflineNoise(definition).advance(actions, 0, 48000)
        baseline = noise.OfflineNoise(definition).advance(baseline_actions, 0, 48000)
        renderer = noise.PersistentNoise(definition, voices=4)
    assert not np.allclose(expected, baseline)
    actual = np.empty_like(expected)
    for start in range(0, 48000, 997):
        end = min(48000, start + 997)
        actual[start:end] = renderer.advance(
            [a for a in actions if start <= a.tick < end], start, end
        )
        if end == 24925:
            snapshot = renderer.snapshot()
    check_audio(tmp_path / f"trigger-motion-{kind}.wav", actual, expected)
    if kind == "fm":
        restored = fm.PersistentFM(definition, voices=4)
    else:
        restored = noise.PersistentNoise(definition, voices=4)
    restored.restore(snapshot)
    replay = restored.advance(
        [a for a in actions if 24925 <= a.tick < 48000], 24925, 48000
    )
    np.testing.assert_allclose(replay, actual[24925:], atol=0)


@pytest.mark.parametrize("playback", ["loop", "ping_pong"])
@pytest.mark.parametrize("named", [False, True])
@pytest.mark.parametrize("staged", [False, True])
@pytest.mark.parametrize("reverse", [False, True])
def test_finite_contour_repeats_match_persistent_native(
    tmp_path: Path, playback: str, named: bool, staged: bool, reverse: bool
) -> None:
    raw = lfo_score("synth", "voice").model_dump(mode="json")
    voice = raw["body"]["voices"][0]
    contour = {
        "kind": "contour",
        "playback": playback,
        "segments": [{"duration": "1/4 s", "to": 1}],
        "repeat_count": 2,
    }
    if named:
        contour.update(
            {
                "markers": [
                    {"name": "start", "position": "1/4"},
                    {"name": "end", "position": "3/4"},
                ],
                "loop_start": "start",
                "loop_end": "end",
            }
        )
    voice["motions"]["motion"]["body"] = (
        {
            "kind": "stages",
            "initial_stage": "sweep",
            "stages": [{"name": "sweep", "motion": contour}],
        }
        if staged
        else contour
    )
    if not staged:
        voice["modulation"]["sources"][0]["minimum"] = 0
        voice["modulation"]["routes"][0]["points"][0]["input"] = 0
    document = SynthInstrumentScore.model_validate(raw)
    events = [onset(0, pitch=0.1).model_copy(update={"controls": {}})]
    if reverse:
        events.append(
            MotionChange(
                tick=14000,
                ordinal=0,
                name="motion",
                part="main",
                trigger_id="note",
                action="reverse",
            )
        )
    actions = prepare_trace(document.body, events, seed=0).actions
    definition = prepare(document)
    expected = OfflineSynth(definition, "numpy").advance(actions, 0, 48000)
    renderer = PersistentSynth(definition, voices=2)
    first = renderer.advance([a for a in actions if a.tick < 13337], 0, 13337)
    snapshot = renderer.snapshot()
    later = [a for a in actions if a.tick >= 13337]
    second = renderer.advance(later, 13337, 48000)
    actual = np.concatenate([first, second])
    check_audio(
        tmp_path / f"finite-contour-{playback}-{named}-{staged}-{reverse}.wav",
        actual,
        expected,
    )
    replay = PersistentSynth(definition, voices=2)
    replay.restore(snapshot)
    np.testing.assert_allclose(replay.advance(later, 13337, 48000), second, atol=0)


def test_host_tempo_quantizes_motion_change_for_both_renderers(tmp_path: Path) -> None:
    raw = lfo_score("synth", "voice").model_dump(mode="json")
    voice = raw["body"]["voices"][0]
    voice["motions"]["motion"]["body"] = {
        "kind": "contour",
        "segments": [{"duration": "1 s", "to": 1}],
    }
    voice["modulation"]["sources"][0]["minimum"] = 0
    voice["modulation"]["routes"][0]["points"][0]["input"] = 0
    document = SynthInstrumentScore.model_validate(raw)
    clock = TempoMap.model_validate(
        {
            "points": [
                {"at_seconds": "0", "beat": "0", "bpm": "120"},
                {"at_seconds": "1/4", "beat": "1/2", "bpm": "60"},
            ]
        }
    )
    start = onset(0, pitch=0.1).model_copy(update={"controls": {}})
    command = MotionChange(
        tick=12000,
        ordinal=0,
        name="motion",
        part="main",
        trigger_id="note",
        action="pause",
        quantize_beats=Fraction(1),
    )
    actions = prepare_trace(
        document.body,
        [start, command],
        seed=0,
        tempo_map=clock,
        sample_rate=48000,
    ).actions
    manual = prepare_trace(
        document.body,
        [start, command.model_copy(update={"tick": 36000, "quantize_beats": None})],
        seed=0,
    ).actions
    assert actions == manual
    definition = prepare(document)
    expected = OfflineSynth(definition, "numpy").advance(manual, 0, 48000)
    actual = PersistentSynth(definition, voices=2).advance(actions, 0, 48000)
    check_audio(tmp_path / "quantized-motion.wav", actual, expected)


def test_reversed_stage_marker_cues_same_trigger_in_native_runtime(
    tmp_path: Path,
) -> None:
    raw = lfo_score("synth", "voice").model_dump(mode="json")
    voice = raw["body"]["voices"][0]
    voice["motions"] = {
        "clock": {
            "body": {
                "kind": "stages",
                "initial_stage": "cycle",
                "stages": [
                    {
                        "name": "cycle",
                        "motion": {
                            "kind": "cycle",
                            "rate": "1",
                            "phase": "1/2",
                            "markers": [{"name": "quarter", "position": "1/4"}],
                        },
                    }
                ],
            }
        },
        "level": {
            "body": {
                "kind": "stages",
                "initial_stage": "waiting",
                "stages": [
                    {"name": "waiting", "motion": {"kind": "hold", "value": 0}},
                    {"name": "bright", "motion": {"kind": "hold", "value": 1}},
                ],
                "transitions": [
                    {
                        "from": ["waiting"],
                        "event": "cue.brighten",
                        "action": {"kind": "enter", "stage": "bright"},
                    }
                ],
            }
        },
    }
    voice["bindings"] = [
        {"name": "clock", "kind": "motion", "reference": "clock"},
        {"name": "level", "kind": "motion", "reference": "level"},
    ]
    voice["modulation"]["sources"] = [
        {"name": "clock", "scope": "voice", "minimum": -1, "maximum": 1},
        {"name": "level", "scope": "voice", "minimum": -1, "maximum": 1},
    ]
    voice["modulation"]["routes"][0]["source"] = "level"
    voice["event_connections"] = [
        {
            "source": "clock",
            "port": "quarter",
            "destination": "level",
            "cue": "brighten",
        }
    ]
    document = SynthInstrumentScore.model_validate(raw)
    events = [
        onset(0, pitch=0.1).model_copy(update={"controls": {}}),
        MotionChange(
            tick=1,
            ordinal=0,
            name="clock",
            part="main",
            trigger_id="note",
            action="reverse",
        ),
    ]
    actions = prepare_trace(document.body, events, seed=0).actions
    definition = prepare(document)
    expected = OfflineSynth(definition, "numpy").advance(actions, 0, 48000)
    actual = PersistentSynth(definition, voices=2).advance(actions, 0, 48000)
    check_audio(tmp_path / "reversed-stage-marker.wav", actual, expected)
    assert abs(expected[12002, 0]) > abs(expected[12001, 0])


@pytest.mark.parametrize(
    ("playback", "port", "boundary", "named"),
    [
        ("loop", "cycle", 12000, False),
        ("ping_pong", "turned", 12000, False),
        ("ping_pong", "cycle", 24000, False),
        ("loop", "cycle", 9000, True),
        ("ping_pong", "turned", 9000, True),
        ("ping_pong", "cycle", 15000, True),
        ("loop", "stage.done", 12000, False),
    ],
)
def test_staged_contour_events_cue_another_motion_in_native_runtime(
    tmp_path: Path, playback: str, port: str, boundary: int, named: bool
) -> None:
    raw = lfo_score("synth", "voice").model_dump(mode="json")
    voice = raw["body"]["voices"][0]
    voice["motions"] = {
        "clock": {
            "body": {
                "kind": "stages",
                "initial_stage": "sweep",
                "stages": [
                    {
                        "name": "sweep",
                        "motion": {
                            "kind": "contour",
                            "playback": playback,
                            "segments": [{"duration": "1/4 s", "to": 1}],
                        },
                    }
                ],
            }
        },
        "level": {
            "body": {
                "kind": "stages",
                "initial_stage": "waiting",
                "stages": [
                    {"name": "waiting", "motion": {"kind": "hold", "value": 0}},
                    {"name": "bright", "motion": {"kind": "hold", "value": 1}},
                ],
                "transitions": [
                    {
                        "from": ["waiting"],
                        "event": "cue.brighten",
                        "action": {"kind": "enter", "stage": "bright"},
                    }
                ],
            }
        },
    }
    if port == "stage.done":
        voice["motions"]["clock"]["body"]["stages"][0]["motion"]["repeat_count"] = 1
    if named:
        voice["motions"]["clock"]["body"]["stages"][0]["motion"].update(
            {
                "markers": [
                    {"name": "start", "position": "1/4"},
                    {"name": "end", "position": "3/4"},
                ],
                "loop_start": "start",
                "loop_end": "end",
            }
        )
        if boundary == 9000:
            level = voice["motions"]["level"]["body"]
            level["stages"].insert(
                1, {"name": "middle", "motion": {"kind": "hold", "value": 0}}
            )
            level["transitions"][0]["from"] = ["middle"]
            level["transitions"].insert(
                0,
                {
                    "from": ["waiting"],
                    "event": "cue.marker",
                    "action": {"kind": "enter", "stage": "middle"},
                },
            )
    voice["bindings"] = [
        {"name": "clock", "kind": "motion", "reference": "clock"},
        {"name": "level", "kind": "motion", "reference": "level"},
    ]
    voice["modulation"]["sources"] = [
        {"name": "clock", "scope": "voice", "minimum": -1, "maximum": 1},
        {"name": "level", "scope": "voice", "minimum": -1, "maximum": 1},
    ]
    voice["modulation"]["routes"][0]["source"] = "level"
    voice["event_connections"] = [
        {"source": "clock", "port": port, "destination": "level", "cue": "brighten"}
    ]
    if named and boundary == 9000:
        voice["event_connections"].insert(
            0,
            {
                "source": "clock",
                "port": "end",
                "destination": "level",
                "cue": "marker",
            },
        )
    document = SynthInstrumentScore.model_validate(raw)
    actions = prepare_trace(
        document.body,
        [onset(0, pitch=0.1).model_copy(update={"controls": {}})],
        seed=0,
    ).actions
    definition = prepare(document)
    expected = OfflineSynth(definition, "numpy").advance(actions, 0, 48000)
    actual = PersistentSynth(definition, voices=2).advance(actions, 0, 48000)
    check_audio(
        tmp_path / f"staged-contour-event-{playback}-{port}-{named}.wav",
        actual,
        expected,
    )
    assert abs(expected[boundary + 1, 0]) > abs(expected[boundary - 1, 0])


def test_reversed_contour_markers_repeat_across_native_snapshot(
    tmp_path: Path,
) -> None:
    raw = lfo_score("synth", "voice").model_dump(mode="json")
    voice = raw["body"]["voices"][0]
    voice["motions"] = {
        "clock": {
            "body": {
                "kind": "stages",
                "initial_stage": "sweep",
                "stages": [
                    {
                        "name": "sweep",
                        "motion": {
                            "kind": "contour",
                            "playback": "loop",
                            "segments": [{"duration": "1/4 s", "to": 1}],
                            "markers": [
                                {"name": "quarter", "position": "1/4"},
                                {"name": "threequarter", "position": "3/4"},
                            ],
                        },
                    }
                ],
            }
        },
        "level": {
            "body": {
                "kind": "stages",
                "initial_stage": "quiet",
                "stages": [
                    {"name": "quiet", "motion": {"kind": "hold", "value": 0}},
                    {"name": "loud", "motion": {"kind": "hold", "value": 1}},
                ],
                "transitions": [
                    {
                        "from": ["quiet"],
                        "event": "cue.raise",
                        "action": {"kind": "enter", "stage": "loud"},
                    },
                    {
                        "from": ["loud"],
                        "event": "cue.lower",
                        "action": {"kind": "enter", "stage": "quiet"},
                    },
                    {
                        "from": ["loud"],
                        "event": "cue.raise",
                        "action": {"kind": "enter", "stage": "quiet"},
                    },
                    {
                        "from": ["quiet"],
                        "event": "cue.lower",
                        "action": {"kind": "enter", "stage": "loud"},
                    },
                ],
            }
        },
    }
    voice["bindings"] = [
        {"name": "clock", "kind": "motion", "reference": "clock"},
        {"name": "level", "kind": "motion", "reference": "level"},
    ]
    voice["modulation"]["sources"] = [
        {"name": "clock", "scope": "voice", "minimum": -1, "maximum": 1},
        {"name": "level", "scope": "voice", "minimum": -1, "maximum": 1},
    ]
    voice["modulation"]["routes"][0]["source"] = "level"
    voice["event_connections"] = [
        {
            "source": "clock",
            "port": "quarter",
            "destination": "level",
            "cue": "raise",
        },
        {
            "source": "clock",
            "port": "threequarter",
            "destination": "level",
            "cue": "lower",
        },
    ]
    document = SynthInstrumentScore.model_validate(raw)
    events = [
        onset(0, pitch=0.1).model_copy(update={"controls": {}}),
        MotionChange(
            tick=18000,
            ordinal=0,
            name="clock",
            part="main",
            trigger_id="note",
            action="reverse",
        ),
    ]
    actions = prepare_trace(document.body, events, seed=0).actions
    definition = prepare(document)
    expected = OfflineSynth(definition, "numpy").advance(actions, 0, 48000)
    renderer = PersistentSynth(definition, voices=2)
    first = renderer.advance([a for a in actions if a.tick < 24001], 0, 24001)
    snapshot = renderer.snapshot()
    second = renderer.advance([a for a in actions if a.tick >= 24001], 24001, 48000)
    actual = np.concatenate((first, second))
    check_audio(tmp_path / "reversed-contour-markers.wav", actual, expected)
    restored = PersistentSynth(definition, voices=2)
    restored.restore(snapshot)
    np.testing.assert_allclose(
        restored.advance([a for a in actions if a.tick >= 24001], 24001, 48000),
        second,
        atol=0,
    )
    for boundary in (3000, 9000, 21000, 27000):
        assert not np.isclose(expected[boundary - 1, 0], expected[boundary + 1, 0]), (
            boundary
        )


def test_persistent_synth_evolves_native_voice_filters(tmp_path: Path) -> None:
    document = filter_score("synth")
    events = [
        onset(pitch=220).model_copy(update={"controls": {"tone": 0}}),
        change(1001, 1, control="tone"),
        onset(6001, "second", pitch=330).model_copy(update={"controls": {"tone": 0.3}}),
        Release(tick=12001, ordinal=0, part="main", trigger_id="note"),
        change(13001, 0, control="tone"),
        Release(tick=18001, ordinal=0, part="main", trigger_id="second"),
    ]
    actions = prepare_trace(document.body, events, seed=0).actions
    definition = prepare(document)
    expected = OfflineSynth(definition, "native").advance(actions, 0, 48000)
    renderer = PersistentSynth(definition, voices=4)
    chunks = []
    boundaries = [0, 997, 1002, 6123, 13002, 24001, 48000]
    for start, end in zip(boundaries, boundaries[1:], strict=False):
        chunks.append(
            renderer.advance([a for a in actions if start <= a.tick < end], start, end)
        )
        if end == 13002:
            snapshot = renderer.snapshot()
    actual = np.concatenate(chunks)
    restored = PersistentSynth(definition, voices=4)
    restored.restore(snapshot)
    replay = restored.advance(
        [a for a in actions if 13002 <= a.tick < 48000], 13002, 48000
    )

    check_audio(tmp_path / "persistent-filter.wav", actual, expected)
    np.testing.assert_allclose(replay, actual[13002:], atol=0)
