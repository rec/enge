from fractions import Fraction
from pathlib import Path

import numpy as np
import pytest
from test_dynamic_synth import onset
from test_sample_instrument import sample_score
from test_synth import check_audio
from ufor.samples import instrument, trace
from ufor.samples.playback import Loop, Playback, Slice

from enge import live, sample_instrument, sampler


@pytest.mark.parametrize("direction", ["forward", "backward", "mirror"])
@pytest.mark.parametrize("plays", [1, 2, 3])
@pytest.mark.parametrize("step", [0.5, 1, 7, 100])
def test_whole_sample_restarts_preserve_envelope_and_first_only_offset(
    tmp_path: Path, direction: str, plays: int, step: float
) -> None:
    raw = sample_score(frames=12).model_dump(mode="json")
    raw["body"]["slots"][0]["playback"].update(
        mode="one_shot", direction=direction, play_count=plays
    )
    offset = -2 if direction == "backward" else 2
    raw["body"]["slots"][0]["alignment_frames"] = offset
    raw["body"]["slots"][0].update(selection="takes", take="one", microphone="close")
    raw["body"]["settings"]["selections"] = [{"name": "takes", "mode": "cycle"}]
    raw["body"]["settings"]["envelope"] = {
        "initial": 0,
        "segments": [{"duration": "1/100 s", "to": 1}],
        "release": [{"duration": "1/100 s", "to": 0}],
    }
    document = instrument.SampleInstrumentScore.model_validate(raw)
    audio = np.column_stack([np.arange(12) / 24, np.arange(12) / 48])
    definition = sample_instrument.prepare(document, {"asset": audio})
    actions = trace.prepare(
        document.body,
        [onset(pitch=440 * step).model_copy(update={"controls": {}})],
        seed=0,
    ).actions
    cycle = list(range(11, -1, -1)) if direction == "backward" else list(range(12))
    if direction == "mirror":
        cycle += list(range(10, -1, -1))
    indices = cycle[2:] + cycle * (plays - 1)
    positions = np.arange(0, len(indices), step)
    values = np.interp(positions, np.arange(len(indices)), indices)
    expected = np.zeros((48000, 2))
    expected[: len(values)] = (
        values[:, None] * [1 / 24, 1 / 32] * (np.arange(len(values)) / 480)[:, None]
    )
    for backend in ("numpy", "native"):
        renderer = sample_instrument.OfflineSampler(definition, backend)
        first = renderer.advance(actions, 0, 11)
        saved = renderer.snapshot()
        tail = renderer.advance([], 11, 48000)
        check_audio(
            tmp_path / f"plays-{backend}.wav", np.concatenate([first, tail]), expected
        )
        renderer.restore(saved)
        np.testing.assert_array_equal(renderer.advance([], 11, 48000), tail)
    source = sample_instrument.PersistentSampler(definition)
    engine = live.NativeLiveEngine({"sample": source}, 48000)
    actual = np.empty_like(expected)
    engine.advance_into({"sample": actions}, [], 0, 11, actual[:11])
    saved = engine.snapshot()
    engine.advance_into({"sample": []}, [], 11, 48000, actual[11:])
    check_audio(tmp_path / "plays-live.wav", actual, expected)
    engine.restore(saved)
    replay = np.empty_like(actual[11:])
    engine.advance_into({"sample": []}, [], 11, 48000, replay)
    np.testing.assert_array_equal(replay, actual[11:])


@pytest.mark.parametrize(
    "direction,overlap",
    [("forward", 0), ("backward", 0), ("mirror", 0), ("forward", 2), ("backward", 2)],
)
@pytest.mark.parametrize("repeats", [0, 1, 2, 3])
@pytest.mark.parametrize("step", [0.5, 1, 1.5, 2, 7, 100])
def test_finite_loops_finish_the_tail_and_restore_pending_turns(
    tmp_path: Path, direction: str, overlap: int, repeats: int, step: float
) -> None:
    raw = sample_score(frames=12).model_dump(mode="json")
    raw["body"]["slices"][0]["loop"] = {
        "start_frame": 3,
        "end_frame": 9,
        "crossfade_frames": overlap,
        "repeat_count": repeats,
        "mode": "through_release",
    }
    raw["body"]["slots"][0]["playback"]["direction"] = direction
    document = instrument.SampleInstrumentScore.model_validate(raw)
    audio = np.column_stack([np.arange(12) / 24, np.arange(12) / 48])
    definition = sample_instrument.prepare(document, {"asset": audio})
    actions = trace.prepare(
        document.body,
        [onset(pitch=440 * step).model_copy(update={"controls": {}})],
        seed=0,
    ).actions
    if direction == "mirror":
        indices = list(range(9))
        for i in range(repeats):
            indices.extend(range(7, 2, -1) if i % 2 == 0 else range(4, 9))
        indices.extend(range(2, -1, -1) if repeats % 2 else range(9, 12))
    elif overlap and repeats:
        indices = list(range(7)) if direction == "forward" else list(range(11, 4, -1))
        for i in range(repeats):
            indices.extend([7, 4] if direction == "forward" else [4, 7])
            if i + 1 < repeats:
                indices.extend([5, 6] if direction == "forward" else [6, 5])
        indices.extend(range(5, 12) if direction == "forward" else range(6, -1, -1))
    elif direction == "forward":
        indices = list(range(9)) + list(range(3, 9)) * repeats + list(range(9, 12))
    else:
        indices = (
            list(range(11, 2, -1))
            + list(range(8, 2, -1)) * repeats
            + list(range(2, -1, -1))
        )
    expected = np.zeros((48000, 2))
    positions = np.arange(0, len(indices), step)
    values = np.interp(positions, np.arange(len(indices)), indices)
    expected[: len(values)] = values[:, None] * [1 / 24, 1 / 32]
    for backend in ("numpy", "native"):
        renderer = sample_instrument.OfflineSampler(definition, backend)
        first = renderer.advance(actions, 0, 8)
        saved = renderer.snapshot()
        tail = renderer.advance([], 8, 48000)
        check_audio(
            tmp_path / f"finite-{backend}.wav", np.concatenate([first, tail]), expected
        )
        renderer.restore(saved)
        replay = np.concatenate(
            [renderer.advance([], 8, 9), renderer.advance([], 9, 48000)]
        )
        np.testing.assert_array_equal(replay, tail)
    source = sample_instrument.PersistentSampler(definition)
    engine = live.NativeLiveEngine({"sample": source}, 48000)
    actual = np.empty_like(expected)
    engine.advance_into({"sample": actions}, [], 0, 8, actual[:8])
    saved_live = engine.snapshot()
    engine.advance_into({"sample": []}, [], 8, 48000, actual[8:])
    check_audio(tmp_path / "finite-live.wav", actual, expected)
    assert not source.native_voices
    engine.restore(saved_live)
    replay = np.empty_like(actual[8:])
    engine.advance_into({"sample": []}, [], 8, 9, replay[:1])
    engine.advance_into({"sample": []}, [], 9, 48000, replay[1:])
    np.testing.assert_array_equal(replay, actual[8:])


@pytest.mark.parametrize("direction", ["forward", "backward", "mirror"])
def test_huge_steps_exhaust_even_the_largest_finite_loop_count(
    tmp_path: Path, direction: str
) -> None:
    definition = sampler.PreparedSample(
        samples=np.arange(1, 13, dtype=np.float64)[:, None] / 16,
        native_rate=48000,
        sample_rate=48000,
        slice=Slice(
            name="sample",
            asset="asset",
            end_frame=12,
            loop=Loop(start_frame=3, end_frame=9, repeat_count=2**64 - 1),
        ),
        playback=Playback.model_validate({"direction": direction}),
    )
    expected = np.zeros((48000, 1))
    expected[0] = (12 if direction == "backward" else 1) / 16
    for backend in ("numpy", "native"):
        actual, state = sampler.sample_frames(
            definition,
            sampler.SampleState.start(definition),
            np.full(48000, 1e100),
            backend=backend,
        )
        check_audio(tmp_path / f"huge-{backend}.wav", actual, expected)
        assert state.repeats_remaining == 0
        assert state.exhaustion_frame is not None


@pytest.mark.parametrize(
    "direction,overlap,release,indices",
    [
        ("forward", 0, 9, list(range(12))),
        ("backward", 0, 9, list(range(11, -1, -1))),
        ("mirror", 0, 8, list(range(12))),
        ("forward", 2, 8, [*range(7), 7, 4, *range(5, 12)]),
        ("backward", 2, 8, [*range(11, 4, -1), 4, 7, *range(6, -1, -1)]),
    ],
)
def test_release_cancels_finite_repeats_but_finishes_committed_overlap(
    tmp_path: Path, direction: str, overlap: int, release: int, indices: list[int]
) -> None:
    definition = sampler.PreparedSample(
        samples=np.arange(12, dtype=np.float64)[:, None] / 16,
        native_rate=48000,
        sample_rate=48000,
        slice=Slice(
            name="sample",
            asset="asset",
            end_frame=12,
            loop=Loop(
                start_frame=3, end_frame=9, crossfade_frames=overlap, repeat_count=2
            ),
        ),
        playback=Playback.model_validate({"direction": direction}),
    )
    expected = np.zeros((48000, 1))
    expected[: len(indices), 0] = np.array(indices) / 16
    for backend in ("numpy", "native"):
        actual, state = sampler.sample_frames(
            definition,
            sampler.SampleState.start(definition),
            np.ones(48000),
            Fraction(release),
            backend=backend,
        )
        check_audio(tmp_path / f"release-{backend}.wav", actual, expected)
        assert state.released
        assert state.repeats_remaining == (1 if overlap else 2)
