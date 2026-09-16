import json
from fractions import Fraction
from itertools import pairwise
from pathlib import Path
from typing import Literal

import numpy as np
import pytest
from pydantic import Field
from test_synth import check_audio
from ufor.base import Model
from ufor.samples.enums import Direction
from ufor.samples.playback import Loop, Playback, Slice

from enge.sampler import PreparedSample, SampleState, sample_frames
from enge.synth import EngineError


class TraversalVector(Model):
    name: str
    source: str
    slice: list[int]
    direction: Direction
    loop: Loop | None = None
    release_at: int | None = None
    pitch_ratios: list[float]
    expected_samples: list[list[float]]
    exhaustion_frame: int | None


class VectorSource(Model):
    native_rate: int
    samples: list[list[float]]


class TraversalVectors(Model):
    output_rate: int
    sources: dict[str, VectorSource]
    cases: list[TraversalVector] = Field(min_length=1)


VECTORS = TraversalVectors.model_validate_json(
    (Path(__file__).parents[1] / "conformance/sampler-traversal.json").read_text()
)


@pytest.mark.parametrize("vector", VECTORS.cases, ids=lambda c: c.name)
def test_traversal_vectors_survive_single_frame_restores(
    backend: Literal["numpy", "native"], tmp_path: Path, vector: TraversalVector
) -> None:
    source = VECTORS.sources[vector.source]
    definition = PreparedSample(
        samples=np.array(source.samples, dtype=np.float64),
        native_rate=source.native_rate,
        sample_rate=VECTORS.output_rate,
        slice=Slice(
            name=vector.name,
            asset=vector.source,
            start_frame=vector.slice[0],
            end_frame=vector.slice[1],
            loop=vector.loop,
        ),
        playback=Playback(direction=vector.direction),
    )
    initial = SampleState.start(definition)
    saved = initial.model_dump_json()
    ratios = np.array(vector.pitch_ratios)
    release = None if vector.release_at is None else Fraction(vector.release_at)
    whole, final = sample_frames(definition, initial, ratios, release, backend=backend)
    assert initial.model_dump_json() == saved
    state = initial
    chunks: list[np.ndarray] = []
    for ratio in ratios:
        state = SampleState.model_validate_json(state.model_dump_json())
        # Deliver release only when its frame arrives. Earlier lookahead must
        # not depend on knowing this future action in the whole-block call.
        audio, state = sample_frames(
            definition,
            state,
            np.array([ratio]),
            release if release is not None and release <= state.frame else None,
            backend=backend,
        )
        chunks.append(audio)
    np.testing.assert_array_equal(np.concatenate(chunks), whole)
    assert state == final
    assert final.exhaustion_frame == vector.exhaustion_frame
    actual = np.zeros((48000, whole.shape[1]))
    expected = np.zeros_like(actual)
    actual[: len(whole)] = whole
    expected[: len(whole)] = vector.expected_samples
    # The harness stops repeating sources after the specified prefix. Finite
    # sources must themselves return silence and preserve exhausted state.
    if vector.exhaustion_frame is not None:
        tail, ended = sample_frames(
            definition, final, np.ones(48000 - len(whole)), backend=backend
        )
        actual[len(whole) :] = tail
        assert ended.model_dump(exclude={"frame"}) == final.model_dump(
            exclude={"frame"}
        )
    check_audio(tmp_path / f"{vector.name}.wav", actual, expected)


@pytest.mark.parametrize("block", [64, 128, 256, 1024, 997])
def test_live_pitch_preserves_fractional_progress_across_blocks(
    backend: Literal["numpy", "native"], tmp_path: Path, block: int
) -> None:
    source_frames = np.arange(60000, dtype=np.float64)
    source = np.column_stack([source_frames / 60000, 1 - source_frames / 60000])
    definition = PreparedSample(
        samples=source,
        native_rate=24000,
        sample_rate=48000,
        slice=Slice(name="ramp", asset="stereo", end_frame=len(source)),
    )
    source[:] = -999
    with pytest.raises(ValueError):
        definition.samples.setflags(write=True)
    ratios = np.repeat([0.5, 1.25, 2, 0.125], 12000)
    ratios.setflags(write=False)
    state = SampleState.start(definition)
    chunks: list[np.ndarray] = []
    boundaries = sorted({0, 48000, 11999, 12000, 12001, *range(0, 48000, block)})
    for start, end in pairwise(boundaries):
        state = SampleState.model_validate_json(state.model_dump_json())
        output, state = sample_frames(
            definition, state, ratios[start:end], backend=backend
        )
        chunks.append(output)
    frames = np.arange(48000)
    progress = (
        np.minimum(frames, 12000) * 0.25
        + np.clip(frames - 12000, 0, 12000) * 0.625
        + np.clip(frames - 24000, 0, 12000)
        + np.maximum(frames - 36000, 0) * 0.0625
    )
    expected = np.column_stack([progress / 60000, 1 - progress / 60000])
    actual = np.concatenate(chunks)
    check_audio(tmp_path / "live-pitch.wav", actual, expected)
    whole, final = sample_frames(
        definition, SampleState.start(definition), ratios, backend=backend
    )
    np.testing.assert_array_equal(actual, whole)
    assert state == final
    assert state.exhaustion_frame is None
    assert state.index == 23250


@pytest.mark.parametrize("block", [64, 128, 256, 1024, 997])
@pytest.mark.parametrize(
    "direction,overlap,intro,cycle",
    [
        ("forward", 0, [0], [10, 20, 30]),
        ("backward", 0, [50, 40], [30, 20, 10]),
        ("mirror", 0, [0], [10, 20, 30, 20]),
        ("forward", 3, [0, 10, 20, 30], [40, 50, 40, 30]),
        ("backward", 3, [90, 80, 70, 60], [50, 40, 30, 40]),
    ],
)
def test_loop_resampling_matches_authored_traversal_over_live_pitch_changes(
    backend: Literal["numpy", "native"],
    tmp_path: Path,
    block: int,
    direction: Direction,
    overlap: int,
    intro: list[float],
    cycle: list[float],
) -> None:
    length = 10 if overlap else 6
    definition = PreparedSample(
        samples=np.arange(length, dtype=np.float64)[:, None] * 10,
        native_rate=44100,
        sample_rate=48000,
        slice=Slice(
            name="slice",
            asset="asset",
            end_frame=length,
            loop=Loop(start_frame=1, end_frame=length - 2, crossfade_frames=overlap),
        ),
        playback=Playback(direction=direction),
    )
    ratios = np.repeat([0.1, 1.2, 0.5, 1.6], 12000)
    frames = np.arange(48000)
    positions = (
        np.minimum(frames, 12000) * 4410
        + np.clip(frames - 12000, 0, 12000) * 52920
        + np.clip(frames - 24000, 0, 12000) * 22050
        + np.maximum(frames - 36000, 0) * 70560
    ) / 48000
    knots = np.concatenate([intro, np.tile(cycle, 48000 // len(cycle))])
    expected = np.interp(positions, np.arange(len(knots)), knots)[:, None]
    state = SampleState.start(definition)
    chunks: list[np.ndarray] = []
    boundaries = sorted({0, 48000, 11999, 12000, 12001, *range(0, 48000, block)})
    for start, end in pairwise(boundaries):
        state = SampleState.model_validate_json(state.model_dump_json())
        audio, state = sample_frames(
            definition, state, ratios[start:end], backend=backend
        )
        chunks.append(audio)
    actual = np.concatenate(chunks)
    check_audio(tmp_path / "resampled-loop.wav", actual, expected)
    whole, final = sample_frames(
        definition, SampleState.start(definition), ratios, backend=backend
    )
    np.testing.assert_array_equal(actual, whole)
    assert state == final


@pytest.mark.parametrize(
    "direction,overlap,release,expected_prefix",
    [
        ("mirror", 0, Fraction(5, 4), [0, 20, 40, 0, 0]),
        ("mirror", 0, Fraction(3, 2), [0, 20, 40, 0, 0]),
        ("mirror", 0, Fraction(7, 4), [0, 20, 20, 0, 0]),
        ("forward", 3, Fraction(5, 2), [0, 20, 40, 60, 80, 0, 0, 0]),
        ("forward", 3, Fraction(11, 4), [0, 20, 40, 40, 40, 60, 80, 0]),
    ],
)
def test_fractional_release_is_ordered_against_source_boundaries(
    backend: Literal["numpy", "native"],
    tmp_path: Path,
    direction: Direction,
    overlap: int,
    release: Fraction,
    expected_prefix: list[float],
) -> None:
    length = 10 if overlap else 6
    definition = PreparedSample(
        samples=np.arange(length, dtype=np.float64)[:, None] * 10,
        native_rate=48000,
        sample_rate=48000,
        slice=Slice(
            name="slice",
            asset="asset",
            end_frame=length,
            loop=Loop(start_frame=1, end_frame=length - 2, crossfade_frames=overlap),
        ),
        playback=Playback(direction=direction),
    )
    ratios = np.full(48000, 2.0)
    actual, final = sample_frames(
        definition, SampleState.start(definition), ratios, release, backend=backend
    )
    expected = np.zeros_like(actual)
    expected[: len(expected_prefix), 0] = expected_prefix
    check_audio(tmp_path / "fractional-release.wav", actual, expected)
    state = SampleState.start(definition)
    chunks: list[np.ndarray] = []
    for start, end in pairwise([0, 1, 2, 3, 4, 48000]):
        state = SampleState.model_validate_json(state.model_dump_json())
        audio, state = sample_frames(
            definition,
            state,
            ratios[start:end],
            release if release < end else None,
            backend=backend,
        )
        chunks.append(audio)
    np.testing.assert_array_equal(np.concatenate(chunks), actual)
    assert state == final


@pytest.mark.parametrize(
    "direction,overlap,first,large,expected_prefix",
    [
        ("forward", 0, 1, 3e12, [0, 10, 40, 50, 0]),
        ("mirror", 0, 1, 4e12, [0, 10, 10, 0, 0]),
        ("forward", 3, 5, 4e12, [0, 50, 50, 60, 70, 80, 90, 0]),
        ("forward", 0, 1, 3 * 2**64, [0, 10, 40, 50, 0]),
        ("forward", 0, 3 * 2**64, 1, [0, 30, 40, 50, 0]),
    ],
)
def test_large_steps_preserve_the_final_pending_boundary(
    backend: Literal["numpy", "native"],
    tmp_path: Path,
    direction: Direction,
    overlap: int,
    first: float,
    large: float,
    expected_prefix: list[float],
) -> None:
    length = 10 if overlap else 6
    definition = PreparedSample(
        samples=np.arange(length, dtype=np.float64)[:, None] * 10,
        native_rate=48000,
        sample_rate=48000,
        slice=Slice(
            name="slice",
            asset="asset",
            end_frame=length,
            loop=Loop(start_frame=1, end_frame=length - 2, crossfade_frames=overlap),
        ),
        playback=Playback(direction=direction),
    )
    ratios = np.ones(48000)
    ratios[:2] = first, large
    actual, state = sample_frames(
        definition, SampleState.start(definition), ratios, Fraction(2), backend=backend
    )
    expected = np.zeros_like(actual)
    expected[: len(expected_prefix), 0] = expected_prefix
    check_audio(tmp_path / "large-step.wav", actual, expected)
    assert state.exhaustion_frame == len(expected_prefix) - 1


def test_decimal_pitch_release_at_mirror_turn_keeps_incoming_direction(
    backend: Literal["numpy", "native"],
    tmp_path: Path,
) -> None:
    definition = PreparedSample(
        samples=np.arange(6, dtype=np.float64)[:, None] / 6,
        native_rate=48000,
        sample_rate=48000,
        slice=Slice(
            name="slice",
            asset="asset",
            end_frame=6,
            loop=Loop(start_frame=1, end_frame=4),
        ),
        playback=Playback(direction="mirror"),
    )
    actual, final = sample_frames(
        definition,
        SampleState.start(definition),
        np.full(48000, 0.1),
        Fraction(30),
        backend=backend,
    )
    expected = np.zeros_like(actual)
    expected[:60, 0] = np.minimum(np.arange(60) / 10, 5) / 6
    check_audio(tmp_path / "decimal-turn.wav", actual, expected)
    assert final.exhaustion_frame == 60


@pytest.mark.parametrize("speed", [7, 187])
def test_rational_release_at_a_turn_is_not_rounded_past_it(
    backend: Literal["numpy", "native"], tmp_path: Path, speed: int
) -> None:
    definition = PreparedSample(
        samples=np.arange(3 * speed, dtype=np.float64)[:, None] / speed,
        native_rate=48000,
        sample_rate=48000,
        slice=Slice(
            name="slice",
            asset="asset",
            end_frame=3 * speed,
            loop=Loop(start_frame=1, end_frame=4),
        ),
        playback=Playback(direction="mirror"),
    )
    actual, final = sample_frames(
        definition,
        SampleState.start(definition),
        np.full(48000, float(speed)),
        Fraction(3, speed),
        backend=backend,
    )
    expected = np.zeros_like(actual)
    expected[:3, 0] = [0, 1, 2]
    check_audio(tmp_path / "rational-turn.wav", actual, expected)
    assert final.exhaustion_frame == 3


@pytest.mark.parametrize("release", [Fraction(1, 2), Fraction(2)])
def test_exhausted_source_ignores_later_releases(
    backend: Literal["numpy", "native"], tmp_path: Path, release: Fraction
) -> None:
    definition = PreparedSample(
        samples=np.array([[0.5]]),
        native_rate=48000,
        sample_rate=48000,
        slice=Slice(name="slice", asset="asset", end_frame=1),
    )
    actual, final = sample_frames(
        definition,
        SampleState.start(definition),
        np.full(48000, 4.0),
        release,
        backend=backend,
    )
    expected = np.zeros_like(actual)
    expected[0] = 0.5
    check_audio(tmp_path / "exhausted-release.wav", actual, expected)
    assert final.exhaustion_frame == 1
    assert not final.released
    tail, ended = sample_frames(
        definition, final, np.ones(48000), Fraction(48000), backend=backend
    )
    assert not np.any(tail)
    assert ended.model_dump(exclude={"frame"}) == final.model_dump(exclude={"frame"})


@pytest.mark.parametrize(
    "ratios", [[0], [-1], [float("nan")], [float("inf")], [[1]], 1]
)
def test_invalid_pitch_does_not_change_source_state(
    backend: Literal["numpy", "native"], ratios: object
) -> None:
    definition = PreparedSample(
        samples=np.zeros((2, 1)),
        native_rate=48000,
        sample_rate=48000,
        slice=Slice(name="slice", asset="asset", end_frame=2),
    )
    state = SampleState.start(definition)
    before = json.loads(state.model_dump_json())
    with pytest.raises(EngineError, match="pitch ratios"):
        sample_frames(definition, state, np.array(ratios), backend=backend)
    assert state.model_dump() == before
