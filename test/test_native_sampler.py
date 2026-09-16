from fractions import Fraction
from itertools import pairwise
from pathlib import Path

import numpy as np
import pytest
from test_sample_instrument import sample_score
from test_synth import check_audio
from ufor.envelope import Envelope, Segment
from ufor.samples.enums import Direction
from ufor.samples.playback import Loop, Playback, Slice

from enge import _native, sample_instrument, sampler, synth


@pytest.mark.parametrize("direction", list(Direction))
def test_native_sample_voice_matches_reference_without_python_dsp(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, direction: Direction
) -> None:
    frames = np.arange(96000)
    sample = sampler.PreparedSample(
        samples=np.column_stack([np.sin(frames / 100), np.cos(frames / 71)]),
        native_rate=44100,
        sample_rate=48000,
        slice=Slice(
            name="sample",
            asset="audio",
            end_frame=96000,
            loop=Loop(
                start_frame=100,
                end_frame=1000,
                crossfade_frames=0 if direction == Direction.mirror else 30,
            ),
        ),
        playback=Playback(direction=direction),
    )
    definition = sampler.PreparedSampleVoice(
        sample_rate=48000,
        slice=sample.slice,
        routes=[[1, -0.5, 0], [0.25, 0.75, 0]],
        gain=1.3,
        minimum_hold_seconds=Fraction(48005, 192000),
        envelope=Envelope(
            initial=0.2,
            segments=[
                Segment(duration=0, target=0.4),
                Segment(duration=Fraction(1, 2), target=1),
            ],
            release=[
                Segment(duration=Fraction(12001, 96000), target=0.7),
                Segment(duration=0, target=0.4),
                Segment(duration=Fraction(1, 8), target=0.25),
            ],
        ),
    )
    reference = sampler.SampleVoiceRenderer.start(definition, sample)
    compiled = sampler.SampleVoiceRenderer.start(definition, sample, backend="native")
    reference.release()
    compiled.release()
    ratios = np.linspace(0.1, 1.9, 96000)[::2]
    gains = np.linspace(0.2, 0.8, 96000)[::2]
    ratios.flags.writeable = gains.flags.writeable = False
    boundaries = [0, 120, 12001, 12002, 18002, 24002, 48000]
    expected = np.concatenate(
        [
            reference.render(sample, b - a, ratios[a:b], gains[a:b])
            for a, b in pairwise(boundaries)
        ]
    )
    buffer = sample.native_buffer

    def forbidden(*args: object, **kwargs: object) -> None:
        pytest.fail("Native sample rendering called Python DSP or recopied its asset")

    monkeypatch.setattr(sampler, "sample_frames", forbidden)
    monkeypatch.setattr(_native, "SampleBuffer", forbidden)
    for name in ("envelope_samples", "route_samples"):
        monkeypatch.setattr(synth, name, forbidden)
    chunks: list[np.ndarray] = []
    for start, end in pairwise(boundaries):
        compiled = sampler.SampleVoiceRenderer.model_validate_json(
            compiled.model_dump_json()
        )
        chunks.append(
            compiled.render(sample, end - start, ratios[start:end], gains[start:end])
        )
    actual = np.concatenate(chunks)
    check_audio(tmp_path / "native-sampler.wav", actual, expected)
    assert sample.native_buffer is buffer
    assert compiled.complete and reference.complete
    assert compiled.source.model_dump(
        exclude={"position", "error"}
    ) == reference.source.model_dump(exclude={"position", "error"})
    assert compiled.source.position == pytest.approx(
        reference.source.position, abs=1e-10, rel=1e-9
    )
    assert compiled.source.error == pytest.approx(
        reference.source.error, abs=1e-10, rel=1e-9
    )
    assert all(not np.shares_memory(a, b) for a, b in pairwise(chunks))
    assert not np.shares_memory(actual, sample.samples)
    assert not np.any(actual[:, 2])


def test_native_buffer_owns_strided_audio_and_keeps_large_frame_coordinates(
    tmp_path: Path,
) -> None:
    original = np.arange(96000, dtype=np.float64).reshape(48000, 2) / 96000
    source = original[:, ::-1]
    expected = source.copy()
    buffer = _native.SampleBuffer(source)
    original[:] = -1
    origin = 2**53 + 125
    actual, state = _native.render_sample(
        buffer,
        (0, 48000, False, None),
        (0, 0, 0, 1, False, False, False, None),
        origin,
        48000,
        np.full(48000, 48000.0),
        (123, False, 0, 48000),
        1,
        np.ones(48000),
        np.array([[0, 48000, 1, 0, 0, 1, 0]], dtype=np.float64),
        np.eye(2),
        48000,
    )
    check_audio(tmp_path / "owned-sample.wav", actual, expected)
    assert state == (48000, 0, 0, 1, False, False, True, None)
    tail, ended = _native.render_sample(
        buffer,
        (0, 48000, False, None),
        state,
        origin + 48000,
        48000,
        np.ones(1),
        None,
        1,
        np.ones(1),
        np.array([[0, 1, 1, 0, 0, 1, 0]], dtype=np.float64),
        np.eye(2),
        1,
    )
    assert not np.any(tail)
    assert ended[-1] == origin + 48000
    assert not np.shares_memory(actual, original)


@pytest.mark.parametrize(
    "invalid",
    [
        "slice",
        "loop",
        "overlap",
        "direction",
        "position",
        "steps",
        "gains",
        "span",
        "routes",
        "release",
        "frame",
    ],
)
def test_native_sample_rejects_invalid_inputs(invalid: str) -> None:
    buffer = _native.SampleBuffer(np.ones((8, 1)))
    selection = (0, 9 if invalid == "slice" else 8, False, (1, 4, 0, True))
    if invalid == "loop":
        selection = (0, 8, False, (2**63 - 1, -(2**63), 0, True))
    state = (
        5 if invalid == "direction" else 0,
        float("nan") if invalid == "position" else 0,
        0,
        1,
        True,
        invalid == "overlap",
        False,
        None,
    )
    steps = np.array([float("inf") if invalid == "steps" else 48000], dtype=np.float64)
    with pytest.raises(ValueError):
        _native.render_sample(
            buffer,
            selection,
            state,
            2**63 - 1 if invalid == "frame" else 0,
            48000,
            steps,
            (1, False, 0, 48000) if invalid == "release" else None,
            1,
            np.ones(2 if invalid == "gains" else 1),
            np.array(
                [[0, 2 if invalid == "span" else 1, 1, 0, 0, 1, 0]], dtype=np.float64
            ),
            np.ones((2 if invalid == "routes" else 1, 1)),
            1,
        )


def test_sampler_restore_rejects_another_backend_even_when_silent() -> None:
    prepared = sample_instrument.prepare(
        sample_score(frames=4), {"asset": np.ones((4, 2))}
    )
    reference = sample_instrument.OfflineSampler(prepared)
    compiled = sample_instrument.OfflineSampler(prepared, backend="native")
    with pytest.raises(synth.EngineError, match="different sampler backend"):
        compiled.restore(reference.snapshot())
