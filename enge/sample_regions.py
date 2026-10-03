"""Realize identified sample regions through the existing prepared sampler."""

from typing import Literal

import numpy as np
from ufor.arpeggiator_capture import CapturedPhrase, SourceNote
from ufor.samples.playback import Slice

from .sampler import PreparedSample, SampleState, sample_frames
from .synth import EngineError


def render_regions(
    phrase: CapturedPhrase,
    notes: list[SourceNote],
    audio: dict[str, np.ndarray],
    channels: list[int],
    backend: Literal["numpy", "native"] = "numpy",
    seam_fade_frames: int = 0,
) -> np.ndarray:
    """Play source-rate regions, fading only discontinuous source boundaries."""
    rate = phrase.timebase.rate
    if seam_fade_frames < 0:
        raise EngineError("seam fade frames must be nonnegative")
    if rate.denominator != 1 or not channels or len(set(channels)) != len(channels):
        raise EngineError(
            "region playback requires an integer rate and selected channels"
        )
    known = {n.note_id: n for n in phrase.notes}
    if any(known.get(n.note_id) != n or n.region is None for n in notes):
        raise EngineError("region note must belong to the source phrase")
    if any(n.region is not None and n.region.loop is not None for n in notes):
        raise EngineError("looped regions require a playback policy")
    if not notes:
        return np.empty((0, len(channels)), dtype=np.float64)
    chunks: list[np.ndarray] = []
    start = notes[0].region
    assert start is not None
    end_frame = start.end_frame
    for note in [*notes[1:], None]:
        region = None if note is None else note.region
        if (
            region is not None
            and region.asset == start.asset
            and region.start_frame == end_frame
        ):
            end_frame = region.end_frame
            continue
        decoded = audio.get(start.asset)
        if decoded is None or any(c < 0 or c >= decoded.shape[1] for c in channels):
            raise EngineError("region asset or channel is unavailable")
        definition = PreparedSample(
            samples=decoded,
            native_rate=rate.numerator,
            sample_rate=rate.numerator,
            slice=Slice(
                name=start.name,
                asset=start.asset,
                start_frame=start.start_frame,
                end_frame=end_frame,
            ),
        )
        count = end_frame - start.start_frame
        rendered, _ = sample_frames(
            definition,
            SampleState.start(definition),
            np.ones(count, dtype=np.float64),
            backend=backend,
        )
        chunk = rendered[:, channels]
        if chunks and seam_fade_frames:
            if seam_fade_frames > min(len(chunks[-1]), len(chunk)):
                raise EngineError("seam fade exceeds an adjacent region run")
            fade = np.arange(1, seam_fade_frames + 1, dtype=np.float64)
            fade /= seam_fade_frames
            chunks[-1][-seam_fade_frames:] *= (1 - fade)[:, None]
            chunk[:seam_fade_frames] *= (fade - 1 / seam_fade_frames)[:, None]
        chunks.append(chunk)
        if region is not None:
            start = region
            end_frame = region.end_frame
    return np.concatenate(chunks)
