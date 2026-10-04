"""Blockwise realization of identified sample regions through the sampler."""

from __future__ import annotations

from typing import Literal, Self

import numpy as np
from pydantic import BaseModel, ConfigDict, Field, model_validator
from ufor.arpeggiator_capture import CapturedPhrase, SourceNote
from ufor.samples.playback import Slice

from .sampler import PreparedSample, SampleState, sample_frames
from .synth import EngineError


class RegionPlayer(BaseModel):
    """Own one ordered region performance and its current sample cursor."""

    phrase: CapturedPhrase
    notes: list[SourceNote]
    audio: dict[str, np.ndarray]
    channels: list[int]
    backend: Literal["numpy", "native"] = "numpy"
    seam_fade_frames: int = 0
    tail_policy: Literal["whole", "gate"] = "whole"
    runs: list[_RegionRun] = Field(default_factory=list)
    run_index: int = 0
    run_offset: int = 0
    definition: PreparedSample | None = None
    state: SampleState | None = None

    @model_validator(mode="after")
    def prepare(self) -> Self:
        rate = self.phrase.timebase.rate
        if (
            rate.denominator != 1
            or not self.channels
            or len(set(self.channels)) != len(self.channels)
        ):
            raise EngineError(
                "region playback requires an integer rate and selected channels"
            )
        if self.seam_fade_frames < 0:
            raise EngineError("seam fade frames must be nonnegative")
        known = {n.note_id: n for n in self.phrase.notes}
        if any(known.get(n.note_id) != n or n.region is None for n in self.notes):
            raise EngineError("region note must belong to the source phrase")
        if any(n.region is not None and n.region.loop is not None for n in self.notes):
            raise EngineError("looped regions require a playback policy")
        runs: list[_RegionRun] = []
        for note in self.notes:
            region = note.region
            assert region is not None
            end = (
                region.end_frame if self.tail_policy == "whole" else note.gate_end_tick
            )
            if not region.start_frame < end <= region.end_frame:
                raise EngineError("sample gate must end within its region")
            decoded = self.audio.get(region.asset)
            if decoded is None or any(
                c < 0 or c >= decoded.shape[1] for c in self.channels
            ):
                raise EngineError("region asset or channel is unavailable")
            if (
                runs
                and runs[-1].asset == region.asset
                and runs[-1].end_frame == region.start_frame
            ):
                runs[-1] = runs[-1].model_copy(update={"end_frame": end})
            else:
                runs.append(
                    _RegionRun(
                        name=region.name,
                        asset=region.asset,
                        start_frame=region.start_frame,
                        end_frame=end,
                    )
                )
        if self.seam_fade_frames and any(
            self.seam_fade_frames > min(a.frames, b.frames)
            for a, b in zip(runs, runs[1:], strict=False)
        ):
            raise EngineError("seam fade exceeds an adjacent region run")
        self.runs = runs
        return self

    def advance(self, frames: int) -> np.ndarray:
        """Render a live-sized block and retain source state for the next pull."""
        if frames < 0:
            raise EngineError("region block length must be nonnegative")
        output = np.zeros((frames, len(self.channels)), dtype=np.float64)
        offset = 0
        while offset < frames and self.run_index < len(self.runs):
            run = self.runs[self.run_index]
            if self.definition is None:
                self.definition = PreparedSample(
                    samples=self.audio[run.asset],
                    native_rate=self.phrase.timebase.rate.numerator,
                    sample_rate=self.phrase.timebase.rate.numerator,
                    slice=Slice(
                        name=run.name,
                        asset=run.asset,
                        start_frame=run.start_frame,
                        end_frame=run.end_frame,
                    ),
                )
                self.state = SampleState.start(self.definition)
            assert self.state is not None
            count = min(frames - offset, run.frames - self.run_offset)
            rendered, self.state = sample_frames(
                self.definition,
                self.state,
                np.ones(count, dtype=np.float64),
                backend=self.backend,
            )
            gains = np.ones(count, dtype=np.float64)
            if self.seam_fade_frames:
                positions = self.run_offset + np.arange(count)
                if self.run_index:
                    gains *= np.minimum(positions / self.seam_fade_frames, 1)
                if self.run_index + 1 < len(self.runs):
                    gains *= np.minimum(
                        (run.frames - 1 - positions) / self.seam_fade_frames, 1
                    )
            output[offset : offset + count] = (
                rendered[:, self.channels] * gains[:, None]
            )
            offset += count
            self.run_offset += count
            if self.run_offset == run.frames:
                self.run_index += 1
                self.run_offset = 0
                self.definition = None
                self.state = None
        return output

    def stop(self) -> None:
        self.run_index = len(self.runs)
        self.run_offset = 0
        self.definition = None
        self.state = None

    @property
    def total_frames(self) -> int:
        return sum(r.frames for r in self.runs)

    model_config = ConfigDict(arbitrary_types_allowed=True)


def render_regions(
    phrase: CapturedPhrase,
    notes: list[SourceNote],
    audio: dict[str, np.ndarray],
    channels: list[int],
    backend: Literal["numpy", "native"] = "numpy",
    seam_fade_frames: int = 0,
    tail_policy: Literal["whole", "gate"] = "whole",
) -> np.ndarray:
    """Render a complete region performance through the blockwise player."""
    player = RegionPlayer(
        phrase=phrase,
        notes=notes,
        audio=audio,
        channels=channels,
        backend=backend,
        seam_fade_frames=seam_fade_frames,
        tail_policy=tail_policy,
    )
    return player.advance(player.total_frames)


class _RegionRun(BaseModel, frozen=True):
    name: str
    asset: str
    start_frame: int
    end_frame: int

    @property
    def frames(self) -> int:
        return self.end_frame - self.start_frame
