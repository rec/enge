"""Bounded recent-input capture with reverse, scrub and overlap-repeat readers."""

from math import floor
from typing import Literal

import numpy as np
from pydantic import BaseModel, ConfigDict, Field
from ufor.audio_effects import CAPTURE_FADE_FRAMES


class CaptureCommand(BaseModel):
    frame: int = Field(ge=0)
    release: bool

    model_config = ConfigDict(extra="forbid")


class CaptureRegion(BaseModel):
    samples: list[list[float]]
    cursor: int = Field(default=0, ge=0)
    active: bool = False
    gain: float = Field(default=0, ge=0, le=1)
    target: float = Field(default=0, ge=0, le=1)
    step: float = 0
    remaining: int = Field(default=0, ge=0, le=CAPTURE_FADE_FRAMES)

    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)


class CapturedBufferState(BaseModel):
    history: list[list[float]]
    write: int = Field(default=0, ge=0)
    count: int = Field(default=0, ge=0)
    regions: list[CaptureRegion] = Field(min_length=2, max_length=2)
    pending: list[CaptureCommand] = Field(default_factory=list)

    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)


def window_frames(seconds: float, sample_rate: int) -> int:
    frames = round(seconds * sample_rate)
    if frames < 2 * CAPTURE_FADE_FRAMES + 1:
        raise ValueError("Captured buffer window must contain at least 129 frames")
    return frames


def initial_state(frames: int, channels: int) -> CapturedBufferState:
    return CapturedBufferState(
        history=[[0.0] * channels for _ in range(frames)],
        regions=[
            CaptureRegion(samples=[[0.0] * channels for _ in range(frames)])
            for _ in range(2)
        ],
    )


def enqueue(
    state: CapturedBufferState, frame: int, release: bool, capacity: int
) -> None:
    if len(state.pending) >= capacity:
        raise ValueError("Captured buffer command capacity exhausted")
    state.pending.append(CaptureCommand(frame=frame, release=release))


def validate_state(
    state: CapturedBufferState,
    frames: int,
    channels: int,
    mode: str,
    capacity: int,
    frame: int,
) -> None:
    if (
        len(state.history) != frames
        or not 0 <= state.write < frames
        or not 0 <= state.count <= frames
    ):
        raise ValueError("Invalid capture history clock")
    for rows in [state.history, *[r.samples for r in state.regions]]:
        if len(rows) != frames or any(len(r) != channels for r in rows):
            raise ValueError("Invalid captured buffer history shape")
    for region in state.regions:
        limit = frames - CAPTURE_FADE_FRAMES if mode == "stutter" else frames
        if region.cursor > frames or (region.active and region.cursor >= limit):
            raise ValueError("Invalid captured reader cursor")
    if len(state.pending) > capacity:
        raise ValueError("Captured command capacity exceeded")
    previous = frame
    for command in state.pending:
        if command.frame < previous:
            raise ValueError("Invalid capture command ordering")
        previous = command.frame


def process(
    state: CapturedBufferState,
    source: np.ndarray,
    position: float,
    mode: Literal["reverse", "scrub", "stutter"],
    frame: int,
    ended: bool,
) -> tuple[np.ndarray, bool]:
    if not np.all(np.isfinite(source)) or not 0 <= position <= 1:
        raise ValueError("Invalid captured buffer input or position")
    frames = len(state.history)
    while state.pending and state.pending[0].frame <= frame:
        command = state.pending.pop(0)
        if command.release:
            for region in state.regions:
                _fade(region, 0)
        else:
            if ended or state.count < frames:
                raise ValueError(
                    "Capture requires live input and a complete history window"
                )
            region = next((r for r in state.regions if not r.active), None)
            if region is None:
                raise ValueError("Captured buffer regions occupied by a crossfade")
            for previous in state.regions:
                _fade(previous, 0)
            region.samples = [
                list(state.history[(state.write + i) % frames]) for i in range(frames)
            ]
            region.cursor = 0
            region.active = True
            region.gain = 0
            _fade(region, 1)
    if not ended:
        state.history[state.write] = source.tolist()
        state.write = (state.write + 1) % frames
        state.count = min(frames, state.count + 1)
    drained = ended and not state.pending and not any(r.active for r in state.regions)
    wet = source.copy()
    for region in state.regions:
        if not region.active:
            continue
        weight = region.gain
        if mode == "reverse":
            sample = np.asarray(region.samples[frames - 1 - region.cursor])
            weight *= min(1.0, (frames - 1 - region.cursor) / CAPTURE_FADE_FRAMES)
        elif mode == "scrub":
            index = position * (frames - 1)
            lower = floor(index)
            a = np.asarray(region.samples[lower])
            b = np.asarray(region.samples[min(lower + 1, frames - 1)])
            sample = a + (index - lower) * (b - a)
        else:
            offset = region.cursor
            period = frames - CAPTURE_FADE_FRAMES
            a = np.asarray(region.samples[offset])
            if offset < CAPTURE_FADE_FRAMES:
                b = np.asarray(region.samples[period + offset])
                mix = offset / CAPTURE_FADE_FRAMES
                sample = b * (1 - mix) + a * mix
            else:
                sample = a
        wet += weight * (sample - source)
        if mode != "scrub":
            region.cursor += 1
            if mode == "stutter":
                region.cursor %= frames - CAPTURE_FADE_FRAMES
            elif region.cursor == frames:
                region.active = False
        if region.remaining:
            region.remaining -= 1
            region.gain = (
                region.target if region.remaining == 0 else region.gain + region.step
            )
        if region.gain == 0 and region.target == 0:
            region.active = False
    if not np.all(np.isfinite(wet)):
        raise ValueError("Non-finite captured buffer output")
    return wet, drained


def _fade(region: CaptureRegion, target: float) -> None:
    region.target = target
    region.step = (target - region.gain) / CAPTURE_FADE_FRAMES
    region.remaining = CAPTURE_FADE_FRAMES
    if target == 0 and region.gain == 0:
        region.active = False
