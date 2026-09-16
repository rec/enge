"""NumPy reference for resolved uFor sample traversal and linear interpolation."""

from copy import copy
from fractions import Fraction
from typing import Literal, Self

import numpy as np
from pydantic import ConfigDict, Field, field_validator, model_validator
from ufor.base import Model
from ufor.samples.enums import Direction, LoopMode, PlaybackMode
from ufor.samples.playback import Playback, Slice

from .synth import EngineError


class PreparedSample(Model, frozen=True):
    """One shared decoded asset and a resolved slice; no file I/O during render."""

    samples: np.ndarray
    native_rate: int = Field(strict=True, gt=0)
    sample_rate: int = Field(strict=True, gt=0)
    slice: Slice
    playback: Playback = Playback()

    @field_validator("samples")
    @classmethod
    def immutable_audio(cls, samples: np.ndarray) -> np.ndarray:
        if samples.ndim != 2 or not all(samples.shape):
            raise EngineError("Sample audio must have shape (frames, channels)")
        if samples.dtype != np.float64 or not np.all(np.isfinite(samples)):
            raise EngineError("Sample audio must contain finite float64 values")
        # Bytes own the prepared copy, so callers cannot re-enable writes through
        # a view or mutate the asset by changing the original decoding buffer.
        return np.frombuffer(samples.tobytes(), dtype=np.float64).reshape(samples.shape)

    @model_validator(mode="after")
    def supported_traversal(self) -> Self:
        if self.slice.end_frame > len(self.samples):
            raise EngineError("Sample slice exceeds decoded audio")
        if self.slice.loop is not None:
            if self.playback.mode != PlaybackMode.while_held:
                raise EngineError("Sample loops require while_held playback")
            if (
                self.playback.direction == Direction.mirror
                and self.slice.loop.crossfade_frames
            ):
                raise EngineError("Mirror loops cannot crossfade")
        return self

    model_config = ConfigDict(arbitrary_types_allowed=True)


class SampleState(Model, frozen=True):
    """State before the next output frame, including uncommitted boundary choices.

    Serialize this value and restore with the same prepared sample. Asset storage
    stays outside numerical state, as oscillator settings do for OscillatorState.
    Position and its addition error use output-rate units, before division.
    """

    frame: int = 0
    index: int
    position: float = 0
    error: float = 0
    direction: Literal[-1, 1] = 1
    looping: bool = False
    overlap: bool = False
    released: bool = False
    exhaustion_frame: int | None = None

    @classmethod
    def start(cls, definition: PreparedSample) -> Self:
        backward = definition.playback.direction == Direction.backward
        return cls(
            index=definition.slice.end_frame - 1
            if backward
            else definition.slice.start_frame,
            direction=-1 if backward else 1,
            looping=definition.slice.loop is not None,
        )


def sample_frames(
    definition: PreparedSample,
    state: SampleState,
    pitch_ratios: np.ndarray,
    release_frame: Fraction | None = None,
) -> tuple[np.ndarray, SampleState]:
    """Read, then advance for each ratio; return audio and independent next state.

    release_frame is the effective voice-relative release coordinate, after uFor
    sustain and the voice's minimum hold. An enclosing voice handles envelopes,
    gain, routing and stop. Ratios already contain resolved tuning, but not the
    native/output rate factor. One-shot key-release policy belongs to uFor.
    """
    if pitch_ratios.ndim != 1 or not np.all(
        np.isfinite(pitch_ratios) & (pitch_ratios > 0)
    ):
        raise EngineError("Sample pitch ratios must be a positive finite vector")
    with np.errstate(over="ignore"):
        steps = pitch_ratios.astype(np.float64) * definition.native_rate
    if not np.all(np.isfinite(steps)):
        raise EngineError("Sample rate-adjusted pitch steps must be finite")
    if release_frame is not None and release_frame < state.frame and not state.released:
        raise EngineError("Sample release cannot precede the current cursor")

    # A local mutable cursor avoids constructing a Pydantic model at every knot.
    cursor = _Cursor(definition, state)
    output = np.zeros((len(steps), definition.samples.shape[1]), dtype=np.float64)
    for i, step in enumerate(steps):
        if cursor.exhaustion_frame is not None:
            break
        frame = state.frame + i
        if release_frame == frame:
            cursor.release()
        cursor.settle(frame)
        if cursor.exhaustion_frame is not None:
            break
        first = cursor.read()
        if cursor.position:
            # Lookahead uses a separate cursor and cannot commit a wrap, turn,
            # or overlap that a release at the next frame might cancel.
            following = copy(cursor)
            following.index += following.direction
            following.position = 0
            following.settle(frame)
            last = first if following.exhaustion_frame is not None else following.read()
            output[i] = first + (cursor.position / definition.sample_rate) * (
                last - first
            )
        else:
            output[i] = first
        if release_frame is not None and frame < release_frame < frame + 1:
            # Multiply the rational time before converting to float. Rounding
            # time first can put an exactly coincident release past a turn.
            before = Fraction(float(step)) * (release_frame - frame)
            cursor.advance(float(before), frame + 1)
            if cursor.exhaustion_frame is None:
                cursor.release()
                cursor.settle(frame + 1)
                cursor.advance(float(Fraction(float(step)) - before), frame + 1)
        else:
            cursor.advance(float(step), frame + 1)
    return output, SampleState(
        frame=state.frame + len(steps),
        index=cursor.index,
        position=cursor.position,
        error=cursor.error,
        direction=cursor.direction,
        looping=cursor.looping,
        overlap=cursor.overlap,
        released=cursor.released,
        exhaustion_frame=cursor.exhaustion_frame,
    )


class _Cursor:
    def __init__(self, definition: PreparedSample, state: SampleState) -> None:
        self.definition = definition
        self.index = state.index
        self.position = state.position
        self.error = state.error
        self.direction = state.direction
        self.looping = state.looping
        self.overlap = state.overlap
        self.released = state.released
        self.exhaustion_frame = state.exhaustion_frame

    def release(self) -> None:
        self.released = True
        loop = self.definition.slice.loop
        if loop is not None and loop.mode == LoopMode.until_release:
            self.looping = False

    def settle(self, frame: int) -> None:
        if self.exhaustion_frame is not None:
            return
        selection = self.definition.slice
        loop = selection.loop
        mirror = self.definition.playback.direction == Direction.mirror
        if loop is not None:
            if self.overlap:
                if self.direction == 1 and self.index == loop.end_frame:
                    self.index = loop.start_frame + loop.crossfade_frames
                    self.overlap = False
                elif self.direction == -1 and self.index == loop.start_frame - 1:
                    self.index = loop.end_frame - 1 - loop.crossfade_frames
                    self.overlap = False
            elif self.looping:
                if mirror:
                    if self.direction == 1 and self.index == loop.end_frame - 1:
                        self.direction = -1
                    elif self.direction == -1 and self.index == loop.start_frame:
                        self.direction = 1
                elif loop.crossfade_frames:
                    self.overlap = self.index == (
                        loop.end_frame - loop.crossfade_frames
                        if self.direction == 1
                        else loop.start_frame + loop.crossfade_frames - 1
                    )
                elif self.direction == 1 and self.index == loop.end_frame:
                    self.index = loop.start_frame
                elif self.direction == -1 and self.index == loop.start_frame - 1:
                    self.index = loop.end_frame - 1
        elif (
            mirror
            and self.direction == 1
            and self.index == selection.end_frame - 1
            and selection.end_frame - selection.start_frame > 1
        ):
            self.direction = -1
        if not selection.start_frame <= self.index < selection.end_frame:
            self.exhaustion_frame = frame
            self.position = self.error = 0

    def read(self) -> np.ndarray:
        audio = self.definition.samples
        loop = self.definition.slice.loop
        if not self.overlap:
            return audio[self.index]
        assert loop is not None
        if self.direction == 1:
            offset = self.index - (loop.end_frame - loop.crossfade_frames)
            incoming = loop.start_frame + offset
        else:
            offset = loop.start_frame + loop.crossfade_frames - 1 - self.index
            incoming = loop.end_frame - 1 - offset
        weight = offset / (loop.crossfade_frames - 1)
        return (1 - weight) * audio[self.index] + weight * audio[incoming]

    def advance(self, step: float, frame: int) -> None:
        if self.exhaustion_frame is not None:
            return
        increment = step - self.error
        total = self.position + increment
        self.error = (total - self.position) - increment
        whole, self.position = divmod(total, self.definition.sample_rate)
        count = int(whole)
        while count:
            boundary, period = self.boundary()
            # Skip complete repetitions but retain the last one. Its final
            # boundary may coincide exactly with a not-yet-delivered release.
            if period:
                count -= max(0, (count - 1) // period) * period
            move = min(count, self.direction * (boundary - self.index))
            self.index += self.direction * move
            count -= move
            if count or self.position:
                self.settle(frame)
                if self.exhaustion_frame is not None:
                    return

    def boundary(self) -> tuple[int, int]:
        selection = self.definition.slice
        loop = selection.loop
        mirror = self.definition.playback.direction == Direction.mirror
        if loop is not None and (self.looping or self.overlap):
            length = loop.end_frame - loop.start_frame
            period = 0
            if mirror:
                if loop.start_frame <= self.index < loop.end_frame:
                    period = 2 * (length - 1)
                return (
                    loop.end_frame - 1 if self.direction == 1 else loop.start_frame,
                    period,
                )
            overlap = loop.crossfade_frames
            if self.direction == 1:
                if (
                    self.looping
                    and loop.start_frame + overlap <= self.index < loop.end_frame
                ):
                    period = length - overlap
                return loop.end_frame - (0 if self.overlap else overlap), period
            if (
                self.looping
                and loop.start_frame <= self.index < loop.end_frame - overlap
            ):
                period = length - overlap
            return loop.start_frame - 1 + (0 if self.overlap else overlap), period
        if (
            mirror
            and loop is None
            and self.direction == 1
            and selection.end_frame - selection.start_frame > 1
        ):
            return selection.end_frame - 1, 0
        return (
            selection.end_frame if self.direction == 1 else selection.start_frame - 1,
            0,
        )
