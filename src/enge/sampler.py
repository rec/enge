"""Reference and native rendering of resolved uFor sample traversal."""

from copy import copy
from fractions import Fraction
from functools import cached_property
from typing import TYPE_CHECKING, Literal, Self

import numpy as np
from pydantic import ConfigDict, Field, field_validator, model_validator
from ufor.base import Model
from ufor.samples.enums import Direction, LoopMode, PlaybackMode
from ufor.samples.playback import Playback, Slice

from . import synth
from .synth import EngineError

if TYPE_CHECKING:
    from ._native import SampleBuffer


class PreparedSample(Model, frozen=True):
    """One shared decoded asset and a resolved slice; no file I/O during render."""

    samples: np.ndarray
    native_rate: int = Field(strict=True, gt=0)
    sample_rate: int = Field(strict=True, gt=0)
    slice: Slice
    playback: Playback = Playback()

    @cached_property
    def native_buffer(self) -> "SampleBuffer":
        """One owned Rust audio copy, reused by voices and restored cursors."""
        from . import _native

        return _native.SampleBuffer(self.samples)

    @field_validator("samples")
    @classmethod
    def immutable_audio(cls, samples: np.ndarray) -> np.ndarray:
        if samples.ndim != 2 or not all(samples.shape):
            raise EngineError("Sample audio must have shape (frames, channels)")
        if samples.dtype != np.float64 or not np.all(np.isfinite(samples)):
            raise EngineError("Sample audio must contain finite float64 values")
        owner = samples
        while isinstance(owner, np.ndarray):
            owner = owner.base
        if isinstance(owner, bytes) and samples.flags.c_contiguous:
            return samples
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


class PreparedSampleVoice(synth.PreparedEnvelope, frozen=True):
    """Resolved voice settings; decoded audio stays in the prepared sample."""

    slice: Slice
    routes: list[list[float]] = Field(min_length=1)
    pitch_ratio: float = Field(default=1, gt=0)
    gain: float = 1

    @model_validator(mode="after")
    def channel_routes(self) -> Self:
        if not self.routes[0] or any(
            len(r) != len(self.routes[0]) for r in self.routes
        ):
            raise EngineError("Sample route rows must have the same output channels")
        return self


class SampleVoiceRenderer(synth.EnvelopeRenderer):
    """A resumable sample voice, with the same envelope timing as the synth."""

    definition: PreparedSampleVoice
    source: SampleState
    backend: Literal["numpy", "native"] = "numpy"

    @classmethod
    def start(
        cls,
        definition: PreparedSampleVoice,
        sample: PreparedSample,
        backend: Literal["numpy", "native"] = "numpy",
    ) -> Self:
        if (
            definition.slice != sample.slice
            or definition.sample_rate != sample.sample_rate
        ):
            raise EngineError("Sample voice must match its prepared source")
        if len(definition.routes) != sample.samples.shape[1]:
            raise EngineError("Sample routes must map every source channel")
        return cls(
            definition=definition, source=SampleState.start(sample), backend=backend
        )

    @property
    def complete(self) -> bool:
        source = self.source
        loop = self.definition.slice.loop
        looping = source.looping and not (
            loop is not None
            and loop.mode == LoopMode.until_release
            and self.release_frame is not None
            and self.release_frame <= self.frame_count
        )
        exhausted = source.exhaustion_frame is not None or (
            not looping
            and not source.overlap
            and not self.definition.slice.start_frame
            <= source.index
            < self.definition.slice.end_frame
        )
        return exhausted or super().complete

    def release(self) -> bool:
        if self.complete:
            return False
        return super().release()

    def active_frames(self, frames: int) -> int:
        return 0 if self.complete else super().active_frames(frames)

    def render(
        self,
        sample: PreparedSample,
        frames: int,
        pitch_ratios: np.ndarray | None = None,
        gains: np.ndarray | None = None,
    ) -> np.ndarray:
        """Render source channels through the voice envelope and explicit routes.

        Restore serialized voices with the same immutable prepared source. Pitch
        arrays replace the prepared ratio; gain arrays multiply prepared gain.
        """
        count = self.active_frames(frames)
        if count and self.backend == "native":
            from . import native, native_sample

            output, self.source = native_sample.render(
                sample,
                self.source,
                pitch_steps(
                    sample,
                    self.source,
                    np.full(count, self.definition.pitch_ratio)
                    if pitch_ratios is None
                    else pitch_ratios[:count],
                    self.release_frame,
                ),
                self.release_frame,
                self.definition.gain,
                np.ones(count) if gains is None else gains[:count],
                native.envelope_spans(
                    self.definition.envelope,
                    self.frame_count,
                    count,
                    self.definition.sample_rate,
                    self.release_frame,
                ),
                np.asarray(self.definition.routes, dtype=np.float64),
                frames,
            )
            self.frame_count += frames
            return output
        output = np.zeros((frames, len(self.definition.routes[0])))
        if count:
            audio, self.source = sample_frames(
                sample,
                self.source,
                np.full(count, self.definition.pitch_ratio)
                if pitch_ratios is None
                else pitch_ratios[:count],
                self.release_frame,
            )
            amplitude = (
                synth.envelope_samples(
                    self.definition.envelope,
                    self.frame_count,
                    count,
                    self.definition.sample_rate,
                    self.release_frame,
                )
                * self.definition.gain
            )
            if gains is not None:
                amplitude *= gains[:count]
            output[:count] = (
                synth.route_samples(audio, self.definition.routes) * amplitude[:, None]
            )
        self.frame_count += frames
        return output


def sample_frames(
    definition: PreparedSample,
    state: SampleState,
    pitch_ratios: np.ndarray,
    release_frame: Fraction | None = None,
    backend: Literal["numpy", "native"] = "numpy",
) -> tuple[np.ndarray, SampleState]:
    """Read, then advance for each ratio; return audio and independent next state.

    release_frame is the effective voice-relative release coordinate, after uFor
    sustain and the voice's minimum hold. An enclosing voice handles envelopes,
    gain, routing and stop. Ratios already contain resolved tuning, but not the
    native/output rate factor. One-shot key-release policy belongs to uFor.
    """
    steps = pitch_steps(definition, state, pitch_ratios, release_frame)
    if backend == "native":
        from . import native_sample

        count = len(steps)
        return native_sample.render(
            definition,
            state,
            steps,
            release_frame,
            1,
            np.ones(count),
            np.array([[0, count, 1, 0, 0, 1, 0]], dtype=np.float64),
            np.eye(definition.samples.shape[1]),
            count,
        )
    if backend != "numpy":
        raise EngineError(f"Unknown sampler backend: {backend}")

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


def pitch_steps(
    definition: PreparedSample,
    state: SampleState,
    pitch_ratios: np.ndarray,
    release_frame: Fraction | None,
) -> np.ndarray:
    """Validate control inputs and convert ratios to output-rate cursor units."""
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
    return steps


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
