"""Bounded, clocked real-time time stretching for finite audio segments.

The Rust stretcher compensates its own startup pad and delay. Each ``advance``
accepts one input clock block and returns one equally sized output clock block.
The final block is padded with silence to the clock boundary. A shortfall in
Rubber Band's approximate real-time output count is also filled with silence.
"""

from collections import deque
from math import ceil, isfinite

import numpy as np

from . import _native
from .synth import EngineError


class RealtimeRubberBand:
    """Stretch a known-length stream with a bounded output queue.

    ``ratio`` is output duration divided by input duration. ``source_frames``
    fixes the segment boundary; ``max_buffer_frames`` is a hard limit on queued
    output frames. Once the output boundary is reached, all remaining buffers
    and the native stretcher are released. ``underrun_frames`` counts silence
    inserted when Rubber Band produces fewer frames than the boundary needs.
    """

    def __init__(
        self,
        sample_rate: int,
        channels: int,
        ratio: float,
        source_frames: int,
        block_frames: int,
        max_buffer_frames: int,
        pitch: float = 1.0,
    ) -> None:
        if (
            sample_rate <= 0
            or channels <= 0
            or source_frames <= 0
            or block_frames <= 0
            or max_buffer_frames <= 0
            or not isfinite(ratio)
            or ratio <= 0
            or not isfinite(pitch)
            or pitch <= 0
        ):
            raise EngineError("Invalid real-time Rubber Band settings")
        self.channels = channels
        self.source_frames = source_frames
        self.block_frames = block_frames
        self.max_buffer_frames = max_buffer_frames
        self.output_frames = round(source_frames * ratio)
        if self.output_frames <= 0:
            raise EngineError("Real-time Rubber Band output must contain a frame")
        if not hasattr(_native, "RubberBandRealtimeStretcher"):
            raise EngineError(
                "Real-time Rubber Band requires the rubberband build feature"
            )
        self._stretcher = _native.RubberBandRealtimeStretcher(
            sample_rate, channels, ratio, pitch, block_frames
        )
        # The finite speedup segment must be available before its shorter
        # playback window ends. Four extra blocks cover the first FFT window
        # and fixed-size clock quantization before output becomes available.
        self.startup_frames = (
            max(0, ceil(source_frames * (1 - ratio)))
            + self._stretcher.start_delay
            + 4 * block_frames
        )
        self._queue: deque[np.ndarray] = deque()
        self._queue_offset = 0
        self.buffered_frames = 0
        self.source_received = 0
        self.playhead = 0
        self.underrun_frames = 0
        self.complete = False

    def advance(self, audio: np.ndarray) -> np.ndarray:
        """Consume the next input block and produce one output clock block."""
        if self.complete:
            raise EngineError("Real-time Rubber Band segment is complete")
        expected = min(self.block_frames, self.source_frames - self.source_received)
        if audio.ndim != 2 or audio.shape != (expected, self.channels):
            raise EngineError(
                "Real-time Rubber Band input must match the next clock block"
            )
        if expected:
            self.source_received += expected
            if self._stretcher is None:
                raise EngineError("Real-time Rubber Band segment is closed")
            produced = self._stretcher.process(
                audio, self.source_received == self.source_frames
            )
            if len(produced):
                self._queue.append(produced)
                self.buffered_frames += len(produced)
        if self.buffered_frames > self.max_buffer_frames:
            self._close()
            self.complete = True
            raise EngineError("Real-time Rubber Band buffer limit exceeded")

        result = np.zeros((self.block_frames, self.channels))
        first = max(self.playhead, self.startup_frames)
        last = min(
            self.playhead + self.block_frames,
            self.startup_frames + self.output_frames,
        )
        cursor = first - self.playhead
        remaining = max(0, last - first)
        while remaining and self._queue:
            head = self._queue[0]
            count = min(remaining, len(head) - self._queue_offset)
            result[cursor : cursor + count] = head[
                self._queue_offset : self._queue_offset + count
            ]
            self._queue_offset += count
            self.buffered_frames -= count
            cursor += count
            remaining -= count
            if self._queue_offset == len(head):
                self._queue.popleft()
                self._queue_offset = 0
        self.underrun_frames += remaining
        self.playhead += self.block_frames
        if self.playhead >= self.startup_frames + self.output_frames:
            self._close()
            self.complete = True
        return result

    def _close(self) -> None:
        self._queue.clear()
        self._queue_offset = 0
        self.buffered_frames = 0
        self._stretcher = None
