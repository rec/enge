"""Stream prepared MIDI performances through enge and encode offline FLAC."""

import subprocess
import sys
import wave
from collections.abc import Iterable
from pathlib import Path
from tempfile import TemporaryDirectory
from time import perf_counter
from typing import Literal

import numpy as np
from pydantic import BaseModel
from reccy.runtime.files import atomic_output
from ufor import synth_trace
from ufor.events import Trigger
from ufor.instrument_trace import TraceAction
from ufor.samples import instrument, trace

from . import fm, sample_instrument, synth
from .midi import MidiPerformance
from .presets import Patch


class RenderReport(BaseModel, frozen=True):
    frames: int
    sample_rate: int
    notes: int
    peak: float
    elapsed_seconds: float


def render_midi(
    performance: MidiPerformance,
    patches: dict[int, Patch],
    destination: Path,
    block_size: int = 4096,
    backend: Literal["numpy", "native"] = "numpy",
    progress: bool = False,
    control_interval: int = 1,
) -> RenderReport:
    """Render fixed channel patches, including their 80ms tails and 170ms silence.

    Native selection applies to oscillator, sampler, and FM engines.
    Runtime includes preparation, rendering, and FLAC encoding. No normalization
    is applied; a non-finite or overloaded mix fails before replacing output.
    """
    if block_size <= 0 or performance.parts.keys() != patches.keys():
        raise ValueError("Positive block size and one patch per MIDI channel required")
    started = perf_counter()
    streams: list[
        tuple[
            fm.OfflineFM | synth.OfflineSynth | sample_instrument.OfflineSampler,
            list[TraceAction],
        ]
    ] = []
    for channel, patch in patches.items():
        score, assets = patch.prepare_score(performance.sample_rate)
        events = performance.parts[channel]
        renderer: fm.OfflineFM | synth.OfflineSynth | sample_instrument.OfflineSampler
        if isinstance(score, instrument.SampleInstrumentScore):
            renderer = sample_instrument.OfflineSampler(
                sample_instrument.prepare(score, assets), backend, control_interval
            )
            actions = trace.prepare(score.body, events, seed=0).actions
        else:
            renderer = (
                fm.OfflineFM(fm.prepare(score), backend, control_interval)
                if patch.engine == "fm"
                else synth.OfflineSynth(synth.prepare(score), backend, control_interval)
            )
            actions = synth_trace.prepare(score.body, events, seed=0).actions
        streams.append((renderer, list(actions)))
    frames = performance.frames + round(performance.sample_rate / 4)

    def blocks() -> Iterable[np.ndarray]:
        cursors = [0] * len(streams)
        next_progress = 0
        for start in range(0, frames, block_size):
            end = min(frames, start + block_size)
            output = np.zeros((end - start, 2))
            for i, (renderer, actions) in enumerate(streams):
                first = cursors[i]
                while cursors[i] < len(actions) and actions[cursors[i]].tick < end:
                    cursors[i] += 1
                output += renderer.advance(actions[first : cursors[i]], start, end)
            if progress and end >= next_progress:
                print(
                    f"Rendered {end / performance.sample_rate:.1f}/"
                    f"{frames / performance.sample_rate:.1f}s audio "
                    f"in {perf_counter() - started:.1f}s",
                    file=sys.stderr,
                    flush=True,
                )
                next_progress = end + 10 * performance.sample_rate
            yield output
        if any(r.voices for r, _ in streams):
            raise ValueError("Render ended with active voices")

    peak = write_flac(blocks(), destination, performance.sample_rate)
    return RenderReport(
        frames=frames,
        sample_rate=performance.sample_rate,
        notes=sum(
            isinstance(e, Trigger) for v in performance.parts.values() for e in v
        ),
        peak=peak,
        elapsed_seconds=perf_counter() - started,
    )


def write_flac(
    blocks: Iterable[np.ndarray], destination: Path, sample_rate: int = 48000
) -> float:
    """Encode float64 stereo blocks as lossless 24-bit PCM; return pre-PCM peak."""
    peak = 0.0
    with TemporaryDirectory(prefix="enge-render-") as directory:
        wav = Path(directory) / "render.wav"
        with wave.open(str(wav), "wb") as output:
            output.setnchannels(2)
            output.setsampwidth(3)
            output.setframerate(sample_rate)
            for block in blocks:
                if (
                    block.ndim != 2
                    or block.shape[1] != 2
                    or not np.all(np.isfinite(block))
                ):
                    raise ValueError("FLAC input must be finite stereo audio")
                if block.size:
                    peak = max(peak, float(np.max(np.abs(block))))
                if peak > 1:
                    raise ValueError(f"Audio exceeds PCM headroom: peak {peak}")
                pcm = np.rint(block * (2**23 - 1)).astype("<i4")
                packed = pcm.view(np.uint8).reshape(-1, 4)[:, :3].copy()
                output.writeframes(packed.tobytes())
        with atomic_output(destination) as temporary:
            subprocess.run(
                [
                    "flac",
                    "--silent",
                    "--force",
                    "--output-name",
                    str(temporary),
                    str(wav),
                ],
                check=True,
            )
    return peak
