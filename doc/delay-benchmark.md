# Fractional delay and retiming measurements

Measured locally on 2026-11-06 at 48 kHz with the ordinary native build, 16 active
oscillator voices and two seconds per case. These are CPU-only calls through
`NativeLiveEngine.advance_into`; they include Python action encoding and native
source-to-output processing, not audio-device scheduling. Action-object creation
and optional FLAC publication are outside the timed region. The first block is
reported separately; median/p99/max exclude that block. Misses include it.

The tap case submits up to 64 different sample-offset delay changes per block.
At 64 frames this keeps the 65-slot read-head pool fully occupied after startup.
Longer blocks reach that occupancy during their retime burst. The modulated case
uses continuously moving stereo reads with opposite oscillator phases, plus
gain/filter automation. Both cases use feedback 0.6 and wet/dry 0.5.

| Profile | Frames | Median µs | p99 µs | Max µs | First µs | Budget µs | Misses |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Fixed-tap retiming | 64 | 278.2 | 306.0 | 407.4 | 573.5 | 1333.3 | 0 / 1500 |
| Fixed-tap retiming | 256 | 807.4 | 886.9 | 919.9 | 1070.2 | 5333.3 | 0 / 375 |
| Fixed-tap retiming | 1024 | 2914.6 | 3074.4 | 3082.2 | 3270.4 | 21333.3 | 0 / 93 |
| Moving modulated delay | 64 | 257.1 | 271.8 | 356.5 | 501.6 | 1333.3 | 0 / 1500 |
| Moving modulated delay | 256 | 776.6 | 867.9 | 1061.0 | 1012.7 | 5333.3 | 0 / 375 |
| Moving modulated delay | 1024 | 2827.1 | 2904.6 | 2907.3 | 3044.0 | 21333.3 | 0 / 93 |

An earlier 64-frame tap run made while the full test suite was still active had
median 358.4 µs, p99 2961.0 µs, max 28469.2 µs and 41 misses / 1500 calls. The
quiet-run figures therefore do not establish a device deadline or a system-load
guarantee. Instrumented conformance separately measured zero Rust heap operations
in 2,510 successful and seven failed callbacks across 55 relevant tests.

Reproduce a case with:

```sh
uv run --no-sync python scripts/benchmark_live.py --seconds 2 --voices 16 --block-frames 64 --effect-actions 64 --delay tap --audio-output .pytest_cache/d/audio/delay-tap.flac
```

Use `--delay modulated` for the moving-read case and 256/1024 for the other
block sizes. The published two-second FLACs are in `.pytest_cache/d/audio/`.
No hardware/device or cross-platform qualification is claimed by these runs.

## Additional work beyond the prompt

None.
