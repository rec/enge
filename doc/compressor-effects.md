# Linked peak compressor and ducker

`ufor.audio_effects.Compressor` has explicit `input` and `detector` ports. Its
output preserves the primary input layout. The detector may have another layout,
including a mono sidechain for a stereo signal. It controls gain without entering
the output mix. Connect the signal to both ports for ordinary compression, or
connect a named host stream or source tap to the detector for ducking.

The approved profile is feed-forward, linked peak detection with a soft knee and
no look-ahead. Each frame uses the largest absolute detector-channel sample and
applies one gain to every signal channel. It adds no latency, retains no audio,
and has no tail. There is no automatic makeup or block normalization.

Parameters are `threshold_db`, `ratio` (at least one), `knee_db` (nonnegative),
`attack_seconds`, `release_seconds` (both nonnegative), `makeup_db`, and the common
`mix`. The defaults are -18 dB, 4:1, 6 dB, 10 ms, 100 ms, 0 dB and fully wet.
Every parameter can be automated or targeted by shared enveLFO bindings. Graph
preparation validates entire modulation domains against the processor limits.

For peak `p > 0`, let `x = 20*log10(p) - threshold_db`, ratio `r`, and knee `k`.
The target gain reduction is zero below `-k/2`, `(1/r - 1)*x` above `k/2`, and
`(1/r - 1)*(x + k/2)^2/(2*k)` inside the knee. A zero knee gives a hard threshold;
a silent detector targets zero reduction. The implementation uses an equivalent
normalized knee expression to avoid unnecessarily squaring large decibel values.

Gain reduction starts at zero decibels. Before producing a frame, it approaches
the target with a one-pole coefficient `exp(-1/(sample_rate*time))`. Attack is
selected for increasing attenuation, release for recovery. A zero time applies
the target immediately. These are time constants, not completion times. Wet gain
is `10^((reduction_db + makeup_db)/20)`, followed by the common linear wet/dry mix.

Bypass keeps the detector and gain state advancing while its existing fade moves
toward dry. Offline JSON snapshots include the reduction scalar alongside
parameter ramps and generator state. Native live snapshots retain the same
reduction and prepared route identity. The offline native path and native live
owner share `src/compressor.rs`; NumPy implements the reference independently.

The offline input-end contract is inherited from the graph: an ended detector
becomes silence and permits recovery; an ended signal ends the compressor's
output. Native live input-end and stop actions use the same contract, including
independent detector endings. Their state survives native snapshot replay.

## Verification

Twelve numerical/audio regressions compare the static knee against known transfer values and
attack/release against closed-form exponential trajectories. They cover mono and
stereo detectors, linked channels, immediate processing, hard knee, enveLFO
threshold motion, makeup automation, bypass and exact snapshot replay. Each runs
against NumPy, native offline rendering or the actual native callback owner and
writes paired one-second 48 kHz WAVs.
The full suite passed 3,178 tests with 21 skipped. A final 13-test focused run
also includes a later-added invalid-ratio callback regression: a failure halfway
through a block silences the block, and restoring an earlier snapshot recovers
the expected audio.

The focused instrumented run passed 62 compressor/routing/live/generator/delay
tests. It recorded zero Rust allocations, reallocations or frees in 3,055
successful and seven failed callbacks. The additional failure regression's
instrumented run measured one successful and one failed callback, again with
zero heap operations: 3,056 successful and eight failed callbacks across the
two runs. Python preparation and offline bridge calls are outside that guarantee.

The benchmark harness accepts `--compressor`, adding a mono host sidechain,
stereo compression and threshold automation to its existing source/filter chain.
Host audio is prepared before timing; the timed call includes admission,
source rendering, native DSP and the output copy. It can also write a 48 kHz
audition FLAC after processing. Device deadlines require separate qualification.

## CPU measurement

On 2026-11-06, ordinary-build runs used 16 voices, 64 effect actions per callback,
two seconds of 48 kHz audio, a mono host sidechain, stereo compression and the
existing gain/filter chain. These commands ran sequentially after the test jobs
finished. Times are microseconds; median/p99/maximum exclude the first block,
which is reported separately. Miss counts include it.

| Frames | Median | p99 | Maximum | First | Budget | Misses |
| --- | --- | --- | --- | --- | --- | --- |
| 64 | 269.6 | 347.1 | 379.5 | 2,368.8 | 1,333.3 | 1 / 1,500 |
| 256 | 775.5 | 885.4 | 925.6 | 1,045.0 | 5,333.3 | 0 / 375 |
| 1,024 | 2,841.0 | 2,959.4 | 2,962.6 | 3,187.1 | 21,333.3 | 0 / 93 |

The 64-frame startup exceeded its budget once; the subsequent measured callbacks
did not. Earlier runs may have overlapped because their launchers returned before
completion. They recorded zero misses at all three sizes, with first-block times
610.2 / 1,094.1 / 3,024.7 and maxima 389.6 / 921.9 / 3,391.8 respectively. Both
observations are retained; they do not establish a startup or device guarantee.

Reproduce a size with:

```sh
uv run --no-sync python scripts/benchmark_live.py --seconds 2 --voices 16 --block-frames 64 --effect-actions 64 --compressor --audio-output .pytest_cache/d/audio/compressor.flac
```

Repeat at 256 and 1,024 frames without overlapping processes. The 64-frame run
also writes a two-second stereo audition FLAC; the numerical regressions remain
the correctness evidence.

## Additional work beyond the prompt

None.
