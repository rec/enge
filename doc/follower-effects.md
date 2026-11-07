# Audio-derived envelope followers

`Follower` consumes one named audio input and measures its maximum absolute
channel amplitude. Initial level is zero. Attack defaults to 10 ms and release
to 100 ms. Each frame uses attack when peak exceeds level, otherwise release:
`level += (peak-level) * (1-exp(-1/(sample_rate*seconds)))`.
Zero seconds follows immediately. Both backends evaluate the exponential factor
with `expm1` to retain precision for slow response times. Attack and release
accept actions, ramps and other acyclic modulation sources.

A dedicated effect-only binding names the analysis processor. It exposes raw
linear amplitude, which may exceed one. The source's declared domain must cover
that output; observations outside the domain fail explicitly. Mapping amplitude
to gain, cutoff or another parameter uses ordinary declared routes. The analysis
cannot feed an audio port or become the graph audio output. There is no wet/dry
or bypass setting on a follower.

Audio and control edges share the graph's dependency ordering and cycle checks.
An analysis observation reaches its consumers in the same frame. Consumers
align earlier audio and control paths to their latest required arrival. Each
route's mapped contribution has its own prepared delay, retaining independent
additive and multiplicative terms until the target is resolved. Parameter bases
and enveLFO contributions follow the same arrival timing. Oversampled processor
controls then receive their existing internal compensation.

Generator contributions reuse the shared control renderer and native generator
bank. Their prepared execution projection uses one scalar target per route;
the original portable definition retains authored targets, units and bounds.
Target-domain validation happens after all aligned contributions are combined.
Each node resolves immediately before DSP, replacing the previous whole-graph
pre-DSP resolution pass.

Input end supplies zero peak. Release continues until level falls below the
declared tail threshold, then level becomes zero and analysis ends. Values below
the state floor become zero earlier. Defaults are 1e-12 and 1e-30 respectively.
A consumer retains its normal tail rules after its aligned audio and required
analysis paths end. Stop remains immediate.

Portable snapshots retain level, route/base delay rings, pending end-marker
distances and ordinary automation/processor state. NumPy and native offline
rendering exchange this state. Native opaque snapshots preserve the same DSP
state together with generator clocks and prepared graph identity. Malformed
levels and delay layouts are rejected before replacing native state.

Regressions use one-second 48 kHz WAVs and analytical attack/release curves.
They cover linked negative-channel peaks, instantaneous response, independent
input endings, mixed follower/enveLFO add/multiply routes, delayed analysis or
audio paths, portable/opaque replay, malformed-state rejection, source-domain
failure, complete failed-block silence and restore recovery.

The ordinary full suite passes 3,293 tests with 29 optional Rubber Band skips.
All four focused follower regressions pass; the related 60-test control/runtime
migration run also passes. Ruff, formatting, type checking, pyupgrade, Rust
formatting and Clippy pass. The pinned follower schema `df1ed1f` passed all
1,524 uFor tests. The later gate/expander schema is separate work.

The 114-test instrumented audit covers follower, persistent effects, shared
enveLFO, mixer, matrix and live input/routing paths. It records zero allocations,
reallocations or frees in 5,138 successful callbacks and 15 failed callbacks.
Preparation, Python bridging, host admission and snapshot conversion remain
outside this native callback guarantee.

After restoring the ordinary build and finishing all test jobs, three sequential
CPU runs used two seconds of 48 kHz audio, 16 voices and 64 effect actions per
callback, including attack automation. Times are microseconds. Median/p99/maximum
exclude the first block; misses include it.

| Frames | Median | p99 | Maximum | First | Budget | Misses |
| --- | --- | --- | --- | --- | --- | --- |
| 64 | 315.9 | 377.5 | 415.5 | 629.2 | 1,333.3 | 0 / 1,500 |
| 256 | 829.1 | 926.8 | 984.2 | 1,104.8 | 5,333.3 | 0 / 375 |
| 1,024 | 2,890.4 | 2,968.2 | 2,992.6 | 3,135.8 | 21,333.3 | 0 / 93 |

Reproduce with `scripts/benchmark_live.py --follower --seconds '2 s' --voices 16`
`--effect-actions 64` and the listed `--block-frames`. These finite measurements
qualify the tested development-machine CPU paths, not an audio device or another
platform.

## Additional work beyond the prompt

None.
