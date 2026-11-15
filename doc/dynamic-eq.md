# Dynamic EQ

The selected second N8 profile is a cut-only bell, authored as
`ufor.audio_effects.DynamicEQ`. Each node preserves its named channels and has
one `input` port. Compose multiple nodes for several independently detected
bands. There is no external sidechain or upward dynamics in this profile.

## Filter and detector

The signal uses the existing one-stage normalized trapezoidal band-pass filter.
Its center gain is unity before dynamics. The existing filter kernel, per-channel
integrators, derived coefficient cache, portable snapshot storage and tail test
are reused in native rendering. The NumPy path calls the existing filter renderer.
The filter design follows the [Cytomic trapezoidal SVF papers](https://cytomic.com/technical-papers/).

Center frequency and Q accept ordinary actions, ramps and scoped modulation,
including enveLFO and follower routes. Coefficients update on the frame their
exact inputs change; integrator state continues across every change. This includes
jumps, which can produce transients. There is no crossfade between filter instances.
Use the existing cutoff bounds: 1 Hz through 0.999 times Nyquist, inclusive.
Q must be finite and positive. Invalid settings fail explicitly.

The detector is the absolute peak across channels of this same band-pass output,
before gain or mix. It feeds the shared compressor soft-knee target calculation.
Cap that target at minus `maximum_cut_db`, then apply the existing attack/release
smoothing in dB. Reduction starts at zero. Lowering the cap releases existing
reduction toward the new target at the current release time; it does not
immediately clamp the retained reduction. Ordinary compressors and multiband
compression use the same recurrence with no reduction limit.

For original input x, band-pass output b and smoothed gain g, fully wet output is:

    y = x + (g - 1) * b

For fixed tuning and gain, this has center gain g and unity gain far from the
band. With g=1 the original samples pass exactly. It adds no buffer latency or
look-ahead. "Cut-only" describes the gain coefficient, not a sample-peak ceiling
on transients or rapidly changing filters. No makeup, clipping or normalization
is added.

## Controls, state and endings

Defaults are center 1 kHz, Q 1, threshold -18 dBFS, ratio 1, knee 6 dB, attack
10 ms, release 100 ms, maximum cut 12 dB and mix 1. The default ratio is neutral.
Center, Q, threshold, ratio, knee, attack, release, maximum cut and mix support
frame actions and scoped controls. Mix is linear; warm bypass keeps the filter
and detector running and uses the ordinary prepared bypass fade.

Each node reserves two filter integrators per channel and one linked reduction
value. Temporary filter audio, attack/release caches and graph control storage
are prepared before callbacks. Portable snapshots retain filter integrators,
reduction, ramps and controls using existing ProcessorState fields; derived
coefficients reconstruct from the current values. Opaque snapshots retain an
equivalent continuation. Invalid state dimensions, non-finite state and positive
reduction are rejected before replacing the renderer state.

Input end feeds zeros through the band until all integrators are below the
prepared tail threshold; reduction history alone does not retain a tail. The
existing state floor snaps small integrators to zero. Stop ends immediately.
Native processing errors silence the entire callback block and latch failure;
restoring a prior opaque snapshot recovers its continuation.

Shared schema/specification/conformance is uFor `650fad1`; enveLFO `99a6d7c`
only aligns its required Python dependency pin. Its runtime and enge's Rust
enveLFO dependency are unchanged.

## Verification

Independent frequency-domain transfer functions verify static bell processing.
An implicit two-by-two state-space solve verifies moving center/Q while retaining
integrators. Nine audio/profile cases cover neutral identity, linked compression,
target capping, instantaneous dynamics, cap changes, frequency/Q ramps and jumps,
scoped enveLFO tuning, warm bypass, mono/stereo layouts, multiple independent
nodes, input-end tails, stop, portable cross-backend and opaque replay, malformed
snapshots, and failed callback recovery. Audio regressions write one-second
48 kHz WAVs.

The full suite passed 3,528 tests with 29 optional skips. The initial 29-test
focused run including existing compressor, multiband and coefficient regressions
passed; all nine final profile cases also pass. uFor passed 1,789 tests with one
optional skip, and enveLFO passed 188 tests with one optional skip. Rust tests,
Clippy, formatting, Ruff, type checks and pyupgrade pass.

Detached DSP recorded 22,401 successful and 61 failed calls with zero allocations,
reallocations or frees in either group. The nine profile cases alone recorded
373 successful and three failed calls, also with zero heap operations. Both
boundaries follow [the performance audit contract](performance-fixes.md): the
containing native method reports 22,401 successful calls with 37,031 allocations,
zero reallocations and 37,031 frees. Its 64 failed calls report 196 allocations,
one reallocation and 130 frees, including admission errors and error construction.
Error cleanup can occur after the measured method returns. The whole method is
reported honestly; only detached DSP is required allocation-free.

Ordinary release runs requested two seconds with 16 synth voices and 64 effect
actions per callback. The dynamic EQ uses ratio 4, threshold -24 dBFS and the
default cut cap, with frequency/Q actions and ramps alongside the existing
gain/filter workload. All three runs were sequential after the tests, with no
selected repeats. Times are microseconds:

| Frames | Median | p99 | Maximum | First | Budget | Misses |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 64 | 375.0 | 485.0 | 578.8 | 686.8 | 1333.3 | 0/1500 |
| 256 | 891.5 | 1100.0 | 3057.0 | 1290.0 | 5333.3 | 0/375 |
| 1024 | 2954.4 | 3079.7 | 3094.6 | 3219.0 | 21333.3 | 0/93 |

These finite shared-machine observations do not qualify audio-device deadlines.
Raw results are `build/verification/dynamic-eq-timing-{64,256,1024}.log`. The full
audit is `dynamic-eq-full.xml` and its summary `dynamic-eq-full-heap.json` in the
same directory; the final nine-case audit is `dynamic-eq-profile-final.xml`.

## Reproduction

Build ordinary release and `allocation-audit` wheels with the existing Maturin
workflow. Install without editable dependencies and extract the selected wheel's
`enge/_native*.so` into the checkout for source-tree tests. Run tests with
`TMPDIR="$PWD/build/tmp"`, a `--basetemp` under `build/verification`,
`--require-callback-no-allocations` and a JUnit report. Summarize both boundaries
with `scripts/summarize_callback_audit.py --reports REPORT`.

For ordinary timing use `scripts/benchmark_live.py --nonlinear dynamic-eq
--seconds '2 s' --voices 16 --effect-actions 64` at block sizes 64, 256 and 1024,
sequentially after tests. This profile exercises frequency and Q actions/ramps
alongside the existing gain/filter workload. Retain every timing result. Finite
CPU runs do not establish audio-device deadline guarantees.

Generated wheels, logs, reports and regression audio under `build/` are
disposable. All development source and reproduction instructions are tracked.

## Additional work beyond the prompt

None. Shared definitions, required dependency pins and the compressor target cap
implement the approved dynamic EQ profile and frequency/Q automation.
