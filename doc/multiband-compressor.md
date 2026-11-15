# Three-band multiband compression

The selected N8 profile is `ufor.audio_effects.MultibandCompressor`. It accepts
one named-channel input and preserves that layout. Fixed lower/upper crossovers
default to 200 Hz and 4 kHz. Each band has independent linked-channel peak
compression, with the existing threshold, ratio, soft knee, attack, release and
makeup recurrence. Ratios default to 1 and makeup to 0, so dynamics are neutral.
There is no external sidechain, look-ahead, internal dry mix or bypass.

## Crossover and phase

The lower crossover splits low and upper branches using cascaded second-order
Butterworth sections (LR4). The upper branch is split again at the upper
crossover into mid/high bands. A second-order allpass at that upper frequency
phase-compensates the low band. The neutral sum is the product of the two
Butterworth allpasses: unity magnitude, frequency-dependent phase. See the
[Linkwitz crossover equations](https://www.linkwitzlab.com/crossovers.htm).

Zero declared frame latency means no added buffer delay; it does not mean zero
phase or group delay. The processor is always wet. Mixing its output with the
unfiltered original can cause frequency cancellations even with neutral dynamics.
Any such blend or bypass is explicitly authored using graph routing.

Prepare crossovers strictly between zero and Nyquist and in increasing order.
Unrepresentable floating-point coefficients are rejected. Frequencies do not
accept actions or modulation. Coefficients are prepared once, using the same
normalized trapezoidal coefficient machinery as existing native filters.

## Controls and lifecycle

Author `low`, `mid`, `high` as `CompressorBand` settings. Parameters are named
`low-threshold_db`, `mid-ratio`, `high-makeup_db`, etc. All six settings accept
ordinary frame actions, ramps and scoped enveLFO/follower modulation. Each band
uses its own peak across channels, with one shared gain for those channels.
No automatic makeup, clipping or normalization is introduced.

Nine second-order sections retain two integrators per channel. Three additional
values retain gain reduction in dB. Native band scratch is prepared once, and
each band reuses its own attack/release coefficient cache. Portable JSON snapshots
carry integrators and reductions; derived coefficients and temporary band audio
are reconstructed from the prepared definition. Opaque snapshots retain equivalent
continuations. Input end feeds zero through the crossover until every integrator
is below `tail_threshold`; gain-reduction history alone does not retain a tail.
Integrator values below `state_floor` snap to zero. Stop ends immediately.
Non-finite processing fails explicitly; failed live callbacks silence the block,
latch failure, and can be restored from a saved continuation.

The shared definitions and conformance example are in uFor `bf94c5f`; enveLFO's
Python dependency pin is aligned in `e7a33d3`, without changing its runtime.
Enge's Rust enveLFO pin is unchanged because no Rust API changed.

## Verification

Independent FFT-domain LR4 transfer functions verify neutral reconstruction and
compressed output. Tests cover stereo linking, mono input, threshold ramps,
scoped enveLFO modulation, arbitrary callback partitions, crossover tails, stop,
portable NumPy/native replay, opaque callback replay, malformed snapshots and
failure recovery. Audio regressions are one second at 48 kHz.

The full suite passed 3,519 tests with 29 optional skips. Detached DSP recorded
22,028 successful and 58 failed calls, with zero allocations, reallocations or
frees in both groups. The seven focused profile tests recorded 297 successful
and one failed detached DSP call, also with zero heap operations. The final
snapshot validation check was rerun after the shape review and passed. uFor's
suite passed 1,736 tests (one optional skip, 23 expected failures); enveLFO's
Python suite passed 188 tests with one optional skip. Rust audit positive
controls, Clippy, formatting, Ruff, type checks and pyupgrade pass.

Both allocator boundaries follow [the performance audit contract](performance-fixes.md).
The containing native method recorded 22,028 successful calls with 36,285
allocations, zero reallocations and 36,285 frees. Its 61 failed calls recorded
187 allocations, one reallocation and 124 frees. This includes admission errors
outside DSP and error construction whose cleanup can occur after the measured
method returns. Only detached DSP is required allocation-free; safe outer NumPy
borrowing and error allocations remain visible.

Ordinary release timings used 16 synth voices, 64 effect actions, all three
bands at -24 dBFS / 4:1, and two seconds of audio. Runs were sequential after
the full suite, with no selected repeats. Times are microseconds:

| Frames | Median | p99 | Maximum | First | Budget | Misses |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 64 | 414.5 | 492.8 | 565.0 | 717.4 | 1333.3 | 0/1500 |
| 256 | 944.7 | 1011.8 | 1056.4 | 1176.3 | 5333.3 | 0/375 |
| 1024 | 3068.5 | 3174.3 | 3293.4 | 3323.2 | 21333.3 | 0/93 |

These finite observations on a shared machine do not qualify audio-device
deadlines. Raw results are `build/verification/multiband-timing-{64,256,1024}.log`;
the full audit is `multiband-full.xml` with summary `multiband-full-heap.json`
in the same directory. The snapshot shape follow-up is recorded in
`multiband-snapshot-final-r2.log`.

Dynamic EQ is the remaining second N8 slice. This profile does not implement
moving crossovers, linear-phase filtering, external detectors or dynamic EQ.

## Reproduction

Build an ordinary release wheel and an `allocation-audit` feature wheel using
the existing Maturin workflow. Install without editable dependencies and extract
the selected wheel's `enge/_native*.so` into the checkout for source-tree tests.
Use `TMPDIR="$PWD/build/tmp"` and pytest `--basetemp` under `build/verification`.
Run the full suite with `--require-callback-no-allocations` and a JUnit report;
summarize it with `scripts/summarize_callback_audit.py --reports REPORT`.

For ordinary timing, run `scripts/benchmark_live.py --nonlinear multiband
--seconds '2 s' --voices 16 --effect-actions 64` once each at `--block-frames`
64, 256 and 1024, sequentially after the tests. Retain every measurement. These
CPU observations do not establish audio-device deadline guarantees.

Generated wheels, reports and regression audio under `build/` are disposable;
all development code and reproduction instructions are tracked.

## Additional work beyond the prompt

None. Shared schema and required dependency-pin updates implement this profile.
