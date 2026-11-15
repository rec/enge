# Patch storage, native-call audit and coefficient caching

This work addresses the Rust allocation review before adding more effects.
The shared Patch fix is enveLFO `0c027e2`. It reuses captured continuation,
command and reset storage, preserving rollback and the public Patch API.
Rust workspace tests with all features and Clippy pass. An initial workspace
test encountered a stale CMake source-directory cache; a fresh generated target
directory passed without changing the source or removing existing artifacts.

## Measurement boundaries

The existing `take_callback_audit` measures the detached native DSP body of
`LiveRuntime.process_into`. `take_native_call_audit` measures the containing
native method body, starting before
NumPy array conversion and including host-input validation/copying, output
copying and construction of returned Python errors. Python wrapper return/error
normalization lies outside the native method body. Counters nest without replacing
each other's active scope. Both report successful and failed calls separately.
Python/foreign allocator activity and host-side action preparation remain outside
these instrumented-Rust-allocator counters. No whole-Python-callback guarantee is
implied.

Host inputs use borrowed sequence access instead of building a temporary Rust
Vec. Existing list and tuple callers retain their behavior. The broader audit
also exposes rust-numpy's per-borrow hash-map allocations: output borrowing
allocates, and each host input is validated and copied under one read borrow.
These are distinct from the detached DSP's zero-heap contract. The user approved
retaining safe NumPy borrowing and reporting outer allocations honestly, while
deferring a prepared-I/O API. Tests require zero heap operations only in detached
DSP and check that every successful native call has a corresponding DSP record.

Current pytest reports record both `native_callback_heap` and `native_call_heap`.
The summarizer requires both properties, retaining detached counts at the top
level and whole-call counts under `native_call`. Historical single-boundary
reports remain original evidence; rerun the documented tests for dual counters.

## Exact coefficient caches

The native filter bank, effect filters and per-voice synth filters reuse
trapezoidal coefficients keyed by effective cutoff, Q and sample rate. Compressor
and follower smoothing retain separate attack/release coefficients keyed by
their time and sample rate. Transient fast/slow detectors have separate caches;
the octave divider reuses the shared follower cache. Validation remains on each
frame, and any exact parameter change takes effect on that frame. No control
rate reduction, quantization or mathematical approximation is introduced.

Caches contain derived data only. Portable snapshots retain their existing
schema; cache keys make restored parameter values safe even if a renderer's
cache came from a later frame. The independent NumPy implementations remain
unchanged. Regressions cover cutoff/Q steps and ramps, zero/nonzero attack and
release, returning to earlier settings, portable cross-backend replay and
native opaque replay. All audio checks write one-second 48 kHz WAVs.

## Reproduction and baseline

Generated source copies are made only from tracked Git input. To build the
pre-fix ordinary baseline without changing branches or the worktree:

```sh
mkdir -p build/tmp build/perf-baseline-source build/wheels/perf-baseline
git archive 2c73c32 | tar -x -C build/perf-baseline-source
```

From that generated source directory, use the existing environment without
editable dependencies:

```sh
TMPDIR=/Users/tom/code/enge/build/tmp uv run --project /Users/tom/code/enge \
  --no-sync --with maturin maturin build --release --locked \
  --out /Users/tom/code/enge/build/wheels/perf-baseline
```

Build the changed ordinary and `allocation-audit` wheels from the main checkout.
Install each measured wheel with `uv pip install --no-deps --reinstall` and
extract its `enge/_native*.so` file into the checkout before running the benchmark.
Use `scripts/benchmark_live.py --seconds '2 s' --voices 16 --effect-actions 64`,
with block sizes 64, 256 and 1024, both without a selected new effect and with
`--nonlinear deesser`. Run sequentially after tests, retaining all misses.
These CPU runs do not establish audio-device deadlines.

Source copies, targets, wheels, reports and regression audio under `build/` are
disposable generated artifacts. Their source inputs and procedures are tracked.

## Verification results

The combined audit suite passed 3,512 tests with 29 optional skips. Detached DSP
recorded 21,731 successful and 57 failed calls, with zero allocations,
reallocations or frees in either group. The containing native method recorded:

| Result | Calls | Allocations | Reallocations | Frees |
| --- | ---: | ---: | ---: | ---: |
| Successful | 21,731 | 35,691 | 0 | 35,691 |
| Failed | 60 | 184 | 1 | 122 |

Admission errors account for the additional failed native calls. Returned errors
can be freed after the measured method returns, so the failed row's unequal
allocation/free totals do not establish a leak. The wider boundary is reported,
not asserted allocation-free. The focused 63-test run, Rust audit positive
controls, Rust formatting/Clippy, Ruff, Python formatting, type checks and
pyupgrade also passed. The shared enveLFO workspace passed all-feature Rust tests
and Clippy using the fresh target described above.

Ordinary release wheels were measured sequentially, once per configuration,
before and after the fixes. All values below are microseconds; misses are the
benchmark's post-first-call deadline count. The first call is listed separately.

| Build | Profile | Frames | Median | p99 | Maximum | First | Misses |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Before | Baseline | 64 | 382.5 | 451.0 | 510.7 | 656.3 | 0/1500 |
| After | Baseline | 64 | 379.4 | 459.2 | 559.6 | 666.3 | 0/1500 |
| Before | Baseline | 256 | 918.6 | 3943.2 | 6378.0 | 16294.8 | 4/375 |
| After | Baseline | 256 | 889.9 | 952.0 | 978.6 | 1141.6 | 0/375 |
| Before | Baseline | 1024 | 3202.1 | 16062.6 | 17419.9 | 3310.1 | 0/93 |
| After | Baseline | 1024 | 2855.7 | 3007.6 | 3017.2 | 3194.1 | 0/93 |
| Before | De-esser | 64 | 414.5 | 1306.1 | 19573.3 | 696.5 | 15/1500 |
| After | De-esser | 64 | 387.5 | 454.6 | 541.4 | 624.8 | 0/1500 |
| Before | De-esser | 256 | 930.5 | 1039.0 | 1079.6 | 1165.0 | 0/375 |
| After | De-esser | 256 | 909.5 | 1013.3 | 1068.1 | 1201.5 | 0/375 |
| Before | De-esser | 1024 | 3079.4 | 3202.6 | 3249.4 | 3304.2 | 0/93 |
| After | De-esser | 1024 | 2972.2 | 3078.6 | 3089.2 | 3211.9 | 0/93 |

Budgets at 48 kHz are 1333.3, 5333.3 and 21333.3 microseconds respectively.
Every paired median decreased, but these short shared-machine runs do not
isolate scheduling noise or establish a statistically reliable speedup. All six
after runs had zero misses; this is not an audio-device deadline qualification.
No repeats were selected to hide misses. Raw reports are
`build/verification/perf-{before,after}-{none,deesser}-{64,256,1024}.log`;
the full audit report and summary are `perf-fixes-full.xml` and
`perf-fixes-full-heap.json` in the same directory.

## Additional work beyond the prompt

None. The prepared Patch reset/command extension and separate native-call/DSP
diagnostics were explicitly approved during implementation.
The prepared-I/O API was explicitly deferred; no array-borrow safety checks were
removed to make the wider counters look allocation-free.
