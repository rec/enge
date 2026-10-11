# Test performance

## Measuring

Run only one benchmark or test suite at a time. Use the same interpreter,
native build, worker count, and test selection for before/after comparisons.
Keep other projects' test runs idle. Profiler timings include substantial
instrumentation overhead and are not ordinary test timings.

```sh
uv run pytest --durations=30
uv run pytest -n 0 test/test_control.py
uv run python -m cProfile -o /private/tmp/enge-latch.prof -m pytest -n 0 -q \
  'test/test_motion_latch.py::test_latch_captures_delivery_frame_with_independent_gates_and_snapshot[1.0-cycle-numpy]'
```

Read the profile without launching the engine:

```sh
printf 'sort cumulative\nstats (control.py|lfo.py|synth.py) 20\nquit\n' | \
  uv run python -m pstats /private/tmp/enge-latch.prof
```

The full suite remains the commit gate. Selecting individual tests is for
development feedback, not a replacement for complete conformance checks.
Linux and Windows validation remains release-only.

## Findings, 2026-10-11

Measurements on macOS with Python 3.13.12, pytest 9.1.1, pytest-xdist 3.8.0,
and the default native build without the optional Rubber Band feature:

- Baseline full suite: 352.74 seconds, 1,082 passed and 21 optional skips,
  with ten workers and work-stealing.
- Optimized full suite: 323.88 seconds, 1,085 passed and the same 21 optional
  skips, with the same ten-worker setup. This single before/after comparison
  saved 28.86 seconds, approximately 8%. The three additional cases cover
  single-frame cancellation and malformed/nonfinite signals.
- Representative Latch test under cProfile: 27.95 seconds before optimization,
  16.60 seconds afterward. This is approximately 41% less test time.
- The profile identified repeated Cycle-to-LFO model validation, one-frame
  LFO span generation, and one-element modulation arrays as bottlenecks.
  WAV regression writing was not a significant cost in this profile.

The worker experiment used the following unchanged 38-case selection. Run it
sequentially with four, six, and ten workers:

```sh
uv run pytest -n 4 --durations=5 \
  test/test_motion_latch.py::test_latch_activation_and_same_frame_captures \
  test/test_motion_quantize.py \
  test/test_control.py::test_array_modulation_matches_scalar_routes_and_weights
```

Four workers took 22.60 seconds, six took 20.69, and ten took 21.02. These are
single runs on one host, not a universal worker recommendation. The small
six-versus-ten difference does not justify changing the portable `-n auto`
default. Re-measure when hardware, Python, or the test workload changes.
After optimization, the same 38-case selection with ten workers took
15.19 seconds, versus 21.02 beforehand, approximately 28% less time.

## Changes and coverage

The renderer retains converted immutable Cycle definitions for its lifetime.
Exact one-frame NumPy LFO queries reuse uFor's scalar evaluator; interpolated
sine queries still use their event-anchored interpolation grid. Native LFO
queries still execute the Rust kernel. One-frame modulation uses the existing
uFor scalar evaluator, while multi-frame queries remain vectorized. Stateful
Patch output is filled into preallocated arrays instead of retaining every
one-frame result dictionary until concatenation.

Motion tests share read-only, session-local reference audio for the same plain
square-wave instrument. Randomized Motion output, snapshots, partitioning,
and NumPy/native/persistent comparisons are still evaluated separately.
No persistent disk cache or new dependency is introduced.

The routing arithmetic test uses explicit clamping and knot-boundary inputs
instead of 48,001 scalar queries. It also compares single-frame and block
results. Linear, step, and exponential mappings, activation weights, validation,
and cancellation retain coverage. Audio regressions continue to generate
at least one second of 48 kHz WAV output; none have been shortened or removed.

## Stages and event-processing follow-up

The first pass did not remove the dominant cost. A representative staged-clock
and cue-routing regression took 51.84 seconds under cProfile, with roughly
688,000 `advance_motion` calls. Its event-processing method accounted for
51.23 seconds of the test. Most calls merely advanced state through frames
that contained no events.

```sh
uv run python -m cProfile -o /private/tmp/enge-stages.prof -m pytest -n 0 -q \
  'test/test_lfo_instrument.py::test_patch_division_is_first_aligned_independent_and_snapshot_safe[1-0.5-0-0-0-False-divisors0-cue-stages]'
```

The follow-up renders event-free seconds-clock intervals using the existing
LFO and linear-envelope block kernels. uFor's natural-boundary query determines
where the next stage event occurs. Cycle markers and queued deliveries also
bound each interval. The original delivery loop still processes those events
at exact fractional times, including multiple events observed on one frame.
State is advanced to the last observed frame of each interval, preserving
snapshot and subsequent-command timing.

Staged Cycle values remain exact even when the host requests interpolated
ordinary LFO controls. Delay/fade weights are applied inside the staged value,
as in the scalar reference. Beat clocks, nonlinear or reversing contours, and
per-frame Latch/Threshold/Slew observation retain their existing evaluation
path. The optimization does not remove these cases or change their coverage.
Block eligibility is checked for all participating Motions before allocating
arrays. A mixed block/scalar-only configuration therefore does not perform
block-sized allocations repeatedly on every frame.

The dependency refresh is committed separately from renderer changes. It
includes the public uFor boundary query and aligns enge's reccy package-source
URL with uFor's HTTPS source; repository Git remotes remain SSH.

### Final measurements

- Dependency-only baseline, before event-span rendering: 282.11 seconds,
  1,085 passed and 21 optional skips.
- Final full suite: 114.70 seconds, 1,091 passed and the same 21 optional skips.
  This saves 167.41 seconds, about 59%, against the matched dependency baseline,
  or about 67% against the original 352.74-second run. Both comparisons use
  ten workers and work-stealing on the same macOS/Python 3.13 host.
- An earlier optimized full run passed 1,089 cases in 121.27 seconds, before
  adding the two mixed-mode regressions and final block-eligibility check.
- The final representative profiled Stages case took 0.95 seconds, versus
  51.84 seconds before. `advance_motion` calls fell from 688,264 to 624;
  event-processing time fell from 51.23 seconds to 0.09 seconds.
- All 1,206 uFor tests passed for the boundary-query change. The six new enge
  cases cover exact staged Cycle delay/fade under control intervals and
  single-frame partitions, plus mixed staged Cycle/looping-Contour evaluation.

Full-suite timings are single-host measurements, not cross-platform guarantees.
The remaining slowest cases involve per-frame threshold/slew observations and
beat-clock evaluation. No regression duration, backend comparison, or existing
test was removed to obtain the improvement.
