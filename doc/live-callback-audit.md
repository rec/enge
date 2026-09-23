# Live callback audit

## Scope

This audit covers the block path introduced by `PersistentSynth`, `PersistentFM`,
`PersistentNoise`, `PersistentSampler`, `LiveEngine`, native effects, and the
bounded action queue. It distinguishes measured callback-shaped performance from
the stronger allocation-free, lock-free, and Python-free contract in
`plan/live-effects.md`.

## Current boundary

The oscillator, FM, and noise runtimes retain voices, controls, LFOs, filters,
routing, and snapshots in Rust. Their `process_actions_into` entry point writes
to caller-owned output and performs no normal-path heap allocation inside the
sample loop. The fixed-size action queue uses `rtrb`, publishes whole batches,
and drains only the batch prefix visible when draining starts.

`LiveRuntime` now closes this boundary for generated and sampled sources. It owns
cloned oscillator, FM, and noise runtimes, immutable sample assets and traversal
cursors, one bounded action queue per source, and preallocated source/mix/output
scratch. Its processing body releases the GIL, captures each queue's published
prefix at entry, advances a prepared gain/multiply/filter/granulator graph with
exact sample-offset parameter ramps, and latches failures before copying output
to NumPy. The graph's parameters, recursive filters, circular grain history,
active grain pool, and ramps participate in snapshots.
The Python method is a callback conformance harness; a native host can call the
same Rust processing body directly.

The complete `LiveEngine` path is not yet ready for a system audio callback:

- Python validates and encodes uFor actions, sorts each block, refreshes active
  slot/context lists, and mixes source arrays.
- `LiveEngine.advance_into()` currently calls the allocating `advance()` path.
- The higher-level `PersistentSampler` still orchestrates its general dynamic
  control/filter profile in Python. `NativeLiveEngine` directly encodes the
  static tuning/gain/envelope/traversal profile and rejects unsupported dynamic
  sample settings during setup.
- `OfflineEffects` still builds parameter and output arrays for every block.
  The live adapter now prepares the supported graph once and encodes parameter
  and bypass actions into bounded batches, but `LiveEngine` does not yet own and
  submit those batches.
- The Python `ActionQueue` binding is a deterministic single-thread harness.
  A native host must own the `rtrb` producer and consumer on separate threads.

`NativeLiveEngine` is the control-side harness for all four source families. It
validates and encodes oscillator, FM, noise, and supported sampler trace actions
plus effect actions, submits bounded batches, then invokes the single Rust owner.
This removes Python DSP, mixing, and effect-array construction from its callback
body. Python still performs admission and submission synchronously before that
call. The native sampler profile currently excludes dynamic controls, generators,
and filters.

Granulators allocate circular history and a fixed captured-grain pool during
setup. Duration, density, lookback, playback ratio, jitter, wet/dry, bypass, and
snapshot continuation stay inside the owner without growing callback storage.
Freeze and unfreeze actions preserve their state across snapshots and match the
reference's circular replay. Frozen source positions wrap through a 64-frame
tail/head overlap, reduced only when a startup freeze has fewer valid frames,
and active grains retain captured samples across unfreeze.

These are correctness and integration milestones, not hidden real-time claims.
The next implementation boundaries are expanding the native sampler profile if
live sample modulation requires it and a native host that owns producer and
callback threads without Python entry. Those paths must use the existing borrowed
input/output and preallocated scratch model rather than wrapping their allocating
Python entry points.

## Stress harness

`scripts/benchmark_live.py` renders a prepared 16-voice triangle synth through a
gain and resonant-filter chain using `NativeLiveEngine` and a caller-owned stereo
buffer. It submits 64 admitted effect actions at every block boundary, including
interrupted gain and cutoff ramps. `--granular` adds bounded granulation, density
changes, and one freeze/unfreeze transition per second. The harness includes
control-side block admission, queue submission, and the single native processing
call; it excludes device I/O and initial event preparation. It discards the first
callback and reports median, p99, maximum, and maximum fraction of the hard block
budget. A good timing result does not waive the host and device blockers above.

Run it with:

```console
uv run python scripts/benchmark_live.py --block-frames 64 --seconds 10 --voices 16
uv run python scripts/benchmark_live.py --block-frames 64 --seconds 10 --voices 16 --granular
```

On 2026-09-23, the local arm64 macOS development build completed 7,500 callbacks
with a 64-frame block, 16 voices, and 64 effect actions per callback. Excluding
the first callback, the gain-plus-filter profile had a 275.1-microsecond median,
482.0-microsecond p99, and 2,017.2-microsecond maximum against a
1,333.3-microsecond block budget. The one observed maximum exceeded the deadline.

With the same action density and bounded granulator enabled, the second run had a
266.6-microsecond median, 403.8-microsecond p99, and 1,157.1-microsecond maximum,
or 86.8 percent of the block budget. These development-host measurements establish
the cost of the current Python admission harness. The outlier reinforces that it
is not a native host deadline guarantee.

## Additional work beyond the prompt

None.
