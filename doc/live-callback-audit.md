# Live callback audit

## Scope

This audit covers the block path introduced by `PersistentSynth`, `PersistentFM`,
`PersistentNoise`, `PersistentSampler`, `LiveEngine`, native effects, and the
bounded action queue. It distinguishes measured callback-shaped performance from
the stronger allocation-free, lock-free, and Python-free contract in
the completed live-effects implementation.

## Current boundary

The oscillator, FM, and noise runtimes retain voices, controls, LFOs, filters,
routing, and snapshots in Rust. Their `process_actions_into` entry point writes
to caller-owned output. This does not establish an allocation-free contract for
every supported profile; the Patch migration and remaining qualification are
detailed below. The fixed-size action queue uses `rtrb`, publishes whole batches,
and drains only the batch prefix visible when draining starts.
Control ramps and instrument/part/trigger context storage are owned by the shared
native `ControlBank`; its storage is prepared before processing and retained in
generated-source snapshots.

### Patch scheduler allocation boundary

The 2026-11-03 audit found temporary input/release vectors, copied names and
allocating shared Patch calls. The approved migration now pins `envelfo-core`
revision `2f2ffb0a5fe45f37b7a67d602e7b7eafa908e259` and uses its prepared
transaction storage for observation, commands, settling and voice reset.
Live-source construction prepares all slots before processing. Retirement retains
their state and capacity; snapshot restoration rebuilds processing storage outside
rendering. Input updates and ordered release scans no longer allocate lists.
Each slot permits 4096 pending connections; no named event outputs leave this
host, so its external event buffer has zero capacity.

Shared-core allocation tests cover reset, observation, cues, delayed events,
capacity rollback, cloning and snapshot continuation. Enge's successful and failed
callback measurements are recorded below.
Low-level native sample sources now attach
the same prepared bank and execute its lifecycle and commands. Their delayed
voices reset generators at the first audible frame, and snapshots retain the
prepared storage. The Python sampler adapter now prepares common/slot banks and
admits shared generator actions before native processing. Shared-core results alone do not
qualify the complete sample-source path or establish hardware deadlines.

### Native heap measurement

The opt-in `allocation-audit` Cargo feature counts Rust allocations,
reallocations and deallocations on the processing thread while
`LiveRuntime::process` drains queues, renders sources, mixes and applies effects.
It excludes preparation, Python action admission, GIL transitions, output copying
and snapshot creation/restoration. A Rust positive-control test verifies that all
three counters detect heap operations. Successful and failed callback totals are
kept separately; the pytest check requires zero heap operations for both outcomes
and records both rows in JUnit properties.

On 2026-11-04 the full instrumented suite passed 2,054 tests, with 21 skipped.
387 tests exercised the callback: 7,276 successful calls and seven failed calls
performed zero Rust allocations, reallocations or frees. Callback errors now
carry only static messages, scalar payloads or the fixed-size shared-core error.
Python exception construction and dynamic message formatting occur after native
processing returns. Error messages, output zeroing, failure latching and snapshot
recovery remain covered. Failure cases include source/effect rejection, sample
filter bounds, queue overflow beyond the callback action capacity and an
EnveLFO rational overflow from an instrument LFO rate command. The latter checks
the shared-core error message and failure latch through the Python boundary.
The audit exposed allocating decimal formatting in LFO numeric commands. The
conversion now uses a stack buffer and checked integer arithmetic, preserving
the existing rational result. Regressions include fractional rates, seeks and
negative shifts, plus sampler threshold events, stage cues and latch captures.
The latest run also exercises onset-key bindings for generated synth, FM and
noise sources through `NativeLiveEngine`, including reused trigger names with
overlapping release tails. These use the existing per-voice event storage.

Reproduce the measurement with the locked environment already installed:

```sh
uv pip install --no-deps --reinstall --editable . --config-settings 'build-args=--features allocation-audit'
cargo test --locked --features allocation-audit
uv run --no-sync pytest --require-callback-no-allocations --junitxml=/private/tmp/enge-callback-heap.xml
uv sync --frozen --reinstall-package enge
```

The last command restores the ordinary extension. Instrumentation is absent from
default builds. This measures the tested Rust paths, not arbitrary future inputs,
foreign-library allocators, device deadlines or scheduling/lock behavior. Optional
Rubber Band processing is not qualified by this run.

`LiveRuntime` owns processing for generated and sampled sources. It owns
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
  tuning/gain/envelope/traversal profile and multichannel filter
  chains. Filter banks are prepared per voice, reset on reuse and retained in
  snapshots. Unsupported dynamic sample settings are rejected during setup.
  The sampler adapter prepares scoped amplitude/tuning, cutoff/Q and crossfade
  controls through `set_sample_controls` and admits addressed actions before
  native processing.
  Context and route scratch storage remain bounded; snapshot ownership includes
  the Python addresses and native control state.
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
call. Native sampler key/velocity event bindings use preallocated per-voice
values, initialized during onset admission and retained in snapshots. The native
sampler adapter also prepares shared generator banks, remaps common/slot routes,
and admits instrument LFO and addressed voice-generator commands. Random keys
come from prepared voice starts; delayed starts initialize generators when audible.
Generated synth, FM and noise key and velocity bindings use the same native
per-voice storage. Their Python velocity metadata shares bounded trigger-context ownership
and is included in source snapshots.
Control-routed sampler filters reuse the fixed bank's stage kernel and retain
preallocated per-voice integrators; parameter evaluation does not allocate.

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
