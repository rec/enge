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
prefix at entry, advances a prepared gain/multiply/filter graph with exact
sample-offset parameter ramps, and latches failures before copying output to
NumPy. The graph's parameter, filter, and ramp state participates in snapshots.
The Python method is a callback conformance harness; a native host can call the
same Rust processing body directly.

The complete `LiveEngine` path is not yet ready for a system audio callback:

- Python validates and encodes uFor actions, sorts each block, refreshes active
  slot/context lists, and mixes source arrays.
- `LiveEngine.advance_into()` currently calls the allocating `advance()` path.
- The higher-level `PersistentSampler` still orchestrates voices in Python. Its
  prepared actions are not yet encoded directly for `LiveRuntime`.
- `OfflineEffects` still builds parameter and output arrays for every block.
  The live adapter now prepares the supported graph once and encodes parameter
  and bypass actions into bounded batches, but `LiveEngine` does not yet own and
  submit those batches. Granulation is not part of the native owner yet.
- The Python `ActionQueue` binding is a deterministic single-thread harness.
  A native host must own the `rtrb` producer and consumer on separate threads.

These are correctness and integration milestones, not hidden real-time claims.
The next implementation boundary is connecting high-level source action
encoding and `LiveEngine` ownership to `LiveRuntime`, followed separately by
native granulation.
Those paths must use its existing borrowed input/output and preallocated scratch
model rather than wrapping their allocating Python entry points.

## Stress harness

`scripts/benchmark_live.py` renders a prepared 16-voice triangle synth through a
native gain effect into a caller-owned stereo buffer. It excludes device I/O and
event preparation, discards the first callback, and reports median, p99, maximum,
and maximum fraction of the hard block budget. This benchmark exercises the
current Python-owned path, so a good timing result does not waive the blockers
above.

Run it with:

```console
uv run python scripts/benchmark_live.py --block-frames 64 --seconds 10 --voices 16
```

On 2026-09-21, the local arm64 macOS development build completed 7,500
callbacks with a 64-frame block and 16 voices. Excluding the first callback, the
median was 91.1 microseconds, p99 was 155.8 microseconds, and the maximum was
277.0 microseconds against a 1,333.3-microsecond block budget. The maximum used
20.8 percent of that budget. This run had no concurrent load and does not include
device, scheduling, or producer work.

## Additional work beyond the prompt

None.
