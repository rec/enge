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

The complete `LiveEngine` path is not ready for a system audio callback:

- Python validates and encodes uFor actions, sorts each block, refreshes active
  slot/context lists, and mixes source arrays.
- `LiveEngine.advance_into()` currently calls the allocating `advance()` path.
- PyO3 persistent-runtime calls retain the GIL; they do not expose a native host
  entry point.
- `PersistentSampler` retains traversal and DSP in Rust but orchestrates voices
  in Python, and the native sampler kernel returns newly allocated arrays.
- Native effect preparation builds parameter and output arrays for every block.
- The Python `ActionQueue` binding is a deterministic single-thread harness.
  A native host must own the `rtrb` producer and consumer on separate threads.

These are correctness and integration milestones, not hidden real-time claims.
The next implementation boundary is one Rust owner for source slots, sample
assets, effect graph scratch, action consumer, and output scratch. Its render
method must accept borrowed input/output buffers, release or avoid the GIL, and
perform no allocation or destruction on the callback thread.

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
