# Shared persistent native effects

`OfflineEffects(..., backend="native")` and `LiveRuntime` use the persistent DSP
core in `src/effects/runtime.rs`. The old native whole-graph function, separate
granular function and per-frame compressor binding have been removed.

The offline owner keeps its existing Python action, ramp and generator
processing. At each output frame it gives the core the declared inputs,
resolved parameters, input-ending flags and granular freeze state. The core
retains filter integrators, grain pools, delay buffers/read heads, compressor
reduction and [reverb network state](reverb-effects.md) between calls. The callback
owner resolves its prepared Rust controls and calls the same processor
implementation. Callback scratch remains fixed
after preparation.

Offline snapshots keep the existing `EffectSnapshot`/`ProcessorState` JSON
representation. Native DSP state is exported at block/snapshot boundaries and
imported explicitly on restoration. No opaque native object is embedded in that
JSON. Valid snapshots can replay between NumPy and native offline renderers.
Import checks matrix shapes, buffer indices, read heads and grain capacity before
replacing running state. A rejected import preserves the renderer and its clock.
The live owner's opaque native snapshot retains its prepared route and width
identity, controls, input endings and terminal graph state.

Native grains accumulate in launch order, independent of their reusable storage
slots. The fixed-size order array is snapshot state and cannot grow in a
callback. This preserves exact native JSON replay when active grains are restored
into different physical slots. Grain duration uses nearest-frame rounding with
ties to even, matching the NumPy profile. Retained input history and captured
output-grain storage are distinct: slow playback can require a longer output
grain than its source-history span. `maximum_duration_seconds` bounds output
grain duration independently of `history_seconds`. Each native slot reserves the
rounded maximum (at least two frames); duration actions and modulation domains
must remain within that cap. The default resolves to the larger of authored
duration and retained history and is serialized explicitly. Raw callback
parameters exceeding the cap fail without allocating and can recover from a
snapshot.

Input-end actions replace only their named input with silence. Ending main also
releases graph envelopes/stages. Filters and delays follow the declared state
threshold; granulation stops new unfrozen launches and drains active grains.
Frozen grains can continue until unfreeze or stop. Compressor detector endings
permit recovery independently of its primary input. Stop silences its frame and
all subsequent blocks. `LiveRuntime.effects_drained` reports graph completion.
Source clocks under the live owner continue advancing.

Delay retirement uses a maintained count of history values at or above the tail
threshold. Each write updates that count; callbacks do not scan a full history
buffer per frame. Filter modulation applies the authored clamp/error policy
after resolving its parameter, including in native callback rendering.

## Verification and current boundary

The 109-test instrumented run covers the existing FX, source routing, portable
state, release/drain/stop behavior, mixed-width detector branches, duration
rounding, slow grains, rejected snapshots and repeated-input-end recovery. It
recorded zero Rust allocations, reallocations or frees in 4,118 successful and
ten failed callbacks. Audio regressions write paired 48 kHz WAVs lasting at least
one second. Native-to-native JSON replay is additionally checked for exact
sample equality; cross-backend replay uses the numerical audio tolerance.
The full ordinary suite, including the explicit grain-cap integration, passed
3,210 tests with 21 skipped.

Python preparation, offline bridge calls and snapshot conversion allocate outside
the callback guarantee. These checks do not qualify device deadlines.

After the test jobs finished, six sequential ordinary-build measurements used
two seconds of 48 kHz audio, 16 voices and 64 effect actions per callback. Times
are microseconds. Median/p99/maximum exclude the first block; misses include it.

| Profile | Frames | Median | p99 | Maximum | First | Budget | Misses |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Compressor, mono host sidechain | 64 | 276.5 | 363.3 | 874.1 | 670.2 | 1,333.3 | 0 / 1,500 |
| Compressor, mono host sidechain | 256 | 793.1 | 887.2 | 907.2 | 1,071.2 | 5,333.3 | 0 / 375 |
| Compressor, mono host sidechain | 1,024 | 2,899.5 | 10,076.2 | 16,474.6 | 2,970.0 | 21,333.3 | 0 / 93 |
| Dense tap retiming | 64 | 298.4 | 368.8 | 458.0 | 563.6 | 1,333.3 | 0 / 1,500 |
| Modulated delay | 64 | 270.8 | 328.0 | 414.3 | 548.8 | 1,333.3 | 0 / 1,500 |
| Granulator | 64 | 265.3 | 326.0 | 645.5 | 549.2 | 1,333.3 | 0 / 1,500 |

Reproduce with `scripts/benchmark_live.py`, choosing `--compressor`,
`--delay tap`, `--delay modulated`, or `--granular`, and supplying `--seconds 2`,
`--voices 16`, `--effect-actions 64`, and the listed `--block-frames` value.
The larger-block compressor run includes scheduling outliers; all observations
are retained. The earlier startup miss in the original compressor measurement
also remains in [its report](compressor-effects.md).

Per-instrument post-effect attachments and spatial processors remain separate
roadmap slices.

## Additional work beyond the prompt

None.
