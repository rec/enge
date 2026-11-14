# Broadband de-esser

The selected N7 profile is `ufor.audio_effects.deesser_graph`. It composes
the existing normalized band-pass filter and linked peak compressor. No new
DSP kernel, processor kind, snapshot representation or intrinsic latency is
introduced. The graph preserves the input's named channel layout.

## Signal and detector paths

The original signal feeds `<name>-compressor.input` directly. A parallel
`<name>-detector` filter feeds only `<name>-compressor.detector`; filtered
audio is never mixed into the output. The detector contains one normalized
two-pole band-pass filter named `band`, centered at 6 kHz with Q 1. Its gain
at the center frequency is unity. The greatest absolute detector sample
across channels drives one common compressor gain for all output channels.

| Setting | Default |
| --- | --- |
| Threshold | -24 dBFS |
| Ratio | 4:1 |
| Soft knee | 6 dB |
| Attack time constant | 1 ms |
| Release time constant | 50 ms |
| Makeup gain | 0 dB |
| Mix | 1 |

The existing [compressor law](compressor-effects.md) defines gain reduction
and smoothing. Reduction starts at zero. This profile has no look-ahead,
normalization, speech classifier or separate maximum-reduction cap. Sibilant
energy attenuates the whole signal, including its low frequencies. Strong
non-vocal energy around the detector band also triggers reduction. This is
a broadband de-esser; it does not reconstruct separately processed bands.

## Controls and lifecycle

The helper accepts the name, attachment scope, graph input, maximum callback
size and optional owner, following the existing composition helpers. It
returns an ordinary graph with explicit processor names. Existing actions
and scoped enveLFO routes can target `band-cutoff_hz` and `band-q` on
`<name>-detector`, or `threshold_db`, `ratio`, `knee_db`, `attack_seconds`,
`release_seconds`, `makeup_db` and `mix` on `<name>-compressor`. The initial
profile uses the defaults above; authoring changes use those processor fields.
Cutoff obeys existing filter bounds. The default 6 kHz center is rejected at
an 8 kHz sample rate, rather than silently moved into another band.

Use the compressor's mix or warm bypass for the complete de-esser. Its
64-frame bypass fade keeps the detector and reduction advancing. Bypassing
only the detector instead supplies unfiltered audio to detection and changes
the effect into broadband compression. There is no additional wrapper bypass.

Input end silences the original signal and output immediately. The filter
may drain its internal state afterward, without an audible output tail.
Stop, portable cross-backend snapshots, native opaque snapshots and callback
failures retain the constituent processors' contracts. Captured envelopes
retain the native-effect preparation restriction documented in
[the transient-shaper profile](transient-shaper.md).

## Verification

The uFor helper and conformance fixture are published in `877f5dc`; its full
suite passes 1,610 tests. The matching enveLFO dependency-only update is
`18405d8`, with 217 tests passing. No package versions outside these required
pins changed. Enge's full instrumented suite passes 3,491 tests with 29
optional skips. The audit records zero allocations, reallocations or frees
in 21,407 successful and 57 failed callbacks. Reports are
`build/verification/n7-full.xml` and `build/verification/n7-heap.json`.

The independent oracle uses the bilinear
band-pass transfer in the frequency domain, followed by the declared static
knee and exponential gain recurrence. It does not call the production filter
or compressor. Regressions cover low-band preservation below threshold,
sibilant bursts in either stereo channel, linked full-band reduction,
portable replay across both backends, native callback replay, threshold
enveLFO modulation, warm bypass, exact unity-ratio signal preservation,
input end and invalid sample-rate bounds. Audio comparisons write paired
48 kHz WAVs lasting one second.

The four focused regressions pass with zero Rust heap operations in 222
successful callbacks. Python lint, formatting, type checks and pyupgrade pass.

The benchmark harness accepts `--nonlinear deesser`. It appends the default
composition to the common 16-voice, 64-action source/filter chain. Run sizes
sequentially with an ordinary build, for example:

```sh
TMPDIR="$PWD/build/tmp" uv run --no-sync python scripts/benchmark_live.py \
  --seconds '2 s' --voices 16 --effect-actions 64 \
  --block-frames 64 --nonlinear deesser
```

Repeat at 256 and 1,024 frames. CPU measurements exclude audio-device I/O.
Build the allocation-audit wheel and run the full suite using the procedure
in [the transient-shaper profile](transient-shaper.md), with `n7` report and
wheel directories. Source and verification code are tracked; reports, wheels
and regression WAVs under `build/` are disposable generated artifacts.

### Ordinary CPU measurements, 2026-11-14

The three runs above were sequential, after pytest finished, using an ordinary
release build, 16 voices and 64 effect actions per callback. Requested audio
duration was two seconds at 48 kHz. The 1,024-frame run processes 93 whole
callbacks. Times are microseconds; median/p99/maximum exclude the first callback,
while misses include it.

| Frames | Median | p99 | Maximum | First | Budget | Misses |
| --- | --- | --- | --- | --- | --- | --- |
| 64 | 641.5 | 13598.0 | 112089.8 | 1239.4 | 1333.3 | 190/1500 |
| 256 | 1649.9 | 20434.3 | 34239.5 | 1871.8 | 5333.3 | 33/375 |
| 1024 | 5417.1 | 45535.4 | 51548.1 | 30041.5 | 21333.3 | 6/93 |

These shared-machine runs include deadline misses at all three sizes. Median
processing times are below their budgets, but the observed long pauses prevent
a deadline qualification. These measurements do not isolate the cause of the
pauses or include audio-device I/O. All results are retained in
`build/verification/n7-deesser-{64,256,1024}.log`; no zero-miss reruns replace them.
Functional and allocation checks pass independently of this timing limitation.

## Additional work beyond the prompt

None. The enveLFO dependency-only pin follows the required uFor revision so
the resolver sees one shared model version; its execution code is unchanged.
