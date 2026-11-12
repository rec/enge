# Folding, coloration and bitcrushing

These profiles extend the existing shared NumPy/native effect graph. The
definitions and numerical contracts are in uFor `5738c05`, including the
triangle-fold schema from `7a51fb8`. They use existing scoped controls and
enveLFO routes, fixed prepared storage, portable JSON snapshots and native
opaque snapshots. They are mathematical effects, without hardware-emulation
or device-deadline claims.

## Triangle folding

`Waveshaper(curve="fold")` adds the symmetric triangle transfer
`f(x) = 1 - abs(((x + 1) modulo 4) - 2)`, where modulo is nonnegative.
It preserves x on [-1,1] and reflects repeatedly outside that interval, with
period four. As with the existing curves, the nonlinear stage is
`f(drive*x + bias) - f(bias)`, preserving zero input at zero. The biased wet
range can extend beyond [-1,1]; output gain is explicit.

Folding reuses the [4x Waveshaper profile](waveshaper-effects.md): two cascaded
65-tap half-band stages in each direction, 48 frames of audio/dry latency and
96 frames of FIR draining after input end. Drive/bias controls wait 24 frames
to the nonlinear stage; output gain/mix/bypass wait 48 frames. Curve selection
and coefficients are fixed at preparation. Raw native actions or modulation
targets for those fixed fields are rejected. Snapshot restore rejects a
different fixed curve.

## Tape/tube-style coloration

`audio_effects.coloration_graph(name, scope, input, profile, maximum_block_frames)`
builds three ordinary processors, preserving named input channels:

| Node | Tape | Tube |
| --- | --- | --- |
| `{name}-dc` | 20 Hz Highpass | 20 Hz Highpass |
| `{name}-drive` | Soft Waveshaper, drive 2, bias 0, output gain 0 dB | Soft Waveshaper, drive 2, bias 0.25, output gain 0 dB |
| `{name}-tone` | Two-pole lowpass, 8 kHz, Q=1/sqrt(2) | Two-pole lowpass, 12 kHz, Q=1/sqrt(2) |

The helper accepts authored drive and output gain. The filter stage is named
`low`, so its existing cutoff control is `low-cutoff_hz`. Ordinary graph
authoring can bind controls to each named node. The complete chain has 48
frames of latency. Each node keeps its existing warm bypass, tail and snapshot
behavior; parallel dry routing uses the graph's existing latency alignment.
Preparation rejects cutoffs outside the render rate's usable range.

These presets add no hysteresis, random transport drift or automatic level
normalization. The tube bias supplies asymmetry through the approved
zero-preserving saturation formula.

## Bitcrusher and sample-rate reduction

`Bitcrusher` defaults to 8 bits, a one-frame hold and fully wet output.
At capture, clip each wet input channel to [-1,1], divide by
`step = 2^(1-bits)`, round nearest with ties away from zero, multiply by step,
then clamp to [-1,1-step]. Bits range from 1 to 24. For example, two bits yield
levels -1, -0.5, 0 and 0.5; one bit yields -1 and 0. There is no dither,
oversampling or automatic normalization. Dry/mixed/bypassed audio follows the
ordinary graph gain rules.

Capture on the first input frame and hold that quantized value for
`hold_frames` output frames, from 1 through 65536. Capture and quantize again
at the next boundary. Bit-depth and hold requests are latched at capture;
shortening a hold does not change the interval already in progress. Both
controls require whole values in range on every frame and reject ramps.
Dimensionless integer counts are used in modulation; authored hold values
also accept frame units. Mix and bypass retain their normal ramps. Processing
continues while bypassed, preserving the capture clock.

There is no added graph latency. After input end, finish the current held
interval without another capture, then emit zero and retire. Stop is immediate.
Snapshot state contains one held value per channel and a remaining count in
0..65535. Corrupt state is rejected before replacing the renderer. Failed
callbacks silence the entire caller block, latch failure and recover through
the existing restore path.

## Verification

uFor's full suite passes 1,576 tests. Focused enge rendering checks compare
folding against an independent floor/parity transfer and full-convolution
oversampling oracle. Coloration is compared with independent frequency-domain
highpass/Butterworth transfers around that nonlinear oracle. Bitcrusher checks
cover signed endpoints and ties, clipping, 1/2/8/24 bits, holds through 65536
frames, capture-latched actions, warm bypass, upstream control latency and
held tails after input end. The maximum-hold case renders two seconds.

Audio regressions write paired 48 kHz WAVs lasting at least one second.
Cross-backend JSON replay, native opaque replay, fixed-curve identity,
enveLFO drive/mix/integer-depth routes and rejected native controls are covered.
The final instrumented enge suite passes 3,433 tests, with 29 optional Rubber
Band tests skipped. Its callback properties record zero Rust allocations,
reallocations or frees in 19,194 successful and 47 deliberately failed
callbacks. Ruff, formatting, type checks, pyupgrade, Rust formatting, Clippy
and the allocation-audit positive-control test pass.

Reproduce the suite and summarize its callback properties with the instrumented
native build described in [the callback audit](live-callback-audit.md):

```sh
mkdir -p build/tmp build/verification
TMPDIR="$PWD/build/tmp" uv run --no-sync pytest --require-callback-no-allocations \
  --basetemp=build/verification/nonlinear-final-temp \
  --junitxml=build/verification/nonlinear-final.xml
uv run python scripts/summarize_callback_audit.py \
  --reports build/verification/nonlinear-final.xml
```

On an ordinary build, run `scripts/benchmark_live.py --seconds '2 s'`
`--voices 16 --effect-actions 64 --block-frames FRAMES`, selecting
`--waveshaper --waveshaper-curve fold`, `--coloration tape`,
`--coloration tube`, or `--bitcrusher`. Each case retains the common synth,
gain/filter chain and authored actions. Timing excludes device I/O and does
not qualify arbitrary host workloads.

### Ordinary CPU measurements, 2026-11-12

The sequential runs below use an ordinary release build, without the heap
instrumentation. Folding starts with drive 3, bias 0.2 and mix 0.6, with the
existing Waveshaper actions. Coloration uses the defaults above. Bitcrushing
uses eight bits and a one-frame hold, exercising capture on every frame.
Requested duration is two seconds; 1024-frame runs process 93 whole callbacks.
All times are microseconds. Median/p99/maximum exclude the first callback;
misses include it. Budgets are 1333.3, 5333.3 and 21333.3 microseconds for
64, 256 and 1024 frames respectively.

| Profile | Frames | Median | p99 | Maximum | First | Misses |
| --- | --- | --- | --- | --- | --- | --- |
| Fold | 64 | 1091.4 | 3481.4 | 41441.9 | 1774.0 | 293/1500 |
| Fold | 256 | 3928.7 | 59836.8 | 77829.0 | 41525.6 | 140/375 |
| Fold | 1024 | 11999.0 | 25575.9 | 27465.6 | 14351.3 | 2/93 |
| Tape | 64 | 997.8 | 1433.6 | 3050.1 | 1314.2 | 39/1500 |
| Tape | 256 | 2083.3 | 2849.7 | 3707.8 | 3806.5 | 0/375 |
| Tape | 1024 | 7557.8 | 8845.8 | 11378.0 | 7171.3 | 0/93 |
| Tube | 64 | 1005.7 | 1358.7 | 1903.2 | 1486.3 | 20/1500 |
| Tube | 256 | 2164.2 | 2534.1 | 2671.0 | 2796.5 | 0/375 |
| Tube | 1024 | 4764.9 | 5086.2 | 5435.5 | 5116.9 | 0/93 |
| Bitcrusher | 64 | 636.5 | 1003.1 | 5667.5 | 894.2 | 6/1500 |
| Bitcrusher | 256 | 880.3 | 993.5 | 1021.5 | 1128.8 | 0/375 |
| Bitcrusher | 1024 | 6073.6 | 7337.8 | 8715.2 | 6580.6 | 0/93 |

Because these runs missed deadlines, the same session also measured the
baseline without a new effect, the existing soft Waveshaper, and one repeat
of each failed profile/block combination. These measurements retain the same
voice count, action count and duration. They do not replace the initial runs.

| Comparison | Frames | Median | p99 | Maximum | First | Misses |
| --- | --- | --- | --- | --- | --- | --- |
| Baseline | 64 | 1099.8 | 3545.6 | 128628.6 | 1835.1 | 141/1500 |
| Existing soft Waveshaper | 64 | 969.4 | 1468.3 | 2565.1 | 1435.2 | 43/1500 |
| Fold repeat | 64 | 1377.9 | 10939.6 | 24588.5 | 3008.2 | 1434/1500 |
| Fold repeat | 256 | 1823.5 | 2109.2 | 2656.9 | 2342.9 | 0/375 |
| Fold repeat | 1024 | 5688.2 | 6402.5 | 6431.1 | 6741.2 | 0/93 |
| Tape repeat | 64 | 692.9 | 892.5 | 1492.2 | 1083.8 | 1/1500 |
| Tube repeat | 64 | 1473.7 | 10200.2 | 29581.3 | 2182.4 | 1488/1500 |
| Bitcrusher repeat | 64 | 806.6 | 1454.3 | 6312.3 | 1936.3 | 25/1500 |

Other build and desktop processes were active during timing. Baseline misses
and large run-to-run variation prevent attributing these stalls to one effect.
Each new profile has a zero-miss finite run at 256 and 1024 frames, but this
session does not qualify a reliable 64-frame deadline. The passing DSP and
storage checks do not remove that timing limit. No device deadline is claimed.
Generated logs, audit reports, wheels and regression WAVs remain disposable
outputs under `build/`; their source and reproduction commands are tracked.

## Additional work beyond the prompt

None.
