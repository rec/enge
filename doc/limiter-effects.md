# Sample-peak limiter

The approved N5 profile uses uFor `6009862`: linked named channels, a default
-1 dBFS ceiling, 5 ms prepared look-ahead, instantaneous gain reduction and
50 ms release. It is a sample-peak limiter, without intersample/true-peak or
transparent attack claims. No external DSP dependency is required.

## Timing and gain

Prepare `L = ceil(lookahead_seconds * sample_rate)` frames of look-ahead, with
positive duration. The latency API now requires the render rate explicitly:
`EffectGraph.latencies(sample_rate)` and `processor_latency(processor, sample_rate)`.
Five milliseconds becomes 221 frames at 44.1 kHz, 240 at 48 kHz and 480 at 96 kHz.
Look-ahead stays fixed at preparation. Ceiling, release, mix and bypass can
change through the existing action/scoped-control path. Ceiling must remain
at most 0 dBFS and release nonnegative.

At time t, take the maximum absolute sample across every channel and the
inclusive input window [t-L,t]. With C=10^(ceiling_db/20), target gain is
min(1,C/peak), or one for a silent window. Release the previous gain toward
one with coefficient `1-exp(-1/(sample_rate*release_seconds))`; zero release
uses one immediately. Apply the smaller of target and released gain to input
frame t-L. Clamp the wet result to [-C,C] to contain floating-point rounding.
One gain links all channels. Initial gain is one and all audio history is zero.

The dry signal receives the same L-frame delay. Ceiling, release and
mix/bypass controls receive upstream latency plus L, aligning authored changes
with output audio. A fully wet limiter satisfies the sample ceiling; mixing
dry, bypassing, summing later branches or adding downstream gain can exceed it.
Bypass remains warm. Input end feeds zeros, emits the final L delayed frames
and retires; stop is immediate, without waiting for gain recovery.

## State and bounds

Portable state consists of the L+1-frame input history, the next ring write
position, gain and remaining tail count. Restore validates dimensions, finite
audio, clocks and gain before replacing the renderer. Rust derives a monotonic
maximum queue from the saved history outside processing. Opaque snapshots
retain the queue directly. All ring positions stay bounded; the queue uses
prepared arrays and no ever-growing sample counter. NumPy uses a direct
window maximum as an independent realization of the same contract.

## Verification

Shared definitions pass 1,591 uFor tests. The complete enge instrumented suite
passes 3,466 tests with 29 optional Rubber Band skips. Its callback properties
record zero Rust allocations, reallocations or frees in 20,731 successful and
54 deliberately failed callbacks. Fourteen focused instrumented checks pass.
Ruff, formatting, ty, pyupgrade, Rust formatting, Clippy and the heap-audit
positive control pass. The rendering regressions use
analytical pulse attenuation/recovery, signed ceilings, linked channels,
zero/default release, look-ahead replay, control alignment, warm bypass,
immediate stop, invalid controls and malformed snapshots. Audio regressions
use 48 kHz WAVs lasting at least one second.

Build and test without editable installs using the same wheel for both the
installed package and the checkout extension. In a POSIX shell, with one
current-platform wheel in the output directory:

```sh
mkdir -p build/tmp build/wheels/n5-audit build/verification
TMPDIR="$PWD/build/tmp" uv sync --locked --no-editable
TMPDIR="$PWD/build/tmp" uv run --with maturin --no-sync maturin build \
  --release --locked --features allocation-audit --out build/wheels/n5-audit
uv pip install --no-deps --reinstall build/wheels/n5-audit/enge-*.whl
unzip -o -j build/wheels/n5-audit/enge-*.whl 'enge/_native*' -d enge
TMPDIR="$PWD/build/tmp" uv run --no-sync python -m pytest \
  --require-callback-no-allocations \
  --basetemp=build/verification/n5-full-temp \
  --junitxml=build/verification/n5-full.xml
uv run --no-sync python scripts/summarize_callback_audit.py \
  --reports build/verification/n5-full.xml
```

For ordinary CPU measurements, rebuild without `allocation-audit`, install and
extract that wheel, then run `scripts/benchmark_live.py --nonlinear limiter`
with the existing voice/action/block settings. Ceiling verification applies to
the represented floating-point control values. Independent ramp calculations
allow a few ulps of numerical difference; constant-ceiling assertions are strict.

### Ordinary CPU measurements, 2026-11-14

Sequential ordinary-release runs use the default limiter after the common
gain/filter chain, 16 synth voices, 64 authored actions and a requested two
seconds at 48 kHz. The 1024-frame run processes 93 whole callbacks. Times are
microseconds; median/p99/max exclude the first callback, while misses include it.

| Frames | Median | p99 | Maximum | First | Budget | Misses |
| --- | --- | --- | --- | --- | --- | --- |
| 64 | 396.0 | 483.2 | 525.2 | 741.5 | 1333.3 | 0/1500 |
| 256 | 922.9 | 1015.6 | 1036.3 | 1174.4 | 5333.3 | 0/375 |
| 1024 | 3029.0 | 3123.6 | 3130.0 | 3295.5 | 21333.3 | 0/93 |

Generated wheels, reports and audio belong under `build/`; implementation and
verification source is tracked. Shared-machine timing misses do not block this
mathematical profile, and finite CPU measurements do not qualify a device
deadline.

## Additional work beyond the prompt

None.
