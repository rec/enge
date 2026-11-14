# Rectification, enhancement and octave division

N4 uses the shared definitions in uFor `b69f30b`. The profiles use the existing
NumPy/native graph, scoped controls and enveLFO machinery. They are explicit
mathematical effects, without hardware-emulation or pitch-tracking claims.

## Rectifiers

`Waveshaper(curve="half_wave")` uses `max(x, 0)`; `full_wave` uses `abs(x)`.
Both apply `f(drive*x + bias) - f(bias)` and the existing output gain. Bias can
make the zero-preserving output bipolar; without bias, rectification produces
DC intentionally. Compose a Highpass when DC removal is wanted.

The [existing oversampled Waveshaper](waveshaper-effects.md) supplies its two
65-tap 2x stages, 48-frame latency, dry/control compensation, warm bypass and
96-frame FIR tail. Curves and coefficients stay fixed at preparation.

## High-band harmonic enhancer

`ufor.audio_effects.enhancer_graph` builds ordinary named processors:

| Node suffix | Definition |
| --- | --- |
| `-high` | 2 kHz one-pole Highpass |
| `-drive` | Soft Waveshaper, drive 2, bias 0.25, output gain 0 dB |
| `-dc` | 20 Hz one-pole Highpass |
| `-mix` | Mixer: unity dry plus 0.25-amplitude processed branch |

The graph preserves named channels and delays dry by 48 frames to align the
branches. Existing node controls remain available. There is no normalization,
output ceiling or hardware model. Mix/bypass is authored with the Mixer and
the component processors' existing controls. Filter tails drain normally.

## Octave divider

`OctaveDivider` detects each channel independently. The detector starts
unarmed and silent. Input at or below `-threshold` arms it; a later input at
or above `threshold` flips square-wave polarity and disarms it. The first
flip produces positive output. Requiring a complete negative-to-positive
traversal prevents small zero crossings from retriggering. Defaults are
threshold 0.01, attack 1 ms, release 50 ms and fully wet output.

Each channel's peak envelope scales the square wave, using the existing
Follower exponential attack/release law. A zero envelope resets detector phase.
Threshold, attack, release and mix accept scoped controls and ramps. Threshold
must stay strictly positive; times must stay nonnegative. Processing remains
warm while bypassed. This is a monophonic effect per channel: chords, noise and
signals below the detector thresholds need not produce a useful octave.

There is no intrinsic latency. Upstream latency delays its ordinary controls
and bypass together with audio. Input end supplies zeros and retains the last
polarity while the envelope decays. The default state floor is 1e-30; retirement
occurs when all channel envelopes are below the default 1e-12 tail threshold,
then clears state. Stop is immediate. Portable snapshots contain four lists:
envelope levels, armed flags, polarity flags and active flags. Malformed state
is rejected before replacing the renderer. Native opaque snapshots retain the
same state, and callbacks use only prepared per-channel storage.

## Verification

uFor passes 1,584 tests with static checks clean. Rectifier regressions use
independent algebraic transfer functions inside the full-convolution
oversampling oracle. The enhancer uses independent frequency-domain highpass
transfers around that oracle and an explicitly delayed dry path. The divider
uses independent periodic square inputs at two frequencies and an analytical
attack/release envelope, covering startup phase, octave-down frequency, tails,
hysteresis, warm bypass, upstream latency, immediate stop, malformed snapshots,
failure latching and restore. Audio regressions write 48 kHz WAVs lasting at
least one second; the default divider tail case lasts two seconds.

Cross-backend JSON replay, native opaque replay, arbitrary callback partitions,
enveLFO routes and generator release are covered. The generic one-second
generator-release fixture uses a 20 ms divider release to fit its capture;
the separate default-profile regression retains 50 ms.

The audit ran 3,446 tests successfully with 29 optional Rubber Band skips.
Six development-tool tests initially failed collection because the standalone
pytest entry point omitted the repository root after removing editable
installs. Those six pass with `python -m pytest`, and full collection succeeds
with that invocation (3,481 cases: 3,452 tests plus 29 skips). No DSP change was
needed for the import path. Combined callback properties show zero Rust
allocations, reallocations or frees in 20,277 successful and 51 deliberately
failed callbacks. The initial report retains one synthetic collection-error
case; it is not an additional test. Ruff, formatting, ty, pyupgrade, Rust
formatting, Clippy and the heap-audit positive control pass.

Build and install a wheel without editable dependencies before the audit:

```sh
mkdir -p build/tmp build/wheels build/verification
TMPDIR="$PWD/build/tmp" uv sync --locked --no-editable
TMPDIR="$PWD/build/tmp" uv run --with maturin --no-sync maturin build \
  --release --locked --features allocation-audit --out build/wheels
uv pip install --no-deps --reinstall build/wheels/enge-*.whl
TMPDIR="$PWD/build/tmp" uv run --no-sync python -m pytest \
  --require-callback-no-allocations \
  --basetemp=build/verification/n4-full-temp \
  --junitxml=build/verification/n4-full.xml
uv run --no-sync python scripts/summarize_callback_audit.py \
  --reports build/verification/n4-full.xml
```

On an ordinary build without `allocation-audit`, the existing CPU benchmark
accepts `--waveshaper --waveshaper-curve half_wave`, `full_wave`, or
`--nonlinear enhancer` / `--nonlinear octave`. Use `--seconds '2 s'`,
`--voices 16 --effect-actions 64` and the desired `--block-frames`.
Shared-machine scheduling misses are measurements, not completion blockers;
these runs do not qualify an audio-device deadline.

### Ordinary CPU measurements

Sequential ordinary-release measurements on 2026-11-14 use 16 voices, 64
authored effect actions and a requested two seconds at 48 kHz. The 1024-frame
runs render 93 whole callbacks. Times below are microseconds; median/p99/max
exclude the first callback and misses include it. Budgets for 64/256/1024
frames are 1333.3/5333.3/21333.3 microseconds. Rectifiers use the existing
benchmark drive/bias/mix and actions; enhancer/divider use their defaults.

| Profile | Frames | Median | p99 | Maximum | First | Misses |
| --- | --- | --- | --- | --- | --- | --- |
| Half-wave | 64 | 487.1 | 1888.9 | 28252.4 | 814.5 | 25/1500 |
| Half-wave | 256 | 1313.9 | 1414.0 | 1436.8 | 1545.4 | 0/375 |
| Half-wave | 1024 | 4601.6 | 4769.6 | 4802.0 | 4806.7 | 0/93 |
| Full-wave | 64 | 466.5 | 545.5 | 1202.6 | 715.3 | 0/1500 |
| Full-wave | 256 | 1283.0 | 1438.5 | 1908.5 | 1523.4 | 0/375 |
| Full-wave | 1024 | 4514.5 | 4747.9 | 4771.4 | 4969.2 | 0/93 |
| Enhancer | 64 | 514.1 | 607.3 | 664.5 | 761.5 | 0/1500 |
| Enhancer | 256 | 1435.9 | 1546.4 | 1681.9 | 1723.0 | 0/375 |
| Enhancer | 1024 | 5090.4 | 5203.4 | 5225.8 | 5255.8 | 0/93 |
| Octave divider | 64 | 386.2 | 464.0 | 511.0 | 632.4 | 0/1500 |
| Octave divider | 256 | 913.6 | 998.3 | 1042.9 | 1166.2 | 0/375 |
| Octave divider | 1024 | 2993.8 | 3121.5 | 3123.7 | 3393.9 | 0/93 |

The half-wave stalls are retained without repeating runs to seek a zero-miss
result. The machine was shared with other work; these measurements are not a
dedicated-host qualification.

All implementation and verification source is tracked. Wheels, reports and
regression audio under `build/` are disposable generated artifacts.

## Additional work beyond the prompt

None.
