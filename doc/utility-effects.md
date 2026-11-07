# Utility and routing effects

## One-pole highpass and DC removal

The `Highpass` processor preserves input channels, defaults to 20 Hz and has
unity gain at Nyquist. It uses one initially zero normalized all-pass memory
per channel. `g = tan(pi*cutoff/rate)`, `a = (g-1)/(g+1)`, and
`b = sqrt((1-a)*(1+a))`. Using the old memory `s`, each frame computes
`allpass = a*x+b*s`, `s_next = b*x-a*s`, `output = (x-allpass)/2`.
At fixed cutoff this gives `(1-a)/2*(1-z^-1)/(1+a*z^-1)`; the normalized
lattice retains an energy balance when cutoff changes.

Cutoff accepts ramps and shared enveLFO/Patch routes. It must remain positive
and strictly below Nyquist, with a representable coefficient strictly between
-1 and 1. Preparation and resolved rows reject invalid values without clamping.
The processor adds no algorithmic latency. Common upstream audio/control/end
compensation applies. Bypass keeps state warm; after input end it drains until
all memories are below `tail_threshold`. Retained values below `state_floor`
become zero. Stop remains immediate.

NumPy and the shared native offline/live core implement the same profile.
JSON snapshots retain normalized memory and common ramp/compensation state;
native opaque snapshots replay it exactly. Invalid snapshot shapes are rejected
before replacing native state. Callback errors zero the caller's complete block
and latch failure until restore/reset.

The focused suite checks the analytical complex frequency response and DC-step
response, tail drain, cutoff ramps, warm bypass, cross-backend JSON replay,
native callback partition/replay, shared generators and coefficient-error
recovery. Audio regressions write one-second 48 kHz WAVs. The 63-test focused
suite passes. Its allocation audit records zero heap operations in 2,530
successful and five failed callbacks. Preparation, Python bridge and snapshot
conversion allocations remain outside that callback guarantee; device deadlines
are not qualified by these checks.

## Always-applied channel matrices

`ChannelMatrix` declares ordered input and output channel names and a finite
coefficient row for each output. It applies without wet/dry or bypass controls;
parameter and bypass actions are rejected. The source layout must match the
declared input names and order. Channel counts may differ. There is no implicit
normalization or clipping.

Static uFor presets emit ordinary matrices for polarity inversion, equal-power
mono-to-stereo pan, attenuation-only stereo balance, width and orthonormal
mid/side encode/decode. Width 0 folds to mono, 1 preserves stereo, and 2 doubles
the side signal. Explicit coefficient rows also cover swap, sum/difference and
arbitrary channel mapping. Time-varying panning is a later profile.

NumPy uses scalar `fsum` over coefficient/sample products. Native rendering
reuses the existing compensated accumulator with one preallocated partial per
input channel. Cancellation retains small terms. Non-finite products and
unrepresentable sums fail rather than clip. The native definition retains its
coefficients independently of control rows; opaque snapshots with different
matrix coefficients are rejected before changing state.

Matrices add no algorithmic latency or audio tail. They end with their input
and preserve upstream latency and end-marker compensation. DSP is stateless;
shared graph lifecycle and upstream state still participate in JSON and opaque
snapshots. The focused regressions cover all presets, unequal channel counts,
cancellation, input end, upstream Waveshaper latency, portable/callback replay,
invalid raw actions, overflow failure/recovery and snapshot identity.
All 13 focused tests pass. Their callback audit records zero heap operations in
743 successful and two failed callbacks. uFor's combined highpass/matrix schema
suite passed 1,519 tests. The subsequent mixer and crossfader implementation
is documented in [mixing profiles](mixer-effects.md).

## Validation and timing

The full ordinary suite passed 3,277 tests with 29 optional Rubber Band skips.
Ruff, formatting, type checking, pyupgrade and Rust lint/format checks passed.
The standalone uFor pin baseline passed 3,252 tests with the same 29 skips;
the renderer integration was then tested directly from the changed checkout,
including all new Highpass and matrix regressions.

After all test jobs finished, six sequential ordinary-build measurements used
two seconds of 48 kHz audio, 16 voices and 64 effect actions per callback. Times
are microseconds. Median/p99/maximum exclude the first block; misses include it.

| Profile | Frames | Median | p99 | Maximum | First | Budget | Misses |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Highpass | 64 | 288.1 | 360.5 | 444.7 | 617.4 | 1,333.3 | 0 / 1,500 |
| Highpass | 256 | 796.4 | 848.4 | 920.8 | 1,025.9 | 5,333.3 | 0 / 375 |
| Highpass | 1,024 | 2,827.1 | 2,974.8 | 2,977.5 | 3,076.8 | 21,333.3 | 0 / 93 |
| Matrix | 64 | 296.4 | 354.7 | 423.7 | 566.8 | 1,333.3 | 0 / 1,500 |
| Matrix | 256 | 807.5 | 914.5 | 1,038.5 | 1,039.5 | 5,333.3 | 0 / 375 |
| Matrix | 1,024 | 2,836.5 | 2,943.2 | 3,000.7 | 3,103.2 | 21,333.3 | 0 / 93 |

Reproduce with `scripts/benchmark_live.py --highpass` or `--matrix`, plus
`--seconds '2 s' --voices 16 --effect-actions 64 --block-frames 64`, selecting
the other listed block sizes for the remaining rows. These finite CPU checks
on the development machine do not qualify an audio device or another platform.

## Additional work beyond the prompt

None.
