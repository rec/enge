# Karplus-style resonant delay

`ResonantDelay` is an input-excited, fixed-tuning feedback resonator. It preserves
named channels independently and uses the existing normalized all-pass filter
machinery with a bounded circular history. It is an effect; excitation comes from
its input rather than an internal note or noise generator.

## Prepared tuning

`frequency_hz` and `damping_hz` are fixed at preparation. Frequency must be
positive and at most one quarter of the output sample rate, inclusive (12 kHz
at 48 kHz). Damping is the existing normalized one-pole low-pass complement,
with a positive cutoff below Nyquist. Unrepresentable coefficients and history
capacities are rejected.

For angular target frequency w, let phi be the damping filter's phase lag.
The integer history is N=floor((2*pi-phi)/w-0.5) samples. A first-order all-pass
supplies the remaining phase theta=2*pi-phi-N*w, using
q=tan(theta/2)/tan(w/2), a=(1-q)/(1+q). This includes damping in the open-loop
phase target. The quarter-rate limit keeps the residual delay around half to
one-and-a-half samples, avoiding near-cancellation tuning coefficients.

The transfer functions are D(z)=z^-N, A(z)=(a+z^-1)/(1+a*z^-1), and the existing
low-pass L(z). The audible wet signal is the tuned read A*D, followed by damping
only inside feedback. For scalar loss g, the complete wet transfer is
`A*D / (1 - g*L*A*D)`. All channels use their own history and filter memories.

The authored frequency is the open-loop phase tuning target. Very short or
strongly damped responses need not have a spectral peak exactly at that
frequency. This follows Smith's [fractional-delay model](https://www.dsprelated.com/freebooks/pasp/Delay_Line_Signal_Interpolation.html)
and [loop-filter phase compensation](https://www.dsprelated.com/freebooks/pasp/Loop_Filter_Identification.html).

## Decay and controls

`decay_seconds`, default 1, specifies nominal undamped T60. Scalar feedback is
`g=exp(-3*ln(10)/(frequency_hz*decay_seconds))`. Damping can shorten audible decay
further. This is a scalar-loss convention rather than a measured modal decay
guarantee. There is no compensating amplification or saturation. Gains that
round to one are rejected; sufficiently short decays can yield zero feedback.

Decay, mix and bypass accept ordinary actions and shared control/generator
routes. Preparation checks declared decay modulation bounds. Frequency and
damping actions/targets are rejected. Mix defaults to fully wet; bypass stays
warm. Musical resonance delay adds no graph alignment latency.

## State and verification

Portable JSON and native opaque snapshots retain history, write position and
both normalized filter memories. Opaque restore also checks fixed tuning and
damping identity. Malformed state is rejected before replacing the renderer.
Input-end retirement requires the history and both memories to be below the
tail threshold. In particular, zero feedback can empty the history while the
tuning all-pass still has audible output. Callback failures silence the complete
caller block and latch failure until restore/reset.

The shared definition/specification is uFor `436b230`, with 1,546 tests passing.
Twenty-one focused resonator cases pass. Their one-second 48 kHz WAVs compare
rendering against an independent frequency-domain feedback transfer, including
55 Hz, 440 Hz, 9,973 Hz and the 12 kHz endpoint. They check compensated loop phase,
active decay ramps, mix and warm bypass, filter-only tails, malformed snapshot
preservation, static tuning identity, invalid actions, finite-input overflow,
failed-block silence and recovery. Existing shared enveLFO and cross-backend
snapshot regressions also include this processor.

The 440 Hz transfer case uses a 20-second nominal decay. Its finite-loop-series
frequency-domain reference prevents later tail energy wrapping into the one-second
record through the FFT. It remains active after input end while portable and
native replay preserve the long resonance.

The full instrumented suite passes 3,376 tests with 29 optional Rubber Band skips.
It records zero Rust allocations, reallocations or frees in 17,115 successful
and 37 failed callbacks. Preparation, Python bridges, host admission and snapshot
conversion remain outside the callback audit. Rust tests, Clippy, formatting,
Ruff, type checking and pyupgrade pass. The separate uFor pin `6239b36` passed
3,350 tests with 29 optional skips and type checking against the unchanged
renderer in an isolated checkout.

## CPU timing

The same renderer source used in the final suite was measured on the ordinary
build, with 440 Hz tuning, 6 kHz damping, 16 synth voices and 64 decay/mix actions
per callback. Three sequential runs each rendered two seconds at 48 kHz. Times
are microseconds; median/p99/maximum exclude the first block, while misses include
it. The interval includes host admission and native processing.

| Frames | Median | p99 | Maximum | First | Budget | Misses |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 64 | 330.8 | 413.4 | 473.2 | 687.6 | 1,333.3 | 0 / 1,500 |
| 256 | 862.1 | 947.0 | 967.4 | 1,153.1 | 5,333.3 | 0 / 375 |
| 1,024 | 2,928.9 | 3,066.1 | 3,103.0 | 3,032.8 | 21,333.3 | 0 / 93 |

Reproduce with `scripts/benchmark_live.py --delay resonant --seconds '2 s'`
`--voices 16 --effect-actions 64` and the listed `--block-frames`. These finite
measurements cover this development machine and authored profile without device
I/O; they do not establish device deadlines for other settings or hosts.

## Additional work beyond the prompt

None.
