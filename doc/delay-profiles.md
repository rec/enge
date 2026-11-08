# Multitap and damped feedback delay

## Multitap composition

uFor's `multitap_graph` expands named `DelayTap` settings into existing TapDelay,
channel-matrix and Mixer nodes. Each tap declares a delay, prepared maximum,
gain in dB and static placement. Mono uses equal-power pan; named left/right
stereo uses linear attenuation-only balance. Output is named left/right stereo.
Dry gain defaults to 0 dB, with equal-power centered mono mapping or unchanged
stereo mapping. Tap gain defaults to -6 dB. There is no automatic normalization
or clipping.

For graph `echo` and tap `one`, `echo-tap-one.delay_seconds` controls retiming;
`echo.tap-one-gain_db` controls level. The mixer also exposes the usual mute/solo
selectors. Placement is a static matrix. Tap count and each tap's maximum delay
fix prepared storage. Musical echo time is not algorithmic latency, so dry audio
is not delayed to match it. The mixer ends after all tap histories drain.

This uses the same DSP in NumPy, native offline and native callbacks. There is
no additional multitap runtime or snapshot format. Standard graph lifecycle,
automation, latency alignment and state replay apply to every constituent node.

## Feedback damping

`TapDelay.damping_hz=None` selects the undamped loop. A positive frequency prepares
one normalized all-pass memory per channel. Its low-pass response is the
complement of the existing highpass kernel, with unity DC gain and zero gain at
Nyquist. For `g=tan(pi*cutoff/rate)` and `a=(g-1)/(g+1)`,
`L(z)=(1+a)/2*(1+z^-1)/(1+a*z^-1)`.

The delayed read stays audible without filtering. Its low-pass complement is
multiplied by feedback gain and added to new input before history is written.
Thus the first echo retains the input's spectrum and subsequent repeats become
progressively darker. Interpolation, continuous fixed-tap retiming, wet/dry and
warm bypass retain their existing contracts. Damping adds no algorithmic latency.

Prepared cutoff accepts actions, ramps and shared enveLFO/follower routes.
Preparation checks authored and declared modulation bounds against Nyquist and
coefficient representability. Resolved invalid values fail without clamping.
An undamped preparation rejects damping actions; another definition must be
prepared to enable the filter.

Native processing reuses one preallocated graph scratch vector for feedback and
keeps only damping memory in persistent delay state. After input end, both delay
history and damping memory must fall below the declared tail threshold before
retirement. State-floor trimming uses the existing threshold. Non-finite feedback
writes fail before entering history, including when damping is disabled.

JSON snapshots retain damping memory. Native opaque snapshots also preserve
enabled/disabled preparation identity; cross-mode and malformed-memory restores
are rejected before replacing state. Callback failures silence the complete
caller block and latch failure until restore/reset.

Regressions write one-second 48 kHz WAVs. Multitap references use independent
fractional-read and placement formulas with gain ramps, interrupted-state replay
and explicit endings. Damping references use the feedback transfer
`D(z)/(1-feedback*L(z)*D(z))`, including fractional reads. Tests cover first-echo
preservation, pending filter-tail replay, shared generator controls, mode identity,
invalid cutoff actions, overflow, failed-block silence and recovery.

The initial ordinary suite passed 3,328 tests with 29 optional Rubber Band skips.
After the final restore-validation refinement, all 82 focused delay/control
regressions passed, followed by the complete instrumented suite: 3,328 passed
with the same 29 skips. That full audit records zero allocations, reallocations
or frees in 15,539 successful and 32 failed Rust callbacks, including existing
synth, sampler and effect paths. Preparation, Python bridges, host admission and
snapshot conversion remain outside the native callback audit.

Ruff, formatting, type checking, pyupgrade, Rust formatting and Clippy pass.
The pinned uFor revision `a8fdc72` includes multitap authoring `2415d82` and
feedback damping; its complete suite passed 1,530 tests.

## CPU timing evidence

After restoring the ordinary build and finishing test jobs, sequential runs used
two seconds of 48 kHz audio, 16 voices and 64 effect actions per callback.
Multitap uses four stereo taps with delay/gain actions. Damped delay uses time
and cutoff actions, starting at 1 kHz; its cutoff sweeps downward within the
prepared profile. Times are microseconds. Median/p99/maximum exclude the first
block; misses include it. These wall-clock measurements include host-side work
inside `advance_into` as well as native rendering.

| Profile/run | Frames | Median | p99 | Maximum | First | Budget | Misses |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Multitap | 64 | 480.8 | 575.7 | 648.7 | 733.5 | 1,333.3 | 0 / 1,500 |
| Multitap | 256 | 1,281.8 | 1,403.5 | 1,550.5 | 1,589.8 | 5,333.3 | 0 / 375 |
| Multitap | 1,024 | 4,478.4 | 4,634.8 | 4,654.9 | 4,818.1 | 21,333.3 | 0 / 93 |
| Damped, initial | 64 | 385.4 | 5,680.5 | 44,952.5 | 636.5 | 1,333.3 | 41 / 1,500 |
| Damped, initial | 256 | 958.4 | 4,044.4 | 25,538.3 | 1,196.0 | 5,333.3 | 3 / 375 |
| Damped, initial | 1,024 | 3,189.2 | 3,306.0 | 3,346.2 | 3,304.8 | 21,333.3 | 0 / 93 |
| Undamped control | 64 | 388.5 | 496.0 | 646.6 | 794.5 | 1,333.3 | 0 / 1,500 |
| Damped, repeat 1 | 64 | 385.2 | 474.9 | 565.8 | 619.4 | 1,333.3 | 0 / 1,500 |
| Damped, repeat 2 | 64 | 374.5 | 447.5 | 514.0 | 719.7 | 1,333.3 | 0 / 1,500 |
| Damped, repeat 3 | 64 | 373.7 | 444.9 | 499.2 | 592.7 | 1,333.3 | 0 / 1,500 |
| Damped, repeat 1 | 256 | 933.5 | 1,022.0 | 1,065.3 | 1,202.9 | 5,333.3 | 0 / 375 |
| Damped, repeat 2 | 256 | 924.4 | 1,031.4 | 1,082.8 | 1,151.4 | 5,333.3 | 0 / 375 |

The initial damped runs had intermittent wall-clock stalls. They did not recur
in the five follow-up runs, but their cause is unresolved; the repeated zero
misses do not erase the initial misses or establish a device-deadline guarantee.
These finite measurements cover this development machine and profile only.

Reproduce with `scripts/benchmark_live.py --delay multitap` or
`--delay tap --damping-hz '1 kHz'`, plus `--seconds '2 s' --voices 16`
`--effect-actions 64` and the listed `--block-frames`. The undamped control omits
`--damping-hz`; that profile spends all its actions on retiming.

## Additional work beyond the prompt

None.
