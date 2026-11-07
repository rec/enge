# Gate and downward expander

Both profiles use linked peak detection across an explicit `detector` input and
apply one gain to all channels of the `input` program. Detector width may differ
from program width. Connecting both ports to the program gives ordinary gating
or expansion; connecting a host or another processor gives a sidechain.
Program channels are preserved, with zero look-ahead or algorithmic latency.

Defaults are threshold -40 dB, 1 ms attack and 50 ms release. Gain smoothing
operates in linear amplitude: `gain += (target-gain)*(1-exp(-1/(rate*seconds)))`.
Increasing gain uses attack; decreasing gain uses release. Zero seconds reaches
the target immediately. The updated gain applies to that same audio frame.

## Gate

The gate starts closed at zero gain. Peak at or above threshold opens it. The
closing threshold is opening threshold minus hysteresis, default 6 dB.
Returning to or above the closing threshold clears elapsed hold. Below it,
the gate remains open for `ceil(hold_seconds*rate)` consecutive frames and closes
on the next low frame. Default hold is 10 ms; zero hold closes on the first low
frame. A closed gate reopens only at the opening threshold.

Hold compares elapsed low frames against the current setting. Shortening an
active hold can close the gate immediately, while lengthening it extends the
wait. Its target gain is one while open and zero while closed. Preparation
rejects hold durations and declared modulation maxima that cannot use the native
integer frame counter. Resolved invalid durations or closing thresholds fail
explicitly in processing.

## Downward expander

The expander starts at unity gain and defaults to ratio 2:1. Below threshold
`T`, detector level `L` dB gives target gain
`10**(((ratio-1)*(L-T))/20)`. At or above threshold the target is one. Ratio one
also targets unity at silence. Otherwise silence targets zero; very small
targets can underflow to zero. Gain never exceeds one and there is no makeup
gain or soft knee.

## Controls and lifetime

Threshold, attack/release, gate hysteresis/hold and expander ratio accept actions,
ramps and shared enveLFO or follower routes. Audio, sidechain, controls and end
markers receive the common upstream latency compensation. Bypass uses the common
fade while keeping detection, hold and gain state warm. Wet/dry mixing follows
the existing graph contract.

An ended detector supplies silence and does not end the program. Neither
processor stores audio, so an aligned program end has no independent dynamics
tail. Required follower-control paths retain their common end rules. Stop is
immediate. JSON snapshots retain gate gain/open/elapsed state or expander gain,
plus automation and upstream alignment state. NumPy and native offline rendering
exchange them; native opaque snapshots replay the same callback state.

Regressions use one-second 48 kHz WAVs and analytical gain curves. They cover
hysteresis, fractional hold rounding, shortened and lengthened active holds,
warm bypass, startup/smoothing, hard-knee expansion, ratio one, tiny/silent
detectors, independent input endings, shared generators, same-frame followers,
delayed sidechains, portable/opaque replay, malformed-state rejection and
failed-block silence/restore recovery.

The ordinary full suite passes 3,312 tests with 29 optional Rubber Band skips.
All 75 focused dynamics, shared-generator and persistent-runtime tests pass.
Ruff, formatting, type checking, pyupgrade, Rust formatting and Clippy pass.
The matching uFor schema `4d51cfa` passed 1,526 tests.

The 133-test instrumented audit records zero allocations, reallocations or frees
in 5,894 successful callbacks and 17 failed callbacks. Preparation, host
admission, Python bridging and snapshot conversion remain outside the native
callback audit.

The separate uFor pin commit was verified against the previous renderer and
its own compiled native module in an isolated source copy: 3,293 tests passed
with 29 skips, and type checking passed with ignored-path filtering disabled.
The runtime integration was tested directly from the changed checkout.

After all test jobs finished, six sequential ordinary-build CPU measurements
used two seconds of 48 kHz audio, 16 voices and 64 effect actions per callback,
including threshold automation and a mono sidechain. Times are microseconds.
Median/p99/maximum exclude the first block; misses include it.

| Profile | Frames | Median | p99 | Maximum | First | Budget | Misses |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Gate | 64 | 339.5 | 422.5 | 457.8 | 713.5 | 1,333.3 | 0 / 1,500 |
| Gate | 256 | 873.6 | 988.5 | 1,015.8 | 1,104.1 | 5,333.3 | 0 / 375 |
| Gate | 1,024 | 2,974.4 | 3,110.0 | 3,111.5 | 3,169.2 | 21,333.3 | 0 / 93 |
| Expander | 64 | 336.7 | 432.7 | 505.4 | 604.2 | 1,333.3 | 0 / 1,500 |
| Expander | 256 | 867.4 | 979.6 | 1,009.5 | 1,129.8 | 5,333.3 | 0 / 375 |
| Expander | 1,024 | 2,980.4 | 3,078.1 | 3,082.5 | 3,201.0 | 21,333.3 | 0 / 93 |

Reproduce with `scripts/benchmark_live.py --dynamics gate` or
`--dynamics expander`, plus `--seconds '2 s' --voices 16 --effect-actions 64`
and the listed `--block-frames`. These finite measurements qualify the tested
development-machine CPU paths, not an audio device or another platform.

## Additional work beyond the prompt

None.
