# Mixer and crossfader effects

`Mixer` declares an ordered, nonempty list of named inputs. The authored port
count fixes native storage at preparation; it is not limited to two. Every
input must have the same channel names in the same order. Each input exposes
`<port>-gain_db`, `<port>-muted` and `<port>-solo` action parameters. Gain accepts
ramps and shared enveLFO routes. Muted and solo accept only instantaneous zero
or one actions.

With no solo selected, every unmuted input contributes. With any solo selected,
only unmuted soloed inputs contribute. Initial eligibility applies immediately.
Later eligibility changes fade linearly from the current amplitude over
`switch_fade_frames`, default 64. The request frame uses the current amplitude;
the target is reached after that many frames. A second change retimes from the
current amplitude. Repeating an unchanged target does not restart a fade.
Input gain in decibels multiplies this eligibility amplitude.

`Crossfader` has named `first` and `second` ports with matching layouts. Its
required law is `linear` or `equal_power`; its position ranges from zero to one
and accepts ramps and shared enveLFO routes. Linear weights are `(1-p, p)`.
Equal-power weights are `(cos(pi*p/2), sin(pi*p/2))`, with exact endpoint
selection. Equal-power mixing preserves power for uncorrelated inputs; it
increases the level of identical signals at the center.

Both processors always apply their authored mix. There is no additional wet/dry
or bypass control, no automatic clipping and no normalization. The mixer uses
compensated accumulation to retain small terms during cancellation. Non-finite
gains, products and output fail explicitly. The native callback silences the
complete failed block and latches failure until restore/reset.

Neither processor adds algorithmic latency or its own audio tail. Upstream
latency compensation aligns every input, parameter action and end marker.
Ending one input supplies silence without reweighting the others. The processor
ends after all aligned inputs end; stop is immediate. JSON snapshots preserve
pending selector fades, gain ramps and upstream alignment state across NumPy
and native rendering. Native opaque snapshots also preserve this state and
reject mismatched input-port identities or crossfader laws.

The focused regressions cover four-input mixing, interrupted mute/solo fades,
gain ramps, independent input endings, both crossfader laws, shared generators,
upstream Waveshaper latency, cancellation, portable and opaque replay, malformed
snapshot rejection and callback failure recovery. Audio comparisons write
one-second 48 kHz WAVs.

The full ordinary suite passes 3,289 tests with 29 optional Rubber Band skips.
All 12 focused mixing tests pass. Ruff, formatting, type checking, pyupgrade,
Rust formatting and Clippy pass. The matching uFor schema revision `0ac37c7`
passed 1,521 tests.

The 54-test instrumented run, including existing live/input/matrix regressions,
records zero allocations, reallocations or frees in 2,757 successful callbacks
and 11 failed callbacks. Preparation, host admission, Python bridges and snapshot
conversion remain outside the native callback audit.

After restoring the ordinary build and finishing all tests, nine sequential
CPU measurements used two seconds of 48 kHz audio, 16 voices and 64 effect
actions per callback. The mixer has four stereo inputs; each crossfader has two.
Times are microseconds. Median/p99/maximum exclude the first block; misses
include it.

| Profile | Frames | Median | p99 | Maximum | First | Budget | Misses |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Mixer | 64 | 329.3 | 413.0 | 544.0 | 637.5 | 1,333.3 | 0 / 1,500 |
| Mixer | 256 | 887.6 | 1,004.9 | 1,090.1 | 1,286.2 | 5,333.3 | 0 / 375 |
| Mixer | 1,024 | 3,078.0 | 3,202.7 | 3,238.7 | 3,370.4 | 21,333.3 | 0 / 93 |
| Linear crossfader | 64 | 307.5 | 397.2 | 444.2 | 609.0 | 1,333.3 | 0 / 1,500 |
| Linear crossfader | 256 | 828.5 | 925.4 | 987.8 | 1,058.4 | 5,333.3 | 0 / 375 |
| Linear crossfader | 1,024 | 2,875.4 | 3,040.3 | 3,065.1 | 3,091.9 | 21,333.3 | 0 / 93 |
| Equal-power crossfader | 64 | 304.8 | 390.4 | 457.0 | 567.1 | 1,333.3 | 0 / 1,500 |
| Equal-power crossfader | 256 | 820.6 | 946.2 | 1,020.5 | 1,048.2 | 5,333.3 | 0 / 375 |
| Equal-power crossfader | 1,024 | 2,904.5 | 3,026.6 | 3,042.4 | 3,246.2 | 21,333.3 | 0 / 93 |

Reproduce with `scripts/benchmark_live.py --mixer`, `--crossfader linear` or
`--crossfader equal_power`, plus `--seconds '2 s' --voices 16 --effect-actions 64`
and the listed `--block-frames`. These finite CPU measurements do not qualify
an audio device or another platform.

## Additional work beyond the prompt

None.
