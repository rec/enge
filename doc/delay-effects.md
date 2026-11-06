# Delay, chorus and flanger profile

`TapDelay` and `ModulatedDelay` use bounded circular histories and linear
interpolation. They retain the existing gain/mix/feedback controls, shared
enveLFO routes, bypass ramps, input-end handling and snapshots.

## Moving reads

`ModulatedDelay` moves its read position continuously. Its base/depth/rate and
channel phase offsets retain their meanings; a shared enveLFO can drive the
base delay with internal depth zero. This produces the pitch motion used by
chorus, flanger and vibrato without integer-sample stepping.

For delay `d = k + f` output frames, the read is `(1-f) * x[n-k] +
f * x[n-k-1]`, with quiet history before the input starts. Effective delays
have a one-frame minimum. History capacity is the ceiling of the prepared
maximum delay in frames, so the older interpolation sample is retained even
when the maximum delay is fractional. There is no hidden dry-path delay.

## Fixed-tap retiming

`TapDelay.retime_fade_frames` is a positive, prepared frame count, default 64.
A new effective delay starts a crossfade from the current read-head mixture.
The current destination begins fading out from its current gain; heads already
retiring keep their existing deadlines. The new destination receives the
remaining weight, so the weights stay continuous and sum to one.

There is at most one new effective tap per output frame, after all same-frame
actions have been applied. Automation ramps and enveLFO routes can therefore
retime on every frame. Each old head retires within the prepared fade duration,
so `retime_fade_frames + 1` slots suffice. Slots are reused; the Rust callback
does not allocate or grow a head collection. The fade duration is a preparation
setting rather than another automated parameter.

The weighted wet read enters the existing feedback recurrence. Feedback remains
in `[0, 1)`. New stored values below `state_floor` are zeroed in both renderers.
Input-end retirement tests the complete history against the declared tail
threshold. The native core maintains a count of stored values at or above that
threshold, updating it as each history cell is overwritten; callbacks do not
rescan the entire delay buffer. Native live input-end and stop actions now use
the same drain rule and survive snapshot replay.

## State and verification

Python JSON snapshots retain history, write position, all head gains/delays,
remaining fades and destination identity. Native snapshots retain the same
execution state and validate prepared capacity/fade/floor identity on restore.
Modulated-delay snapshots retain history, phase and channel offsets.

`test/test_delay_interpolation.py` supplies independent interpolation and
overlapping-crossfade audio oracles, fractional maximum-history cases,
samplewise retiming beyond one pool lifetime and interrupted JSON/native replay.
The audio comparisons write paired 48 kHz WAV files of at least one second.
Full-suite and allocation-audit results are recorded in the implementation
inventory when each verification gate finishes.

This profile passed 3,156 tests with 21 skipped; its targeted audit passed 55
tests with zero Rust heap operations in 2,510 successful and seven failed
callbacks. See the [source-to-output measurements](delay-benchmark.md) for peak
read-head occupancy, timings, observed deadline misses and audition artifacts.

## Additional work beyond the prompt

None.
