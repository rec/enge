# Sampler numerical contract

## Scope and status

This specifies the sampler's source traversal boundary. It extends the
[execution contract](engine-execution.md) and reuses
[uFor's playback and sustain-loop rules](../../ufor/doc/sample-performance.md#playback-direction).
The NumPy source renderer in [sampler.py](../enge/sampler.py) passes the
accompanying [vectors](../conformance/sampler-traversal.json). Complete NumPy
sample voices and prepared-action integration are implemented. Rust sampler
rendering is the next step.

uFor continues to own sample selection, effective slot settings, trigger IDs,
sustain, and prepared release/stop actions. Inputs here are already selected and
resolved. Linear interpolation is the first implemented reference profile. The
native-rate direction/loop examples follow the existing uFor rules.

`PreparedSample` validates already-decoded float64 audio and owns one immutable
copy, shared by every cursor rendering that prepared sample. It reuses uFor's
`Slice` and `Playback` definitions. `SampleState.start()` initializes a cursor;
`sample_frames()` accepts per-output-frame pitch ratios and an optional effective
release coordinate, and returns source-channel audio plus independent next state.
The input state and audio are unchanged. Serialize `SampleState` as JSON and
restore it with the same prepared sample; decoded assets stay outside snapshots.

`SampleVoiceRenderer` adds the same `PreparedEnvelope` and `EnvelopeRenderer`
timing used by oscillator voices, including fractional minimum hold, release
capture, and exact completion. It uses the common envelope and channel-routing
functions. Its serialized state contains no decoded audio; rendering and restore
use the same prepared sample. uFor currently has no minimum-hold field for sample
instruments, so the instrument adapter uses zero. The standalone voice API accepts
an explicit minimum hold, as the oscillator voice does.

`sample_instrument.prepare(score, decoded_assets)` validates decoded array shapes,
integer native/output rates, resolved settings, and the supported profile.
Immutable asset storage is reused across slots, voices, and instances. Preparation
fingerprints decoded content separately from the score's encoded-file hashes.
File decoding remains with the caller.

`OfflineSampler` consumes prepared uFor actions through the same `render_actions`
scheduler and `ControlRenderer` used by `OfflineSynth`. It combines instrument
and effective slot volume/tuning in dB/cents, multiplies amplitude routes, applies
resolved pitch/gain variation once, and uses whole slot/group envelope overrides
with instrument fallback. Snapshot restore checks the prepared score and decoded
fingerprints, restores independent voice/control state, and shares audio storage.

The first adapter supports held linear envelopes and control bindings targeting
amplitude or tuning. Filters/EQ, spatial processing, named generators, event
bindings, layer crossfades, fade retirement, latched parameter overrides, and
delayed/aligned/offset starts fail explicitly. They are not silently approximated.

## Coordinates and source ownership

Decoded audio is finite float64 `(native frames, source channels)` data. A
selected slice `[A, B)` addresses that data in native frames. Playback starts at
`A` for forward/mirror and `B - 1` for backward. The same traversal position and
weights apply to every channel. Reversing playback changes frame order only.
Decoding and validation happen during preparation, outside the render call.
Voices share immutable decoded arrays; each voice owns its cursor and traversal
state. No per-voice copying or pre-expansion of the complete sample is required.

At output frame `n`, read at the current position, then advance by
`pitch_ratio[n] * native_rate / output_rate`. Ratios are finite and positive;
direction is a separate resolved uFor setting. A changed ratio affects the next
position and never recalculates position from total elapsed time. The native-rate
factor is applied once, after uFor's pitch-ratio composition.

Accumulate `pitch_ratio[n] * native_rate` in output-rate units with compensated
addition, dividing by `output_rate` only to obtain interpolation progress. This
avoids committing a mirror turn early through repeated rounded division, such
as a release at native frame 3 after thirty 0.1-speed increments. Retain the
addition correction across calls and serialized restores, like the oscillator.

Output-frame scheduling remains integer and half-open. Fractional native-frame
progress belongs to the voice, persists across blocks, and is restored with it.
The reference and Rust must agree on discrete crossings, even when one output
step crosses several source frames, turns, or loop repetitions.

## Finite traversal

For source frames `A B C D`, the ordered native-frame knots are:

| Direction | Knots |
| --- | --- |
| forward | A B C D |
| backward | D C B A |
| mirror | A B C D C B A |

The turning frame is never duplicated. Without a loop, mirror is one outward
and return traversal, not perpetual ping-pong playback. A one-frame slice has
one knot in all three directions. Selected source boundaries exclude other
asset frames, including during interpolation.

## Linear interpolation profile

Let `V[k]` be a native-frame knot in the ordered traversal, after any loop
overlap blend. For native progress `x`, let `k = floor(x)` and `a = x - k`.
The output in each source channel is:

```text
y = V[k] + a * (V[k + 1] - V[k])
```

There is no hidden smoothing, normalization, clipping, or extra resampling
filter. This baseline is not bandlimited: linear interpolation changes the
high-frequency response and does not provide adequate alias rejection for
arbitrary pitch changes. A windowed-sinc profile would need its filter support,
ratio-dependent cutoff, edge extension, transition behavior, and numerical
tolerances specified separately. It could not silently replace this equation.
See Julius O. Smith's treatments of
[linear interpolation](https://www.dsprelated.com/freebooks/pasp/Linear_Interpolation_Frequency_Response.html)
and [windowed-sinc interpolation](https://www.dsprelated.com/freebooks/pasp/Windowed_Sinc_Interpolation.html).

For a finite traversal of `K` knots, coordinates `[0, K)` are active. In the
last cell `[K - 1, K)`, extend the last knot constantly: `V[K] = V[K - 1]` for
that interpolation only. At `x >= K`, produce exact zero and mark source
exhaustion before reading or advancing again. Do not create an implicit final
fade or read outside the selected slice. For `[10, 20]` at half speed the output
is `[10, 15, 20, 20, 0]`, with exhaustion at output frame 4. A single-frame slice
at half speed likewise produces its frame twice, then ends.

No-loop forward/backward traversals have `K = B - A`; mirror has
`K = 2 * (B - A) - 1`. The same final-cell rule applies to the finite remaining
path after leaving a loop. Constant unit speed reproduces uFor's exact frame
counts. For constant native step `d`, a finite unlooped traversal exhausts at
output frame `ceil(K / d)`. Under changing pitch, test the accumulated position,
not an onset-duration estimate.

At a wrap or reflection, interpolation follows adjacent traversal knots rather
than clamping each source read independently. In a `10, 20, 30` wrap, halfway
between the final `30` and next `10` is `20`. Mirror interpolates toward the
turning frame and then away from it without an extra held endpoint.

Interpolate already-blended overlap knots. Do not separately interpolate the
two source positions and a continuous crossfade weight and then multiply them:
that generally produces a different curve. For outgoing knots `10, 30, 50`,
incoming knots `0, 4, 0`, and `M = 3`, the overlap knots are `10, 17, 0`.
Halfway through the first pair the required value is `13.5`, not `15.5`.

## Loop topology

Use the slice's existing uFor `Loop`: `[start_frame, end_frame)` is in absolute
asset frames, is contained in the slice, and contains at least two frames.
Loops require effective playback mode `while_held`.

Forward starts at the slice start, enters the loop, and wraps toward increasing
frames. Backward starts at the slice end and wraps toward decreasing frames.
Mirror enters from the slice start and reflects at the first and last included
loop frames, without duplicating either turn. Source material outside the loop
is still available for an `until_release` exit.

For source `A B C D E F` and loop `B C D`:

| Direction | Unreleased knots |
| --- | --- |
| forward | A B C D B C D B C D ... |
| backward | F E D C B D C B D C ... |
| mirror | A B C D C B C D C B ... |

Release changes future topology. With `until_release`, cancel future wrapping
and reflection and continue in the current direction toward the slice boundary.
Release before loop entry continues toward the slice boundary in the current
direction without entering repetition, including for mirror playback. With
`through_release`, repetition continues until the amplitude envelope completes;
there is no subsequent tail after envelope completion.

## Event and boundary ordering

Apply all actions at an output frame in canonical `(tick, ordinal)` order before
observing that frame. Stop and envelope completion prevent any further source
read or source-state advancement at that frame.

A boundary exactly coincident with an action is pending until that action is
applied. In particular:

- At a forward wrap coordinate, release chooses the next ordinary source frame
  beyond the loop instead of the wrapped head.
- At a mirror turning frame, the direction is still the incoming direction for
  the release decision. Release at the upper turn continues toward the upper
  slice boundary; release at the lower turn continues toward the lower boundary.
- Release exactly at overlap entry prevents a new overlap from starting.
  Release after entry follows the already-active overlap rule below.

Crossings strictly before the new output-frame coordinate have already happened.
A large pitch step must resolve them in order. A release at the new frame cannot
undo a turn or wrap crossed strictly inside the preceding step.

A deferred minimum-hold release can occur at a rational output-frame coordinate
between observed samples. Merge that effective release with boundary crossings
in time order, using the preceding sample's pitch ratio throughout its interval.
Release wins an exact tie. Do not round the release to a whole frame and thereby
allow an extra loop or turn. The envelope captures its release level at that
same exact coordinate. The initial vectors below exercise integer releases;
fractional minimum-hold integration remains an explicit implementation test.

Interpolation lookahead does not commit a transition, consume a loop head,
change direction, or start an overlap. Future actions are never anticipated.
Rendering one block with a future release in its action list must equal rendering
up to that frame and delivering the release in the next block.

## Overlap knots

For a forward/backward loop of length `L`, use uFor's overlap length `M`, with
`M >= 2` and `2 * M < L`. Mirror requires `M = 0`.

The final `M` outgoing native-frame knots overlap the first `M` incoming knots.
At integer overlap index `j`, the knot value in every source channel is
`(1 - j / (M - 1)) * outgoing[j] + j / (M - 1) * incoming[j]`.
After knot `M - 1`, continue at incoming traversal index `M`. The consumed head
is not played twice. Subsequent repeat length is `L - M` native frames.

If `until_release` is delivered after overlap entry, finish the active overlap
and continue from the incoming head in the same direction, with no later wrap.
Do not abandon the incoming head, jump to a tail, or start a second overlap.
An envelope completing before that point still ends the voice immediately.

## Integration and acceptance

The full sample voice reuses the established envelope, minimum-hold,
control-scope, channel-routing, and snapshot contracts. Effective release must
reach the envelope and loop state at the same coordinate. It is distinct from
physical key release and from uFor's sustain decision.

Natural source exhaustion and release-envelope completion are separate end
conditions; the first ends rendering. One-shot voices ignore ordinary key
release according to prepared uFor actions, but still obey explicit stop/choke
retirement. Audio exhaustion never reruns selection or trigger ownership.

Live voice parameter evaluation stops at source exhaustion, including release
at a loop endpoint. A control ramp becoming invalid later cannot make a long
block fail after the voice has ended. Both backends resolve modulated voices
one output frame at a time to preserve that boundary; this API is not a
real-time performance claim. Unmodulated voices can render in spans.

Complete voice snapshots must retain fractional progress and its compensation,
direction, whether a coincident transition is pending, loop enablement, active
overlap progress, and the shared prepared-asset identity. The exact internal
layout is an implementation choice; no speculative interpolation state may leak
into the committed snapshot.

## Vector format

`conformance/sampler-traversal.json` is test data, not a new score format or
public preparation API. All sources declare their native rate and frame-major
channel arrays; the output rate is 48000. Each case chooses a source, half-open
slice, resolved direction, and optional canonical uFor loop settings.
`pitch_ratios` supplies one value per observed output frame. `release_at`, when
present, is an effective release immediately before that integer output frame;
the amplitude envelope is deliberately held open to expose traversal behavior.
Loop cases assume `while_held`; non-loop cases observe source exhaustion only.

`expected_samples` is the complete observed prefix. `exhaustion_frame` is the
first output-frame coordinate where the source is exhausted, or null if it has
not exhausted within that prefix. This distinguishes a zero-valued source frame
from completion. Source state must stop advancing after exhaustion. Fractional
cases describe the implemented linear profile.

Each vector now has NumPy and Rust regressions writing at least one second
of 48 kHz WAV output and comparing float arrays before encoding. Finite cases pad
with exact silence after exhaustion. Repeating cases stop source calls in the harness
after the observed prefix. The expected prefix is not repeated to fabricate a
long recording. Both backends must pass the same vectors and independent audio
oracles, with the existing `atol=1e-10`, `rtol=1e-9` budget.

The shared tests repeat cases with 64/128/256/1024-frame and irregular partitions,
including single-frame splits around release, wrap, reflection, and overlap
entry/exit. They restore during those states and during changing pitch, and
check fractional releases, decimal-speed boundary ties, and very large steps.
The fixture values test source traversal before envelopes or output routing;
Voice integration tests also apply envelopes, scoped controls, and channel routes,
and verify exact retirement, prepared sustain/replacement/stop decisions, asset
sharing, and independent JSON restores.

The native kernel retains an owned immutable audio buffer per prepared source
and runs without the GIL or Python callbacks. Python converts a rational release
into before/after source distances only after exact multiplication by that
interval's step. Rust retains compensated progress and pending transitions;
large steps skip complete cycles without narrowing their counts to machine
integers. Frame and native-knot coordinates use signed 64-bit integers at the
binding, with checked frame advancement. Voice and instrument snapshots retain
their backend. Direct native regressions disable reference DSP and verify
buffer ownership and integer timing beyond `2**53`.

## Additional work beyond the prompt

None.
