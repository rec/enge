# LFO realization

## Supported profile

enge renders uFor's seconds-clock LFOs for both synths and samplers, selecting
NumPy or Rust with the instrument's existing `backend` argument. Sine, square,
and triangle shapes retain the canonical duty-cycle equations, including duty
zero and one. LFO values are bipolar. Delay and fade-in affect a separate
activation weight, never the phase clock.

`enge.lfo.lfo_samples(definition, state, start, frames, sample_rate, backend)`
returns a float64 `(frames, 2)` array containing value and weight. `state` is the
canonical `ufor.lfo.LFOState`. Frames address the same absolute seconds clock as
its rational event anchors. Rendering observes state without re-anchoring it.
Use `ufor.lfo.initial_lfo` and `ufor.lfo.lfo_event` to initialize and apply events;
serialize the canonical state to restore a standalone source.

Beat clocks require a host tempo mapping and remain explicitly unsupported in
this audio profile. Named envelopes and other modulation targets remain outside
this milestone.

## Timing and numerical boundaries

At time `t`, phase is exactly `(state.phase + state.rate * (t-state.at)) mod 1`.
A rate event advances using the previous rate before installing the next one.
It preserves phase and activation age. Zero rate freezes phase. Rate changes are
piecewise constant; continuous rate ramps are not part of uFor's current LFO
contract. An event affects observations at its time and later. For a fractional
event time, render the earlier integer observations with the previous state,
apply the event, then render from the first integer observation at or after it.

Reset policies follow uFor: `free` ignores trigger/transport events, `trigger`
responds to trigger, and `transport` responds to explicit transport events. An
explicit reset always resets phase and activation age, retaining current rate.
A render boundary is never a reset or transport event.

The NumPy reference samples uFor's independent scalar equations. Rust evaluates
waveforms and activation arithmetic in a detached numerical loop with owned
input and output arrays. Python resolves rational phase, wrap/duty boundaries,
and activation boundaries before crossing the binding. Square polarity is
resolved before converting to float. Triangle spans use a normalized coordinate
within the selected branch, avoiding cancellation when duty approaches zero or
one. Complete unobserved cycles are skipped using exact rational reduction.
The numerical kernel receives arrays, without Pydantic or Fraction operations.

Phase continues during delay and fade-in. At delay's endpoint weight is zero
when a fade exists, otherwise one. During fade it increases linearly to one.
Route evaluation applies weight after mapping: addition fades from zero;
multiplication fades from one. Rust bounds final observations to their declared
bipolar/unit domains to contain floating-point roundoff at endpoints.

## Instrument ownership and event boundary

LFO names are local to a sound-setting definition. Repeated bindings to one
named source observe the same state. Voice scope creates a fresh source at the
voice's start frame. Instrument and part sources use frame zero as their clock
origin and continue algebraically through silent intervals. A joining voice does
not send a trigger event to an existing shared source. Part sources are separate
for each part; instrument instances never share state. Sampler slot/group
ownership follows uFor validation; shared sampler LFOs belong in instrument
settings. Instrument and slot modulation still compose per voice before mixing.

Both instrument snapshots include LFO anchors and their owner identities,
alongside existing control ramps and voice-to-source references. A new onset
using an old trigger ID cannot reassign a release tail's voice LFO.

Prepared uFor instrument traces currently contain lifecycle and control actions,
not addressed LFO rate/reset events. Instrument integration therefore realizes
authored LFO settings and voice/shared ownership. Dynamic rate, trigger,
transport, and explicit reset events are supported through the standalone
canonical state API. Exposing those events through prepared instrument traces
requires a later uFor contract extension; this implementation adds no private
trace vocabulary and does not reinterpret control observations as LFO events.

## Acceptance

Shared tests compare float arrays before writing at least one second of 48 kHz
WAV audio, using `atol=1e-10`, `rtol=1e-9`. Coverage includes shapes/duty endpoints,
fractional delay/fade boundaries, irregular partitions, rational rate events,
zero rate, all reset policies, exact duty ties, large frame coordinates, scoped
silent advancement, release tails, reused trigger IDs, and JSON restores.
Native checks disable scalar reference evaluation and reject invalid span layouts.

The two-second listening regression combines vibrato, tremolo, and a live gain
ramp: a sine synth on the left and a sampled harmonic tone on the right. Expected
pitch is integrated independently at higher precision; the sampler oracle uses
linear interpolation of the supplied asset. Both backends publish verified
lossless FLAC through reccy's existing atomic-output helper. This establishes
numerical behavior, not live callback performance.

## Additional work beyond the prompt

None.
