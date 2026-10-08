# Remaining engine execution work

## Control integration

Carry named envelopes and addressed LFO rate/reset actions through every
offline and persistent runtime that supports their scopes. Add portable
modulation targets individually, with exact action ordering, partitions, and
snapshot continuation.

## Dynamic filters

The complete per-voice filter chain defaults to `before_amplitude` and can use
`after_amplitude`. Amplitude includes the envelope, prepared gain, and changing
amplitude controls. Pan/channel routing follows both stages. Sampler slot/group
filters precede instrument filters and active chains must agree on their order.

NumPy and Rust renderers carry this choice through preparation and snapshots.
FM keeps envelope-scaled operator signals for modulation and feedback while
placing the audible carrier envelope and level in the chosen amplitude stage.
Source exhaustion or envelope completion discards filter state without an extra
tail.

## Live execution

Move remaining prepared-action orchestration into the native callback owner.
Native hosts retain device ownership. Expand live sampler modulation only with
explicit bounded-resource and callback contracts.

## Additional work beyond the prompt

None.
