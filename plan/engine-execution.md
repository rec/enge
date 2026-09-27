# Remaining engine execution work

## Control integration

Carry named envelopes and addressed LFO rate/reset actions through every
offline and persistent runtime that supports their scopes. Add portable
modulation targets individually, with exact action ordering, partitions, and
snapshot continuation.

## Tensor implementation

Specify a supported recurrence subset and eager/compiled conformance before
adding PyTorch. Keep model validation, action dispatch, and rational boundaries
outside numerical kernels. Do not promise compiled feedback-loop performance
before measuring it.

## Live execution

Move remaining prepared-action orchestration into the native callback owner.
Native hosts retain device ownership. Expand live sampler modulation only with
explicit bounded-resource and callback contracts.

## Additional work beyond the prompt

None.
