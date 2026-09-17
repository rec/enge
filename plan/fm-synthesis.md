# Two-operator FM synthesis

## Purpose and scope

Add FM as a third sound engine beside enge's oscillator synth and sampler, with
an independent NumPy reference and a Rust implementation selected through the
same Python API. Both backends run the same conformance tests. Discrete behavior
must agree exactly; audio and floating-point state use explicit tolerances.

The first profile has two sine operators, one modulation connection, independent
operator envelopes, and optional delayed self-feedback on the modulator. It must
handle live controls, release, arbitrary render partitions, and snapshot/restore
from its first implementation. It is preparation for larger FM instruments, not
an emulation of a particular hardware synth or preset format.

This document is a proposed implementation plan. Writing it does not implement
the engine or change uFor's current contract.

## Existing contracts and ownership

Follow the [engine execution contract](engine-execution.md), especially action
ordering, control evolution, output ownership, errors, and numerical boundaries.
Reuse [uFor's instrument contract](../../ufor/doc/instrument-format.md),
[modulation semantics](../../ufor/doc/modulation-format.md), and
[control evolution](../../ufor/doc/control-evolution.md).

uFor owns portable FM definitions, parameter units and domains, validation,
trigger contexts, and prepared lifecycle actions. enge owns phase, feedback,
envelope/filter realization, and audio snapshots. Hosts retain device I/O,
transport, raw event adaptation, output files, and encoding.

The current `ufor.synth.SynthVoice` and `ufor.synth_trace.VoiceStart` explicitly
carry an oscillator. FM therefore needs a deliberate uFor definition and prepared
start representation before enge integration; do not disguise it as a waveform
or create a second event-policy implementation. Reuse common lifecycle actions
and existing preparation logic where applicable, with only the extraction needed
to support the third engine.

## Initial musical model

Use a list of two operators with stable names and the same operator definition
type. Represent their connection separately. The first supported topology is
modulator into carrier, with only the carrier connected to the audio output.
Reject other operator counts or topologies explicitly in the initial profile.

Each operator has a positive frequency ratio to the prepared note pitch, a tuning
offset in cents, an initial phase in cycles, and its own held linear envelope.
The carrier has an output level; the connection has a modulation index in radians.
The modulator has an optional nonnegative feedback amount in radians. Define
finite parameter domains in uFor and use existing modulation domain handling.
Do not add a second modulator-level control that duplicates the connection index.

Reuse the existing envelope shape and exact release timing. Operator envelope
levels are bounded to [0, 1] in this profile. All operators receive the voice's
effective release, including minimum hold. Carrier-envelope completion ends the
voice and discards remaining modulator/filter state. A finished modulator
envelope contributes zero; it must not terminate an audible carrier. An immediate
prepared stop ends the whole voice. Duplicate releases follow existing semantics.

The mono carrier feeds the existing ordered filters, processing amplitude/gain,
and explicit channel routes. Its operator envelope is applied once, before those
filters. Do not accidentally apply the existing synth amplitude envelope a
second time. This ordering is part of the new FM contract.

## Numerical definition

The public name is FM, but this profile is explicitly phase modulation (PM).
Its index is a phase deviation in radians, not a frequency deviation in Hz.
These differ under changing index/envelopes, so implementations must follow the
equations rather than substitute a frequency-modulation recurrence.

For frame `n`, let `p_m` and `p_c` be phase accumulators in cycles, `e_m` and
`e_c` the operator envelope values, `I` the connection index, `B` the feedback
amount, and `L` the carrier level:

```text
m[n] = e_m[n] * sin(2*pi*p_m[n] + B[n]*m[n-1])
c[n] = L[n]*e_c[n] * sin(2*pi*p_c[n] + I[n]*m[n])
p_i[n+1] = wrap_cycles(p_i[n] + f_i[n]/sample_rate)
```

The stored feedback sample is the envelope-scaled modulator output `m[n-1]`,
initialized to zero at voice onset. It is updated after evaluating the current
modulator. This is an intentional one-output-sample delay, not a block delay or
an implicit equation. There is no carrier-to-modulator path in the first profile.

Compute each frequency from the prepared note pitch, the operator ratio, and
final common/operator tuning, applying each pitch contribution exactly once.
Specify this composition in uFor before implementation. Ratios remain positive;
fixed-Hz operators and through-zero frequency modulation are later work.

Evaluate the current sample before advancing phases. Reuse compensated phase
accumulation where appropriate, preserving its correction in snapshots. Keep
unmodulated phase state separate from the instantaneous PM offset. The first
profile starts at authored phases on each trigger; global phase synchronization
is outside this profile.

The initial numerical profile runs at the output sample rate with float64 state
and output. It permits aliasing and makes no alias-free claim. Preserve enough
headroom and never normalize or clip implicitly. Oversampling, its filtering,
latency, and feedback-delay meaning require a later explicit quality profile.

## Dynamic parameters and state

Support existing scoped control/LFO sources for common tuning and amplitude,
operator ratios/tuning, carrier level, connection index, modulator feedback, and
filter cutoff/Q. Add canonical structured uFor addresses for the new targets;
do not add a string-path setter or separate smoothing system.

Resolve values at each output frame after that frame's ordered actions. A pitch
or ratio change preserves accumulated phase. An index or feedback change changes
the PM offset immediately unless its control trajectory supplies smoothing;
discontinuities may therefore click. Block boundaries never add smoothing or
reset state. Topology, initial phases, and envelope definitions are latched at
onset; live topology editing is excluded.

Snapshots retain both phases and numerical corrections, the previous modulator
sample, operator envelope/release state, controls and scoped LFOs, filter state,
voice identity, cursor, prepared-definition identity, and backend. Restore must
reject incompatible definitions/backends and reproduce the continuation,
including silent instances and release tails.

## Implementation sequence

1. **Specify and test uFor's FM profile.** Settle portable definitions, parameter
   addresses/domains, envelope ownership, frequency composition, and prepared
   actions. Add validation and language-neutral lifecycle/control examples.
   Preserve the existing oscillator and sample formats. Publish the required
   uFor change before updating enge's dependency lock in a separate commit.
2. **Implement the NumPy numerical core.** Add a dedicated FM module with
   prepared voice/state models and explicit render inputs/outputs. Reuse envelope,
   phase, filter, and routing primitives without coupling the reference to Rust.
   Establish independent analytic and recurrence oracles before integration.
3. **Integrate the offline instrument.** Expose the existing behavioral surface:
   preparation, backend selection, `advance(actions, start, end)`, snapshot, and
   restore. Reuse control resolution and lifecycle infrastructure. Verify live
   changes, overlapping voices, release, and filters together.
4. **Implement the Rust kernel.** Use the existing PyO3/rust-numpy packaging and
   explicit `backend="native"` selection, with no fallback. Own native working
   buffers, release the GIL, and perform no Python callbacks during DSP. Move
   both operator recurrences and feedback into the kernel, not one binding call
   per frame. Run exactly the same cases against NumPy and Rust.
5. **Publish the audible regression and document the API.** Add a short musical
   demo, verified WAV/FLAC artifacts, README usage, and measured render/test costs.
   Update the execution roadmap to record the supported FM profile and limits.

Keep numerical inputs as parameter arrays plus explicit previous state. Follow
the existing `torch.compile` guidance: Python model validation, rational boundary
resolution, and action dispatch stay outside the numerical region. PyTorch is
future work; do not add it as a dependency or promise that a scalar feedback loop
will compile efficiently.

## Acceptance and verification

- Zero index and feedback reproduce an independently calculated sine carrier.
  Constant-frequency/index cases match the closed-form nested-sine equation;
  delayed-feedback cases match an independent scalar recurrence.
- Ratios, tuning, phase offsets, operator envelopes, and release produce the
  specified samples. Cover zero-duration segments, minimum hold, repeated
  release, early modulator completion, carrier completion, and immediate stop.
- Exercise abrupt and smoothed index/feedback/ratio changes, interrupted ramps,
  LFO modulation, same-frame action ordering, and multiple independent voices.
  Verify filter state and explicit output routes as well as raw FM generation.
- Whole renders agree with partitions of 64/128/256/1024/997 frames and splits
  immediately around changes and releases. JSON snapshot/restore agrees with
  uninterrupted continuation, including a nonzero feedback history.
- Use identical cases and declared absolute/relative tolerances for NumPy and
  Rust; compare meaningful internal state as well as audio. Derive tolerances
  from measured numerical error, not from a failed regression. Include long
  renders and demanding feedback to expose accumulated divergence. Native tests
  must prove they do not call Python DSP.
- Audio regressions write at least one second of 48 kHz WAV. The short demo
  should include a bell-like decay, a sustained index sweep, and feedback with
  release, using independent expectations. Verify lossless FLAC encoding and
  publish one listenable file per backend using the existing demo helpers.
- Run focused conformance tests, then the full parallel suite, Ruff, formatting,
  type checking, pyupgrade, Cargo formatting/clippy, and diff checks as required
  by repository instructions. Report existing unrelated failures separately.
  Measure short-block and longer-render costs without claiming callback safety.

## Path to four or more operators

Keep operator identity, equations, envelopes, and state independent of their
modulator/carrier role. The separate connection definition should allow a later
profile to admit additional operators and audible outputs without changing what
an existing two-operator patch means. Do not implement a generic graph scheduler
or a full algorithm library for this first release.

A larger profile must specify summing multiple modulation inputs, operator
evaluation order, multiple carriers, and explicit delays for permitted feedback
edges. It also needs expanded aliasing/performance tests, output-level policy,
and usability work. The two-operator engine proves the foundational recurrence
and dynamic behavior; it does not by itself establish production quality for
every larger topology or hardware compatibility.

## Additional work beyond the prompt

None.
