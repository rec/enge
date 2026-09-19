# Synth and sampler execution contract

## Status and purpose

This is the proposed execution contract for enge's Python/NumPy reference and
one Rust implementation with PyO3 and rust-numpy bindings. Rust is the chosen
native language. This document records both the implemented synth profile and
requirements for future sampler and processing extensions.

Both implementations must realize the same prepared uFor definitions and actions,
using one Python-facing API and one conformance suite. Discrete behavior must
agree exactly; floating-point results agree within declared numerical tolerances.
Bitwise floating-point identity is not a requirement.

Synth and sampler share scheduling, control evolution, lifecycle realization,
routing, and snapshot rules within each implementation. Their source-generation
state differs. The Python reference remains independent of native DSP code and
prioritizes readable equations and transitions over speed.

The current implementation in `src/enge/synth.py` provides the first dynamic
synth reference: held linear envelopes, explicit routes, minimum hold, phase
synchronization, static tuning, and live amplitude/tuning control routes. It
consumes uFor trigger contexts, retains scoped smoothing trajectories, and
restores them with voice state. Named seconds-clock LFOs now drive the same
amplitude/tuning routes. Both backends now implement dynamic resonant filters;
named envelopes and other targets remain unimplemented. The Rust backend realizes
the same numerical synth profile through explicit `backend="native"` selection;
`"numpy"` remains the default. PyTorch compilation remains future work.
Existing waveform start/length/period behavior remains a regression requirement
during synth consolidation.

The third source profile is now the NumPy and Rust two-operator FM engine in
`src/enge/fm.py`, under the [FM plan and status](fm-synthesis.md). It shares uFor
synth lifecycle preparation, scoped controls/LFOs, envelope arithmetic, filters,
and routing. Its pure array kernel keeps phase and one-sample feedback state
explicit for a future tensor port. The Rust FM kernel computes both operators,
envelopes, feedback, filters, gain, and routing with owned buffers and the GIL
released. PyTorch remains deferred.

The fourth source profile is white noise in `src/enge/noise.py` and `src/noise.rs`,
under the [noise plan](noise.md). Both backends use the portable noise-v1
SplitMix64 stream contract, with per-voice keys carried by prepared actions and
an exact sample counter. Noise shares filters, amplitude envelopes, controls,
LFOs, routing, and lifecycle semantics; source pitch/tuning is unsupported.
The filter precedes the amplitude envelope, as in the oscillator synth. Snapshots
preserve the stream and reject backend mismatches, even for silent engines.

`src/enge/sampler.py` implements NumPy and Rust source traversal with linear
interpolation, shared immutable decoded audio, live pitch arrays, fractional
effective releases, and serializable cursor state. `SampleVoiceRenderer` and
`src/enge/sample_instrument.py` now integrate shared envelope timing, scoped
controls, routing, and prepared uFor actions. Both instrument renderers now share
scoped LFO sources under the [LFO numerical contract](lfo-numerics.md).

## Ownership and existing contracts

uFor owns portable definitions, parameter units/domains, source scope, performance
preparation, selection, trigger ownership, sustain, and retirement decisions.
enge consumes the resulting actions and owns audio realization and its state.
It must not infer sample selection or pedal policy again from raw events.

Sample decoding and immutable asset preparation can belong to enge preparation.
Asset discovery, session management, input adaptation, device clocks, MIDI/OSC,
GUI, output files, encoding, and plugin integration remain outside the engine.
No decoding, file access, or Python callbacks belong inside a native render call.

The existing specifications to reuse are:

- [Instrument definitions and routing](../../ufor/doc/instrument-format.md).
- [Sample performance and traversal](../../ufor/doc/sample-performance.md).
- [Oscillator and synth definitions](../../ufor/doc/musical-format.md).
- [Envelope, LFO, and modulation semantics](../../ufor/doc/modulation-format.md).
- [Scalar timeline automation](../../ufor/doc/automation-format.md).
- [Shared lifecycle actions](../../ufor/ufor/instrument_trace.py).

The [corrected enge handover](../../recs/plan/enge.md) governs the project boundary.
The older [Recs offline contract](../../recs/plan/sample-playback.md) also describes
a combined performance preparer and renderer, and planar output. This proposal
uses the corrected prepared-action boundary and preserves enge's existing
`(frames, channels)` output layout. It does not introduce a second raw-event API.

Control smoothing, trigger-context initialization, and pitch composition are now
specified in [uFor control evolution](../../ufor/doc/control-evolution.md), with
language-neutral cases in uFor commit `2e5b02e`. Reuse those definitions rather
than maintaining an enge-specific version of their musical meaning.

## Designing numerical functions for torch.compile

A future PyTorch realization should run the same numerical function eagerly and
through `torch.compile` against the shared conformance cases. PyTorch is not a
dependency of this first NumPy implementation, and this guidance does not choose
the native backend or replace the independent reference requirement.

Use an explicit numerical boundary:

```text
render(parameter arrays, previous numerical state) -> audio, next numerical state
```

Keep changing pitch, gain, phase, ramp, and DSP values in tensor inputs rather
than Python scalar arguments that can cause specialization and recompilation.
Keep static configuration separate. Treat shapes, dtypes, devices, strides, and
voice counts as deliberate compilation choices; do not assume arbitrary changing
block sizes will reuse one compiled graph. Preserve arbitrary-partition semantics
even if an implementation uses several compiled shape variants.

Use tensor operations for numerical computation. Avoid extracting tensor values
into Python with `.item()` or branching in Python on those values. Use suitable
tensor selection or supported structured control flow; selection must not evaluate
invalid arithmetic in an unselected branch. Tensorization must preserve the
specified recurrence and state transitions, including phase and loop boundaries.

Keep Pydantic validation, rational-clock resolution, context lookup, event
interpretation, asset loading, and logging outside the compiled numerical region.
Pass state explicitly and return its replacement. This is a design preference for
clear ownership and testing, not a claim that PyTorch forbids all mutation.

During development, compile numerical kernels with `fullgraph=True` to expose
graph breaks, inspect `TORCH_LOGS="graph_breaks,recompiles"`, and compare eager and
compiled results. Benchmark steady-state calls after warm-up on the intended CPU
or accelerator, separately from compilation cost. Successful compilation does
not demonstrate a speedup or suitability for live audio deadlines.

Oscillators, gain trajectories, and mixing are early candidates. Recurrent filters,
changing voice populations, and small 64/128-frame calls need early measurement.
Avoid a large unrolled Python sample loop in the compiled path; choose a supported
recurrence formulation only after verifying its numerical behavior and performance.
The NumPy reference may retain a clear scalar recurrence where it establishes the
required behavior. No compiled performance claim is made for that reference.

Sources: [graph breaks](https://docs.pytorch.org/docs/2.14/user_guide/torch_compiler/compile/programming_model.common_graph_breaks.html),
[recompilation](https://docs.pytorch.org/docs/2.14/user_guide/torch_compiler/compile/programming_model.recompilation.html),
and [torch.compile](https://docs.pytorch.org/docs/stable/generated/torch.compile.html).

## Common API behavior

Keep the existing operations as the common behavioral surface; concrete sample
types and native bindings can be named when their profiles are implemented.

| Operation | Required behavior |
| --- | --- |
| `prepare(...)` | Accept a validated instrument and the immutable assets needed by its declared profile. Resolve the output rate, channel order, supported source/processing settings, and asset bounds. Return a prepared definition or an explicit unsupported/invalid-input error. |
| New instance | Begin at output frame zero with no voices and controls at declared defaults. Share immutable prepared data only; all mutable state belongs to this instance. Random selection/variation has already been resolved by uFor. |
| `advance(actions, start, end)` | Require `start` to equal the instance cursor and `end > start`. Render exactly `[start, end)` and move the cursor to `end`. Supply only actions in this interval. Return audio, not another preparation trace. |
| `snapshot()` | Capture state at the current cursor without advancing it. The state describes the next sample to render, before any actions at that cursor. |
| `restore(snapshot)` | Restore into an instance with the same prepared definition, assets, output rate, and channel order. Replaying the same future actions must reproduce the continuation. |

The output is a C-contiguous float64 array of shape `(end - start, channels)`.
The first profile uses float64 for DSP state and accumulation too. There is no
implicit clipping, normalization, integer encoding, or channel conversion.
Values outside [-1, 1] are permitted. Output ownership must keep returned buffers
valid and unchanged after later engine calls; they must not alias mutable engine
scratch storage.

Known unsupported settings must fail during preparation. Unknown action kinds,
unresolved required targets, and unsupported runtime operations must fail
explicitly rather than being silently ignored. Errors identify the feature or
target and, for runtime failures, the action or frame. Matching error categories
and relevant context is required across backends; identical prose is not.
After a failed render, continuation requires restoring a valid snapshot or
creating a fresh instance; transactional rollback is not required.

## Frame timing and ordering

The execution clock is integer output frames at a positive integer rate `R`.
The caller resolves native uFor event ticks onto that clock before rendering.
enge performs no implicit rounding of non-frame event coordinates. A host must
declare any quantization used when adapting another clock.

At frame `n`, apply the actions for that frame before rendering sample `n`.
Order actions by `(tick, ordinal)`, preserving trace-list order for records
with equal coordinates. A single prepared performance event can produce several
actions; do not invent an action-type priority or reorder its fan-out.
An action at `end` belongs to the next call, never the current call.

Advance continuous state to the frame coordinate before applying its actions.
Consume envelope segment endpoints and zero-duration segments according to
uFor's existing rules. Apply ordered actions, resolve parameter values, generate
the sample, and update source/DSP state for the next frame. A sample-loop boundary
transition at the frame is resolved after its release actions, as required by
uFor's release-at-loop-boundary rule.

Block boundaries have no musical meaning. They do not reset phase, restart a
ramp, quantize a release, update a parameter early, or change a filter cadence.
Controls delivered in later calls can change the future but cannot revise
already rendered audio. Late actions are errors, not instructions to rewind.
The engine needs only the current interval's actions, not a complete performance.

## Parameters over time

Every supported live numeric parameter has a defined value at each output sample.
Constant spans and ramps may use compact representations or vectorized evaluation,
but the result must match the same sample-by-sample semantics. Evaluating one
value per caller-supplied block is not conforming.

Use uFor's existing control IDs, scoped source bindings, structured parameter
addresses, and units. Do not add a parallel string-path setter API. The first
dynamic profile uses addressed control actions routed to amplitude and tuning
in cents. Standalone automation and generated modulation must use the same final
parameter evaluation when their bindings are supported; they are not implicitly
connected merely because uFor can describe them.

For each frame and resolved voice context:

1. Observe the relevant control trajectories and envelope/LFO states.
2. Resolve the base value, including a supported explicit replacement automation.
3. Apply uFor source mappings, activation weights, and route arithmetic in its
   specified stable order.
4. Validate the resulting domain and apply only explicitly declared boundary
   policies, such as a filter's clamp policy.
5. Use the result for that frame's gain, phase increment, traversal, or processing.

Do not smooth the final sum again or introduce a hidden conversion between Hz,
cents, dB, and ratios. Frequency automation declared in Hz is linear in Hz;
a linear cents trajectory becomes an exponential frequency trajectory. A source
explicitly configured for immediate changes remains immediate.

### Control smoothing contract

Interpret a control binding's existing `smoothing` duration as a finite linear
transition in the source's declared numeric domain, before route mapping. This
is the portable rule implemented by uFor's scalar control reference.

For a target `b` received at frame `k`, let `a` be the trajectory's value at `k`
before that action. For duration `S` seconds and exact rational `D = S * R`:

```text
S = 0:  value(x) = b for x >= k
S > 0:  value(x) = a + (b - a) * min((x - k) / D, 1) for x >= k
```

Observe this trajectory at integer sample coordinates. Keep `D` exact for
endpoint comparisons; do not round each duration to whole frames. The target
is observed at the first sample at or after the endpoint. A second action
captures the first trajectory's current value and starts a new full-duration
transition there. Same-frame actions execute in their trace order. A positive
duration does not jump at entry; a zero-duration action does.

Contexts begin at the declared default or an explicit initial control value,
without an artificial ramp from zero. Instrument and part controls retain their
history when no voices are sounding. Voices joining a shared context observe its
current trajectory, rather than restarting it. Trigger contexts remain independent
and survive physical release while their voices still need them. Scopes do not
implicitly override one another: bindings select the addressed context.

Bindings with different smoothing durations can share the raw control history
but require distinct trajectories. Bindings sharing context, control, and
smoothing behavior observe the same trajectory. A voice-local modulation source
is still distinct from another voice's source, even when both use one template.

### What may change in the first dynamic profile

| Setting | Treatment |
| --- | --- |
| Amplitude and tuning cents through declared control routes | Live per-sample values, including during release. |
| Logical gate and retirement | Exact prepared actions; never smoothed with expression controls. Sustain decisions remain in uFor. |
| Envelope durations derived from key/velocity | Latched at onset, following the existing uFor contract. |
| Envelope structure, waveform shape/duty, sample identity, slice/loop bounds, direction, output layout, routing topology | Fixed for the prepared voice definition; no active-voice mutation API in this profile. |
| LFOs | Seconds-clock value/weight realization is implemented; addressed instrument rate/reset actions remain a future uFor extension. |
| Filter cutoff and Q | Live per-sample control/LFO values, with retained trapezoidal integrator state. Response, stage count, and list order stay fixed. |
| Richer automation and structural transitions | Separate declared extensions with their own dynamic conformance cases before support is enabled. |

This is an initial support boundary, not a claim that structural edits can never
be supported. Such edits need explicit state transfer or crossfade semantics.
Rebuilding a voice silently when a control changes is not an acceptable substitute.

## Persistent source and processing state

### Synth

Let `p[n]` be the phase used to produce sample `n`, and `f[n]` its resolved Hz:

```text
source[n] = waveform(p[n])
p[n+1] = (p[n] + f[n] / R) modulo 1
```

Frequency changes preserve `p[n]`. Never derive a running voice's phase as
`elapsed * current_frequency`. For the first dynamic profile, resolved onset
frequency is the base and tuning modulation contributes cents according to uFor's
parameter-base rules; the preparer must make clear which static adjustments are
already included so no offset or tuning is applied twice.

Preserve the current phase convention: ordinary voices start at phase zero;
transport-synchronized starts initialize phase from the absolute start frame
and resolved onset frequency. Later modulation advances that phase; it does not
retroactively change the start or continuously relock the voice. Shared evolving
oscillators and other synchronization behavior need an explicit extension.

The recurrence defines discrete audio phase integration, including during ramps.
It does not replace uFor's separately specified scalar LFO clock mathematics.
The NumPy reference and Tuney share `OscillatorState` and `oscillator_samples()`.
They accumulate frequency in output-rate units with compensated addition,
observe phase by dividing by R, and carry both accumulator and rounding
correction across blocks and snapshots. This avoids the one-sample
square-wave boundary errors caused by repeatedly adding rounded `frequency / R`.
These are numerical state details, not an alternative musical phase convention.
Square and triangle retain their existing endpoint rules; band-limiting is a
future declared rendering profile, not an optimization allowed to change output.

Tuney preserves its existing synchronized-onset adapter: it wraps the transport
frame at the base note's period, then uses that sample-position origin for each
oscillator, including both binaural frequencies. Its running oscillators now use
the common state recurrence. This preserves existing onset behavior rather than
silently changing binaural synchronization to enge's absolute-frame convention.

### Sampler

A voice owns a position along its selected traversal, loop/direction state, and
any active loop overlap. At frame `n`, read the current position, then advance by
the resolved positive pitch ratio times `native_rate / R`. Pitch changes affect
future progress without resetting position. Direction determines how progress
maps to source frames; it is not inferred from a negative pitch ratio.

Reuse uFor's forward/backward/mirror endpoint, loop, overlap, and release rules.
Decoded immutable arrays are shared across voices and restored instances;
traversal, envelopes, and processing state are private. Stereo channels share
the voice's traversal position and weights without collapsing their audio.

The [sampler numerical contract](sampler-numerics.md) specifies traversal and
release ordering, with exact small [vectors](../conformance/sampler-traversal.json).
The NumPy source core implements its linear interpolation kernel, finite-sample
edge treatment, fractional loop/reversal behavior, and exhaustion coordinate.
All vectors now pass 48 kHz WAV regressions, including single-frame serialized
restores. Longer tests exercise rate conversion and changing pitch across block
partitions. Native and Python backends may not select different resamplers and
call the difference a numerical tolerance.

### Release and processing

Apply prepared stop immediately before the addressed sample. Apply prepared
release through the declared envelope and minimum-hold behavior; capture the
envelope level at the effective release coordinate. A duplicate release does
not restart the tail. Gain/pitch controls continue evolving through release.
Ending audible output must not make enge rerun uFor's trigger ownership policy.

Natural completion and release completion must retire audio state at their
defined coordinates, not merely at the end of whichever block contains them.
The sample profile must specify whether exhaustion or envelope completion ends
each traversal, using uFor's playback-mode rules. No amplitude threshold may
silently terminate a voice or discard a nonzero authored tail.

### Dynamic filters

Implemented under the revised [uFor filter contract](../../ufor/doc/instrument-format.md#resonant-filters),
uFor commit `1239e45`. The static RBJ transfer functions remain the reference for
frequency response. The former direct-form II transposed recurrence is replaced
by the specified trapezoidal state-variable recurrence for both constant and
changing parameters. There is one realization per backend, with no legacy path.

`src/enge/filters.py` supplies the NumPy reference and shared parameter validation;
`src/filters.rs` computes coefficients and advances filter state inside the native
synth/sample kernels. Each voice has two integrators per stage per source channel.
Values have shape `(active frames, filters, 2)`, with cutoff in Hz then Q. Resolve
modulation before checking the filter's authored clamp/error bounds. At a sample,
install its values and retain both integrators without extra smoothing or reset.
The kernels reject non-finite output/state; they do not clip audio.

Filters run in list order before amplitude, gain, and routing. Sample effective
slot/group filters precede instrument filters, whose local parameter namespaces
remain distinct even when names match. All states are voice-owned and captured
in snapshots. Release keeps the existing lifetime; source exhaustion, envelope
completion, or stop discards filter state. Modulated samplers batch a prefix
whose declared maximum tuning cannot exhaust the source, then resolve the
remaining uncertain frame alone so parameters after exhaustion cannot cause
errors.

Acceptance tests compare all four responses and one/two stages against uFor's
independent RBJ transfer functions. Dynamic cases use a coupled matrix solve,
alternate cutoff between 1 kHz and 20 kHz, change Q per sample, check zero-input
state decay, cover 64/128/256/1024/997-frame partitions and JSON restoration, and
exercise independent voices/channels, control ramps, LFOs, and sample exhaustion.
The two-second FLAC demo has independent audio expectations and lossless encoding
verification. These are numerical conformance checks, not live deadline or
alias-free modulation guarantees. Named envelopes remain the next discussion.

## Snapshots and restoration

An audio snapshot must preserve all mutable information needed for continuation:

- Cursor, active voice identities, and pending effective release coordinates.
- Oscillator phase or sample traversal position, direction, loop overlap, and
  delayed-start state where applicable.
- Envelope traversal, captured entry/release values, and generator state.
- Control contexts, current ramp anchors/targets/endpoints, and applicable
  modulation state, including histories still needed by future joining voices.
- Independent filter histories and other processing state for supported profiles.

For the fixed linear-envelope profile, the release level is reconstructed from
the immutable envelope definition and exact effective release coordinate. It
does not need a second stored value. `envelope_samples()` performs the same
calculation for the offline synth and Tuney's fade-setting adapter.

A snapshot must not alias mutable live state. Immutable definitions and decoded
assets can be referenced and shared, not copied per voice or snapshot. Prepared
data identity and output configuration must be checked on restoration.
The caller owns undelivered future actions; a snapshot is not a copy of that
future stream. Retaining a pending scheduled transition already accepted by the
engine is nevertheless part of its state.

uFor's semantic snapshots and enge's audio snapshots serve different purposes.
Seeking a combined performance requires appropriate preparation state as well
as audio state. enge cannot reconstruct missing sustain, selection, or control
history from a list of active voices.

Within-backend restoration is required. Cross-backend snapshot exchange and a
durable on-disk snapshot format are not requirements of the first profile.
Shared tests compare logical state and restored behavior without requiring
identical private layouts or exposing a second mutable control interface.

## Numerical and behavioral conformance

Use one parameterized suite against the reference and native API. Also compare
the implementations directly on identical prepared inputs. Retain independent
expected values and uFor vectors so agreement cannot hide a shared mistake.
The native backend must execute native rendering, not delegate DSP to Python.

| Value | Comparison |
| --- | --- |
| Frame coordinates, action order, identities, routing destinations, lifecycle status, array shapes | Exact. |
| Rational durations and discrete endpoint decisions | Exact, regardless of floating representation used for arithmetic. |
| Floating controls and audio/state values | Explicit finite absolute/relative tolerances, declared before validating a backend. No bitwise requirement. |
| Deliberately silent/unrouted output and stopped voices | Exact zero. |

Initial float64 audio comparisons use elementwise
`abs(actual - expected) <= 1e-10 + 1e-9 * abs(expected)`.
Scalar uFor conformance retains its existing tolerances, including `1e-12` where
specified. Check finite output explicitly. Do not use relative error alone near
zero or an average error metric that could conceal a click. Report peak absolute
error and its frame/channel for failed audio comparisons.

Phase comparisons account for wraparound, but waveform discontinuities and
sample-loop boundary choices must still occur on the same samples. Numerical
tolerance does not excuse a one-frame branch error. If a future algorithm or
long-duration case needs a different numerical budget, document its error source
and agree that case's bound before implementation acceptance; do not relax a
test merely to make a backend pass.

Every audio regression writes at least one second of 48 kHz WAV output through
the test harness. Compare float arrays before WAV encoding so quantization cannot
hide errors. Short exact scalar/traversal vectors supplement those audio cases.
WAV writing is a test responsibility, not an engine output feature.

Render each continuity case uninterrupted and in 64, 128, 256, and 1024-frame
partitions, plus irregular partitions and boundaries immediately before, at,
and after changes. Include one-frame calls around critical boundaries. Restore
inside ramps, during release, and near source discontinuities and loop edges;
compare the continuation and logical ending state. Both fixed and changing
parameters need long-duration phase/traversal drift coverage as well.

### First dynamic-control acceptance cases

1. At 48 kHz, start a 100 Hz sine at phase zero and step to 200 Hz at frame 120
   through a declared tuning route. Phase at frame 120 is 1/4 cycle, sample 120
   is 1 before gain, and phase at frame 121 is `1/4 + 1/240`. Repeat through
   release and after snapshot restoration.
2. Route a control directly to amplitude. Start at zero, target one at frame zero
   with smoothing `1/200` second, then target zero at frame 120. Values at frames
   0, 120, 240, and 360 are respectively 0, 1/2, 1/4, and 0. Include repeated
   same-frame targets and zero smoothing as separate cases.
3. Deliver a part control while silent, then start two overlapping voices at
   different points in its ramp. They must observe the same continuing control
   trajectory. Trigger-scoped controls must remain isolated from the other voice.
4. Exercise gain and pitch changes at the same frame as starts, releases, stops,
   and minimum-hold endpoints, in both meaningful ordinal orders. Pair synth and
   sampler cases with identical lifecycle expectations when the sampler exists.
5. Cover fractional smoothing endpoints, large absolute frame coordinates,
   interrupted ramps, zero-duration envelope stages, nonzero terminal levels,
   and square/triangle phase transitions under pitch changes.
6. Require matching explicit rejection of unsupported settings and operations;
   a backend cannot pass by ignoring a parameter it does not implement.

Sampler acceptance adds forward/backward/mirror vectors, loop entry/exit and
overlap, fractional pitch, stereo routing, release tails, and decoded-asset reuse
without shared mutable voice state. Backend-specific allocation tests may verify
asset reuse separately from the common behavioral suite.

## uFor prerequisites and implementation order

The portable prerequisites below were completed in uFor commit `2e5b02e`:

1. Specify the smoothing equation, interruption, initial values, scope lifetime,
   and behavior of voices joining an already evolving control context. Add scalar
   language-neutral vectors for the examples above.
2. Ensure prepared inputs preserve trigger-initial controls and relevant scoped
   control history. An onset's evaluated parameters alone cannot initialize all
   future modulation correctly. Reuse the canonical uFor representation rather
   than inventing an enge-only substitute.
3. Confirm that resolved onset pitch, static tuning, and routed tuning have one
   unambiguous composition rule, with no double application across preparation
   and realization.
4. Use the corrected lifecycle contract. uFor snapshots are explicitly
   observational, not resumable preparation checkpoints; audio state belongs to
   enge. Recheck contracts after future uFor changes rather than preserving defects
   as compatibility behavior.

Implementation status and remaining order:

1. The first dynamic synth reference and its timing/control conformance are
   implemented. Tests cover 64/128/256/1024-frame and irregular partitions,
   restores inside active ramps and release, trigger-ID reuse with an old tail,
   multiple smoothing durations, and exact fractional release boundaries. The
   reference uses uFor scalar control/route evaluation before a numerical
   oscillator function with explicit phase input/output. Ten-second integer-oracle
   regressions cover changing and held pitch, all three waveforms, starts beyond
   2^53 frames, irregular partitions, and serialized oscillator-state restoration.
   Longer stress runs and performance optimization remain future work.
2. The reusable Tuney voice renderer is consolidated in enge. `PreparedVoice`
   carries oscillator frequencies, envelope, minimum hold, gain, and explicit
   source-to-output route rows. `VoiceRenderer` owns oscillator state, the relative
   frame cursor, effective release, exact completion, and bounded rendering.
   `OfflineSynth` uses it with per-sample frequency and gain arrays; Tuney uses
   its prepared constant values. Their NumPy path uses `route_samples()` for mixing.
   Tuney's old `VoiceState` implementation is removed. Its host adapter retains
   mono duplication, stereo-to-mono averaging, and left-channel duplication for
   other output layouts, along with keyboard/polyphony policy and device code.
   Existing audio fixtures remain unchanged. Shared regressions cover restoration
   before and during release, fractional retirement with a nonzero terminal
   level, explicit routes, and exact silence after completion. Prepared definitions
   are shared; mutable renderer snapshots use deep copies or serialized state.
   The full prepared-action renderer remains offline; this consolidation does
   not add a live prepared-action host or change Tuney's note-selection policy.
   Each shared API update is published in enge before updating Tuney's pinned
   revision; tests use the local editable dependency.
3. The corresponding Rust slice is implemented with PyO3, rust-numpy, and maturin.
   The shared suite runs with explicit NumPy and native selection, retaining its
   independent numerical oracles, exact lifecycle checks, irregular partitions,
   and serialized restores. Direct comparisons cover fractional release during
   attack, zero-duration stages, duty-cycle endpoints, multiple source routes,
   and read-only strided parameter arrays. A test disables the reference DSP
   functions while rendering native output to prevent a silent Python fallback.
   `src/lib.rs` computes phase recurrence, waveform, envelope arithmetic, gain,
   and mixing while the GIL is released. The crate forbids `unsafe` code; the
   binding checks dimensions and span bounds, copies input arrays into owned
   Rust buffers, and transfers owned audio and next-state arrays back to NumPy.
   Tests verify that inputs and outputs do not alias. Default Rust floating-point
   operations preserve the compensated recurrence without fast-math reassociation.
   `src/enge/native.py` prepares exact rational envelope span boundaries. uFor
   actions and per-sample control resolution remain in Python. Snapshots retain
   their renderer backend; active voices cannot be restored into a different
   backend. Tuney continues to select NumPy by default.
4. Sampler: the NumPy and Rust cores implement the [numerical contract](sampler-numerics.md)
   and pass all 34 [traversal vectors](../conformance/sampler-traversal.json).
   `PreparedSample` validates and isolates decoded audio; `SampleState` and
   `sample_frames()` preserve position through live pitch, fractional releases,
   loop transitions, block partitions, and JSON restores. `SampleVoiceRenderer`
   shares `PreparedEnvelope`, `EnvelopeRenderer`, envelope evaluation, and routing
   with the synth. `OfflineSampler` shares `ControlRenderer` and `render_actions`
   scheduling, consumes prepared uFor actions, composes instrument/slot processing,
   and restores voices against shared decoded assets with content fingerprints.
   Tests cover sustain, group envelope overrides, gain/pitch variation, scoped
   controls, source exhaustion, one-shot release, replacement, and stop. Sample
   minimum hold is available on the reusable voice; uFor sample instruments have
   no authored minimum-hold field and therefore use zero.
   Explicit `backend="native"` selects Rust for the source, voice, or offline
   instrument API. `src/sampler.rs` performs traversal, overlaps, interpolation,
   gain, and routing with the shared Rust envelope arithmetic. Python prepares
   rational release splits before converting distances to float, preserving
   release-before-turn ties. Each prepared source caches a Rust-owned audio
   buffer; detached render calls share that buffer and own their control arrays.
   Integer frame/index state is never packed into float arrays. The same vectors,
   WAV oracles, changing controls, and restore tests now run on both backends.
   Direct native checks cover no Python DSP fallback, owned strided inputs,
   invalid binding inputs, and frame coordinates beyond float integer precision.
   Sampler snapshots reject backend mismatches even when silent. Modulated
   voices batch a prefix whose declared maximum tuning cannot reach natural
   exhaustion, then resolve the remaining uncertain frame alone; this milestone
   does not claim real-time callback performance.
5. Named seconds-clock LFOs now render through NumPy and Rust with shared
   conformance. They retain uFor's exact phase anchors, rate/reset semantics,
   separate activation weight, and scope ownership. Both synth and sampler
   consume them through existing amplitude/tuning routes and restore their
   source identities. Tests cover silent shared sources, release tails, exact
   duty ties, fractional timing, and large clocks. A two-second FLAC regression
   demonstrates tremolo, vibrato, and live gain on synth and sampled tones.
   The standalone source API accepts canonical LFO event states; prepared
   instrument traces still need a portable addressed rate/reset action before
   those events can be delivered through `advance`. See [details](lfo-numerics.md).
6. Dynamic filters are implemented in NumPy and Rust under the revised uFor
   contract. Cutoff/Q controls and LFOs preserve independent stage/channel states;
   static responses, rapid modulation, boundaries, lifetime, restores, and a
   listening demo have shared coverage. See [dynamic filters](#dynamic-filters).
7. Further generators, processing, and structural changes one specified feature
   at a time. Do not introduce a generic DSP graph or host to complete these steps.

### Phase consolidation timing check, 2026-09-16

A local arm64 macOS test measured one second of `Mixer.render()` calls at 48 kHz
after one warm-up block, using ten triangle oscillators: either ten mono notes
or five binaural notes. Blocks of 32, 64, 128, 256, and 1024 were measured. The
figures below are microseconds per call; setup, device callbacks, encoding, and
the dynamic-action preparer were excluded.

| Block / notes | Previous median | Shared median | Shared p95 | Buffer budget |
| --- | ---: | ---: | ---: | ---: |
| 32 / ten mono | 157 | 200 | 253 | 667 |
| 32 / five binaural | 133 | 198 | 247 | 667 |
| 1024 / ten mono | 277 | 1581 | 1741 | 21333 |
| 1024 / five binaural | 296 | 1595 | 1803 | 21333 |

The scalar compensated recurrence costs more than Tuney's previous vectorized
constant-pitch calculation. At 32 frames the largest observed shared-render call
was 480 microseconds for mono and 438 for binaural. These short mixer measurements
support this consolidation on the measured host; they do not establish live
deadline guarantees or the cost of the complete dynamic-action renderer. Retain
the recurrence as the correctness reference when evaluating a faster backend.

### Voice and routing consolidation timing check, 2026-09-16

The same one-second, ten-oscillator mixer test was repeated before and after
sharing the complete voice renderer. Final measurements ran without concurrent
checks. Times are microseconds per call, with setup and device I/O excluded.

| Block / notes | Previous median | Shared median | Shared p95 | Buffer budget |
| --- | ---: | ---: | ---: | ---: |
| 32 / ten mono | 197 | 223 | 283 | 667 |
| 32 / five binaural | 193 | 182 | 228 | 667 |
| 1024 / ten mono | 1529 | 1621 | 1795 | 21333 |
| 1024 / five binaural | 1552 | 1576 | 1835 | 21333 |

An initial run while other checks were active observed 32-frame calls up to
766 microseconds for mono and 945 for binaural, above the 667-microsecond budget.
After removing an unnecessary render-buffer copy, the isolated run's maxima
were 563 and 408 microseconds. These results establish the measured cost, not
live deadline guarantees; scheduler contention and note-on preparation still
need host-level measurement before claiming reliable live performance.

### Rust voice timing check, 2026-09-16

A local arm64 macOS pytest harness rendered one second at 48 kHz after one
warm-up block. Each call summed ten triangle oscillators, either ten mono voices
or five two-oscillator voices, using the public `VoiceRenderer` API. Python buffer
preparation, binding overhead, and Rust-owned input copies were included. Note-on
setup, dynamic uFor control evaluation, device I/O, and encoding were excluded.
Times are microseconds per call.

| Block / voices | NumPy median | Rust median | Rust p95 | Buffer budget |
| --- | ---: | ---: | ---: | ---: |
| 32 / ten mono | 209.8 | 132.4 | 150.3 | 667 |
| 32 / five stereo | 179.0 | 74.2 | 79.7 | 667 |
| 1024 / ten mono | 1690.2 | 273.2 | 336.6 | 21333 |
| 1024 / five stereo | 1632.3 | 160.3 | 187.0 | 21333 |

Blocks of 64, 128, and 256 were also faster with Rust. This measures the voice
API, not the earlier Tuney mixer harness or live deadline reliability. The native
backend still copies inputs and allocates per call, and the full dynamic-action
renderer still evaluates controls in Python. These are separate costs to measure
before choosing a live host integration or optimization.

## Additional work beyond the prompt

None.
