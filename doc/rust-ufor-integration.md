# Rust uFor integration in enge

Completed on 2026-11-21. This record describes the adopted preparation
boundaries, consumer behavior and qualification for the completed integration.

## Goal and scope

Use Rust uFor's validated tuning, modulation and generator definitions during
enge's native preparation. Replace the corresponding Python-to-native conversion
code incrementally, preserving supported audio behavior and the existing
prepared runtime representation.

Implementation and the four decisions below were approved on 2026-11-16.
The first slice covered literal LFOs and Envelopes.

## Current boundary and ownership

The initial Cargo manifest depended on `envelfo-core`, but not Rust uFor.
The first prerequisite migrates that dependency and imports to the consolidated
Git-distributed `engelfo` crate, which includes MIDI mapping and depends on middy.
`enge/synth.py::_persistent_modulation` discovers host parameter/filter metadata
and calls native `prepare_modulation` to compile the indexed source/route tables;
`_prepare_native_generators` configures the shared native `GeneratorBank`.
`enge/envelfo_graph.py::prepare_graph` builds the standalone scheduling graph.
Native definitions, route maps and execution adapters live in `src/runtime.rs`,
`src/runtime/generators.rs` and `src/runtime/patch_host.rs`.

Synth, FM and noise use the shared generator bank. Sampler preparation also
uses that machinery, including common/slot remapping.

| Owner | Responsibility |
|---|---|
| uFor | Authored definitions, units, score materialization, tuning/modulation mathematics and portable validation |
| enveLFO | Generator execution, transitions, retained Patch state and continuation |
| enge | Supported audio profiles, parameter/source indices, frames/clocks, voice ownership, prepared storage, rendering and callback policy |
| Python host | Authoring, resource/reference resolution and orchestration until explicitly migrated |

Rust uFor's score codec currently supports Tuning and Envelfo scores. This does
not provide a native decoder for complete instrument/effect documents. Existing
Python document and target validation remain necessary. The partial binding
validator reports unresolved reference, follower, event and target requirements;
enge must discharge applicable requirements before building an executable plan.

## Implementation stages

### Approved first-slice boundary and inventory

On 2026-11-16 the user approved replacing the native GeneratorBank constructor
and named-Envelope setters with one constructor taking canonical JSON LFO and
named-Envelope definitions, separate host routing/index metadata, and keyword
`max_records=10000`. The limit covers these supplied definitions and route
records together. Canonical per-definition text keeps uFor's validated parser;
aggregate preparation text is also bounded. Existing beat-clock profiles and
SampleHold route slots remain supported. Main voice-amplitude conversion is a
separate follow-on slice, as are Stages/Patch graph and modulation conversion.

The current LFO encoder combines waveform/clock and rational body fields with
scope/parameter/operation/affine route metadata. Named Envelopes combine exact
frame-scaled segment arrays with release/loop/retrigger options, and attach
captured approaches through a later setter. The replacement reads authored
quantities in Rust, converts exact seconds to frames once, and retains enge's
scope/reset policy and prepared state ownership. Named loop bounds retain the
existing normalized native representation. Main amplitude arrays and Stages
encoders retained consumers at this boundary; the follow-on slices below replace
them.

The user also approved mapping uFor's valid `retrigger='reset'` to the native
Initial policy, fixing the old encoder's lookup of the nonexistent value
`initial`. Native FM/noise named captured approaches must retain their authored
parameters instead of losing them as placeholder segments. Add focused consumer
regressions for these two approved alignment fixes.

### 1. Contract inventory and first preparation slice

The inventory and approved boundary above are implemented. `fa6ea68` migrates
the native dependency/imports to `engelfo`; the baseline passed 3,570 Python tests
with 29 optional skips. `137773a` separately adds the pinned Rust uFor and JSON
preparation dependencies.

`src/runtime/preparation.rs` validates canonical definitions with uFor and
constructs the existing engelfo bodies. `_persistent_modulation` carries JSON
definitions alongside enge routing metadata. Synth, FM, noise, effects and sampler
bank construction use the new interface; the old LFO rational encoder and named
Envelope/approach/beat-flag setters are removed. Stages initially retained their
existing encoder and are migrated by the later slice below. Native stubs and
callers have been updated together.

The first slice retains existing seconds/beat clocks, scope/reset policy,
SampleHold slots, shared ownership and normalized loop bounds. Named captured
approaches are prepared directly, including the approved FM/noise correction;
reset retriggering is mapped to Initial. The user approved signed-i64 portable
repeat counts, rejecting values above `2**63 - 1` during preparation.

Preparation tests cover invalid definitions, aggregate record overrides, exact
frame overflow, repeat-count boundaries, independent gain equations and native
partitioned snapshot continuation. Existing audio/ownership tests use the new
constructor. JSON decoding stays outside callback execution. Python dependency
pins are unchanged; tests use installed non-editable enge and locked Cargo inputs.

Verification on 2026-11-16 passed: 3,583 Python tests with 29 optional skips under
`--require-callback-no-allocations`, including the new preparation regressions.
The audit recorded 24,703 successful and 68 failed callbacks with zero callback
allocations, reallocations or frees. The broader native-call boundary still
records allocations outside callbacks; this is not an allocation-free API claim.
Both Rust allocation-audit unit tests, Clippy, Ruff, formatting, ty and pyupgrade
passed. The ordinary extension is restored after qualification. Hosted
Windows/Linux evidence and hardware deadlines remain outside these local checks.

Reproduce the audited run with:

```sh
uv sync --frozen --no-editable --reinstall-package enge --config-settings-package enge:build-args=--features=allocation-audit
uv run --no-sync pytest -q --require-callback-no-allocations --junitxml=build/rust-ufor-callback-audit.xml
uv run --no-sync python scripts/summarize_callback_audit.py --reports build/rust-ufor-callback-audit.xml
uv sync --frozen --no-editable --reinstall-package enge
```

### Main voice-amplitude preparation

On 2026-11-17 the user approved canonical Envelope JSON at each native amplitude
constructor, replacing the initial/segment-array arguments and captured-release
setters. Each operation has keyword `max_records=10000`; FM combines all supplied
operator definitions under one budget. Graph FM receives one Envelope per operator,
without a duplicate carrier definition. The bank and amplitude constructors are
separate preparation operations with independent budgets.

The inventory found that main amplitude already uses normalized uFor Envelope
levels. Public parameter materialization, scalar voice gain, minimum-hold timing,
release timing and sample exhaustion remain host-owned. The supported profile is
a voice-scoped, seconds-clock, held, once Envelope with numeric initial,
activation start, no markers and an explicit release. Captured approaches are
supported only in release. Finite curves and indefinite holds retain their
existing behavior. Voice reuse owns fresh activation and keeps the native Current
retrigger policy regardless of the authored named-generator retrigger setting.

`src/runtime/preparation.rs` reuses the validated generator decoder and Envelope
converter for synth, two-operator FM, graph FM, noise and live sampler construction.
Exact durations convert to frames once; captured releases are present before
initial state and command storage are prepared. Runtime commands, continuation
state, snapshot identity checks and retirement are unchanged. Raw native arrays
that bypassed the supported profile now fail during definition validation.

The Python amplitude and captured-release encoders and native setters are removed.
At this boundary `_runtime_segments` remained only for Stages; the subsequent
Stages slice removes it.
Python dependency pins and the Rust dependency set are unchanged.

`test/test_amplitude_preparation.py` covers unsupported profiles, invalid portable
types and ratios, exact frame overflow, every FM operator, and combined record
limits with explicit overrides. Existing amplitude, runtime and live tests cover
minimum hold, captured release interruption, voice reuse, all four audio consumers
and partitioned snapshot continuation.

Verification on 2026-11-17 passed: 3,604 Python tests with 29 optional skips under
`--require-callback-no-allocations`. The audit recorded 24,703 successful and 68
failed callbacks with zero callback allocations, reallocations or frees. Both
Rust allocation-audit unit tests, Clippy, Ruff, formatting, ty, pyupgrade and
`git diff --check` passed. The broader native-call boundary retains allocations
outside callbacks. The generated report is `build/rust-ufor-amplitude-audit.xml`;
reproduce it with the commands above, substituting that report path. Local
qualification used macOS; Windows/Linux and hardware deadlines were not tested
here. The ordinary extension was restored afterwards and all 39 focused amplitude
preparation/release tests passed again.

### 2. Stages, Patch and generator-aware bindings

#### Stages preparation slice

The Stages body encoder is replaced by canonical JSON during native preparation.
`GeneratorBank.set_staged_envelfos` takes `(document, optional parameter index,
operation)` tuples plus the existing host connection records, with keyword
`max_records=10000`. One operation counts the preparation root, route slots,
all nested uFor records and each host connection. The existing document bounds
also apply to the supplied JSON. This is independent of the bank constructor's
LFO/Envelope budget.

Rust uFor validates concrete Stages definitions and enge converts their child
LFOs/Envelopes, markers, loop bounds and transitions into the existing runtime.
Stages remain voice-scoped; beat clocks still require a prepared host clock.
Exact quantities convert once. Public rates must already be materialized.
Stage names remain numeric indices in native bodies, declaration and transition
order are retained, and the host's held/Current Envelope-entry policy is preserved.
Commands, connection scheduling, continuation storage and snapshot layout are
unchanged. Noninitial current-value indefinite holds are supported by the shared
model; the initial stage still cannot capture a current value.

The user approved accepting valid child `retrigger='reset'` on 2026-11-17. The old
Python encoder rejected it by looking for `initial`. This corrects acceptance;
stage-entry playback keeps its existing fresh state and Current policy. Portable
repeat counters use signed i64, consistently with the preceding migrations.

The nested Python stage, segment and Envelope-option encoders, tuple aliases and
old native stage-array decoder are removed. All native consumers use the new
setter through their existing prepared bank. Python offline graph preparation
still has live consumers and remains in place. Dependency pins are unchanged.

Preparation tests cover child/schema failures, unknown transition targets,
route metadata, time overflow, aggregate limits and overrides. The existing
Stages playback regression now also exercises reset acceptance, named loop
bounds, release and snapshot replay. Focused native Patch, sampler, indefinite,
clock and mapping regressions cover the surrounding callers.

Qualification on 2026-11-17 passed 3,619 Python tests with 29 optional skips under
`--require-callback-no-allocations`. The report
`build/rust-ufor-stages-audit.xml` records 24,703 successful and 68 failed callbacks
with zero callback allocations, reallocations or frees. The broader native-call
boundary retains allocations outside callbacks. Reproduce this report using the
installed-build commands above with its path substituted. Both Rust
allocation-audit unit tests, Clippy, Ruff, formatting, ty, pyupgrade and diff
checks passed. Verification was local to macOS; Windows/Linux and hardware
deadlines were not tested here.
The ordinary extension was restored after qualification; all 19 focused Stages
preparation/playback tests passed again.

#### Patch configuration and generator-aware binding preparation

On 2026-11-17 the user approved one native Patch configuration operation with
canonical generator JSON, existing modulation/binding/control/event declarations,
and separate host slot/index/seed metadata. Its configurable record budget defaults
to 10,000. Unresolved generator/follower/event requirements must fail before
executable preparation. Full processor-target validation remains in Python.
The user also approved preserving FM/noise's existing supported profiles, while
applying the same generator-aware checks to them; this slice does not add their
missing full Patch configuration path.

The approved shared API prerequisite is implemented in uFor `bcef1e7`:
`codec::parse_value` exposes the existing document-budget decoder without claiming
domain validation or object-order retention. Enge's dependency-only commit
`cab3a53` pins that revision. No dependency package versions or Python pins change.
Generator bodies still pass through `parse_generator` on their original text.
Delays outside the portable numeric domain now fail during shared declaration
decoding with its 64-bit-domain error, before the host frame-conversion check.
The existing overflow regressions use that shared error wording.

`GeneratorBank.configure_patches(context, generators, slots, targets, orders,
max_records=10000)` validates a private preparation context containing only the
existing `modulation`, `bindings`, `controls` and `event_connections` fields.
Named generator documents are separate ordered `(name, JSON)` entries. Slot
metadata maps names to LFO/Envelope/Stages bank indices, target metadata supplies
host parameter indices, and Patch child-order metadata retains the current native
layout. The context is not a score kind, a full ModulationSettings model, or a
new persisted document format.

The operation counts all shared declaration/reference records, supplied generator
occurrences and host layout/index records. Aggregate text and per-document
character/value/depth bounds apply. The override reaches embedded modulation and
generator validation. Rust checks real generator/output/range/release/port facts
using `validate_generator_bindings`; remaining generator/follower/event checks
are rejected. Processor targets remain checked by Python settings and the host
profile, with native target-index lookup establishing only the prepared address.

The Rust compiler constructs the existing Sum/Product/Affine/Quantize/Slew/Latch/
Threshold plans, selected outputs, marker events, stage starts and threshold
events. Authored input order, legacy host node order, event-category ordering,
stream-key identities, exact frame delays and threshold frame offsets are
retained. Host order is validated against actual dependencies rather than
replaced with another valid topological order. Existing internal native indexed
plan/profile checks remain; configuration uses a prepared copy and replaces the
bank only after all checks pass. Callback execution and snapshots are unchanged.

Synth, effects and sampler use this operation. Persistent FM/noise use the same
declaration decoder and generator-aware validator before their existing bank
construction. SampleHold ranges come only from canonical definitions;
`set_sample_hold_keys` supplies host stream keys without duplicating range input.
The old Python Patch operator/event encoders and their public native setters are
removed. The single Python host parameter-index map is reused for Patch routes.

At this slice's boundary `envelfo_graph.py`, Patch-output materialization,
source/owner indices and generic Stage-to-Stage connection metadata retained
Python consumers. The modulation slice below moves the indexed source maps and
generic Stage connection encoding into native preparation. The offline graph
adapter, Patch-output materialization and owner orchestration remain host work.

New preparation cases cover unresolved declarations, explicit layout order,
aggregate budget boundaries and overrides through both generator and modulation
bodies, plus rejected configuration without mutation. Existing Patch/Threshold/
Latch/SampleHold, sample, effect and continuation regressions qualify the migrated
consumers.

Qualification on 2026-11-17 passed 3,628 Python tests with 29 optional skips under
`--require-callback-no-allocations`. The generated report
`build/rust-ufor-patch-audit.xml` records 24,703 successful and 68 failed callbacks
with zero callback allocations, reallocations or frees. The broader native-call
boundary retains allocations outside callbacks. Reproduce this run with the
installed-build commands above and this report path. Both Rust allocation-audit
tests, Clippy, Ruff, formatting, ty, pyupgrade and diff checks passed. Verification
was local to macOS; Windows/Linux and hardware deadlines were not tested here.
The ordinary extension was restored and all 20 focused preparation, delay-limit,
SampleHold continuation and bank-ownership tests passed again.

### 3. Modulation preparation and mathematics

On 2026-11-17 the user approved shared allocation-free mapping and accurate-sum
kernels, preserving enge's current contribution category order and full-weight
shortcuts. Nonfinite intermediate mapping/sum arithmetic must fail explicitly,
including mappings that formerly returned finite results after overflow.
The user also approved replacing the Python route/control/smoothing encoder with
one native preparation operation, retaining Python's DSP target/default/filter
discovery and orchestration.

uFor `1296471` provides `modulation::math::{Mapping, add_partial, sum_partials}`
and fixed-size `CalculationError`. Both uFor's allocating scalar reference
evaluator and enge's prepared runtimes use these kernels. Enge's separate
dependency commit `4b0bd5b` pins that implementation. No dependency package
versions or Python pins change.

`prepare_modulation(sample_rate, context, generators, targets, filters,
max_records=10000)` compiles canonical declarations and separate host metadata
into the existing 23-field prepared tuple. Source maps retain insertion order;
routes within each category retain `(source, route)` order. Control scopes,
key/velocity slots, exact frame smoothing, generator sharing, neutral event-only
slots, generic Stage connection delays and stream keys keep their existing
meaning. Canonical parameter declarations override the discovered host defaults
and domains as before. Full processor validation and supported renderer profiles
remain host-owned; FM/noise do not gain the full Patch path.

The operation counts declaration and generator records, host targets/filters,
and each expanded generator slot/body against one configurable budget. Expanded
generator text also has a combined document character cap, preventing repeated
routes from multiplying preparation work without a bound. Existing per-document
value/depth limits apply. The override reaches nested bodies and modulation.
Exact smoothing/frame conversion rejects overflow in signed 64-bit components.
Native stubs and callers use the same boundary; the old Python encoder and
Stage-event delay converter are removed. FM/noise's former standalone binding
validation call and its native export are also removed: route preparation now
performs those shared checks for every consumer.

Enge's old mapping and summation implementations are removed. Synth, FM, noise,
sampler and effect modulation use prepared shared mappings and caller-owned sum
storage. Effect routing matrices, mixers and feedback-row preparation also used
the old sum helper and now reuse the shared kernel. Callback errors retain a
copyable payload; Python exception construction stays outside the callback.

Sharing these kernels does not make enge's composition policy identical to the
reference evaluator: enge retains controls/events, LFOs, named generators,
Stages and Patch-transform category order. It also retains direct full-weight
contributions where the reference computes `1 + weight * (amount - 1)`, which can
round differently for tiny multipliers. Global route interleaving and removal of
these shortcuts are deferred behavior changes requiring explicit qualification
and, for global ordering, route-order metadata in prepared runtimes/snapshots.
The general uFor evaluator remains an allocating reference API outside callbacks.

`test/test_modulation_preparation.py` covers scope/event layout, exact smoothing,
host facts, combined/expanded limits and overrides, frame overflow and shared
mapping/sum failures with zeroed callback output. Existing mapping/audio,
generator ownership, effect matrix/mixer and continuation regressions cover
the migrated consumers. Matrix sum-overflow rejection also retains its contextual
diagnostic, zeroed output and snapshot recovery.

Qualification on 2026-11-17 passed 3,641 Python tests with 29 optional skips under
`--require-callback-no-allocations`. The generated report
`build/rust-ufor-modulation-audit.xml` records 24,704 successful and 71 failed
callbacks with zero callback allocations, reallocations or frees. The broader
native-call boundary retains heap operations outside callbacks. Reproduce this
run with the installed-build commands above and this report path. Both Rust
allocation-audit tests, Clippy, Ruff, formatting, ty, pyupgrade and diff checks
passed. Local qualification used macOS; Windows/Linux and hardware deadlines
were not tested here. The ordinary extension is restored after qualification.
All 50 focused preparation, mapping and feedback-limit cases passed again with
that ordinary extension.

### 4. Tuning integration and closeout

Tuning preparation is implemented in `enge.performance.MidiConverter` in
commit `5c7d1af`.
Each referenced tuning is prepared once using Rust uFor's portable JSON codec.
Note-on evaluation uses the retained Rust definition and converts to float Hz
when creating `Trigger.pitch_hz`. Missing or invalid definitions fail
construction. Unavailable notes still fail feed and invalidate pairing state.
The host resolves references and the unconfigured default remains A4 = 440 Hz
in 12-tone equal temperament.

The consumer audit found no additional tuning evaluation needed in the renderers.
Performance traces carry resolved Hz into voice actions, with existing
voice-specific offsets or explicit trigger pitches applied during tracing.

| Consumer | Voice preparation contract |
|---|---|
| Synth | Uses positive finite `VoiceStart.pitch_hz` as the base oscillator frequency. |
| FM | Uses positive finite `VoiceStart.pitch_hz` as the base for operator frequencies and their existing ratios and offsets. |
| Noise | Uses the resolved random stream key. Pitch does not enter the noise signal path. |
| Sampler | Pitch-tracked slots derive a playback ratio from positive finite `VoiceStart.pitch_hz` and the sample reference pitch. Unpitched slots can start without a frequency. |

These are the existing offline and native paths. Cents modulation, sample
variation, and playback ratios are renderer processing of already resolved
pitches. Callers that supply their own `Trigger.pitch_hz` can use the same
renderer contract directly. See the [performance binding contract](performance-binding.md).

The tuning implementation passed 90 targeted tests on macOS, including MIDI
file and live input, rendering and portamento. Closeout reran 50 focused cases
covering converter and MIDI file behavior, synth pitch admission, persistent FM,
pitch-independent noise, and sampler pitch and continuation. All passed.

Python continues to resolve resources and references, build performance traces,
and validate complete instrument and effect settings. The
[native modulation contract](envelfo-contract.md#modulation-route-preparation-and-shared-mathematics)
describes the later SoundSettings, FM, and initial effect target integration.
Complete instrument and effect document loading and TempoMap remain separate
milestones, as do changes to global modulation ordering and weighting.

## Approved decisions

1. **Preparation input.** Recommend canonical JSON text at the preparation-only
   PyO3 boundary, decoded by Rust uFor. It reuses the portable format and ordered
   text decoding without binding Rust to Python model internals. It adds encoding
   work at preparation. Direct Python-object extraction avoids that encoding but
   requires another field conversion layer. Confirm the exact payload and method
   signature during the first-stage inventory.
2. **Partial adoption.** Recommend migrating one complete supported profile and
   its callers at a time, keeping Python orchestration and enge's prepared bank.
   Moving whole instrument loading now would require substantially more uFor
   definitions and resource-resolution policy.
3. **Exact numbers and acceptance.** Recommend checked conversion of uFor's i64
   rationals into core time types, preserving exact values without float bridges.
   uFor uses checked 64-bit intermediates; the current core uses wider
   intermediates. Inventory currently accepted inputs that portable uFor rejects,
   including broad authoring syntax and oversized Python Fractions. Normalize
   authoring conveniences before the boundary; ask before changing acceptance of
   existing valid documents. Do not silently clamp, approximate or widen uFor.
4. **Limits and errors.** Recommend one explicit per-operation preparation budget,
   default 10,000 records, propagated through decoding and combined validation.
   Keep independent document limits and enge's voice/event/storage capacities.
   Translate validation errors into the existing Python preparation exception
   convention. Callback errors retain the fixed-size native error path.

## Conformance and consumer verification

uFor's shared corpora define the portable contracts. Enge's consumer tests
exercise the conversion and host behavior using existing audio regressions and
standalone core expectations.

The verification scope included:

- Literal LFO phase/rate/reset/onset and Envelope curves, indefinite holds,
  release interruption, markers and supported captured approaches.
- Exact seconds/beats and frame conversion at boundaries, zero and large values,
  signed i64 limits, overflow rejection and unsupported host profiles.
- Stage cues, inherited settings, Patch output order, shared children, event
  routing/delay, release timing and deterministic random identities.
- Linear/step/exponential maps, endpoint clamping, cancellation, operation order,
  activation weights, parameter domains and unresolved binding rejection.
- Synth/FM/noise/sampler output, common/slot remapping, delayed starts, voice
  retirement/reuse and overlapping contexts for the migrated profiles.
- Whole versus partitioned rendering, snapshot continuation/replay and isolation
  between separately prepared instances. Existing snapshots must retain their
  meaning; any required snapshot-format change needs a separate decision.

Audio regressions use 48,000 samples per second and at least one second of output.
The stage records give the results from installed Python builds with locked
Cargo dependencies. The opt-in allocation audit measured successful and failed
callbacks for the migrated runtime paths. The results describe local macOS
testing. Windows and Linux execution and hardware deadlines require separate
qualification.
