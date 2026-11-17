# Integrating Rust uFor into enge

## Goal and scope

Use Rust uFor's validated tuning, modulation and generator definitions during
enge's native preparation. Replace the corresponding Python-to-native conversion
code incrementally, preserving supported audio behavior and the existing
prepared runtime representation.

Implementation and all four recommendations below were approved on 2026-11-16.
Begin with literal LFOs and Envelopes. No further
large uFor port is a prerequisite. Record concrete API gaps discovered during
integration before proposing additional shared-library work.

## Current boundary and ownership

The initial Cargo manifest depended on `envelfo-core`, but not Rust uFor.
The first prerequisite migrates that dependency and imports to the consolidated
Git-distributed `engelfo` crate, which includes MIDI mapping and depends on middy.
`enge/synth.py::_persistent_modulation` compiles parameter/source/route metadata;
`_prepare_native_generators` configures the shared native `GeneratorBank`.
`enge/envelfo_graph.py::prepare_graph` builds the standalone scheduling graph.
Native definitions, route maps and execution adapters live in `src/runtime.rs`,
`src/runtime/generators.rs` and `src/runtime/patch_host.rs`.

Synth, FM and noise already use the shared generator bank. Sampler preparation
also uses that machinery, including common/slot remapping. Inspect current
callers before each stage because these paths continue to evolve.

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

#### Remaining graph and binding work

Extend the same boundary to the remaining SampleHold host setup, Patch signal nodes, output
selection and event connections. Preserve declaration order, authored arithmetic
order, shared-child identity, random stream identities and connection identities.
Retain the current host's supported profiles and explicit rejection behavior.

Use uFor's generator-aware checks for supplied definitions. Resolve references
before executable preparation; retain Python's full target/follower checks until
the required definitions are available natively. Never treat an empty subset of
outstanding checks as proof of complete instrument validation.

Replace the migrated native graph encoders. Review `envelfo_graph.py` by caller:
Python offline execution may still need its graph adapter, even after native
preparation moves. Remove code only when all its consumers have migrated.

### 3. Modulation preparation and mathematics

Use uFor definitions and validation while compiling routes into enge's existing
indexed tables. Preserve source scopes, key/velocity latching, control smoothing,
activation weights, target defaults/domains and mapping/composition order.

Adopt shared mapping/composition calculations where their interface fits the
runtime's allocation and error contracts. Measure before substituting a general
evaluator in callbacks; an allocating API can remain a preparation/reference
tool. Keep frame observations, controls and retained execution state in enge.

### 4. Tuning integration and closeout

Identify where native preparation or execution needs tuning evaluation rather
than assuming all Python pitch work should move. Use uFor's validated tuning
types at that boundary, preserving note addressing, exact ratios and existing
frequency preparation. Inspect synth, FM, noise and sampler consumers separately.

Remove obsolete conversion code for the completed stages, update native stubs
and documentation, and record the remaining Python-owned work. A useful next
uFor port would be `SoundSettings`, processing declarations and target resolution,
if the integration confirms these are the next blocker. Full instrument/effect
documents and TempoMap are separate milestones.

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

Ask immediately if implementation requires a changed host profile, wire shape,
snapshot contract, library ownership or dependency upgrade beyond adding uFor.

## Conformance and consumer verification

Keep uFor's shared corpora authoritative for portable definitions and mathematics.
Add enge tests for the conversion and actual host behavior, without copying the
entire library corpus or retaining a second production implementation as an
oracle. Reuse existing audio regressions and standalone core expectations.

Cover the following as their stages are implemented:

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
Run focused tests first, then the repository's required checks for implementation
commits. Exercise the opt-in allocation audit on successful and failed callbacks
affected by the migration. Preparation may allocate; callback behavior must retain
the existing zero-heap-operation guarantees for the qualified paths.

Test declared, non-editable Python installation and locked Cargo dependencies.
Preserve Python 3.13 and macOS/Windows/Linux compatibility. Specific Rust-version
targets are not a project requirement. Hosted
platform evidence comes from the existing workflows; do not launch release
workflows for every commit. Local tests do not establish hardware deadlines or
qualify unavailable platforms.

## Completion criteria and exclusions

Each stage is complete when its consumers use the new preparation boundary,
obsolete encoders are removed, expected accept/reject behavior is documented,
and affected audio, continuation and allocation checks pass. Record evidence and
remaining limitations in this plan as work lands.

No generator runtime rewrite, broad instrument/effect port, resource loader,
Python/Rust binding package in uFor, callback JSON parsing, new execution backend,
TempoMap migration, hardware access or performance claim is included.

## Additional work beyond the prompt

None.
