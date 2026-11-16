# Integrating Rust uFor into enge

## Goal and scope

Use Rust uFor's validated tuning, modulation and generator definitions during
enge's native preparation. Replace the corresponding Python-to-native conversion
code incrementally, preserving supported audio behavior and the existing
prepared runtime representation.

This document authorizes no implementation or dependency changes. Begin with
literal LFOs and Envelopes after the decisions below are confirmed. No further
large uFor port is a prerequisite. Record concrete API gaps discovered during
integration before proposing additional shared-library work.

## Current boundary and ownership

The current Cargo manifest depends on `envelfo-core`, but not Rust uFor.
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

### 1. Contract inventory and first preparation slice

Inventory the exact generator fields and transformations crossing the current
PyO3 boundary: rates, phase, curves, held/release segments, markers, scope,
ownership, onset, origins, indices and preparation capacities. Identify the
callers and old encoders that the first slice will replace. Keep audio-host
restrictions separate from portable-format validity.

Add a Rust uFor dependency pinned to a reviewed commit, in a separate dependency
commit. Use the repository's declared setup, with no editable or sibling-path
dependencies. Avoid upgrading `envelfo-core` or unrelated packages as part of
that dependency change.

Implement preparation of literal seconds-clock LFOs and explicit-initial
Envelopes for the already supported host profiles. Validate canonical definitions
through uFor, translate them into existing core types, and populate the existing
bank. Check main amplitude preparation separately from named modulation
Envelopes; migrate each supported caller when its contract is covered.

Keep decoding, validation, graph work, strings and allocation in preparation.
Preserve prepared indices and callback storage contracts. Replace each migrated
encoder completely, updating its callers and removing obsolete code. Do not
introduce a production fallback that retries the old encoder after rejection.

### 2. Stages, Patch and generator-aware bindings

Extend the same boundary to Stages, SampleHold, Patch signal nodes, output
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

## Decisions before implementation

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
Preserve Python 3.13, Rust 1.85 and macOS/Windows/Linux compatibility. Hosted
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
