# enveLFO project extraction

Plan, 2026-10-11. Spin the existing Envelfo system out into its own project,
with independently usable Python and Rust libraries. This document authorizes
no implementation by itself. It describes extraction and integration, not a
new modulation architecture.

## Goal

enveLFO replaces separate envelope and LFO workflows with one control model:
simple envelopes and periodic modulation remain easy, while Stages, named
outputs, composition, and event connections support more involved behavior.

The Python library must work without NumPy, enge, or a compiled extension.
The Rust library must work without Python, PyO3, NumPy, or enge. Both must
implement the same documented behavior and pass shared conformance cases.
Neither core owns audio rendering, MIDI ports, clock acquisition, or application
configuration. A host supplies time and inputs and consumes signals and events.

Use `enveLFO` as the project display name, `envelfo` for the repository and
Python package, `envelfo-core` for the Rust crate, and `Envelfo` in type names.
The recent breaking rename remains canonical. Do not introduce old-name aliases
or backward readers during extraction.

## Current starting point

- `ufor/envelfo.py` combines portable definitions, validation, state models,
  exact-position semantics, evaluation, and stage/event processing.
- `ufor/envelfo_random.py` defines deterministic stream identities and random
  words. These identities and vectors are part of observable behavior.
- `ufor/control.py`, `ufor/lfo.py`, `ufor/envelope.py`, `ufor/segments.py`, and
  `ufor/oscillator.py` supply timing, shapes, and existing semantic helpers.
- `enge/synth.py` owns host preparation, control evaluation, routing, scope and
  voice lifecycle integration, and accelerated block evaluation.
- `enge/src/runtime.rs` contains native Envelfo execution inside an audio runtime
  exposed through a Python extension. It is not yet a standalone control crate.
- uFor and enge tests cover exact boundaries, transport changes, event routing,
  composition, randomness, continuation, and snapshots. Backend coverage is
  substantial, but not every Python capability is an independent Rust feature.
- arpeg already provides a standalone `arpeg-core` crate and Python runtime,
  with shared language-neutral event traces. Reuse these integration boundaries.

Inventory these sources before moving code. Mark each capability as Python,
Rust, or host-only, with the tests that establish its behavior. In particular,
distinguish the existing scalar semantics from enge's block accelerators and
record restrictions currently imposed by particular renderers.

## Ownership and dependencies

| Project | Owns | Does not own |
| --- | --- | --- |
| uFor | Authored definitions, validation, score kinds, library references, unit normalization, portable interchange contracts | Executable Envelfo runtime |
| enveLFO | Evaluation, runtime state transitions, clocks as supplied observations, event processing, composition, deterministic randomness, continuation | Audio rendering, note selection, application I/O |
| enge | Audio rendering, synthesis integration, voice ownership, mapping Envelfo signals to sound parameters | A second implementation of Envelfo semantics |
| arpeg | Note selection, rhythm, expressive note/controller ownership, arpeggiator scheduling | A second modulation engine or independently acquired clock |
| Integrating host | Shared time/transport observations, input delivery, explicit connections, output scheduling and combined state | Hidden changes to either library's semantics |

Python dependency direction is `envelfo -> ufor`; uFor must not import envelfo.
enge and arpeg may depend on both. Keep uFor's definitions and schema validation
independent of runtime evaluation. Classify interleaved helpers carefully:
definition validation and structural port checks stay with their definitions;
execution and state transitions move to envelfo. Shared format-level mathematics
can remain in uFor where consumers already need it, without retaining a duplicate
runtime there.

The Rust crate exposes native definition and state types corresponding to the
portable contract. Using it does not require installing the Python uFor package.
Rust types are a language representation of that contract, not a separate format
with different defaults or behavior.

Keep Pydantic and reccy's Pint normalization on the Python authored-document
boundary. Rust execution consumes normalized numbers and explicit clock units.
Do not implement another project-specific unit parser or require arbitrary Pint
expression parsing in the Rust core. Document which Rust entry points accept
normalized definitions and how authored documents reach them.

## Library boundaries

### Python

Move the reference evaluator into the `envelfo` package. Accept uFor definitions
and existing typed events rather than introducing replacement Python schemas.
Keep an idiomatic public interface for initialization, timestamped commands,
advancement, signal queries, emitted events, and snapshot continuation.

The pure Python path is first-class. It must not import NumPy, enge, or native
bindings, even indirectly. Preserve scalar and event-stream execution. Move or
adapt existing block algorithms only where they are part of the control core;
NumPy/audio-specific acceleration remains an enge adapter concern.

### Rust

Extract a normal Rust library crate, not only a Python extension. Separate
Envelfo definitions, state, and event decisions from synthesis voices, audio
buffers, and PyO3 conversions. enge should call the extracted implementation
instead of keeping copies of the old native algorithms.

Use exact rational time for observable boundaries and event timestamps, and
integer frame positions where the caller explicitly uses frames. Keep seconds,
local beats, transport position, and frames distinct. Inspect arpeg's existing
rational representation before selecting one for interoperation. Document
supported numeric ranges and overflow errors rather than silently converting
event ordering to floating-point time.

Signal values may use floating point with declared comparison tolerances.
Commands and event ordering must not depend on polling frequency or block size.
Keep event capacities and unsupported-operation errors explicit where existing
execution contracts already require them. Do not add new concurrency machinery
or broad execution guarantees as part of extraction.

### Optional Python bindings

Bindings are an optional adapter over `envelfo-core`, with their own packaging
boundary. They must not be needed to install or run the pure Python library.
Keep the Python reference implementation independent: it is not merely a wrapper
around Rust. Use the same definitions and conformance inputs for both paths.

## Preserve behavior during extraction

The extraction is an import and ownership change, not a redesign. Preserve:

- Cycle shapes, phase, delay, fade, reset, and signal ranges.
- Contour segments, release timing, loops, direction, finite repeats, and
  boundary completion behavior.
- Stages, transitions, captured entry values, and named outputs.
- Patch operations, SampleHold, Latch, Quantize, Threshold, and Slew.
- Event connection ordering, cue routing, probability, every/offset behavior,
  delayed delivery, and cancellation rules.
- Explicit seeds, stream identities, random words, and independent streams.
- Host tempo/transport observations, beat quantization, cancellation on seek,
  and the distinction between elapsed and transport-position playback.
- Trigger-addressed commands, ownership scope, continuation, and restoration.

At a named boundary, preserve marker-before-boundary ordering. For finite
contour traversal completion, preserve boundary-before-done ordering and the
held boundary value. Keep immediate release as the default with the existing
optional voice-timing behavior supplied by the host integration.

If a Rust feature is missing or currently approximated, record it and implement
it against a shared case. Do not label it equivalent solely because two paths
agree on one audio test. Any intentional behavioral change needs a separate
decision and commit, not an incidental extraction fix.

## Shared conformance

The new repository owns language-neutral runtime cases. Start by moving the
existing cases and promoting suitable existing tests, not inventing a separate
test framework. Follow arpeg's JSON trace approach where it fits.

Each case records the normalized definition, initial state or seed, supplied
clock observations, ordered timestamped inputs, query/advance boundaries,
expected signal observations, and expected events. Include final state or
continuation expectations when relevant. Both languages read the same fixtures;
neither generates the other's expected answers during testing.

Require exact comparisons for timestamps, event identities/order, discrete
state, and random vectors. Use explicit tolerances only for floating signal
values. Independently derived simple cases anchor the fixtures so agreement
between implementations cannot conceal the same mistake in both.

Cover:

1. Hold, ordinary Cycle, and attack/release Contour common cases.
2. Fractional boundaries, simultaneous events, named markers, loops, finite
   repeats, reverse/pause/resume/seek, and delayed commands.
3. Seconds and beat clocks, tempo changes, stops, transport seeks, and
   quantized commands.
4. Stages, named Patch outputs, signal transforms, and routed child events.
5. Deterministic random streams, connection gates, and every/offset selection.
6. Whole-span versus irregular partitions versus single-position queries.
7. Snapshot/restore followed by equivalent continuation in each language.
8. Existing capacity errors, invalid definitions, and explicitly unsupported
   combinations without silent fallback.

Keep shared document-schema cases in uFor and audio integration regressions in
enge. Test the Python package in an environment with no NumPy, enge, or bindings;
test Rust directly with no Python process. Cross-language snapshot interchange
is not assumed: preserve each implementation's continuation contract first,
and document any common snapshot representation separately if required.

## Integration with arpeg

Connect the existing libraries through host-supplied time and typed inputs and
outputs. Neither core opens ports or independently acquires tempo. The same
accepted clock/transport observations reach both, while each retains its own
documented local position and lifecycle.

Start with a small deterministic composition:

- arpeg emits a hit event that triggers an enveLFO contour.
- A separate enveLFO Cycle supplies a named expression output for played notes.
- Explicit mapping delivers that output to arpeg's existing expression path,
  without discarding the note's captured breath, bend, or controller ownership.

Specify the controlled parameter's existing update boundary: continuous
expression updates are not interchangeable with selection or rhythm decisions
made when a hit is scheduled. Begin with supported expression targets; add
density, gate, or rhythm targets only through separately specified arpeg APIs.

Use existing timestamp and ordinal conventions for same-time delivery. An arpeg
hit must reach enveLFO before observing its triggered result for that hit. Define
an explicit phase for returning mapped expression to arpeg. Start with an
acyclic connection graph; feedback must use already specified delayed event
semantics, not an unbounded same-time evaluation loop.

Combined conformance traces must exercise irregular advancement, tempo changes,
stop/seek behavior, final note releases, expressive controllers between notes,
and restoration of both libraries plus pending host deliveries. Verify that
stop/seek actions follow each library's existing policy rather than inventing
one blanket reset rule. Keep these integration fixtures in arpeg, consuming
released/pinned enveLFO libraries.

This plan does not convert or remove arpeg executables. Its existing Rust crate
is already the initial library boundary. A separate application can link both
cores without making either library depend on the other. Update arpeg's old
Envelfo terminology and integration notes when that integration is implemented.

## Implementation sequence

Step 1 is complete as of 2026-10-11: see the
[inventory and baseline contract](../doc/envelfo-contract.md), the shared
`conformance/envelfo-runtime.json` traces, and `test/test_envelfo_contract.py`.
These establish reference behavior and record extraction/parity gaps; they do
not claim that either standalone library has already been extracted. Step 2 is
next. Existing consumer tests remain in place.

1. **Inventory and freeze the contract.** Build the capability/parity matrix,
   identify pure versus host-specific responsibilities, and establish baseline
   conformance cases. Record Rust gaps without treating them as completed work.
2. **Extract the Python evaluator.** Create the new repository/package, leave
   portable definitions in uFor, move executable semantics and random execution,
   and change consumers to direct imports. Remove replaced implementations;
   verify no dependency cycle and NumPy-free installation/execution.
3. **Extract the Rust core.** Separate the existing algorithms from enge's audio
   runtime and binding types. Establish exact event-time interoperation, add
   missing behavior in focused slices, and pass the shared fixture set directly.
4. **Integrate enge and optional bindings.** Make its Python and native paths
   consume the new libraries. Retain voice/routing/audio ownership in enge and
   preserve existing audio regressions, snapshots, and measured performance.
5. **Integrate arpeg.** Implement the hit/contour/expression example with shared
   time observations, explicit scheduling phases, and composed conformance
   traces. Keep runtime dependency direction acyclic.
6. **Finish the extraction.** Remove remaining duplicate execution paths and
   stale imports, document install/use examples for each independent library,
   and verify the full parity matrix and consumer suites before release.

Use reviewable commits for each coherent slice. Coordinate breaking consumer
changes and dependency revisions; do not leave a committed consumer using a
removed API. Dependency-only commits remain separate where independently
testable; ask before combining inseparable changes. Commit and push each
completed slice after its applicable verification gates pass.

Cross-platform release checks cover Linux, Windows, and macOS. Follow the
existing release-only policy rather than introducing per-commit platform runs.
Review dependency licenses and publish reproducible library build instructions
before distributing releases. No new dependencies or tooling are prescribed
without an implementation need and review.

## Completion criteria

- Python installs and evaluates Envelfos without NumPy, enge, or native bindings.
- Rust consumers can link and evaluate `envelfo-core` without Python or enge.
- Both implementations pass one shared runtime conformance suite; any remaining
  parity gaps are explicit and prevent claiming the corresponding feature done.
- uFor remains the format owner and does not import the runtime package.
- enge consumes the extracted runtimes without duplicate control semantics or
  lost audio/backend coverage and without a material performance regression.
- arpeg demonstrates shared-clock, expressive integration in Python and Rust.
- Old imports and replaced execution paths are removed, without aliases.
- Each library has documented APIs, dependencies, ownership, and release checks.

## Additional work beyond the prompt

None.
