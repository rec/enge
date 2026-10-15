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
Python package, `envelfo-core` for the Rust crate, and `EnveLFO` as the shared
definition base.
The recent breaking rename remains canonical. Do not introduce old-name aliases
or backward readers during extraction.

## Original inventory starting point, 2026-10-11

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

- LFO shapes, phase, delay, fade, reset, and signal ranges.
- Envelope segments, release timing, loops, direction, finite repeats, and
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
envelope traversal completion, preserve boundary-before-done ordering and the
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

1. Constant indefinite Envelope, ordinary LFO, and attack/release Envelope common cases.
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

- arpeg emits a hit event that triggers an enveLFO envelope.
- A separate enveLFO LFO supplies a named expression output for played notes.
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

The Envelope model now supports indefinite segments in place of a separate
constant generator. `duration: null` holds the incoming value, and `advance`
leaves the currently reached hold. Note-off enters release directly. These
Envelopes use segment-local elapsed coordinates, not normalized total positions;
normalized seek/shift, markers and repeated playback are rejected. Finite
Envelope behavior remains unchanged. Python/Rust continuation state retains the
segment cursor. Enge's voice-amplitude profile now supports these waits too,
using the shared scalar Python and Rust Envelope core. Host lifetime policy
keeps minimum hold, source exhaustion and retirement in enge. Commands use
`target="envelope"` with an optional FM `op`; default and explicit carrier
addresses share one state. Unbounded releases retain the allocated voice until
advanced to completion or forcibly retired.

Step 1 is complete as of 2026-10-11: see the
[inventory and baseline contract](../doc/envelfo-contract.md), the shared
`conformance/envelfo-runtime.json` traces, and `test/test_envelfo_contract.py`,
now owned by the sibling enveLFO repository.

Step 2 is in progress: the NumPy-free Python package now owns scalar and Stages
execution, operator helpers, random words, 48 moved execution tests, and the
18 baseline conformance checks. uFor retains definitions, validation, format
mathematics and stream identities. enge imports the extracted runtime directly.
Standalone Patch execution is now available in `envelfo.patch`: named outputs,
retained operators, exact natural boundaries, start/sample/capture/cue connections,
division/chance gates, seconds delays, host tempo maps and snapshots. Its host
boundary uses explicit observation and next-delivery times, not an audio-rate
grid. Two additional shared Patch traces and 27 focused unit cases cover that
boundary. enge reuses the connection gate and retained signal evaluator while
keeping its frame queue, cross-instance routing and NumPy block acceleration.
Replacing that entire audio-host adapter with the standalone scheduler remains
consumer integration work, not completed parity.

Step 3 started on 2026-10-12. `crates/envelfo-core` now provides checked signed
64-bit rational components, exact LFO playback/commands and portable random
words, independently of Python, NumPy and enge. Arithmetic reduces i128
intermediates before checking the final i64 components and returns explicit
overflow errors; failed LFO commands preserve the original state. Four shared
LFO traces and the portable random vectors run directly in Rust, alongside
boundary/overflow tests in debug and optimized builds. enge consumes the extracted
random-word kernel rather than retaining a duplicate. Native playback still uses
floating-point event positions. Envelopes, Stages, marker emission, Patch
operations/scheduling, host tempo maps and bindings remain Rust extraction work;
the full parity matrix is not complete.

Step 3 continued on 2026-10-13. The standalone Rust core now owns Envelope and
Stages playback, including zero-duration and curved segments, release capture,
retrigger policies, pause/reverse/seek/shift, named loops, finite-repeat crossing
counts, captured Stage entry levels, exact marker times and ordered transitions.
All ten shared runtime traces now execute directly in Rust. Debug and optimized
tests also cover irregular advance partitions, cloned continuation, reversal,
same-time preemption, exact ordering near float precision limits and transactional
rollback on overflow or the 4096-event capacity limit. enge pins this revision,
but its native audio adapter still uses the existing floating-point Envelope and
Stage state. Standalone Patch operations/scheduling, top-level LFO marker
emission, host tempo maps, serialized Rust snapshots, bindings and exact native
adapter integration remain unfinished. No new musical policy decisions were
needed for this slice.

Step 3 continued with Patch orchestration on 2026-10-13. The standalone Rust
core now executes materialized Patch graphs, all seven signal operators,
SampleHold children and LFO markers, named signal/event outputs, Stage cues,
start/sample/capture connections and delayed deliveries. Exact scheduling keeps
external commands, natural boundaries, zero-delay cascades and host observations
in their established phases. Division/offset/chance retain independent stream
identities and draw only for eligible nonconstant probability gates. Rust uses
SHA-256 for the same identity bytes as uFor, and shared identity vectors check
both implementations. Signal topological ordering preserves Python's
first-encountered prerequisites, not a sorted-name order.

Hosts supply monotonic local-clock mappings and inverse boundary times through
the Rust `ClockMapping` contract. Observations, explicit next deliveries and
connection delays remain in seconds; Slew advances while a beat clock is stopped,
and externally commanded events retain their host timestamp. This provides
beat-clock orchestration without adding a built-in tempo-map implementation.
All ten body traces and four Patch traces pass directly in both languages.
Rust tests cover random gates, delayed/cascaded cues, irregular partitions,
retained observation state, bounded queues, overflow and full-call rollback.
enge pins the extracted scheduler revision but does not yet execute it in the
audio adapter. Consumer replacement, built-in tempo maps, serialized Rust
snapshots, bindings and arpeg integration remain unfinished. No new musical
policy decision was required.

Step 4 continued on 2026-10-14. Native named Envelope playback now owns
`envelfo-core::envelope::EnvelopeState`, rather than a second playback/repeat
implementation. Making fractional routed cues usable with checked rational
time also required replacing native Stages playback and natural boundaries
with `envelfo-core::stages`. Both adapters retain enge's voice and route owners.
Prepared durations, loop bounds, marker positions, tempo-map observations,
minimum-hold boundaries and connection delays now enter as exact ratios.
Routed LFO marker crossings, Stage events, delayed queues and Envelope starts
keep exact timestamps through ordering and snapshot continuation. No epsilon
or rounding repairs event times. Numeric command payloads still enter through
the existing floating-point action protocol and are checked when converted.

Tests cover one-third-frame Stage and LFO cues, fractional delayed delivery
before and after restoration, rational named loops across stopped beat clocks,
and voice-timed release at a fractional minimum-hold boundary. The existing
linear preparation restrictions remain; this slice does not enable curved
audio-host Envelopes. Ordinary native LFO signal playback and retained Patch
operators/orchestration still use enge adapters, so the full scheduler replacement
and removal of all duplicated execution are not finished.

Step 4 continued with native LFO playback. enge now stores the shared Rust
`LFOState` for instrument/part and voice-owned clocks, using core waveform,
activation-weight and command evaluation instead of its separate phase and
fade implementations. Voice starts initialize their owning clock; instrument
and part signals retain the existing absolute clock. Transport-position LFOs
read exact song beats, including negative positions and integers beyond float
precision. Snapshot continuation retains core state and host delivery cursors.

The core's non-mutating `LFO::next_marker` query is shared by standalone Patch
scheduling and native marker routing. Pause, resume, reversal and phase changes
therefore affect routed crossings through the same state used for signals.
Numeric LFO command payloads use their decimal value, matching Python
`Fraction(str(value))`, with checked rational overflow rather than binary-float
denominator expansion. Existing Patch-level command restrictions remain.
At that stage, native retained Patch operators/orchestration and Python host
scheduler replacement remained unfinished.

Step 4 continued on 2026-10-15 with native Patch orchestration and connection
scheduling. Each native voice now owns one shared `PatchState`, including child
playback, retained signals, connection gates, random draws and delayed commands.
The old native transform evaluator, event counters, marker dispatcher, delayed
queues and deferred start dispatcher are removed. Prepared source owners preserve
shared children and named outputs. Prepared random-stream keys preserve existing
sample and probability sequences when independently named graphs are flattened.

The core clock interface now addresses each child, allowing exact seconds,
elapsed beats and signal-only transport positions in one scheduler. enge maps
host tempo observations, supplies instrument/part-owned signals, and retains
voice admission, minimum hold, release timing and retirement. Same-time commands
are bracketed by an exclusive event phase and one inclusive audio observation;
Threshold retains next-frame delivery, and Slew measures host seconds even while
beats are stopped. Aggregate snapshots clone core state and pending host release
policy. Cross-clock audio regressions cover fractional delayed cues, probability,
stopped clocks, song seeks and restoration into a fresh native renderer.

At that stage, Python audio-host scheduler replacement, serialized Rust snapshots
and the combined arpeg integration remained unfinished.

Step 4 continued with the Python audio-host scheduler migration. enge now
materializes its bindings, inherited Patch children, operator dependencies and
cross-instance connections into `envelfo.patch.PreparedPatch`. It preserves each
child clock, scope ownership and prepared random-stream identity. Shared
`PatchState` owns voice-local playback, retained values, gates and exact delayed
deliveries. The former Python connection dispatchers, per-source event queues,
gate counters and Patch signal evaluator have been removed.

The shared scheduler exposes explicit command phases, event-only settling and
non-mutating next-boundary queries. enge retains its event-free audio sampling
adapters; stateless block signals are evaluated by the standalone library without
NumPy. Retained operators use the host's individual observation schedule.
Instrument/part-owned signals and minimum-hold release policy remain in enge.
Synth, FM, noise and sampler snapshots now save shared state under `graphs`;
old per-source snapshot records are not read through compatibility adapters.

Regressions cover saved JSON with pending captures, original snapshot isolation,
seconds/beat clocks, stopped transport, fractional events, random streams and
all existing audio backends. Serialized Rust snapshots remain unfinished,
along with the final extraction/release audit.

Step 5 now has an optional Python `arpeg.envelfo` bridge and separate
`arpeg-envelfo` Rust crate. Both connect arpeg hits to a standalone event-started
Envelope and named Patch outputs to existing expression lanes. Hosts supply
exact timestamps, per-child tempo mappings and observation/delivery boundaries;
neither core depends on the other or acquires its own clock. One shared wire
trace checks controllers between notes, irregular advancement, tempo changes,
stop/start, seeking, final release and explicit lane ownership. Pending Patch
deliveries survive Python JSON restoration and Rust bridge cloning. Python
also restores the player alongside the bridge. A serialized combined Rust
host/player continuation format has not been introduced.

The first integration deliberately returns hit-triggered expression after
arpeg's note messages at the same timestamp. Revisit this ordering: instruments
that latch expression at note-on may need observation before note realization,
which would require separating those phases in arpeg. This is documented in
arpeg's `doc/envelfo.md` and covered by its composition fixture, rather than
claimed to provide pre-note-on expression already. Feedback and additional
selection/rhythm mappings are outside this first integration.

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
5. **Integrate arpeg.** Implement the hit/envelope/expression example with shared
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

Final extraction audit, 2026-10-15: the standalone implementations and the enge
and arpeg integrations are complete under the accepted host/continuation
contracts. The current capability matrix is in enveLFO's
`doc/extraction-audit.md`; enge's `doc/envelfo-contract.md` records its host boundary.
Earlier dated progress notes above are history, not current gap lists.

The audit found and removed a remaining legacy `ufor.lfo` execution path.
Lighting now uses the canonical LFO definition and stateless shared mathematics;
enge's block helper consumes canonical state directly, without old-model
conversion. The public legacy model/state/event/value classes and execution
functions have no aliases. uFor's generated schema now contains one LFO model.

Additional shared host traces cover independently clocked children, elapsed
beats across stops and transport seeks, direct song-position LFOs, delayed
capture during stopped transport, seconds-based Slew, division/offset sampling,
curved release and retained/pending state across continuation. Both languages
read the same expectations. Rust's body runner visits every shared body case,
instead of naming individual LFO cases and silently missing future additions.

Distribution verification includes Python source/wheel builds, tests against
an installed wheel outside the checkout without NumPy or enge, and standalone
Rust package compilation with the MIT notices included. A release-only workflow
gates Linux/Windows/macOS, Python 3.13/3.15 and Rust debug/release builds. It is
not run per commit, and cross-platform release results are not claimed now.

Rust JSON snapshot follow-up, 2026-10-15: the optional `snapshots` feature now
bundles normalized LFO, Envelope, Stages, SampleHold and Patch definitions with
complete continuation state. Version 1 preserves exact rational times, 64-bit
counters/random words and signal floats. Save/load validates definitions, child
types, indexes, graph order and pending deliveries. Unknown versions are rejected
without migration. The shared Rust conformance traces serialize and restore at
every step when enabled; default builds remain free of serialization dependencies.
Release-only checks cover both feature configurations.

Clock mappings, external queues and combined enge/arpeg application snapshots
remain host-owned. Optional standalone Python bindings remain a separate
feature. Host-supplied clock mapping is the intended boundary, not
a missing built-in tempo implementation.

arpeg expression-ordering follow-up: both Python and Rust players now separate
scheduling from MIDI realization. Hosts deliver all same-time hits, observe once,
then realize fresh expression before note-on. Ordinary playback uses that same
path; note-offs, captured/current lane ownership and exact beat timestamps are
preserved. Python snapshots support the intermediate phase, and stopping cancels
unsent attacks while releasing only realized notes. Shared wire traces exercise
these phases, including tempo changes, overlap and captured phrases. Combined
Rust application snapshots remain host-owned work. No tag or publication is part
of this change.

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
