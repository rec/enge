# enveLFO extraction contract and closeout evidence

Updated 2026-10-15 after the final extraction audit. The original inventory and
implementation history are retained in [the plan](../plan/envelfo.md).
The standalone capability matrix, distribution checks and explicit exclusions
are maintained in [enveLFO's audit](https://github.com/rec/envelfo/blob/main/doc/extraction-audit.md).

## Current ownership and evidence matrix

Python and Rust below mean the independent control libraries, not an audio
backend that happens to use Python orchestration. Consumer tests supplement,
rather than replace, shared standalone conformance.

| Capability | Python owner | Rust owner | Evidence / host boundary |
| --- | --- | --- | --- |
| Definitions, units and references | uFor canonical EnveLFO models | Host-normalized native definitions | uFor schema/codec/unit tests; Rust validates normalized definitions |
| LFO playback, rate, reset, phase and onset | enveLFO runtime | Core LFO state | Shared body traces and LFO tests; enge retains event-free sampling acceleration |
| Finite/indefinite Envelopes, curved release and repeats | enveLFO runtime | Core Envelope state | Shared runtime/host traces and focused Envelope tests |
| Stage transitions, cue events and entry capture | enveLFO runtime | Core Stages state | Shared body traces, command-preemption and partition tests |
| Patch outputs, signal operators and retained state | Standalone Patch scheduler | Core Patch/signal modules | Shared Patch/host traces and operator tests |
| Division, offset, chance, random words and delayed queues | Standalone Patch/random modules | Core Patch/random modules | Shared portable vectors and host traces; focused gate/cascade/capacity rollback tests |
| Per-child clocks and elapsed beats versus song position | Host TempoMap supplied to Patch | Host ClockMapping supplied to Patch | Shared stop/seek/tempo trace; frame conversion and clock acquisition stay in hosts |
| Continuation and pending deliveries | JSON snapshots | Independent state clones | Shared restore points; enge snapshot isolation and exact fractional delivery tests |
| Voice scope, triggers, minimum hold and retirement | enge | enge | Host policy, not duplicate Envelope playback |
| Audio integration | enge sampling and shared Python Patch state | Native adapters over shared core state | Full enge suite and Python/native Patch regressions |
| arpeg expression integration | Optional arpeg bridge | Separate arpeg-envelfo crate | Shared composition MIDI trace, mapped clocks and pending-delivery continuation |

The legacy low-level `ufor.lfo.LFO` and its separate event/state/value records and
initializer/query/command functions have been removed, with no aliases. enge
block sampling consumes canonical LFO definitions and EnvelfoState directly.
Lighting also uses that definition; stateless shape and activation mathematics
remain in uFor without a runtime dependency cycle.

enge's named Envelope, LFO, Stages and retained Patch scheduling all consume
the extracted library in both languages. NumPy spans, audio interpolation and
native waveform kernels remain acceleration. enge still restricts amplitude
profiles and curved native preparation; those are explicit host restrictions,
not missing standalone Envelope capabilities.

## Exact current reference boundary

These public functions form the current scalar boundary:

- `initial_envelfo(use, at, seed=0)` initializes LFO, Envelope, Stages, or
  SampleHold. References must already be materialized. Patch initialization
uses the separate standalone `envelfo.patch` orchestration API.
- `envelfo_at(use, state, at)` observes a scalar result. For Stages it computes
  an advance internally but returns only the value: a caller that needs events
  and continuation must retain the result of `advance_envelfo` instead.
- `envelfo_event(use, state, event)` applies an ordered command and returns state.
  For Stages this delegates to advancement but does not return emitted events.
- `advance_envelfo(use, state, at, event=None)` currently accepts only Stages and
  returns state, value and ordered events. It is not a general graph evaluator.
- `next_stage_event` is a nonmutating natural-boundary query. Consumers must
  still perform advancement to obtain state transitions and emitted events.

Supported command sets are body-specific. For example, LFO rejects cue/sample,
Envelope rejects cue/reset/transport/rate/sample, and transport-position LFO
rejects phase commands. Do not silently add support or manufacture empty event
lists for features that have no standalone event-emission API today.

### Time and ordering

Python boundary times and positions use `Fraction`, with explicit seconds or
beats on the owning use. The same rational number in different clocks is not
interchangeable. Integer audio frames and host conversion remain adapter inputs,
not an implicit replacement for these clocks.

Elapsed-state queries cannot go back before their saved anchor. Commands must
increase in `(at, ordinal)` order; separate ordinals allow multiple commands at
one time. Direct transport-position LFO queries are different: `at` is song
position, so a backward song seek can be observed without replaying history.

For staged advancement with a command, natural events strictly before its time
are processed first, then the command. A command that changes stage at a natural
boundary can discard the obsolete old-stage completion. Without such a change,
eligible same-time natural boundaries are processed afterwards.

Natural markers at a named loop boundary precede `cycle`/`turned`. Named
forward loop wraps emit both start and end markers at the wrap time in their
existing marker order, including the final crossing. Ping-pong turns do not
invent the opposite endpoint marker. These rules are already covered by uFor's
`test_named_envelope_loop_boundaries_preserve_markers_and_events`.

Finite repeat counts count boundary crossings, including reversed crossings.
The final crossing emits its boundary events then `stage.done`; an explicit finish
transition additionally emits `done`. Without a finish transition, completion
still holds the boundary value but does not invent a `done` event.

At inventory time the Rust runtime had `f64` beat points, playback positions and
delayed event times, and snapped near-integer staged event times using a tolerance.
The native migration has removed that integration gap. The standalone
boundary uses reduced signed i64 numerators and positive i64 denominators,
with i128 intermediates and explicit errors when normalized components do not
fit. It supports checked conversion to/from arpeg's `Ratio<i64>`; it does not
promise arbitrary Python Fraction range. Command overflow leaves the supplied
LFO state unchanged. Floating-point conversion is limited to signal values,
not comparisons, phase wraps or position advancement. Rust snapshots currently
use cloned state, not a shared serialized wire format.

### Signals and observation schedules

Results contain a floating `value`, activation `weight`, and exact status tag
(`idle`, `running`, `held`, `complete`). The current scalar LFO returns its
unweighted centered waveform and reports delay/fade through `weight`. A staged
LFO applies fade to depth inside its value, preserves the center, and reports
`weight=1`. These are distinct existing observable contracts. Do not normalize
them during extraction or apply the fade twice in adapters.

Signal graphs are acyclic and evaluated in dependency order. Sum/Product/Affine
do not implicitly clip to a modulation domain. Quantize half-step ties select
the higher grid point. Patch signals remain dimensionless.

Latch, Threshold and Slew have observable state on a supplied observation grid.
Threshold initializes silently and holds through its deadband. enge schedules
resulting routed commands on the following frame; Slew receives the elapsed
time per observation. Partition independence means identical logical
observations and ordered inputs split across different calls. It does not mean
changing that observation schedule leaves these operators unchanged.

The standalone graph evaluator consumes explicit host observations and
next-delivery times through its Patch API, without an implicit audio grid or
continuous-crossing detector. Ordinary LFO control interpolation is also a host optimization, not the
definition of exact LFO/Staged LFO observations.

### Events, randomness and state

Keep connection declaration order, every/offset counters, random streams and
pending deliveries as state. Division happens before chance testing; probability
zero/one does not consume a random draw. Number source occurrences from zero:
an occurrence is eligible when `index >= offset` and
`(index - offset) % every == 0`. Thus `every=3, offset=0` selects musician-facing
occurrences 1, 4, 7; `offset=2` selects 3, 6, 9, and offsets larger than the
division are an initial skip, not reduced modulo that division. Chance compares
the upper 53 random bits against `int(probability * 2**53)` after division.

SplitMix64 advances modulo `2**64`. Signal draws use the upper 53 bits. The
existing stream-key byte prefix is `ufor-envelfo-v1` followed by a zero byte;
renaming the package must not change it or the seed/identity encoding.
uFor traces already call `stream_key` to assign voice keys, so that portable
identity helper must remain usable by uFor without importing the runtime.
Random draw execution can move to enveLFO; do not move all seed derivation behind
an `envelfo -> ufor -> envelfo` dependency cycle.

Preserve current staged event and host connection/queue capacity errors
(4096 limits at their existing enforcement points). These are not a new global
graph-size guarantee. Unsupported scope, curves, or Patch-level commands remain
explicit errors until a separately tested implementation supports them.

JSON restoration of Python scalar state and restoration of native aggregate
state are distinct existing contracts. Preserve equivalent continuation in
each. Cross-language snapshot interchange is not required by this baseline.

## Extraction ownership map

| uFor | enveLFO | enge/adapters |
| --- | --- | --- |
| Definitions, score registration, unit validation, reference materialization, structural port/range checks, portable event/state records | Initialization, body/state advancement, command execution, boundary scheduling, retained graph state, random draws, event connection execution | Voice/context allocation, trigger-to-instance addressing, minimum-hold release policy, audio modulation mapping, output buffers and renderer snapshots |
| Format mathematics needed by validation, including quantizer range calculations and declared shape/time helpers | Pure graph orchestration shared by ControlRenderer and standalone hosts | NumPy block acceleration and native/PyO3 conversion |
| Portable stream identity construction used by traces | Independent Rust control types and equivalent execution, with a fixture reader for conformance | Aggregation of audio and control states at the consumer boundary |

Each algorithm must have one implementation per language after extraction.
Do not leave uFor evaluation copies or a second enge native implementation.
Definition helpers needed for validation stay dependency-safe and are reused,
not duplicated under new names. Portable state records remaining in uFor do not
authorize uFor to depend on their executing library.

## Shared fixtures and verification

The sibling enveLFO repository owns `envelfo-runtime.json`,
`envelfo-patch.json`, `envelfo-host.json` and `envelfo-streams.json`.
Both standalone implementations read them directly. Exact times, events,
pending delivery boundaries and random words compare exactly; signal values
use absolute tolerance `1e-12` and zero relative tolerance. Expectations are
handwritten from simple waveform/time/order rules, not generated by one backend.

The host corpus covers mixed clocks, running versus stopped beats, forward and
backward song seeks, direct song-position signals, delayed capture while
transport is stopped, seconds-based Slew, division/offset sampling, curved
Envelope commands, and pending deliveries across continuation. The Rust fixture
host integrates supplied observations only; it is not a new core tempo API.

Python restore steps serialize and reload state; Rust restore steps clone it.
Cross-language snapshot interchange is not required. arpeg owns its composed
expression wire trace and verifies its explicit lane ownership and same-time
delivery phases independently of enge.

The audit verifies clean wheel installation outside the source tree without
NumPy/enge, source-distribution and standalone crate builds, Python 3.13/3.15
on macOS, Rust debug/release tests and consumer suites. Cross-platform checks
remain release-only; configuring them is not evidence that a release has passed.
No release or hardware validation is implied.

## Separately scoped follow-ups

Serialized Rust disk snapshots, standalone Python bindings to Rust, and arpeg
expression observation before note-on are not extraction blockers under the
accepted continuation and host-boundary contracts. arpeg deliberately returns
newly triggered expression after note messages at the same timestamp; revisit
that ordering explicitly before promising support for instruments that latch
expression at note-on. Built-in Rust tempo acquisition and broader audio-host
Envelope profiles are not silently added here.

## Additional work beyond the prompt

None.
