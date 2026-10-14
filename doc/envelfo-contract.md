# enveLFO extraction inventory and baseline contract

Inventory date: 2026-10-11. This completes the first slice of
[the extraction plan](../plan/envelfo.md), not the extraction itself.
Implementation anchors below refer to enge and its sibling uFor, enveLFO and arpeg
repositories. These are current capabilities, not promises of a standalone
Rust API.

Extraction update: scalar/Stages execution, stateless operator execution, random
words, and their unit tests now live in `envelfo/runtime.py`, `envelfo/random.py`
and enveLFO's `test/`. The matrix retains the inventory's original uFor test
names as evidence; runtime tests cited there have moved, while schema and trace
tests remain in uFor. Format-level shape, envelope and quantizer mathematics
remain in uFor where validation or other consumers require them. Patch
orchestration now also has a NumPy-free entry point in `envelfo.patch`. enge's
audio-host adapter shares its gate and retained signal evaluation, but retains
frame queues and cross-instance routing. The standalone scheduler is covered
by `test/test_patch.py` and `conformance/envelfo-patch.json` in enveLFO; replacing
the entire audio adapter and proving scheduler parity remain integration work.

Rust extraction update, 2026-10-12: the sibling `envelfo-core` crate now owns
checked rational time, exact scalar LFO state/commands, and the native random
word kernel. It passes the four LFO cases and random vectors directly from
the shared corpus, without a Python process. enge's native runtime imports that
kernel. Envelopes, Stages, markers, composition and native exact-time adapter
integration are still missing from the standalone Rust core; the matrix's
native column continues to describe enge, not completed standalone parity.

Rust extraction update, 2026-10-13: `envelfo-core` now also provides normalized
Envelope/Stages definitions and exact playback. Curved segments, current-level
release/retrigger and Stage entry capture, named-loop finite repeats, markers
and transitions pass direct Rust tests; all ten shared body traces now run in
both languages. Clone continuation and failed-update rollback are covered.
Hosts resolve Pint quantities and named loop bounds before entering the core;
seconds/beat positions use the same checked rational time type, without implicit
conversion. Rust Stage advancement returns ordered events and explicitly errors
above 4096 events per advance. enge pins the new core, but its existing native
audio adapter has not been replaced: its float event positions and curved-segment
preparation restriction still apply. Patch scheduling, host tempo maps, top-level
LFO marker emission, serialized Rust snapshots and bindings remain gaps.

Patch extraction update, 2026-10-13: the standalone Rust core now evaluates
materialized Patch graphs and their exact connection queues. Sum, Product,
Affine, Quantize, Latch, Threshold, Slew and SampleHold execute without Python.
Named outputs, LFO markers, Stage events/cues and start/sample/capture actions
use the existing ordering, division/offset/chance and 4096 capacity contracts.
Commands do not insert observations; Threshold edges use host-supplied delivery
times, and captures read the next observation after delivery. Failed calls
restore the entire prior state, including retained values and random streams.
`ClockMapping` accepts host seconds and monotonic elapsed local time; delays and
Slew stay in seconds even during stopped beat clocks. It does not implement a
tempo map or use seekable song positions as elapsed beats.

Four shared Patch traces now pass directly in Python and Rust, including
first-encountered graph dependency ordering and retained signal observations.
`conformance/envelfo-streams.json` in enveLFO checks portable SHA-256 stream keys
against independently computed vectors in both languages. enge's locked core
revision includes this scheduler, but its audio adapter still has separate
orchestration and floating-point native event positions. Consumer replacement,
built-in tempo maps, serialized Rust snapshots, bindings and combined arpeg
expression traces remain integration work.

Native adapter update, 2026-10-14: named Envelope playback and Stages now use
the standalone core directly. Their definitions and local clock observations
enter as exact ratios. LFO marker crossings, Stage outputs, routed cues,
connection delays and pending Envelope starts retain rational host timestamps
through queues and snapshots, without floating-point ordering or epsilon
adjustments. Fractional minimum-hold boundaries also remain exact when entering
named release. `test_native_named_envelope.py` anchors thirds of an audio frame,
non-binary delays and loops, stopped beat clocks and restored continuation.
Ordinary native LFO signal state and retained Patch operators/orchestration are
still separate adapters; this does not claim that the entire native scheduler
uses the standalone Patch runtime.

Native LFO adapter update: ordinary playback now also consumes shared Rust
`LFO`/`LFOState` for shapes, activation weight, commands and continuation.
The duplicate floating-point event state and modular phase/fade evaluator have
been removed. `LFO::next_marker` supplies exact routed crossings from that
same state and is also used by the standalone Rust Patch scheduler.
`test_native_lfo_core.py` covers decimal commands and paused resets, exact large
and negative transport positions, and pause-retimed markers across restoration.
Scope ownership and retained Patch orchestration remain host responsibilities.

## Capability and evidence matrix

Python reference means enveLFO scalar semantics unless the row explicitly names
enge orchestration. Native means execution inside enge's Rust extension;
an offline `backend="native"` test can still run control orchestration in Python.
Persistent renderer tests are stronger evidence of Rust-owned control state.
Neither kind proves an independent Rust library exists.

| Capability | Python implementation | Rust implementation today | Evidence and remaining gap |
| --- | --- | --- | --- |
| Definitions, units, library references | uFor `envelfo.py`, `library.py`, `segments.py`, `score_types.py` | Prepared tuples supplied by Python, not equivalent authored-document validation | uFor `test_envelfo_score_round_trips_simple_bodies`, `test_example_envelfo_library_resolves_all_scores`, `test_units.py`; normalized Rust definition boundary still to extract |
| Indefinite Envelope segments | Scalar and staged Envelope playback, explicit `advance` | Scalar/staged core and native Envelope cursors | Shared `indefinite-segments-advance-and-restore` and `indefinite-envelope-is-a-stage` traces; no separate constant generator |
| Voice-amplitude Envelope waits | Shared scalar state plus enge minimum-hold and completion policy | Shared Rust Envelope state per voice/operator; native sample state | `test_amplitude_envelope.py` covers attack/release waits, FM carrier/operator addressing, continuation and sample exhaustion; `test_envelope_policy.py` covers minimum hold and paused release |
| LFO signals and commands | enveLFO `initial_envelfo`, `envelfo_at`, `envelfo_event` | Shared Rust `LFO` and `LFOState`; enge owns scope and delivery cursors | `test_native_lfo_core.py` and existing scoped/playback audio tests; no duplicate native persistent phase/fade evaluator; stateless block acceleration remains an audio adapter |
| Envelope attacks, release, loops, reversal, finite repeats | enveLFO scalar Envelope execution | Shared Rust `EnvelopeState`; enge owns queued voice release and routes | enge `test_finite_envelope_repeats_match_persistent_native` and `test_native_named_envelope.py`; no duplicate native named Envelope evaluator |
| Curved Envelope segments | Shared scalar curve mathematics used by Envelopes and Stages | Standalone core supports curves; enge named/staged preparation still rejects them | enge `_persistent_modulation` and `_runtime_staged_envelfo`; removing audio-host restrictions is separate work |
| Stage transition events and entry capture | enveLFO `advance_envelfo`, `next_stage_event` | Shared Rust `Stages` and `StageState`; exact routed timestamps | enge `test_staged_envelope_events_cue_another_envelfo_in_native_runtime` and fractional routed delivery tests; native Stage playback/boundary helpers removed |
| Patch named signal/event outputs | Definitions/ranges/order in uFor; execution and child materialization in enge | Prepared transform nodes, children and connection tables in persistent runtime | enge `test_patch_named_outputs_match_separate_envelfos`, `test_patch_named_event_output_cues_stage`; `initial_envelfo` does not evaluate a Patch independently |
| Sum, Product, Affine, Quantize | uFor stateless helpers; enge graph execution | `envelfo_transform_values` evaluation | uFor `test_patch_transforms_follow_dependencies_without_clipping`, `test_quantize_half_steps_choose_the_higher_value`; enge `test_quantize_grid_matches_across_clocks_partitions_and_snapshots` |
| SampleHold | uFor `SampleHoldState`, explicit sample-event draws; enge owns per-voice seed derivation | SampleHold lowered through named envelope state; SplitMix64 implementation | uFor `test_sample_hold_draws_only_on_sample_events_and_restores_its_stream`; enge `test_sample_hold_is_shared_by_reference_and_persistent_fm_and_noise` |
| Latch, Threshold, Slew | uFor scalar helpers; enge owns observation grid, retained graph state and resulting deliveries | Persistent transform state and connection processing | enge `test_latch_activation_and_same_frame_captures`, `test_threshold_gate_and_next_frame_commands_survive_partitions_and_snapshots`, `test_slew_shared_state_limits_rates_and_restores_mid_ramp`; independent Python graph execution still missing |
| Event division, offset, chance, delayed delivery | uFor connection validation; enge `ControlRenderer` event queues and counters | Prepared connection tables, `forward_event`, delayed-event queues | enge `test_patch_division_is_first_aligned_independent_and_snapshot_safe`, `test_sample_hold_events_are_seeded_independent_delayed_and_snapshot_safe`, `test_delayed_envelfo_events_report_queue_and_clock_limits` |
| Beats, tempo, stop, seek, quantized commands | uFor `TempoMap` and trace preparation; enge converts host time to the instance clock | Prepared `BeatPoint` observations; native playback between host points | uFor `test_synth_trace_cancels_pending_envelfo_command_on_transport_seek`; enge `test_beat_stages_follow_tempo_and_stop_across_restore`, `test_host_tempo_quantizes_envelfo_change_for_both_renderers`; clock acquisition is not part of either core |
| Direct transport-position LFO | uFor `PositionDriver.transport`, `envelfo_at` | Native LFO transport-position flag | enge `test_transport_position_cycle_follows_seek_without_crossing_history`; only beat-clock LFOs with transport reset and no delay/fade are allowed |
| Scope, trigger addressing and release ownership | uFor trace contracts; enge instance admission/routing/lifecycle | Voice/context slots and prepared source owners | enge `test_trigger_addressed_envelfo_changes_only_matching_voice`, `test_trigger_envelfo_playback_matches_persistent_fm_and_noise`; host ownership remains outside the control core |
| Snapshots and continuation | uFor scalar state JSON; enge aggregate snapshots | Cloned opaque `SynthRuntimeSnapshot` | enge `test_trigger_envelfo_playback_matches_persistent_native` and operator snapshot tests; no shared Rust/Python snapshot wire format exists |
| Event-free block acceleration | enge `_envelfo_span`, `lfo_samples`, `_envelope_values`; scalar fallback for unsupported spans | Native block LFO/audio kernels and persistent frame processing | enge `test_staged_cycle_fade_remains_exact_across_single_frame_partitions`, `test_mixed_staged_cycle_and_looping_envelope_keep_scalar_boundary_semantics`; acceleration is not the reference contract |
| arpeg control/event interoperation | arpeg runtime and explicit ports, not yet an enveLFO dependency | `arpeg-core` `ports.rs`, exact `Beat = Ratio<i64>` | Rust `InputPort` includes expression names, but `check_control` rejects breath/bend/pressure with "expression controls require the MIDI player"; standalone expression integration is unfinished |

## Exact current reference boundary

These public functions describe existing behavior; this slice adds no new
runtime API:

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

The existing Rust runtime has `f64` beat points, playback positions and delayed
event times, and snaps near-integer staged event times using a tolerance. This
is still an exact-event-time native integration gap. The approved standalone
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

The standalone graph evaluator must expose or consume the logical observation
schedule rather than baking an audio rate into the core. Its concrete API is
step 2/3 work; no new default grid or continuous-crossing detector is selected
here. Ordinary LFO control interpolation is also a host optimization, not the
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

| Keep in uFor | Move into enveLFO | Keep in enge/adapters |
| --- | --- | --- |
| Definitions, score registration, unit validation, reference materialization, structural port/range checks, portable event/state records | Initialization, body/state advancement, command execution, boundary scheduling, retained graph state, random draws, event connection execution | Voice/context allocation, trigger-to-instance addressing, minimum-hold release policy, audio modulation mapping, output buffers and renderer snapshots |
| Format mathematics needed by validation, including quantizer range calculations and declared shape/time helpers | Pure graph orchestration currently embedded in `ControlRenderer`, separated from its frame/audio adapter | NumPy block acceleration and native/PyO3 conversion |
| Portable stream identity construction used by traces | Independent Rust control types and equivalent execution, with a fixture reader for conformance | Aggregation of audio and control states at the consumer boundary |

Each algorithm must have one implementation per language after extraction.
Do not leave uFor evaluation copies or a second enge native implementation.
Definition helpers needed for validation stay dependency-safe and are reused,
not duplicated under new names. Portable state records remaining in uFor do not
authorize uFor to depend on their executing library.

## Baseline fixtures

`conformance/envelfo-runtime.json` and `test/test_envelfo_contract.py` establish
the initial language-neutral runtime baseline. They now live in the sibling
enveLFO repository, together with the extracted scalar/Stages runtime and its
execution tests. uFor retains the authored definitions and validation; enge
depends on enveLFO and imports execution functions directly.

The JSON uses the current uFor definition representation with canonical unit
literals, including explicit `s` segment suffixes. Missing definition fields use
the existing uFor defaults. Each case starts at zero with seed zero. Step times
are rational strings in the definition's clock; events carry their exact time
and ordinal. Expected values/weights have absolute tolerance `1e-12`, relative
tolerance zero. Status, event fields/order and selected runtime fields compare
exactly. Random states/words use hexadecimal strings to avoid JSON number
precision loss in other readers.

`restore` requests a same-implementation snapshot round trip before the next
observation. Steps with no event observe the result; Stages retain the returned
advance state and events, whereas ordinary scalar bodies use their current
query/command API. The partition test repeats each staged trace without its
intermediate queries and compares final state and ordered events.

The runtime traces anchor constant indefinite Envelopes, triangle quarter-turns, both fade contracts,
phase-preserving rate/pause/resume, attack/hold/release, same-time release
precedence, exact fractional markers, final named-loop event ordering, and
direct song-position queries across a backward jump. Three existing portable
SplitMix64 vectors anchor randomness. Expectations are handwritten from these
simple mathematical/ordering rules, not generated from either backend.

The runner uses only pytest, standard-library code, uFor and enveLFO. Run it
in enveLFO's environment, without NumPy or enge installed:

```sh
# From the sibling envelfo checkout:
uv run pytest test/test_envelfo_contract.py
```

This verifies the baseline reference semantics and the new package's dependency
boundary without NumPy or an enge import. The future
Rust runner must read the same cases directly; the current native audio tests
remain supporting evidence, not a shared standalone conformance runner.

Four shared Patch traces now anchor exact LFO-to-Envelope starts, explicit
Threshold delivery, retained signal observations and graph ordering, without
importing enge or NumPy. Standalone unit cases also
exercise staged cues, elapsed beat clocks, stop/seek maps, seconds delays,
division/offset/chance, retained observations, bounded queues, and restoration.
The host must supply matching observations and random stream identities when
comparing standalone and audio-host results. Matching a different host's
flattened connection index is not assumed.

Pending shared additions are broader Patch/connection and observation traces,
curved envelopes, mapped tempo/quantized transport,
capacity/error cases and arpeg combined expression traces. Existing tests above
remain in place; this initial baseline does not replace their coverage or claim
those extraction gaps are resolved.

## Next slice and decisions

Proceed to the Python extraction with the dependency direction and ownership
above. Before choosing its public graph scheduling API, review the existing
logical observation-grid and same-time delivery phases. Before Rust extraction,
settle checked rational ranges and verify the curved-segment and event-time gaps.
Before arpeg expression integration, expose the existing player-owned expression
path at an appropriate library boundary; do not assume enum names mean that
`arpeg-core` already accepts those controls.

No new behavioral decision is required to freeze this existing baseline.

## Additional work beyond the prompt

None.
