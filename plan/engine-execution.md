# Remaining engine execution work

## Control integration

Addressed LFO and Envelfo playback actions already have offline and persistent
coverage, including snapshots, named envelopes and stages. Shared execution now
belongs to enveLFO; enge supplies audio observation and voice lifecycle policy.
Do not treat the original event-integration proposal as wholly unimplemented.

Persistent routes now accept piecewise linear, step and positive exponential
maps with endpoint clamping. Native prepared tables also cover Patch outputs;
LFO activation weights apply after mapping. Multiple routes can share a source
or target: additive contributions combine with the base before multiplication,
and domain validation applies to the composed value. Routes share generator
state, so marker events, playback commands and snapshots affect every route.
Offline and native route sums retain small terms under cancellation and resolve
halfway rounding using lower remainders. Native scratch is allocated when routes
are prepared, never during parameter evaluation. Regression coverage compares
scalar, block, persistent and native live output against scalar `fsum` results,
including snapshot replay.
Amplitude, named and staged envelope adapters now preserve authored curves in
NumPy/native block and persistent rendering. Regression coverage includes large
positive/negative curves, held segments, release, partitions and continuation.

## Sampler coverage

The offline NumPy/native sampler and its Python persistent adapter already
support control routes, generators, filters, pan laws and key/velocity bindings.
Static key/velocity crossfades now use onset-latched linear or equal-power gains
in offline and native live rendering, including zero-weight voice ownership and
snapshot continuation. Control-driven crossfades now work in offline NumPy/native
rendering, the Python persistent adapter and native live rendering, using
existing scoped control ramps.
Smoothing precedes the fade law, differing durations remain independent, and
snapshots retain ramps and voice context ownership. Native crossfades use prepared
position mappings and scalar fade laws with bounded storage. Coverage includes
delayed onset and the fade's amplitude ordering through fixed filter chains.
Fade retirement now works offline and in native live playback;
repeated requests retime continuously from the current fade gain. Earlier source or
envelope completion still ends the voice. Audit remaining restrictions
against current code before editing because separate work may land concurrently.

Signed alignment and resolved random offsets now work offline and in native live
playback, including snapshots. Random offsets follow playback direction; alignment
adds to the source index. Out-of-slice positions exhaust immediately without
clamping or changing slice/loop coordinates, as decided by the user.

Delayed starts now combine slot and resolved variation delays, rounding up to the
next output frame with traversal and envelopes beginning at age zero there.
uFor's explicit `trigger_release` observation cancels pending start-triggered
voices independently of sustain and one-shot retirement. Release-triggered
voices keep their own timer. Pending starts survive offline/native live snapshots
and consume bounded persistent voice capacity. Retirement before delayed onset,
including a fade choke, cancels the pending voice without sounding. At or after
onset, fade retirement applies the normal continuous ramp.

Native live sample voices now reuse the block renderer's multichannel filter
bank for fixed and control-routed filter chains. Each voice has preallocated
integrator state, reset on slot reuse and retained in snapshots. Slot filters precede instrument
filters; both amplitude orders are supported, without a tail after exhaustion.
Native live registration still rejects generators.
The user approved extending that sampler with the existing
native control/filter machinery on 2026-11-02, preserving bounded callback storage.
The generated runtimes' existing control ramps, scoped contexts and voice
bindings now live in a shared native `ControlBank`, including context retirement
and snapshot state. The native sampler now consumes that component through
`LiveRuntime.set_sample_controls`: common and slot amplitude/tuning routes retain
separate domains, support all three scopes and mapping modes, and share the
generated runtimes' precise route accumulation. Ramps advance during delayed
starts, trigger contexts retire with their voices, and snapshots retain ramps
and bindings. Direct native regressions cover interrupted fractional smoothing,
context initialization and partition-independent continuation. Python sampler
preparation now shares the generated-source modulation compiler and admits
authored amplitude/tuning controls. Context capacity is explicit and snapshots
retain Python address ownership alongside native state. Regressions cover common
and inherited slot routes with different smoothing, controls while silent,
overlapping parts, reused trigger names with release tails and capacity rejection.
Control-routed cutoff and Q now share the same parameter preparation and native
stage kernel. Common/slot filter domains remain independent, stage order is
preserved, and authored cutoff clamp/error policy runs after route composition.
Regressions cover all four responses, both amplitude orders, changing Q,
independent voices, reuse, exhaustion, snapshots and cutoff-boundary rejection.
Native sampler key/velocity event bindings now latch values in bounded per-voice
storage. They use the shared mapping/composition path, remain independent of
same-named controls, survive snapshots and reset on slot reuse. Coverage includes
delayed starts and event/control composition on filter Q. Sampler generators
remain work. Generated persistent synth, FM and noise renderers now admit
velocity bindings using the same native voice latch. Python velocity metadata
uses bounded trigger-context ownership, survives snapshots and is retired with
the context. Reused trigger names retain independent release-tail values.
Preset and repeated-trigger regressions cover offline/persistent/native-live
parity and exact partitioned snapshot replay. Key bindings remain outside the
generated engines' existing supported profile.

The generator reuse audit found that enge's Patch host and the pinned
`envelfo-core::Patch::advance` allocate during processing. Finite voice/context
capacity alone does not establish bounded callback storage for these generators.
The [callback audit](../doc/live-callback-audit.md) records the exact paths.
Extending the shared Rust scheduler with caller-owned processing storage needs
a cross-project API decision before native sampler generators can use it.

Finite loop `repeat_count` now counts additional traversals in both offline
backends and the native live sampler, including mirrored and crossfaded loops.
The remaining count survives snapshots, and exhaustion continues toward the
trimmed source boundary. Counts use unsigned 64-bit storage, validated during
preparation. Regression coverage includes zero repeats, fractional interpolation,
steps crossing multiple loop boundaries, release coinciding with a turn, and
release during an already committed overlap.

Whole-sample `play_count` now runs in both offline backends and native live
rendering, preserving voice state and remaining plays across snapshots.
Only the first traversal uses alignment/random offsets; subsequent traversals
restart at the trimmed boundary. Counts are bounded to 2**64 plays.

The completion audit also found an authored playback field that is not yet
consumed by the sampler: `end_fade_seconds`.
These remain required rendering work. The final-fade
contract leaves loop/repetition interactions unspecified; changing pitch also
needs an explicit timing decision before implementation.

## Dynamic filters

The complete per-voice filter chain defaults to `before_amplitude` and can use
`after_amplitude`. Amplitude includes the envelope, prepared gain, and changing
amplitude controls. Pan/channel routing follows both stages. Sampler slot/group
filters precede instrument filters and active chains must agree on their order.

NumPy and Rust renderers carry this choice through preparation and snapshots.
FM keeps envelope-scaled operator signals for modulation and feedback while
placing the audible carrier envelope and level in the chosen amplitude stage.
Source exhaustion or envelope completion discards filter state without an extra
tail.

## Live execution

The Rust LiveRuntime already owns mixing, supported effects, generated sources
and static sample playback. Python still performs action admission/submission;
the higher-level LiveEngine also has allocating orchestration paths. Native
hosts retain device ownership. See [the callback audit](../doc/live-callback-audit.md)
for the measured boundary; offline conformance is not device qualification.

## Additional work beyond the prompt

None.
