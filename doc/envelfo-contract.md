# enveLFO extraction contract and closeout evidence

Updated 2026-11-17 for native uFor generator preparation. The original inventory and
implementation history are retained in [the plan](https://github.com/rec/enge/blob/main/plan/envelfo.md).
The standalone capability matrix, distribution checks and explicit exclusions
are maintained in [enveLFO's audit](https://github.com/rec/envelfo/blob/main/doc/extraction-audit.md).

## Current ownership and evidence matrix

Captured `exponential_approach` stages now use the same ownership boundary:
uFor owns the stage definition and nanosecond resolution contract; enveLFO's
Python and Rust evaluators capture the phase entry value and retain resolved
curved/linear segments in continuation state. enge's offline block adapters
consume that resolved state, and native named-generator preparation carries
the authored stage to the shared Rust evaluator. Frame-clock conversion follows
rounding in seconds. Existing interpolation kernels are reused.

The supported host profile is named voice modulation Envelopes with one-shot
seconds clocks. Main voice-amplitude releases now also support captured stages
through the same shared state and native amplitude lifetime. Captured onset
stages remain rejected; markers, repeats,
indefinite holds and nesting in Stages are rejected by the shared model.
`test/test_captured_filter_envelope.py` checks delay/attack/sustain interruption,
independent gain equations, both offline backends, JSON continuation and native
persistent continuation with two-second 48 kHz WAV output.

Main amplitude commands reserve resolved-stage and command continuation storage
during preparation. Voice reuse resets that prepared state without allocating;
captured release commands use enveLFO's shared prepared evaluator. The focused
`test/test_amplitude_approach.py` suite covers synth, sampler, FM and noise,
interrupted onset phases, zero and minimum positive time constants, snapshots
and block partitioning. Its allocation-audit run recorded 100 successful native
callbacks with zero allocations, reallocations or frees. This evidence applies
to main amplitude commands, not every named-generator or Patch command path.
The integration suite reports 3,505 passed and 29 skipped. After rebasing onto
the concurrent de-esser commit, its four new tests and the 18 release tests
also pass together with callback allocation auditing enabled. Rust compilation,
typing, formatting and the selected Ruff checks pass.

Python and Rust below mean the independent control libraries, not an audio
backend that happens to use Python orchestration. Consumer tests supplement,
rather than replace, shared standalone conformance.

| Capability | Python owner | Rust owner | Evidence / host boundary |
| --- | --- | --- | --- |
| Definitions, units and references | uFor canonical EnveLFO models | Rust uFor validates generator and binding declarations; enge compiles Patch configuration from those typed definitions | uFor conformance and enge preparation/audio tests |
| LFO playback, rate, reset, phase and onset | enveLFO runtime | Core LFO state | Shared body traces and LFO tests; enge retains event-free sampling acceleration |
| Finite/indefinite Envelopes, curved release and repeats | enveLFO runtime | Core Envelope state | Shared runtime/host traces and focused Envelope tests |
| Stage transitions, cue events and entry capture | enveLFO runtime | Core Stages state | Shared body traces, command-preemption and partition tests |
| Patch outputs, signal operators and retained state | Standalone Patch scheduler | Core Patch/signal modules | Shared Patch/host traces and operator tests |
| Division, offset, chance, random words and delayed queues | Standalone Patch/random modules | Core Patch/random modules | Shared portable vectors and host traces; focused gate/cascade/capacity rollback tests |
| Per-child clocks and elapsed beats versus song position | Host TempoMap supplied to Patch | Host ClockMapping supplied to Patch | Shared stop/seek/tempo trace; frame conversion and clock acquisition stay in hosts |
| Continuation and pending deliveries | JSON snapshots | Clones; optional core JSON snapshots | Shared traces restore JSON with the Rust feature enabled; enge snapshot isolation and exact fractional delivery tests |
| Voice scope, triggers, minimum hold and retirement | enge | enge | Host policy, not duplicate Envelope playback |
| Audio integration | enge sampling and shared Python Patch state | Native adapters over shared core state | Full enge suite and Python/native Patch regressions |
| arpeg expression integration | Optional arpeg bridge | Separate arpeg-envelfo crate | Shared composition MIDI trace, mapped clocks and pending-delivery continuation |

The legacy low-level `ufor.lfo.LFO` and its separate event/state/value records and
initializer/query/command functions have been removed, with no aliases. enge
block sampling consumes canonical LFO definitions and EnvelfoState directly.
Lighting also uses that definition; stateless shape and activation mathematics
remain in uFor without a runtime dependency cycle.

## Native uFor preparation boundary

The native dependency is now the single Git-distributed `engelfo` crate. Rust
uFor parses canonical JSON LFO, Envelope and Stages definitions
during preparation, then enge converts them to the existing execution definitions. Python dependency
pins remain unchanged for this slice; their canonical generator serialization is
accepted by the native adapter. Python authoring syntax is normalized before
crossing that boundary. Runtime execution remains in engelfo.

The internal `GeneratorBank` constructor takes sample rate, slot and parameter
counts, LFO routing metadata, JSON LFO definitions, JSON named-Envelope definitions
and Envelope routing metadata, plus keyword `max_records=10000`.
LFO routing rows contain parameter index, add/multiply operation, intercept and
slope. Envelope routing tuples contain parameter index, operation and voice-release
policy. Definitions follow route-slot order; existing owner maps retain shared
source identity. SampleHold retains its established bank slots and host seed setup.

The budget counts one preparation root, each route record, and uFor's definition,
segment and marker records for every supplied occurrence. Raising it permits
larger prepared banks. Aggregate JSON text is capped at 16,777,216 characters;
each definition also passes uFor's document-value/depth checks. These limits cover
this constructor operation; Stages/Patch configuration is still a later slice.
Portable counters are signed i64, including Envelope repeat counts. Invalid
definitions, unresolved public rates, out-of-range counters, route mismatches
and frame-conversion overflow fail before constructing the executable bank.

Exact seconds convert to frame coordinates once, while beat quantities retain
their clock. Existing scope/reset policy and normalized named loop bounds remain
host-owned. Authored reset retriggering now maps to engelfo's Initial policy.
Named captured approaches are prepared in the same pass for synth, FM, noise and
sampler consumers. The old LFO rational-array encoder and named-Envelope segment,
approach and beat-flag setters are removed.

Main amplitude constructors (`SynthRuntime`, its FM/noise variants, and
`LiveRuntime.add_sample`) take canonical Envelope JSON instead of initial/segment
arrays. Graph FM takes one definition per operator, including its carrier. Each
constructor has keyword `max_records=10000`, counting one root and each amplitude
slot, definition and segment, with the same text and portable validation bounds.
All FM operators share one budget; it is independent of a separately prepared
GeneratorBank. Captured releases are prepared directly, and their old setters and
Python encoder are removed.

Main amplitude retains the existing voice/seconds/held/once profile, numeric
initial, activation start, no markers and explicit release. Captured approaches
remain release-only. A fresh voice activation uses enge's Current retrigger policy;
authored retrigger settings still govern named generators. Scalar voice gain,
minimum hold, sample exhaustion, retirement, command storage and snapshots remain
host-owned. Empty segment arrays that previously bypassed Python profile
validation are no longer accepted at the native boundary.

The 2026-11-17 amplitude-preparation qualification passed 3,604 Python tests with
29 optional skips. Its 24,703 successful and 68 failed callbacks recorded zero
allocations, reallocations or frees inside the callback boundary. The broader
native-call boundary retains allocations outside callbacks. Preparation limits
and profile rejection tests are in `test/test_amplitude_preparation.py`; existing
amplitude, runtime and live regressions cover audio and snapshot continuation.
See the integration record for the installed-build audit reproduction commands.

`GeneratorBank.set_staged_envelfos` takes `(JSON Stages definition, optional
parameter index, operation)` tuples and the existing connection metadata, plus
keyword `max_records=10000`. Its budget combines one root, route slots, all nested
definition/stage/segment/marker/transition/reference records and each host
connection. Document text/value/depth bounds apply as at the other preparation
boundaries. Beat-clock Stages still require a prepared host clock.

The converter preserves numeric native stage identities, transition expansion
order, marker order, normalized loop bounds, child LFO affine values and the
held/Current Envelope-entry policy. Valid `reset` retrigger input is now accepted;
the old encoder looked for the nonexistent spelling `initial`. The old stage
tuples, Python segment/option encoders and native array decoder are removed.
Connection scheduling, commands, retained state and snapshot layout remain
unchanged.

Stages preparation qualification passed 3,619 Python tests with 29 optional
skips. The 24,703 successful and 68 failed callbacks recorded zero callback heap
operations. The generated report is `build/rust-ufor-stages-audit.xml`.
`test/test_stages_preparation.py` covers portable preparation boundaries; the
expanded runtime regression covers reset acceptance with loop bounds, release
and snapshot replay. Local qualification used macOS.

`GeneratorBank.configure_patches` prepares Patch operators, selected output routes
and marker/stage/threshold events from canonical generator documents and the real
modulation/binding/control/event declarations. Separate metadata supplies bank
slots, parameter indices and the legacy node order. This private preparation
context is not a full SoundSettings/ModulationSettings decoder or a score format.
Named generator text remains ordered and uses the typed parser; other declarations
reuse uFor's bounded `codec::parse_value` adapter.

`max_records=10000` covers shared declarations, generator occurrences and host
layout/index references together. The override reaches embedded modulation and
generator bodies. Aggregate text and per-document value/depth bounds also apply.
Generator-aware checks reject unresolved generator, follower and event facts.
Python retains processor-target validation; an available native parameter index
does not establish full processor capability/domain compatibility.

Node and input order, event-category order, stream identities, frame delays,
threshold offsets and shared owner maps retain their previous meaning. Native
configuration commits only after all indexed-plan/profile checks succeed.
SampleHold ranges come from its canonical definition; its host setter now accepts
only stream keys. Old Python Patch operator/event encoders and public native
transform/event setters are removed. The Python offline graph adapter and host
Patch-output/source/index materialization retain their consumers.

Synth, effects and sampler use the full configuration operation. FM/noise keep
their current supported profiles and run the same generator-aware checks before
their existing native preparation. See the
[integration record](rust-ufor-integration.md) for completed stages and qualification.

Patch/binding qualification passed 3,628 Python tests with 29 optional skips. The
24,703 successful and 68 failed callbacks recorded zero callback allocations,
reallocations or frees. The broader native-call boundary retains allocations
outside callbacks. `test/test_patch_preparation.py` checks unresolved requirements,
host layout order, combined budgets and failure without mutation; the existing
audio and continuation suites cover the consumers. The generated report is
`build/rust-ufor-patch-audit.xml`; see the integration record for reproduction.

### Modulation route preparation and shared mathematics

`_native.prepare_modulation` consumes canonical modulation/binding/control/event
declarations, ordered generator JSON and host target metadata. Synth, FM, noise
and sampler also supply the processing portion of a canonical SoundSettings
body. Rust uFor resolves the authored amplitude, tuning cents, cutoff and Q
facts for the shared native slots, and enge checks authored parameter units,
defaults and minimum rules against those facts.

Effect graph preparation compiles canonical modulation declarations into
native control tables. Rust uFor checks every target's unit, default, and bounds
against its authored processor. This includes named mixer inputs, multiband
compressor bands, discrete bitcrusher bounds, and captured-buffer scrub
positions. Rust checks the combined base-delay and depth ranges for modulated
delays and admits fixed-delay damping targets when the processor declares
damping. Rust also validates input and processor connections, and provides
the layered execution order, named channel layouts, and cumulative frame
latencies used during offline and live preparation. Follower dependencies
participate in both ordering and alignment. The prepared Rust graph also
validates authored actions before offline application and live queue encoding,
including processor capabilities, parameter bounds, discrete controls, and
input names. Python retains asset resolution, frame windows, and native slot
assignment. Enge owns queue admission and runtime state.
Topology preparation uses the shared codec limits and the default 10,000-record
budget. Cumulative latencies use checked signed 64-bit frame counts.
The compiler preserves declaration order for target slots and sorts route
contributions by source and route name. Follower assignments and individual
generator contribution slots retain their existing meaning. The preparation
budget covers modulation, supplied processor declarations, and host slot and
follower metadata.

The focused macOS audit passed 162 tests covering effect controls, persistent
effects, followers, frequency shifting, routing, mixing, and matrix delay.
It recorded zero heap operations in 6,627 successful and 12 failed callbacks.
The generated report is `build/rust-ufor-effect-targets-audit.xml`.
The dynamics extension passed 98 focused macOS tests with zero heap operations
inside 4,481 successful and 15 failed callbacks. The generated report is
`build/rust-ufor-dynamics-targets-audit.xml`.
The initial delay changes passed 101 focused macOS tests with zero heap operations
inside 4,890 successful and 12 failed callbacks. The generated report is
`build/rust-ufor-delay-targets-audit.xml`.
With Rust validation required for every effect modulation target, 335 focused
macOS tests passed. The audit recorded zero heap operations inside 13,810
successful and 51 failed callbacks. It includes mono and stereo regressions
for authored zero-mix reverb during live rendering and snapshot replay.
The generated report is `build/rust-ufor-all-effect-targets-audit.xml`.
Topology preparation passed 456 focused macOS tests. The audit recorded zero
heap operations inside 18,719 successful and 68 failed callbacks. Follower
tests compared order and cumulative latencies to Python's authoring graph
and checked named channel layouts, playback, and snapshot replay.
The generated report is `build/rust-ufor-effect-topology-audit.xml`.
Action validation passed 457 focused macOS tests. The audit recorded zero
heap operations inside 18,719 successful and 68 failed callbacks. The
generated report is `build/rust-ufor-effect-actions-audit.xml`.

Native generator preparation reads the host's canonical TempoMap through Rust
uFor and derives exact frame points for transport and elapsed beats. The
Python reference renderer uses the Python TempoMap. Native snapshots retain
their existing clock point representation. Preparation uses the generator
bank's integer sample rate and the shared 10,000-record default.
TempoMap preparation passed 510 focused macOS tests with zero heap operations
inside 2,227 successful and one failed callback. The generated report is
`build/rust-ufor-tempo-map-audit.xml`.

The offline native sampler and the live sampler share a prepared traversal
from Rust uFor's authored slice, loop and resolved playback declarations.
Sampler preparation uses Rust uFor to resolve the output and native asset
timebases, check slice extents, and match the decoded array shapes to the
referenced assets. The asset inventory budget defaults to 10,000 records and
can be overridden for a preparation call. Enge loads and fingerprints the
audio buffers.
Asset inventory preparation passed 718 focused macOS tests with zero Rust
heap operations in 6,507 successful and seven failed callbacks. The generated
report is `build/rust-ufor-sample-inventory-audit.xml`.

Sampler, oscillator synth, FM, noise, and Yamaha preparation share Rust
uFor's physical-timebase rate validation. Their integer-rate profiles
require an authored denominator of one. Physical timebase integration
passed 446 focused macOS tests with zero Rust heap operations across 2,533
successful and six failed callbacks. The generated report is
`build/rust-ufor-physical-timebase-audit.xml`.

The MIDI file adapter uses Rust uFor's PPQN clock to round absolute event
times across tempo changes. Enge decodes and merges the file records and
dispatches events to the configured performance converters. Track ends
extend the frame count without consuming performance ordinals.
MIDI file and live converters share Rust uFor's prepared channel assignments
and controller and pitch-bend mapping. The Rust held-note state handles
note admission, pitch, pairing, trigger IDs, and restart. Enge resolves
references and supplies the instrument control declarations and tuning
bodies. Rust uFor checks the mapped ranges and prepares the tunings by
channel. Enge validates event order and emits performance events with the
exact host ticks and ordinals.

Native synth-family and sampler trace preparation validate performance
payloads through Rust uFor. This includes trigger control overrides,
addressed control changes, and LFO and Envelfo command fields. The trace
planner retains event order and voice lifecycle state.

Native sampler trace preparation asks Rust uFor to validate and prepare its
latched keyswitch lookup. The sampler trace applies switches per part and
records explicit clears in its selection snapshot.

Native sampler and synth-family traces prepare their initial sustain state
from the declared unipolar control default using Rust uFor. A default equal
to the threshold starts with the pedal pressed.

Persistent synth, FM and noise voices use Rust uFor to convert minimum
hold seconds to exact frame durations, including fractional frames.

Both samplers use Rust uFor's cursor for pending release and loop decisions,
forward and reverse movement, mirror turns, whole plays, overlap taps, and
end fades. A shared read plan determines overlap and interpolation weights
once per frame and applies them to each decoded channel supplied by enge.
Enge owns filtering and routing.
Snapshot and Python bridge coordinates retain their existing integer and
floating-point fields.
The offline native bridge passes a release as a batch frame offset and an
exact fraction within the frame. Rust uFor rounds both parts of the advance
from their exact products, then applies the release between them. The
fractional part must fit signed 64-bit numerator and denominator fields.
Traversal preparation passed 670 focused macOS tests with zero heap
operations inside 6,755 successful and seven failed callbacks. The generated
report is `build/rust-ufor-sample-traversal-audit.xml`.
The shared cursor subsequently passed 711 focused macOS tests with zero Rust
heap operations in 6,507 successful and seven failed callbacks. The generated
report is `build/rust-ufor-sample-cursor-audit.xml`.
The fractional release split passed 740 focused macOS tests with zero Rust
heap operations in 6,507 successful and nine failed callbacks. The generated
report is `build/rust-ufor-sample-release-split-audit.xml`.
The shared read plan passed 584 focused macOS tests with zero Rust heap
operations in 3,793 successful and six failed callbacks. The generated report
is `build/rust-ufor-sample-read-audit.xml`.

Offline native control batching uses the same cursor to bound the active
prefix from the maximum tuning, remaining finite traversal, and any latched
end fade. Rust uFor reads and caches the common and slot tuning bounds during
renderer preparation. The host splits the request at a pending release before
calculating the bound. The NumPy reference applies the same rule. This keeps control
evaluation inside the audible lifetime of a finite loop.
The batch bound passed 713 focused macOS tests with zero Rust heap operations
in 6,507 successful and seven failed callbacks. The generated report is
`build/rust-ufor-sample-prefix-audit.xml`.

The native sampler prepares a Rust uFor pitch mapping per slot. Each onset
uses the resolved `pitch_hz`, the sample's reference frequency, and the voice's
pitch variation. Rust also combines the onset key and velocity crossfades
with instrument, slot, and variation volume. Native offline playback applies
common and slot tuning to that ratio before converting it to cursor units
with the decoded asset rate. The same calculation supplies its control batch
bound. Live playback applies
both tuning layers to the onset step, which already carries the asset rate.
Both native paths use the checked Rust uFor mathematics and preserve their
existing multiplication order. The NumPy renderer uses Python's mapping and
array arithmetic.
Native offline and live onsets use the Rust cursor's initial direction and
play budget. Rust applies asset alignment and direction-relative variation in
signed 64-bit frames and marks an onset outside the slice as exhausted.
Pitch mapping passed 670 focused macOS tests with zero heap operations inside
6,755 successful and seven failed callbacks. The generated report is
`build/rust-ufor-sample-pitch-audit.xml`.
Routed pitch and rate conversion passed 713 focused macOS tests with zero
Rust heap operations in 6,507 successful and seven failed callbacks. The
generated report is `build/rust-ufor-sample-routed-pitch-audit.xml`.

Rust uFor also prepares slot playback overrides against instrument defaults.
Enge uses the same prepared slot to calculate onset delays for offline and
live playback. Slot and variation delay are added as checked rational seconds
before rounding up to an output frame. The total play count can still reach
`2^64` because the cursor tracks only plays remaining after the first.
Slot playback passed 671 focused macOS tests with zero heap operations
inside 6,756 successful and seven failed callbacks. The generated report is
`build/rust-ufor-slot-playback-audit.xml`.

Native sampler onset and offline control fades use Rust uFor's authored sample
crossfades. The live callback uses the same gain curve on its normalized
control position. Enge retains control smoothing and voice selection, and
the NumPy renderer retains its reference calculation.
Crossfade preparation passed 671 focused macOS tests with zero heap
operations inside 6,756 successful and seven failed callbacks. The generated
report is `build/rust-ufor-sample-crossfade-audit.xml`.

`sample_instrument.prepare_trace` prepares sampler actions using Rust uFor to
filter slots in instrument declaration order, advance cycle, random, and
shuffle take-selection partitions, and resolve seeded random ranges and voice
variation. Native MIDI rendering uses this preparation path for sample
instruments. Python supplies the event context, eligible candidates, and
take or slot identity, manages trigger and voice lifecycle, and records the
existing trace snapshots. The offline and live samplers consume the prepared
actions.
Sampler eligibility passed 691 focused macOS tests with zero heap operations
inside 7,383 successful and seven failed callbacks. The generated report is
`build/rust-ufor-sample-eligibility-audit.xml`.
Take selection passed 128 focused macOS tests with zero heap operations
inside 3,409 successful and six failed callbacks. The generated report is
`build/rust-ufor-sample-selection-audit.xml`.
Seeded ranges and variation passed 129 focused macOS tests with zero heap
operations inside 3,409 successful and six failed callbacks. The generated
report is `build/rust-ufor-sample-variation-audit.xml`.

Native MIDI trace preparation uses Rust uFor to plan voice retirements
before starting a selected batch for sampler, synth, FM and noise. The
planner applies chokes, per-part sampler pools in name order, then the
instrument-wide policy. Python supplies the selected batch and active voices
and records the resulting retirement actions in the trace.
Voice admission passed 201 focused macOS tests with zero heap operations
inside 3,421 successful and six failed callbacks. The generated report is
`build/rust-ufor-voice-admission-audit.xml`.

Native MIDI trace preparation also uses Rust uFor to classify key releases
and sustain transitions for sampler, synth, FM and noise. The planner
keeps sustain scoped to each part and returns deferred releases in onset
order. Python applies release layers against the current voice state and
records the actions and snapshots. The bridge reads the host's accumulated
trigger history without applying score document limits to that state.
Release planning passed 204 focused macOS tests with zero heap operations
inside 3,421 successful and six failed callbacks. The generated report is
`build/rust-ufor-voice-release-audit.xml`.

Native MIDI rendering uses `synth.prepare_trace` for oscillator synth, FM
and noise. Rust uFor validates the mapped voices, SoundSettings, and shared
instrument declarations before selecting voices in authored order for onset,
release and sustain events. FM modulation targets resolve against each
voice's own graph. The trace provides the original onset velocity for
release layers and resolved `pitch_hz` for the renderer.
One record budget covers the instrument and its voices. The prepared
instrument also supplies the initial sustain state and the control
declarations used to validate performance events.
Voice eligibility passed 352 focused macOS tests with zero heap operations
inside 3,421 successful and six failed callbacks. The generated report is
`build/rust-ufor-voice-eligibility-audit.xml`.

Native sampler and synth-family trace preparation derives voice-start pitch
and random stream keys with Rust uFor. Synth-family voices apply their
authored frequency offset when a `pitch_hz` is supplied. Noise and EnveLFO
derive keys from the performance seed and voice ID using their own
SHA-256 domains. Native Patch and modulation preparation
also use the shared EnveLFO graph identity calculation.
Voice-start and graph stream preparation passed 640 focused macOS tests
with zero heap operations inside 3,537 successful and six failed callbacks.
The generated report is `build/rust-ufor-voice-start-audit.xml`.

When a host supplies a tempo map and sample rate, `synth.prepare_trace`
schedules quantized EnveLFO commands with Rust uFor. It uses exact beat
arithmetic, rounds up to the dispatch frame, and assigns moved commands
ordinals after the authored events at that frame. A seek or unreachable
target cancels a pending command. Python applies the scheduled coordinates
to the existing trace actions.
Command scheduling passed 113 focused macOS tests with zero heap operations
inside 2,119 successful and six failed callbacks. The generated report is
`build/rust-ufor-command-schedule-audit.xml`.

Native offline and persistent synth rendering use Rust uFor to prepare
oscillator waveform, duty cycle, and key-scaled gain. Enge reuses prepared
oscillator declarations across voice starts and blocks. The audio kernels
receive a binary64 duty value rounded directly from the exact ratio so that
the position of a square-wave transition matches the Python reference.
For transport-synchronized onsets, Rust uFor reduces the exact frame-frequency
product modulo the sample rate before rounding the initial phase. The native
bridge keeps whole-frame origins as signed 64-bit integers.
The synth oscillator and FM graph renderers share uFor's binary64 waveform
evaluation in both block and persistent rendering. FM graph operators use
half duty for square and triangle waves.
Native synth and FM renderers also use uFor's compensated phase advancement.
Snapshots carry the wrapped position and the compensation in sample-rate units.
Oscillator preparation passed 198 focused macOS tests with zero heap
operations inside 2,119 successful and six failed callbacks. The generated
report is `build/rust-ufor-oscillator-audit.xml`.
The shared waveform evaluator passed 249 focused render tests and an 80-test
live audit on macOS. The live audit recorded zero Rust heap operations in
1,875 successful and six failed callbacks. Its generated report is
`build/rust-ufor-waveform-audit.xml`.
Compensated phase advancement passed 350 render and live tests on macOS
with zero Rust heap operations inside 1,875 successful and six failed
callbacks. The generated report is `build/rust-ufor-phase-advance-audit.xml`.

Native offline and persistent oscillator and FM rendering use Rust uFor
to apply routed tuning to the resolved onset `pitch_hz`. FM applies the
operator ratio first, then the combined voice and operator cents.
Persistent evaluation rejects nonfinite or nonpositive rendered frequencies
with fixed-size errors inside the callback.
Pitch realization passed 285 focused macOS tests with zero heap operations
inside 2,119 successful and six failed callbacks. The generated report is
`build/rust-ufor-synth-pitch-audit.xml`.

Sampler renderer and native trace preparation use Rust uFor to resolve
slot-group inheritance. The resolver selects whole SoundSettings categories
and resolves take-set inheritance, including an explicit `selection = false`.
Unspecified categories use the SoundSettings defaults when there is no
group value. Python decodes the selected bodies and continues to validate the
instrument document and expand Patch outputs. Nested generator bodies keep
their authored declaration order through the bridge.
Slot inheritance passed 186 focused macOS tests with zero heap operations
inside 5,815 successful and seven failed callbacks. The generated report is
`build/rust-ufor-slot-settings-audit.xml`.

Rust uFor resolves authored channel routes into input-by-output gain matrices
for sampler, oscillator synth, FM and noise. Sampler preparation also applies
the Native or SFizz pan law to mono-to-stereo routing, or the corresponding
balance law to stereo-to-stereo routing. Other layouts use their authored
matrix directly. The renderer consumes the prepared gains.
Channel routing passed 400 focused macOS tests with zero heap operations
inside 3,605 successful and six failed callbacks. The generated report is
`build/rust-ufor-channel-routing-audit.xml`.

Persistent FM prepares a canonical uFor FM body once. Rust validates the
operator Envelopes and current-sample topology, then compiles the ordered
indexed graph for enge. Native modulation uses the same prepared graph to
resolve operator ratios and tuning, edge indices and carrier level. Slots
follow the existing order: amplitude and tuning, operator pairs, edges,
carrier level, then shared filters. The constructor consumes the prepared
graph and retains enge's voice-time Envelope profile and renderer state.

The FM integration passed 184 focused macOS tests with allocation auditing
enabled, covering FM, amplitude and modulation preparation, live rendering
and MIDI rendering. The run recorded 1,870 successful and eight failed
callbacks with zero Rust heap operations. The generated report is
`build/rust-ufor-fm-audit.xml`.

Enge's native preparation assigns filter slots in authored order and uses
Rust uFor to compute the sample-rate and Nyquist bounds. Sampler preparation
reuses those indexed filter rows. Persistent instrument and effect renderers
use uFor's allocation-free error or clamp policy for each cutoff and continue
to use enge's trapezoidal filter coefficients and integrator state.
The authored cutoff remains the default used for target validation. The
runtime cutoff fallback for an undeclared parameter uses the clamped value.
Existing runtime capabilities govern which targets get slots. Python retains
owner validation, Patch-output materialization, orchestration and the offline
graph adapter.

Filter boundary qualification passed 272 focused macOS tests with zero Rust
heap operations inside 4,227 successful and 14 failed callbacks. The generated
report is `build/rust-ufor-filter-cutoff-audit.xml`.

Native modulation preparation's `max_records=10000` budget combines shared
declarations, supplied generators, the SoundSettings processing body, any
prepared FM graph (including default operator Envelopes), host target/filter
records and expanded generator slots/bodies. An override reaches embedded
modulation and generator validation. Expanded JSON text is also
bounded, alongside existing per-document limits. Exact seconds convert to frame
ratios with checked signed-64-bit components. Source-map insertion order, sorted
routes within categories, scopes, latched events and deterministic stream keys
retain the existing host layout.

Prepared runtimes use uFor's `modulation::math::Mapping`, `add_partial` and
`sum_partials`; the duplicate enge kernels are removed. Accurate-sum scratch is
allocated during preparation and reused. Mapping/sum errors are fixed-size inside
callbacks, including nonfinite intermediate rejection. Effect matrices and mixers
share the same accurate-sum implementation. The general uFor scalar evaluator
remains an allocating reference API, not a callback implementation.

Enge retains its controls/events, LFO, named-generator, Stages and Patch-transform
contribution order and direct full-weight shortcuts. These differ from the
reference evaluator's global ordering and universal weighting expression; tiny
multiplier rounding can therefore differ. Changing those policies is separate
work, not an implied consequence of sharing the numerical kernels. Callback state
ownership, execution profiles and snapshot representation are unchanged.

Modulation qualification passed 3,641 Python tests with 29 optional skips. The
24,704 successful and 71 failed callbacks recorded zero Rust allocations,
reallocations or frees. The generated report is
`build/rust-ufor-modulation-audit.xml`; the integration record describes reproduction
and the local macOS qualification boundary. The broader native-call boundary
still performs heap operations outside callbacks.

After adopting SoundSettings for shared targets, the full macOS allocation
audit passed 3,649 tests with 29 optional skips. It recorded 24,704 successful
and 71 failed callbacks with zero Rust allocations, reallocations or frees.
Use `build/rust-ufor-sound-settings-audit.xml` as the report path when
reproducing the audit with the integration record's commands.

JSON parsing, validation and conversion allocate only during preparation. They
are not called from audio processing. `test/test_generator_preparation.py`
checks rejection/budget boundaries, independently calculated gain curves and
partitioned native snapshot continuation with two-second 48 kHz WAV regressions.

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
not comparisons, phase wraps or position advancement. Rust supports cloned state
and optional, self-contained versioned JSON through core's `snapshots` feature.
This is not a shared Python/Rust wire format or a change to enge's audio snapshots.

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

Python restore steps serialize and reload state. With `snapshots` enabled, the
Rust shared body/Patch/host traces serialize and reload every continuation step;
default Rust builds retain clone-based continuation checks.
Cross-language snapshot interchange is not required. arpeg owns its composed
expression wire trace and verifies its explicit lane ownership and same-time
delivery phases independently of enge.

The audit verifies clean wheel installation outside the source tree without
NumPy/enge, source-distribution and standalone crate builds, Python 3.13/3.15
on macOS, Rust debug/release tests and consumer suites. Cross-platform checks
remain release-only; configuring them is not evidence that a release has passed.
No release or hardware validation is implied.

## Separately scoped follow-ups

Optional Rust JSON snapshots now cover LFO, Envelope, Stages, SampleHold and
Patch definitions plus complete state, including pending core deliveries and RNG
streams. Hosts still save clock mappings and external queues themselves. This
does not add serialized combined enge/arpeg application snapshots.

arpeg's Python and Rust players now separate scheduling and MIDI realization.
Hosts deliver same-time hits, observe the bridge once, then realize controllers
before note-on, retaining existing release and expression-lane ownership rules.
Python snapshots also retain pending unrealized batches. Combined Rust host
snapshot serialization remains separate work.

Standalone Python bindings to Rust remain optional. Built-in Rust tempo
acquisition and broader audio-host Envelope profiles are not silently added here.

## Additional work beyond the prompt

None.
