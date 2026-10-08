# Synthesis engines: review, common design, and new directions

Design and static code review, 2026-09-28. This replaces the earlier candidate
shortlist with a broader proposal. No runtime changes, performance measurements,
listening tests, or crash reproductions were performed for this review.
Source symbols below are evidence anchors, not a claim of an exhaustive audit.

The recommendation is to improve the quality and consistency of the current
engines, then make new instruments by combining reusable sound processes.
Alongside useful established techniques, explore **imprint synthesis**: a
resonant instrument whose internal structure remembers how it has been played.

## 1. What is actually here

| Component | Current role | Important boundary |
| --- | --- | --- |
| Oscillator synth | Sine, square, and triangle-derived shapes; phase, envelopes, tuning, filters, routing | NumPy/reference, native block, and persistent paths |
| Graph FM | Two through six operators; waveforms, current edges and delayed feedback | Actually phase modulation; one selected carrier output; old fixed-FM native machinery still exists |
| Sampler | Prepared immutable audio, pitched traversal, reverse/mirror/loop behavior, release, filtering | Has its own cursor and exhaustion lifecycle; native live playback and Python block adaptation are different paths |
| Noise | Deterministic white-noise stream with envelope and processing | Random identity/counter state is different from oscillator phase |
| Granulator | Captures and overlaps grains from incoming audio/history | Exists as an effect, including native/live code; a granular instrument is an extension, not a blank-slate grain engine |
| Effects and live mixer | Gain, multiplication, filters, saturation, granular and delay processing | Stateful sound-producing processes already exist outside the named synthesis engines |
| Controls | Envelopes, LFOs, smoothing, routes, scoped contexts | Shared in places, but lifecycle and backend support are not consistently shared |

Relevant files: [synth.py](../enge/synth.py), [fm.py](../enge/fm.py),
[noise.py](../enge/noise.py), [sampler.py](../enge/sampler.py),
[sample_instrument.py](../enge/sample_instrument.py),
[effects.py](../enge/effects.py), [live.py](../enge/live.py), and their Rust
implementations under [src](../src).

Existing strengths worth preserving: prepared trace actions, exact authored
timing, compensated phase/cursor accumulation, immutable sample buffers, stable
random streams, explicit routing, backend comparisons, snapshots, and bounded
native live buffers. Uniformity should build on these rather than discard them.

## 2. Findings that should precede architectural expansion

“Observed” means directly visible in the inspected code. “Inferred” means a
consequence deduced from that code, still requiring a focused reproduction.
Priority reflects likely user impact, not measured frequency.

### A. Sampler named-envelope state is not carried through its lifecycle

**Priority: high. Observed omission; runtime failure inferred.**

In [sample_instrument.py](../enge/sample_instrument.py), _validate_settings calls
the common generator validator, and _start_voice obtains named-envelope sources
through ControlRenderer.sources. However, SamplerSnapshot, snapshot, and restore
carry contexts and LFOs without the envelope-state collection. Its release path
calls voice.renderer.release without calling controls.release.

A sample patch with a named envelope can therefore play with a live envelope
source index, then restore into a fresh control renderer whose envelope list is
empty. The next evaluation should fail when indexing that list. Restoring into
an already used renderer could instead read stale state. Named envelopes also
do not receive the matching release transition.

**Next verification:** route a named envelope to sample gain, release midway,
snapshot, restore into a new sampler, and compare both signals and state.
Exercise instrument-level and slot-level bindings, including the persistent
adapter. Fix the missing lifecycle state before advertising generator parity.

### B. Minimum hold can make native named envelopes evaluate negative release time

**Priority: high. Observed path; numerical/error outcome inferred.**

In [runtime.rs](../src/runtime.rs), apply_action records a release coordinate
equal to max(voice age, minimum hold). parameter immediately uses
age - release_frame for a named envelope once that coordinate is present.
Unlike the main envelope branch, it does not wait until release_frame.

For example, release a note at 0.1 seconds with a 0.5-second minimum hold. The
named envelope then receives -0.4 seconds of release progress. A linear release
can extrapolate outside its target domain. A zero-duration first release segment
can reach division by zero in envelope_value, leading to a non-finite parameter
and an error. Native live processing can consequently silence the block and
enter its failure state.

**Recommendation: option 3, selectable release timing for each named-envelope
binding.** Responding immediately to release is an expressive capability: timbre
can relax while minimum hold keeps the sound audible. Coordinated release is
also useful, so expose the choice rather than imposing one policy on every
envelope. This can be implemented now without introducing the full motion/event
graph proposed elsewhere.

Add one field on envelope bindings, not on the reusable envelope shape:

```toml
[[bindings]]
name = "brightness"
kind = "envelope"
reference = "filter-contour"
release_timing = "event"
```

- `event` (default): release when the voice's prepared release action arrives,
  independent of minimum hold. This preserves the existing offline behavior and
  gives the release event immediate expressive effect.
- `voice`: release at the main sound envelope's effective release coordinate,
  after minimum hold. Until then, continue the named envelope's normal attack
  or held progression.

For a release action at 100 ms and minimum hold of 500 ms, these mean 100 ms
and 500 ms respectively. Each envelope captures its own value at its chosen
release time. `event` means the existing prepared voice-release action, which
may already reflect sustain-pedal or other instrument policy; it does not bypass
that policy by listening directly to raw MIDI note-off.

**Implementation boundary:** validate the field only for envelope bindings;
carry it into the prepared native named-envelope definitions. Resolve and retain
each source's chosen release coordinate in both backends. The offline path must
retain a pending release rather than advancing its event state into the future
and then querying earlier samples. Native evaluation must continue pre-release
behavior until the coordinate is reached, never pass negative elapsed time to
release evaluation, and handle zero-duration segments at the boundary.

Duplicate releases must not restart or move a pending/active release. Snapshot
and restore must preserve pending release state; stop/voice retirement discards
it. The main sound envelope and FM operator lifetime rules remain unchanged,
and a named envelope cannot extend its owning voice's lifetime. Include the
sampler when fixing A so the policy is uniform across supported source engines.
This is a bounded schema/lifecycle change, not just the numerical guard: it
requires coordinated ufor and enge changes, but no new scheduler framework.

**Next verification:** place two named envelopes on one voice, one using each
policy, and release before minimum hold. Compare offline and persistent values
before, at, and after both release coordinates, including fractional sample
boundaries, zero-time segments, duplicate releases, and snapshot/restore while
release is pending. Confirm omitted policy behaves as `event`, both policies
agree when minimum hold has elapsed, and immediate stop cancels pending work.

### C. Low-level graph FM validation is weaker than its callers

**Resolved, 2026-10-08:** native graph rendering now rejects non-finite inputs,
fractional/non-finite edges, and non-finite output/state. Persistent graph
preparation requires a permutation ordered before every current-sample edge.
Direct native regressions cover these failures and overflow without input mutation.
The original review findings follow for context.

**Priority: high for native-boundary correctness. Observed.**

[fm.rs](../src/fm.rs), render_graph_fm, validates layout but does not comprehensively
validate finite frequencies, envelopes, indices, levels, phases, or history.
Edge indices are floating-point values checked by range, then cast to integers;
fractional values are not rejected and NaNs evade ordinary range comparisons.
Unlike the older render_fm, the graph function does not finish with an equivalent
finite-output/state check.

[runtime.rs](../src/runtime.rs), fm_graph, accepts a supplied order with the right
length and in-range entries without verifying a permutation or that every
current edge precedes its destination. A repeated or incorrect order can use
stale outputs and produce the wrong graph.

Normal Python preparation filters out many such inputs. That is not a sufficient
contract for an exposed native entry point.

**Next verification:** direct native calls with duplicate order entries, reversed
dependencies, fractional/NaN edge entries, and non-finite state. Require useful
errors before state mutation. This review did not demonstrate a process crash;
these are concrete validation gaps, not evidence of memory unsafety.

### D. Error recovery can leave partially advanced state

**Partially resolved, 2026-10-08:** live snapshot restoration preflights every
source and the effect graph before changing any state. Regressions verify that
an incompatible later source or effect graph leaves audio continuation unchanged.
Processing-error recovery and the reset-only policy still need a decision.

**Priority: high for live operation. Observed ordering; recovery risk inferred.**

[runtime.rs](../src/runtime.rs) validates and applies actions while rendering.
An invalid later action can follow already applied actions and audio-state
updates. In [live.rs](../src/live.rs), sources process sequentially, and
process_into silences output and latches failure on an error. reset_failure only
clears that flag. It does not roll back earlier source advancement or consumed
batches. LiveRuntime.restore likewise restores sources sequentially before
validating every later source/effect snapshot.

Silencing a bad block is useful, but “clear the error” must not imply that all
source clocks and states are synchronized again.

**Next verification:** inject an invalid second source action after a valid first
source block; inspect state before and after recovery. Try an incompatible later
snapshot in a multi-source restore. Preflight configuration/action structure
before mutation where possible. For runtime numerical failures, require an
explicit restore/restart policy rather than a costly per-callback full snapshot.

### E. FM does repeated allocation and preparation in its hot path

**Priority: medium. Observed; cost not benchmarked.**

[fm.rs](../src/fm.rs), render_graph_fm, allocates the operator-output vector inside
each sample iteration and collects a new history vector each iteration. It
also recomputes the topological order for every call. [fm.py](../enge/fm.py),
VoiceRenderer.render, repeatedly rebuilds named-edge indices, waveform codes,
and carrier lookup and materializes frame-by-operator/edge arrays.

Its offline native path then calls filters.filter_samples without selecting the
native filter backend, so the filtered graph-FM path still runs the Python
filter loop. “Native” currently describes part of this route, not all its DSP.

**Improvement:** prepare immutable graph layout once, preallocate recurrence
scratch, fill history in place, and keep the selected backend through filtering.
Use constant/ramp/stream parameter representations internally instead of
expanding every static value across a block. Verify complete voices, not just
the oscillator kernel. The persistent graph runtime already preallocates its
graph buffers; do not attribute these allocations to that path.

### F. Persistent parameter lookup scales with unrelated modulation routes

**Priority: medium. Observed; cost not benchmarked.**

[runtime.rs](../src/runtime.rs), parameter, scans controls, LFO routes, and named
envelopes on every parameter lookup. Graph FM calls it repeatedly per operator
and edge per sample; filter processing calls it again. A source connected to
several targets can be evaluated repeatedly rather than once for its scope.

**Improvement:** compile target-to-route adjacency and evaluate each unique
source once per declared evaluation instant and scope. Share constant filter
coefficients where valid. Do not reduce a modulation rate or replace a recurrence
with coarse interpolation merely to make a benchmark faster.

### G. Oscillators and FM have an aliasing ceiling

**Priority: medium to high for bright/high-pitched sounds. Observed algorithms;
audible severity requires measurement and listening.**

[synth.py](../enge/synth.py), _waveform_angles, and the native oscillator/FM paths
sample piecewise square/triangle waveforms directly. There is no band-limiting
step in those paths. FM creates additional sidebands even with sine operators;
large indices and feedback can move energy above Nyquist. A post-filter cannot
remove components already folded into the audible spectrum.

**Improvement:** specify oscillator band-limiting and an explicit quality policy
for modulated/nonlinear networks. Options include bandlimited tables, appropriate
discontinuity corrections, and oversampling with proper resampling filters.
A basic waveform correction is not automatically sufficient for rapidly
phase-modulated square waves. Keep an intentional raw/lo-fi option if musically
wanted, but label its behavior.

Critical detail: an FM feedback edge currently means one sample of delay.
Oversampling changes that delay in seconds unless the contract compensates for
it. Choose whether it is an internal-solver delay or a fixed reference-time
delay before adding quality modes; otherwise “better quality” changes the patch.

### H. Pitched sample playback has interpolation, not full anti-alias resampling

**Priority: medium to high for transposition. Observed.**

[sampler.py](../enge/sampler.py), sample_frames, and
[sampler.rs](../src/sampler.rs), render_sample, linearly interpolate adjacent
cursor samples. This is useful and deterministic but is not a pitch-dependent
low-pass resampler. Bright material pitched upward can alias; interpolation can
also alter the high-frequency response at other ratios.

**Improvement:** compare linear playback with a bounded bandlimited resampler.
Specify filter support, latency, and how lookahead traverses wrap, mirror,
overlap, and release boundaries. Reuse the careful existing cursor semantics;
do not make resampling accidentally commit a pending loop transition early.
An ordinary output filter is not a substitute for anti-alias filtering during
rate conversion.

### I. Granular effect buffers have avoidable offline costs

**Priority: medium. Observed in the offline Rust effect.**

[granulator.rs](../src/granulator.rs), render_granulator, stores history as rows,
allocates incoming rows and wet scratch in the sample loop, and removes the first
history row when full. That removal shifts the remaining history entries.
Capturing grains also copies sample data.

**Improvement:** use a circular history buffer, reusable scratch, and a deliberate
grain-ownership strategy. This observation does not imply the live granulator
has the same implementation: [live.rs](../src/live.rs) has separate prepared
grain/history storage. Establish which grain scheduling/state code can be shared
before turning granular processing into an instrument.

### J. Tail and level semantics vary between source families

**Priority: medium. Observed structure; perceptual consequence depends on patch.**

Oscillator/noise amplitude shaping occurs after voice filtering, while FM
operator envelopes are inside the modulation graph and feed its filter. FM
voice lifetime follows the selected carrier's envelope. The shared
EnvelopeRenderer limits active frames at release completion; it does not
independently wait for resonator/filter energy to decay. Immediate stop removes
the voice.

These are meaningful differences, not automatically defects. They can truncate
ringing or click, and adding modal or waveguide synthesis makes the ambiguity
much more visible. End of excitation, end of amplitude movement, end of source
data, and end of audible tail must become separate states.

The normalized tanh shaper in effects.py/effects.rs is also not a guaranteed
[-1, 1] output limiter: dividing by tanh(drive) can produce a magnitude above
one for sufficiently large input. Do not present soft clipping as overload
protection. Finite output is not necessarily bounded or click-free output.

### K. Shared control state can accumulate across an offline performance

**Priority: medium for long renders. Observed append-only collections in the
inspected control renderer; memory cost needs measurement.**

ControlRenderer appends contexts and voice generator instances. Retiring an
audio voice does not itself remove those entries; snapshots copy the collections.
Long performances can therefore retain historical contexts/generator state.
Its cached arrays are also copied on return.

**Improvement:** establish ownership-based retirement with stable handles before
introducing shared motions. Measure long renders and snapshot sizes. Recycling
must preserve deterministic identities and must not make old queued events
refer to a newly reused instance.

## 3. Concepts that need clearer names and contracts

1. **FM versus PM.** Keep “FM” as the familiar family name, but explain that
   edges currently add phase in radians. True frequency modulation is a different
   connection with a different unit and integration behavior.
2. **Native versus live.** Native code does not by itself imply allocation-free,
   complete native processing or callback suitability. LiveEngine,
   PersistentSampler.advance, and NativeLiveEngine serve different execution
   paths. Publish a feature/capability matrix.
3. **One carrier versus a mix of operators.** Current graph FM selects one output
   carrier. Multiple audible carriers should eventually be explicit output taps
   and a mix, not an overloaded meaning of “carrier”.
4. **Envelope versus lifetime.** Release can change timbre without ending a
   resonant body. A sampler reaching its last source frame need not kill an
   attached effect tail.
5. **Gain versus loudness.** Ratio, dB, modulation depth, operator output, and
   normalized control values need explicit units. No automatic normalization
   whenever a musician adds a layer.
6. **Graph versus engine.** A graph is a composition tool, not a requirement to
   hand-author twenty nodes for a pluck. Familiar instrument presets remain the
   entry point.
7. **Quality versus performance.** Declare resampling, alias behavior, control
   rate, latency, and tail policy. A backend label is not a quality guarantee.

The old plan's “two-operator FM” inventory and future four/six-operator work were
outdated. Its PyTorch suggestion also contradicted the newer roadmap; this
proposal does not reintroduce that work.

## 4. Organizing principle: excite, evolve, observe

An instrument is a collection of **sound processes** with persistent state,
driven by events and **motions**, connected through typed ports.

For many instruments, three musical questions explain the design:

- **Excite:** what introduces energy or information? A note strike, bow pressure,
  oscillator, noise burst, sampled attack, grain, or external audio.
- **Evolve:** what remembers and transforms it? Phase, a delay loop, resonant
  modes, a grain population, a spectral frame, or an evolving field.
- **Observe:** what do we listen to? A carrier tap, virtual pickup, channel mix,
  microphone position, or spectral resynthesis.

This is a design vocabulary, not a claim that all sound models must be energy
conserving or physically realistic. An oscillator has no separate external
exciter; a sample reader's recording already contains its excitation history.
Simple source nodes may combine all three roles.

| Familiar engine | Persistent state | Excitation/drive | Observation |
| --- | --- | --- | --- |
| Oscillator | Phase and waveform reader | Frequency, sync, shape controls | Waveform value |
| FM/PM | Several phases and delayed edge outputs | Pitch, indices, operator envelopes | One or more operator taps |
| Sampler | Cursor, loop/release state | Note pitch, speed, start/release events | Interpolated source channels |
| Noise | Stable stream identity and position | Seed, spectral controls | Noise or filtered noise |
| Granular | History and active grains | Scheduler, grain position/pitch | Windowed grain sum |
| Waveguide | Delay memory and loss state | Strike, bow, breath, external audio | Pickup points |
| Modal body | Resonator states | Impulse/noise/audio | Weighted modes |
| Spectral process | Analysis/synthesis frames | Audio or partial trajectories | Resynthesized spectrum |

There should be one instrument lifecycle and connection language, with distinct
specialized numerical kernels. Do not create a universal sample-by-sample
interpreter or move every algorithm into the existing SynthRuntime source_kind
switch. The old fixed-FM branch should be retired when its callers/tests have
been accounted for, rather than carried alongside graph FM indefinitely.

### A simple authoring surface

Illustrative future TOML, not supported schema:

~~~toml
[instrument]
name = "felt-string"

[sound]
kind = "pluck"
pitch = { input = "note.pitch" }
decay = "2 s"
brightness = 0.35
~~~

The pluck is a library recipe with public musical controls. An advanced author
can inspect or replace its excitation, string, pickup, and processing. Use the
existing ufor score library and public-parameter machinery; do not create another
preset repository mechanism.

A more explicit composition might be:

~~~toml
[processes.breath]
kind = "noise"
color = "white"

[processes.body]
kind = "modal"
model = { selector = "factory:small-metal-body" }

[[audio]]
source = "breath.output"
target = "body.excitation"

[[controls]]
source = "pressure.value"
target = "breath.gain"
operation = "multiply"

[output]
source = "body.output"
~~~

The sample is deliberately small; node-specific defaults remain visible when
the recipe is expanded. Different composition levels are views of the same
definition, not independent competing implementations.

### Data contracts

| Contract | Contents |
| --- | --- |
| Sound process definition | Tagged kind, typed inputs/outputs, parameter declarations, immutable assets |
| Instrument definition | Named process instances, connections, motions, public controls, output selections |
| Prepared instrument | Resolved library/asset identities, schedule, constant data, preallocated state, capacities |
| Voice instance | Stable identity, ownership scope, event state, process state, motion instances |
| Process state | Algorithm-specific phases, cursors, buffers, modes, or solver state |
| Execution profile | Required rate, latency, tail behavior, memory/work bounds, supported backend capabilities |
| Snapshot | State plus prepared-definition identity, pending events, clocks, delays, random/provider state |

Use tagged types rather than a giant class of optional fields. Share contracts
and appropriate kernels, not meaningless fields: a sampler need not allocate
FM phase arrays, and a modal body need not pretend to have a sample cursor.

Keep three port families: audio blocks with channel layouts; control signals
with units and an evaluation policy; timestamped events with typed payloads.
Analysis/spectral frames can be specialized typed ports when a real spectral
engine needs them, with explicit hop size, window, and latency.

A control stream may run at audio rate, but does not silently become an audio
connection or trigger. The proposed [motions](a-new-view-of-lfos-and-envelopes.md)
and [segments](segments.md) supply control behavior and readable time notation.

### Preparation and scheduling

Compile constant data once: topology, incoming edge lists, operator codes,
asset references, routing matrices, parameter dependencies, and resource bounds.
Build an acyclic schedule between stateful components. Feedback edges have an
explicit positive delay and an initial condition; never use block size as the
meaning of a delay.

An algorithm with instantaneous internal coupling can own its own specified
solver, convergence policy, and cost. The generic graph scheduler must not
invent an algebraic-loop solver.

Allow whole-block kernels where possible, small fixed chunks where useful, and
per-sample execution only where recurrence requires it. Fuse common simple
paths after measuring them. Reuse immutable tables/assets, but never share
mutable voice state accidentally.

Signals at different sample rates need explicit resampling boundaries. Clock,
latency, and group delay must remain understandable when oversampled processes
are mixed with ordinary ones. Snapshot semantics include those boundaries.

### Lifecycle and failure

Use activation, excitation release, tail, and finished as distinct concepts.
A process declares whether it has a finite authored duration, a known tail
bound, an energy-based completion rule, or indefinite sustain. Energy-based
retirement needs a stated threshold and hold duration; it is an audible policy.

An instrument chooses its lifetime owner and how child tails are collected.
Voice stealing is an explicit bounded transition, not arbitrary immediate state
erasure. Shared bodies may continue ringing after contributing voices end.

Validate input contracts before callbacks. Diagnostic errors identify the
instrument, voice, process, parameter, and frame. After a numerical failure,
define whether the host stops, restores a checkpoint, or restarts the affected
instrument; clearing an error bit is not state recovery.

## 5. Useful new engines, reorganized around reusable work

These are candidates, not simultaneous commitments.

| Direction | Musical value | Reuse and difficult part | Relative effort |
| --- | --- | --- | --- |
| Bandlimited virtual analog | Better bright leads, bass, PWM and sync | Existing phase engine; discontinuities under modulation, nonlinear oversampling | Medium |
| Morphing wavetable | Evolving periodic timbres | Asset preparation and phase reader; pitch-indexed bandlimiting, phase-aligned morphs | Medium |
| Modal body | Bells, struck materials, sympathetic resonance | Shared excitation and controls; modal coefficients, tuning, tails, energy | Medium |
| Plucked/bowed waveguide | Strings, tubes, unusual resonant objects | Noise/sample excitation; fractional delay, tuning, nonlinear interaction and stable feedback | Medium for pluck, high for bow/breath |
| Granular instrument | Transform recordings into playable clouds | Existing grain/history work; pitched excitation, immutable assets, allocation and scheduling | Medium-high |
| Additive/partial field | Organs, inharmonic tones, independently moving partials | Shared phase/control infrastructure; bounded partial count and Nyquist policy | Medium |
| Spectral resynthesis | Freeze, morph, hybridize recorded timbres | New analysis/frame engine; latency, transient handling and phase coherence | High |
| Formant instrument | Vowels, breath, choirs | Prefer an excitation-plus-resonator recipe first; distinguish formants from pitch | Low-medium after resonators |
| Feedback instrument | Self-sustaining, unstable-sounding but bounded textures | Explicit delay cycles, drive and pickup; stability and headroom | Medium-high |
| Corpus instrument | Play neighborhoods of recorded gestures | Prepared feature index and bounded selection; continuity and search latency | High |

“Modal” and “waveguide” are not simply two presets of the existing output filter.
They carry a different state, energy, and lifetime contract. Conversely,
“drum kit” is often an instrument recipe, not a new DSP engine.

A granular effect consumes ongoing audio/history, while a granular instrument
can read an immutable sample immediately at note-on. Their scheduling/windowing
can share implementation without conflating those different input/lifetime
contracts.

Prefer **modal bodies** as the first genuinely contrasting new family after
correctness/quality work. They connect the existing noise, sampler, and FM
sources to physical and experimental synthesis. Wavetable remains the strongest
compact expansion of familiar oscillator sounds.

## 6. More adventurous ideas

These are speculative design seeds. Their relationships to established methods
are stated deliberately; attractive naming is not evidence of a new technique.

### Phase weaving

Several oscillators exchange phase influence through musical relationships:
attract, repel, divide, and temporarily lock. A player controls cohesion and
friction rather than individual modulation indices. Outputs include audible
taps and synchronization events that can drive motions.

Prototype against coupled-oscillator/PM baselines. Preserve phase when changing
relationships, bound coupling, and avoid claiming that nonlinear locking will
always preserve keyboard tuning.

### Event grammar synthesis

Make the basic object a rule for producing sound events. A strike can branch
into quieter descendants, which excite short wavelets or grains. Rules control
rhythm, pitch interval, decay, and whether an event reproduces.

Sparse activity sounds like gestures; dense bounded activity becomes texture
or pitch. A deterministic scheduler, positive event delays, and an explicit
event/energy budget are essential. Compare it to stochastic granular synthesis
before giving it a separate engine.

### Counterfactual resonance

Maintain several related resonant bodies representing alternative responses to
the same excitation: softer material, different tension, different boundary.
A control chooses how their states exchange energy rather than merely
crossfading their output signals.

The interesting hypothesis is that exchanging internal state can produce
transitions unavailable from a static output crossfade. A prototype must specify
how states are mapped and avoid injecting energy by arbitrary interpolation.

### Constraint synthesis

The musician specifies tendencies such as “a stable fundamental, wandering odd
partials, and an attack whose brightness falls faster than loudness”. A bounded
solver steers a partial field toward those constraints.

Keep a deterministic reference and expose constraint conflicts. Start with
closed-form projections or preparation-time solutions, not an unconstrained
optimizer in the audio callback. The result may be better understood as a
composition tool for additive synthesis than a separate live engine.

### Topology as an instrument control

A gesture changes which string touches which body, which FM operator feeds
which carrier, or which partial exchanges energy with its neighbors. Prepare a
bounded set of possible connections, then vary their weights or switch through
declared transitions.

Do not allocate new graphs in the callback. Arbitrary phase/state crossfading
is not universally safe; each process declares what a transition preserves.

### Recorded silence as a resonant material

Analyze the noise floor and tiny resonances of a recording, then excite that
structure with notes. A room, tape hiss, or a nearly silent object becomes a
playable material. This is a creative application of analysis/resynthesis, not
a claim that silence contains a hidden musical scale.

## 7. Detailed experiment: imprint synthesis

### Musical idea

**An instrument that remembers being played, through changes in how its
resonances interact.**

A loud low note leaves a slow-decaying imprint. A later high note travels
through the changed resonant network and acquires a different attack or spectral
motion. A performer can polish a sound by repetition, erase its history, freeze
it as a material, or let it recover between phrases.

The history is structural, not an audio recording. It need not audibly replay
previous notes. Existing resonant energy can ring separately from the persistent
imprint, allowing either or both to be reset.

This is a proposed original instrument design, **not a verified claim of a
previously unknown synthesis method**. It is related to modal synthesis,
state-dependent coupling, scattering networks, and hysteretic systems.
Research on nonlinearly interconnected modal networks already exists. The
distinctive hypothesis is a musically useful, explicitly playable slow memory
of past excitation governing bounded energy exchange.

### A concrete numerical starting point

Use a fixed bank of M complex resonator states z[i], with nominal frequencies,
losses, and pickup weights. Connect selected pairs with a fixed list of edges.
Each edge also stores a slow imprint h[e] in [0, 1].

For every internal audio sample:

1. Apply the timestamped bounded excitation to selected resonators.
2. Rotate each resonator by its phase increment and apply a decay factor between
   zero and one.
3. Exchange state along edges using a specified sequence of pair rotations.
4. Read audio from the real part of a weighted sum of resonator states.
5. Update the slow imprint from post-exchange local energy, using a bounded
   leaky accumulator; use it for the next sample's coupling.

For an edge joining i and j:

~~~text
a = z[i]
b = z[j]
z[i] =  cos(theta) * a + sin(theta) * b
z[j] = -sin(theta) * a + cos(theta) * b

theta = (base_rate + depth * h[edge]) / internal_sample_rate
h_next = (1 - alpha) * h + alpha * drive(local_energy)
~~~

drive is a specified saturating nonnegative function with range [0, 1];
one candidate is energy / (energy + reference_energy), with a positive declared
reference energy. For memory time tau, use alpha = 1 - exp(-1 / (rate * tau)).
Define coupling strength in radians per second and divide by the internal sample
rate to obtain the pair angle. Changing oversampling must not accidentally
multiply the speed of imprint formation or coupling. If imprint formation and
forgetting need different times, use explicitly separate attack/recovery
coefficients. Every sample reads the old imprint values before committing the
next ones.

The pair operation is an orthogonal rotation, also norm-preserving for complex
a and b. Therefore changing theta redistributes sum(abs(z[i])^2) without increasing
it in exact arithmetic. Decay decreases that norm; excitation introduces energy.
This gives a tractable stability starting point, not a guarantee against
aliasing, large output peaks, or floating-point errors.

Edge ordering is part of the prepared definition: rotations on overlapping pairs
do not generally commute. Coupling also changes the composite modes, so nominal
oscillator frequencies do not by themselves guarantee perceived pitch. Both
facts require musical calibration and exact snapshot/partition tests.

With zero external excitation, bounded pair rotations and decay cannot increase
the state norm. With repeated excitation, impose a documented drive budget or
bounded excitation with a uniform decay bound strictly below one to obtain a
state bound. Normalize pickup weights by
a stated norm and meter output; do not silently normalize each rendered block.

### Parameters a musician could understand

| Control | Meaning |
| --- | --- |
| Material | Nominal mode distribution, losses, and allowed connections |
| Strike | Where/how excitation enters |
| Impression | How strongly recent energy writes the imprint |
| Memory | How quickly the imprint fades |
| Mobility | How strongly imprint changes exchange between resonances |
| Damping | How quickly the audible energy dies |
| Pickup | Which resonances are heard, including spatial outputs |
| Erase | Reset the imprint independently of the ringing energy |
| Freeze | Stop imprint changes while sound evolution continues |

Proposed sketch, not supported TOML:

~~~toml
[sound]
kind = "imprint"
material = "glass-thread"
modes = 32
memory = "8 s"
impression = 0.6
mobility = 0.25
damping = "2 s"
memory_scope = "instrument"
~~~

A voice-scoped imprint forgets when that voice is retired unless explicitly
saved. An instrument-scoped imprint survives note releases and is part of the
instrument snapshot. Reset-on-transport behavior must be authored, not accidental.

Two possible implementations need comparison: independent resonant voices
sharing only an imprint, or one shared resonant body receiving all note
excitations. The first gives clearer note ownership; the second permits true
sympathetic interaction but needs a tuning/excitation map for chords. Start with
one shared body and a small fixed pitch set for research, not a promised
fully chromatic instrument.

### Why it might sound different

A conventional envelope depends mainly on the current note. A fixed resonator
depends on its present ringing state. Here a slower material state survives
after that ringing disappears. The same isolated note can sound different after
different preceding phrases, even when its initial audible state is zero.

Possible sounds: glass that grows breathier as it is played, metal whose attack
changes after repeated strikes, a bass sound with phrase-dependent harmonic
paths, or two players sharing a material that each reshapes for the other.
These are hypotheses, not listening results.

### Research prototype and rejection criteria

Build a reference experiment only after the design is selected. Use 16 or 32
modes, a fixed sparse edge list, bounded excitation, one pickup, and explicit
oversampling if needed. Work and memory are O(M + E); no learning system,
dynamic graph growth, or unbounded search is needed.

Compare matched-level renders of:

- Fixed modal body with no imprint.
- Fixed coupling and the same damping.
- Imprint-controlled coupling.
- The same coupling variation driven by unrelated slow noise.
- Imprint on, audible state reset before a probe note; then imprint erased and
  the same probe repeated.

The last comparison isolates structural memory from an ordinary lingering tail.
Ask listeners whether phrases create recognizable, controllable differences,
not just whether one version sounds more complicated. Measure pitch drift,
spectral change, aliasing, peak level, state energy, and worst-case CPU.

Reject or redesign it if its useful range is tiny, if pitch becomes unplayable,
if it sounds equivalent to random filter modulation, or if its history is too
unpredictable to control. A convincing failure is more useful than promoting a
new engine because its name is appealing.

## 8. Quality, performance, and safety evaluation

Do not equate NumPy/Rust agreement with correctness: both can implement the same
aliasing or lifecycle mistake.

Use independent analytical cases where possible, oversampled reference renders
where meaningful, exact event/state assertions, and matched-level listening.
Retain deterministic WAV regressions at 48 kHz lasting at least one second, plus
short musical demonstrations for perception. Hardware callback tests are
separate evidence from offline tests.

A useful matrix includes:

- Pitch sweeps, high notes, extreme pulse widths, fast pitch/phase modulation,
  large FM indices, and feedback.
- Bright sample transposition, loop seams, mirror turns, release inside a loop
  overlap, and source/output sample-rate differences.
- Releases during attacks and before minimum hold, zero-time segments, stops,
  voice stealing, long ringing tails, and empty/exhausted sources.
- Many voices with shared motions versus unique motions, filters, modulation
  fan-out, and long performances with repeated snapshots.
- Tiny/irregular blocks, simultaneous events, capacity boundaries, invalid
  native inputs, incompatible snapshots, and recovery after partial failures.

Report worst-case block time against its deadline, high-percentile timing,
allocation counts, memory growth, and snapshot cost. Distinguish oscillator
cost, modulation cost, asset traversal, graph overhead, and encoding. Benchmark
the actual native callback route rather than an offline adapter with a similar
name. Denormal behavior and solver stability under rapid coefficient changes
need dedicated checks; no defect is asserted here without that evidence.

Each quality mode needs a precise contract: which processing is oversampled,
resampler/filter design, latency, feedback delay interpretation, and transition
rules. Keep gain staging explicit and diagnose non-finite samples before output.

## 9. Suggested order of work

1. Reproduce and repair sampler motion lifecycle and minimum-hold named-envelope
   behavior. Add boundary, invalid-input, and live recovery tests.
2. Remove obvious per-sample allocations and prepare graph/parameter adjacency.
   Measure before promising a speedup.
3. Establish source/engine capability reporting and explicit voice-tail semantics.
   Review native coverage, old fixed-FM duplication, and backend-specific limits.
4. Improve oscillator and sampler quality under an explicit policy. Treat sound
   changes as authored/versioned behavior, not unnoticed “optimizations”.
5. Introduce the common process contract around existing engines incrementally.
   Prove one mixed instrument using existing noise, sampler/oscillator, motions,
   and processing before designing a large catalog of new node types.
6. Add a modal body and a few expressive library instruments. Then choose
   wavetable, waveguide, or granular instruments according to desired sounds.
7. Explore imprint synthesis as a bounded research prototype with the rejection
   criteria above. Promote it only if it earns a clear musical identity.

The architectural steps require a separate implementation decision. This plan
does not authorize a repo-wide refactor, dependency changes, or new engines.
The findings are actionable independently of the larger architecture.

## 10. Technical precedents

These sources inform the proposals; they do not establish novelty or validate
our implementation.

- Csound's [vco2](https://csound.com/manual/opcodes/vco2/) documents bandlimited
  oscillator generation, relevant to improving the current raw waveform path.
- Csound's [poscil](https://csound.com/manual/opcodes/poscil/) provides an established
  table-oscillator reference for wavetable work.
- Csound's [partikkel](https://csound.com/manual/opcodes/partikkel/) combines granular
  techniques with per-grain control and synchronization, illustrating how one
  scheduling engine can support many musical forms.
- Csound's [pvsmorph](https://csound.com/manual/opcodes/pvsmorph/) illustrates spectral
  interpolation; a spectral engine still needs analysis and phase/latency contracts.
- [Real-time modal synthesis of nonlinearly interconnected networks](https://www.dafx.de/paper-archive/2023/DAFx23_paper_17.pdf)
  is directly relevant prior work for interacting resonators. Imprint synthesis
  must be evaluated against this broader family, not claimed as unprecedented.
- [Antialiasing Piecewise Polynomial Waveshapers](https://www.dafx.de/paper-archive/details/cu51KU_JsiXj1cvMi-2DOw)
  is relevant to discontinuities and nonlinear processing, not a universal fix
  for every PM or feedback topology.

## Additional work beyond the prompt

None. This change updates the engine review/design document only.
