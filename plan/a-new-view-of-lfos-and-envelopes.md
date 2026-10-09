# A new view of LFOs and envelopes

Design proposal, 2026-09-28. This document proposes a shared model; its TOML
examples are proposed syntax, not files that today's engine can load. It does
not authorize implementation or change the existing roadmap.

## 1. Recommendation: call them motions

A **motion** produces control signals over time, can receive events, and can
emit events of its own. It can be as simple as a sine wave or as involved as a
collection of interacting processes. An envelope and an LFO are familiar kinds
of motion. They should share ownership, connections, clocks, event handling,
libraries, and runtime state.

Use `motion` in the format, `Motion` in data types, and “motions” in conversation:
“Give the filter a slow motion”, “load the bowed motion”, “reverse this motion”.
Keep envelope and LFO in descriptions, search tags, and documentation. A
musician should not have to unlearn those words to use the system.

Other names considered:

| Name | Strength | Limitation |
| --- | --- | --- |
| Motion | Short; includes periodic, triggered, and reactive behavior | Needs a one-sentence introduction |
| Modulator | Established synthesizer terminology | Also used for FM audio operators; suggests a destination is necessary |
| Gesture | Expressive and musical | Suggests a finite performance rather than a continuous process |
| Contour | Good for a shaped trajectory | Too narrow for followers, randomness, and external programs |
| Function | General, with modular-synth precedent | Abstract; suggests a stateless calculation |

“Motion” is the working recommendation, not an implementation dependency.

The crucial qualification: **unify the contract, not every implementation**.
A waveform reader, an envelope follower, and an external program do not have
the same internal state. Making all of them pretend to be a list of envelope
points would make the system less expressive and harder to understand.

## 2. What we already have

The current ufor definitions separate `LFO`, `Envelope`, and autonomous `Curve`.
LFOs already have phase, rate, scope, delay, and fade-in. Envelopes already have
arbitrary segments, release segments, hold behavior, curves, and retrigger
policies. The modulation system already separates sources, mappings, units,
and parameter targets. These are useful pieces to retain.

enge currently renders seconds-based generators. Sound envelopes use held
linear segments; named modulation envelopes can use curves offline, while the
persistent native path requires linear segments. Named envelopes and instrument
LFO rate/reset actions now have native runtime support. Stateful continuation
already belongs in snapshots.

We also already have a library foundation, not a blank slate:

- `LFOScore` and `EnvelopeScore` are registered score types and can be library
  entries.
- `Library` resolves names, tags, relative references, and optional hashes. It
  detects ambiguity and dependency cycles.
- `PresetScore` configures public parameters of interface scores.
- Instrument settings currently embed separate `lfos` and `envelopes`
  dictionaries. They do not offer a unified motion library instance with a
  shared event/parameter interface.

The proposal extends those facilities. It does not introduce a second library
resolver, an unrelated preset format, or another instrument modulation router.

Source anchors: `ufor/lfo.py`, `ufor/envelope.py`, `ufor/modulation.py`,
`ufor/library.py`, `ufor/interface.py`, `ufor/preset.py`, and `enge/synth.py`.
The ufor paths refer to the sibling ufor repository.

## 3. The musician's model

A motion answers five questions:

1. **What does it produce?** A waveform, contour, sequence, random movement,
   measurement, recording, or external calculation.
2. **What moves it?** Elapsed time, musical time, incoming pulses, or another
   signal controlling its position.
3. **What happens when something happens?** Start, release, enter a stage,
   reverse, jump, pause, change a parameter, or emit another event.
4. **Who owns it?** A voice, trigger, part, or instrument.
5. **Where do its outputs go?** Sound parameters, another motion, or an event
   destination.

Most motions only need the first two answers. The rest have documented defaults
or appear at the point where the motion is connected to an instrument.

There are two port families:

- **Signals** have a value at each evaluation time: amplitude, position, rate,
  brightness, filter cutoff, or a gate level.
- **Events** happen at a particular time: note-on, marker crossing, cycle end,
  release, beat, or a named cue. Events can carry typed payloads.

Neither is silently converted into the other. A threshold detector turns a
signal into events; a pulse or latch turns events into a signal. This avoids
the ambiguity of calling both a one-sample impulse and a timestamp a “trigger”.

Every motion exposes a primary signal named `value`. Compound motions may expose
additional named signals such as `brightness` and `pressure`, and named events
such as `peak` and `done`. Common lifecycle events have one shared meaning.

## 4. Keep the first examples small

These are standalone score documents. Existing common score header defaults
apply; no new format version number is chosen by this proposal.

### A simple LFO

```toml
kind = "motion"
name = "slow-sine"
title = "Slow sine"

[body]
kind = "cycle"
shape = "sine"
rate = "5 Hz"
```

This starts when its instance activates, has phase zero, runs forwards, loops
indefinitely, and produces -1 to +1. A voice instance activates at voice onset;
an instrument instance activates at the instrument's timeline origin. There is
no hidden dependence on the first audio block requested.

### A simple envelope

```toml
kind = "motion"
name = "soft-pluck"
title = "Soft pluck"

[body]
kind = "contour"
initial = 0.0
segments = [
  { duration = "10 ms", to = 1.0 },
  { duration = "100 ms", to = 0.6 },
]
release = [{ duration = "200 ms", to = 0.0 }]
```

By default this waits for `note_on`, traverses its segments, holds the last
value, and enters release on `note_off`. Release starts at its current value.
Completion holds the final value and emits `done` exactly once. An instrument
may use that event for voice retirement; an arbitrary modulation envelope must
not kill the voice just because it finishes.

The musician writes neither a graph nor a state machine for either example.
`cycle` and `contour` are distinct useful body types with the same external
contract. A contour without release is an autonomous one-shot: activation
starts it and reaching its end completes it. An explicit `playback` setting can
instead loop or reflect it. These defaults are selected by the documented body
variant, never guessed from the instrument target.

### Units and simple defaults

Use a small, specified unit grammar, not an expression language:

- Durations: `"10 ms"`, `"0.2 s"`, `"1/4 beat"`.
- Rates: `"5 Hz"`, `"2 cycles/beat"`.
- Positions: numeric fractions of a cycle or contour, with named markers for
  musically meaningful places.

Parse decimal and fractional time literals exactly. Beats require a named clock
binding. Define a beat as a quarter note; bar lengths come from the clock's
meter map, never an assumed four beats. The first grammar need not include bars
or note-length aliases. Numeric signal values carry units in their port or
parameter declarations, not in every sample.

Defaults: linear segments; forward playback; no smoothing; no event emission
from a discontinuous seek; no implicit clipping or unit conversion. Existing
waveform phase and duty-cycle conventions remain authoritative.

## 5. An envelope that becomes an LFO

This is naturally a motion with named stages. The stages say what to do; the
transitions say when to change behavior. It is one instance with one `value`
output throughout.

```toml
kind = "motion"
name = "bloom"
title = "Attack, sway, release"

[body]
kind = "stages"
initial_stage = "waiting"

[[body.stages]]
name = "waiting"
motion = { kind = "hold", value = 0.0 }

[[body.stages]]
name = "attack"
motion = { kind = "contour", initial = "current", segments = [
  { duration = "300 ms", to = 0.7 },
] }

[[body.stages]]
name = "sway"
motion = { kind = "cycle", shape = "sine", rate = "3 Hz", center = 0.7, depth = 0.2 }

[[body.stages]]
name = "release"
motion = { kind = "contour", initial = "current", segments = [
  { duration = "500 ms", to = 0.0 },
] }

[[body.transitions]]
from = ["waiting", "attack", "sway", "release"]
event = "note_on"
action = { kind = "enter", stage = "attack" }

[[body.transitions]]
from = ["attack"]
event = "stage.done"
action = { kind = "enter", stage = "sway" }

[[body.transitions]]
from = ["attack", "sway"]
event = "note_off"
action = { kind = "enter", stage = "release" }

[[body.transitions]]
from = ["release"]
event = "stage.done"
action = { kind = "finish" }
```

`current` captures the parent's output immediately before entering the stage;
it is a special start-value token, not embedded code. Stage entry resets the
child's local position. Only the active stage runs. At the natural attack end,
the output is 0.7 and the sine starts at its center, so that transition is
continuous. Note-off during either attack or sway releases from the actual
current value.

Repeated note-ons retrigger the attack here. Binding a footswitch cue to
`enter(sway)` or `enter(release)` provides extra triggers for different stages.
For a polyphonic patch, repeated notes normally create separate voice instances;
sharing this progression across notes requires a part or instrument instance.

This example also exposes a useful distinction: *an envelope followed by an
LFO* changes the active stage; *an envelope fading in an LFO* multiplies two
simultaneous signals. Both are needed, and neither should be encoded as a
special case in the other.

## 6. Playback, triggers, and movement through a shape

Separate the shape from the mechanism moving through it. A cycle reads a
periodic shape; a contour reads a bounded trajectory; a recording reads stored
values. These bodies can share a position driver.

| Command | Meaning |
| --- | --- |
| `start` | Begin from the declared initial state |
| `release` | Enter a declared release behavior from the current value |
| `enter` | Enter a named stage, resetting its child and capturing entry values |
| `pause` / `resume` | Freeze/resume local advancement; incoming events still work |
| `reverse` | Change the sign of local travel without changing position |
| `seek` | Set an absolute position or named marker |
| `shift` | Move by a relative position, including negative distances |
| `set` | Change a declared input parameter |
| `finish` | Complete once and hold the current output |

Commands are tagged data with explicit arguments. `seek(position=0.75)` is
not the same operation as `enter(stage="release")`. Only bodies supporting
position accept position commands; sending `reverse` to a follower is a
preparation error rather than a silently ignored message.

Playback policies are `once`, `loop`, and `ping_pong`, with optional named
loop boundaries and a finite repeat count. A repeat count means completed
traversals; a ping-pong traversal is one leg, with `turned` at each endpoint
and `cycle` after the complete outward-and-back journey. Loop and turn boundary
rules must be shared by all readers.

Signed rate supports continuous slowing, stopping at zero, and reversing.
This is a signed playback-rate type, separate from the existing strictly
positive Hz domain used for sound-frequency targets. A rate change preserves
position. Changing direction does not retrigger or restart delay/fade behavior.
For a periodic reader, the unwrapped coordinate continues counting positive
or negative turns; the shape is sampled at its fractional phase. Contours
reverse through their authored shape, including the curvature of each segment.
For a contour starting from `current`, retain the captured entry value when
reversing that traversal. Do not recapture it on every direction change.

A position input, such as `{ source = "scrub.value" }`, can replace an internal
rate driver. It must not coexist with another position owner. A pedal can then
scrub an envelope or freeze it anywhere. A transport-locked motion similarly
has an externally owned position: it follows seeks and reversals of that clock;
local seek/reverse commands are rejected until its driver is explicitly changed.

There is an essential limit: moving backwards through a shape does **not** undo
events it already emitted, reverse an external process, or restore a previous
world state. Rewinding a performance requires snapshot restore and event replay.
An arbitrary live input has no past to revisit unless it was recorded.

Discontinuous jumps may change the output abruptly. Make optional transition
smoothing explicit, with a duration in seconds. It blends from the captured
old output while the destination advances normally. Jump smoothing changes
signal output, not the timestamps or traversal meaning of marker events.

## 7. Signals, event outputs, and connections

Implemented voice-owned Patch `latch` nodes: capture a named input on activation,
then hold until an event connection delivers `capture`. Observation occurs on
the first sample frame at or after delivery, including connection delay; several
commands on one frame observe the same final input. Existing Cycle, Stages, and
threshold event sources retain division, probability, delay, and capacity rules.
Python and native snapshots preserve held values and queued captures; stopping
invalidates pending commands and voice-slot reuse initializes fresh state.
One-second 48 kHz regressions cover startup, independent gates, fractional-frame
and zero delay, shared inputs, partitions, snapshots, and voice-slot reuse.

Implemented stateless Patch `quantize` nodes: positive finite `step`, finite
`origin` defaulting to zero, and nearest-grid rounding with halfway values
choosing the higher point, including negative inputs. Output is not clipped;
range inference quantizes both endpoints. Python and native graphs share these
semantics for voice, part, and instrument scopes. One-second 48 kHz regressions
cover shifted grids, exact ties, wider outputs, shared consumers, partitions,
and snapshot continuation.

Implemented voice-owned Patch `slew` nodes with independent rise/fall limits in
signal units per second. Activation starts at the input's value unless an
explicit initial value requests a ramp; movement begins on the next sample and
does not overshoot. Zero prevents movement in that direction. Slew continues in
seconds even when a beat-clock input stops. Shared consumers read one state,
and Python/native snapshots retain the previous output. One-second 48 kHz
regressions cover startup, asymmetric and zero rates, interrupted ramps, output
ranges, stopped clocks, partitioning, snapshots, and voice-slot reuse.

Implemented voice-owned Patch `threshold` nodes with lower/upper hysteresis,
a [0, 1] gate, and named `rising`/`falling` events. Activation is silent;
subsequent jumps can fire. Detectors observe the sample-frame grid, not estimated
between-sample crossings. Commands arrive on the following frame plus their
seconds delay, using existing division/probability gates and bounded queues.
Python and native snapshots retain detector state and pending commands.

Implemented Patch signal transforms: named `sum`, `product`, and `affine`
children form a preparation-checked dependency DAG. Python and persistent
native synth evaluate shared child state in dependency order without clipping.
Output source domains must cover conservative inferred ranges. Cycle onset
weights scale transform inputs toward zero; the transformed output has weight
one. Regression coverage includes envelope-shaped vibrato, seconds and beat
clocks, tempo changes, release, partitioning, and exact snapshot replay.
Contour input bindings retain explicit voice-timed release even when they have
no direct parameter route; unbound input contours default to immediate release.

Reuse existing typed modulation routes for signal-to-parameter connections.
A motion's rate, depth, phase offset, curve, and other declared inputs are
ordinary parameter targets. Multiple additive/multiplicative contributions
retain ufor's current combination and domain rules.

Add typed event connections alongside those routes. A connection identifies a
source event port, destination instance, and command, optionally mapping named
payload fields into typed command arguments. Do not place executable strings
inside the document.

Event connections support `every`, a positive integer defaulting to 1. Division
is first-aligned: `every = 3` forwards matching events 1, 4, 7, and so on. Each
connection also accepts `offset`, a nonnegative integer defaulting to 0, which
skips that many matching events initially before forwarding every nth event.
Offsets 1 and 2 with `every = 3` forward 2, 5, 8 and 3, 6, 9. Offsets can exceed
the divisor: offset 4 starts at event 5, not event 2. Each connection counts
independently for each voice, resets on voice activation, and
retains its count through child restarts, stage transitions, render partitions,
and snapshot restoration. Division filters delivery, not source emission or
named event outputs. This applies to Patch starts and cues and to instrument
Motion cue connections in both reference and native rendering.

Connections also accept `probability` in [0, 1], defaulting to 1. Apply it after
division: each selected event draws independently per connection and voice;
rejection does not shift the divider. Probabilities use 53-bit integer cutoffs;
cutoffs zero and 2**53 consume no random words. Prepared voice starts carry a
Motion key derived from the trace seed and voice identity. Connection identity
selects an independent SplitMix64
stream; snapshots preserve its state and voice activation initializes it. Child
restarts do not initialize it. Matching score seed, voice IDs, and authored
connection order reproduces results across partitions and Python/native
renderers. Native preparation encodes the 64-bit key as two exact 32-bit halves
before voice start, using one additional action row for probabilistic voices.

Connections accept `delay`, a nonnegative rational number of seconds defaulting
to zero. Division and probability gates run at source emission; accepted commands
are delivered once at emission time plus delay. Pending commands survive render
partitions and snapshots, remain active through the voice's release tail, and are
discarded when that voice stops or retires. Both renderers cap pending delayed
commands at 4096 per voice and report overflow. Native preparation reserves this
queue only for voices with delayed connections; enqueueing does not grow it.
At a shared instant, natural source events precede delayed commands. Delayed cues
with the same deadline retain enqueue order; Contour starts retain their existing
time/connection ordering. Zero delay keeps immediate
delivery. A native deadline that cannot advance or remain finite at its floating
frame coordinate is an error, not an immediate delivery.

For example, within a proposed compound `patch` body:

```toml
[body]
kind = "patch"

[body.motions.wobble]
kind = "cycle"
shape = "sine"
rate = "2 Hz"
markers = [{ name = "peak", position = 0.25, direction = "forward" }]

[body.motions.accent]
kind = "contour"
initial = 0.0
start = "event"
segments = [{ duration = "80 ms", to = 1.0 }, { duration = "120 ms", to = 0.0 }]

[[body.events]]
source = "wobble.peak"
target = "accent"
action = { kind = "start" }

[body.outputs]
value = "accent.value"
```

Here `start="event"` overrides the autonomous contour's activation default.
References such as `wobble.peak` have the fixed grammar `instance.port`; instance
names cannot contain dots. They are references, not expressions.

Useful event sources include:

- Named shape positions, stage entry/exit, completion, wraps, and turns.
- Signal threshold crossings with direction, hysteresis, and optional minimum
  time between firings.
- Counters, divisions of an incoming pulse train, probabilistic gates, and
  comparisons of typed event payloads.

Marker traversal uses an open starting boundary and closed arrival boundary,
in either direction. Wrapping owns a boundary crossing once. Turning at an
endpoint must not emit the same marker twice without further travel. A seek
does not invent crossings by default; an explicit `crossings="emit"` requests
all crossed markers in travel order and is subject to event capacity limits.
Landing on a marker without traversal can instead use a separate `arrived`
event. Define this once, not separately in each waveform kernel.

Connections can broadcast from a shared motion to matching voices. Connecting
many voices back to one shared signal requires a declared reduction, such as
sum, maximum, latest, or average. Event fan-in requires a defined merge order.
No implicit “whichever voice ran last” behavior.

Signal feedback needs an explicit positive-time delay with an initial value;
its unit is frames or a specified clock duration, never “one render block”.
Same-time signal dependencies must form an acyclic graph. Immediate event cycles
and zero-duration stage cycles are rejected. Musically useful event feedback
is allowed through a positive delay, with a declared maximum event rate.

## 8. Clocks and external synchronization

Clocks are shared services with position, rate, running state, and discontinuity
events. A motion selects or binds a clock; it does not open a MIDI port itself.
The host adapts physical or application inputs into timestamped clock/events.

Keep these separate:

| Relationship | Musical behavior |
| --- | --- |
| Rate follows tempo | Change speed with tempo, preserving local phase |
| Position follows transport | Stay at a defined song position, including host seeks |
| Restart on pulse | Every selected note-on or beat restarts the shape |
| Advance on pulse | Every selected event advances a step or stage |
| Lock to pulse train | Estimate period and correct phase between observed pulses |
| Quantized command | Queue a command until the next selected beat division |

Repeated note-ons can therefore restart an LFO, tap its tempo, advance it one
step, or select a stage. These must be distinct settings. They answer different
musical intentions.

Clock adapters may accept host tempo maps, MIDI clock, transport messages,
tap tempo, detected onsets, OSC cues, or another motion's events. The MIDI
adapter owns protocol pulse counting and transport conversion; the motion
only sees a normalized musical clock.

For pulse estimation, declare the acquisition rule, smoothing/phase-correction
policy, outlier handling, timeout, and behavior on loss: hold position, continue
at the last tempo, or stop. These belong to the chosen clock adapter. Record
its resolved clock observations so offline playback does not depend on arrival
jitter. Predictions made before the next live pulse cannot be retroactively
corrected; live capture and offline replay use the same accepted observations.

Seconds-based release can remain 200 ms while a beat-based sustain loop follows
tempo. Stage durations carry their own units; beat-based durations bind to a
clock. A paused musical clock freezes beat progress but not independent seconds
progress. Whole-instance pause is a separate command.

For quantization, a command exactly on a grid boundary executes there; otherwise
it targets the next boundary. Pending beat commands follow tempo-map changes.
Seek/stop behavior is explicit: cancel, retain musical target, or re-quantize.
No hidden choice based on the size of the next audio block.

## 9. More musical possibilities

The following are design use cases, not a request to implement every feature
in the first release.

| Family | Ideas and the pieces that express them |
| --- | --- |
| Evolving gestures | Attack into looping sustain; release after the next loop; several note-offs exposing successive release stages |
| Playable movement | Pedal scrubbing; velocity choosing an entry stage; pressure slowing or reversing travel; legato continuation |
| Rhythmic patterns | Step sequences; Euclidean event patterns; ratchets; swing; alternating lengths; event counting and clock division |
| Variation | Seeded sample-and-hold; smooth noise; random walks with boundaries; weighted stage choices; probability changing with velocity |
| Coupled movement | One motion changes another's speed; quadrature outputs; polymetric loops; explicit delayed feedback and phase coupling |
| Reactive movement | Audio-level followers; pitch/onset analysis; pressure, breath, or sensor inputs; threshold-triggered contours |
| Memory | Capture a played control gesture; repeat or stretch it; hold the value present at a marker; latch a peak until a cue |
| Structured expression | Morph between two shapes at the same position; bias/skew a contour; quantize to musical pitch intervals; smooth only falling edges |
| Physical models | A damped spring after each trigger; bouncing/settling gestures; resonant responses to impulses |
| Multi-output gestures | A bow stroke producing pressure, brightness, and vibrato depth from shared progress; correlated spatial trajectories |
| Performance structure | Evolve across a phrase or song section; reset on a bar cue; transition at a zero crossing to reduce discontinuities |
| External intelligence | A program generates motions from analysis, a simulation, a model, or an external sequencer |

Distinguish independent voices from intentionally correlated variation. A random
seed can derive from the score seed, motion identity, and stable instance key.
Implemented the first variation primitive: voice-owned `sample_hold` draws on
activation and on connected Patch `sample` commands. Cycle markers and Stages
ports use the existing division, probability, and seconds-delay rules. Each
named child owns an independent stream; signal transforms can consume held
values without drawing again. Seeks, reversals, pause/resume, and note-off do
not rewind or redraw. Python and persistent native synth retain the value,
stream, and pending deliveries in snapshots, including across voice-slot reuse.
Direct bindings also use the shared native setup in FM and noise. WAV
regressions cover those paths and both event-source types at 48 kHz.

For reversible random shapes, derive each value from seed plus coordinate/index;
reverse revisits the same values. Stateful random walks and physical models
instead need recorded history or replay to revisit their past.

An oscillator with a sufficiently high rate is still a motion, but its target
may require audio-rate evaluation and appropriate band-limiting. The type system
must not promise that every control-rate process is a safe audio oscillator.

## 10. Proposed data types

These are schema-level contracts, not Python implementations. Use frozen
Pydantic data classes where practical, discriminated unions for alternatives,
and lists/dictionaries for collections. Avoid one giant class of optional fields.

### Definition, use, and state

| Type | Fields and responsibility |
| --- | --- |
| `MotionScore` | Existing score header; `kind="motion"`; public `parameters`, `inputs`, `outputs`; one `body` |
| `MotionBody` | Tagged union of the body types below |
| `MotionUse` | Exactly one inline body or existing `ScoreReference`; public parameter values and input bindings |
| `MotionInstance` | A use plus `scope`, owner identity, clock bindings, and lifecycle bindings |
| `MotionState` | Definition identity, local position/stage, captured values, direction, paused/completed flags, child states, random state, pending events |
| `MotionSnapshot` | Instance states, graph delays, clock state, queued commands, evaluation-grid origin, and resolved dependency identities |
| `PreparedMotion` | Resolved immutable definition, typed ports, ordered graph, backend requirements, resource bounds |

Definition reuse is not state sharing. Two uses of the same library entry get
independent state. Sharing happens by connecting to one explicitly owned instance.
Scopes are voice, trigger, part, and instrument; a trigger can own several voices.
The host maps physical key presses and retriggers to these stable owners.

### Bodies

| Body kind | Principal fields | Meaning |
| --- | --- | --- |
| `hold` | `value` | Constant signal, also useful as a waiting stage |
| `cycle` | `shape`, rate or position driver, `phase`, `center`, `depth`, optional markers/playback | Periodic waveform or reusable periodic shape |
| `contour` | `initial`, `segments`, optional `release`, `start`, `playback`, markers | Traversable finite shape; release makes it a triggered envelope by default |
| `stages` | `initial_stage`, named child uses, ordered transitions | One active child; events select the behavior |
| `patch` | Named child uses, signal routes, event connections, output selections | Children run together and interact |
| `random` | Algorithm, seed, range, update driver, interpolation | Deterministic random process with stated reversibility |
| `recording` | Asset reference, channel selections, position driver, interpolation | Read captured control data |
| `follower` | Input port, measurement, response times | Derive controls from an input signal |
| `transform` | Operation and typed inputs/options | Sum, product, map, slew, quantize, delay, latch, or other registered operation |
| `external` | Provider identity/version, port contract, configuration, execution contract | Call a separately implemented generator |

`steps` can be a library contour using held segments. ADSR can be a library
contour with exposed attack/decay/sustain/release parameters. Delay and fade-in
can be library compositions. Do not add a second runtime implementation just
because a common preset deserves a friendly name.

### Supporting contracts

| Type | Proposed contents |
| --- | --- |
| `Segment` | `duration`, `to`, `curve` (linear by default; held or a specified curved interpolation) |
| `StartValue` | A number, or `current` where an entry context supplies a previous output |
| `PositionDriver` | Timed signed rate, clock-locked position, or typed signal reference; exactly one |
| `Playback` | Mode, bounded region, optional traversal count, end behavior |
| `Marker` | Name, position, direction filter |
| `SignalPort` | Name, scalar unit, range when known, evaluation requirement |
| `EventPort` | Name and declared payload field types |
| `MotionCommand` | Tagged command from section 6; arguments depend on the tag |
| `Transition` | Allowed source stages, event, optional typed predicate, action |
| `EventConnection` | Source event reference, target instance, command, payload mapping, optional delay/quantization |
| `ClockBinding` | Clock identity and synchronization relationship |
| `PublicParameter` | Name, description, type/unit, default, domain; body fields refer to it explicitly |
| `ParameterValue` | A typed literal or `{ parameter = "name" }`; resolved against a declared public parameter |
| `ExecutionContract` | Supported evaluation modes, bounded work/state, determinism, latency, snapshot/seek support |

Stage-local transitions take precedence over program-wide transitions; within
one list, the first matching transition wins. Predicates use a small tagged set
of comparisons and boolean combinations over typed inputs/event fields. More
complex decisions belong in a decision motion or external provider, not a new
general-purpose language in TOML.

Known signal ranges help catch errors, but not every external process has a
provable bound. A connection to a bounded target must declare an accepted range
or include an explicit limiting transform. Unit conversions such as decibels to
linear gain are explicit. Scalar ports cover simple motions; named scalar ports
compose multi-dimensional ones without forcing vectors onto every user.

## 11. Libraries are a first-class requirement

Add `MotionScore` to the existing score union and codec. A library should contain
individual motions, reusable shapes, and compound motions with musical public
parameters. Library resolution remains in the preparation/host layer.

A reusable sine exposes speed, rather than requiring callers to edit its body:

```toml
kind = "motion"
name = "vibrato"
title = "Vibrato"
tags = ["#pitch", "#periodic"]

[parameters.speed]
unit = "hz"
default = 5.0
minimum = 0.1
maximum = 20.0

[body]
kind = "cycle"
shape = "sine"
rate = { parameter = "speed" }
```

The reference supplies the rate from the declared parameter, including its unit.
There is only one authored default. Preparation checks that each reference's
unit and domain suit its field; this is not arbitrary object mutation. A future
motion instance inside an instrument could use it as follows:

```toml
[motions.vibrato]
score = { selector = "factory:vibrato" }
scope = "voice"
parameters = { speed = 6.0 }
```

This is an embedding fragment, not a complete instrument document. The proposed
`motions` field replaces the separate generator dictionaries when integration
is implemented. Parameter routing to the sound still uses the existing
modulation system.

Required library behavior:

- Resolve exactly one definition using existing paths/selectors. Show useful
  errors for missing or ambiguous references.
- Expose only declared public parameters. No deep-merge inheritance, textual
  templates, arbitrary nested overrides, or implicit dependency on another file's
  private stage names.
- Permit compound motions to include library motions and re-expose selected
  ports/parameters. Recursive inclusion is invalid.
- Retain resolved identities and content hashes in prepared artifacts and
  snapshots. Editing a library file does not mutate an already playing instance.
- Reuse `PresetScore` after extending the public-parameter contract to motions;
  today its parameter handling requires an interface score.
- Store title, tags, parameter descriptions, unit/range information, and capability
  requirements so both humans and hosts can explain what a preset does.

Useful initial library entries: sine/triangle/pulse, ADSR/AHD, delayed vibrato,
looping sustain, tremolo, bouncing decay, sample-and-hold, stepped filter rhythm,
ducking, and attack-to-sway. Presets should teach the composition vocabulary.
Discovery by tags is useful; loading must resolve a specific unambiguous entry.

## 12. External programs and the meaning of completeness

“Any possible mechanism” cannot mean that every algorithm has bounded memory,
can run inside an audio callback, can be reversed, or is deterministic. Those
properties are incompatible with some useful generators. The achievable promise
is **a common connection contract for every computable generator we can host**,
with honest capability requirements.

An external provider receives timestamped inputs/events, a requested output
interval, parameters, and its state. It returns timestamped signal data, emitted
events, and updated state. Its manifest declares units, ports, configuration
schema, provider version, latency, and snapshot/replay capabilities.

Distinguish preparation-time generation, offline streaming, and live buffered
delivery. Preparation-time output can be frozen as a recording motion. A live
provider runs outside the audio callback; no process launch, file access,
blocking wait, network request, or interpreted user code belongs in the callback.
TOML references a host-registered provider, not a shell command to execute merely
because a preset was loaded.

Define buffer capacity, timestamps, acceptable latency, late-data behavior, and
failure behavior at the provider boundary. For example, hold the last value,
use a declared neutral value, or stop with a diagnostic. The choice is explicit
because these have different musical consequences. The host aligns signal and
event outputs consistently; it cannot claim zero-latency results from future
analysis.

An unrecorded nondeterministic provider cannot promise exact replay. Offline
reproduction either restores a provider-supported state or reads recorded
outputs and events. Missing capabilities fail preparation for the requested
execution mode rather than silently changing the sound.

## 13. Execution semantics that keep this understandable

### One performance, independent of block boundaries

Use the established exact-time/event ordering model. Internal event times use
an exact canonical coordinate; rendering maps them to sample boundaries by a
specified rule, initially the first sample at or after the event time. Retain
the exact time so simultaneous rounding does not erase original order.

At an event boundary: advance to the boundary, collect natural boundary events,
apply external events in their existing ordinal order, then dispatch internally
generated events in stable connection order. Parent actions precede their
descendants. Events from a stage that was exited before dispatch are discarded
using the stage activation identity. Thus a note-off exactly at attack completion
can enter release without an obsolete attack completion subsequently entering
sustain. Sample the resulting signal state for that boundary.

For ties, use the following contract: external events retain their host-supplied
ordinal; natural events sort by exact time, stable instance path, and authored
marker/port order; connections run in authored list order. Dispatch descendants
through a FIFO queue, after the already queued events at that timestamp. Scope
broadcasts enumerate stable owner identities. Stage-local rule precedence and
first-match selection apply within each receiving instance. Preserve these
rules in conformance fixtures; dictionary iteration and thread scheduling must
not influence the musical result.

Evaluation can be analytic, audio-rate, or on a declared control grid. A reduced
control grid has a fixed origin and an explicit hold/interpolation rule. Event
times and stage boundaries are not quantized to that grid. Declaring a lower
evaluation rate is an audible choice, not an invisible optimization.

### Bounded live execution

Preparation resolves library references, validates units/scopes, detects
instantaneous cycles, computes evaluation order, and allocates state. The live
profile has fixed limits for instances, active stages, delay memory, transitions,
and events per interval. A large seek with marker emission may exceed that
profile; report it according to the declared failure policy, never drop an
arbitrary subset of markers silently.

Snapshots include all behaviorally relevant state: stage entry captures,
positions, loop counts, random seeds/counters, pending quantized commands,
detector hysteresis, delay lines, clock estimation, and external stream cursors.
Restoring against a different prepared definition is an error.

### Definition ownership and voice lifetime

ufor owns portable declarations, validation, and scalar reference semantics.
enge owns preparation, sample/control evaluation, native state, and resource
limits. Hosts own physical clocks, MIDI/device input, and external services.

An amplitude motion may be the explicitly selected lifetime owner of a voice.
Other motions can complete independently. Instrument/part motions survive
individual note releases; voice motions retire with their voices. On voice
stealing, deliver the chosen stop policy and invalidate that instance's queued
events so they cannot act on a reused slot.

## 14. Implementation sequence and decision gates

This is a proposed architectural change. Review the contract and examples before
implementation; do not use this document as permission to refactor adjacent
systems. Each implemented slice should replace the corresponding old path,
with shared reference/native semantics, rather than accumulating parallel
generator engines.

1. **Settle vocabulary and semantics.** Agree on Motion, ports, scope, position
   ownership, lifecycle, units, completion, and event boundary rules. Have a
   musician read the simple and staged TOML examples without explanatory prose.
2. **Establish the common contract.** Introduce motion definitions and instances
   for cycle/contour; reuse existing waveform/envelope kernels and routing.
   Preserve current behavior in conformance fixtures. Keep the schema tagged,
   and reject unsupported body kinds explicitly.
3. **Add library uses and public parameters.** Integrate existing references,
   presets, and dependency resolution. Supply the small familiar library first.
4. **Add staged behavior and event ports.** Deliver attack-to-LFO-to-release,
   alternate triggers, markers, and deterministic event connections end to end.
5. **Generalize playback and clocks.** Reverse, seek, loops, ping-pong, external
   position, musical tempo maps, quantization, and pulse-clock adapters.
6. **Add composition primitives.** Transforms, event logic, delayed feedback,
   randomness, recording, and followers, driven by concrete musical examples.
7. **Add the provider boundary.** First support recorded/offline generation;
   add live providers only with their buffering and execution contracts.

When replacing old declarations, convert repository documents and call sites
together. If users need an import converter, decide that explicitly; do not
quietly retain two permanent schemas or add compatibility layers. Update the
top-level roadmap only as a separate requested planning decision.

### Acceptance evidence

- Simple LFO and ADSR-like definitions remain short and understandable.
- An attack becomes a loop and releases from its actual value, including when
  interrupted exactly at a stage boundary.
- Repeated note-ons, stage cues, reversal, forward/backward seeks, pause, and
  resumed playback have exact, documented outcomes.
- Marker events occur once with correct ordering through wraps, turns, seeks,
  and multiple events in one sample; event storms are diagnosed.
- Shared versus per-voice instances behave predictably, including retriggers,
  voice stealing, and cross-scope reductions.
- Tempo changes preserve the intended phase relationship; transport jumps,
  missing external pulses, and quantized commands follow the selected policies.
- Whole renders, irregular partitions, and snapshot continuation agree for
  signals and event traces in reference and native backends.
- Random results reproduce from seed and identity; external results reproduce
  only under their declared recording/state contract.
- Libraries resolve reproducibly and diagnose unknown public parameters,
  ambiguous selectors, incompatible units, and dependency cycles.
- Audio regressions are listenable WAV files at 48 kHz, at least one second
  long; event behavior also has exact trace assertions. Physical synchronization
  needs host/device validation beyond these automated checks.

## 15. Related systems and why this proposal goes further

Bitwig's documented Segments modulator includes one-shot, hold, looping, and
ping-pong playback, while its Curves modulator exposes phase and several trigger
relationships. This is useful precedent for separating a shape from how it is
played. Its broader modulator set also includes followers and signal processors.
See the [Bitwig modulator reference](https://www.bitwig.com/userguide/latest/modulator/).

Ableton's unlinked clip envelopes give an envelope an independent loop length.
That reinforces the idea that an envelope need not end with the note or clip
whose parameter it controls. See [Ableton's clip-envelope manual](https://www.ableton.com/en/live-manual/11/clip-envelopes/).

Those precedents inform the vocabulary; they are not specifications to copy.
The proposal here makes the shared state/event contract, reusable libraries,
cross-motion connections, and external-provider capabilities explicit in a
human-readable format.

## Additional work beyond the prompt

None. This change adds a design document only.
