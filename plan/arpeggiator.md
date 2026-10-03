# The expressive arpeggiator

Design proposal, updated 2026-10-03. This plan covers the shared arpeggiator model,
its integration with Motions and instruments, and a standalone MIDI program.
Names and TOML examples below are proposed syntax, not currently loadable scores.
This document adds no implementation and does not change the main roadmap.

## 1. The musical idea

An arpeggiator chooses **which note to play next, when to play it, and how to
perform it**. A note is more than its pitch and velocity: it can contain a breath
gesture, a bend, articulation changes, an audio region, and the space following
it. Reordering notes must preserve this material and its ownership.

The same instrument should let a musician:

- Hold C, E, G and immediately hear a familiar ascending sixteenth-note pattern.
- Play C, E, G sequentially on a wind controller, capture their individual
  breath and bend gestures, then hear G, C, E with those gestures intact.
- Keep blowing into a wind controller while a remembered note bank supplies
  pitches and the current breath shapes the resulting line.
- Mark syllables, drum hits, or notes in a recording and play their regions as
  an arpeggio. Ascending through their assigned keys can reproduce the recording.
- Put five attacks in eight steps, walk through a chord with weighted choices,
  and give pitch, rhythm, velocity, and articulation different cycle lengths.
- Send the result to a hardware synth from a standalone MIDI-only executable.

Provide named musical presets over one model. A simple ascending arpeggio needs
no graph editor, controller-routing table, scripting language, or sample engine.
Advanced behavior comes from composing a small set of explicit decisions.

“Ultimate” means broad, testable musical coverage. It cannot honestly mean
undocumented compatibility with every proprietary pattern library or every
algorithm anyone might invent. Maintain a feature-family coverage matrix and
add a conformance fixture whenever a concrete missing behavior is identified.

## 2. Existing range we should cover

These are reference behaviors, not an instruction to clone product interfaces.
Sources were checked on 2026-10-02; proposed behavior elsewhere in this document
is our design, not a claim about those products.

| Family | Required behavior and primary reference |
| --- | --- |
| Classic and performance arps | Up/down, endpoint-repeat variants, played order, converge/diverge, chord repeats, octave traversal, latch, retrigger, finite repeats, groove, gate overlap, and several distinct random orders. [Ableton Live 12 Arpeggiator](https://www.ableton.com/en/live-manual/12/live-midi-effect-reference/#arpeggiator) provides a useful concrete checklist. |
| Programmable patterns | Rests, ties, chords, independent velocity and note length, live pattern entry, and capture into an editable sequence. [Logic Pro pattern parameters](https://support.apple.com/en-ae/guide/logicpro/lgce1465941c/mac) demonstrates this family. |
| Algorithmic and polymetric patterns | Independent pattern loops, Euclidean gates, chance, scale mapping, swing, and meaningful ordering of chained processors. [Hapax effects](https://squarp.net/hapax/manual/modefx/) is a reference for these combinations. |
| Phrase and accompaniment arps | Fixed-note material, input-note substitution, and chord-responsive phrase conversion, rather than sorting alone. [Yamaha MONTAGE M conversion modes](https://manual.yamaha.com/mi/synth/montage_m/en/om02screenparameters0050.html) distinguishes these behaviors. |
| Independent generative dimensions | Pitch selection, rhythm, duration, clusters, velocity, controller generation, and bending can contribute independently. [KARMA's documented model](https://karma-lab.wikidot.com/karma2:how-does-karma-work) motivates this separation. |

Euclidean rhythm is a rhythm generator, not a pitch-order algorithm. Its musical
basis is documented in [Toussaint's original paper](https://cgm.cs.mcgill.ca/~godfried/publications/banff.pdf).
We must specify our rotation and initial phase instead of assuming every
implementation produces the same array orientation.

## 3. Fit within the current system

Use the established boundaries:

| Owner | Responsibility |
| --- | --- |
| uFor | Portable definitions, exact timing, note/control ownership, library references, validation, and reference semantics. |
| New arpeggiator project | First-class Python and Rust runtimes in one repository, MIDI adapters, CLI, examples, and shared conformance tests. Runtime cores own capture, selection, rhythm, note realization, deterministic scheduling, and snapshots, without audio processing or device ownership. |
| Motions | Clocks, control signals, recorded gestures, parameter modulation, and typed events used by the arpeggiator. |
| enge | Consume generated performance events; realize sample regions and synth voices through existing instruments and renderers. |
| Host or standalone MIDI shell | MIDI ports, device profiles, clock acquisition, controller decoding, channel allocation, and output scheduling. |

An arpeggiator is an **event processor with note input and note output**, with
Motion-compatible signal and event ports. It should not pretend to be a scalar
LFO. Reuse the Motion clock/event contract and library resolver; do not require
every note to be represented as a signal crossing a threshold.

Source anchors checked for the initial proposal on 2026-10-02; recheck their
implementation status when beginning development:

- Sibling `ufor/ufor/events.py`: `Trigger`, `Release`, addressed `ControlChange`,
  stable `trigger_id`, ordered events, raw MIDI and UMP storage.
- Sibling `ufor/ufor/control.py`: exact tempo maps, host song beats, and elapsed
  running beats that ignore transport seeks.
- Sibling `ufor/ufor/motion.py`: Cycle/Contour/Stages and ordered Motion events.
- `enge/synth.py` and `src/runtime.rs`: standalone beat-clock Contours/Cycles in
  offline and persistent synth rendering. Beat-based Stages and explicit
  transport-position drivers remain separate unfinished Motion work.
- `enge/midi.py`: a limited offline adapter. It currently handles CC1/CC11 and
  bend but rejects breath CC2 and other unsupported messages. It also fixes a
  particular bend interpretation in its supplied patches. It is not yet the
  expressive capture or bidirectional MIDI layer this plan needs.
- `enge/sample_instrument.py` and sibling `ufor/ufor/samples/`: existing sample
  playback and instrument definitions to reuse when lowering region notes.

Existing normalized performance events do not preserve every wire detail,
release velocity, or device-specific controller interpretation. Extend their
shared contract where needed, while retaining the raw source event ledger;
do not quietly claim an existing `Trigger` is already a lossless MIDI note.

## 4. One pipeline, several ways to supply notes

```text
MIDI performance / event clip / marked sample
    capture and ownership
    note bank
    rhythm opportunities + note selection
    pitch, articulation, and gesture realization
    ordered performance events
    MIDI destination or enge instrument
```

Each stage is independently understandable and can be inspected in a trace.
The note bank need not be a simultaneously held chord.

### The note object

Use an immutable source note and a distinct emitted occurrence. Replaying a
source note twice creates two occurrences with different `trigger_id` values.
Pitch is never an identity, and an input MIDI channel is never a permanent
output voice identity.

| Information | Meaning |
| --- | --- |
| Source identity | Capture/session identity, note identity, original onset and ordered source-event references. |
| Musical identity | Pitch/key or region-selection key, optional exact frequency/tuning, velocity, release velocity, articulation, tags. |
| Gate | Note-on to note-off interval, distinct from the interval to the next note and any release tail. |
| Entry state | Effective note-owned controller values at onset, with provenance and explicit unknown values. |
| Expression | Ordered events or declared curves in note-local time, including bend, breath, pressure, and timbre. |
| Following space | Captured gap duration and retained inter-note events, when the source is a sequential phrase. |
| Optional payload | A reference to an audio asset and a marked region; the event core never loads the audio. |

A phrase owns the original event ledger, note objects, inter-note material,
and leading/trailing material. Notes reference the ledger so capture does not
duplicate or erase events merely to make them convenient for playback.
Independent overlapping notes may legitimately share references to a captured
channel-wide observation. That is shared evidence, not two original events.

An emitted occurrence records its source note, bank revision, algorithm decision,
destination, transposition, timing map, and output identity. These records make
“why did it play this?” answerable and allow generated phrases to be captured
and subsequently arpeggiated again.

### Bank modes

- **Held:** current input notes; immediate classic arpeggiation. Keep repeated
  pitches as distinct identities. A preset may explicitly deduplicate by pitch.
- **Latched:** retain a chord after release, with explicit replace/add/toggle
  behavior. Sustain pedal and arp latch are separate controls.
- **History:** last N completed notes or a bounded musical-time window. Suitable
  for monophonic input. Initial default: last eight completed notes, retaining
  repeated pitches, published on the next step boundary.
- **Phrase:** capture between record/commit commands or a declared bar/count
  boundary, then play a frozen revision. Overdub creates the next revision.
- **Regions:** a fixed collection of marked sample notes or loaded note phrases.

Replacing a nonempty bank must not invalidate sounding occurrences. Clearing
the bank is an explicit stop condition and retires its owned output notes.
At a scheduled step, select from one published revision. For a changing held bank, continue
from the previous selection's stable ordering key; explicit retrigger settings
can restart at the first note. Freeze permutation-based random orders for the
cycle and regenerate them when the next revision becomes active.

## 5. Monophonic capture is a first-class instrument

Three performance presets make the causal choices visible:

1. **Live repeat:** arpeggiate the current note immediately. Sample its current
   expression at each generated onset and follow subsequent live changes. No
   future gesture is implied.
2. **Remembered notes, live breath:** select pitches from history while current
   breath/bend controls the generated output. Recorded gestures remain stored
   but these selected lanes use the live performance instead.
3. **Phrase reorder:** wait for completed notes or an explicitly committed
   phrase, then rearrange complete recorded gestures, including their gaps.

Do not silently switch between these behaviors as a buffer fills. Reversing or
randomizing a future phrase is impossible with zero latency. A recorded gesture
becomes eligible only when its declared end, including any capture-tail window,
has arrived. A streaming mode can follow an open gesture at original speed;
time compression or arbitrary order requires sufficient captured material.

Show capture state, available notes, latency, and the current bank revision.
Provide record, commit, replace, overdub, undo-last-capture, and clear commands.
A bar boundary or explicit command is a reliable phrase delimiter. An optional
silence timeout needs a declared duration and a hard maximum phrase length;
breath noise or a stuck note must not create an unbounded recording.

For nominally monophonic MIDI with overlapping note-ons, the device profile
must state whether overlap denotes legato handoff or two independent notes.
Default wind behavior follows the newest active note and snapshots current
breath and bend at each handoff. Keep the original overlap in the capture.

For the WX7 behavior reported by the user, a legato passage contains successive
nonzero-velocity note-ons with no intervening note-offs, followed by a final
note-off encoded as a velocity-zero note-on. Its capture profile treats each
new onset as the end of the preceding note segment and the start of the next,
without resetting breath or bend. Normalize velocity-zero note-ons as releases
for interpretation, but preserve their original encoding in the ledger.
Controllers belong to the current segment; each new segment snapshots the
inherited state. These inferred boundaries do not insert note-offs into the
source recording. Do not apply this monophonic rule to ordinary polyphonic
input. Verify the device profile against a hardware capture before claiming
WX7 conformance, including the reported absence of breath messages after release.

## 6. Expression and the information between notes

### Ownership before transformation

Classify observations by semantics rather than simply copying every MIDI CC
between two note-ons:

| Scope | Capture and replay rule |
| --- | --- |
| Per-note | Travel with their note and are addressed to each generated occurrence. |
| Channel expression | A device profile can bind breath/bend/pressure to the active monophonic note, or snapshot and share observations among the active notes. The chosen interpretation is recorded. |
| Phrase context | Retain gaps, pedal history, and phrase-local automation; replay on the phrase timeline unless explicitly lifted into note gestures. |
| Destination/global | Transport, bank/program changes, tuning setup, and configuration stay on their declared timeline. Do not replay them every time a note repeats. |
| Unknown/opaque | Preserve raw bytes/words and order in the ledger. Default to retained, not emitted in transformed playback, with a visible diagnostic. Explicit passthrough can be selected. |

For a sequential monophonic phrase, the default cell is the half-open interval
from one onset to the next, or to the explicit phrase end. A note-off can occur
inside its cell. Subsequent unowned local controller changes remain in its gap
lane; they are not reassigned to the nearest sounding note. The next note also
captures its own entry state. Leading material belongs to the phrase prefix;
the final cell includes the selected trailing material.

Gate intervals may overlap cells. Expression with explicit note ownership can
continue past another onset or note-off when the destination supports a release
tail. Never truncate it solely because the next note has started.

Retain global events once in the original ledger. Rearrangement policies may
move cells and their gap lanes, preserve gaps as a separate timeline, or omit
them from output explicitly. The default recorded-phrase preset carries the
following gap. “Not emitted” is not “deleted from the capture.”

For a controller genuinely arriving after C's release and before E's onset:

- Keep the event in C's following gap, and update the captured controller state
  used to determine E's entry state. This storage relationship alone does not
  make the event expression owned by C.
- During rearranged playback, emit it as C's tail expression only when the
  capture profile assigns that meaning and the destination can still address
  C's release tail independently. Otherwise retain it without emitting it.
- Never apply it to whichever rearranged note happens to be sounding. E starts
  with its own captured entry state regardless of the new playback order.
- Original-phrase playback preserves the event's original timing and scope,
  including its channel-wide effect, rather than applying the rearrangement rule.

This retained-but-not-emitted rule is the default for unowned gap controllers,
not permission to discard explicitly owned note expression. Capturing a stream
as a Motion does not assign note ownership; the capture profile does that.

### The wind-controller example

This is a generic controller example, not a claim about WX7 output. Suppose
the input contains:

```text
0 ms    C note-on, current breath = 0
8 ms    breath = 0.30
40 ms   breath = 0.80
90 ms   bend = +35 cents
180 ms  C note-off
190 ms  breath = 0.10             (in the following gap)
220 ms  E note-on, breath = 0.10  (inherited entry state)
235 ms  breath = 0.65
400 ms  E note-off
```

Selecting E before C must initialize E with breath 0.10 and deliver its 0.65
change 15 ms later under original timing. It must not inherit whatever breath
the previous output note happened to leave behind. C must retain its initial
silence and its later breath attack; filling in future breath at onset would
change the performance. The bend travels with C and remains an offset from its
new base pitch after transposition.

A profile explicitly maps a device's breath source, commonly CC2, to the desired
destination control. CC2, expression CC11, velocity, and channel pressure are
not interchangeable merely because all can make a sound louder. Store raw
controller values and resolution alongside their declared normalized meaning.

### Timing and transformation

Keep event order and exact relative times. Motions need not be continuous.
Distinguish discrete events, step-valued signals that hold their last value,
and explicitly interpolated signals. A captured MIDI controller lane retains
its ordered original events and may expose a step-valued signal for modulation.
Repeated messages remain in the event view even when their values are equal;
holding a value does not generate additional MIDI messages. Before a known
initial value exists, preserve unknown state rather than inventing zero.
Converting the lane to an interpolated Motion requires an explicit
interpolation/reduction choice, with an error bound and retained original data.
Controller state can exist while no note is sounding.

Separate three operations:

- **Order:** choose another note/cell. Its internal time still runs forward.
- **Retiming:** map a complete gesture to an output duration. With source gate
  `G` and output gate `H`, fit maps a local offset `u` to `u * H / G` when `G > 0`.
  Apply the same mapping to note expression and declared post-gate material.
- **Gesture reversal:** reverse a recorded curve or audio region deliberately.
  Reversing note order does not reverse breath, swap note-on/off, or undo events.

Provide original timing, fit, and crop/hold policies. Original timing is the
phrase/sample identity preset. Fit is the default for completed gestures placed
on a rhythmic grid. For zero-duration source gates, keep simultaneous events
ordered at onset and obtain output duration from the selected grid policy;
there is no division by zero. Cropping must emit required release/cleanup events.
An optional attack-preserving mapping can be added as a documented preset when
a concrete musical fixture requires it.

With source-timed rhythm, the selected cell's duration determines the next
onset, including its carried gap. With grid rhythm, the grid determines the
next onset independently: fitting a gate does not move later grid opportunities.
Carried gap/tail events can therefore overlap the next occurrence. Explicitly
owned events retain their occurrence ownership; unowned gap events follow the
retention rule above. Emitted events must obey the destination's overlap
policy; never send an old gesture into a channel reassigned to a new note.
A profile that cannot realize the requested overlap must diagnose it or use
an explicitly selected crop/handoff policy, rather than silently losing data.

Recorded, live, and Motion-generated expression are distinct lane sources.
Default ownership is one source per lane. Combining them requires the existing
typed modulation operations, for example live breath multiplying recorded
breath, or live bend adding cents. Record the combination and its bounds.

## 7. Marked samples are note material

A sample note references an immutable asset, channel selection, and integer
source-frame boundaries. Reuse sample playback definitions and library identity;
do not introduce a second audio asset loader or a private resampler.

An authored marker has a stable note identity and an ordering/selection key.
That key need not describe the acoustic pitch. For syllables A, B, C in source
order, virtual keys 60, 62, 64 mean that ascending selects A, B, C even if the
speaker's actual pitch falls. Optional root pitch is separate metadata used
only for deliberate pitch transposition. Preserve “unknown pitch” for noise,
drums, and unpitched speech.

Onset markers alone do not determine releases, gaps, or tails. Default regions
run from each marker to the next, with the final region ending at the asset
boundary. Include a prefix region or explicit prefix reference when the first
marker is not at frame zero. Optional attack/gate/release markers describe more
musical slicing without erasing the original boundaries.

The identity contract is strong:

> Ascending selection keys, one traversal, original timing, original playback
> rate, and contiguous exhaustive regions reproduce the source sample stream.

For this case, keep a continuous source cursor across adjacent regions. Bypass
extra attack/release envelopes, seam fades, time stretching, resampling, and
normalization. Compare decoded sample frames exactly at the same sample rate.
An arbitrary pitched recording sorted by detected pitch does not satisfy this
contract; source-order keys do.

Reordered or repeated regions may need explicit seam fades, tail overlap,
one-shot/gated/looped sustain behavior, or time stretching. Those are playback
choices and need their own listening regressions. They do not inherit the
identity guarantee. Rubber Band can remain an optional realization backend;
the arpeggiator's event model and standalone MIDI build never require it.

MIDI output can trigger an external sampler using an explicit key map for these
regions. It cannot reproduce an unmapped audio asset itself. Diagnose that
destination mismatch before starting playback.

## 8. Selection, rhythm, and performance are independent

### Selection vocabulary

Support ascending, descending, as-played, reverse-played, alternating with and
without repeated endpoints, inside-out, outside-in, bass/treble alternation,
pedal-note patterns, chord repeats, and explicit index patterns. Define empty
and single-note behavior and use source identity to break equal-pitch ties.

Octave expansion, scale-degree transposition, register limits, inversions, and
voicing are separate transformations. State whether traversal iterates notes
inside each octave or octaves inside each note. Keep pitches unrounded in the
musical model; a destination decides how microtonal pitches are represented.
Out-of-range pitches require a chosen drop/fold/error policy, never silent clamp.

Algorithmic selectors include weighted choice, shuffle-once, reshuffle-per-cycle,
no-immediate-repeat, bounded random walks, interval contours, transition tables,
and finite grammar expansions. All must still select or derive identified notes.
Deriving a new pitch retains a source gesture reference or explicitly chooses
a synthetic gesture. No hidden pitch generation from an empty bank.

Phrase selectors cover fixed-note/drum playback, rank substitution from the
current input bank, and chord-relative interval patterns. Chord recognition is
an optional explicit mapping policy with declared ambiguity rules. Begin with
literal interval templates; do not require machine learning or PyTorch.

Arbitrary external algorithms use the planned provider boundary: prepare or
record their decisions first. There is no `eval` field or unrestricted program
execution inside an event or audio callback.

### Rhythm vocabulary

Support regular divisions, tuplets, rests, ties, explicit duration patterns,
Euclidean gates, swing/groove templates, ratchets, bursts, accents, conditional
steps, probability, finite repeats, and independent pattern lengths. Polyphony,
strums, and several independently clocked lanes use the same scheduling model.

Specify Euclidean phase exactly. For `n > 0`, `0 <= k <= n`, and step `i`, the
canonical unrotated gate is `(i * k) mod n < k`. Positive rotation moves hits
later by replacing `i` with `(i - rotation) mod n`. Thus `k=3, n=8` is
`10010010`; zero pulses is silence and `k=n` fills every step. Persist this
convention and test all rotations instead of depending on a library's default.

Pitch selection advances on emitted hits by default. Advancing on every grid
step is an explicit alternative. State whether probability-rejected hits advance
the selector, whether ratchets repeat one choice or select again, and whether
ties extend a particular active occurrence or become rests when none exists.
Default: rejected hits do not advance, ratchets repeat one choice, and a tie
without a preceding occurrence is a rest.

### Deterministic operation order

At each rhythm opportunity:

1. Publish eligible bank/configuration changes and evaluate step parameters.
2. Evaluate the rhythm mask and conditions, then seeded chance.
3. Select identified source notes from that bank revision.
4. Apply pitch/voicing transforms and choose the expression sources.
5. Expand ratchets/strums; compute exact onsets, gates, and gesture mappings.
6. Admit output occurrences against destination and resource limits.
7. Emit ordered lifecycle/control events and trace the decision.

Swing and human timing affect exact scheduled onsets within declared limits;
negative offsets require lookahead. Live input that arrives too late follows an
explicit next-opportunity/drop policy. No event is inserted into already emitted
time. Rate changes preserve rhythmic phase unless retrigger was requested.

Seed random decisions by instance identity, named lane, bank revision where
relevant, and logical decision count. Do not use audio block size, wall time,
or a single random stream shared with unrelated lanes. Snapshot all counters.

## 9. Clocks, lifecycle, and Motions

Keep the agreed time coordinates: local running beats, elapsed seconds, and
sample frames at an audio destination. A MIDI-only process has no sample rate;
it schedules exact beat/second positions against a monotonic clock. Audio hosts
convert absolute times to frames once using their declared rounding rule.

Local beat timing follows tempo and freezes on a musical stop, preserving phase
through seeks. Explicit transport-position timing follows song position. Host
seeks cancel pending quantized commands, as already chosen for Motions. For
arpeggiation, release owned output notes, discard their future scheduled events,
and begin again at the destination clock's next eligible boundary; a seek does
not spray all crossed notes into the output. A transport stop preserves pattern
position but releases owned sounding notes by default. A performer pause/hold
command may deliberately use a different gate policy.

MIDI Clock acquisition belongs in the standalone host adapter. Define internal,
external, and incoming-note-pulse modes separately. For external clock, document
Start/Continue/Stop/song-position interpretation, acquisition, phase correction,
timeout, and loss behavior. Default loss policy: stop generating attacks and
release owned output notes. Record accepted clock observations so playback does
not depend on the next run's packet jitter. This adapter must be implemented and
tested explicitly; the existing tempo map is not a MIDI pulse estimator.

Expose a small useful Motion interface:

- Inputs: pulse, reset, start/stop, capture/commit, bank selection, density,
  gate, transposition, selection offset, and named expression lanes.
- Outputs: step, hit, rest, cycle, note-start, note-end, capture-ready, plus
  optional phase/selected-index signals with declared domains.
- Changes: scalar values at the next opportunity by default; structural pattern
  and phrase changes at a declared step/cycle boundary. Sounding occurrences
  retain their realized parameters except for explicitly live expression.

Use the existing event ordering/capacity rules for connections. Feedback requires
an explicit delay of at least one scheduling quantum and a bounded event budget.
Do not add an unbounded same-timestamp loop between a Motion and an arpeggiator.

## 10. MIDI realization and faithful cleanup

The core emits note/control intent. A destination profile maps that intent to
MIDI messages, bend ranges, channel allocation, and device-specific articulation.
Preserve high-resolution data and atomic groups such as paired controllers and
parameter-selection/data transactions. Do not split such groups while reordering.

MIDI 1.0 channel-wide bend and expression affect all notes on a channel. MPE
uses channel allocation to make that expression independently addressable.
This limitation is part of the destination contract, not something a note bank
can remove. [MIDI Association MPE overview](https://midi.org/midi-polyphonic-expression-mpe-specification-adopted)
and [MIDI 2.0 architecture](https://midi.org/details-about-midi-2-0-midi-ci-profiles-and-property-exchange-updated-june-2023)
provide the protocol context; conformance work must use the applicable full specs.

Provide explicit output profiles:

- Monophonic MIDI 1.0: one channel, one independent expression owner at a time.
  Default conflicting overlap behavior is to end the previous occurrence.
- Shared-expression MIDI 1.0: chords can share one declared controller stream;
  differing recorded per-note gestures cannot silently merge.
- MPE/channel-per-note: allocate channels, initialize state before onset, and
  respect a declared release-tail reservation and channel exhaustion policy.
- MIDI 2.0: use supported per-note messages and preserve group/channel identity;
  do not assume every controller has a per-note equivalent or infer capability
  merely from a UMP container.

Default same-time output ordering is old note-off, safe channel reassignment,
new entry-state initialization, new note-on, then ordered local events at zero.
Keep captured ordinal order within each lifecycle bundle. MIDI wire output is
serial: use a stable serialization order and timestamps where supported, and
report bandwidth overload instead of promising physically simultaneous messages.

Do not reset a shared channel while another owned note still needs its state.
Preserve pedal state separately from arp latch and key-up; track key release
and pedal-held sounding lifetime. A generated note-off does not guarantee silence
if sustain is still active. Sustain behavior must be realized within the channel
ownership contract, including how a monophonic handoff releases prior sound.

Pair repeated same-key MIDI 1.0 input notes according to a documented profile
(FIFO by default); preserve distinct source identities. Avoid independent
overlapping same-key occurrences on one output channel unless the device profile
defines their release behavior. MPE allocation alone does not guarantee arbitrary
post-note-off tail control on every receiver.

Maintain an output-ownership ledger. Stop, bypass, empty bank, seek, preset
replacement, disconnect, and failure must retire owned occurrences and cancel
their future controls. Send only the cleanup the process owns; preserve unrelated
passthrough notes. Bound release-tail channel reservations. Panic is an explicit
stronger command. Snapshot restore into hardware requires release/reconciliation
of external state; restoring internal memory cannot rewind a synthesizer.

## 11. Separate project and standalone MIDI-only delivery

Develop the arpeggiator as a new project, with Python and Rust implementations
in the same repository. Its name and repository location remain to be chosen.
uFor owns portable definitions and shared musical/event types; the new project
owns executable arpeggiator behavior, adapters, and their conformance fixtures.
enge consumes its output and owns audio realization. Do not introduce an enge
dependency into either standalone runtime.

Python is a first-class independently runnable implementation, not merely a
reference for Rust. A contributor can edit a selector and hear it through MIDI
without compiling Rust. Its core uses ordinary Python collections and exact
timing arithmetic, with Pydantic accepted for models. Installing, importing,
and running it must not require NumPy, a Rust extension, or any audio package.
Live MIDI needs an optional device adapter; event-file processing and core
tests do not. This is not a standard-library-only requirement.

Both runtimes implement the same built-in semantics and use shared behavioral
fixtures. Python-only custom algorithms need not execute in Rust; choosing the
Rust backend for an unsupported algorithm must report that limitation explicitly.
Keep these extensions outside bounded native/audio callbacks, consistent with
the provider policy. Measure runtime footprint and scheduling limits rather
than assuming Python meets the same real-time constraints as Rust.

Make this an acceptance requirement from the first implementation, not packaging
work deferred until after the audio engine has absorbed everything.

Deliver a self-contained executable that needs only the operating system and
its MIDI service. No enge installation, Python runtime, NumPy, sound device,
audio driver setup, sample assets, GUI, Rubber Band, or network service is needed.
Protocol/configuration libraries can be included at build time; users should
not have to install them separately. A physical device still needs whatever
support the operating system normally requires for that MIDI device.

Rust implementation boundary: a small portable event-core crate in the new project,
shared by the enge native adapter and the standalone executable. It depends on
time/event/algorithm facilities only. Keep PyO3, audio buffers, DSP, and sample
decoding in the enge adapter. uFor supplies authoring definitions and reference
semantics, with shared fixtures checking the native configuration parser and
runtime. The standalone executable reads the same authored arp profile; it
must not require a Python compilation step for each user preset.

The standalone shell owns port enumeration, input/output selection, timestamped
I/O, internal/external clock, capture controls, file import/export, and a bounded
status trace. Start with macOS CoreMIDI for the user's hardware, keeping OS
adapters outside the kernel. Other supported platforms get explicit builds and
their own MIDI tests. Do not run audio initialization just to obtain a timer.

Expose a concise CLI for ordinary playing and explicit subcommands for listing
ports, validating a profile, capturing a phrase, and rendering event files.
No daemon or GUI is required. Python CLIs follow the project's Tyro/Pydantic
conventions; they are not prerequisites for the standalone
executable. Choose its final public name during implementation.

Test the release artifact in an environment without the Python/audio stack.
Inspect the dependency graph and verify that forbidden libraries are neither
linked nor loaded. An in-memory/SMF event test must run without an available
MIDI port; a loopback-port test then verifies device scheduling separately.
Also test the Python package in an environment with Pydantic and its declared
non-audio dependencies but without NumPy, enge, or the Rust extension. Exercise
profile loading and event processing there, and MIDI playing with its optional
adapter. Both entry points must use the same authored profile semantics.

## 12. Small authoring examples

These examples use one canonical schema. Presets resolve to that schema rather
than maintaining a second set of runtime options.

### Familiar held-chord arp

```toml
kind = "arpeggiator"
name = "up"
title = "Up"

[body]
selection = { kind = "ascending" }
rhythm = { kind = "grid", step = "1/4 beat" }
```

Defaults: held bank, no latch, one octave, restart on first note after an empty
bank, local beat clock, gate 80% of a step, and current source velocity/expression.
Start at the next grid opportunity; an on-grid onset is eligible immediately.
The MIDI shell supplies a visible internal tempo or a selected external clock.

### Monophonic expressive phrase

```toml
kind = "arpeggiator"
name = "wind-memory"
title = "Remember eight gestures"

[body]
bank = { kind = "history", notes = 8, publish = "step" }
selection = { kind = "played", direction = "reverse" }
rhythm = { kind = "grid", step = "1/2 beat" }
expression = { source = "recorded", timing = "fit", gaps = "carry" }
```

The selected input profile identifies the controller's breath/bend messages and
capture-tail boundary. An otherwise identical live-breath preset replaces only
the breath lane's source. Do not discard the recorded version of that lane.

### Euclidean walk

```toml
kind = "arpeggiator"
name = "five-in-eight"
title = "Five attacks, walking through the bank"

[body]
bank = { kind = "latched", update = "replace" }
selection = { kind = "walk", moves = [-1, 1, 2], weights = [1, 3, 1], boundary = "wrap" }
rhythm = { kind = "euclidean", steps = 8, pulses = 5, rotation = 0, step = "1/4 beat" }
seed = 47
```

### A sample in its original order

```toml
kind = "arpeggiator"
name = "sample-notes"
title = "A recording as a keyboard"

[body]
bank = { kind = "regions", reference = "phrases:spoken-scale" }
selection = { kind = "ascending", key = "selection_key", repeats = 1 }
rhythm = { kind = "source" }
expression = { source = "recorded", timing = "original", gaps = "carry" }
```

The referenced bank maps increasing selection keys to consecutive source regions.
Its playback settings select original rate and continuous adjacent playback.
Changing the selection order rearranges those regions; choosing a MIDI output
requires an external sampler mapping.

## 13. State, limits, and observability

Snapshot the bank revisions, open captures, current entry-state tables, rhythm
phase, selector state, random counters, pending occurrences, output ownership,
channel allocations, and accepted clock observations. Reference immutable audio
assets and captured event pages by identity rather than copying them every step.

Bound bank size, open notes, captured events/bytes, phrase length, queued output,
polyphony, ratchet expansion, feedback depth, and pending channel reservations.
Retain data until no active occurrence references it. On capacity exhaustion,
reject new admission with a clear diagnostic and execute already-owned release
obligations. Do not evict a still-needed breath stream to make room for a note.

Reserve capacity for lifecycle and cleanup events. Dense continuous expression
may use declared coalescing with a value/time error bound, preserving initial,
final, and discontinuous changes. Notes, releases, pedal transitions, and atomic
MIDI message groups are not ordinary redundant controller samples.

Preparation resolves assets, presets, profiles, and bounds. The running event
kernel performs no file/network I/O or unbounded allocation. Trace output goes
to a bounded queue; the host handles formatting and storage. Deterministic
offline execution and live execution consume the same decisions and clocks.

## 14. Delivery in coherent slices

1. **Contracts and executable fixtures.** Specify note/cell/occurrence identity,
   ownership, exact times, default ordering, and the canonical profile schema in
   uFor. Write held-chord, wind-breath, gap, and marked-sample identity fixtures.
   Establish the new project's Python/Rust and event-core/audio boundaries
   before adding features.
2. **Classic arp plus standalone proof.** Implement held/latch banks, familiar
   orders, grid/gate/retrigger, releases, and the minimal MIDI executable. Ship
   usable Python and Rust plain keyboard arps, check their shared fixtures,
   and verify Python installation without NumPy or compilation and Rust
   installation without Python or the audio stack.
3. **Expressive MIDI and monophonic capture.** Add device profiles, controller
   entry state, note-local gestures, history/phrase capture, gap retention,
   recorded/live lane choice, and destination allocation. The wind-controller
   example is a release gate, not a later embellishment.
4. **Marked sample notes.** Lower region notes through existing sample playback.
   Prove exact source-order reconstruction, then explicit reordered playback,
   seams, tail policies, and optional pitch/time processing.
5. **Rhythmic and algorithmic range.** Euclidean masks, custom steps, ties,
   ratchets, probability, deterministic selectors, phrase substitution, voicing,
   polymetric lanes, and multi-arp composition. Cover each reference family with
   an authored profile and an exact decision trace.
6. **Motions and clock integration.** Share signal/event ports and snapshots;
   complete external MIDI-clock acquisition and explicit transport-position
   behavior on top of the common clock work. Keep live provider execution
   deferred until its capacity and timing contract exists.
7. **Performance and release hardening.** Measure event bandwidth, scheduling
   jitter, capture limits, recovery, and hardware expression. Publish supported
   destination profiles and known protocol limitations with the standalone builds.

Each slice gets focused commits and its own passing checks. No dependency or
roadmap changes are authorized by writing this plan. During implementation,
coordinate required uFor/enge changes as separately reviewable commits; do not
silently fold unrelated unfinished Motion features into an arp change.

## 15. Acceptance evidence

| Area | Required evidence |
| --- | --- |
| Common playing | Three held notes produce the exact expected up/down/played-order sequences, with documented single-note, empty-bank, duplicate-pitch, latch, retrigger, and chord-edit behavior. |
| Monophonic memory | Sequential notes build a bank without requiring overlap; completed gestures become eligible at the declared boundary and playing occurrences survive bank replacement. |
| Breath and bend | Reordering wind notes carries CC2-derived breath, the correct inherited initial state, later attack, bend offsets, release velocity when supplied, and selected tails. Legato onsets without intervening releases form segments under the explicit input profile; velocity-zero releases retain their original encoding. No silent default velocity substitution. |
| Between notes | Prefix, gap, suffix, pedal, and unknown events survive capture; each playback policy has exact emitted and retained-event traces. Unowned post-release breath updates the next source note's entry state without affecting an unrelated output note. Original-phrase replay preserves source semantic ordering. |
| Discrete Motions | Repeated equal-valued MIDI messages preserve their timestamps and order; the held-value view creates no extra messages or implicit interpolation and remains independent of note ownership. |
| MIDI allocation | Independent gestures on overlapping notes never collide silently on one channel; exhaustion, repeated keys, stop, bypass, disconnect, and snapshot reconciliation release exactly the owned notes. |
| Euclidean/algorithmic | Exact canonical masks and rotations, pulse counts, deterministic random choices, rest/tie/ratchet semantics, and polymetric cycle behavior. |
| Timing | Tempo changes, fractional divisions, stop/resume, seek cancellation, incoming pulses, and equal-time ordering agree across whole runs, irregular blocks, and restored snapshots. |
| Samples | Exhaustive marked regions reconstruct the decoded source exactly under identity settings; shuffled/repeated regions and seam policies have listenable regressions. |
| Motion integration | A Motion can change density or expression and receive a hit event without duplicate clock ownership or unbounded feedback. |
| Standalone | A shipped MIDI-only executable validates profiles and processes files without Python or audio libraries; loopback tests verify note/control order and scheduling. |
| Python accessibility | Install and run without NumPy, enge, audio libraries, or a Rust extension. A modified Python selector works without compilation; shared fixtures establish built-in parity with Rust. |
| Limits | Event storms and dense breath streams obey explicit capacity/quality policies while preserving release obligations. |

Protocol and selection tests use exact event traces; they do not need an audio
device. Audio regressions write WAV at 48,000 Hz and last at least one second.
Include an EWI phrase whose note-ons alone produce silence, and compare the
audible reordered performance with the correct expression attached. Hardware
checks on the user's wind controller and synth complement automated tests;
software parity alone cannot establish the receiver's articulation or tail rules.

## Handoff to the new project

Begin a dedicated project chat after choosing the name and creating its
repository. Use this plan as the handoff, starting with slice 1 and the dependency
boundaries in section 11. Keep enge/Motions work in its existing project context.
This documentation update creates neither the repository nor a new chat and
does not start implementation. Coordinate shared uFor changes explicitly when
development begins; keep one authoritative plan rather than diverging copies.

## Additional work beyond the prompt

None. This change adds the design plan only.
