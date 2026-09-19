# Live, chainable audio effects

## Purpose and current state

Add reusable, stateful audio effects that can process every enge sound engine,
with the same behavior in offline rendering and live block processing. The main
new musical effect is a granulator, usable on individual voices, instrument
mixes, or the final output mix.

Today, oscillator, sampler, FM, and noise voices share ordered dynamic filters.
Those filters run before amplitude and mixing, retain voice-owned state, and
stop when the voice completes. They are not a general effects system. uFor
already reserves shared post-mix processing for a separate processor graph.
The existing native render paths also do not establish an allocation-free,
Python-free audio callback path.

This plan proposes that architectural addition. Use fixed prepared acyclic
processor graphs at explicit attachment points. A serial chain remains the
concise common form, while named audio input ports permit processors such as
ring modulators and sidechain compressors. This is a portable data and execution
model, not an arbitrary routing graph editor. Existing voice filters keep their
specified position and lifetime.

## Ownership and scope

- uFor owns portable effect definitions, named audio ports and connections,
  attachment points, parameter units/domains, modulation bindings, and prepared
  actions.
- enge owns preparation, numerical processing, buffers, scheduling, effect state,
  independent NumPy/Rust implementations, and snapshots.
- Hosts own devices, transport, input capture, UI, MIDI/OSC adaptation, files,
  encoding, serialization of multiple control producers, and recovery after a
  reported processing failure.

Reuse the [engine execution contract](engine-execution.md), uFor control
evolution, existing LFO/control semantics, and explicit channel routing. Do not
introduce another raw musical-event API or reinterpret instrument selection.
PyTorch, C++, JUCE, plugin hosting, device backends, user-addressable reusable
send/return buses, convolution, feedback connections, and hot editing of graph
topology are outside this plan. Fixed connections directly to named processor
inputs, including sidechains, are included.

## Signal path and attachment points

Define three ordered insertion points:

1. Per voice: after its existing source filters and amplitude/gain, before its
   channel routing and contribution to the instrument mix.
2. Per instrument: after its voices have been routed and summed into that
   instrument's output layout.
3. Master: after the instrument outputs have been summed.

Each attachment exposes one or more named input streams and owns a prepared
processor graph with one public output. Its default `main` input is the signal at
the insertion point described above. Additional inputs are explicitly routed
from declared host streams or named instrument pre-effect mixes/post-effect
outputs. A per-voice graph may also use its own source output, but cannot name
another dynamically allocated voice. A master output cannot feed its ancestors.
Extend uFor's stream references and existing channel-map semantics to express
these connections; current channel routing alone is not an implementation of
cross-instrument dependencies. Validate cycles across the entire expanded
source/mix/processor graph, not just inside each attachment. For example, two
instruments may reference each other's pre-effect mixes, but mutually dependent
post-effect outputs are a cycle and must be rejected.

Every processor type declares fixed named input ports and one output. Gain,
filter, and granulator have `input`; a ring modulator can have `carrier` and
`modulator`; a sidechain compressor can have `input` and `detector`. Each port
declares its channel layout, whether it is required, and its exact behavior after
that input ends. A processor with wet/dry behavior identifies the input used for
the dry path; the detector input of a sidechain processor is never mixed into its
audio output. There is no universal fallback for an unconnected port. A type may
declare a specific fallback, such as using `input` as its detector, but otherwise
preparation rejects a missing required connection.

Audio ports carry sample-rate signals and remain distinct from scalar control,
automation, envelope, and LFO inputs. An audio-rate carrier, modulator, or
detector is never silently converted into a control trajectory, and a control
route cannot satisfy an audio port.

A connection joins an attachment input or one processor output to exactly one
processor input port. Outputs may fan out. A port has only one connection; an
explicit mixer is required when several signals must feed it. Validate channel
layouts, required connections, references, and graph acyclicity during
preparation, then establish one stable topological execution order, breaking ties
by canonical instance path. Execute a producer once per processing span and share
its immutable output with all consumers; fan-out must not advance its state twice.
Feedback edges remain unsupported until a later contract supplies explicit delay and
cycle semantics. Serial-chain notation is shorthand that connects each
processor's primary input to the preceding output; normalize it into the same
graph before validation, without a separate chain runtime or snapshot format.
The initial processors preserve the primary input's channel count; no implicit
stereo conversion occurs.

Every processor definition has a stable string ID unique within its graph. A
parameter address consists of attachment scope, attachment owner, processor ID,
and parameter. A per-voice processor instance is identified internally by voice
ID plus processor ID. Snapshots and diagnostics use these identities rather than
list positions.

Per-voice effects receive independent instances. Instrument and master effects
retain state across note boundaries. The same processor implementation serves
all three placements, including direct processing of host-supplied audio blocks.
The normal block API can therefore granulate external live audio without owning
an audio input device.

## Preparation and processing contract

Preparation fixes sample rate, every port's channel layout, maximum block length,
graph topology and execution order, and resource limits. Allocate every input
buffer, delay/history buffer, scratch region, grain pool, voice/tail slot, and
action capacity before processing begins. Reject invalid definitions during
preparation wherever possible.

Use an explicit numerical boundary: named input audio arrays, parameter
values/trajectories, and previous state produce output audio and next state. Port
names resolve to fixed prepared indices before processing; the callback does not
perform string or dictionary lookup. The NumPy reference keeps this boundary
readable and suitable for a later tensor implementation; Rust may mutate
exclusively owned prepared state and caller-owned output buffers. Do not require
allocating and returning a new state object for each native block.

Audio keeps enge's `(frames, channels)` layout and float64 reference convention.
Calls advance a contiguous absolute frame cursor. Support variable block sizes
up to the prepared maximum and define a zero-frame call as a no-op. Rendering
the same input and actions with different block partitions must agree within
the numerical tolerance. Define buffer ownership and aliasing explicitly; the
initial public processing API uses read-only input buffers and disjoint writable
output buffers. Inputs may share immutable storage through fan-out. All connected
ports cover the same frame interval and sample rate, though channel layouts may
differ where the processor type explicitly supports it. Host conversion and
resampling happen before this boundary.

No Python calls, allocation/deallocation, blocking locks, file access, decoding,
logging, or unbounded work belong in the native live processing path. Prepare
typed actions and bounded control trajectories outside it; evaluate ongoing
controls and apply actions in Rust using the same semantics as the reference.
Existing Python orchestration must not be invoked from that path.

Live controls use absolute integer sample-frame timestamps. A non-audio thread
converts them into bounded typed action batches and submits complete batches to
a fixed-capacity single-producer/single-consumer queue. A host with several
producers serializes them before submission. The audio thread consumes the queue
without allocating or locking. Bound batch size, pending actions, and actions per
frame; a finite queue alone does not bound work if its producer keeps refilling it.
At callback entry, capture the published queue boundary and consume at most that
prefix. New submissions become eligible on the next callback. Publish the first
unclaimed frame so the host can schedule untimestamped observations beyond the
currently claimed interval; retain a configured scheduling lead to allow delivery.

The producer submits actions in nondecreasing frame/ordinal order and retains
distant future score events off the queue, so they cannot block live controls.
A batch preserves every action emitted by one prepared musical event. Reject an
invalid or over-capacity batch before publication, without committing its producer
lifecycle transition. Never drop only a release, start, or context from that event.
The consumer reports a raced late batch without applying any of it. A rejected
control-only batch is nonfatal; a late lifecycle batch requires stopping and
resynchronizing the stream, since later prepared actions may depend on it.
Provide an out-of-band bounded stop request that cannot be blocked by queue
exhaustion. No late batch is applied retroactively. Offline slices and admitted
live batches use the same ordered action executor; live admission errors are a
transport policy, not a change to strict offline validation.

Expose a native processing entry point usable by a native host without the GIL.
Python bindings remain useful for reference tests, offline rendering, setup, and
inspection; releasing the GIL inside a Python call alone is not proof of live
callback safety. Integrate the existing native sources through prepared,
nonallocating render-into paths as part of completing live source-to-output use.

Preparation and submission reject every error detectable outside processing. A
fatal callback error, including non-finite input, non-finite DSP state/output, or
an internal invariant failure, discards the current graph render, zeros the entire
caller's output block, and latches the graph instance as failed. Rendering into
preallocated scratch before publishing prevents a failing branch from leaking
partial or non-finite output. Later calls write zeros until reset or reconstruction
outside the callback. No rollback is promised. Record the first failing processor
and frame in bounded preallocated status storage; do not unwind across the native
boundary. Numerical block-partition invariance applies to successful rendering;
the discarded block after a fatal error depends on the host's block boundary.

Recursive DSP can enter the subnormal range during long decays and cause
processor-dependent timing spikes. After each recursive state update, replace a
finite, explicitly designated decaying audio state whose magnitude is below
`1e-30` with positive zero in both NumPy and Rust. This is an intentional signal
floor, not the IEEE float64 subnormal boundary. Never apply it indiscriminately
to oscillator phase, time, control values, or compensated-summation corrections.
Specify the state fields and comparison point in each applicable recurrence.
Reusing the source filter kernel requires an explicit contract update and
regression review for this numerical change; do not silently change old filters.

State trimming does not prevent subnormal input or intermediate arithmetic, so
it is not a complete real-time performance guarantee. Benchmark tiny inputs and
long decays on each supported target, and document whether a native thread-local
floating-point mode is needed. Such a mode must be scoped/restored and validated
against reference tolerances before enabling it. See Intel's distinction between
[denormal operands and results](https://www.intel.com/content/www/us/en/docs/dpcpp-cpp-compiler/developer-guide-reference/2024-2/set-the-ftz-and-daz-flags.html).

## Dynamic controls and chain lifetime

Reuse exact frame ordering, ramp semantics, modulation mapping, and deterministic
LFO state. Events split processing at their scheduled frame. Changes take effect
without resetting processor history. Structural settings and resource bounds
remain fixed until the stream is stopped and prepared again. Current per-voice
instances receive controls through their existing scope; a new instance observes
the current trigger context, and an old tail retains its old trajectory under
uFor's trigger-ID reuse rules.

Use control interval 1 as the numerical reference. Apply the existing configurable
control interval only where its defined approximation permits; delay feedback,
grain playback, resonators, and other audio recurrences remain audio-rate.
Specify smoothing per parameter when the effect requires it, rather than adding
an undocumented smoothing layer to all controls.

Provide a shared wet/dry control `m` in `[0, 1]`, with
`output = (1 - m) * aligned_dry + m * wet`. Linear mixing preserves unity when
dry and wet are identical; no implicit equal-power gain boost is introduced.
Smooth bypass by ramping it to dry while still processing the effect and advancing
its state. Suspending an effect is not a
second bypass mode in this profile. Reset is an explicit operation performed
while stopped; transport position jumps require reset or snapshot restoration.
Store the authored wet/dry trajectory independently from the bypass ramp so
unbypassing returns to the current authored mix. Preparation must define the
bypass ramp duration in frames. Shared instrument/master effects use persistent
control contexts; they cannot implicitly inherit the most recent note's velocity
or trigger context. Trigger-scoped bindings require a per-voice owner.

Each attachment input ends independently at an exact frame. A processor receives
an end marker separately for each port; ended ports supply zeros if the processor
continues. The processor type declares which inputs govern its lifetime and its
behavior for every relevant combination of ended inputs. A ring modulator can
finish when its carrier ends, while a sidechain compressor can continue its main
input and release gain reduction after its detector ends.

A processor continues producing its tail and reports the exact frame at which
its output is drained. Only then does output-end propagate along its outgoing
edges, including when the boundary falls within a block. The graph is drained
when its public output drains. Reject processors disconnected from the public
output. Connected inputs must originate at a declared attachment stream;
type-declared optional-port fallback is resolved explicitly during preparation.
A voice with an effect tail retains its identity and routing during this process.
Instrument and master graphs follow the same rule after their inputs finish.
Existing source filters still end with their source; they do not acquire new
tails implicitly.

Temporary silence or having no active voices does not end an instrument/master
input: later notes must still be accepted. Only explicit stream completion ends
those inputs. Once a graph output ends, it cannot restart without reset. End
markers denote the first frame with no further output, and compensation buffers
must drain before forwarding them. A shared upstream producer remains alive for
its other consumers when one consumer ends. The wet/dry wrapper remains alive
until both the designated dry path and the processor's wet tail end, including
while bypassed; an infinite auxiliary input must not retain a voice after its
declared lifetime-driving input and tails have ended.

Define separate operations for ending input and explicitly stopping processing.
Ending an input starts propagation and draining under the processor's port rules.
Stopping cuts the graph output and retires state at its specified frame. Each
processor declares whether its current state can drain and reports an exact
`drained_at` frame rather than only a block-level boolean. An active freeze can
sustain indefinitely. Offline callers must give a finite render boundary for
such a state or schedule its release. Never infer completion solely from one
silent block.

Prepare separate capacities for active source voices, retained effect tails,
queued action batches/actions, and grains. Offline preparation verifies voice
capacity against its lifecycle. Live preparation declares a voice limit and the
producer reserves capacity before accepting an onset and committing its lifecycle
transition; unknown future live input cannot be guaranteed at initial setup.
A full tail pool retires its oldest tail using a fixed de-click fade. Specify
oldest by `(tail-entry frame, canonical instance path)`, and include fading tails
in the resource budget. Derive reserved fade slots from the admitted maximum
tail-entry rate and fade duration, including simultaneous source completions.
Reject preparation if that bound is unavailable; a fade must not assume an
unlimited supply of replacement slots. Record the exact fade equation and duration
in the portable profile before implementing it.
A full grain pool skips a scheduled grain while advancing its scheduler and random
sequence. Record bounded counters for each condition without logging in the
callback.

## Latency and snapshots

Each processor declares fixed algorithmic latency for its prepared settings.
Propagate cumulative latency through the graph's topological order and delay
every path entering a multi-input processor to the greatest input-path latency.
Apply the same rule where voices enter an instrument and instruments enter the
master mix. Delay a processor's designated dry input by its algorithmic latency
when mixing wet and dry. Allocate compensation buffers during preparation. The
prepared engine reports one fixed total latency in integer frames. An action at
frame `n` affects input frame `n`, and its result is audible at output frame
`n + total latency`; a host can translate user-facing timestamps when it wants
perceptual rather than input-time scheduling. Compute these delays from all
prepared paths, including currently silent voices, so starting a voice cannot
change the latency of another voice.

That timing rule requires delaying processor controls as well as audio: a node
whose aligned inputs have cumulative delay `d` applies an action authored at
frame `n` at processing frame `n + d`. Evaluate its ramps/LFOs on the corresponding
logical clock. Shift port end markers through each edge's compensation delay too.
For example, a 128-frame upstream delay followed by automated gain must apply a
gain change at logical frame 100 to the audio arriving at frame 228. This is
alignment of logical time, not a promise that a lookback granulator responds
audibly immediately. Intentional musical delays keep their processor semantics.

Distinguish algorithmic latency from intentional musical delay: a delayed grain
or echo is part of the effect and must not be automatically canceled. Specify
the granulator's timeline accordingly. Device latency remains the host's concern.
Reject configurations requiring a changing compensation topology during playback.

Snapshots include effect definitions/identity checks, frame cursor, modulation
state, every port's ended state, all audio history and per-edge compensation
buffers, active grains, random counters, and graph drain status. Snapshot
extraction/restoration happens while processing is stopped, outside the callback.
Pending future actions remain caller-owned, as in the engine execution contract:
stop submissions and processing, recover all unconsumed transport batches to the
caller without applying or dropping them, then take the snapshot. Actions already
accepted by the executor but delayed internally for latency alignment belong to
the snapshot, along with their ordering and reservation state. Restore with the
same graph/asset digest and capacities, then resubmit only the returned transport
batches. Test that every event executes exactly once. Follow existing backend
compatibility rules.

## Initial processors

### Gain and shared filters

Use gain to establish chaining, exact automation, attachment points, and tails
without starting with a complex algorithm. Expose the existing resonant filter
kernel through the processor contract, reusing its implementation and dynamic
cutoff/Q rules. This adds post-mix filtering without replacing or moving the
existing per-voice filter stage. Define a finite tail cutoff for processor-mode
filters; document and test the truncation policy.

Use a stateless two-input multiplication processor as the first conformance case
for named ports, fan-out, per-input end behavior, and latency alignment. Its
`carrier` and `modulator` ports have explicitly routed matching channel layouts,
and its output is their sample-by-sample product. This is also a basic ring
modulator; it should remain small rather than expanding this plan into an effect
catalog. A sidechain compressor is not part of the initial implementation, but
must fit the same port and detector-input contract without changing the graph API.

### Granulator

Record input into a bounded per-instance circular history buffer and schedule
windowed grains that read it. Initially support grain duration, density, lookback,
playback pitch ratio, position jitter, wet/dry, and freeze. Use a fixed window
shape and forward playback first; reverse grains and separate sample instruments
can follow only if requested. No feedback control is included initially.

Use shared scheduling across channels to preserve their timing relationship.
Specify deterministic per-instance random keys and counters using the existing
noise infrastructure, independently of block partitioning and other instances.
Latch duration, pitch, and source position when a grain starts; later changes
affect subsequent grains. Wet/dry changes apply continuously.

Initialize storage to zero for memory safety, but mark startup history unavailable.
Do not launch a grain until its complete requested read range contains captured
input. Preparation and grain launch validation must also ensure a grain's read
trajectory cannot reach future input or history overwritten before it ends.
Use absolute history indices with an explicit valid interval, including the
interpolation kernel's neighboring samples. A skipped unavailable-history grain
advances the same scheduler/random counters as a skipped full-pool grain. Bound
density, duration, pitch, jitter, and history at preparation; specify scheduler
phase accumulation and parameter-change ordering so splitting a block cannot
change which grains launch.

Use portable, tested interpolation and window equations, explicit maximum grain
count, and a documented gain rule. Do not hide clipping with normalization or a
limiter. Freeze latches the valid history interval; it cannot expose unrecorded
storage. Define its wrap and boundary crossfade explicitly. On unfreeze, existing
grains must retain their captured samples until completion: use preallocated
per-grain storage or an equivalently bounded retention scheme, never overwrite
samples still referenced by a grain. History storage and freeze transitions must
stay bounded under repeated toggles, with no callback allocation.

On end-of-input without freeze, stop launching grains and drain active grains;
the tail is bounded by maximum grain duration. Frozen operation continues until
freeze is released or an explicit stop occurs. Define this behavior identically
for standalone processing and every attachment point.

## Implementation sequence and acceptance

1. Specify the portable graph and action contracts in uFor, including named ports
   and connections, stable processor identities, timing, scope, capacities,
   per-input lifetime, tails, latency, live admission errors, fatal processing
   errors, and granular equations. Add conformance examples before updating
   enge's dependency in its own dependency commit. Before DSP implementation,
   settle the remaining numerical profile constants/equations: filter tail cutoff,
   retirement and bypass fades, granular window/interpolation and gain law,
   scheduler, and frozen-history wrap/crossfade. These are required contract
   deliverables, not backend-specific implementation choices.
2. Implement preparation and the independent NumPy graph reference with gain,
   filter, and two-input multiplication processors. Integrate offline
   instrument/master placement, then per-voice placement with retained tails.
   Preserve no-effects rendering and serial-chain shorthand.
3. Implement Rust graph processing, native control evaluation, and prepared
   source integration. Audit all reachable processing paths for allocations,
   locks, Python entry, and unbounded work. Keep one native DSP implementation
   per processor, used by both live and offline execution.
4. Implement granular reference and Rust processing against the same cases.
   Exercise history limits, pitch extremes, freeze transitions, pool exhaustion,
   and snapshot continuation before adding musical presets.
5. Add a short listening demo using oscillator, sampler, FM, and noise inputs,
   including chained filtering and granulation. Keep demo orchestration small
   and reusable processing in enge. Compare per-voice and shared placement.
6. Benchmark complete source-to-output processing under a native callback-like
   harness. Measure worst observed block duration and deadline misses, not just
   average throughput, with changing controls and maximum prepared occupancy.
   Include short blocks, variable partitions, multiple chains, worst admitted
   action density, fan-out, repeated tail retirement, and freeze toggles. Measure
   subnormal external input and long decays separately from ordinary signal levels.
   Report machine, sample rate, block sizes, capacities, and headroom. A harness
   result does not establish physical device reliability; host/device validation
   remains a separate integration step before claiming end-to-end live readiness.

Run the same behavior-focused cases against NumPy and Rust: frame-exact action
ordering, stable topological order, channel independence, all attachment scopes,
required and missing ports, fan-out, rejected cycles and implicit fan-in,
two-input multiplication, independent input endings, dry/wet and bypass
transitions, processor addressing, latency alignment at every processor and mix
fan-in, graph tail completion, every capacity boundary, late action handling,
fatal failure latching, subnormal state truncation, random determinism, and
snapshots. Compare contiguous rendering with irregular block partitions.
Discrete results agree exactly; audio uses declared tolerances.

Include regressions for an idle instrument receiving another note, a cross-scope
cycle, a fanned-out stateful source advancing once, and controls/end markers
traversing unequal-latency paths. Exercise queue publication races, rejected
whole-event batches, producer lifecycle rollback, bounded stop under queue
exhaustion, and snapshot restoration with pending compensated actions. Granular
cases must include insufficient startup history and unfreezing while old grains
still read their captured material. Verify unchanged source-only audio except
for any explicitly approved shared-kernel numerical change.

Audio regression fixtures write at least one second of 48 kHz WAV, with listenable
FLAC demos. Include input ending mid-block and controls changing at block edges.
Keep allocation/deadline measurements separate from numerical unit tests.
Completion requires all four sources to use the shared effects path, both
backends to pass conformance, and the native processing audit and timing results
to be documented. It does not require implementing the deferred effect families.

## Additional work beyond the prompt

None.
