# Live, chainable audio effects

## Purpose and current state

Add reusable, stateful audio effects that can process every enge sound engine,
with the same behavior in offline rendering and live block processing. The first
musical effect is a granulator, usable on individual voices, instrument mixes,
or the final output mix.

Today, oscillator, sampler, FM, and noise voices share ordered dynamic filters.
Those filters run before amplitude and mixing, retain voice-owned state, and
stop when the voice completes. They are not a general effects system. uFor
already reserves shared post-mix processing for a separate processor graph.
The existing native render paths also do not establish an allocation-free,
Python-free audio callback path.

This plan proposes that architectural addition. Start with fixed serial chains
at explicit attachment points, not an arbitrary routing graph editor. Existing
voice filters keep their specified position and lifetime.

## Ownership and scope

- uFor owns portable effect definitions, ordered chains, attachment points,
  parameter units/domains, modulation bindings, and prepared actions.
- enge owns preparation, numerical processing, buffers, scheduling, effect state,
  independent NumPy/Rust implementations, and snapshots.
- Hosts own devices, transport, input capture, UI, MIDI/OSC adaptation, files,
  encoding, serialization of multiple control producers, and recovery after a
  reported processing failure.

Reuse the [engine execution contract](engine-execution.md), uFor control
evolution, existing LFO/control semantics, and explicit channel routing. Do not
introduce another raw musical-event API or reinterpret instrument selection.
PyTorch, C++, JUCE, plugin hosting, device backends, general sends/returns,
sidechains, convolution, and hot editing of chain topology are outside this plan.

## Signal path and attachment points

Define three ordered insertion points:

1. Per voice: after its existing source filters and amplitude/gain, before its
   channel routing and contribution to the instrument mix.
2. Per instrument: after its voices have been routed and summed into that
   instrument's output layout.
3. Master: after the instrument outputs have been summed.

Each attachment owns an ordered list of processors. A processor receives the
previous processor's output. The initial processors preserve channel count;
existing explicit channel maps handle layout changes. No implicit stereo
conversion or feedback edges between processors are allowed.

Every processor definition has a stable string ID unique within its chain. A
parameter address consists of attachment scope, attachment owner, processor ID,
and parameter. A per-voice processor instance is identified internally by voice
ID plus processor ID. Snapshots and diagnostics use these identities rather than
list positions, while chain order still determines processing order.

Per-voice effects receive independent instances. Instrument and master effects
retain state across note boundaries. The same processor implementation serves
all three placements, including direct processing of host-supplied audio blocks.
The normal block API can therefore granulate external live audio without owning
an audio input device.

## Preparation and processing contract

Preparation fixes sample rate, channel layout, maximum block length, chain
topology, and resource limits. Allocate delay/history buffers, scratch space,
grain pools, voice/tail slots, and action capacity before processing begins.
Reject invalid definitions during preparation wherever possible.

Use an explicit numerical boundary: input audio, parameter values/trajectories,
and previous state produce output audio and next state. The NumPy reference
keeps this boundary readable and suitable for a later tensor implementation;
Rust may mutate exclusively owned prepared state and caller-owned output buffers.
Do not require allocating and returning a new state object for each native block.

Audio keeps enge's `(frames, channels)` layout and float64 reference convention.
Calls advance a contiguous absolute frame cursor. Support variable block sizes
up to the prepared maximum and define a zero-frame call as a no-op. Rendering
the same input and actions with different block partitions must agree within
the numerical tolerance. Define buffer ownership and aliasing explicitly;
the initial public processing API uses separate input and output buffers.

No Python calls, allocation/deallocation, blocking locks, file access, decoding,
logging, or unbounded work belong in the native live processing path. Prepare
typed actions and bounded control trajectories outside it; evaluate ongoing
controls and apply actions in Rust using the same semantics as the reference.
Existing Python orchestration must not be invoked from that path.

Live controls use absolute integer sample-frame timestamps. A non-audio thread
converts them into bounded typed action batches and submits complete batches to
a fixed-capacity single-producer/single-consumer queue. A host with several
producers serializes them before submission. The audio thread consumes the queue
without allocating or locking. Queue exhaustion rejects the whole batch before
acceptance. An action whose frame has passed is discarded, reported as late, and
never applied retroactively. The host schedules an untimestamped observation at
the next block boundary. Offline action slices and live queued batches enter the
same ordered action consumer rather than creating two execution semantics.

Expose a native processing entry point usable by a native host without the GIL.
Python bindings remain useful for reference tests, offline rendering, setup, and
inspection; releasing the GIL inside a Python call alone is not proof of live
callback safety. Integrate the existing native sources through prepared,
nonallocating render-into paths as part of completing live source-to-output use.

Preparation and submission reject every error detectable outside processing. A
fatal callback error, including non-finite input, non-finite DSP state/output, or
an internal invariant failure, writes zeros for the rest of the block and latches
the instance as failed. Later calls write zeros until reset or reconstruction
outside the callback. Record the first failure in bounded preallocated status
storage; do not unwind across the native boundary. Rejected or late actions are
admission errors, leave current processing unchanged, and do not fail the engine.

Recursive DSP can enter the subnormal range during long decays and cause
processor-dependent timing spikes. After each recursive state update, replace a
finite state value whose magnitude is below `1e-30` with positive zero in both
NumPy and Rust. Apply this portable float64 rule to recursive state only, not to
input or output samples. Specify the comparison point in every recurrence so
block partitioning and backend comparisons retain one numerical meaning; do not
depend on host floating-point flush-to-zero modes.

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

Provide a shared wet/dry control. Smooth bypass by ramping it to dry while still
processing the effect and advancing its state. Suspending an effect is not a
second bypass mode in this profile. Reset is an explicit operation performed
while stopped; transport position jumps require reset or snapshot restoration.

Source completion ends input to the first processor. That processor continues to
feed its tail to the next processor and reports the exact frame at which it is
drained; only then does the next processor's input end. This propagates serially
through the chain, including when boundaries fall within a block. A chain is
drained only when its final processor drains. A voice with an effect tail retains
its identity and routing during this process. Instrument and master chains follow
the same rule after their inputs finish. Existing source filters still end with
their source; they do not acquire new tails implicitly.

Define separate operations for ending input and explicitly stopping processing.
Ending input starts the serial drain process. Stopping cuts output and retires
state at its specified frame. Each processor declares whether its current state
can drain and reports an exact `drained_at` frame rather than only a block-level
boolean. An active freeze can sustain indefinitely. Offline callers must give a
finite render boundary for such a state or schedule its release. Never infer
completion solely from one silent block.

Prepare separate capacities for active source voices, retained effect tails,
queued action batches/actions, and grains. Source-voice capacity must cover the
prepared uFor lifecycle rather than rejecting an already accepted start in the
callback. A full live action queue rejects an entire batch during submission. A
full tail pool retires its oldest tail at the incoming instance's frame using a
fixed portable de-click fade; allocate the fade capacity in advance and specify
the maximum tails, fade duration, and `(start frame, identity)` tie-break order.
A full grain pool skips a scheduled grain while advancing its scheduler and random
sequence. Record bounded counters for each condition without logging in the
callback.

## Latency and snapshots

Each processor declares fixed algorithmic latency for its prepared settings.
Sum latencies through a chain and align every path entering a fan-in to the
greatest upstream algorithmic latency, both where voices enter an instrument and
where instruments enter the master mix. Delay the dry path by the processor's
declared latency when mixing wet and dry. Allocate compensation buffers during
preparation. The prepared engine reports one fixed total latency in integer
frames. An action at frame `n` affects input frame `n`, and its result is audible
at output frame `n + total latency`; a host can translate user-facing timestamps
when it wants perceptual rather than input-time scheduling.

Distinguish algorithmic latency from intentional musical delay: a delayed grain
or echo is part of the effect and must not be automatically canceled. Specify
the granulator's timeline accordingly. Device latency remains the host's concern.
Reject configurations requiring a changing compensation topology during playback.

Snapshots include effect definitions/identity checks, frame cursor, modulation
state, all audio history and compensation buffers, active grains, random counters,
and drain status. Snapshot extraction/restoration happens while processing is
stopped, outside the callback. Pending future actions remain caller-owned, as in
the engine execution contract: stop submissions and empty the live transport
queue before taking a snapshot, then resubmit future batches after restoration.
Follow existing backend compatibility rules.

## Initial processors

### Gain and shared filters

Use gain to establish chaining, exact automation, attachment points, and tails
without starting with a complex algorithm. Expose the existing resonant filter
kernel through the processor contract, reusing its implementation and dynamic
cutoff/Q rules. This adds post-mix filtering without replacing or moving the
existing per-voice filter stage. Define a finite tail cutoff for processor-mode
filters; document and test the truncation policy.

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
Use portable, tested interpolation and window equations, explicit maximum grain
count, and a documented gain rule. Do not hide clipping with normalization or a
limiter. Freeze stops history writes while grain scheduling continues; define
wrap and unfreeze transitions so old history is never mistaken for new input.

On end-of-input without freeze, stop launching grains and drain active grains;
the tail is bounded by maximum grain duration. Frozen operation continues until
freeze is released or an explicit stop occurs. Define this behavior identically
for standalone processing and every attachment point.

## Implementation sequence and acceptance

1. Specify the portable chain and action contracts in uFor, including timing,
   stable processor identities, scope, capacities, tails, latency, live admission
   errors, fatal processing errors, and granular equations. Add conformance
   examples before updating enge's dependency in its own dependency commit.
2. Implement preparation and the independent NumPy chain reference with gain
   and filter processors. Integrate offline instrument/master placement, then
   per-voice placement with retained tails. Preserve no-effects rendering.
3. Implement Rust chain processing, native control evaluation, and prepared
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
   Include short blocks, variable partitions, multiple chains, and tail overlap.
   Report machine, sample rate, block sizes, capacities, and headroom. A harness
   result does not establish physical device reliability; host/device validation
   remains a separate integration step before claiming end-to-end live readiness.

Run the same behavior-focused cases against NumPy and Rust: frame-exact action
ordering, chain order, channel independence, all attachment scopes, dry/wet and
bypass transitions, processor addressing, latency alignment at every fan-in,
serial tail completion, every capacity boundary, late action handling, fatal
failure latching, subnormal state truncation, random determinism, and snapshots.
Compare contiguous rendering with irregular block partitions. Discrete results
agree exactly; audio uses declared tolerances.

Audio regression fixtures write at least one second of 48 kHz WAV, with listenable
FLAC demos. Include input ending mid-block and controls changing at block edges.
Keep allocation/deadline measurements separate from numerical unit tests.
Completion requires all four sources to use the shared effects path, both
backends to pass conformance, and the native processing audit and timing results
to be documented. It does not require implementing the deferred effect families.

## Additional work beyond the prompt

None.
