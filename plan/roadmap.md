# Roadmap

Reconciled with local implementation on 2026-11-04. Current code and tests take
precedence over historical proposals in the other plans.

## Completed foundations

- Graph FM supports two through six operators, sine/triangle/square waveforms,
  delayed feedback, controls, filters and snapshots in NumPy and Rust, including
  persistent graph state.
- enveLFO owns standalone modulation execution. enge consumes its Python and Rust
  runtimes and retains voice lifetime, routing and audio integration.
- Authored units use uFor and reccy's shared parser; prepared DSP stays numeric.
- Native live mixing, bounded effect queues and sampler control/filter/generator
  integration exist. The tested Rust callback paths pass the heap-allocation
  gate; device deadlines and native host integration are separate boundaries.
- Effects include gain, filters, multiplication, granulation, soft clip,
  tap/modulated delay and optional Rubber Band adapters.

## Completed control/runtime parity and sampler coverage

The user authorized this work on 2026-11-02 after correcting the old plans.
Audit supported behavior against code and tests before treating old limitations
as missing features. Addressed LFO and Envelfo actions already have persistent
runtime coverage; do not implement a second event path.

1. Persistent modulation supports piecewise linear, step and exponential
   mappings and multiple-route composition with bounded prepared storage and
   shared generator state.
2. Control-driven crossfades now work offline, in the Python persistent
   sampler and in native live rendering. Delayed starts,
   alignment, directional random offsets, fade retirement, amplitude curves,
   key-event bindings, static key/velocity crossfades and Patch-child
   continuation are implemented.
   Regressions cover ordering, source exhaustion, partitions and snapshot continuation.
3. Native live sampler controls, generators and filters now reuse the existing
   native control/filter machinery with bounded callback storage, under the
   architecture approved on 2026-11-02. The instrumented suite measured zero
   Rust heap operations in 7,276 successful and seven failed callbacks across
   387 tests. This closes the measured shared-storage gate for tested paths;
   it does not qualify device deadlines, foreign allocators or arbitrary inputs.
   Device ownership remains in hosts.

Track verified details and remaining restrictions in
[engine-execution.md](engine-execution.md). Changes to portable schemas or
ambiguous musical behavior require a decision before implementation.

## Remaining committed work

The user confirmed on 2026-11-03 that the completion goal covers the concrete
roadmap and Yamaha requirements, while optional catalogues remain deferred.

- Complete and qualify [Yamaha emulation](yamaha-emulation.md); dedicated
  offline DX7 and provisional TX81Z renderers are already started.
  Outstanding work includes remaining controller behavior, integer/DAC
  behavior, full reference comparison and hardware qualification. Firmware
  tracing is optional: the user has no dump available and asked us to proceed
  without making it a prerequisite.
  The Yamaha adapter integrates seconds- and beat-clock enveLFO generators,
  generator snapshots and minimum-hold release timing. Beat-clock sources use
  the existing caller-owned tempo map, with tempo identity checked on restore.

## Deferred choices

- Select further effects from [new-effects.md](new-effects.md).
- Choose sound-quality improvements or new synthesis from
  [new-engines.md](new-engines.md). Its architectural proposals are not approved.
- PyTorch work remains excluded.

## Ownership outside enge

uFor owns portable definitions and preparation; reccy owns shared unit parsing;
enveLFO owns modulation execution; meedee owns MIDI protocols and transport;
safaz owns SFZ conversion; arpeg owns arpeggiator scheduling. enge retains the
audio realization of their contracts. Hosts own devices and application policy.

## Additional work beyond the prompt

None.
