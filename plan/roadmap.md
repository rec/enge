# Roadmap

## 1. Finish graph FM

Replace the fixed two-operator FM renderer with the graph-owned state model.
Integrate the existing four-operator profile through prepared actions, controls,
filters, snapshots, NumPy, and Rust. Add the six-operator profile, shared
topology vectors, and listenable 48 kHz regressions.

## 2. Add FM operator waveforms

Specify sine, triangle, and square phase/output conventions in uFor. Make the
choice per operator and latched at onset, then add matching NumPy/Rust tests.
Add Yamaha-compatible waveform and algorithm import only after the general
profile is stable.

## 3. Complete control integration

Carry named envelopes and addressed LFO rate/reset changes through every offline
and persistent runtime that supports their scopes. Add portable modulation
targets one at a time, with partition and snapshot conformance.

## 4. Extend real-time execution

Move remaining prepared-action orchestration into the native callback owner.
Keep device ownership in hosts. Expand live sampler modulation only with bounded
resource and callback contracts.

## 5. Add selected effects

Use the live-effects graph for the categories in `new-effects.md`. Start with
effects that fit its fixed resource model; add Rubber Band time/pitch processing
as an optional GPL-linked integration when its offline and live contracts are
specified.

## 6. Evaluate PyTorch last

Define a supported recurrence subset and eager/compiled conformance before
adding a PyTorch implementation. Do not make it a dependency or performance
claim before measurement.

## Additional work beyond the prompt

None.
