# Yamaha closeout record

Reconciled on 2026-11-05 against the original Yamaha plan (`7bdf9ea`), the
subsequent approved model/API decisions in [yamaha-emulation.md](yamaha-emulation.md),
the current implementation and the committed comparison reports.

The user explicitly requested a solid, prompt closeout of the Yamaha work on
2026-11-05, recognizing it as a small part of enge. That decision supersedes the
expanded nine-package completion backlog below. The tested provisional
[offline profile](../doc/yamaha-profile.md) includes the completed EG-bias change;
the other feature/fidelity investigations are deferred. Do not
continue adding Yamaha slices automatically. Deferred work is not represented
as implemented or hardware-qualified.

## Delivery contract

- Deliver the implemented provisional offline DX7, TX81Z and WT11 profiles, keeping their
  portable schemas and state separate from general FM.
- Preserve lossless imported bytes, reject malformed/unimplemented playable
  configurations explicitly, and retain owned continuation through snapshots.
- Retain the existing named-patch/software comparisons, source revisions,
  inputs, metrics and WAV artifacts, including their failing qualification
  results. Further reference convergence is deferred.
- Preserve the user's approved provisional models and intentional departures
  from reference bugs. Explain and bound discrepancies; do not silently change
  approved timing or sound merely to match a different implementation.
- Keep the optional effects/new-engine catalogues deferred. Imported DSP bytes
  do not automatically select those effects for implementation.

The original delivery boundary is **offline first**. A callback-safe live Yamaha
backend was conditional on measured fixed state/allocation bounds, not an
additional promised backend. Hardware captures and firmware tracing are useful
qualification routes; the user has no ROM and explicitly asked to proceed
without one. Neither is a prerequisite for software delivery. Do not claim
hardware equivalence without the corresponding evidence.

## Completed baseline

The general engine/control/sampler roadmap is recorded as complete in
[roadmap.md](roadmap.md). Existing Yamaha support includes all three offline
voice profiles, shared clocks, ordinary routing/processing, generators and tempo
maps, minimum hold, fades, mono/glide handling where supported, MIDI-file
dispatch and owned JSON continuation. WT11 import includes all five edit buffers
and VMEM/PMEM/PMEM2 banks; its note renderer retains its pitch projection.

Commit `d8efe21` adds WT11 aftertouch LFO pitch/amplitude assignments through
the direct and scoped-control paths. The verified baseline is **3,119 tests
passed, 21 skipped**, with Ruff, formatting, typing and pyupgrade checks clean.
The focused aftertouch run passed 36 cases. These counts establish regression
health, not full completion or measured hardware accuracy.

## Feature dispositions after the closeout decision

All entries except C1 are deferred. Their future acceptance conditions are
retained for an explicit later request, not as current completion gates.

| ID | Package | Delivered evidence or future work if explicitly resumed |
| --- | --- | --- |
| C1 | Delivered: WT11 aftertouch EG bias | ACED2 EG bias reuses measured TX81Z curves with an explicit strongest-relief combination model. Eighteen focused regressions cover per-operator EBS, zero/default pressure, TL/foot saturation, mixed operator identity, scoped EG-only controls, invalid-input atomicity and held/released JSON replay. Final full suite: 3,135 passed, 21 skipped. |
| C2 | TX81Z/WT11 breath pitch bias | Establish and implement signed assignment-to-pitch and pressure scaling, each model's wire domain, fixed/ratio behavior and composition with tuning/glide/bend. A provisional model must be labelled and justified; the unidentified firmware table alone is insufficient. Replace the current nonneutral-bias rejection only when that path renders and has regression evidence. |
| C3 | WT11 aftertouch pitch bias | Implement the separate direct-pitch assignment and its normalized pressure route, preserving the center-50 field and full WT11 domain. Establish its transfer model and fixed/ratio behavior explicitly. Verify composition, pressure endpoints, release and snapshots; LFO modulation does not close this package. |
| P1 | TX81Z/WT11 performance allocation and settings | Connect imported instrument allocations and explicit voice references to playback: allocation limits/priority, ordinary versus alternate assignment, key ranges/shifts, detune, level and output selection, with retained identity and continuation. Cover performance-mode mono key memory separately from the implemented Single-mode behavior. Imported numbers must resolve explicitly or fail admission. Classify every PCED field, including hardware-ignored compatibility fields, rather than silently treating inspection as execution. |
| P2 | Performance LFO/control ownership | Complete the performance-to-owner/selector integration, including WT11 and its PCED2 breath/lip/off LFO control. Preserve the first-two-allocated-owner rule independently of sounding-note order, shared clocks, off and independent vibrato behavior, and scoped host extensions. Existing explicit TX81Z LFO groups and provisional vibrato are the starting point, not missing primitives to reimplement. Demonstrate allocation/owner changes and snapshot replay with complete performances. |

## Deferred comparison and expanded audit

| ID | Deferred package | Future work if explicitly resumed |
| --- | --- | --- |
| R1 | DX7 complete-note envelope/reference differences | Isolate the remaining attack/release trajectories, block gain interpolation and silence admission in the pinned complete-note comparisons. Preserve approved stable-duration holds and the integer core. Resolve unexpected differences or attribute them quantitatively to the selected model using independent envelope evidence; feeding enge's own envelopes into a reference kernel does not qualify those envelopes. Retain named inputs, metrics and paired WAVs. |
| R2 | Four-operator frequency/shift reference boundaries | Resolve the software-reference interpretation of fixed-frequency increments and the OP1/register-slot envelope-shift disagreement. Existing nominal-frequency equations and passing ratio/core comparisons are evidence to reuse. Do not adopt ymfm's identified fixed-frequency scale inconsistency as an oracle. Record a checkable reference for the chosen model and disposition of conflicting evidence, with mixed fixed/ratio and shifted-envelope render comparisons. Physical chip qualification remains a separate evidence limit. |
| R3 | Complete-patch reference matrix | Consolidate complete named DX7/TX81Z/WT11 patch comparisons with the licensed or constructed bytes, pinned references and declared model differences. Cover asymmetric operators, feedback, dynamic envelopes/modulation, fixed/ratio modes, release and continuation. Reuse existing passing captures and WAVs. Distinguish a component pass from a complete-note pass, and a projected WT11 model from independently measured WT11 behavior. Publish which cases pass, differ intentionally or still fail unexpectedly. |
| A1 | Final parameter/requirement disposition audit | Map the original numbered requirements and every admitted voice/performance field to execution, documented hardware-ignored behavior, an explicit unsupported boundary, or a justified delivery boundary. In particular, verify the four-operator sustain/portamento foot-switch flags (VCED 68/69) against their own manuals; validation or a missing indexed read alone does not prove correct behavior. Check model-specific performance/DSP fields without adding the deferred effects catalogue. Reconcile stale plan statements, rerun the required checks for changed code/data, and confirm all required software packages above have closure evidence before marking the goal complete. |

These deferred packages remain unimplemented or unqualified as described.
Closing the current profile does not claim that they passed. Source-level
controls, integer-core agreement, a green suite and stored performance bytes
are not substitutes for the missing execution/reference evidence.

## Evidence limits retained after software delivery

The current models include intentionally provisional DX7 calibration, WT11 pitch
projection, OPZ waveform/control/clock behavior and host-side ownership choices.
List their precise remaining hardware/firmware uncertainties in the comparison
reports even after the required software evidence is complete. Device deadlines,
foreign allocators, listening qualification, a native live Yamaha backend and
new optional synthesis/effects are not silently added to this checklist.

## Closeout evidence

The final full suite passed: **3,135 passed, 21 skipped** in 884.43 seconds.
Ruff (`B,E,F,I`), formatting, `ty check enge`, pyupgrade on changed Python files
and `git diff --check` are clean. The supported-profile page documents explicit
rejections and metadata/performance limits. The roadmap and WT11 model report
reflect the narrowed scope. This closes the current software milestone;
no further Yamaha feature slice is scheduled under this goal.

## Additional work beyond the prompt

None.
