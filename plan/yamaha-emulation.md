# Yamaha-compatible FM

## Status as of 2026-11-04

`enge/yamaha.py` already contains a Python DX7 note renderer, operator envelopes,
frequency and level scaling, algorithm routing and feedback. This is partial
implementation, not verified hardware equivalence. The requirements below remain
the target contract; hardware/reference qualification, completion of TX81Z/WT11
rendering and full engine integration require further work. Do not restart the existing DX7 implementation.

An initial [pinned MSFA comparison](../doc/yamaha-reference/README.md) now includes
constructed named patches, the reference harness and paired 48 kHz WAV files.
It found whole-file relative RMS differences of 5.20% and 4.13% for algorithms
1 and 32. Qualification is not passed; onset/release timing and the intended
reference target require resolution before treating these as passing fixtures.
An all-32-algorithm sweep subsequently exposed incorrect algorithm 23 routing
in uFor. Its correction reduced held-section relative RMS error from 35.234%
to 0.2432%; the other algorithms were unchanged. The report includes sweep
measurements and the corrected algorithm 23 recordings. Uniform operator
settings and disabled feedback limit what this sweep proves.
An asymmetric all-algorithm sweep now gives each operator a distinct ratio and
level. Held-section relative RMS errors range from 0.213% to 0.665%, with no
new routing outlier. Whole-file errors remain 4.16% to 5.86%; qualification is
still incomplete. The report includes the sweep and worst-held-case audio pair.

Pitch envelopes now decode the packed/unpacked four-rate/four-level fields,
start from level four, hold level three and release toward level four. Integer
Q24 octave state and a fractional tick carry adapt
[MSFA's measured pitch rate/level data](https://github.com/google/music-synthesizer-for-android/blob/master/app/src/main/jni/pitchenv.cc)
to this renderer's per-sample execution. Pitch affects fixed and ratio operators.
Tests cover octave calibration, release, partitioned continuation and invalid
pitch parameters. This is not yet a comparison against a complete reference
renderer's WAV output.

`DX7Renderer.snapshot()` now captures the patch/note definition, six operator
envelopes, pitch envelope, oscillator phases and delayed feedback. Restore
rejects a different patch, note, velocity or sample rate and copies mutable
state so repeated restores do not mutate the saved snapshot. JSON round-trip
and one-second audio regressions cover held and released notes, feedback and
partitioned continuation. These snapshots remain specific to the dedicated
Yamaha renderer, separate from general FM snapshots.

Operator modulation now uses cycles before sine conversion. Feedback uses two
delayed operator outputs and the power-of-two level scaling described by
[MSFA's feedback kernel](https://github.com/google/music-synthesizer-for-android/blob/master/app/src/main/jni/fm_op_kernel.cc).
Both history samples survive snapshots. Analytic one-second WAV regressions
cover feedback off, minimum and maximum levels, with both a feedback carrier
and a feedback modulator. A subsequent pinned MSFA sweep now covers algorithms
1, 2, 16 and 32 at feedback 1/4/7, with asymmetric and maximum-feedback-operator
output profiles. Held-section errors remain below 0.671% in the former profile
but reach 133.016% in the latter. The report preserves raw patches and paired
WAVs for the worst case; this is a failing qualification result whose cause
has not yet been isolated.

The user approved replacing uFor's feedback-operator integer with an explicit
source/destination edge. Algorithm 4 returns OP4 to OP6; algorithm 6 returns OP5
to OP6; other algorithms retain self-feedback. The dedicated note renderer now
uses those endpoints, including two-sample return history in snapshots. Scalar
one-second WAV regressions cover both multi-operator loops at feedback 1 and 7.
The pinned MSFA core explicitly omits those loops and cannot qualify them.
The user approved Dexed Mk I's provisional quarter-strength feedback rule for
algorithms 4, 6 and 32. The note renderer applies that rule, and the offline
adapter now admits algorithms 4 and 6 with feedback. The earlier MSFA algorithm-32
sweep values and recordings above are historical results from before this
strength change. A [constant-gain Mk I comparison](../doc/yamaha-reference/dexed-mki-core.md)
now fixes operator gain and exact phase increments. Unit-gain/feedback-7 relative
RMS errors remain 78.34% for algorithm 4 and 41.02% for algorithm 6; a reference-only
one-bit phase perturbation also demonstrates sensitivity. This does not prove
that quantization explains all discrepancies. A subsequent shared-integer-kernel
control executes all 32 exported topologies with equal and asymmetric operator
settings: all 24,576,000 samples across 512 cases match Mk I exactly. This isolates
the graph/feedback scheduling under fixed gains; it does not independently
qualify the kernel or dynamic envelope behavior. Replacing the floating core
with the provisional integer model awaits approval. A complete reference patch
comparison and hardware qualification remain work.

The renderer rejects uFor's diagnosed algorithm, feedback and operator parameter
errors before preparing DSP state. It also rejects envelope rates/levels above
99, which raw voice inspection preserves. Tests exercise packed and unpacked
voices without changing their stored bytes. This does not yet qualify all
global Yamaha parameters. Offline template integration is described below.

DX7 note admission now validates oscillator sync, LFO speed/delay/depth,
LFO sync/waveform, pitch sensitivity and transpose ranges in both packed and
unpacked inputs. Unpacked oscillator mode must also be 0 or 1. Rejected voices
remain losslessly inspectable. Boundary tests include waveform 5 (sample and
hold), which is implemented in the pinned MSFA reference. Transpose execution
is described below. The note-level phase ownership added below implements
oscillator-sync reset and offline polyphonic sync-off allocation.

A transpose audit of pinned Dexed revision
`fea2f0f0f2c397799f59b27eb11c038cd34b22c5` finds that its
[standard-tuning shift](https://github.com/asb2m10/dexed/blob/fea2f0f0f2c397799f59b27eb11c038cd34b22c5/source/Tuning.cpp)
is the patch transpose minus 24. Its
[note admission](https://github.com/asb2m10/dexed/blob/fea2f0f0f2c397799f59b27eb11c038cd34b22c5/source/PluginProcessor.cpp)
applies that shift before calling the note renderer, whose
[operator preparation](https://github.com/asb2m10/dexed/blob/fea2f0f0f2c397799f59b27eb11c038cd34b22c5/source/msfa/dx7note.cc)
uses the resulting key for level scaling, envelope rate scaling and ratio
frequency. This supports transposed-key intrinsic scaling for the provisional
DX7 model. It does not establish TX81Z hardware behavior.

The user approved transposed-key intrinsic scaling for both Yamaha models and
explicit rejection of DX7 sounding keys outside MIDI 0–127. DX7 now applies
patch transpose to ratio frequency, detune, envelope rates and level scaling;
TX81Z level scaling now uses the same transposed key as its existing frequency
and envelope preparation. Fixed frequencies remain independent of transpose.
Explicit pitch retains its microtuning interval relative to the original trigger
key, and patch transpose is applied in addition. Portable key bindings and
snapshot note identity retain the original trigger key. The offline adapter
admits non-neutral transpose. One-second 48 kHz regressions cover positive and
negative shifts, mixed fixed/ratio operators, key scaling, microtuning, packed
DX7 voices, shared clocks, portable bindings and JSON continuation through
release. This is the approved provisional model, not hardware qualification.

The user approved a caller-owned shared Yamaha clock/LFO context on 2026-11-04.
`DX7LFOContext` now emits normalized waveform/delay rows once per render interval;
the caller passes those same rows to each note's `render(frames, rows)` call.
Notes never advance the shared clock. Explicit `keydown()` delivery applies
LFO sync and restarts delay without resetting sample-and-hold random state.
`DX7LFOSnapshot` retains configuration identity, output-frame position, Q32 phase,
fractional carries, delay progression and random state. Restore checks the clock
definition. Hosts retain that snapshot alongside their existing note snapshots.

The six DX7 LFO waveforms and speed/delay calibration follow the pinned
[MSFA LFO implementation](https://github.com/google/music-synthesizer-for-android/blob/f67d41d313b7dc85f6fb99e79e515cc9d208cfff/app/src/main/jni/lfo.cc).
The context starts from zero phase/random state and advances per output frame,
carrying fractional reference ticks. Delay zero applies full modulation from the
first frame. This is the nominal per-sample model, without MSFA block-held control
artifacts or integer sine-table quantization. The outstanding envelope-reference
target decision remains separate.

DX7 notes now apply their own PMD/PMS calibration to shared LFO rows in octave
units before phase advancement. Effective pitch modulation requires explicit
rows; missing/invalid rows are rejected before note state advances. Zero
sensitivities leave their corresponding modulation depths inaudible.
The offline adapter now owns these clocks per template; native engine ownership
remains work. TX81Z LFO status is recorded below.
One-second 48 kHz WAV regressions cover all waveform shapes, known random values,
rate/delay calibration, sync, fractional continuation, pitch calibration and two
simultaneous notes restoring their clock and note snapshots together.

The user approved Hexter's amplitude calibration provisionally on 2026-11-04.
DX7 notes now apply AMD and each operator's AMS before operator output enters
carrier mixing, downstream modulation or delayed feedback. The shared waveform
column has Hexter polarity: zero adds no attenuation and one applies the full
reduction, gated by the shared delay column. Effective amplitude modulation also
requires explicit shared rows.

Calibration follows Hexter revision `1cf1bfea5962f7c9726e0cf809b762b3b2655225`:
its [AMD fit](https://github.com/smbolton/hexter/blob/1cf1bfea5962f7c9726e0cf809b762b3b2655225/src/dx7_voice_tables.c)
is `exp(AMD * 0.0428993 - 0.285189)`, rounded to six decimal places for AMD 1–98,
with published endpoints 0 and 52.75 for AMD 0 and 99. Its
[AMS reduction ratios](https://github.com/smbolton/hexter/blob/1cf1bfea5962f7c9726e0cf809b762b3b2655225/src/dx7_voice_render.c)
are 0, 0.238058, 0.460510 and 1. These output-level reductions are adapted to
enge's existing logarithmic gain with one step equal to 1/8 octave. This retains
the existing envelope and oscillator model; it does not reproduce Hexter's own
envelope/output lookup tables or controller-combination approximations.
Hexter describes the measured fit as rough, so hardware qualification remains
incomplete. The original pinned MSFA renderer omits amplitude modulation and
cannot qualify this path. WAV regressions cover all four sensitivities, depth
boundaries, zero depth/level/delay, modulator/carrier feedback ordering, mixed
pitch/amplitude modulation and aggregate snapshot continuation.

Packed-bank operator decoding was corrected in uFor `217bf7d`: curve byte 11,
detune/rate-scaling byte 12, sensitivity byte 13, output byte 14,
mode/coarse byte 15 and fine-frequency byte 16 now follow the 17-byte VMEM
layout used by the pinned reference unpacker. A 48 kHz one-second regression
compares equivalent packed and unpacked notes with six distinct operators,
fixed/ratio frequencies, detune, key/velocity scaling, feedback and release.
Both imported byte payloads remain unchanged. The existing malformed-output
regression now targets packed output byte 14. The earlier reference sweeps used
unpacked VCED inputs, so this correction does not resolve their measured errors.

Fixed-frequency operators now use the low two coarse bits and key-independent,
positive-only logarithmic detune, following the frequency equations in
[MSFA's DX7 note implementation](https://github.com/google/music-synthesizer-for-android/blob/master/app/src/main/jni/dx7note.cc).
One-second 48 kHz regressions cover repeated decades, different played keys and
negative/neutral/positive detune. The renderer still uses floating-point
oscillators; these checks do not establish bit-exact hardware equivalence.
Ratio-mode detune now applies the existing Dexed measured-fit coefficients to
log-frequency, as approved on 2026-11-04. The former implementation incorrectly
used the fitted value as a linear frequency multiplier. The
[detune audit](../doc/yamaha-reference/README.md#ratio-mode-detune-audit)
records the discrepancy and the difference from MSFA's constant-cents model.
One-second 48 kHz WAV regressions use independently calculated frequency anchors
at low, middle and high keys, negative/neutral/positive detune, coarse/fine ratio
scaling and JSON snapshot replay. This is the continuous Dexed fit; Q24 pitch
rounding, frequency-table quantization and hardware qualification remain work.

Implement a dedicated Yamaha renderer for the DX7 and TX81Z instead of
projecting their patches into the general FM voice profile. The general graph
remains useful for authored enge patches, but it cannot represent Yamaha's
rate/level envelopes, key scaling, fixed-frequency operators, pitch envelope,
or feedback arithmetic.

## DX7

- Decode each validated VCED or packed bank voice into six Yamaha operators,
  the selected one of 32 algorithms, carrier mix, and delayed feedback path.
- Preserve Yamaha operator numbering, frequency mode, coarse/fine ratio,
  detune, level scaling, velocity sensitivity, and oscillator sync.
- Render the four-rate/four-level amplitude and pitch envelopes in a dedicated
  fixed-point-compatible state model. Compare its generated output against
  known DX7-compatible renderers using named patches and 48 kHz WAV fixtures.
- Keep the lossless SysEx objects as the import boundary. A malformed dump must
  remain inspectable and never become a playable voice.

## TX81Z / WT11

`TX81ZRenderer(voice, additional, note, velocity, sample_rate=48000)` now accepts
explicit VCED/ACED objects and connects nominal frequency, chip detune, output
level, the shared envelope clock, eight waveforms/algorithms and OP4 feedback.
`render`, `release`, `snapshot` and `restore` provide offline note lifetime and
owned JSON continuation. Snapshot identity includes both edit buffers, key,
velocity and output rate. Tests cover all algorithms, mixed waveforms, fixed
and ratio operators, detune, held/released snapshots and mismatched definitions.
The initial supported pitch span after transpose is MIDI 13–108 (chip key codes
0–31); out-of-span notes are rejected. Velocity scaling now applies the measured
KVS curves to positive note-on velocities. Key-level scaling now applies the
measured three-key groups at the transposed key. Zero-delay LFO modulation uses explicit
shared OPZ rows as described below. Velocity has no gain effect
when KVS is zero; its existing zero-velocity metadata behavior remains unchanged
pending the note-on contract decision. This is an incomplete
profile, not completion of the full TX81Z/WT11 requirement. Controller integration,
DAC/integer quantization and hardware/reference
qualification remain work.

The user approved a provisional OPZ LFO context on 2026-11-04.
`TX81ZLFOContext` now prepares the measured VCED speed/PMD/AMD register mappings
and emits signed PM and unsigned AM chip-unit rows. Sample-and-hold uses the
measured alternate speed mapping. The single-mode model advances LFO1 plus the
reset-configured secondary clock, retaining their shared noise history and latch
table. Key-down sync resets the primary counter without rewinding noise or the
chip clock. JSON snapshots retain all of that state and reject a different
clock definition.

Register maps follow [androuet's hardware register measurements](https://nornand.hatenablog.com/entry/2020/11/25/171659).
Clock, waveform, noise and modulation arithmetic follow pinned
[ymfm OPZ](https://github.com/aaronsgiles/ymfm/blob/81aec25ccbb98f4873a255f7551ac4dadac59b4a/src/ymfm_opz.cpp).
Note rendering consumes the same shared rows for simultaneous voices, applies
PMS before ratio frequency multiplication while leaving fine-detune offsets
outside it, and bypasses fixed-frequency operators. AMS contributes envelope
attenuation only to operators with AME enabled. Effective modulation requires
explicit rows; malformed rows are rejected before voice state advances.

The [pinned LFO comparison](../doc/yamaha-reference/README.md) has exact agreement
for all 48,000 PM/AM rows of each of four waveforms. Its compressed reference
rows are used by one-second WAV regressions alongside fractional-clock/noise
continuation, fixed-frequency bypass, detune, AME/AMS gating and two-note
snapshot replay. This qualifies the component against that software model,
not hardware or the complete TX81Z renderer. Delayed pitch modulation now uses
the provisional measured timing below; active delayed AMD remains rejected.
Multitimbral LFO routing, software
LFOs and full engine integration also remain work.

The [delay measurements](https://nornand.hatenablog.com/entry/2020/11/30/185005)
specify completion times in units of 1/48 second, with observed variation of one
unit. Both depths remain zero until half that time; PMD then rises linearly,
whereas AMD follows a curve that the report does not quantify. Delay 1 completes
at 15/48 second and delay 99 at 1022/48 seconds. Implementing a linear AMD ramp
would therefore invent behavior inconsistent with the stated observation.
The user approved the provisional pitch-only implementation. It uses the measured
completion table and floors the linear depth to integer register values at output
frame time before OPZ ticks. Frames without a chip tick retain the previous chip
output. Key-down restarts the delay even with sync off, while preserving the
noise and secondary clocks. Snapshot identity includes authored delay; elapsed
delay time survives partitioned JSON continuation. Rounding and update cadence
are explicit model choices, not qualified firmware behavior. Regression WAVs
cover the silent half, ramp, final depth, key-down restart, fractional chip ticks
and continuation, including maximum delay. Offline integration covers shared
clock restart by overlapping notes. Active delayed AMD remains rejected; zero
depth, zero AMS or disabled operator AME permits the pitch-only path.
Delayed AMD remains a separate unresolved requirement.

VCED globals 54–76 and ACED globals 20–22 now reject values above the manual's
documented parameter ranges before note preparation. Unused payload bytes and
voice-name bytes remain losslessly owned by the imported edit buffers. Boundary
tests cover LFO controls, transpose, performance controls and ACED foot controls.
Range validation does not imply those external controller paths are implemented.

ACED reverb rate now prepares a capped effective envelope rate and selects an
additional post-release stage once raw attenuation reaches 192 (-18 dB in the
model). Rate zero remains off; a nonzero reverb rate cannot accelerate release.
The implementation follows ymfm's OPZ rate preparation and envelope transition.
WAV regressions cover the exact threshold/increment schedule, a slow release,
repeated key release and JSON continuation. The note-renderer integration also
covers reverb tails. This parameter controls envelope decay; hardware
qualification of that behavior remains incomplete.

`tx81z_detune_step` now maps the seven VCED detune settings (neutral 3) and a
five-bit chip key code to the signed OPZ phase-step offset. It follows
[ymfm's key-dependent detune table](https://github.com/aaronsgiles/ymfm/blob/main/src/ymfm_fm.ipp)
and the [OPZ ordering before the frequency multiplier](https://github.com/aaronsgiles/ymfm/blob/main/src/ymfm_opz.cpp).
Tests cover sign, neutral, low-key zero offsets and high-key limits, and render
the resulting offset through the operator bank at 48 kHz. This primitive takes
chip key codes explicitly; the note renderer now supplies MIDI-to-chip key
mapping and applies detune before ratio multiplication. Multiplier/phase
quantization and hardware qualification remain work.
Fixed-frequency mode bypasses this detune path in the reference model.

`tx81z_output_gain` now converts the VCED 0–99 output control through
[DXconvert's total-level curve](https://github.com/richotero/dxconvert/blob/master/DXconvert/dxcommon.py)
and its [reported TX81Z carrier-headroom measurements](https://github.com/richotero/dxconvert/blob/master/DXconvert/fb01.py).
Carrier offsets are 8, 13 and 16 TL steps for two-, three- and four-carrier
algorithms; modulators retain the unadjusted level curve. Saturation is at TL
127. This nominal floating gain does not include integer DAC/sine quantization
or envelope attenuation. Regressions cover curve boundaries
and headroom, plus a one-second paired VCED/ACED integration through nominal
frequency, shared envelope clock, gain conversion, operator mixing and release.
These are model-level checks, not new hardware measurements.

The same gain preparer now accepts a decoded operator, MIDI key and velocity
and sums key-level/velocity attenuation before the TL clamp. The discrete tables
come from [androuet's TX81Z register analysis](https://nornand.hatenablog.com/entry/2020/11/21/201911),
including three-key level-scaling groups, the documented high-key wrap and
all seven nonzero KVS curves. KVS zero adds no attenuation. Velocity zero with
nonzero KVS is rejected because the measurement table covers only 1–127.
Focused WAV regressions cover boundaries and combined saturation. The note
renderer now connects both terms for the unambiguous inputs above. Analytic
one-second WAV regressions cover velocity curves, transposed-key scaling,
their sum and TL saturation. All eight algorithm continuation tests now use
distinct nonzero key/velocity sensitivities. The velocity-zero note-creation
contract still awaits a decision. These model-level
checks do not establish hardware qualification.

The numeric `tx81z_operator_samples` kernel now renders all eight four-operator
graphs in visible OP1–OP4 order, with OP4 two-sample feedback, independent
waveform selections and caller-prepared per-frame frequency/gain rows. Its
`TX81ZState` retains phases and feedback through JSON continuation without
mutating the caller's saved state. Routing/modulation scaling follows
[ymfm's four-operator output path](https://github.com/aaronsgiles/ymfm/blob/main/src/ymfm_fm.ipp).
Analytic 48 kHz WAV regressions cover all graphs, asymmetric operators,
feedback, waveform selection and partitioned continuation. This is raw carrier
mixing. The note renderer now connects chip detune, nominal VCED gain conversion
and the envelope clock, plus key/velocity scaling in the supported profile.
DAC saturation and the full voice profile remain work.

The user approved ymfm's waveform model provisionally on 2026-11-03, including
the inferred signed sine-squared shape. `tx81z_wave_samples` now realizes all
eight continuous shapes using zero-based ACED waveform numbers. The model
follows [ymfm's OPZ waveform construction](https://github.com/aaronsgiles/ymfm/blob/main/src/ymfm_opz.cpp);
it does not reproduce logarithmic table quantization or establish measured
hardware equivalence. One-second 48 kHz regressions distinguish all eight
shapes at sixteen positions per cycle, including negative/unwrapped phases
and the silent half-cycle boundaries. The complete TX81Z renderer remains work.

`TX81ZEnvelope` now prepares matching VCED/ACED operator data into a separate
OPZ attenuation state. Its `clock(counter)` consumes the host's shared envelope
counter, not output-frame time. Rates include key scaling, attack/decay/sustain/
release transitions, saturated attenuation and envelope shift (disabled for
OP1). D1L is inverted into chip attenuation, so level 15 means no initial decay,
as described by the Yamaha manual. Increment sequencing follows
[ymfm's envelope clock](https://github.com/aaronsgiles/ymfm/blob/main/src/ymfm_fm.ipp)
and [OPZ rate preparation](https://github.com/aaronsgiles/ymfm/blob/main/src/ymfm_opz.cpp).
One-second WAV regressions cover full-level sustain, release and every shift,
including JSON state continuation. A pinned ymfm envelope comparison now adds
eight profiles spanning zero/immediate/slower attack, decay, key-rate scaling,
sustain extremes, release and reverb. All 1,536,000 shifted attenuation values
match exactly across four shifts and 48,000 frames per profile. The reference
harness and compressed oracle are retained in
[the comparison report](../doc/yamaha-reference/README.md#tx81z-envelope-comparison).
This qualifies those profiles against the provisional software model, including
fractional-clock and JSON continuation; exhaustive rates, hardware timing,
output-level scaling and complete voice integration remain unqualified.

`TX81ZEnvelopeBank` now places four envelopes on a shared integer counter and
converts chip-clock intervals to output frames with an integer remainder.
The nominal clock is 3,579,545 Hz; clock and output rates remain explicit state.
The envelope divider is 2 * 32 * 3, following ymfm's OPZ engine constants.
Rows observe attenuation before advancing the next output-frame interval.
Counter/remainder values can be retained by a host across voice/block boundaries.
One-second 48 kHz regressions verify exact and fractional clock ratios, all four
shifts and partitioned JSON continuation. This establishes clock conversion,
not measured hardware phase alignment or the complete patch renderer.

`tx81z_base_frequency` now prepares nominal frequency from matching VCED/ACED
operators. Ratio mode orders the 64 multiplier/coarse-detune combinations and
adds fine steps before coarse detune. Half-multiple entries reject unavailable
fine values above seven. Fixed mode uses the VCED coarse index shifted by two,
plus fine and the ACED range, reproducing the manual's 8–32,640 Hz span and its
range-dependent steps. The equations are cross-checked against ymfm's OPZ
register model and [DXconvert's `fix_4op`](https://github.com/richotero/dxconvert/blob/master/DXconvert/fourop.py);
the [Yamaha manual](https://www.polynominal.com/site/studio/gear/sold/yamaha_tx81z/yamaha-tx81z-manual.pdf)
and [TX81Z Programmer ratio displays](https://github.com/mgregory22/tx81z-programmer/blob/master/src/freqratios.c)
corroborate the control mapping. Audio regressions cover ratio,
fine, fixed-frequency boundaries and different played keys. The note renderer
now applies the chip detune primitive. Phase quantization and full voice-profile
qualification remain work.

- Pair VCED and ACED edit-buffer messages explicitly before conversion.
- Decode its four-operator, eight-algorithm routing, frequency modes, feedback,
  and ACED waveform selections into the same Yamaha renderer family.
- Preserve the TX81Z waveform definitions rather than mapping them to enge's
  generic sine, square, and triangle waveforms.

A pinned waveform-table comparison now measures all eight continuous shapes
against ymfm at 440 Hz in paired one-second, 48 kHz WAVs. Relative RMS differences
range from 0.338% to 0.803%; table quantization and nonzero zero-substitution remain
explicit approximation differences. See the [comparison report](../doc/yamaha-reference/README.md#tx81z-waveform-table-comparison).
This does not qualify hardware or complete the full-renderer requirement.

## Delivery boundaries

- The compatibility renderer is offline first. Do not claim callback-safe live
  Yamaha emulation until its fixed state and allocation bounds are measured.
- The general FM graph and Yamaha profiles have separate schema tags, prepared
  state, snapshots, and regressions. Neither is an alias for the other.
- Add a small set of public imported-patch fixtures only when their licensing
  permits redistribution. Tests may construct parameter bytes directly.

## DX7 oscillator phase ownership

The user approved explicit caller-owned phase initialization on 2026-11-04.
`DX7Renderer` now accepts six `initial_phases` in cycles, ordered OP1 through
OP6. Sync-on resets them to zero; sync-off copies the supplied phases. Omitting
phases starts a standalone note at zero. Invalid lengths, nonfinite values and
values outside `[0, 1)` are rejected. Caller mutation cannot affect a live note.
Existing JSON note snapshots preserve running phases and the last per-frame
phase increments across continuation.
One-second WAV regressions cover both sync states, input ownership and restored
continuation. The approved offline adapter owns reusable slots per template for sync-off
patches. Overlapping notes receive distinct slots. Retired slots advance at their
last effective frequencies, including the final pitch-envelope/LFO contribution,
and a new note takes the oldest idle slot or creates a zero-phase slot when all
are active. A retirement-frame origin makes idle continuation independent of
render block size. Stop, fade and envelope completion all return a slot; snapshots
retain idle slots alongside active voices. Regressions cover overlapping notes,
stop/fade retirement and restoration with idle and active slots. This policy has
no new voice cap and is not hardware voice-allocation or phase qualification.

## Offline instrument integration

Dedicated uFor `YamahaDX7Voice` and `YamahaTX81ZVoice` templates now preserve
raw patch bytes and carry trigger velocity through shared lifecycle preparation.
`enge.yamaha_instrument` renders those actions with one shared LFO context per
template, routing, gain/filter processing, minimum hold, release, stop
and continuous fade retiming. MIDI-file rendering dispatches Yamaha instruments
to this NumPy-only adapter. Snapshots include clocks, notes, filter state and
pending retirement state.

Instrument snapshot restore now prepares both notes and shared clocks before
replacing live state. A rejected later clock definition cannot rewind an earlier
template's clock. Regression coverage reproduces this with both DX7/TX81Z template
orders, checks unchanged aggregate state after rejection, and compares the next
one-second audio continuation against an untouched snapshot reference.

Explicit trigger pitch microtunes ratio operators while the transposed MIDI key continues
to drive key scaling. Fixed operators retain their authored frequencies. Velocity
uses nearest-integer rounding with ties upward, preserves exact zero and maps
any positive value to at least 1. Existing TX81Z rejection of velocity zero with
nonzero velocity sensitivity remains in force.

The adapter admits transpose for both models and TX81Z delayed pitch modulation.
Active delayed TX81Z amplitude modulation remains rejected.
Dynamic tuning routes, external envelopes and unsupported portable processing
remain rejected. Hardware-specific assignment fields are range-validated and
preserved, but are not connected to controller events. They are not all rejected
at admission. Released DX7
carriers with nonzero final
envelope levels remain active until explicitly stopped or faded. This integration
adds no native callback renderer and does not establish hardware equivalence.

Shared `ControlRenderer` integration now supplies instrument-, part- and
trigger-scoped controls to outer amplitude and filter cutoff/Q targets. Authored
smoothing, route mappings/composition and both filter/amplitude orders reuse the
existing synth machinery. Key and unquantized trigger velocity can also drive
onset-latched routes, independently of the patch's intrinsic Yamaha key/velocity
scaling. Snapshots retain control contexts and each voice's source ownership
alongside filter and note state. Regressions cover overlapping parts, interrupted
ramps, restoration during smoothing and both Yamaha models. This is portable
processing control integration, not emulation of hardware-specific breath,
mod-wheel or other assignment matrices.

### Remaining integration gates

- TX81Z VCED 63–76 retains poly/mono, pitch-bend range, portamento, foot volume,
  sustain, chorus and modulation-wheel/breath assignments. ACED 21–22 retains
  foot-controller pitch/amplitude assignments. Current rendering does not execute
  those patch-defined behaviors. Portable trigger/release/control preparation is
  not evidence that this hardware assignment layer is implemented.
- Portable dynamic tuning is explicitly rejected in `prepare`. Reusing scoped
  control values requires a note-renderer tuning input and a defined fixed/ratio
  operator policy. The proposed ratio-only per-render rows await approval.
- TX81Z velocity-zero note creation with nonzero KVS still needs a contract;
  the measured attenuation table covers positive velocities only.
- DX7 integer oscillator/feedback replacement remains awaiting approval.
  The existing fixed-gain graph comparison does not qualify dynamic envelopes,
  complete patches or hardware output.
- TX81Z delayed AMD, software/multitimbral LFO routing and complete-renderer
  reference qualification remain incomplete. No passing component test closes
  those requirements.

## Additional work beyond the prompt

None.
