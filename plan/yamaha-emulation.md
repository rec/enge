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
to this renderer's per-sample execution. Pitch-envelope modulation affects ratio
operators only; the controller reference audit corrected the earlier fixed-mode
behavior. Hardware pitch bend still affects both operator modes.
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
qualify the kernel or dynamic envelope behavior. The user subsequently approved
replacing the floating core with the provisional integer model. The renderer
now uses Q24 phase/increments and signed Q24 feedback, a 4096-position logarithmic
sine, and 1/1024-octave attenuation. Independent mathematical tables reproduce
the pinned Mk I table precision. The reference's near-silence threshold is
applied per frame; the existing per-frame envelopes and provisional controller
calibrations remain, rather than adopting Dexed's block gain ramps.
Initial phases remain caller-supplied cycles and truncate onto the Q24 grid;
frequency increments round to the nearest Q24 unit and wrap modulo one cycle.
Note and idle-slot snapshots store integers consistently, and idle slots advance
using the retained quantized increments. The old floating core is removed.
A complete reference patch comparison and hardware qualification remain work.

[Controlled dynamic comparisons](../doc/yamaha-reference/dexed-mki-core.md#isolating-reference-envelope-behavior)
now isolate two reference behaviors before any further runtime redesign.
Dexed's inner operators in the special algorithm-4/6 feedback kernels jump to
new block attenuation rather than using the ordinary interpolation rule.
Changing only that behavior alters reference audio by 10.00–56.99% relative RMS
in the tested profiles but does not resolve enge's mismatch. Separately,
correcting only the reference's inverse hold-duration scaling reduces the
unmodulated hold-profile difference from 43.08% to 5.77%. All nine negative
controls across the two experiments remain bit-identical. Preserve enge's
approved stable-duration holds; remaining attack/release trajectories, block
interpolation and silence admission require isolation before treating the
full-note comparisons as passed. The retained patches are experiments, not
production changes or verified hardware corrections.

The subsequent
[matched-trajectory comparison](../doc/yamaha-reference/dexed-mki-core.md#matched-per-frame-attenuation-trajectories)
feeds captured enge envelope attenuation to upstream Mk I integer kernels with
the same per-frame silence policy. All 432,000 samples across nine fast, held
and asymmetric-envelope cases match exactly, including algorithms 4 and 6 at
feedback seven. One-unit attenuation perturbations change every case. Retained
fixtures and production regressions cover exact audio and JSON replay through
release. This qualifies the tested dynamic-core arithmetic under controlled
inputs; it does not independently qualify the captured envelope trajectories or
resolve the failing whole-note comparisons.

A subsequent [dynamic-envelope comparison](../doc/yamaha-reference/dexed-mki-core.md#dynamic-envelope-comparison)
combines upstream Dexed envelopes with the Mk I core. It confirms that
constant-gain agreement does not close attack/release or equal-level timing.
The approved provisional equal-level holds are now implemented using the
measured 44.1 kHz counts as durations, preserving seconds across sample rates
instead of inheriting Dexed's inverse rate conversion. Remaining frame counts
survive snapshots and release interrupts a hold. Zero-level attacks use the
reference's faster hold rule. Focused regressions cover table/high-rate
boundaries, prepared rate scaling, multiple sample rates and JSON continuation.
The moving stages now convert Dexed's 64-frame increment to a per-frame
increment before sample-rate scaling. Previously they ran approximately 64
times too fast. A pinned `Env` release trace verifies the rate-50 slope;
regressions cover 44.1, 48 and 96 kHz while retaining the approved hold durations.
Full-note reference errors remain explicit in the report; hardware timing and
the remaining envelope/block-model differences are not qualified.

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
negative/neutral/positive detune. Frequencies now feed the approved integer
oscillator core; these checks do not establish bit-exact hardware equivalence.
Ratio-mode detune now applies the existing Dexed measured-fit coefficients to
log-frequency, as approved on 2026-11-04. The former implementation incorrectly
used the fitted value as a linear frequency multiplier. The
[detune audit](../doc/yamaha-reference/README.md#ratio-mode-detune-audit)
records the discrepancy and the difference from MSFA's constant-cents model.
One-second 48 kHz WAV regressions use independently calculated frequency anchors
at low, middle and high keys, negative/neutral/positive detune, coarse/fine ratio
scaling and JSON snapshot replay. The continuous Dexed fit now rounds at the Q24
phase-increment boundary; frequency-table quantization and hardware qualification
remain work.

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
explicit VCED/ACED objects and connects native ratio/nominal fixed frequency,
chip detune, output level, the shared envelope clock, eight waveforms/algorithms
and OP4 feedback.
`render`, `release`, `snapshot` and `restore` provide offline note lifetime and
owned JSON continuation. Snapshot identity includes both edit buffers, key,
velocity and output rate. Tests cover all algorithms, mixed waveforms, fixed
and ratio operators, detune, held/released snapshots and mismatched definitions.
After transpose, keys below 13 or above 108 now wrap by octaves into MIDI
13–108 (chip key codes 0–31), following the
[published register analysis](https://nornand.hatenablog.com/entry/2021/07/30/183520).
This replaces the initial out-of-span rejection. Pitch, chip detune, envelope
rate scaling and level scaling use the wrapped key; snapshot identity and
portable key bindings retain the original trigger key. Explicit pitch remains
a ratio relative to the original key's equal-tempered frequency, applied to
ratio operators only. Regression renders cover both boundaries, extreme
transpose, microtuning, mixed fixed/ratio operators and snapshot continuation.
Portamento stores nominal sounding Hz after transpose and wrapping, so a new or
legato note starts from the preceding sounding fundamental rather than applying
the destination's octave shift to the previous trigger pitch. The direct
renderer's `start_pitch_hz` argument uses these sounding-Hz coordinates.
Per-part history, interrupted glides, retired voices and snapshots use the same
coordinates. Regression renders cover both wrap boundaries, positive/negative
transpose, microtuned legato notes and JSON replay. This is the provisional
continuous-between-endpoints model; firmware glide and bend behavior across
wrap boundaries remains unqualified.
Velocity scaling now applies the measured
KVS curves to positive note-on velocities. Key-level scaling now applies the
measured three-key groups at the transposed key. Zero-delay LFO modulation uses explicit
shared OPZ rows as described below. Velocity has no gain effect
when KVS is zero; explicit zero-velocity triggers use the provisional attenuation
model described below when KVS is nonzero. This is an incomplete
profile, not completion of the full TX81Z/WT11 requirement. Controller integration,
Whole-patch reference and hardware qualification remain work; the provisional
integer core and explicit shared DAC path are covered below.

A [fixed-gain OPZ core comparison](../doc/yamaha-reference/opz-core.md) now
covers all eight algorithms and feedback levels with uniform and asymmetric
profiles: 128 one-second cases, or 6,144,000 samples. It bypasses patch decoding,
envelope movement, LFO and DAC processing. Unmodulated carriers differ by about
0.34%, while the worst modulated case reaches 141.1251% relative RMS. These are
failing qualification results. The report retains the pinned-source harness,
complete metrics and paired WAVs. A subsequent shared-integer-kernel control
executes the exported enge graphs with upstream operator kernels and independent
feedback history. All 6,144,000 samples match exactly; a deliberately removed
edge produces 46,406 mismatches. This supports graph and feedback scheduling
for the tested fixed-gain profiles. The production core now uses the provisional
integer OPZ model and matches all 128 captured reference cases exactly,
including partitioned JSON continuation. This closes that fixed-gain comparison,
not dynamic-patch, native-clock or hardware qualification.

A subsequent [dynamic integration comparison](../doc/yamaha-reference/opz-core.md#dynamic-envelope-integration)
combines the integer channel path with attack, decay and release at the existing
rational envelope clock. It found a missing raw-envelope quiet gate: pinned
ymfm silences raw attenuation above 896 before applying operator shift. The bank
now returns raw attenuation; the note renderer applies that provisional gate
and then shift. All 1,536,000 samples across 32 cases match, including JSON
continuation. The fixture leaves both endpoint operators unshifted because
ymfm's register-slot exclusion conflicts with the manual's OP1 restriction
under the channel mapping. Enge retains its manual-based OP1 rule; complete
shift mapping, controller-modulated patches and hardware remain unqualified.

The [DAC audit](../doc/yamaha-reference/opz-dac.md) verifies the pinned helper
over all 65,536 signed 16-bit inputs and records its positive-overflow difference
from explicit encode/decode. The chip renderer uses the optimized helper after
summing hardware channels. Per-note conversion or conversion after the current
portable per-voice processing would be a different signal path. Implementing
this provisional DAC model requires explicit shared output-group ownership and
post-sum processing, not an isolated converter inside each note. uFor now
declares those groups, and `OfflineYamaha` sums their raw voices, converts once,
then applies group gain, filters and routing. Group filter tails and JSON
continuation survive voice retirement. Exhaustive DAC-code tests and 48 kHz WAV
regressions cover overlapping voices, separate groups, clipping, host processing
order and fades. Raw ungrouped note output remains available. This closes the
explicit provisional DAC boundary; native-clock and physical DAC behavior,
eight-note chip allocation and multitimbral LFO integration remain unqualified.

Renderer admission checks the raw per-operator VCED amplitude-modulation-enable
and ACED fixed-frequency flags before using their decoded booleans. Values above
one are rejected rather than silently treated as true. Regressions cover all
four operator positions in both buffers and preserve the original dump bytes.

The user approved a provisional OPZ LFO context on 2026-11-04.
`TX81ZLFOContext` prepares the measured VCED speed register mapping and emits
four integer columns: raw signed PM, raw unsigned AM, delay-progress numerator
and delay-progress denominator. Notes apply the measured PMD/AMD register maps
and depths. This replaces the earlier two-column, depth-scaled row API; old
rows are rejected. Sample-and-hold uses the
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
for all 48,000 PM/AM rows of each of four waveforms after applying the reference
depth of 127 to the raw rows. Its compressed reference rows are unchanged and
used by one-second WAV regressions alongside fractional-clock/noise
continuation, fixed-frequency bypass, detune, AME/AMS gating and two-note
snapshot replay. This qualifies the component against that software model,
not hardware or the complete TX81Z renderer. Delayed pitch modulation now uses
the provisional measured timing below; delayed AMD now uses the provisional
authored-depth ramp described below.
Software-LFO firmware qualification and the remaining full performance integration
still need work.

The [owner's manual, printed page 40](https://usa.yamaha.com/files/download/other_assets/9/316769/TX81ZE.pdf)
defines the performance-mode ownership boundary: the first two instruments
whose Max Notes allocation is nonzero supply the shared LFO choices. This is
allocation order, not the order of currently sounding notes. `vib` uses the
instrument's own voice settings for pitch modulation only; `off` disables both
pitch and amplitude modulation. Integration uses explicit LFO owners and
selectors derived from those allocations. Existing DAC output groups do not
provide that information and must not silently infer LFO ownership from their
template list or current polyphony.

The format prerequisite is implemented in uFor `85bce42`: lossless PCED bulk
inspection retains all 110 parameter bytes and decodes the eight instrument
allocations, including maximum notes and LFO selectors. Invalid parameters
remain inspectable with diagnostics, and performance messages are distinguished
from ACED operator data. The decoder is included in enge's uFor pin `a32cfe3`,
which also adds explicit `TX81ZLFOGroup` owner-to-template bindings.
Software-vibrato qualification and remaining performance settings still need work;
the decoder alone does not qualify or execute those behaviors.

The existing `TX81ZLFOContext` now accepts a second voice definition and clocks
both hardware LFOs with one shared noise generator/table. Its paired output is
`(frames, 2, 4)`; single-mode output remains `(frames, 4)`. Both rates, waveform
latches, delays and sync flags survive JSON snapshots. A keydown resets each
enabled sync phase and both configured delays. The two-phase reset follows
the pinned ymfm model; restarting both performance delay ramps is provisional
until firmware or hardware evidence establishes that behavior. All 16 waveform
pairings match the pinned reference over one second at 48 kHz, including the
shared sample-and-hold table. This implements the shared hardware-clock
primitive, not software vibrato or complete performance
rendering. See the [paired comparison](../doc/yamaha-reference/README.md#tx81z-paired-hardware-lfo-comparison).

`OfflineYamaha` now consumes the explicit `tx81z_lfos` owner/selector groups.
One group clocks its first and optional second owners once per render interval,
even when neither owner has sounding notes. Recipients inherit selected owner
base PMD/AMD while retaining their own sensitivity, AM-enable bits and scoped
controller assignments. Keeping scoped assignments independent is a host
extension, not qualified firmware controller ownership. Prepared actions remain
validated against the original authored templates; derived render patches and
group clocks restore through the existing JSON snapshots. Off routes suppress
both PM and AM. Software vibrato routes use the provisional model below.
One-second WAV regressions cover one/two owners, silent owners, overlapping
notes, independent owner rates/delays/sync, off routes and JSON continuation.
This does not impose a hardware voice allocator or implement other PCED fields.

The `vibrato` selector now has a provisional independent pitch-only path. Each
selected template owns a separate `TX81ZLFOContext` using its own patch settings;
notes of that template share it, and only its own onsets restart its delay/sync.
Shared hardware clocks still receive every group onset, so an owner can select
vibrato for itself while supplying a hardware LFO to other members. AM waveform
rows are zeroed, suppressing both patch and controller AM without suppressing
independent EG-bias or volume controls. Separate `vibrato_clocks` are required in
Yamaha JSON snapshots; restoration validates their identities before committing
any state. One-second WAV coverage combines two vibrato owners and a hardware
recipient, independent keydowns, additive controllers and partitioned replay.

The manual establishes own-patch pitch-only behavior, but does not specify the
software oscillator update cadence, phase quantization or noise algorithm.
Reusing OPZ rate/waveform/noise machinery is an explicit provisional proxy,
not firmware or hardware equivalence. The software-LFO qualification gate stays
open until those details can be compared against an independent reference.

Firmware-derived depth integration:
[ax81z Math.md at 5848832](https://github.com/iflyhigh/ax81z/blob/5848832989c7864092a277e9428d44f817c6173f/Math.md)
describes firmware-derived additive wheel/breath/foot contributions, saturated
before PMD/AMD register conversion, with delay applied only to the patch depth.
Its exponential conversion and LFO table reproduce all 100 existing measured
rate/PMD/AMD rows exactly with zero controllers and completed delay. This is
stronger evidence than the provisional strongest-depth model, which is now
replaced by additive byte-depth contributions and controller regressions. The
source is a partial reverse engineering, not an independent hardware trace;
its controller arithmetic still needs that qualification. It does not document
the software vibrato or breath pitch-bias implementation.

The [delay measurements](https://nornand.hatenablog.com/entry/2020/11/30/185005)
specify completion times in units of 1/48 second, with observed variation of one
unit. Both depths remain zero until half that time; PMD then rises linearly,
whereas AMD follows a curve that the report does not quantify. Delay 1 completes
at 15/48 second and delay 99 at 1022/48 seconds. Implementing a linear AMD ramp
would therefore invent behavior inconsistent with the stated observation.
The delay clock uses the measured completion table.
The clock latches exact rational delay progress at each OPZ observation; each
note applies the byte-depth conversion below before multiplying the raw waveform.
Frames without a chip tick retain the previous observation.
Progress and byte conversion use integer arithmetic without float rounding.
Key-down restarts the delay even with sync off, while preserving noise state
and unsynchronized phases. Snapshot identity includes authored delay; elapsed
delay time survives partitioned JSON continuation. Rounding and update cadence
are explicit model choices, not qualified firmware behavior. Regression WAVs
cover the silent half, ramp, final depth, key-down restart, fractional chip ticks
and continuation, including maximum delay. Offline integration covers shared
clock restart by overlapping notes.

The firmware-derived byte-depth model now replaces the provisional authored
AMD interpolation and linear PMD register interpolation. Rational delay progress
is floored to a byte in 0..255, multiplied by `authored_depth * 165 >> 6`,
then shifted right by eight. Undelayed controller byte contributions are added
and saturated to 255. PMD is half that value, truncated; AMD uses the full
firmware lookup curve indexed by 255 minus that value, then halves it.
This retains all measured static endpoints and gives nonlinear AMD progression.
The conversion is firmware-derived; the measured completion times and the
linear progression of the delay byte remain provisional timing choices.
One-second WAV regressions cover quarter progress, large rational denominators,
additive controllers, saturation, delay bypass, shared noise and snapshots.
Exact delayed AMD firmware timing remains unqualified.

Polyphonic full-time portamento now uses the
[published register-step model](https://nornand.hatenablog.com/entry/2020/12/14/185801).
`enge/yamaha_portamento.py` retains the signed quarter-step distance, asymmetric
descending correction and 100-entry time coefficient table. The implementation
uses exactly 96 updates per second, with the first update one tick after onset;
the source reports approximately that cadence. Fractional sample-clock carry
keeps partitioned renders and non-48-kHz rates consistent. Time zero bypasses
glide. An explicit `start_pitch_hz` gives standalone notes a sounding source
pitch after transpose and octave wrapping.
Ratio operators receive glide before additive detune; fixed operators remain
fixed. Dynamic tuning multiplies the same ratio. Start and target microtuning
remain exact, while intermediate pitch steps use the firmware model's grid
relative to the target. Envelopes and key scaling use the target key throughout.

The offline adapter keeps the most recently started pitch separately for each
template and part. A new note inherits that voice's current glide pitch if it
is still active, or the pitch retained at its retirement. A part's first note
starts at its own pitch. Retiring an older overlapping note does not overwrite
the latest history. This is the recommended provisional polyphonic ownership
policy; it does not claim hardware voice-slot allocation equivalence. Snapshots
retain per-note glide state and the instrument's pitch history. One-second WAV
regressions cover measured ascending/descending traces, slow settings, target
clamping, fixed-frequency bypass, detune, dynamic tuning, interrupted glides,
part isolation, retirement history and snapshot replay.

The synth, FM, noise and Yamaha renderers now accept physical `trigger_release`
observations without retiring sustained audio. Persistent native admission
discards these observations before encoding DSP actions. Shared synth traces
expose physical key release independently of logical
retirement under sustain. One-second WAV regressions cover every supported
backend, partitioned rendering and snapshot replay.

TX81Z mono playback now owns one sounding voice per template and part, following
the [manual's Single-mode rules](https://usa.yamaha.com/files/download/other_assets/9/316769/TX81ZE.pdf)
(printed page 21). Overlapping onsets select the newest key without restarting
oscillator phases, feedback, envelope attenuation/stages or the envelope counter.
Physical release returns to the newest remaining held key, even under sustain;
logical retirement releases the final sounding key. Eight previous keys are
remembered in addition to the current key. The oldest is forgotten on overflow.
Performance-mode five-key memory is outside the current Single-mode adapter.
Each selected key restores its scoped controls. Snapshots include held starts,
their source contexts and the selected note's continuing DSP state.

Legato updates intrinsic key-scaled rates, detune, output levels and velocity for
the selected key while preserving envelope progress. Shared LFO and filter state
also continue; a separated onset restarts the note and LFO and replaces any
sustained/releasing mono voice. These register-update and ownership details are
provisional, not firmware-qualified. Explicit stop, fade or envelope-release
actions remove that key from priority memory without reviving another key.
Non-onset templates retrigger rather than enter held-key memory. Fingered mode
glides only across overlapping held keys, including return to a previous key,
and ignores the full-time portamento switch. Poly mode always uses full-time
glide regardless of the stored fingered flag. Regressions cover analytic phase
continuity, developing envelopes, sustain, memory overflow, separate parts,
retirement, switch-off fingered glide and JSON snapshot replay in 48-kHz WAVs.

The TX81Z-specific `yamaha.portamento` target now controls full-time glide through
the existing scoped-control path. Its normalized 0..1 domain defaults to 1,
matching the manual's initial enabled state. Values below 0.5 switch glide off;
values at or above 0.5 enable subsequent glides. The recommended provisional
mid-glide rule snaps to the target on the first off frame and does not resume
that ended glide when the switch returns on. The retained pitch history then
contains the reached target. Direct note renderers accept equivalent
`portamento_enabled` rows. Invalid rows are rejected before state changes.
MIDI CC65 uses the existing controller binding and switches between values 63
and 64; smoothing is applied before the threshold. One-second WAV regressions
exercise actual converted CC65 messages, independent parts, threshold crossing,
new notes after re-enabling, interrupted smoothing and snapshot replay. This
does not qualify the hardware's exact mid-glide switch response. Mono fingered
mode ignores this switch as specified by the manual.

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
mapping and applies detune before ratio multiplication. Native chip frequency
generation and hardware qualification remain work; phase increments now round
to the integer core's Q20 grid.
Fixed-frequency mode bypasses this detune path in the reference model.

`tx81z_output_gain` now converts the VCED 0–99 output control through
[DXconvert's total-level curve](https://github.com/richotero/dxconvert/blob/master/DXconvert/dxcommon.py)
and its [reported TX81Z carrier-headroom measurements](https://github.com/richotero/dxconvert/blob/master/DXconvert/fb01.py).
Carrier offsets are 8, 13 and 16 TL steps for two-, three- and four-carrier
algorithms; modulators retain the unadjusted level curve. Saturation is at TL
127. This preparer's nominal gain excludes envelope attenuation; the operator
core subsequently quantizes combined attenuation and waveform output. Explicit
output groups now apply the provisional shared DAC model. Regressions cover curve boundaries
and headroom, plus a one-second paired VCED/ACED integration through nominal
frequency, shared envelope clock, gain conversion, operator mixing and release.
These are model-level checks, not new hardware measurements.

The same gain preparer now accepts a decoded operator, MIDI key and velocity
and sums key-level/velocity attenuation before the TL clamp. The discrete tables
come from [androuet's TX81Z register analysis](https://nornand.hatenablog.com/entry/2020/11/21/201911),
including three-key level-scaling groups, the documented high-key wrap and
all seven nonzero KVS curves. KVS zero adds no attenuation.
Focused WAV regressions cover boundaries and combined saturation. The note
renderer now connects both terms for the unambiguous inputs above. Analytic
one-second WAV regressions cover velocity curves, transposed-key scaling,
their sum and TL saturation. All eight algorithm continuation tests now use
distinct nonzero key/velocity sensitivities. These model-level checks do not
establish hardware qualification.

Explicit zero-velocity synth triggers now render with every KVS setting rather
than being rejected. The recommended provisional extension combines the
[published KVS calculation](https://nornand.hatenablog.com/entry/2021/01/01/153911)
with the leading `FE` value in the
[reported firmware table at $DF94](https://nornand.hatenablog.com/entry/2023/06/26/115707).
For nonzero sensitivity `s` and table value `v`, integer TL attenuation is
`((((s * v + (7 - s) * 16) >> 3) + 1) >> 1)`.
The 127 trailing firmware values reproduce all 889 existing positive-velocity
entries exactly; those measured rows are unchanged. Using `v = 254` yields
`[22, 37, 52, 67, 81, 96, 111]` for zero velocity at KVS 1..7. This is an
explicit-trigger model choice, not proof that hardware creates notes from MIDI
velocity zero: MIDI Note On with velocity zero retains its existing Note Off
interpretation. Output, carrier and key attenuation still sum before TL
saturation. WAV regressions cover all sensitivities, saturation, polyphonic
onsets, mono legato, analytic output and JSON continuation.

The numeric `tx81z_operator_samples` kernel now renders all eight four-operator
graphs in visible OP1–OP4 order, with OP4 two-sample feedback, independent
waveform selections and caller-prepared per-frame frequency/gain rows. Its
`TX81ZState` retains Q20 integer phases and signed integer feedback through
JSON continuation without
mutating the caller's saved state. Routing/modulation scaling follows
[ymfm's four-operator output path](https://github.com/aaronsgiles/ymfm/blob/main/src/ymfm_fm.ipp).
Pinned integer-core 48 kHz WAV regressions cover all graphs, asymmetric operators,
feedback, waveform selection and partitioned continuation. This is raw carrier
mixing. The note renderer now connects chip detune, nominal VCED gain conversion
and the envelope clock, plus key/velocity scaling in the supported profile.
Explicit output groups apply DAC saturation after this raw mixing. The full
voice profile and hardware qualification remain work.

The user approved ymfm's waveform model provisionally on 2026-11-03, including
the inferred signed sine-squared shape. `tx81z_wave_samples` now realizes all
eight quantized logarithmic shapes using zero-based ACED waveform numbers. The model
follows [ymfm's OPZ waveform construction](https://github.com/aaronsgiles/ymfm/blob/main/src/ymfm_opz.cpp);
independent mathematical tables reproduce the pinned sine and power tables.
This does not establish measured hardware equivalence. One-second 48 kHz
regressions compare all 1024 cells of each waveform, including negative/unwrapped
phases and the reference's nonzero half-cycle substitution. The complete TX81Z
renderer remains work.

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
The expanded independent rate sweeps now cover every authored AR/D1R/D2R/RR
value in 116 distinct profiles, including the original eight. All 22,272,000
attenuation values match the pinned reference exactly, including all four shifts.
Every profile also exercises release, partitioned JSON continuation and a paired
one-second 48 kHz WAV regression. This is not the Cartesian product of all rates,
keys and levels, and slow stages may outlast the observation interval.
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

`tx81z_base_frequency` now prepares native ratio frequency or nominal fixed
frequency from matching VCED/ACED operators, before chip detune. Ratio mode
uses the pinned 768-entry phase table, octave saturation and integer multiplier
rounding. Both modes reject unavailable fine values
above seven when VCED coarse is below four. Fixed mode uses the coarse index
shifted by two, plus fine and the ACED range, reproducing the manual's
8–32,640 Hz span and its range-dependent steps. The fixed equations are
cross-checked against
[DXconvert's `fix_4op`](https://github.com/richotero/dxconvert/blob/27a3c6dbfa97d57dfee0b2bcdbe32ed7bc98d8fa/DXconvert/fourop.py);
the [Yamaha manual](https://www.polynominal.com/site/studio/gear/sold/yamaha_tx81z/yamaha-tx81z-manual.pdf)
and [TX81Z Programmer ratio displays](https://github.com/mgregory22/tx81z-programmer/blob/master/src/freqratios.c)
corroborate the control mapping. Audio regressions cover ratio,
fine, fixed-frequency boundaries and different played keys. The note renderer
adds chip detune before multiplier rounding, and PM enters the table position
before saturation. Host microtuning, scoped tuning and portamento retain their
approved continuous ratio overlay, leaving the resulting detune difference
unscaled. The operator core quantizes the resulting
frequency to Q20 increments; full voice-profile qualification remains work.
The [native frequency audit](../doc/yamaha-reference/opz-frequency.md) records
96 ratio and 1,984 fixed register cases. The pinned ymfm fixed path produces
increments about ten octaves below its stated frequencies, so it is not used
as a fixed-frequency oracle. Native output-rate and full-voice qualification
remain open.
The expanded ratio sweep covers all 95,232 legal coarse/fine/key combinations
with zero chip detune and modulation. It finds 5,456 native top-octave clamps,
up to 853.530 cents difference from continuous nominal frequency, and up to
10.037 cents difference even outside saturation. This motivated the native
ratio replacement above. All 95,232 neutral-detune base frequencies now match
the pinned register export exactly. Audio regressions cover signed detune,
fine rounding, upper/lower PM saturation and restored continuation. This
qualifies the covered frequency arithmetic against the provisional model,
not hardware equivalence or native-rate complete-patch rendering.

The [complete ratio-note comparison](../doc/yamaha-reference/opz-note.md)
now programs the pinned YM2414 through register writes and clocks the full
channel and DAC through release. Across eight algorithms, all eight feedback
levels and two distinct patches, production host-clock error ranges from 2.94% to 137.79% relative
RMS. A control drives enge's current operator, envelope and DAC components on
native ticks and matches all 6,144,000 host observations. The next implementation
gate is a common native tick origin, tick-aligned events and LFOs, per-tick
voice summation/DAC and partition-independent host observation. Native timing
and held output must survive snapshot restoration.

- Pair VCED and ACED edit-buffer messages explicitly before conversion.
- Decode its four-operator, eight-algorithm routing, frequency modes, feedback,
  and ACED waveform selections into the same Yamaha renderer family.
- Preserve the TX81Z waveform definitions rather than mapping them to enge's
  generic sine, square, and triangle waveforms.

A historical pinned waveform-table comparison measured all eight continuous shapes
against ymfm at 440 Hz in paired one-second, 48 kHz WAVs. Relative RMS differences
range from 0.338% to 0.803%; table quantization and nonzero zero-substitution remain
explicit approximation differences in that former core. The integer replacement
now matches the captured tables exactly. See the [comparison report](../doc/yamaha-reference/README.md#tx81z-waveform-table-comparison).
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
OP6. Sync-on resets them to zero; sync-off quantizes the supplied phases onto the
Q24 grid. Omitting
phases starts a standalone note at zero. Invalid lengths, nonfinite values and
values outside `[0, 1)` are rejected. Caller mutation cannot affect a live note.
JSON note snapshots now preserve integer running phases and the last per-frame
phase increments across continuation, replacing the former floating fields.
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
any positive value to at least 1. TX81Z explicit zero-velocity triggers use the
provisional KVS extension above.

The adapter admits transpose for both models and TX81Z delayed pitch and
amplitude modulation using the provisional models above.
External envelopes and unsupported portable processing remain rejected.
DX7 hardware controller assignments, TX81Z wheel/breath/foot LFO assignments
and TX81Z breath EG bias
now use the scoped-control path described below. Other TX81Z hardware-specific
fields remain stored but inactive; they are not all rejected at admission.
Released DX7
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

The approved dynamic-tuning integration adds optional per-render cents rows to
both note renderers. Rows affect ratio operators only; TX81Z additive chip detune
remains outside the tuning multiplier, matching static microtuning. Invalid row
shape, nonfinite cents and nonpositive/nonfinite resulting ratios are rejected
before note state advances. The offline adapter uses existing scoped control
mapping and smoothing, subtracting static tuning already applied on admission.
It evaluates tuning only while the voice is active. Note snapshots retain phase
continuation; instrument snapshots also retain the controlling ramps and scopes.
One-second WAV regressions compare dynamic rows against successive static
microtuned references with mixed fixed/ratio operators, then cover instrument,
part and trigger scopes, interrupted ramps, static tuning, filters and restored
continuation. This implements portable tuning routes, not hardware pitch-bend
assignments or portamento.

### Remaining integration gates

- The approved shared controller schema is available in uFor `047c741`:
  `yamaha.pitch_bend` uses a normalized -1..1 domain; `modulation_wheel`,
  `breath` and `foot` use 0..1. All have unit `ratio` and default zero, and
  retain the existing scoped bindings, mappings and smoothing. This is the
  input contract. All four DX7 targets now use the existing scoped-control path;
  TX81Z wheel/breath/foot targets drive LFO depths, and pitch bend now uses the
  patch's 0–12-semitone range through the existing ratio-only tuning path.
  The user's standing instruction to use the recommended model resolves the
  earlier provisional bend decision. Bend combines additively with portable
  tuning in cents; fixed operators and additive chip-detune offsets are unchanged.
  This follows the [OPZ phase-step model](https://github.com/aaronsgiles/ymfm/blob/81aec25ccbb98f4873a255f7551ac4dadac59b4a/src/ymfm_opz.cpp),
  not verified TX81Z firmware register updates. Regressions cover negative,
  centered and positive positions, zero/full range, mixed operator modes,
  nonzero detune, scoped interrupted smoothing and snapshot continuation.
- The approved DX7 performance settings are available in uFor `e056201`,
  outside the lossless voice dump: a 0..12-semitone bend range (default two),
  and wheel/breath/foot ranges with independent pitch, amplitude and EG-bias
  switches, all disabled by default. Enabled assignments now render through
  the approved provisional combination model.
- DX7 note rendering accepts per-frame hardware bend cents, applied to both
  fixed and ratio operators, following the pinned Dexed `dx7note.cc` bend path.
  Portable tuning remains ratio-only and composes with bend. The offline
  adapter multiplies normalized bend by the authored semitone range. Snapshots
  retain oscillator continuation and existing scoped control ramps. WAV
  regressions compare positive/negative bend against equivalent static mixed
  fixed/ratio patches and cover scoped smoothing with concurrent tuning.
  This is continuous bend; hardware MIDI quantization and bend-step settings
  are not qualified by these tests.
- DX7 wheel/breath/foot use the approved provisional
  [Dexed controller combination](https://github.com/asb2m10/dexed/blob/fea2f0f0f2c397799f59b27eb11c038cd34b22c5/source/msfa/controllers.h):
  positions truncate to seven-bit values, range scaling truncates again, and
  each destination takes the strongest assigned contribution. EG attenuation
  is disabled when no controller has an EG assignment. Missing controller
  targets supply position zero, including for active EG assignments.
  Controller modulation bypasses patch delay and competes with patch LFO depth.
  The approved hybrid retains Hexter's 52.75-unit full-depth attenuation scale
  and existing patch-only amplitude sound, rather than Dexed's attenuation
  approximation. Controller pitch modulation affects ratio operators only.
  The same reference audit corrects the older patch path: fixed operators now
  ignore pitch-envelope and patch LFO pitch modulation as well, while retaining
  hardware pitch bend. This changes fixed-operator patches that previously
  received those modulations incorrectly.
  One-second WAV regressions cover analytic attenuation, static pitch references,
  all controller routes, scope ownership, interrupted smoothing and restoration.
  This remains a provisional model, not hardware qualification.
- The [TX81Z owner's manual, page 22](https://usa.yamaha.com/files/download/other_assets/9/316769/TX81ZE.pdf)
  routes controller-driven modulation separately from delayed patch PMD/AMD.
  The firmware-derived additive model supersedes the earlier strongest-depth
  approximation. Each position is floored to MIDI CC 0..127; its assignment
  becomes `assignment * 165 >> 6`, and the contribution is
  `(assignment_byte * cc * 2) >> 8`. Contributions sum and saturate with the
  delayed patch byte before PMD/AMD conversion. Controller PMD/AMD bypass delay.
  Direct `controller_depths` rows now carry summed integer byte contributions
  in 0..255, not authored 0..99 values. Depths are applied per note
  to shared raw observations, so scoped controls do not advance extra clocks.
  Regressions cover all waveforms, patch addition and saturation, quantized
  controller positions, modulation during patch delay, scoped smoothing, overlapping
  notes and snapshot restoration. This is provisional, not firmware qualification.
  Breath pitch bias remains unresolved. The provisional
  bend model leaves fixed operators unchanged; firmware qualification remains
  open. Breath routes reject
  nonneutral pitch bias rather than ignoring that assignment.
  Direct `TX81ZRenderer.render` also rejects positive breath rows when VCED 75
  is nonneutral, before advancing oscillators, envelopes or portamento. Omitted
  or all-zero breath rows remain valid because the pitch assignment is inactive;
  neutral pitch bias still permits the measured EG-bias path. This closes a
  direct-API admission gap, not the missing pitch-bias renderer.
  The manual's compatibility section (printed page 48) explicitly describes
  the TX81Z bias curve as exponential, unlike the DX27/100 curve. Direction and
  a neutral setting alone therefore do not justify a linear cents model.
  The VCED wire table (printed page 72) specifies 0..99 with center 50, although
  the user-interface description spans -50..+50. The
  [published firmware memory map](https://nornand.hatenablog.com/entry/2023/06/26/115707)
  reports an internal maximum of 100 and an unidentified 51-entry table at
  `$A485`. These are useful leads, not proof of wire-format conversion or pitch
  arithmetic. Keep the documented 0..99 wire validation until that conversion
  is established. Remaining evidence must identify assignment-to-depth mapping,
  depth times CC2 arithmetic, pitch units and fixed-operator behavior. Do not
  treat the unidentified table as a qualified breath-bias curve.
  A further source check of
  [Deicsonze's render source at `b50ba70`](https://github.com/ngeiswei/deicsonze/blob/b50ba700e069ed64d80c7efb40e0f645e5172f80/src/deicsonze/deicsonze.cpp)
  found `bcPitchBias` assignments for import and controller edits, but no
  consumption of that field in its rendering code. That source does not supply
  the missing conversion formula. The owner's manual and published memory map
  still do not establish assignment scaling, CC2 arithmetic or fixed-operator
  behavior, so the explicit unsupported boundary remains necessary.
- The [measured breath EG-bias table](https://nornand.hatenablog.com/entry/2020/11/24/224423)
  is complete: 100 bias settings × 8 operator sensitivities × 128 CC2 values.
  The 102,400 values are all in 0..127; attenuation decreases monotonically with
  breath, reaches zero at CC2 127, and is zero for bias or sensitivity zero.
  The ordered uint8 payload (bias, sensitivity, CC2 ascending) has SHA-256
  `80910f5f9803def7904843145433f9428407cd2395cc72ff2da3c8779f3da3ee`.
  These measurements put EG bias in the TL sum before saturation. Firmware
  updates are only approximately 48 Hz and the TL high-bit behavior is uncertain.
  The approved provisional implementation ships this table in
  `enge/yamaha_breath.npy` and uses the existing per-frame scoped-control cadence.
  Normalized breath positions truncate to CC2; omitted input means CC2 zero.
  Each operator applies its own sensitivity and adds attenuation before TL
  saturation. Audio regressions cover measured values, fractional CC2 inputs,
  zero sensitivity, saturation, scoped smoothing and snapshot restoration.
  Firmware cadence and TL high-bit behavior remain unqualified.
- TX81Z FC Volume now uses a provisional range mapping onto the
  [measured master-volume attenuation table](https://nornand.hatenablog.com/entry/2020/11/21/201911).
  The owner's manual specifies full control at assignment 99 and residual sound
  at intermediate assignments, but does not specify the arithmetic. Normalized
  foot position truncates to CC 0..127; the selected internal volume position is
  `127 - ((127 - CC) * assignment // 99)`. Assignment zero has no effect;
  omitted foot input means position zero, consistent with the portable target.
  The measured 128-entry volume table applies only to carriers, before TL
  saturation, alongside breath bias. Minimum volume reaches the renderer's
  existing TL-127 floor. Per-frame updates reuse scoped controls and smoothing.
  One-second WAV regressions cover disabled, partial and full ranges, fractional
  CC input, unchanged modulators, combined breath saturation, overlapping notes,
  all control scopes and snapshot restoration. Range arithmetic, firmware
  cadence and hardware silence remain unqualified.
- TX81Z VCED 63–76 retains poly/mono, pitch-bend range, portamento, foot volume,
  sustain, chorus and modulation-wheel/breath assignments. ACED 21–22 retains
  foot-controller pitch/amplitude assignments. Wheel/breath/foot LFO modulation
  and breath EG bias, foot volume, ratio-only pitch bend and polyphonic full-time
  portamento with switch control are implemented as above. Mono legato and
  fingered portamento are implemented provisionally as above. Breath pitch bias
  and hardware qualification remain unfinished. Chorus is a compatibility-data
  field, not an unfinished TX81Z effect: the owner's manual explicitly says
  this instrument has neither chorus nor a pitch envelope (printed page 48).
  Preserve the imported bytes, but do not add a chorus renderer to emulate a
  feature absent from this hardware. This corrects the earlier backlog wording
  and does not select the separate optional effects catalogue.
- TX81Z explicit zero-velocity note creation with nonzero KVS is implemented
  using the provisional ROM-table extension above; positive-velocity measured
  attenuation remains unchanged. MIDI zero-velocity Note On remains Note Off.
- The approved DX7 integer oscillator/feedback replacement is implemented.
  Constant-gain reference comparisons and scoped-control continuation cover the
  new core; complete dynamic patches and hardware output remain unqualified.
- TX81Z multitimbral LFO routing is implemented for shared hardware owners and
  provisional independent vibrato. Software-LFO firmware behavior and
  complete-renderer reference qualification remain incomplete. No passing
  component test closes those qualification requirements.

### TX81Z firmware-trace qualification path

The [2023 interrupt investigation](https://nornand.hatenablog.com/entry/2023/04/06/114613)
identified missing OPZ IRQ and CPU Timer 2 handling in the author's MAME setup.
It locates periodic register writes in the firmware's Timer 2 handler at
`$8F1F`, relevant to LFO delay and portamento. This explains why a trace from an
older emulator cannot automatically qualify those behaviors.

Source inspection on 2026-11-04 pins MAME at
`8f02776aa011cf9d249a35b1f95b434560ad70f6`:

- Its [TX81Z driver](https://github.com/mamedev/mame/blob/8f02776aa011cf9d249a35b1f95b434560ad70f6/src/mame/yamaha/ymtx81z.cpp)
  connects the OPZ IRQ handler to `HD6301_IRQ1_LINE` and selects an HD6303X CPU.
- Its [CPU implementation](https://github.com/mamedev/mame/blob/8f02776aa011cf9d249a35b1f95b434560ad70f6/src/devices/cpu/m6800/m6801.cpp)
  advances Timer 2 from the selected internal clock, sets the compare flag,
  and dispatches the enabled counter-match interrupt through vector `$FFEC`.
- The driver identifies firmware versions 1.0 through 1.6. Its default 1.6 image
  is 65,536 bytes with SHA-1 `208a72c0dc615825c442240e520a6a3c5fe860ea`.
  It still declares imperfect sound.

These are source-level findings, not an executed firmware qualification. No ROM
was acquired and no emulator was run. The user has no dump available and asked
us to work around this on 2026-11-04. Firmware tracing is therefore an optional
qualification route, not a prerequisite for further implementation. Continue
with published measurements and approved provisional software models; unresolved
behavior still needs an explicit model decision. If tracing becomes available,
before using its results, verify
that the IRQ handlers execute and periodic register writes occur. Then capture
timestamped writes for isolated delayed AMD, foot volume and breath pitch bias,
including parameter endpoints, intermediate settings and fixed/ratio operators.
Keep the firmware hash, emulator revision and input events with each trace.
Measured register behavior would resolve these controller contracts separately
from OPZ audio accuracy; it would not establish hardware equivalence.

## Additional work beyond the prompt

None.
