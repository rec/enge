# Yamaha reference comparisons

The [WT11 pitch-envelope audit](wt11-pitch.md) records a pinned provisional
conversion model and its timing/level limits before renderer integration.

The [complete ratio-note comparison](opz-note.md) clocks a programmed YM2414
through key-on and release. Native-clock enge components match all 6,144,000
observations. The public note renderer now matches the same 128 captures.

The [OPZ native frequency audit](opz-frequency.md) records the pinned reference's
ratio-table differences and fixed-frequency phase-scale inconsistency.

Measured on 2026-11-03 against MSFA revision
`f67d41d313b7dc85f6fb99e79e515cc9d208cfff`, using enge `59287dd`.
This is a failing qualification result, not evidence of hardware equivalence.

The two patches are constructed for this comparison, not imported factory
patches. Both use six ratio-mode operators at coarse 1, fine 0, neutral detune,
output level 80, amplitude rates 99 and levels 99/99/99/0. Pitch levels are
neutral (50), feedback is off, oscillator sync is on and transpose is neutral
(24). Other parameters are zero. Titles identify algorithm 1 and algorithm 32.
The raw 155-byte VCED payloads are included alongside the audio.

Both implementations render MIDI key 69 at velocity 100 and 48,000 Hz for
48,000 frames. Release occurs at frame 24,000. MSFA pitch bend is centered,
LFO pitch input is neutral and LFO depth is zero. `render-msfa.cc` initializes
the note's storage to zero and invokes the upstream renderer in its native
64-frame blocks.

| Patch | Whole-file relative RMS error | Held-section relative RMS error | Largest error frame |
| --- | ---: | ---: | ---: |
| ENGE-REF01 | 5.2000% | 0.4261% | 6 |
| ENGE-REF32 | 4.1339% | 0.1007% | 24021 |

Relative RMS error is `norm(enge - reference) / norm(reference)`. The held
section is frames 12,000 through 23,999. No alignment, frequency adjustment or
independent gain fitting is applied. Each WAV pair uses one shared peak scale
to fit signed 32-bit PCM; pairs have independent scales. No samples are clipped.

- [ENGE-REF01 MSFA](ENGE-REF01-msfa.wav), [enge](ENGE-REF01-enge.wav)
- [ENGE-REF32 MSFA](ENGE-REF32-msfa.wav), [enge](ENGE-REF32-enge.wav)

The measured differences are concentrated around onset and release. Upstream
amplitude envelopes update once per 64-frame block, with interpolated gains;
enge updates envelopes per sample. MSFA also uses integer frequency and sine
lookup tables, whereas enge uses floating-point oscillators. These observations
identify candidate causes, not a complete attribution of the measured error.
Matching those block artifacts versus retaining per-sample execution requires
a target decision. Hardware qualification, nonzero feedback, time-varying
pitch/LFO modulation and arbitrary patches remain unverified.

## All-algorithm sweep and algorithm 23 correction

A follow-up sweep used the same uniform operator settings for all 32 algorithms,
changing only the algorithm byte at offset 134 and the patch title. The
[before/after measurements](algorithm-sweep.csv) store relative errors as
fractions, not percentages. The baseline used uFor `68c693e`; the corrected run
used `72e9002`. The enge DSP implementation and MSFA revision were unchanged.

Algorithm 23 was the only held-section outlier: 35.234% error before correction.
uFor duplicated algorithm 19's routing. Algorithm 23 instead has edges OP3 to
OP2 and OP6 to OP4/OP5, with carriers OP1/OP2/OP4/OP5. Correcting the table reduced
its held-section error to 0.2432% and whole-file error to 6.3522%.
The other 31 algorithms' measured errors were unchanged; all corrected held
sections are below 1% error for this patch family. This is not a general pass:
uniform settings cannot distinguish every operator-numbering error, feedback
is disabled and transient differences remain.

The corrected [ENGE-REF23 MSFA](ENGE-REF23-msfa.wav) and
[enge](ENGE-REF23-enge.wav) recordings and the raw VCED payload are included.
Construct each sweep patch from ENGE-REF01 by changing offset 134 to
`algorithm - 1` and the ten title bytes to `ENGE-REFnn`, where `nn` is the
two-digit algorithm number.

## Asymmetric operator sweep

The [asymmetric sweep](asymmetric-algorithm-sweep.csv) uses enge `bffac67` and
the same pinned MSFA source, note, velocity, timing and neutral controller state.
The six visible operators OP1 through OP6 use ratios 1/2/3/4/5/6 and output
levels 80/77/74/71/68/65, respectively. Other patch parameters match ENGE-REF01.
Titles are `ENGE-ASYnn`, with `nn` identifying the algorithm.

Held-section relative RMS error ranges from 0.2135% (algorithm 26/27) to
0.6655% (algorithm 16). Whole-file relative RMS error ranges from 4.1626%
to 5.8586%. No further routing outlier appeared. Differing operators remove
some symmetries of the uniform sweep, but this remains one constructed patch
family with feedback disabled, not exhaustive topology or hardware proof.

The raw ENGE-ASY16 VCED payload and paired
[MSFA](ENGE-ASY16-msfa.wav) and [enge](ENGE-ASY16-enge.wav) recordings capture
the largest held-section discrepancy in this sweep. WAV scaling and the held
section use the same definitions as above. CSV error columns are fractions;
maximum absolute error and signal RMS values use unscaled upstream Q24/2**24
amplitude units.

## Feedback sweep

The [feedback sweep](feedback-sweep.csv), measured on 2026-11-04 with enge
`273def5`, uses the same pinned MSFA source and render timing. Algorithms
1, 2, 16 and 32 each run at feedback settings 1, 4 and 7, covering different
feedback operators, modulator chains and a feedback carrier. Every case has
two profiles:

- `asymmetric` retains the ratios and levels from the asymmetric sweep.
- `high-output` changes only the feedback operator's output level to 99:
  OP2 for algorithm 2, OP6 for the other selected algorithms.

Construct each patch from ENGE-ASY16 by setting offset 134 to `algorithm - 1`,
offset 135 to the feedback setting and the ten title bytes to `ENG-FBnn-f`.
For `high-output`, also set byte `(6 - operator) * 21 + 16` to 99, where
`operator` is the visible feedback operator above. All other parameters,
controller inputs, rendering, WAV scaling and CSV units are unchanged.

The twelve `asymmetric` cases have held-section relative RMS errors from
0.2599% to 0.6702%. The twelve `high-output` cases range from 0.5855% to
133.0160%. At feedback 7, their held-section errors are 24.6804%, 113.2326%,
83.2009% and 133.0160% for algorithms 1, 2, 16 and 32 respectively.
The worst whole-file error is 132.5152%, also algorithm 32 at feedback 7.
High-output feedback is therefore an explicit failing qualification case.
The comparison does not isolate quantization, gain interpolation or nonlinear
feedback sensitivity as the cause, and the earlier low-level results cannot
be generalized to this profile.

The matching raw payloads and WAV pairs retain algorithm 32 at feedback 7:

- Asymmetric: [VCED](ENG-FB32-7.vced), [MSFA](ENG-FB32-7-msfa.wav),
  [enge](ENG-FB32-7-enge.wav).
- High output: [VCED](ENG-FB32-7-high-output.vced),
  [MSFA](ENG-FB32-7-high-output-msfa.wav),
  [enge](ENG-FB32-7-high-output-enge.wav).

Algorithms 4 and 6 are excluded from this reference sweep. The pinned
[MSFA core](https://github.com/google/music-synthesizer-for-android/blob/f67d41d313b7dc85f6fb99e79e515cc9d208cfff/app/src/main/jni/fm_core.cc)
explicitly leaves multi-operator feedback unimplemented: its feedback kernel
requires a single operator with both feedback flags. Its algorithm 4/6 tables
put those flags on different operators. uFor now represents feedback with an
explicit source/destination edge, and enge's note renderer uses OP4-to-OP6 and
OP5-to-OP6 for algorithms 4 and 6 respectively. Comparison against this MSFA
revision still cannot qualify those loops.

Inspection of [Dexed Mk I at the pinned revision](https://github.com/asb2m10/dexed/blob/fea2f0f0f2c397799f59b27eb11c038cd34b22c5/source/engine/EngineMkI.cpp)
confirms those return endpoints in its two- and three-operator feedback kernels.
Its dispatcher also adds two to the feedback shift for algorithms 4, 6 and 32,
reducing feedback strength by four relative to its other algorithms. The user
approved that rule provisionally. Enge now follows it, including the change to
algorithm 32. Scalar regressions establish the chosen feedback arithmetic and
graph behavior, not Mk I waveform equivalence.

The earlier MSFA feedback sweep, sensitivity measurements and retained algorithm-32
WAV pairs above predate this strength change. They remain historical evidence of
the previous model; they do not quantify the current algorithm-32 renderer. A
new complete Mk I patch comparison remains required. The subsequent
[constant-gain core comparison](dexed-mki-core.md) isolates oscillator arithmetic
and retains the large high-gain discrepancies in algorithms 4 and 6. Its subsequent
[shared-kernel graph control](dexed-mki-core.md#integer-kernel-graph-isolation)
matches all 24,576,000 integer samples across 512 cases. This supports the graph
and feedback scheduling under fixed gains, not complete renderer equivalence.

## TX81Z paired hardware LFO comparison

The same pinned ymfm register harness now accepts two optional trailing
arguments: the secondary rate and waveform. In that mode it sets secondary
PMD/AMD to 127, selects primary-only AM on channel 0 and secondary-only AM on
channel 1, and emits four native-endian signed 32-bit values per frame:
primary PM, primary AM, secondary PM, secondary AM. Both clocks run inside
one `opz_registers`, sharing its noise generator and sample-and-hold table.

The [paired oracle](../../test/data/tx81z-dual-lfo.npz) contains arrays `wave00`
through `wave33`, covering all 16 primary/secondary waveform pairings. Each is
48,000 rows by four columns, captured directly from that harness. Primary
authored speed 75 maps to rate 242 for periodic waves and 193 for noise;
secondary speed 50 maps to 223 for periodic waves and 128 for noise. Primary
PMD/AMD are also 127. For example, the two-noise capture arguments are
`193 127 127 3 output.i32 128 3`. No phase reset is injected during capture.
The source `ymfm_opz.cpp` Git blob is
`949061baf4716b5d9932650685ebedcd2726e341` at the revision below.

All 768,000 four-column rows match exactly after applying depth 127 to enge's
raw waveform rows. Tests write paired one-second 48 kHz WAV files and resume
both clocks from JSON at frame 17,003. Separate tests check unequal delay
progress, selective phase reset from the two sync flags, preservation of noise
state on keydown, and rejection of a mismatched secondary definition before
restore. The compressed oracle's SHA-256 is
`f9684e8a0a38ae8c282d65b8851a9affda81c1d760fbecaacb2d17af963c926d`.

`TX81ZLFOContext.advance_ticks(frames)` exposes every native observation.
It returns an array of final tick indices for the host frames and an array
of observations. Observation zero holds the preceding value. A frame with
zero completed ticks reuses that value, and a frame with multiple ticks can
access the intervening rows. `advance(frames)` projects these observations
onto the host frame indices and stores one row per frame. Both calls advance
the same clock. A caller can advance once and share the resulting rows across
all voices rendering that interval.

The reference harness accepts `--ticks` before its register arguments to
capture every completed chip tick. The paired capture used
`--ticks 193 127 127 3 output.i32 223 2`. Its 55,930 four-column rows are
stored as the `rows` array in `test/data/tx81z-lfo-ticks.npz`, compressed as
little-endian signed 16-bit integers. The archive's SHA-256 is
`0267752f7816c998eb02e56a765d846095088c1e246e8b779a8a7e6077d09ef1`.
The primary noise and secondary triangle observations match the capture at
both 24 kHz and 96 kHz host rates. Tests verify frame indices, zero-length
advances and JSON continuation after 17 frames. Delay progress retains the
existing host-frame timing model.

This qualifies paired clock arithmetic against ymfm. It does not qualify
hardware, firmware delay-reset policy, multitimbral allocation/routing, software
vibrato, or the full performance renderer. Restarting both configured delay
ramps on keydown is a provisional extension of the existing single-clock rule.

## TX81Z single-mode LFO comparison

The TX81Z shared LFO context is compared against ymfm revision
`81aec25ccbb98f4873a255f7551ac4dadac59b4a`, using the included
[register-level harness](render-ymfm-lfo.cc). It clocks `opz_registers` directly,
with the secondary LFO left at reset settings. The master clock is 3,579,545 Hz,
with one native step per 64 clocks; an integer remainder converts this to 48 kHz
output rows. Each case contains 48,000 signed-PM/unsigned-AM pairs. No phase or
noise reset is injected after initialization.

All cases use authored speed 75, PMD 99 and AMD 99. The measured register values
are rate 242 for saw/square/triangle and 193 for sample-and-hold, with PMD/AMD
both 127. The harness selects channel AMS 1 so the AM result is the raw,
depth-scaled LFO output. Waveform numbers are 0, 1, 2 and 3 respectively.

| Waveform | Compared PM/AM rows | Maximum absolute error |
| --- | ---: | ---: |
| Saw | 48,000 | 0 |
| Square | 48,000 | 0 |
| Triangle | 48,000 | 0 |
| Sample-and-hold | 48,000 | 0 |

The [compressed oracle](../../test/data/tx81z-lfo.npz) retains the reference as
four portable NumPy arrays named `wave0` through `wave3`, each containing signed
16-bit pairs without loss of register values. Tests compare complete rows,
write paired one-second 48 kHz WAVs and check partitioned JSON continuation.
The shared context now returns raw PM/AM plus rational delay progress. The
comparison applies the oracle's PMD/AMD register value of 127 to the first two
columns before comparing these unchanged fixtures. Per-note depth application
allows simultaneous voices to consume one clock with independent controllers.
The initial chip noise state, shared latch table and secondary counter matter
for sample-and-hold; reproducing only a free-running random value is insufficient.

Compile the harness as C++17 with `-O2`, `ymfm_opz.cpp` and the pinned upstream
include directory. Its five arguments are rate, PMD, AMD, waveform and output
filename. The output is 48,000 native-endian signed 32-bit pairs. For example,
the saw case uses `242 127 127 0 output.i32`; sample-and-hold uses
`193 127 127 3 output.i32`. The upstream source files used by the harness are
`ymfm_opz.cpp`, `ymfm_opz.h`, `ymfm_fm.ipp`, `ymfm_fm.h` and `ymfm.h`.

This is software-model conformance of the single-mode LFO component. It does
not qualify note synthesis, nonzero delay, multitimbral routing, the firmware's
software LFO or physical hardware behavior. ymfm's documented OPZ uncertainties
remain applicable.

## Reproducing the DX7 reference

Use the pinned [upstream source](https://github.com/google/music-synthesizer-for-android/tree/f67d41d313b7dc85f6fb99e79e515cc9d208cfff).
Compile the included harness with the upstream JNI directory on the include
path and these upstream sources: `dx7note.cc`, `env.cc`, `pitchenv.cc`,
`fm_core.cc`, `fm_op_kernel.cc`, `sin.cc`, `exp2.cc` and `freqlut.cc`.
The measured build used Apple Clang, C++11 and `-O2` on macOS arm64.

Invoke the harness with the VCED filename and an output filename. Its output is
48,000 native-endian signed 32-bit values in upstream Q24 amplitude units.
Divide by 2**24 to obtain the unscaled reference signal. For enge, construct
`DX7Renderer` from the same payload, render 24,000 frames, release, then render
24,000 more. Preserve the raw amplitude relationship when comparing or writing
the WAV pair.

## Feedback sensitivity diagnostic

A scalar recurrence diagnostic on 2026-11-04 separates sensitivity to initial
feedback state from differences between renderers. It uses neither renderer,
no envelope transient and no sine-table approximation. The
[measurements](feedback-sensitivity.csv) compare two otherwise identical
floating-point recurrences for 48,000 samples at 48 kHz:

```
y[n] = gain * sin(2*pi*(phase[n] + (y[n-2] + y[n-1])/2**(9-feedback)))
phase[0] = 0
phase[n+1] = (phase[n] + 2640/48000) % 1
```

The reference starts with both history values zero; the other starts with
`y[-2]=0` and `y[-1]=2**-24`, one MSFA output-unit LSB. Both use Python's
binary64 `math.sin` and the same separately maintained phase. Output levels
65 and 99 give held gains `2**(-3.25)` and `2`, using the existing neutral
scaling and level-99 sustain equations. Feedback settings are 1, 4 and 7.
Held relative RMS difference uses frames 12,000 through 23,999, as above.
A blank first-error field means no sample exceeded one percent of gain.
This diagnostic is not a replacement for the reference WAV comparisons.

At output 99 and feedback 7, the initial difference exceeds one percent of
gain at frame 26 and gives 138.026% held relative RMS difference. Every other
case converges to identical floating-point samples before the held interval.
Thus a large sample-wise discrepancy is possible even with identical
frequency, gain, feedback scaling and sine implementation. It does not by
itself establish a routing or gain bug, nor does this experiment attribute the
actual MSFA mismatch to one particular numerical difference.

For equal phase and gain, the sine derivative bounds the next output error by
`2*pi*gain/2**(9-feedback)` times the sum of the two history errors. The CSV
records twice that coefficient. Below one, the maximum history error contracts
over each pair of steps; above one, this bound gives no convergence guarantee.
The high-output/feedback-7 case has bound `2*pi`, while all other cases are below
one. This is a sufficient contraction bound, not a proof of chaos or a hardware
accuracy criterion.

Source inspection also confirms the nominal feedback denominator agrees:
MSFA sets `fb_shift=8-feedback` and shifts the history sum by `fb_shift+1`;
enge divides by `2**(9-feedback)`. MSFA additionally quantizes the history,
sine, gain and frequency and interpolates block gains. Their individual effects
remain unisolated. Qualification still fails; resolving the execution/reference
target remains necessary before changing arithmetic to chase sample equality.

## Ratio-mode detune audit

Source inspection on 2026-11-04 found that enge's ratio-mode detune equation
uses the coefficients from
[Dexed's measured fit](https://github.com/asb2m10/dexed/blob/fea2f0f0f2c397799f59b27eb11c038cd34b22c5/source/msfa/dx7note.cc)
but applies them in a different numerical domain. Write `f` for the played
key's frequency, `L=log2(f)`, and
`d=0.0209*exp(-0.396*L)*(detune-7)/7`. Before correction, enge multiplied frequency by
`1+d`. Dexed instead adds `d*L` to log-frequency, equivalent to multiplying
frequency by `2**(d*L)` before coarse/fine ratio scaling. The following values
use continuous equations, not Dexed's Q24 rounding or frequency lookup:

| MIDI key | Neutral Hz | Previous enge Hz | Dexed-fit Hz | MSFA-model Hz |
| --- | ---: | ---: | ---: | ---: |
| 21 | 27.500000 | 27.586532 | 27.788284 | 27.600440 |
| 69 | 440.000000 | 440.284037 | 441.732272 | 441.607036 |
| 108 | 4186.009045 | 4186.755126 | 4192.235613 | 4201.297829 |

All rows use coarse 1, fine 0 and detune 14, with neutral transpose and no
pitch modulation. The last column instead multiplies `f` by
`2**(12606*7/2**24)`, following the pinned MSFA approximation. Its detune is
constant in cents; Dexed's fit is key-dependent. At key 69, the previous enge
shift is 1.117219 cents versus 6.802455 cents for the Dexed fit. Neutral detune
is unchanged in all three models, so the previous neutral-detune sweeps do not
exercise this discrepancy.

The user approved correcting the existing Dexed fit on 2026-11-04. enge now
uses `2**(d*L)` and preserves the existing fixed-mode equation. Tests use
50-digit decimal calculations for positive-detune anchors, reciprocal ratios
for negative detune and neutral key frequencies, followed by coarse/fine
scaling. They compare complete one-second 48 kHz renders against analytic
sines with the existing envelope, and verify JSON snapshot continuation.
These checks qualify the continuous equation, not Dexed's Q24 rounding or
frequency lookup, and do not establish measured hardware equivalence.

## TX81Z envelope comparison

The [envelope harness](render-ymfm-envelope.cc) uses the same pinned ymfm
revision `81aec25ccbb98f4873a255f7551ac4dadac59b4a`. It creates OPZ operators
through the public API and reads their debug attenuation and prepared shift.
The upstream source is unchanged. Four operators share each parameter profile,
with envelope shifts 0, 1, 2 and 3; the unshifted operator uses OP1's register
position. Key-on is prepared before frame zero and release before frame 24,000.

| Case | AR | D1R | D2R | RR | D1L | RS | Key code | Reverb |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 31 | 0 | 0 | 15 | 15 | 0 | 0 | 0 |
| 1 | 0 | 0 | 0 | 0 | 15 | 0 | 0 | 0 |
| 2 | 12 | 9 | 5 | 4 | 7 | 0 | 0 | 0 |
| 3 | 12 | 9 | 5 | 4 | 7 | 3 | 31 | 0 |
| 4 | 31 | 31 | 7 | 10 | 0 | 1 | 18 | 0 |
| 5 | 31 | 0 | 0 | 15 | 15 | 0 | 0 | 2 |
| 6 | 31 | 0 | 0 | 0 | 15 | 0 | 0 | 7 |
| 7 | 24 | 20 | 11 | 8 | 11 | 2 | 13 | 4 |

The profiles cover immediate and zero-rate attack, slower attack/decay, key-rate
scaling, sustain-level extremes, release, reverb transition and a release slower
than the requested reverb rate. D1L is the authored control; the harness writes
`15-D1L` to the chip sustain-level register. The five-bit key code is written
into the upper channel pitch bits, independently of MIDI-to-key-code mapping.

The host observes attenuation before each 48 kHz output-frame interval and
advances the shared envelope counter using a 3,579,545 Hz master clock divided
by 192, retaining its integer remainder. The harness passes each envelope tick
as `counter << 2` to ymfm's operator clock, whose counter uses two fractional
bits. This tests envelope arithmetic and the chosen host clock conversion; it
does not independently qualify physical chip key-on timing.

The 2026-11-04 rate sweep extends these eight profiles to 116 distinct cases.
Starting with `31 0 0 0 15 0 0 0`, vary AR through 0–31, then D1R through
0–31 with D1L set to zero, then D2R through 0–31, then RR through 0–15.
Each sweep starts from that base independently; retain the original eight
profiles first and omit duplicate parameter rows. This tests every authored
rate value independently, with immediate attack for the non-attack sweeps,
zero sustain attenuation for the D2R sweep, and held full level for the RR sweep.

All 48,000 four-operator rows in each case match exactly: 22,272,000 attenuation
values, maximum absolute error zero. The
[compressed reference](../../test/data/tx81z-envelope.npz) stores 116 signed
16-bit arrays, `case0` through `case115`, and their eight-column `parameters`
table. Regressions also check partitioned JSON continuation after release and
write paired one-second 48 kHz WAVs using these attenuations on a 440 Hz carrier.
No production renderer changes were needed for these profiles.

Compile as C++17 with `-O2`, the pinned upstream include directory and
`ymfm_opz.cpp`. The harness includes `ymfm_fm.ipp` for the operator template
implementations. Its nine arguments are the eight parameter columns followed
by an output filename; case 2 is `12 9 5 4 7 0 0 0 output.i32`. Output is
48,000 rows of four native-endian signed 32-bit attenuation values, which can
be converted losslessly to the fixture's signed 16-bit arrays.

The pinned source files were verified against their upstream Git blob hashes
before rebuilding the unchanged harness for the sweep. This qualifies these
envelope profiles against the provisional software model. Slow stages need not
complete within the half-second held or released interval, and this is not the
Cartesian product of rates, levels and keys. It does not qualify every rate/key
combination, full TX81Z synthesis, WT11
behavior, controller interaction, integer audio output or physical hardware.

## TX81Z waveform-table comparison

The measurements in this section describe the former continuous-waveform core.
The [production integer replacement](opz-core.md#production-integer-replacement)
now matches all captured waveform cells and fixed-gain core cases exactly.

`render-ymfm-waveforms.cc` reads all eight waveform tables through the public
operator cache of the same pinned ymfm revision
`81aec25ccbb98f4873a255f7551ac4dadac59b4a`. It converts logarithmic attenuation
with ymfm's own integer volume lookup. It does not apply envelope attenuation,
feedback, modulation, chip-clock pitch quantization or DAC processing.

From the repository root, build and capture the table, with `YMFM` pointing
to that revision's source folder:

```sh
mkdir -p build
c++ -std=c++17 -O2 -I"$YMFM" doc/yamaha-reference/render-ymfm-waveforms.cc "$YMFM/ymfm_opz.cpp" -o build/ymfm-waveforms
build/ymfm-waveforms > build/ymfm-waveforms.csv
```

The comparison uses 48,000 frames at 48 kHz and 440 Hz, with each WAV's eight
channels containing waveform numbers 0 through 7. For frame `n`, the reference
index is `floor(n * 440 * 1024 / 48000) % 1024`, computed with integer arithmetic;
the reference sample is the corresponding table value divided by 8192. The enge
sample is `tx81z_wave_samples(w, ((n * 440) % 48000) / 48000)` using the continuous
phase, without midpoint alignment or fitted gain. Each file stores samples
rounded to signed 16-bit PCM after multiplication by 32767. Metrics in
[waveform-comparison.csv](waveform-comparison.csv) precede PCM conversion.

| Waveform | Relative RMS error | Maximum absolute error |
| --- | ---: | ---: |
| 0 | 0.3376% | 0.0046013 |
| 1 | 0.3887% | 0.0063892 |
| 2 | 0.5489% | 0.0046013 |
| 3 | 0.6331% | 0.0063892 |
| 4 | 0.6955% | 0.0093416 |
| 5 | 0.8031% | 0.0115181 |
| 6 | 0.6955% | 0.0093416 |
| 7 | 0.8031% | 0.0115181 |

Relative RMS divides the error norm by the reference norm. The paired
[enge WAV](TX-WAVEFORMS-enge.wav) and [ymfm WAV](TX-WAVEFORMS-ymfm.wav) preserve
these differences without normalization. In particular, ymfm substitutes its
first sine-table entry where the continuous shapes use zero; waveform 4 and 5
therefore have reference DC 0.00152587890625, versus approximately zero in enge.
The table also samples near phase-cell midpoints and quantizes logarithmic
amplitude. No large shape mismatch appears in this comparison, but agreement is
not exact and no hardware qualification threshold has been established.

The [pinned source's waveform construction](https://github.com/aaronsgiles/ymfm/blob/81aec25ccbb98f4873a255f7551ac4dadac59b4a/src/ymfm_opz.cpp)
explicitly treats the signed sine-squared shape as an inference. This comparison
qualifies enge against that provisional model only, not measured TX81Z or WT11
hardware. It does not resolve the outstanding full-renderer qualification.
