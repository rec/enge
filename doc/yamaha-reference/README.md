# Yamaha reference comparisons

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
put those flags on different operators. uFor's current single-operator feedback
field also cannot express those return paths. Extending that shared topology
contract requires a decision; comparison against this MSFA revision cannot
qualify those loops even after enge implements them.

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
