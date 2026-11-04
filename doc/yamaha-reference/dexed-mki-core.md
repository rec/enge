# Dexed Mk I constant-gain core comparison

The initial measurements below describe the historical floating core. The
approved integer replacement and its current regression oracle are described
at the end of this report; do not treat the old mismatch figures as results for
the current implementation.

The latest [controlled dynamic comparisons](#isolating-reference-envelope-behavior)
separate two reference behaviors from enge's intended per-frame model. Neither
experiment closes whole-note or hardware qualification.

Measured on 2026-11-04 against Dexed revision
[`fea2f0f0f2c397799f59b27eb11c038cd34b22c5`](https://github.com/asb2m10/dexed/blob/fea2f0f0f2c397799f59b27eb11c038cd34b22c5/source/engine/EngineMkI.cpp).
The enge renderer includes the approved explicit feedback endpoints and
quarter-strength rule for algorithms 4, 6 and 32. Its `enge/yamaha.py` SHA-256 is
`4f9d45516a947cb926fc76f8ba3b9f291094dcad386cc7ac745741b28692d6bc`.
The reference uses the actual Mk I core, including its integer sine/amplitude
lookups, feedback shifts and two- and three-operator kernels.

## Method

The [standalone harness](render-dexed-mki.cc) renders 750 blocks of 64 samples,
exactly one second at 48 kHz. Algorithms 3, 4, 5, 6, 31 and 32 each run at
feedback 0, 1, 4 and 7 with constant, equal operator gains of 0.03125, 0.25 and 1:
72 cases in [the complete CSV](dexed-mki-core.csv).

For every operator, the Q24 frequency increment is exactly 153791 and the initial
phase is zero. `level_in` is `L * 2**24` for `L` in 9, 12 and 14; `gain_out` starts
at `(16 - L) * 1024` to avoid an initial block gain ramp. This corresponds to
continuous gain `2**(L - 14)`. Feedback history starts at zero. Each raw signed
32-bit reference output is divided by `2**24` before comparison.

The enge comparison constructs an unpacked DX7 voice with the same algorithm and
feedback; all operators use ratio mode, coarse 1, fine 0 and neutral detune 7.
Oscillator sync is on, transpose is neutral, and pitch-envelope levels are 50.
LFO depths and sensitivities are zero. Explicit `pitch_hz` is
`153791 * 48000 / 2**24`. A public note snapshot sets every amplitude envelope's
`level` to `L * 2**24`, `stage` to 3 and `held` to true before restoring and rendering
48,000 frames. Thus there is no amplitude ramp, pitch drift, release or LFO
variation in either stream. The remaining differences are within core execution;
this experiment does not determine whether they all follow from quantization.

Relative RMS is `norm(enge - reference) / norm(reference)` over all 48,000 frames.
The CSV also records maximum absolute error and each stream's RMS before any
WAV scaling. Retained WAV pairs use one common divisor per pair, recorded in the
CSV, and round to signed 32-bit PCM using `round(sample / divisor * (2**31 - 1))`.
No independent level fitting or time alignment is applied.

## Results

| Constant operator gain | Smallest relative RMS error | Largest relative RMS error |
| --- | ---: | ---: |
| 0.03125 | 1.2274% | 1.3017% |
| 0.25 | 0.1665% | 0.3249% |
| 1 | 0.0816% | 78.3386% |

At unit gain and maximum feedback:

| Algorithm | Relative RMS error | Reference RMS | enge RMS | Paired recordings |
| --- | ---: | ---: | ---: | --- |
| 4 | 78.3386% | 1.17170 | 1.17598 | [Mk I](MKI-ALG04-FB7-mki.wav), [enge](MKI-ALG04-FB7-enge.wav) |
| 6 | 41.0212% | 1.90573 | 1.90841 | [Mk I](MKI-ALG06-FB7-mki.wav), [enge](MKI-ALG06-FB7-enge.wav) |
| 32 | 0.0816% | 4.18646 | 4.18894 | [Mk I](MKI-ALG32-FB7-mki.wav), [enge](MKI-ALG32-FB7-enge.wav) |

The two multi-operator loops still fail reference qualification at the high-gain
setting. Their similar overall RMS values do not establish waveform agreement.
The large differences persist after removing envelope timing, block gain ramps
and pitch drift. At gain 0.25, maximum-feedback errors for algorithms 4 and 6 are
0.2329% and 0.2990% respectively. These observations are consistent with feedback
sensitivity to arithmetic differences, but do not prove that explanation or
exclude a core-model error. Integer arithmetic isolation remains required.

This is a controlled FM-core comparison, not a complete named-patch comparison,
DAC model, native callback qualification or hardware validation. The reference's
hardware accuracy is not established by agreement with it. These results do not
replace the earlier MSFA patch/envelope comparisons, which exercise other paths.

## One-bit phase sensitivity control

The [reference-only control](dexed-mki-sensitivity.csv) runs the same Mk I core
twice, changing only OP6's initial phase from zero to one Q24 least-significant
bit: `2**-24` cycles. At maximum feedback and unit operator gain:

| Algorithm | Relative RMS difference | First differing frame | Perturbed reference WAV |
| --- | ---: | ---: | --- |
| 4 | 19.9127% | 1729 | [one phase bit](MKI-ALG04-FB7-phase-lsb.wav) |
| 6 | 2.2912% | 705 | [one phase bit](MKI-ALG06-FB7-phase-lsb.wav) |
| 32 | 0.0006941% | 2753 | [one phase bit](MKI-ALG32-FB7-phase-lsb.wav) |

These recordings use the same divisors as their unperturbed Mk I recordings
above. At gain 0.03125, the one-bit perturbation produces identical output in all
three cases; at gain 0.25 all relative differences remain below 0.0021%.
Thus the high-gain loops amplify a tiny initial-state difference even with
identical arithmetic. This demonstrates sensitivity, but does not fully explain
the larger enge differences or prove that enge's core is otherwise correct.

## Reproduction

Download `source/engine/EngineMkI.cpp` and `.h`, plus `source/msfa/` files
`synth.h`, `aligned_buf.h`, `fm_op_kernel.h`, `fm_op_kernel.cc`, `controllers.h`,
`fm_core.h`, `fm_core.cc`, `sin.h`, `sin.cc`, `exp2.h` and `exp2.cc` from the pinned
revision. Place the engine files in a temporary directory and the others in its
`msfa` subdirectory. In that temporary `controllers.h` only, omit the unused
`#include "../debugger/Logger.h"` to avoid pulling in the JUCE application logger.
No DSP code is changed. Set `DEXED_CORE` to that directory and compile:

```sh
c++ -std=c++17 -O2 -fwrapv -I"$DEXED_CORE" render-dexed-mki.cc \
  "$DEXED_CORE/EngineMkI.cpp" "$DEXED_CORE/msfa/fm_core.cc" \
  "$DEXED_CORE/msfa/fm_op_kernel.cc" "$DEXED_CORE/msfa/sin.cc" \
  "$DEXED_CORE/msfa/exp2.cc" -o /tmp/render-dexed-mki
/tmp/render-dexed-mki 4 7 14 0 /tmp/mki-alg4.i32
/tmp/render-dexed-mki 4 7 14 1 /tmp/mki-alg4-phase-lsb.i32
```

`-fwrapv` specifies the signed integer wrap used by the reference phase
accumulators. The output is 48,000 native-endian signed 32-bit samples; this run
used little-endian macOS. The harness was also checked with `-Wall -Wextra
-Werror -Wno-unused-parameter -fsyntax-only`. Reference source licenses remain
with Dexed; the repository contains the harness and generated measurements,
not copied reference DSP source.

## Integer-kernel graph isolation

The [graph-isolation harness](compare-dexed-mki-graph.cc) executes the exported
[uFor topology data](dx7-feedback-graphs.txt) using the pinned Mk I `mkiSin`
function, then compares it with Mk I's own algorithm dispatcher and feedback
kernels. It includes `EngineMkI.cpp` directly to reuse that exact integer kernel;
there is no independently implemented sine or gain lookup in this control.
The graph data comes from uFor revision `829ba12dbc926489101aa5376a10a5e0803c3914`.
Each row contains algorithm number, feedback source/destination, carrier count
and carrier numbers, followed by edge count and source/destination pairs.

All 32 algorithms run at feedback 0, 1, 4 and 7 with four profiles:

| Profile | Constant log levels | Q24 increments | Initial Q24 phases |
| --- | --- | --- | --- |
| 0 | all 9 | all 153791 | all zero |
| 1 | all 12 | all 153791 | all zero |
| 2 | all 14 | all 153791 | all zero |
| 3 | OP1 through OP6: 9 through 14 | `153791 * operator_number` | `123456 * (operator_number - 1)` |

The asymmetric profile separates operator order, gain and frequency. Every case
renders 48,000 frames. Graph execution sums the exported modulation edges,
reads the two preceding samples from the feedback source, shifts their sum into
the feedback destination with the approved strength rule, and sums the exported
carriers. All operators in these DX7 graphs have feed-forward edges toward lower
operator numbers, so descending operator order is valid. Each case resets phases,
operator gains and both feedback histories. No amplitude envelopes are advanced.

**All 512 cases agree exactly: zero mismatches across 24,576,000 integer samples.**
The [complete CSV](dexed-mki-graph.csv) records mismatch count, maximum absolute
Q24 error and relative RMS error for each case. The retained unit-gain,
feedback-7 graph WAVs are byte-identical to the corresponding Mk I WAVs above:
[algorithm 4](MKI-ALG04-FB7-integer-graph.wav),
[algorithm 6](MKI-ALG06-FB7-integer-graph.wav) and
[algorithm 32](MKI-ALG32-FB7-integer-graph.wav).

This establishes agreement of the exported graph, operator ordering, carrier
sum and feedback scheduling with the pinned dispatcher under these fixed-gain
conditions. It does not establish an independent implementation of the integer
kernel, dynamic-gain behavior, the production floating renderer's equivalence,
complete patch rendering or hardware accuracy. It supports replacing the
floating core as the next model decision, rather than another topology change.

Build with the same temporary reference source directory described above. Since
the harness includes `EngineMkI.cpp`, do not also list that file as a separate
translation unit:

```sh
c++ -std=c++17 -O2 -fwrapv -I"$DEXED_CORE" compare-dexed-mki-graph.cc \
  "$DEXED_CORE/msfa/fm_core.cc" "$DEXED_CORE/msfa/fm_op_kernel.cc" \
  "$DEXED_CORE/msfa/sin.cc" "$DEXED_CORE/msfa/exp2.cc" \
  -o /tmp/compare-dexed-mki-graph
/tmp/compare-dexed-mki-graph dx7-feedback-graphs.txt /tmp > /tmp/mki-graph.csv
```

The output directory must exist. The harness also writes `graph-4.i32`,
`graph-6.i32` and `graph-32.i32` there; convert them to WAV using the matching
`wav_divisor` values in `dexed-mki-core.csv`, as in the original comparison.

## Production integer-core comparison

The approved replacement uses independently calculated logarithmic sine and
exponential tables with the reference's float32 intermediate table precision.
Phases and feedback now use Q24 integers. Phase observations select one of 4096
half-bin-centered positions, folding a 1024-entry quarter-wave table. Attenuation
uses 1/1024-octave units and negative samples use one's-complement magnitude.
The near-silence cutoff of 16,284 attenuation units runs per frame. Frequency
increments round to the nearest Q24 unit, while caller-supplied initial phases
truncate onto that grid. Snapshot and idle-slot state use the same integer units.

The [new oracle](../../test/data/dx7-mki-core.npz) contains 176 one-second signed
Q24 reference recordings. Its first 72 cases reproduce the constant-gain matrix
above. Append algorithms 1 through 32, feedback 0 and 7, and log levels 9 and 14,
omitting triples already present. The `parameters` array records those triples
and `case0` through `case175` retain all 48,000 samples per case. Regenerate each
case with the unchanged `render-dexed-mki.cc` harness and phase perturbation zero,
then save the native signed 32-bit samples in a compressed NumPy archive.

`test/test_dx7_integer.py` compares the production note renderer directly with
these independent samples and checks JSON restoration at frame 17,003. Each
case writes paired 48 kHz WAVs and also requires exact numerical equality.
All 176 cases pass: 8,448,000 samples match exactly, including algorithms 4 and 6
at unit gain and maximum feedback. The full suite passed 2,365 tests with 21
skipped after this replacement.
The upstream `EngineMkI.cpp` Git blob was reverified as
`75b0d33720ca10fcb875e443adf2724523187908` at the pinned revision.

The existing amplitude envelopes and Hexter-scaled controller model still
advance per frame. Their attenuation is quantized before oscillator evaluation;
the implementation does not reproduce Dexed's 64-frame gain interpolation or its
block-wise silence admission. Dynamic controller regressions therefore check
the approved per-frame model, not complete Dexed patch equivalence. Integer-core
agreement does not qualify the DAC, native callbacks or physical hardware.

## Dynamic envelope comparison

The [envelope harness](render-dexed-mki-envelope.cc) combines the same pinned
Mk I core with upstream `Env` from `source/msfa/env.cc` and `.h`. It preserves
the reference's 64-frame envelope updates, gain interpolation and silence
admission. Only the unused application logger include is omitted from the
temporary `env.cc`; no DSP code is changed. Add `msfa/env.cc` to the earlier
core harness compile command and use this harness as its translation unit.
Arguments are algorithm, feedback, profile (0 or 1), and output filename.

All six operators use ratio 1, neutral detune, output level 99, no key/velocity
scaling and exact Q24 increment 153791. The harness initializes each envelope
with output level 4064 and rate scaling zero. Profile 0 uses rates 99/99/99/99
and levels 99/99/99/0. Profile 1 changes rate 2 to 50 and level 3 to 50, requiring
a timed equal-level hold before the decay. Both release at frame 24,000 and
produce 48,000 samples at 48 kHz. Enge receives the matching constructed VCED
files `DYN-aa-p.vced` and explicit pitch `153791 * 48000 / 2**24`.

The [initial results](dexed-mki-envelope.csv) use enge `707513d`.
The [hold results](dexed-mki-envelope-holds.csv) add the approved provisional
equal-level timing. Relative RMS compares unscaled Q24/2**24 samples without
alignment or gain fitting; the held window is frames 12,000–23,999. Each retained
WAV pair uses the CSV's common peak divisor and signed 32-bit PCM.

| Algorithm | Feedback | Profile | Initial whole-file error | With holds |
| --- | ---: | --- | ---: | ---: |
| 4 | 7 | Fast | 91.1231% | 91.1231% |
| 4 | 7 | Equal-level hold | 99.2407% | 100.1092% |
| 6 | 7 | Fast | 43.2738% | 43.2738% |
| 6 | 7 | Equal-level hold | 99.3228% | 72.0716% |
| 32 | 0 | Fast | 5.1516% | 5.1516% |
| 32 | 0 | Equal-level hold | 98.3994% | 42.6478% |

These remain failing full-note qualification results. With feedback disabled,
the held window matches exactly, but onset/release differs. High-feedback loops
also retain waveform differences after differing attacks; constant-gain equality
does not imply equality after a different excitation history. The hold model
fixes missing behavior, not complete agreement with this reference.

### Per-frame rate correction

The tables above predate the moving-stage rate correction. Pinned `Env::advance`
includes `LG_N=6` in its increment because `getsample()` advances once per
64-frame block. Enge previously applied that increment on every output frame,
making moving stages approximately 64 times too fast. It now omits the block
factor before sample-rate scaling and rounds to the nearest per-frame integer.

A standalone trace of the pinned, otherwise unchanged `Env` uses output level
4064, no rate scaling, rates 99/99/99/50 and levels 99/99/99/0. After 750 held
updates at 48 kHz, key-up produces logarithmic levels 251417396, 251176552 and
250935708 at release frames 64, 128 and 192. The decrement is 240844 per block,
or about 3763.1875 per frame. Enge uses 3763; the unscaled 44.1 kHz increment is
4096 per frame, and 96 kHz rounds to 1882. Regression tests check a full second
of release at each rate and write the 48 kHz result to WAV.

This corrects the time units without adopting block gain ramps. Attack
integration granularity, rounding, hold timing and gain interpolation still
differ from Dexed. The [corrected-rate results](dexed-mki-envelope-rates.csv)
repeat the same six full-note comparisons:

| Algorithm | Feedback | Profile | Corrected whole-file error |
| --- | ---: | --- | ---: |
| 4 | 7 | Fast | 91.2675% |
| 4 | 7 | Equal-level hold | 101.4955% |
| 6 | 7 | Fast | 42.7878% |
| 6 | 7 | Equal-level hold | 72.0260% |
| 32 | 0 | Fast | 3.1404% |
| 32 | 0 | Equal-level hold | 43.0782% |

The unmodulated fast-note error falls from 5.1516% to 3.1404%; the other cases
remain divergent. Correct time units do not establish complete reference or
hardware agreement. Corrected carrier WAVs are retained for
[fast](DYN-32-0-enge-rates.wav) and
[equal-level hold](DYN-32-1-enge-rates.wav) profiles.

The user approved interpreting Dexed's measured hold table as 44.1 kHz durations
and preserving seconds across output rates. Enge rounds up to the next frame,
uses the provisional additive rate-scaling rule, and retains the reference's
high-rate approximation and faster zero-level attack rule. Thus rate 50 holds
for 150 ms at both 44.1 and 48 kHz. Dexed's current inverse sample-rate multiplier
instead shortens its 48 kHz hold, so exact agreement is not the adopted contract.
Hold countdowns survive snapshots; release interrupts a pending hold. Sustain
does not consume a release-stage hold until key-up. Measurements remain
provisional and do not establish hardware envelope timing.

Listen to the unmodulated carrier cases:

- Fast: [Mk I](DYN-32-0-mki.wav), [enge](DYN-32-0-enge-holds.wav).
- Equal-level hold: [Mk I](DYN-32-1-mki.wav),
  [enge before holds](DYN-32-1-enge.wav), [enge with holds](DYN-32-1-enge-holds.wav).

## Isolating reference envelope behavior

These controls use enge `ce019b2` and the same pinned Dexed sources, patch bytes,
exact frequency increment, 48 kHz rate, release frame and one-second duration
as the dynamic comparison above. The upstream `EngineMkI.cpp` blob was checked
against GitHub again: `75b0d33720ca10fcb875e443adf2724523187908`. The reference's
`env.cc` differs from upstream only by omitting the unused application logger.
Production enge code is unchanged by these experiments.

Apply each diagnostic patch to its own prepared reference copy with
`git apply --unidiff-zero /absolute/path/to/the.patch` from that copy's root.
The patches omit context lines to avoid importing upstream trailing whitespace;
they target the exact pinned files, not arbitrary Dexed versions.

### Inner feedback-operator ramps

In `compute_fb2` and `compute_fb3`, the pinned reference overwrites each inner
operator's `gain_out` before reading its previous value. For the nonzero target
attenuations in these profiles, that operator consequently jumps to the new
block attenuation and has zero interpolation increment. OP6 still interpolates.
The ordinary `compute`/`compute_pure` kernels instead interpolate from the
previous attenuation using `(new - old + 32) >> 6`.

The retained [experimental patch](dexed-mki-feedback-ramp.patch) makes only
those inner operators use the ordinary interpolation rule. It is a diagnostic
control, not an adopted reference correction or hardware claim. Build the
existing `render-dexed-mki-envelope.cc` harness once against the original core
and once against an isolated copy with this patch applied. All envelopes,
tables, phase arithmetic and feedback strengths stay unchanged.

The [complete results](dexed-mki-feedback-ramp.csv) include algorithms 4 and 6
with feedback off and at seven, plus algorithm 32 with feedback off, each with
both dynamic profiles. All six feedback-off controls are exactly unchanged.

| Algorithm | Profile | Changed reference samples | Reference-to-control RMS difference | enge vs pinned | enge vs ramp control |
| --- | --- | ---: | ---: | ---: | ---: |
| 4 | Fast | 504 | 10.0001% | 91.2675% | 91.2358% |
| 4 | Hold | 514 | 19.8750% | 101.4955% | 101.3775% |
| 6 | Fast | 8,231 | 33.1432% | 42.7878% | 42.9870% |
| 6 | Hold | 6,281 | 56.9853% | 72.0260% | 72.9480% |

Every changed case first differs at frame zero. The reference change materially
alters the waveform but does not resolve enge's mismatch. Listen to the algorithm-4 fast
case: [pinned core](DYN-04-feedback-pinned.wav) and
[ramp control](DYN-04-feedback-ramp.wav). Both WAVs divide Q24/2**24 samples by
the same peak, `3.998046875`, before rounding to signed 32-bit PCM. No alignment
or independently fitted gain is used.

### Hold sample-rate scaling

A separate [one-line reference patch](dexed-mki-hold-scaling.patch) changes only
the hold conversion from multiplication by the inverse sample-rate ratio to
division by that ratio, rounding upward. The reference retains its Q24 rate
multiplier and 64-frame countdown granularity, so this is not an exact copy of
enge's per-frame countdown. The original Mk I core, including the inner
feedback behavior above, remains unchanged in this control. Apply this patch
to a separate copy of `msfa/env.cc` and compile the same envelope harness.

All three fast-profile controls remain exactly unchanged. The hold profiles
each change 1,278 reference samples. The
[complete results](dexed-mki-hold-scaling.csv) show:

| Algorithm | Feedback | enge vs pinned hold profile | enge vs stable-duration control |
| --- | ---: | ---: | ---: |
| 4 | 7 | 101.4955% | 91.5254% |
| 6 | 7 | 72.0260% | 57.1671% |
| 32 | 0 | 43.0782% | 5.7693% |

Thus the intentional stable-duration choice accounts for most of the
unmodulated hold-profile discrepancy. It does not explain the remaining
attack/release differences or high-feedback excitation histories. Reversing
enge's approved timing to reduce error against the unmodified reference would
violate the intended duration contract.

Across both experiments, 16 one-second cases were compared, including nine
unchanged negative controls. These results narrow the remaining qualification
work to envelope trajectories, block attenuation interpolation and silence
admission, followed by modulation integration. Neither modified reference is
adopted as hardware truth, and neither comparison passes whole-note
qualification. Do not replace the failing full-note fixtures with these controls.

## Matched per-frame attenuation trajectories

The next control removes envelope evolution and block interpolation from the
comparison by supplying identical per-frame attenuation to both cores. It uses
the unchanged production `DX7Renderer` from enge `51fca69`, with a separate copy
of its public amplitude-envelope state to capture the inputs ahead of rendering.
No reference samples come from the enge oscillator implementation.

For each of 48,000 frames, the capture advances the six envelope copies and
stores `16384 - (level >> 14)` in OP1 through OP6 order. Key-up occurs before
frame 24,000. The note renderer independently processes the original patch,
including its envelopes. The
[reference harness](render-dexed-mki-trajectory.cc) consumes those captured rows
and calls the pinned upstream `mkiSin` kernel with an independently traversed
exported graph and feedback history. Its phase increment is exactly 153791 Q24
units per frame, as in the prior comparisons. Compile with signed wrap enabled.

Both paths use enge's existing per-frame silence policy: attenuation above
16284 yields zero. That policy is an explicit controlled input, not something
this experiment qualifies against Dexed's block admission or hardware. The
reference graph scheduler was separately checked against the complete Mk I
block engine for constant gains above; this experiment exercises its dynamic
inputs without adopting the block engine's interpolation.

Algorithms 4 and 6 use feedback seven; algorithm 32 uses feedback zero. Each
runs three profiles:

- Profile 0 is the prior fast envelope.
- Profile 1 is the prior equal-level hold envelope.
- Profile 2 gives the six operators distinct moving envelopes. For slot index
  `i = 0..5` in VCED order OP6 through OP1, rates are
  `[70 + 4*i, 30 + 7*i, 40 + 5*i, 65 + 5*i]` and levels are
  `[99, 80 - 5*i, 30 + 7*i, 0]`. The authored bytes are retained as
  `DYN-aa-trajectory-2.vced`.

All **432,000 samples match exactly** in the
[complete results](dexed-mki-matched-trajectory.csv). A sensitivity control
increases only OP6's attenuation by one 1/1024-octave unit for frames
1,000–1,255. Every case changes: mismatch counts range from 223 to 22,962.
Thus this comparison is sensitive to small trajectory differences, including
their continuing effect on high-feedback histories.

The captured inputs, patch bytes, case parameters and independent reference
outputs are retained in `test/data/dx7-mki-dynamic.npz`, SHA-256
`2326047c946991aa3115d00a6acb1b3e7432eb7c0bb3183b5acf654fad1ab3ee`.
Arrays `attenuation0` through `attenuation2` contain little-endian signed-32-bit
OP1..OP6 rows; `parameters` lists algorithm, feedback and profile; `patchN`
contains VCED bytes; `caseN` contains raw signed Q24 reference samples.
Export a profile array unchanged as native little-endian `int32_t` rows, then
run the retained harness against the pinned reference directory:

```sh
c++ -std=c++17 -O2 -fwrapv -I"$DEXED_CORE" render-dexed-mki-trajectory.cc \
  "$DEXED_CORE/msfa/fm_core.cc" "$DEXED_CORE/msfa/fm_op_kernel.cc" \
  "$DEXED_CORE/msfa/sin.cc" "$DEXED_CORE/msfa/exp2.cc" \
  -o /tmp/render-dexed-mki-trajectory
/tmp/render-dexed-mki-trajectory dx7-feedback-graphs.txt 4 7 \
  /tmp/attenuation0.i32 /tmp/trajectory-output.i32
```

The harness requires exactly 48,000 rows and rejects truncated or extra input.
The nine production regressions render one-second paired 48 kHz WAVs and require
exact unscaled sample equality. They also replay JSON snapshots from frame
17,003 through key-up and release. Changing the intentional envelope model may
require recapturing inputs and rerunning the reference; these captured rows are
not an independent hardware-envelope oracle.

This closes the tested dynamic-core arithmetic comparison under matched inputs.
It does not close the original full-note comparisons: envelope trajectories,
block interpolation, silence admission, controller modulation and hardware
qualification remain separate. Do not interpret the exact result as evidence
that the original 43–101% full-note discrepancies have disappeared.
