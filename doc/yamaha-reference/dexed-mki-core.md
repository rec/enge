# Dexed Mk I constant-gain core comparison

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
