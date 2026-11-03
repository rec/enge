# Initial DX7 reference comparison

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
a target decision. Hardware qualification, nonzero feedback, modulation,
other algorithms and arbitrary patches remain unverified.

## Reproducing the reference

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
