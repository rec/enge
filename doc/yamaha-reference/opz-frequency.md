# OPZ native frequency audit

Measured on 2026-11-04 against unmodified ymfm revision
`81aec25ccbb98f4873a255f7551ac4dadac59b4a`. The Git blob hash of
`src/ymfm_opz.cpp` is `949061baf4716b5d9932650685ebedcd2726e341`.
[The harness](check-ymfm-frequency.cc) records exact accumulated phase steps in
[the CSV](opz-frequency.csv). These are register arithmetic checks, not audio
or measured hardware comparisons.

With the pinned source files in `$YMFM`, reproduce from the repository root:

```sh
mkdir -p build
c++ -std=c++17 -O2 -I"$YMFM" doc/yamaha-reference/check-ymfm-frequency.cc \
  "$YMFM/ymfm_opz.cpp" -o build/enge-check-ymfm-frequency
build/enge-check-ymfm-frequency > build/opz-frequency.csv
diff doc/yamaha-reference/opz-frequency.csv build/opz-frequency.csv
```

Each case accumulates 4,096 native ticks, covering the fixed-frequency carry
period. Frequency is the accumulated increment times 3,579,545 Hz, divided by
64 clocks per tick, 2^20 phase units per cycle, and the tick count. The core's
`phase()` shifts the accumulator by ten bits before its ten-bit waveform lookup.

| Cases | Count | Difference from nominal frequency |
| --- | ---: | ---: |
| Ratio multiplier 1, MIDI keys 13–108, zero fine and detune | 96 | −4.197910 to +0.188415 cents |
| Legal fixed coarse/fine register combinations, all eight ranges | 1,984 | −11,999.803828 cents |

Ratio key 69 produces 439.943181872368 Hz against nominal 440 Hz. This documents
the measured native ratio table's quantization; it does not qualify enge's
continuous nominal equations as equivalent to that table.

The fixed path adds `75 * frequency` to a substep accumulator and shifts by
12 to produce phase increments. Its resulting frequency is approximately
1/1,024 of nominal: requested 32,640 Hz produces 31.878612062428 Hz. The source
comment's approximately 8 Hz minimum is inconsistent with those accumulated
steps. This audit neither patches that reference nor adopts a guessed correction.
Enge retains the nominal fixed-frequency equations pending a sound native oracle.

The valid control domain is independently supported by
[TX81Z Programmer](https://github.com/mgregory22/tx81z-programmer/blob/08dd21c3e15bd664e82cb1c48685135a475c2978/src/dlg/voicedlg.c):
`vd_CheckFreqSliderRange` limits fine to seven for VCED coarse below four in
both modes, and `vd_UpdateFrequencyDisplay` applies that limit before choosing
fixed or ratio display. `VoiceDlg_FixedParamsToFreq` agrees with enge on valid
fixed controls, as does
[DXconvert](https://github.com/richotero/dxconvert/blob/27a3c6dbfa97d57dfee0b2bcdbe32ed7bc98d8fa/DXconvert/fourop.py).
Enge now rejects the unavailable fixed combinations as it already did for
ratio mode; uFor still preserves their original bytes. Boundary regressions
cover fine seven at low coarse and fine eight at the next coarse range.

## Complete legal ratio-control sweep

The [ratio harness](check-ymfm-ratio.cc) expands the check to all 992 legal
VCED coarse/ACED fine pairs at each of the 96 wrapped keys, with chip detune
and modulation disabled. It writes [64 aggregate rows](opz-ratio.csv), one
per VCED coarse value. Each row retains the tested count, extrema, worst key
and fine setting, and top-key frequencies at fine zero. No audio is generated.

```sh
mkdir -p build
c++ -std=c++17 -O2 -I"$YMFM" doc/yamaha-reference/check-ymfm-ratio.cc \
  "$YMFM/ymfm_opz.cpp" -o build/enge-check-ymfm-ratio
build/enge-check-ymfm-ratio > build/opz-ratio.csv
diff doc/yamaha-reference/opz-ratio.csv build/opz-ratio.csv
```

The harness builds the 64-value displayed-ratio ordering from the four
documented coarse-detune families and 16 multipliers, matching enge's nominal
model. It writes those controls to the reference registers, compares cached
and dynamically computed phase steps (exit 2 on disagreement), and compares
native Hz with the continuous equations. Fine increments enter the multiplier
before its final integer shift. All 95,232 cases agree between the reference's
cached and dynamic paths.

Of those cases, 5,456 hit the native top-octave clamp. The reference applies
coarse detune in 1/64-semitone units, adjusts the octave, and saturates its base
phase step at the final table entry before multiplication. Enge's continuous
formula has no corresponding clamp. Across the full sweep, native minus
nominal error spans −853.530124156 to +0.221538103 cents. Without clamped cases
the span is −10.036815924 to +0.221538103 cents; the low-key half multiplier
and fine-step rounding account for the larger negative extreme.

For example, key 108 with coarse 3 (half multiplier and 950-cent coarse detune),
fine zero, is nominally 3,623.143872985092 Hz but the reference yields
2,212.944074869156 Hz. At coarse zero, key 13, fine three, the native result is
10.036815924 cents below nominal. These differences preclude treating the
nominal equations as a qualified native frequency implementation.

The next implementation must preserve the native table, octave saturation,
detune addition and multiplier rounding order together. A fitted correction
factor or a clamp on the final multiplied frequency would not implement those
rules. This sweep does not yet cover nonzero chip detune, key fractions,
modulation, tuning, portamento or output-rate conversion. It does not establish
hardware equivalence or resolve the independent fixed-mode reference defect.

## Production ratio replacement

Enge now uses the pinned 768-entry native phase table, in
`enge/yamaha_phase.npy`, for ratio operators. Coarse detune and integer PM
adjust the table position before octave saturation. Chip detune is added to
that phase step before the multiplier/fine product is shifted right four bits.
The helper `tx81z_base_frequency` returns native ratio Hz before chip detune;
fixed-mode values retain their nominal equations. Keys outside 13–108 wrap
by octaves into the supported span, as in the note renderer.

Microtuning, scoped tuning and portamento keep the approved continuous host
overlay. They scale the neutral-detune frequency; the difference between
detuned and neutral native steps remains unscaled. PM recomputes both steps
through the native table, including saturation and rounding. This also keeps
PM correct after legato retargeting and snapshot restoration. It does not
claim firmware-quantized tuning or portamento.

The [fixture exporter](export-ymfm-frequency.cc) builds unmodified register
caches for all keys, coarse/fine pairs and seven signed detune settings. Invalid
fine settings occupy zero placeholders. A second array uses the pinned native
key-code helper for multiplier-one PM deltas −512 through 511, with signed
detune. Reproduce the raw little-endian uint32 exports with:

```sh
mkdir -p build
c++ -std=c++17 -O2 -I"$YMFM" doc/yamaha-reference/export-ymfm-frequency.cc \
  "$YMFM/ymfm_opz.cpp" -o build/enge-export-ymfm-frequency
build/enge-export-ymfm-frequency build/base.u32 build/pm.u32
```

Pack these arrays with NumPy, preserving the loop order:

```python
import numpy as np

base = np.fromfile("build/base.u32", dtype="<u4").reshape(96, 64, 16, 7)
pm = np.fromfile("build/pm.u32", dtype="<u4").reshape(96, 7, 1024)
np.savez_compressed("build/tx81z-frequency.npz", base=base, pm=pm)
```

The committed `test/data/tx81z-frequency.npz` SHA-256 is
`1e4f1f77c6c67534ffd76bfb24a097b8b66950fe0589bff9f56c238ca4cd8fd1`.
The production table is the pinned `s_phase_step[12*64]` values saved as a
one-dimensional little-endian int64 NumPy array; its `.npy` SHA-256 is
`db421ae93bb06c307dfe59edab9eb8d9c6d37b1e4289a18b2683afc1b4859e89`.
The upstream BSD notice is retained in the repository license.

All 95,232 legal neutral-detune base frequencies compare exactly against the
register export. One-second 48 kHz WAV tests additionally cover signed detune
before fine-multiplier rounding, low-key quantization, top-octave saturation,
PM at both key boundaries, and partitioned snapshot continuation. Existing
amplitude/controller tests now use exported native frequencies rather than
ideal equal-tempered frequencies in their audio expectations. Fixed-frequency
reference qualification remains open. The subsequent [native note comparison](opz-note.md)
qualifies the public clock and DAC path for two ratio-mode patches across all
algorithms and feedback levels.
