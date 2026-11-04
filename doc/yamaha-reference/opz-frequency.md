# OPZ native frequency audit

Measured on 2026-11-04 against unmodified ymfm revision
`81aec25ccbb98f4873a255f7551ac4dadac59b4a`. The Git blob hash of
`src/ymfm_opz.cpp` is `949061baf4716b5d9932650685ebedcd2726e341`.
[The harness](check-ymfm-frequency.cc) records exact accumulated phase steps in
[the CSV](opz-frequency.csv). These are register arithmetic checks, not audio
or measured hardware comparisons.

With the pinned source files in `$YMFM`, reproduce from the repository root:

```sh
c++ -std=c++17 -O2 -I"$YMFM" doc/yamaha-reference/check-ymfm-frequency.cc \
  "$YMFM/ymfm_opz.cpp" -o /tmp/enge-check-ymfm-frequency
/tmp/enge-check-ymfm-frequency > /tmp/opz-frequency.csv
diff doc/yamaha-reference/opz-frequency.csv /tmp/opz-frequency.csv
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
