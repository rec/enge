# TX81Z fixed-gain OPZ core comparison

This is a failing qualification result, not a passing audio regression. Enge
`566a63f` is compared with ymfm revision
`81aec25ccbb98f4873a255f7551ac4dadac59b4a` using the
[standalone harness](render-ymfm-core.cc). All five source files used by the
harness were checked against that revision's Git blob hashes on 2026-11-04:

| File | Git blob |
| --- | --- |
| `ymfm_opz.cpp` | `949061baf4716b5d9932650685ebedcd2726e341` |
| `ymfm_opz.h` | `997ba32f9f2e67595e888e7f497cd2710000e6a5` |
| `ymfm_fm.ipp` | `675424839ce9cf3c589e550e7f638effbfe5483c` |
| `ymfm_fm.h` | `d40409fddfd62bdd7232347de5a318bd75c97f92` |
| `ymfm.h` | `6427e34cda2c94f013443a49647119b99ff861cd` |

## Method

The harness uses the upstream public operator and channel APIs without changing
reference source. Maximum attack initializes attenuation to zero; the harness
checks that state, freezes envelope rates, and sets cached total levels and
phase increments. No LFO, patch decoding, dynamic controllers, native clock
resampling or DAC processing is included. All phases and feedback histories
start at zero. Output precedes each phase/feedback clock, matching enge's order.
Each case contains 48,000 frames interpreted at 48 kHz.

The sweep covers algorithms 1–8, feedback 0–7 and two profiles, for 128 cases
and 6,144,000 compared samples. Values below use enge's OP1..OP4 order; upstream
channel operators use the reverse order.

| Profile | Q20 phase increments | TL values | Waveform indices |
| --- | --- | --- | --- |
| 0, uniform | 9600, 9600, 9600, 9600 | 0, 0, 0, 0 | 0, 0, 0, 0 |
| 1, asymmetric | 24000, 19200, 14400, 9600 | 48, 32, 16, 0 | 7, 4, 1, 0 |

Enge's `tx81z_operator_samples` receives constant frequency rows
`increment * 48000 / 2**20` and gain rows `2**(-TL / 8)`, starting with a default
`TX81ZState`. These phase increments are exactly representable in binary,
avoiding nominal-frequency or native-clock differences. The reference uses
`output_4op` with zero output shift and clip limit 32767, which accommodates the
four-carrier sum. Its signed integer output is divided by 8192 for comparison.
Enge retains continuous phases, waveform evaluation, gains and feedback.

Build with `YMFM` pointing to the pinned source directory:

```sh
c++ -std=c++17 -O2 -I"$YMFM" render-ymfm-core.cc "$YMFM/ymfm_opz.cpp" -o /tmp/ymfm-core
/tmp/ymfm-core 8 0 0 /tmp/opz-carriers.i32
```

Arguments are one-based algorithm, feedback, profile and output path. Output is
48,000 host-endian signed 32-bit integers. The recorded run used macOS arm64
little-endian output. Metrics compare raw normalized samples without time
alignment or fitted gain. Relative RMS is the error norm divided by reference
norm; all reference norms are nonzero.

## Results

The [complete CSV](opz-core-comparison.csv) records every case. Selected results:

| Algorithm | Uniform, feedback 0 | Uniform, feedback 7 | Asymmetric, feedback 0 | Asymmetric, feedback 7 |
| --- | ---: | ---: | ---: | ---: |
| 1 | 134.3497% | 139.1521% | 52.0522% | 89.8425% |
| 2 | 131.9464% | 141.1251% | 21.1998% | 90.5287% |
| 3 | 85.8901% | 140.5135% | 14.6041% | 115.7502% |
| 4 | 85.8901% | 140.6395% | 56.7324% | 114.6200% |
| 5 | 5.5096% | 96.1413% | 6.9210% | 139.9104% |
| 6 | 5.5096% | 139.1978% | 7.2813% | 140.0222% |
| 7 | 2.4321% | 58.9823% | 6.7979% | 137.1002% |
| 8 | 0.3384% | 43.5601% | 0.3416% | 135.8016% |

The worst case is uniform algorithm 2, feedback 7, at 141.1251% relative RMS.
The unmodulated algorithm-8 control is consistent with the earlier isolated
waveform-table discrepancy. Deep modulation and feedback greatly amplify the
combined differences, but this does not isolate quantization from graph or
feedback scheduling. A shared-integer-kernel graph comparison is the next
useful control before replacing the floating TX81Z core.

Paired one-second 48 kHz, signed 32-bit PCM listening files share the CSV's
`wav_divisor` within each pair, without independent normalization:

- Unmodulated carriers: [enge](TX-CORE-carriers-enge.wav),
  [ymfm](TX-CORE-carriers-ymfm.wav).
- Worst case: [enge](TX-CORE-worst-enge.wav), [ymfm](TX-CORE-worst-ymfm.wav).

No result here qualifies complete patches, firmware controller behavior,
WT11 performance mode, native-rate output, DAC behavior or hardware equivalence.
The upstream OPZ model itself remains provisional.
