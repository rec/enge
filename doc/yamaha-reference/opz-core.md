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
/tmp/ymfm-core 8 0 0 /tmp/opz-carriers.i32 tx81z-graphs.txt
```

Arguments are one-based algorithm, feedback, profile, output path and exported
graph path. Output is
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
combined differences. The shared-integer-kernel control below subsequently
isolates graph and feedback scheduling from operator arithmetic.

Paired one-second 48 kHz, signed 32-bit PCM listening files share the CSV's
`wav_divisor` within each pair, without independent normalization:

- Unmodulated carriers: [enge](TX-CORE-carriers-enge.wav),
  [ymfm](TX-CORE-carriers-ymfm.wav).
- Worst case: [enge](TX-CORE-worst-enge.wav), [ymfm](TX-CORE-worst-ymfm.wav).

No result here qualifies complete patches, firmware controller behavior,
WT11 performance mode, native-rate output, DAC behavior or hardware equivalence.
The upstream OPZ model itself remains provisional.

## Shared integer-kernel graph control

The harness now also evaluates enge's exported graph with the same upstream
integer `compute_volume` kernel and operator phase state. It traverses enge's
OP4..OP1 order, sums incoming edges, shifts their integer modulation sum right
by one, and maintains its own two-sample feedback history. The return path
uses `(previous + older) >> (10 - feedback)` when enabled. Carrier selection
comes from the exported graph, independently of ymfm's algorithm table.

The [graph export](tx81z-graphs.txt) is generated directly from the literal
`_TX_ALGORITHMS` in `enge/yamaha.py` at `ab8a9c3`. Each of its eight lines contains
five decimal masks: incoming sources for OP1, OP2, OP3 and OP4, then carriers.
Bit zero denotes OP1. For example, algorithm 1 is `2 4 8 0 1`, representing
OP4 to OP3 to OP2 to OP1, with OP1 as the carrier. The export was regenerated
from the source AST and checked against the committed graph data.

The [full control results](opz-graph-comparison.csv) show **zero mismatches in
6,144,000 samples across all 128 cases**. The harness reports one CSV row to
stderr and exits with status 6 if any sample differs. A negative control removed
algorithm 1's OP2-to-OP1 edge: the uniform, feedback-zero case then produced
46,406 mismatches and exited with status 6. This verifies that the graph control
can detect an actual topology change.

This supports the exported topology, carrier selection and two-sample feedback
scheduling under fixed gains and shared integer kernels. It does not establish
equivalence of the production floating kernel, dynamic gain processing, native
clock conversion or DAC output. The large floating-core differences above
remain unresolved production behavior. The next implementation step is to
replace the continuous operator arithmetic with the provisional integer OPZ
model, retaining these graphs and qualifying the resulting production core.
