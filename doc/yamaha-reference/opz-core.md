# TX81Z fixed-gain OPZ core comparison

The initial comparison below is a failing qualification result. The subsequent
production integer replacement passes the fixed-gain cases, as described at
the end. Initially, enge
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

From the repository root, build with `YMFM` pointing to the pinned source directory:

```sh
mkdir -p build
c++ -std=c++17 -O2 -I"$YMFM" doc/yamaha-reference/render-ymfm-core.cc "$YMFM/ymfm_opz.cpp" -o build/ymfm-core
build/ymfm-core 8 0 0 build/opz-carriers.i32 doc/yamaha-reference/tx81z-graphs.txt
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
motivated the production replacement described next.

## Production integer replacement

The production `tx81z_operator_samples` now uses integer Q20 phase state,
nearest-integer Q20 frequency increments, 10-bit waveform indices, 1/64-octave
gain attenuation and signed integer feedback. Incoming modulation sums shift
right by one before waveform lookup; feedback uses the same two-sample shift
as the control above. The raw carrier sum is divided by 8192. No DAC clipping
or native-clock resampling is introduced.

Gains must lie in 0..1 and round to the nearest 1/64-octave attenuation;
zero produces silence. Derived patch gains already use the chip's attenuation
units. Standalone arbitrary gains are now quantized instead of multiplying
continuous waveform values. Frequency increments round to the Q20 grid at the
requested output sample rate. This output-frame model is not a claim of native
chip frequency generation. Snapshots retain integer phases and feedback; the
former floating continuation representation is replaced without migration.

Independent mathematical reconstruction matches the pinned logarithmic sine
and power tables. The [table export harness](render-ymfm-tables.cc) writes 8192
unsigned 16-bit waveform entries (eight consecutive 1024-entry tables), followed
by 256 unsigned 16-bit unshifted power values. Build it with the same source and
flags as the core harness. Its output and all 128 captured core outputs are
retained in `test/data/tx81z-opz-core.npz`, SHA-256
`0217c8aa8bee97929897a6d9d3c054c6a4c0fa15fe9156fe90e041dca318dd8e`.
The archive's `parameters` rows are algorithm/feedback/profile; `case0` through
`case127` contain signed reference output. `waveforms` and `power` hold the
independent table captures.

All **6,144,000 production samples now match exactly**. Each case also passes
partitioned execution and JSON snapshot replay. The eight waveform regressions
cover every table cell. Existing dynamic-control and lifecycle regressions now
use captured integer-table carrier oracles where they previously assumed ideal
sines or multiplication after waveform quantization.

This closes the fixed-gain software-core comparison, not complete dynamic patch,
controller, native-clock, DAC, WT11 or hardware qualification. The earlier CSVs
and listening files remain historical measurements of the removed floating core.

## Dynamic envelope integration

The [dynamic harness](render-ymfm-dynamic.cc) combines upstream envelopes and
integer channel output. It sets exact Q20 increment 9600 for every operator,
observes output at 48 kHz and advances envelopes with the same rational
`3579545 / (192 * 48000)` carry as enge. Output-frame phase and feedback clocks
are separated from envelope ticks through the public operator cache and clock
APIs. Key-up occurs before frame 24000. This deliberately qualifies enge's
output-frame model, not native chip-rate resampling.
After the native ratio-frequency replacement, the former Python test adjusted its host
pitch ratio by `440 / native_A4` to retain that exact injected phase increment.
The captured reference audio remains unchanged. This controlled stimulus keeps
the envelope/core comparison independent of the separate frequency audit.

The 32 cases cover eight algorithms, feedback 0/7 and two profiles. Both use
MIDI key 69, velocity 127, ratio 1, neutral detune, output level 99 and rate
scaling zero. Carrier headroom follows the same measured TL offsets used by
patch preparation. VCED/ACED are constructed explicitly in the Python test:

| Profile | AR/D1R/D2R/RR/D1L | Waveforms, OP1..OP4 | Shifts, OP1..OP4 |
| --- | --- | --- | --- |
| 0 | 31/12/4/9/7 | 0, 0, 0, 0 | 0, 0, 0, 0 |
| 1 | 20/16/12/15/0 | 7, 4, 1, 0 | 0, 1, 2, 0 |

An initial attempt enabled shift on OP4 as well. It exposed a reference mapping
conflict: pinned ymfm disables shift for register offsets whose `opoffs & 0x18`
is zero, which is the first channel operator and maps to enge's OP4 in the
four-operator graph. Its comment cites the manual's OP1 restriction. Enge
retains the manual-based OP1 rule. Both endpoint operators are unshifted in
this fixture to isolate the integration; this does **not** qualify OP4 shift
against the complete reference channel or resolve the upstream conflict.

The [before-correction CSV](opz-dynamic-before.csv) uses enge `700261a` and the
profiles above. All 16 unshifted cases and eight shifted-modulator cases already
matched. The eight cases with shifted carriers differed
because ymfm's `compute_volume` returns zero when raw envelope attenuation is
strictly greater than `0x380` (896), before applying envelope shift. Enge had
passed only shifted attenuation to its core. The mixed-waveform algorithm-8,
feedback-zero case differed in 22,619 samples, with 7.3643% relative RMS error.

`TX81ZEnvelopeBank.attenuation_samples` now returns raw, unshifted attenuation.
The note renderer computes the raw quiet mask, applies the per-operator shift,
then combines LFO and total-level attenuation. Quiet operators receive zero
gain. The gate follows the provisional pinned software model; it is not a new
hardware measurement. Existing envelope component regressions explicitly apply
shift when comparing against their captured shifted observations.

After the correction, all **1,536,000 samples match exactly** across the 32
cases. The historical integer reference output remains in
`test/data/tx81z-opz-dynamic.npz`. Public note regressions now use the
[complete native-clock note captures](opz-note.md) for partitioned
attack/release rendering and JSON continuation. Separate WAV regressions distinguish raw
attenuation 896 from 897 and 1023 with shift enabled. The archive's `parameters`
rows are algorithm/feedback/profile; `case0` through `case31` contain upstream
signed integer output before division by 8192.
The archive's SHA-256 is
`eca6142602178e955fcab6dad38590d64d16f28408b4a57e5e9a6fc3b8a02ffa`.

Build as for the fixed-gain harness, substituting `render-ymfm-dynamic.cc`.
Arguments are algorithm, feedback, profile and output path; output is 48,000
host-endian signed 32-bit samples. No graph-file argument is required.
Complete controller-modulated patches, disputed shift mapping, native frequency
preparation, DAC behavior and hardware qualification remain open.
