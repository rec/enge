# WT11 pitch-envelope model audit

Audited on 2026-11-05 before connecting the WT11 schema to audio rendering.
WT11 SysEx inspection and portable voice preparation exist in uFor. The direct
note renderer now implements the provisional pitch path below; this audit does
not establish measured WT11 envelope timing or complete instrument support.

## Sources and stage semantics

The [WT11 manual, printed Add-5](https://data.yamaha.com/files/download/other_assets/2/333982/WT11E.PDF)
defines three pitch rates at VCED bytes 87–89 and three pitch levels at 90–92.
All range from 0 through 99, with level 50 centered. It does not give an
absolute rate-to-time curve in that table.

The [DX21 manual, printed page 42](https://usa.yamaha.com/files/download/other_assets/5/320775/DX21E1.pdf)
documents the related three-rate/three-level envelope: a note begins at PL3,
PR1 moves toward PL1, PR2 moves toward PL2, and PL2 holds until release.
PR3 then returns to PL3. It identifies level 50 as normal pitch, levels 0/99
as minus/plus four octaves, and rate 99 as instantaneous. These are DX21
specifications, not independent WT11 measurements.

DXconvert's [four-operator conversion model](https://github.com/richotero/dxconvert/blob/89aaf6cf48ff2eec7e11e82bb92e315b2e6e6621/DXconvert/fourop.py)
provides a concrete provisional projection onto a DX7 pitch envelope. The file
revision is pinned to `89aaf6cf48ff2eec7e11e82bb92e315b2e6e6621` (2025-02-05).
Its `pr` table maps four-operator rates to DX7 rates. `vmm2vmem` prepares:

| DX7 field | Four-operator source |
| --- | --- |
| Rate 1 | `pr[PR1]` |
| Rate 2 | `min(98, pr[PR2])` |
| Rate 3 | 99 |
| Rate 4 | `pr[PR3]` |
| Level 1 | PL1 |
| Level 2 | PL2 |
| Level 3 | PL2 |
| Level 4 | PL3 |

For example, source rates 0, 50 and 99 map to 1, 60 and 99, respectively,
before the second-stage cap. The repeated sustain level permits reuse of the
existing DX7 state machine without introducing a separate envelope executor.

## Limits of that projection

This is a patch-conversion approximation. It is not a measured WT11 firmware
model and must not be presented as passing hardware qualification. In particular:

- enge's DX7 level table maps 0 to -4 octaves but 99 to 127/32 octaves, which
  is 37.5 cents below the DX21 manual's stated +4-octave endpoint. Copying the
  levels therefore does not exactly satisfy that related manual's endpoints.
- The existing DX7 executor advances by a finite increment even at rate 99.
  Reusing it does not implement the DX21 manual's instantaneous-rate statement.
- The DX21 stage description and DXconvert mapping do not prove WT11's
  intermediate level curve, rate curve, firmware update cadence, equal-level
  transitions or interaction with fixed-frequency operators.

## Direct note implementation

`enge.yamaha.WT11Renderer` now accepts explicit WT11 VCED, ACED and ACED2 objects,
note, velocity, sample rate, optional pitch/glide origin and absolute start frame.
It translates the WT11 operator order into the existing TX81Z/OPZ core and adds
the projected pitch envelope through that core's tuning path. The projection
advances once per host frame, reusing `DX7PitchEnvelope`; original rates and
levels remain in the WT11 patch. All 100 legal entries in the new conversion
table were checked against the pinned source table.

Caller tuning cents add to the pitch-envelope cents. Both affect ratio operators
only, preserving the existing fixed-frequency and chip-detune behavior. That
choice is provisional for intrinsic WT11 pitch envelopes. Shared modulation
rows, controller depths, breath, foot and portamento rows use the existing OPZ
processing path. A nonneutral WT11 breath pitch bias rejects positive breath
input; neutral input is allowed, including patches whose stored bias is 100.
The original bias is retained rather than clamped into the TX81Z domain.

`WT11Snapshot` retains the mapped OPZ state, all three original WT11 buffers and
the projected pitch state. Restore checks the original definitions and projected
parameters before changing state; a plain TX81Z snapshot cannot supply the
missing pitch state. Invalid render controls do not advance the pitch envelope.

Active aftertouch assignments are explicitly rejected at note construction.
The admitted ACED2 assignment tuple is pitch 0, amplitude 0, pitch bias 50 and
EG bias 0; reserved bytes remain unrestricted seven-bit data. Aftertouch
rendering, performance execution and hardware qualification remain incomplete.

Thirty-four focused tests cover all eight algorithms with asymmetric operators,
neutral-pitch audio equivalence to the reused core, nonzero absolute start,
mixed fixed/ratio operators, projected early/held release, hold and final pitch,
JSON snapshot replay, partitioned continuation, rejected controls and foreign
snapshots. Sample-rate cases use 24, 48 and 96 kHz with equal release times in
seconds; their paired WAV regressions are written at 48 kHz for at least one
second. The tests establish the projection and state contract, not measured
WT11 timing or complete-note hardware equivalence.

The full enge suite passes with this implementation: 3,082 tests passed and
21 skipped. Ruff, formatting, type checking and pyupgrade checks also pass.

## Offline instrument integration

`enge.yamaha_instrument.prepare` now admits `YamahaWT11Voice` through the same
generator, processing and control validation as the other Yamaha profiles.
`OfflineYamaha` creates a `WT11Renderer` for each prepared start and a shared
OPZ LFO context per WT11 template. Simultaneous notes observe the same native
rows; the context advances once per render interval. The clock uses the
existing provisional TX81Z/ymfm model with WT11's documented global-byte layout,
not a separately measured WT11 firmware mapping. WT11's stored breath-bias
domain includes 100; declaring a breath route with a nonneutral bias remains
an explicit admission error. Active aftertouch assignments also fail admission.

The existing scoped wheel/breath/foot, pitch-bend, tuning, filter and gain paths
now accept WT11 templates. Mono held-note retention, glide history, minimum
hold, fades, seconds/beat generators and caller-owned tempo maps reuse their
existing execution paths. TX81Z-specific LFO/DAC groups remain restricted by
the shared schema; WT11 uses its ordinary channel routes.

`YamahaSnapshot` now includes the distinct WT11 note snapshot type, retaining
its original buffers and pitch state through instrument JSON round trips.
Clock restore keeps the existing prepare-before-mutation behavior. MIDI-file
rendering dispatches WT11 to this offline NumPy adapter; native live rendering
is not implemented by this integration.

Twenty-seven additional focused cases exercise overlapping shared-clock notes,
delay and transpose, scoped hardware and portable controls, filter order,
minimum hold/fade, mono generator retention, staged generator events, host tempo
changes, MIDI rendering, full stored-bias range and unsupported-assignment
admission. The audio comparisons write at least one second of paired 48 kHz WAVs.
This integrates the current note profile; performance allocation and PCED2
LFO-control/DSP behavior are not thereby implemented or qualified.

The full suite with offline WT11 integration passes: 3,109 tests passed and
21 skipped. Ruff, formatting, type checking and pyupgrade checks are clean.

## Additional work beyond the prompt

None.
