# WT11 pitch-envelope model audit

Audited on 2026-11-05 before connecting the WT11 schema to audio rendering.
WT11 SysEx inspection and portable voice preparation exist in uFor; this audit
does not establish a working WT11 renderer or measured WT11 envelope timing.

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

Use the pinned conversion as an explicitly provisional starting point when
implementing the WT11 pitch path, preserving all original WT11 bytes in note
and instrument snapshots. Keep its timing and level qualification open.
Tests must exercise onset at PL3, arrival/hold at PL2, early and held release,
partitioned renders, snapshot continuation and sample-rate behavior. Component
projection tests alone cannot qualify a complete WT11 note or performance.

## Additional work beyond the prompt

None.
