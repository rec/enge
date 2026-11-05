# TX81Z native-clock note comparison

The public TX81Z renderer matches all 6,144,000 captured YM2414 samples across
eight algorithms, eight feedback levels and two ratio patches. Revision
`f99514a` supplies the historical baseline from before native clock integration.
The sweep fixes
a host observation rule so that the two clock schedules can be compared.

## Method

The [register harness](render-ymfm-note.cc) uses the public `ym2414` interface
at ymfm revision `81aec25ccbb98f4873a255f7551ac4dadac59b4a`. It resets the
chip, writes the registers for one channel and keys all four operators on.
It calls `generate` to clock the chip, produce the channel mix and apply the
reference DAC. The two output channels agree at every observation.

The enge [comparison driver](compare-opz-note.py) prepares matching VCED/ACED
data. Both notes use key 69, velocity 127 and operator output level 99. Profile
zero uses four sine operators at ratio one, fine zero and neutral detune.
Its envelopes use AR/D1R/D2R/RR/D1L 31/12/4/9/7. Profile one uses slower
attack, sustains through key-off and exercises asymmetric operator settings.
In VCED storage order (OP4 to OP1), its settings are:

| Setting | OP4 | OP3 | OP2 | OP1 |
| --- | ---: | ---: | ---: | ---: |
| VCED coarse | 4 | 5 | 6 | 7 |
| ACED fine | 0 | 3 | 8 | 15 |
| VCED signed detune setting (neutral 3) | 0 | 2 | 4 | 6 |
| Waveform | 0 | 1 | 4 | 7 |
| Envelope shift | 0 | 2 | 1 | 0 |

Profile one's envelope settings are 20/16/0/15/10. Both profiles use the
algorithm's carrier headroom. Release is scheduled at half a second.

The reference advances one tick per 64 master clocks at 3,579,545 Hz. Each host
frame observes the latest completed tick at the right edge of its interval.
For zero-based frame `f`, the number of completed ticks is
`floor((f + 1) * 3579545 / (64 * 48000))`. At the half-second host boundary
27,965 ticks have completed. Release is written before the next chip clock,
so its first possible effect is the first clock after the host event.
These are explicit event and observation conventions for the experiment.

The first comparison uses the production note renderer and DAC. The controlled
comparison uses two native-tick operations in enge. The envelope bank returns
attenuation after each completed chip tick and carries its three-tick divider
through release. The operator bank accepts integer Q20 increments and observes
the new phase on each tick. It starts at zero phase, just as the chip does.
Both comparisons use integer DAC units, a common origin and the same authored
levels. The control now exercises the production components directly.

## Historical host-clock results

| Profile zero | Host relative RMS error | Native mismatches / 48,000 |
| --- | ---: | ---: |
| Algorithm 8, feedback 0 | 2.9431% | 0 |
| Algorithm 8, feedback 7 | 25.0144% | 0 |
| Algorithm 1, feedback 0 | 129.6414% | 0 |
| Algorithm 1, feedback 7 | 137.7875% | 0 |

All cases have nonzero held and released output. The native control's maximum
integer sample error is zero across all 6,144,000 observations. Historical
relative RMS error ranges from 2.94% to 137.79% for profile zero and from
9.40% to 115.55% for profile one. The former host-clock result
reaches 16,336 DAC units absolute difference. Full results are in the
[comparison CSV](opz-note.csv).

The one-second 48 kHz WAVs preserve the signed DAC units directly as PCM16.

- Algorithm 8, feedback 0: [enge](note-8-0-enge.wav), [ymfm](note-8-0-ymfm.wav).
- Algorithm 1, feedback 7: [enge](note-1-7-enge.wav), [ymfm](note-1-7-ymfm.wav).

## Reproduction

From the repository root, set `YMFM` to the pinned source directory containing
`ymfm_opz.cpp`. The five reference source hashes are recorded in the
[core comparison](opz-core.md).

```sh
mkdir -p /tmp/enge-chip-notes
c++ -std=c++17 -O2 -I"$YMFM" doc/yamaha-reference/render-ymfm-note.cc \
  "$YMFM/ymfm_opz.cpp" -o /tmp/enge-ymfm-note
for a in 1 2 3 4 5 6 7 8; do
  for f in 0 1 2 3 4 5 6 7; do
    for p in 0 1; do
      /tmp/enge-ymfm-note "$a" "$f" "$p" "/tmp/enge-chip-notes/note-$a-$f-$p.i32" || exit 1
    done
  done
done
uv run python doc/yamaha-reference/compare-opz-note.py \
  --source /tmp/enge-chip-notes > /tmp/opz-note-native.csv
diff doc/yamaha-reference/opz-note-native.csv /tmp/opz-note-native.csv
```

## Production native-clock results

The public note clocks its operators, envelopes and feedback for each completed
chip tick. Host frames observe the latest completed output. Snapshots carry
the absolute host frame and held sample so a frame containing zero ticks can
reproduce its output after restoration. Direct callers can supply a host-frame
modulation array, held throughout each frame, or the native observations from
`TX81ZLFOContext.advance_ticks`. The renderer validates the supplied tick grid
against its own position before advancing the note.
Persisted snapshots require the new clock fields. Start a fresh render when
switching from snapshots produced by the older host-clock renderer.

Each note in the offline instrument starts at its trigger's absolute frame.
Simultaneous notes share one LFO advance per interval. An output group sums
the aligned voice observations before DAC conversion. Selecting the same tick
from each voice gives the same instantaneous sum as sampling the per-tick mix.
The group keeps its last DAC value across frames without a chip clock, including
when the constituent voices retire. Group gain and filters then run at host rate.

The [public renderer results](opz-note-native.csv) have zero differing samples
in all 128 cases. The automated comparison uses the native signed PCM captures
in `test/data/tx81z-ym2414-notes.npz` and checks partitioned attack/release and
JSON replay. The archive stores one little-endian int16 array per case under
`a{algorithm}f{feedback}p{profile}`. Its SHA-256 is
`712558eb3283cfd46314c2362386823418990eb7e6f4b1db46baecd2d286d522`.
Separate regressions exercise held output at 96 kHz and grouped DAC hold after
retirement at 192 kHz.

This qualification covers the pinned software model and the stated sampling
convention. Mixed operator modes, modulation and controller assignments,
polyphonic group output, physical hardware and output reconstruction filters
require additional evidence.
