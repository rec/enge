# TX81Z native-clock note comparison

Enge revision `f99514a` renders each operator once per host frame. At 48 kHz its
output is compared with the pinned YM2414 across all eight algorithms, eight
feedback levels and two ratio patches. The sweep fixes
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
comparison advances enge's envelope counter every three native ticks, computes
gain and quiet-gate values for each tick and supplies native phase steps to the
production operator bank. The first oscillator phase includes one step because
YM2414 clocks before observing its output. Both comparisons use integer DAC
units, a common origin and the same authored levels.

## Results

| Profile zero | Host relative RMS error | Native mismatches / 48,000 |
| --- | ---: | ---: |
| Algorithm 8, feedback 0 | 2.9431% | 0 |
| Algorithm 8, feedback 7 | 25.0144% | 0 |
| Algorithm 1, feedback 0 | 129.6414% | 0 |
| Algorithm 1, feedback 7 | 137.7875% | 0 |

All cases have nonzero held and released output. The native control's maximum
integer sample error is zero across all 6,144,000 observations. Production
relative RMS error ranges from 2.94% to 137.79% for profile zero and from
9.40% to 115.55% for profile one. The production host-clock result
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
  --source /tmp/enge-chip-notes > /tmp/opz-note.csv
diff doc/yamaha-reference/opz-note.csv /tmp/opz-note.csv
```

## Production acceptance gate

The public renderer still needs a shared native timebase for oscillators,
feedback, envelopes and LFOs. It must carry tick remainder and the held output
across render partitions and snapshots, including host intervals with zero or
multiple chip ticks. Overlapping voices need a common tick origin. Per-voice
controls and release events need defined application on that grid. Hardware
group output must sum voices and apply DAC conversion before observing the
host frame. The comparator gives a direct end-to-end check for the covered
patches when that path is implemented.

This qualification covers the pinned software model and the stated sampling
convention. Mixed operator modes, modulation and controller assignments,
polyphonic group output, physical hardware and output reconstruction filters
require additional evidence.
