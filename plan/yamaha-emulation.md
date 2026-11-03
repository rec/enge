# Yamaha-compatible FM

## Status as of 2026-11-02

`enge/yamaha.py` already contains a Python DX7 note renderer, operator envelopes,
frequency and level scaling, algorithm routing and feedback. This is partial
implementation, not verified hardware equivalence. The requirements below remain
the target contract; oscillator-sync behavior, reference
qualification, TX81Z/WT11 rendering and full engine integration require further
work. Do not restart the existing DX7 implementation.

An initial [pinned MSFA comparison](../doc/yamaha-reference/README.md) now includes
constructed named patches, the reference harness and paired 48 kHz WAV files.
It found whole-file relative RMS differences of 5.20% and 4.13% for algorithms
1 and 32. Qualification is not passed; onset/release timing and the intended
reference target require resolution before treating these as passing fixtures.
An all-32-algorithm sweep subsequently exposed incorrect algorithm 23 routing
in uFor. Its correction reduced held-section relative RMS error from 35.234%
to 0.2432%; the other algorithms were unchanged. The report includes sweep
measurements and the corrected algorithm 23 recordings. Uniform operator
settings and disabled feedback limit what this sweep proves.
An asymmetric all-algorithm sweep now gives each operator a distinct ratio and
level. Held-section relative RMS errors range from 0.213% to 0.665%, with no
new routing outlier. Whole-file errors remain 4.16% to 5.86%; qualification is
still incomplete. The report includes the sweep and worst-held-case audio pair.

Pitch envelopes now decode the packed/unpacked four-rate/four-level fields,
start from level four, hold level three and release toward level four. Integer
Q24 octave state and a fractional tick carry adapt
[MSFA's measured pitch rate/level data](https://github.com/google/music-synthesizer-for-android/blob/master/app/src/main/jni/pitchenv.cc)
to this renderer's per-sample execution. Pitch affects fixed and ratio operators.
Tests cover octave calibration, release, partitioned continuation and invalid
pitch parameters. This is not yet a comparison against a complete reference
renderer's WAV output.

`DX7Renderer.snapshot()` now captures the patch/note definition, six operator
envelopes, pitch envelope, oscillator phases and delayed feedback. Restore
rejects a different patch, note, velocity or sample rate and copies mutable
state so repeated restores do not mutate the saved snapshot. JSON round-trip
and one-second audio regressions cover held and released notes, feedback and
partitioned continuation. These snapshots remain specific to the dedicated
Yamaha renderer, separate from general FM snapshots.

Operator modulation now uses cycles before sine conversion. Feedback uses two
delayed operator outputs and the power-of-two level scaling described by
[MSFA's feedback kernel](https://github.com/google/music-synthesizer-for-android/blob/master/app/src/main/jni/fm_op_kernel.cc).
Both history samples survive snapshots. Analytic one-second WAV regressions
cover feedback off, minimum and maximum levels, with both a feedback carrier
and a feedback modulator. Integer sine/gain quantization and multi-operator
feedback loops still require reference qualification.

The renderer rejects uFor's diagnosed algorithm, feedback and operator parameter
errors before preparing DSP state. It also rejects envelope rates/levels above
99, which raw voice inspection preserves. Tests exercise packed and unpacked
voices without changing their stored bytes. This does not yet qualify all
global Yamaha parameters or dump-to-engine integration.

Fixed-frequency operators now use the low two coarse bits and key-independent,
positive-only logarithmic detune, following the frequency equations in
[MSFA's DX7 note implementation](https://github.com/google/music-synthesizer-for-android/blob/master/app/src/main/jni/dx7note.cc).
One-second 48 kHz regressions cover repeated decades, different played keys and
negative/neutral/positive detune. The renderer still uses floating-point
oscillators; these checks do not establish bit-exact hardware equivalence.
Ratio-mode detune retains the existing approximation and still needs reference
qualification.

Implement a dedicated Yamaha renderer for the DX7 and TX81Z instead of
projecting their patches into the general FM voice profile. The general graph
remains useful for authored enge patches, but it cannot represent Yamaha's
rate/level envelopes, key scaling, fixed-frequency operators, pitch envelope,
or feedback arithmetic.

## DX7

- Decode each validated VCED or packed bank voice into six Yamaha operators,
  the selected one of 32 algorithms, carrier mix, and delayed feedback path.
- Preserve Yamaha operator numbering, frequency mode, coarse/fine ratio,
  detune, level scaling, velocity sensitivity, and oscillator sync.
- Render the four-rate/four-level amplitude and pitch envelopes in a dedicated
  fixed-point-compatible state model. Compare its generated output against
  known DX7-compatible renderers using named patches and 48 kHz WAV fixtures.
- Keep the lossless SysEx objects as the import boundary. A malformed dump must
  remain inspectable and never become a playable voice.

## TX81Z / WT11

`TX81ZRenderer(voice, additional, note, velocity, sample_rate=48000)` now accepts
explicit VCED/ACED objects and connects nominal frequency, chip detune, output
level, the shared envelope clock, eight waveforms/algorithms and OP4 feedback.
`render`, `release`, `snapshot` and `restore` provide offline note lifetime and
owned JSON continuation. Snapshot identity includes both edit buffers, key,
velocity and output rate. Tests cover all algorithms, mixed waveforms, fixed
and ratio operators, detune, held/released snapshots and mismatched definitions.
The initial supported pitch span after transpose is MIDI 13–108 (chip key codes
0–31); out-of-span notes are rejected. Nonzero key-level/velocity scaling and LFO
modulation are rejected until implemented. Velocity is accepted as
note metadata but has no gain effect when KVS is zero. This is an incomplete
profile, not completion of the full TX81Z/WT11 requirement. Controller integration,
DAC/integer quantization, broader malformed-global checks and hardware/reference
qualification remain work.

ACED reverb rate now prepares a capped effective envelope rate and selects an
additional post-release stage once raw attenuation reaches 192 (-18 dB in the
model). Rate zero remains off; a nonzero reverb rate cannot accelerate release.
The implementation follows ymfm's OPZ rate preparation and envelope transition.
WAV regressions cover the exact threshold/increment schedule, a slow release,
repeated key release and JSON continuation. The note-renderer integration also
covers reverb tails. This parameter controls envelope decay; hardware
qualification of that behavior remains incomplete.

`tx81z_detune_step` now maps the seven VCED detune settings (neutral 3) and a
five-bit chip key code to the signed OPZ phase-step offset. It follows
[ymfm's key-dependent detune table](https://github.com/aaronsgiles/ymfm/blob/main/src/ymfm_fm.ipp)
and the [OPZ ordering before the frequency multiplier](https://github.com/aaronsgiles/ymfm/blob/main/src/ymfm_opz.cpp).
Tests cover sign, neutral, low-key zero offsets and high-key limits, and render
the resulting offset through the operator bank at 48 kHz. This primitive takes
chip key codes explicitly; MIDI-to-chip key mapping, multiplier quantization
and the complete patch-frequency preparation path still need integration.
Fixed-frequency mode bypasses this detune path in the reference model.

`tx81z_output_gain` now converts the VCED 0–99 output control through
[DXconvert's total-level curve](https://github.com/richotero/dxconvert/blob/master/DXconvert/dxcommon.py)
and its [reported TX81Z carrier-headroom measurements](https://github.com/richotero/dxconvert/blob/master/DXconvert/fb01.py).
Carrier offsets are 8, 13 and 16 TL steps for two-, three- and four-carrier
algorithms; modulators retain the unadjusted level curve. Saturation is at TL
127. This nominal floating gain does not include integer DAC/sine quantization,
velocity, key scaling or envelope attenuation. Regressions cover curve boundaries
and headroom, plus a one-second paired VCED/ACED integration through nominal
frequency, shared envelope clock, gain conversion, operator mixing and release.
These are model-level checks, not new hardware measurements.

The numeric `tx81z_operator_samples` kernel now renders all eight four-operator
graphs in visible OP1–OP4 order, with OP4 two-sample feedback, independent
waveform selections and caller-prepared per-frame frequency/gain rows. Its
`TX81ZState` retains phases and feedback through JSON continuation without
mutating the caller's saved state. Routing/modulation scaling follows
[ymfm's four-operator output path](https://github.com/aaronsgiles/ymfm/blob/main/src/ymfm_fm.ipp).
Analytic 48 kHz WAV regressions cover all graphs, asymmetric operators,
feedback, waveform selection and partitioned continuation. This is raw carrier
mixing, not the completed patch renderer: DAC saturation, detune integration,
velocity/key scaling and the full voice profile remain work. Nominal VCED gain conversion
and the envelope clock are now connected in the component integration test.

The user approved ymfm's waveform model provisionally on 2026-11-03, including
the inferred signed sine-squared shape. `tx81z_wave_samples` now realizes all
eight continuous shapes using zero-based ACED waveform numbers. The model
follows [ymfm's OPZ waveform construction](https://github.com/aaronsgiles/ymfm/blob/main/src/ymfm_opz.cpp);
it does not reproduce logarithmic table quantization or establish measured
hardware equivalence. One-second 48 kHz regressions distinguish all eight
shapes at sixteen positions per cycle, including negative/unwrapped phases
and the silent half-cycle boundaries. The complete TX81Z renderer remains work.

`TX81ZEnvelope` now prepares matching VCED/ACED operator data into a separate
OPZ attenuation state. Its `clock(counter)` consumes the host's shared envelope
counter, not output-frame time. Rates include key scaling, attack/decay/sustain/
release transitions, saturated attenuation and envelope shift (disabled for
OP1). D1L is inverted into chip attenuation, so level 15 means no initial decay,
as described by the Yamaha manual. Increment sequencing follows
[ymfm's envelope clock](https://github.com/aaronsgiles/ymfm/blob/main/src/ymfm_fm.ipp)
and [OPZ rate preparation](https://github.com/aaronsgiles/ymfm/blob/main/src/ymfm_opz.cpp).
One-second WAV regressions cover full-level sustain, release and every shift,
including JSON state continuation. Lower-rate timing, output-level scaling and
complete voice integration remain unqualified.

`TX81ZEnvelopeBank` now places four envelopes on a shared integer counter and
converts chip-clock intervals to output frames with an integer remainder.
The nominal clock is 3,579,545 Hz; clock and output rates remain explicit state.
The envelope divider is 2 * 32 * 3, following ymfm's OPZ engine constants.
Rows observe attenuation before advancing the next output-frame interval.
Counter/remainder values can be retained by a host across voice/block boundaries.
One-second 48 kHz regressions verify exact and fractional clock ratios, all four
shifts and partitioned JSON continuation. This establishes clock conversion,
not measured hardware phase alignment or the complete patch renderer.

`tx81z_base_frequency` now prepares nominal frequency from matching VCED/ACED
operators. Ratio mode orders the 64 multiplier/coarse-detune combinations and
adds fine steps before coarse detune. Half-multiple entries reject unavailable
fine values above seven. Fixed mode uses the VCED coarse index shifted by two,
plus fine and the ACED range, reproducing the manual's 8–32,640 Hz span and its
range-dependent steps. The equations are cross-checked against ymfm's OPZ
register model and [DXconvert's `fix_4op`](https://github.com/richotero/dxconvert/blob/master/DXconvert/fourop.py);
the [Yamaha manual](https://www.polynominal.com/site/studio/gear/sold/yamaha_tx81z/yamaha-tx81z-manual.pdf)
and [TX81Z Programmer ratio displays](https://github.com/mgregory22/tx81z-programmer/blob/master/src/freqratios.c)
corroborate the control mapping. Audio regressions cover ratio,
fine, fixed-frequency boundaries and different played keys. Applying the chip
detune primitive, phase quantization and complete voice integration remain work.

- Pair VCED and ACED edit-buffer messages explicitly before conversion.
- Decode its four-operator, eight-algorithm routing, frequency modes, feedback,
  and ACED waveform selections into the same Yamaha renderer family.
- Preserve the TX81Z waveform definitions rather than mapping them to enge's
  generic sine, square, and triangle waveforms.

## Delivery boundaries

- The compatibility renderer is offline first. Do not claim callback-safe live
  Yamaha emulation until its fixed state and allocation bounds are measured.
- The general FM graph and Yamaha profiles have separate schema tags, prepared
  state, snapshots, and regressions. Neither is an alias for the other.
- Add a small set of public imported-patch fixtures only when their licensing
  permits redistribution. Tests may construct parameter bytes directly.

## Additional work beyond the prompt

None.
