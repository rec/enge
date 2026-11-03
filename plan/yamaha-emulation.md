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
