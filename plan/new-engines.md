# Possible new sound engines

## Purpose

Candidate additions to the existing oscillator synth, sampler, two-operator FM
synth, and noise generator. This is a shortlist for choosing future work, not a
commitment to implement every candidate. Effort estimates below are relative
engineering judgments, including dynamic controls and state management.

## Candidates

### Morphing wavetable

- **Sound:** evolving sustained tones, moving harmonic textures, and bright
  electronic leads and basses.
- **First version:** prepared periodic tables, a continuous table-position
  control, and interpolation between adjacent tables. Reuse oscillator phase
  handling and the immutable asset approach used by the sampler.
- **Main challenges:** alias management across pitch, preserving phase during
  modulation, and avoiding discontinuities when moving between tables. Specify
  table preparation and level normalization rather than accepting arbitrary
  sample loops as equivalent inputs.
- **Relative effort:** medium. A broadly useful next engine with a compact
  initial control surface.

Table lookup is an established oscillator technique; see Csound's
[poscil](https://csound.com/docs/manual/poscil.html).

### Granular sampler

- **Sound:** sample clouds, frozen textures, scattered attacks, and evolving
  transformations of recorded material.
- **First version:** bounded overlapping grains from an immutable sample, with
  controls for source position, grain duration, density, and playback pitch.
  Grain scheduling and playback rate provide separate timing and pitch controls;
  this does not promise transparent time stretching.
- **Main challenges:** deterministic scheduling and randomization, smooth grain
  windows, bounded CPU and memory, and snapshots containing every active grain.
  Reuse sampler assets, interpolation, and deterministic random infrastructure.
- **Relative effort:** high, chiefly because each voice owns many independently
  evolving grains.

Csound's [partikkel](https://csound.com/docs/manual/partikkel.html) illustrates
the musical range of grain scheduling and per-grain parameters.

### Plucked string

- **Sound:** plucked, struck, and damped string-like tones with a different
  character from the existing oscillator and FM engines.
- **First version:** a Karplus–Strong style filtered delay loop, excited by a
  deterministic noise burst, with pitch, damping, and excitation controls.
- **Main challenges:** fractional-delay tuning, pitch bends without abrupt delay
  jumps, stable feedback, and predictable decay. Reuse noise excitation, but
  implement the loop's damping inside its feedback path rather than relying on
  the existing output filter chain.
- **Relative effort:** medium for a deliberately limited plucked-string profile;
  realistic bowing, dispersion, and instrument bodies would be separate work.

Julius O. Smith describes this family in
[Virtual Musical Instruments](https://dsprelated.com/freebooks/pasp/Virtual_Musical_Instruments.html).

### Modal resonator

- **Sound:** bells, mallets, metallic percussion, and synthetic resonant bodies.
- **First version:** excite a bank of damped modes with an impulse or short noise
  burst. Each mode has frequency, decay, and gain; harmonic or inharmonic mode
  ratios determine the instrument's character.
- **Main challenges:** stable coefficient changes, sustained ringing after the
  excitation ends, headroom, and CPU cost as the mode count grows. These modes
  need a dedicated resonator representation, not another copy of the shared
  output filters.
- **Relative effort:** medium. Particularly attractive for playable percussion.

Smith's [Physical Audio Signal Processing](https://dsprelated.com/freebooks/pasp/)
covers modal synthesis and related physical models.

### Additive synth

- **Sound:** organ-like tones, inharmonic spectra, and timbres whose individual
  partials evolve independently.
- **First version:** a bounded bank of sine partials with frequency ratios and
  gains, plus shared amplitude shaping. Reuse phase and control-array machinery.
- **Main challenges:** phase continuity during tuning changes, partials crossing
  Nyquist, headroom, and computation proportional to active partial count.
  Avoid allocating an unbounded frames-by-partials intermediate array.
- **Relative effort:** medium; richer per-partial envelopes increase both the
  parameter surface and control cost.

Csound's [adsyn](https://csound.com/docs/manual/adsyn.html) provides an example
of synthesis from time-varying sinusoidal partials.

### Formant synth

- **Sound:** vowel-like tones, synthetic choirs, and breathy vocal textures.
- **First version:** voiced or noisy excitation with a small parallel bank of
  formants controlled by center frequency, bandwidth, and gain. This is musical
  voice coloration, not text-to-speech.
- **Main challenges:** smooth vowel transitions and gain balance. First assess
  whether existing sources plus the proposed resonator bank can express this as
  a preset; a separate engine is justified only by distinct DSP or lifecycle.
- **Relative effort:** medium, potentially lower after a modal engine exists.

Csound's [fof](https://csound.com/docs/manual/fof.html) demonstrates another
formant synthesis approach based on overlapping formant wave functions.

## Extensions to existing engines

These are useful directions, but do not initially need separate engines:

- Four- or six-operator FM: extend the topology and feedback work outlined in
  [the FM plan](fm-synthesis.md).
- Virtual analog: extend the oscillator synth with bandlimited saw and pulse
  waveforms and pulse-width modulation. Nonlinear filters would be a separate
  filter project. Csound's [vco2](https://csound.com/manual/opcodes/vco2/)
  illustrates a table-based bandlimited approach.
- Pink or brown noise: extend the existing noise profile with explicit spectral
  and state semantics.
- Drum kits: start with presets combining existing FM, oscillator, noise, and
  sample sources. Introduce a new engine only for a genuinely different model.

## Suggested priorities

1. **Wavetable** for a broadly useful addition with familiar phase and asset
   handling.
2. **Plucked string or modal** for a contrasting family of acoustic-like sounds.
3. **Granular** when transforming recordings is the priority and the extra
   scheduling complexity is justified.

Additive and formant remain alternatives driven by desired sounds. Choose one
candidate and write its detailed contract before implementation.

## Shared implementation requirements

Follow the [engine execution contract](engine-execution.md). Reuse controls,
envelopes, LFOs, output filters, routing, and lifecycle handling. uFor owns
portable definitions and event semantics; enge owns DSP and numerical state.

For a selected engine, build an independent NumPy reference and Rust backend,
with shared behavioral tests and numerical tolerances rather than bitwise float
identity. Keep the numerical boundary suitable for a later PyTorch port.
Design live parameter changes alongside the initial static sound, including
phase continuity, transitions, release behavior, and resource bounds.

Require deterministic rendering, snapshot continuation, and independence from
host block partitioning. Use control interval 1 as the accuracy reference;
coarser control evaluation must not reduce the rate of audio feedback loops or
other audio-rate recurrences. Benchmark complete voices with changing controls,
not just isolated kernels, before claiming real-time performance.

Include meaningful audio regressions of at least one second at 48 kHz, written
to WAV, and a short listenable FLAC demo for musical evaluation.

## Additional work beyond the prompt

None.
