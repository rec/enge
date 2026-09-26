# Possible new effect categories

## Purpose

This is a catalogue for choosing the next effect after gain, resonant filtering,
ring multiplication, and granulation. It is deliberately broad: a category names
a useful sound-processing family, not a commitment to implement every familiar
plugin. Select one bounded profile at a time, specify its audible and numerical
behavior in uFor, then implement matching NumPy and Rust paths in enge.

Effects belong at a prepared voice, instrument, master, or direct host-audio
attachment. A candidate must declare its audio ports, channel layouts, parameter
units, control scope, fixed resource limits, latency, tail behavior, bypass
behavior, snapshot state, and response when an input ends. Existing filters remain
voice-source processing until a selected post-mix filter profile explicitly reuses
their kernel.

## Rubber Band for pitch and time processing

Rubber Band Library is the preferred implementation route for high-quality pitch
shifting and time stretching. Enge can use its C API through the existing
[`rubberband-sys`](https://docs.rs/crate/rubberband-sys/latest) crate rather than
creating another public Rust wrapper project. Before adopting it, audit that
crate's generated declarations, API-version support, static build behavior, and
licence metadata against the official [Rubber Band C API](https://breakfastquay.com/rubberband/code-doc/rubberband-c_8h_source.html).

The planned distribution statically links a vendored Rubber Band build and is GPL
when that feature is included. Preserve its licence notices and corresponding
source in source distributions. Keep FFI `unsafe` code confined to the dependency
or a tiny isolated binding crate: enge's main native crate continues to forbid
unsafe code.

Rubber Band accepts planar float32 channel buffers, whereas enge's numerical
boundary uses interleaved `(frames, channels)` float64 arrays. The adapter owns
preallocated layout-conversion scratch for live use and performs explicit offline
conversion. Rubber Band output is an external algorithmic result, so tests compare
documented behavior and tolerances rather than using the NumPy effect path as an
independent sample-for-sample oracle.

Start with an offline, prepared time-and-pitch processor around
`RubberBandStretcher`. High-quality offline processing studies the complete input
before processing and retrieving output, so it belongs on prepared finite audio,
not a live callback. Its output length, latency, final drain, transient mode,
formant option, channel-together option, and parameter-change policy are part of
the effect contract. A later real-time pitch-only processor can wrap
`RubberBandLiveShifter`, whose fixed block size and start delay must participate
in graph latency accounting. See the official [API overview](https://www.breakfastquay.com/rubberband/code-doc/).

## Current foundation

- **Gain:** one input, stateless apart from parameter and bypass ramps.
- **Resonant filter:** one input, retained per-channel integrator state, with
  lowpass, highpass, bandpass, and notch responses.
- **Multiply/ring modulator:** matching `carrier` and `modulator` inputs, with
  sample-by-sample multiplication.
- **Granulator:** one input, bounded history and grain pool, deterministic
  scheduling, freeze, circular frozen replay, and a finite drain tail.

The categories below build on this prepared graph model. A two-input processor
must use named ports. A processor needing a delayed self-reference belongs in a
later explicit-feedback profile, not in the current acyclic graph by accident.

## Utility, routing, and level effects

These are small building blocks, but they define useful graph conventions before
more expensive processors depend on them.

- **Polarity, DC blocking, and one-pole highpass:** polarity is stateless;
  DC blocking has one recursive state per channel and needs a documented startup
  condition and subnormal floor.
- **Pan, balance, width, and mid/side matrix:** fixed channel layouts and exact
  gain laws matter. First decide whether the effect only accepts stereo or has
  explicit mono and surround mappings.
- **Channel mapper, swap, and sum/difference:** a fixed matrix processor is more
  honest than implicit conversions. It is useful for routing, M/S processing,
  and sidechain preparation.
- **Mixer and crossfader:** multiple named inputs with declared gains, mute/solo
  semantics, and no implicit fan-in. An equal-power crossfader should be a
  separate profile from the current linear wet/dry rule.
- **Envelope follower:** converts an audio input into a smoothed internal control
  signal. It is an analysis processor, not a substitute for an audio port, and
  its attack/release recurrence must be frame-partition invariant.
- **Gate and expander:** thresholded level control driven by `input` or a named
  `detector` port. Define detector filters, hold time, hysteresis, look-ahead,
  and behavior at silence.
- **Ducker:** a simple detector-driven gain reduction, useful before a full
  compressor. It can establish the sidechain port and gain-smoothing semantics.

## Delay, echo, and feedback families

These introduce delayed state, large bounded buffers, and tails. They should use
integer sample frames internally; fractional delay needs a chosen interpolation
kernel.

- **Tap delay:** one or more fixed taps, with delay time, gain, pan, and optional
  wet/dry. Dynamic delay time must state whether it jumps, crossfades, or reads a
  moving fractional delay position.
- **Feedback delay:** a bounded circular buffer with feedback gain, damping, and
  a finite tail cutoff. Limit feedback below instability, or specify saturation
  inside the loop.
- **Ping-pong and cross-feedback delay:** stereo or multi-channel delay whose
  feedback matrix is explicit. It should not be inferred from channel count.
- **Modulated delay, chorus, and flanger:** a fractional delay modulated by an
  LFO or control signal. Chorus uses longer, lightly modulated delays; flanger
  uses short delays and usually feedback. Define interpolation, modulation phase,
  zero-delay behavior, and anti-click delay changes.
- **Doubling, ensemble, and ADT:** several decorrelated short delay voices with
  fixed capacity and deterministic modulation seeds. These are variations of the
  same delayed-read primitive, not distinct graph architectures.
- **Karplus-style resonant delay:** a feedback loop excited by input. It can be
  used as an effect, but its pitch and stability requirements overlap physical
  modeling and deserve a separate contract.
- **Reverb:** start with a bounded algorithmic network such as a feedback delay
  network or Schroeder-style comb/allpass structure. Specify matrix, damping,
  diffusion, predelay, tail retirement, denormal treatment, and channel profile.
  Convolution reverb is a different asset-and-partitioning project.
- **Convolution:** static impulse responses, partition size, latency, channel
  routing, IR changes, and tail handling must be prepared. It needs FFT assets,
  scratch, and a numerical tolerance suitable for different FFT implementations.

## Nonlinear, dynamics, and distortion families

Nonlinear processors need strict finite-input/output rules, oversampling policy,
and headroom behavior. Do not hide clipping with undocumented normalization.

- **Hard clip, soft clip, waveshaper, and wavefolder:** transfer functions with
  optional drive and bias. State whether they are sample-rate-only or use a
  prepared oversampling profile to reduce aliasing.
- **Saturation and tape/tube-style drive:** usually a waveshaper plus filtering,
  bias, and optional hysteresis approximation. Start with a named mathematical
  curve rather than a vague analogue claim.
- **Bit crusher and sample-rate reducer:** quantization step, rounding rule,
  dither policy, held-sample clock, and behavior when the reduction rate changes
  must be explicit.
- **Rectifier, harmonic enhancer, and octave divider:** simple nonlinear sources
  for effect chains. Octave division often needs edge/state tracking and can
  misbehave on noisy inputs, so define its detection profile.
- **Compressor:** `input` plus optional `detector`, threshold, ratio, knee,
  attack, release, makeup gain, and detector mode. Define whether gain reduction
  uses peak, RMS, or another measurement and how linked channels behave.
- **Limiter:** a bounded-ceiling compressor. A useful transparent limiter usually
  needs look-ahead, so its fixed latency and delay compensation become part of the
  graph contract.
- **Transient shaper:** separate attack and sustain control derived from an
  envelope follower. It needs clear level detection and a finite response time.
- **De-esser:** detector filtering plus a sidechain compressor or dynamic EQ.
  It is better built after detector and dynamic-gain primitives exist.
- **Dynamic EQ and multiband compression:** split into prepared bands, derive a
  detector per band, then apply time-varying gain. Crossovers, reconstruction,
  phase behavior, and latency are first-class concerns.

## Pitch, time, and resynthesis families

These are musically valuable but require more than ordinary scalar automation.
They should begin with deliberately limited profiles instead of promising general
transparent transformation.

- **Vibrato and tremolo:** existing LFO/control routing may already express them
  at the source. A post-mix tremolo can be a gain profile; audio-rate vibrato
  requires delay or granular traversal and is not merely gain modulation.
- **Frequency shifter:** analytic-signal or quadrature processing, with a fixed
  latency and channel policy. It differs from ring modulation because it moves
  spectral components rather than creating both sum and difference products.
- **Pitch shifter:** use Rubber Band's offline stretcher first, with an explicitly
  prepared finite input and declared pitch range, transient mode, formant mode,
  output timing, and drain. A real-time pitch-only profile can follow through its
  dedicated live shifter when fixed latency and block-size requirements fit the
  native host.
- **Time stretcher:** use Rubber Band's two-pass offline processing to change
  duration while preserving approximate pitch. Live variable-rate stretching is
  deferred until its latency, callback work bound, and dynamic-ratio behavior can
  be specified and measured. Do not promise universal transparency.
- **Reverse, scrub, and stutter:** capture a bounded live buffer and replay it
  under deterministic trigger/control timing. They resemble frozen granulation
  but need a direct repeat/segment contract.
- **Looper:** records and replays prepared finite buffers. It is partly transport
  behavior, so distinguish host-owned recording/file operations from the audio
  processor that replays already-prepared memory.
- **Spectral freeze and spectral blur:** FFT-frame processing with prepared
  window/hop sizes, overlap-add latency, and defined phase evolution. This is
  distinct from the existing time-domain frozen granulator.
- **Vocoder and cross-synthesis:** at least `carrier` and `modulator` inputs,
  with filter-bank or spectral analysis. Specify channel mapping, band count,
  latency, and the behavior of each input ending.

## Frequency-domain and filter-bank families

These build richer coloration from prepared filter or FFT structures.

- **Graphic, parametric, and shelving EQ:** fixed bands with gain, frequency, and
  Q. A first profile should reuse stable biquad or trapezoidal filter semantics
  only after its response and gain conventions are specified.
- **Tilt EQ and tone control:** a compact two-shelf or complementary-filter
  profile that is easier to automate than an unrestricted EQ.
- **Comb, notch-bank, and resonator-bank filters:** static or controllable tuned
  modes for metallic, vocal, and body coloration. Decay and feedback must be
  bounded and their tails retired predictably.
- **Formant and vowel filter:** a small prepared parallel/serial filter bank.
  It may be a preset over resonators rather than a separate effect type.
- **Phaser:** a modulated allpass cascade, usually with feedback. It belongs with
  modulated-delay effects in testing, but its state and frequency response differ.
- **Auto-wah and envelope filter:** filter cutoff driven by an internal envelope
  follower. Declare detector routing, range mapping, and response at silence.
- **Dynamic notch and spectral suppression:** detector-driven frequency changes
  or FFT masking. Begin only after dynamic filter control has a portable action
  representation.

## Modulation and amplitude-pattern families

Some of these overlap source modulation. They are useful as shared effects on a
bus or host input, where source-level controls cannot express them.

- **Ring modulator:** the current multiplication processor is the minimal form.
  A richer profile may add carrier generation, DC removal, and depth control.
- **Amplitude modulation:** multiply by a unipolar or bipolar generated carrier,
  with a declared carrier phase/reset model. It should not silently create a new
  instrument voice.
- **Auto-pan and spatial LFO:** a channel matrix whose coefficients are driven by
  a deterministic LFO. Define the pan law and channel layout.
- **Step sequencer, rhythmic gate, and trance gate:** a prepared step pattern
  addressed by absolute frames or transport position. The host must provide the
  transport contract; the processor should not invent tempo ownership.
- **Sidechain pump:** an envelope-shaped gain curve triggered by a named audio
  detector or prepared events. Its audio and event forms should remain distinct.
- **Sample-and-hold modulation:** captures a control or detector value at a
  defined clock. It needs a deterministic clock and snapshot state.
- **Follower-controlled effects:** map an envelope follower into gain, cutoff,
  delay, or distortion drive. Prefer composing a declared follower with an
  existing target to adding one opaque monolithic effect per mapping.

## Spatial and channel-field families

These require an explicit output layout. No effect should infer speaker geometry
from channel count alone.

- **Stereo panner, width, rotation, and binaural balance:** fixed stereo profile
  with a documented pan law and mono compatibility.
- **Surround matrix and up/down-mix:** prepared matrices labelled by named
  layouts. Downmix coefficients and clipping/headroom rules must be visible.
- **Ambisonic encode, rotate, and decode:** a separate channel-ordering and
  normalization contract, potentially useful before binaural rendering.
- **Binaural/HRTF renderer:** prepared directional filters or convolution assets,
  head-tracking control semantics, and substantial latency/cost constraints.
- **Doppler and distance:** combine fractional delay, gain, and filtering. The
  spatial trajectory belongs to the host or control system; the processor owns
  only the prepared audio transformation.
- **Early reflections:** a sparse multi-tap spatial delay. It can be a smaller,
  lower-latency complement to full reverb.

## Repair, degradation, and creative capture families

- **Noise gate, hum/DC remover, click reducer, and denoiser:** restoration tools
  often need analysis windows and can be unsuitable for low-latency live use.
  Keep conservative repair profiles separate from creative gating.
- **Vinyl, tape, radio, telephone, and speaker simulation:** usually a controlled
  combination of filters, noise, modulation, saturation, and dropout. Start as
  presets when existing primitives suffice; add a processor only for state that
  cannot be represented by a stable chain.
- **Glitch, buffer repeat, dropout, and slice rearrangement:** bounded captured
  audio with deterministic event scheduling. They need explicit latency and
  input-end behavior.
- **Granular delay and particle delay:** granulation whose source is a delay
  history rather than an immutable sample. The current granulator provides much
  of the storage and scheduling foundation.
- **Freeze variants:** amplitude freeze, pitch freeze, shimmer freeze, and
  resonator freeze are different algorithms. Keep their retention and release
  semantics separate from frozen granular replay.

## Multi-input and analysis-driven effects

These validate the graph's named-port design and are best added after simple
one-input effects have settled their tail and latency rules.

- **Sidechain compressor/expander/gate:** `input` and `detector` ports, with a
  declared detector fallback only if the type explicitly permits it.
- **Ring modulation and cross-modulation:** `carrier` and `modulator` ports;
  the existing multiply processor is the baseline conformance case.
- **Vocoder, cross-synthesis, and spectral morph:** carrier/modulator pairs,
  typically with fixed transform latency and prepared analysis buffers.
- **Keyed reverb or ducked delay:** an audio input plus detector that changes wet
  gain or feedback. Feedback changes require strict stability bounds.
- **Audio-driven filter or pitch tracker:** a detector produces an internal
  control signal, then drives a separate transformation. Pitch detection needs a
  confidence/fallback policy and should not imply MIDI extraction.
- **Correlation, stereo linking, and phase scope:** analysis-only processors that
  may expose bounded metering rather than changing audio. They belong in the same
  prepared lifecycle only if the host needs callback-safe measurements.

## Priority order

1. **Delay/chorus/flanger:** a reusable bounded delay primitive unlocks a large
   musical family and establishes latency and tail rules.
2. **Compressor/ducker:** the first useful detector-input processor validates
   sidechains, linked channels, and internal gain smoothing.
3. **Algorithmic reverb:** high musical value once feedback matrices, tail
   retirement, and subnormal handling have a tested shared profile.
4. **Waveshaper/saturation:** compact, audible, and useful for testing nonlinear
   numerical safety. Choose an explicit transfer curve before analogue styling.
5. **Phaser or auto-wah:** builds on filters, LFOs, and followers without FFT
   infrastructure.
6. **Rubber Band offline pitch/time:** add a prepared finite-audio processor with
   a vendored static build, exact source/output timing contract, and audible
   regressions before considering its real-time API.
7. **Other spectral processing:** choose one narrowly specified profile only when
   its latency and quality target are clear.

Start with a single stable implementation slice, rather than an omnibus
"multi-effect" processor. A delay primitive does not automatically authorize
feedback graphs, convolution, looper recording, or arbitrary modulation routing.

## Shared requirements for every selected category

Use the [live callback audit](../doc/live-callback-audit.md) and [engine execution
contract](engine-execution.md). uFor owns portable type definitions and action
semantics; enge owns preparation, NumPy reference processing, Rust processing,
resource bounds, and snapshots.

For each chosen effect:

- Specify offline and live numerical behavior before implementation, including
  interpolation, randomization, nonlinear transfer functions, detector equations,
  and any tail cutoff.
- Prepare topology, channel layouts, maximum block size, latency, history,
  scratch, queues, and worst-case active state before processing starts.
- Support exact action-frame ordering and irregular block partition invariance.
  Audio-rate recurrences remain audio-rate even when ordinary control evaluation
  uses a coarser interval.
- Preserve independent state for channels, graph instances, and voice instances.
  Snapshot continuation must not alias mutable buffers.
- Treat non-finite values as a failed processing block in live execution, and
  identify any decaying recursive states that need the explicit subnormal floor.
- Add focused NumPy/Rust parity, capacity, input-end, bypass, action, snapshot,
  and partition regressions. Audio regressions write at least one second of
  48 kHz WAV and publish a short listenable FLAC when the sound needs audition.
- Benchmark complete prepared source-to-output graphs with changing controls,
  tails, and maximum declared occupancy. Report misses and outliers, not only
  average throughput.
- For Rubber Band, test the vendored static build on every supported platform,
  include licence/source material in distributable artifacts, and distinguish its
  offline two-pass path from a live callback-safe processor.

## Additional work beyond the prompt

None.
