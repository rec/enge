# Enge

Enge is the shared synth and sampler engine for Ufor instruments. Hosts supply
prepared Ufor actions, advance the engine at exact output-frame boundaries, and
own transport, device I/O, MIDI/OSC, GUI, output files, and encoding. Sample
decoding and preparation can belong to the engine's sampler preparation.

The NumPy reference and Rust backend render held linear-envelope voices with
live amplitude and tuning control routes. It consumes uFor trigger contexts,
smooths controls in their declared scopes, preserves phase through pitch changes,
and snapshots active ramps and release tails. Static tuning and the prepared Hz
offset are applied once. Output is float64 in `(frames, channels)` order.

Tuney and the offline synth use the same `PreparedVoice` and `VoiceRenderer` for
oscillator state, gain, envelopes, minimum hold, release, completion, and explicit
channel routes. Each voice renders float64 `(frames, channels)` buffers and can
serialize its state for restoration. Completed voices return silence without
advancing their oscillators. `OscillatorState` retains the phase rounding
correction, and the stateless `waveform_samples()` keeps the established
sample-position/period interface. Tuney translates its mono/binaural settings
and output-layout policy into routes; note policy and device handling stay local.

The NumPy and Rust sampler core in [sampler.py](src/enge/sampler.py) renders decoded
float64 assets with linear interpolation, forward/backward/mirror traversal,
loops, overlaps, live pitch ratios, and fractional effective releases.
`PreparedSample` owns immutable audio shared by cursors; `SampleState` serializes
progress, and `sample_frames()` returns audio plus independent next state.
It passes the [traversal vectors](conformance/sampler-traversal.json) and longer
48 kHz WAV regressions under the [numerical specification](plan/sampler-numerics.md).

`OfflineSampler` in [sample_instrument.py](src/enge/sample_instrument.py) now
consumes prepared uFor sample actions. It applies held linear envelopes,
instrument/slot amplitude and tuning controls, static dB gain, resolved pitch/gain
variation, and explicit channel routes. Selection, sustain, replacement, and
one-shot release decisions come from uFor. Source exhaustion or envelope
completion ends each voice; prepared stops are immediate.

Prepare with `sample_instrument.prepare(score, decoded_assets)`, where the asset
dictionary maps uFor asset IDs to float64 frame/channel arrays, then construct
`OfflineSampler(prepared, backend="native")` for Rust, or omit the backend for
NumPy, and call `advance(actions, start, end)`. The same selection is available on
`SampleVoiceRenderer.start()` and `sample_frames()`. Snapshots
serialize voices and control ramps, and verify the score and decoded-content
fingerprints on restore. Sampler snapshots also retain their backend and reject
restoration into a different backend, including when no voices are active.
Decoded NumPy audio is shared across slots and instances. Each `PreparedSample`
lazily retains one Rust-owned audio copy, reused by its voices and restored
cursors across calls and engine instances.

Rust computes sample traversal, loop overlap, interpolation, envelopes, gain, and
routing while the GIL is released. Python resolves controls and exact rational
release splits; frame/index coordinates cross the binding as signed 64-bit
integers. The shared conformance suite covers both backends, and direct tests
disable Python DSP to prevent a fallback. Modulated sample voices batch a prefix
whose declared maximum tuning cannot exhaust the source, then resolve the
remaining uncertain frame alone so parameters after exhaustion are not
evaluated. This API allocates per call and does not establish live callback
deadline guarantees.

Named seconds-clock LFOs drive amplitude, tuning, and filters in both instruments.
They support sine, square, and triangle shapes, delay/fade-in, and voice, part,
or instrument ownership as allowed by uFor. Shared sources continue through
silence, and snapshots retain their exact phase anchors. Standalone
`enge.lfo.lfo_samples()` also samples canonical uFor rate/reset event states.
Prepared instrument traces do not yet carry addressed LFO events; their LFO
rates remain authored settings. See the [LFO contract](plan/lfo-numerics.md).

Both backends render ordered lowpass, highpass, bandpass, and notch filters with
one or two stages. Cutoff and Q follow per-sample control/LFO routes. Each source
channel owns trapezoidal integrator state, preserved through parameter changes
and snapshots. Filters run before the amplitude envelope and routing; sampler
slot/group filters precede instrument filters. Voice completion discards their
state without adding a tail. See the revised [uFor filter contract](../ufor/doc/instrument-format.md#resonant-filters)
and [enge's realization](plan/engine-execution.md#dynamic-filters).

Decoding remains with the caller. Curved envelopes, named envelopes, equalizers,
layer crossfades, delayed or offset sample starts, event bindings, other
modulation targets, and fade
retirements fail explicitly. The PyTorch backend is not yet implemented.

The NumPy and Rust two-operator FM engine in [fm.py](src/enge/fm.py) uses explicit
phase modulation: a sine modulator drives a sine carrier, with independent held
linear envelopes and optional one-sample modulator feedback. Operator ratios and
tuning, modulation index, feedback, carrier level, common amplitude/tuning, and
filters support the existing control/LFO routes. Pitch changes retain phase;
snapshots retain phase corrections and feedback history. Carrier completion ends
the voice. Rendering is at output rate and permits aliasing.

Load a `SynthInstrumentScore` containing `fm` voice definitions, such as
[the FM fixture](conformance/fm-instrument.json), and prepare its performance with
`ufor.synth_trace.prepare`. Use `fm.OfflineFM(fm.prepare(score))`, then the same
`advance(actions, start, end)`, `snapshot()`, and `restore(snapshot)` operations
as the other engines. Each engine rejects a score containing another source
profile. Select Rust with `fm.OfflineFM(fm.prepare(score), backend="native")`.
Rust computes both envelopes, operator phases, feedback, filters, gain, and
routing in one call with owned buffers and the GIL released. Snapshots reject
a different backend, including when no voices are active.

`fm.fm_samples` is the pure numerical boundary: parameter/envelope arrays plus
phase and feedback arrays produce audio and independent next-state arrays.
Validation, models, control evaluation, and rational envelope boundaries stay
outside it. This is a NumPy reference for a future tensor port, not an existing
`torch.compile` implementation; feedback still requires a sequential recurrence.
See the [FM plan and implementation status](plan/fm-synthesis.md) and
[uFor FM semantics](../ufor/doc/fm-synthesis.md).

The white-noise engine in [noise.py](src/enge/noise.py) accepts uFor `NoiseVoice`
definitions (`noise: "white"`, mapping `pitch_tracking: false`). Prepare the score
with `noise.prepare(score)`, construct `noise.OfflineNoise(prepared,
backend="native")` for Rust or omit the backend for NumPy, then use the same
`advance`, `snapshot`, and `restore` operations. See the
[noise fixture](conformance/noise-instrument.json) and
[portable noise contract](../ufor/doc/noise-synthesis.md).

Each prepared note carries its own deterministic stream key derived from the
performance seed. Block sizes, muting, and snapshot continuation preserve that
stream. The source is mono uniform white noise, with existing resonant filters
before the amplitude envelope/gain and channel routes. Cutoff, Q, and amplitude
support controls and LFOs. Pitch does not affect noise; source tuning is rejected.
NumPy generates random samples using vectorized integer arithmetic; Rust owns
its buffers and performs the full voice DSP with the GIL released. Neither
backend normalizes or clips, and pink/brown noise are not implemented.

Run `uv run pytest test/test_noise.py` for shared regressions and the two-second
filtered-noise demo. Verified 48 kHz WAVs produce listenable
`.pytest_cache/d/audio/noise-demo-numpy.flac` and
`.pytest_cache/d/audio/noise-demo-native.flac`.

Select the Rust backend with `OfflineSynth(prepare(score), backend="native")`
or `VoiceRenderer.start(definition, backend="native")`. The default is `"numpy"`;
there is no fallback if the native extension is unavailable. Active-voice snapshots
retain their backend and restore only into that backend. Tuney continues to use
the default NumPy renderer.

Rust computes oscillator phase, waveforms, filter coefficients/state, envelope
samples, gain, and channel mixing. Python resolves uFor actions, control values, and exact rational
envelope boundaries before the call. The Rust kernel owns its working buffers,
releases the GIL during rendering, and returns new audio and state arrays without
mutating its inputs. It contains no Python callbacks or `unsafe` code. Copying
inputs and allocating output still happen per call, so this is not an
allocation-free device callback API.

Build with `uv sync`; Rust/Cargo 1.85 or newer is required. Python packaging uses
[maturin](https://www.maturin.rs/project_layout.html),
[PyO3](https://pyo3.rs/v0.29.0/), and
[rust-numpy](https://docs.rs/numpy/0.29.0/numpy/).
After changing Rust, run `uv sync --reinstall-package enge` to rebuild the extension.
Run `uv run pytest` for the shared NumPy/native conformance suite. It uses all
available pytest workers and work-stealing to balance long audio cases. Use
`uv run pytest -n 0` to reproduce a failure serially. Native cases are required,
not skipped when compilation is unavailable. `cargo fmt --check` and
`cargo clippy --locked --all-targets -- -D warnings` check the Rust source.

The [proposed execution contract](plan/engine-execution.md) defines the common
timing, dynamic-control, state, and conformance requirements for Python/NumPy
reference engines and a native implementation. It also records the numerical
function boundaries and verification requirements for a future `torch.compile`
implementation.

Run `uv run pytest test/test_synth_demo.py` to generate a 4.25-second stereo demo
at 48 kHz: an arpeggio followed by a held note with smoothed volume and pitch
changes. The regression compares the render against an independently calculated
waveform, writes WAV artifacts, and verifies lossless FLAC encoding using the
`flac` command (required on PATH). Listen to
`.pytest_cache/d/audio/synth-demo-numpy.flac` and
`.pytest_cache/d/audio/synth-demo-native.flac`.
Completed demos are published with Reccy's shared `atomic_output` helper, so a
failed copy preserves the previous demo. Reccy also supplies atomic publication
for offline rendering.

Run `uv run pytest test/test_lfo_demo.py` for a two-second vibrato/tremolo demo,
with a sine synth on the left and a sampled harmonic tone on the right. Both
receive a live gain change. Listen to `.pytest_cache/d/audio/lfo-demo-numpy.flac`
or `.pytest_cache/d/audio/lfo-demo-native.flac`; the test compares independent
audio oracles and verifies lossless encoding before publishing either file.

Run `uv run pytest test/test_filter_demo.py` for a two-second cutoff sweep with
LFO-modulated resonance: triangle synth on the left, sampled harmonics on the
right. The test checks an independent matrix-equation oracle before publishing
`.pytest_cache/d/audio/filter-demo-numpy.flac` and
`.pytest_cache/d/audio/filter-demo-native.flac`.

Run `uv run pytest test/test_fm.py` for FM conformance and a one-second demo with
feedback, interrupted timbre smoothing, and independent operator release tails.
The test checks an independent scalar oracle, writes 48 kHz WAV regressions, and
verifies lossless encoding before publishing
`.pytest_cache/d/audio/fm-demo-numpy.flac` and
`.pytest_cache/d/audio/fm-demo-native.flac`.

For a longer benchmark, run `uv run python scripts/bach.py`. It renders the
four-minute [BWV 578 MIDI adaptation](scripts/bwv-578.md) to `bwv-578.flac` at
48 kHz using FM bass/soprano, triangle tenor, and sampled alto, with subtle
expression, timbre, and pitch automation. `--block-size` controls render blocks;
`--backend native` selects Rust for all three engines. MIDI adaptation, reusable
presets, and streaming FLAC rendering live in
`enge.midi`, `enge.presets`, and `enge.render`. Reccy is now a runtime dependency
for atomic output publication.
