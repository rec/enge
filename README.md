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
disable Python DSP to prevent a fallback. Modulated sample voices still resolve
controls one frame at a time so evaluation stops at source exhaustion. This API
allocates per call and does not establish live callback deadline guarantees.

Named seconds-clock LFOs now drive amplitude and tuning in both instruments.
They support sine, square, and triangle shapes, delay/fade-in, and voice, part,
or instrument ownership as allowed by uFor. Shared sources continue through
silence, and snapshots retain their exact phase anchors. Standalone
`enge.lfo.lfo_samples()` also samples canonical uFor rate/reset event states.
Prepared instrument traces do not yet carry addressed LFO events; their LFO
rates remain authored settings. See the [LFO contract](plan/lfo-numerics.md).

Decoding remains with the caller. Curved envelopes, named envelopes, filters,
layer crossfades, delayed or offset sample starts, event bindings, other
modulation targets, and fade
retirements fail explicitly. The PyTorch backend is not yet implemented.

Select the Rust backend with `OfflineSynth(prepare(score), backend="native")`
or `VoiceRenderer.start(definition, backend="native")`. The default is `"numpy"`;
there is no fallback if the native extension is unavailable. Active-voice snapshots
retain their backend and restore only into that backend. Tuney continues to use
the default NumPy renderer.

Rust computes oscillator phase, waveforms, linear-envelope samples, gain, and
channel mixing. Python resolves uFor actions, control values, and exact rational
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
Run `uv run pytest` for the shared NumPy/native conformance suite. Native cases
are required, not skipped when compilation is unavailable. `cargo fmt --check`
and `cargo clippy --locked --all-targets -- -D warnings` check the Rust source.

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
failed copy preserves the previous demo. Reccy is a development dependency.

Run `uv run pytest test/test_lfo_demo.py` for a two-second vibrato/tremolo demo,
with a sine synth on the left and a sampled harmonic tone on the right. Both
receive a live gain change. Listen to `.pytest_cache/d/audio/lfo-demo-numpy.flac`
or `.pytest_cache/d/audio/lfo-demo-native.flac`; the test compares independent
audio oracles and verifies lossless encoding before publishing either file.
