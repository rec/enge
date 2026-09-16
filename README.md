# Enge

Enge is the shared synth and sampler engine for Ufor instruments. Hosts supply
prepared Ufor actions, advance the engine at exact output-frame boundaries, and
own transport, device I/O, MIDI/OSC, GUI, output files, and encoding. Sample
decoding and preparation can belong to the engine's sampler preparation.

The current NumPy reference renders held linear-envelope oscillator voices with
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

Sample traversal, curved envelopes, named generators, filters, other modulation
targets, and fade retirements remain unsupported and fail explicitly. Native and
PyTorch backends are not yet implemented.

The [proposed execution contract](plan/engine-execution.md) defines the common
timing, dynamic-control, state, and conformance requirements for Python/NumPy
reference engines and a native implementation. It also records the numerical
function boundaries and verification requirements for a future `torch.compile`
implementation.

Run `uv run pytest test/test_synth_demo.py` to generate a 4.25-second stereo demo
at 48 kHz: an arpeggio followed by a held note with smoothed volume and pitch
changes. The regression compares the render against an independently calculated
waveform, writes WAV artifacts, and verifies lossless FLAC encoding using the
`flac` command (required on PATH). Listen to `.pytest_cache/d/audio/synth-demo.flac`.
