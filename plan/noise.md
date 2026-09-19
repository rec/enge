# Noise synthesis

## Purpose and scope

Add a noise generator as a fourth enge engine beside the oscillator synth,
sampler, and FM synth. Implement an independent NumPy reference and a Rust
backend with the same lifecycle, controls, filters, routing, and snapshot surface.
Follow the [engine execution contract](engine-execution.md) and the integration
pattern established by the [FM engine](fm-synthesis.md).

The proposed first profile is uniform white noise, shaped by the existing
resonant filters and held linear amplitude envelope. This supports percussion,
wind, breath, and swept noise patches without adding another filter implementation.
Pink and brown noise, live changes of noise distribution, sample-and-hold noise,
stereo-independent generators, and oversampling are outside the initial scope.
Filtered white noise is not advertised as an exact pink or brown noise model.
PyTorch and C++ implementations are also deferred.

The first profile is implemented in `src/enge/noise.py` and `src/noise.rs`.
uFor defines `NoiseVoice` and prepared `noise_key` semantics in
[the noise contract](../../ufor/doc/noise-synthesis.md). The shared regression
suite is `test/test_noise.py`; both backends publish a two-second FLAC demo.

## Ownership and portable definitions

uFor owns a distinct `NoiseVoice` source definition under the existing synth
instrument model, portable parameter semantics, and prepared lifecycle actions.
Reuse `VoiceTemplate` and synth performance preparation rather than treating
noise as a periodic oscillator waveform or creating another lifecycle preparer.
The source discriminator must remain unambiguous in score and action JSON.

The noise voice owns one amplitude envelope and uses existing processing filters,
volume, channel routes, scoped controls, and LFO bindings. Define the initial
noise distribution in the source contract without adding unused color choices.
The first engine accepts noise-only scores and rejects other source profiles,
matching the current separate oscillator and FM engines.

enge owns numerical random generation, envelope/filter realization, audio state,
and the NumPy/Rust implementations. Hosts retain transport, device I/O, raw
performance adaptation, encoding, and output files.

Pitch is not a noise-generator frequency. Keys still select voices; the initial
enge profile keeps the existing control/LFO binding restrictions. Resolved note pitch does
not change the random stream. Reject nondefault source tuning/frequency offsets
and unsupported tuning modulation explicitly; do not silently accept controls
that have no audible meaning. Specify this restriction in uFor and enge together.

## Deterministic random streams

Use a specified counter-based integer generator whose output depends only on a
resolved per-voice stream key and that voice's sample index. The chosen algorithm
is fixed-increment SplitMix64, with exact unsigned 64-bit arithmetic and the upper-53-bit float mapping below. Constants, stream-key
SHA-256 derivation, and counter exhaustion are specified in the uFor contract.
`conformance/noise-v1.json` carries the portable vectors. Do not rely on NumPy's
default RNG, Rust's random library defaults, Python hashing, global state, wall time, or entropy.

The existing `synth_trace.prepare(..., seed=...)` seed should be the single
performance seed. Callers can pass only `trace.actions` to engines, so retaining
a seed only on the trace object is insufficient. Prepared noise starts now carry a resolved
stream key. Define its deterministic derivation from the performance seed and canonical voice identity in uFor,
with language-neutral examples. Do not add a second engine seed argument.
Repeated voices receive distinct streams; rerunning the same prepared actions
reproduces the same streams. Rendering order and block partitioning cannot
change an already assigned stream.

For each active voice frame, consume exactly one random word and advance its
sample index once. Use its upper 53 bits, `u`, to define float64 white noise as
`2 * (u / 2**53) - 1`, in `[-1, 1)`. This mapping is exact in float64; integer
stream output must agree exactly across backends. Filtered/enveloped floating
output follows the existing tolerance-based comparison policy.

Advance through zero gain, zero envelope values, and release tails while the
voice is active. Zero-length renders do not advance state. After completion or
immediate stop, generate no more samples and emit silence. Reject counter
exhaustion explicitly rather than silently repeating a stream.

## Signal path and dynamic behavior

Use this order, matching the oscillator engine:

```text
mono white noise -> existing ordered filters -> amplitude envelope and gain
                 -> explicit output channel routes
```

All output channels receive routed copies of the same mono source. Multiple
voices have separate random streams and filter states. Independent stereo noise
would need an explicit later source/channel contract.

Reuse `enge.filters` for NumPy filtering and the Rust `FilterBank` for native
filtering, including lowpass, highpass, bandpass, notch, stages, and cutoff/Q
boundary policy. Do not duplicate their algorithms or smooth parameters again.
Use the existing amplitude and filter cutoff/Q control targets, smoothing,
scopes, and LFO evaluation. Apply ordered actions before evaluating their frame.
Changing controls must preserve both the random stream and filter state.

Reuse exact envelope boundaries, minimum hold, sustain handling, duplicate
release behavior, and immediate stop. Envelope completion ends the voice and
discards filter state without adding a separate filter tail. Latched envelope
and source definitions do not change during a note. No implicit normalization,
clipping, DC removal, or gain compensation is applied. Authored demo levels must
leave headroom for resonant filters and overlapping voices.

## Numerical API and snapshots

Add `enge.noise` with prepared definitions, a voice renderer, and `OfflineNoise`.
Expose `prepare`, `advance(actions, start, end)`, `snapshot`, and `restore` using
the existing engine conventions. NumPy is the default; `backend="native"`
selects Rust explicitly, without fallback.

Keep the NumPy numerical boundary in arrays and explicit integer state. Generate
counter ranges and random words with vectorized integer array operations rather
than a Python loop per noise sample. Model validation, exact rational boundary
resolution, and action dispatch remain outside DSP. Preserve an independent
reference; a future tensor port must separately establish supported integer
operations and eager/compiled parity, without promising compilation today.

Rust owns working buffers and computes random words, envelopes, filters, gain,
and routing in one block call with the GIL released. Reuse the current PyO3 and
rust-numpy packaging. No Python callbacks, file I/O, or per-sample binding calls
belong inside rendering.

Snapshots retain the resolved stream key and next sample index, envelope cursor
and effective release, filter state, control/LFO state, voice identity, engine
cursor, prepared-definition identity, and backend identity. Encode integer state
losslessly in JSON. Reject incompatible definitions/backends even for silent
snapshots. Snapshot continuation must neither skip nor repeat a random sample.

## Implementation sequence

1. **Specify the uFor source and random contract.** Define `NoiseVoice`, supported
   targets, pitch restrictions, prepared stream-key derivation, and the exact
   generator/mapping. Add portable validation, lifecycle, and random test vectors.
   Preserve existing source formats. Publish uFor first; update enge's dependency
   lock in a separate dependency commit.
2. **Implement the NumPy source and voice.** Establish independent integer and
   conversion oracles, vectorized generation, shared envelope/filter processing,
   and owned outputs with explicit next state.
3. **Integrate the offline engine.** Reuse lifecycle dispatch and control
   infrastructure. Cover overlapping notes, modulation, release, and JSON
   continuation before introducing native execution.
4. **Implement Rust parity.** Add the native block kernel and explicit backend
   selection. Run the same conformance cases against both implementations and
   prove native rendering cannot fall back to Python DSP.
5. **Publish sound and performance evidence.** Add a short noise-percussion and
   resonant-sweep demo, verified WAV/FLAC output per backend, README usage, and
   execution-roadmap status. Measure both the source/voice kernel and the whole
   offline engine so Python control overhead remains visible. Keep the existing
   Bach score and four-voice arrangement unchanged.

## Acceptance and verification

- Fixed vectors verify exact stream-key derivation, random words, conversion,
  and counter advancement, including zero seed and supported boundary values.
  Use an independent scalar oracle rather than another call to the same kernel.
- Different voice identities get different streams; replaying the same actions
  reproduces them. Interleaving renders of separate voices cannot alter either
  stream. Advancing a muted voice preserves its later audible continuation.
- Whole renders agree with 64/128/256/1024/997-frame partitions and splits around
  note starts, same-frame controls, releases, and completion. Verify snapshot
  continuation during sustain, release, and silence, including integer state.
- Exercise abrupt and smoothed amplitude/cutoff/Q changes, interrupted ramps,
  LFOs, multiple voices, repeated releases, minimum hold, stop, and channel routes.
  Check filter state as well as audio, and reject unsupported source parameters.
- Raw output has the specified range. Deterministic fixed-seed statistical tests
  check mean, variance near `1/3`, and low nonzero-lag correlation with documented
  sample sizes and tolerances. These supplement exact vectors rather than
  replacing them or claiming proof of randomness.
- NumPy and Rust share audio tests with declared tolerances; compare integer
  state exactly. Verify owned/strided inputs, no mutation or output aliasing,
  invalid binding inputs, and no Python DSP fallback in the native path.
- Every audio regression writes at least one second of 48 kHz WAV. Publish a
  listenable multi-second FLAC demo per backend using the existing helpers and
  verify lossless encoding. Do not judge filtered noise only by statistics.
- Run focused and full tests plus the repository's Python/Rust checks when
  implementation changes land. Measure short blocks and a longer render; report
  sample rate, voice count, filters, block size, and whether control evaluation
  is included. Offline speed does not establish real-time callback safety.

## Implementation evidence

The shared suite covers NumPy and Rust, independent scalar/matrix oracles,
block partitions and JSON continuation, live filter controls, LFO amplitude,
muted/interleaved streams, owned strided native buffers, and no Python fallback.
Noise-v1 vectors are copied from uFor so installed-package tests do not depend
on a sibling checkout. Both backends publish verified two-second FLAC demos.

A local timing check on 2026-09-19 used 48 kHz float64 stereo routing and one
1200 Hz lowpass stage at Q=0.7. Voice timings include envelope/filter parameter
preparation and binding overhead; medians use 90 blocks after ten warmups.

| Render | NumPy | Rust |
| --- | ---: | ---: |
| One voice, 128-frame block | 0.888 ms | 0.032 ms |
| One voice, 1024-frame block | 6.909 ms | 0.062 ms |
| Four voices, two seconds, 1024-frame blocks, including controls | 4.850 s | 2.251 s |

The last row is a single offline run with constant authored controls and no file
encoding. At that point, the shared Python control/orchestration path was much more
expensive than voice DSP. The subsequent [control optimization](engine-execution.md#vectorized-control-evaluation)
removes per-sample Python evaluation and records updated full-engine timings.
Neither measurement establishes live callback safety.

## Additional work beyond the prompt

None. The implementation follows the profile described above.
