# Effects implementation inventory

The user selected the entire `new-effects.md` catalogue on 2026-11-06, including
the spectral appendix. This supersedes the earlier selection-only/deferred
status for effects. It does not select the separate new-engine catalogue.
Complete the first six priority entries in order, then the remaining families.
Implement named, bounded profiles and reuse primitives for compositions; a
shared primitive does not by itself complete every family using it.

## Approved first six

| ID | Entry | Accepted design | State |
| --- | --- | --- | --- |
| P1 | Delay / chorus / flanger | Moving linear-interpolated reads for modulation; continuous fixed-tap retiming from bounded read-head mixtures. Existing retiring heads finish on schedule; capacity is fade frames plus one. | Selected profile complete: schema `21078ca`, pin `8ab7e48`, interpolation/retiming and shared native lifecycle verified. Shared-runtime suite: 3,210 passed / 21 skipped. |
| P2 | Compressor / ducker | Feed-forward soft knee, linked stereo peak detection, explicit detector input, no look-ahead. | Selected profile complete: schema `43bfb83`, pin `b4a20b2`, shared native lifecycle verified. Shared-runtime suite: 3,210 passed / 21 skipped, zero callback heap operations. See the [contract and CPU measurements](../doc/compressor-effects.md). |
| P3 | Algorithmic reverb | Bounded eight-line feedback delay network, stereo output, predelay, damping and decay. Mono dry maps to both stereo outputs at minus 3 dB each; stereo dry passes through. | Selected profile complete: schema/specification `9c2336f`, pin `3830aa1`, shared foundation `2a800e8`. Full suite: 3,222 passed / 21 skipped. Twelve focused regressions and a 64-test callback audit pass with zero heap operations. See [the profile](../doc/reverb-effects.md). |
| P4 | Waveshaper / saturation | Explicit hard/soft curves, zero-preserving bias, drive, output gain, 4× oversampling, graph latency reporting and audio/control/end-marker compensation. Two cascaded 2× stages use 65-tap linear-phase half-band filters with fixed 48-frame latency. | Selected profile complete: schema `d9da8f6`, uFor 1,515 tests passed. Runtime verified: 3,237 passed / 21 skipped; 124-test audit records zero heap operations in 5,008 successful and 12 failed callbacks. Three finite ordinary CPU runs record zero misses. User approved committing this API migration atomically. See [the profile](../doc/waveshaper-effects.md). |
| P5 | Phaser | Six identical energy-normalized all-pass stages; shared enveLFO bipolar sweep, center frequency and cents/octave depth; feedback limited to [-0.95, 0.95] with one frame of delay. | Complete. Schema `c0f8dbe` and numerical clarification `e29fff1`, uFor 1,516 tests passed. Enge 3,252 passed, 21 skipped; 139-test audit recorded zero heap operations in 5,684 successful and 15 failed callbacks. Three ordinary timing runs had zero misses. Response, energy, lifecycle, upstream latency and replay checks pass. Auto-wah remains a separate family. See [the profile](../doc/phaser-effects.md). |
| P6 | Rubber Band pitch / time | Reuse and qualify the existing optional offline and finite realtime adapters; preserve their explicit timing/storage boundaries. | Existing adapters; closeout audit pending. |

The user also approved extending the prepared native graph API with explicit
port widths, source/host input routing, preallocated input buffers and a declared
output layout before P2. This is required for native sidechains, mixers and
spatial effects. The routing implementation now supports main/source/host inputs,
explicit widths and a separately declared output layout in both live owners:
3,166 tests passed / 21 skipped, with zero Rust heap operations in the focused
callback audit. See [the contract](../doc/live-inputs.md).
Per-instrument effect attachments and their post-effect taps remain unfinished.
Native input-end/stop handling is implemented in the shared-runtime
migration, with release, drain and stop regressions. On 2026-11-06,
the user approved extending the existing native effect graph into a persistent
offline renderer, exporting/importing explicit processor state for JSON snapshots
and replacing the old native whole-graph path. Use this foundation for the next
stateful effects rather than adding separate bindings for their history buffers.
Named stereo/5.1/7.1 layouts and first-order ACN/SN3D ambisonics, with explicit
matrices for other layouts, were also approved on 2026-11-06. FFT implementation
and external IR/HRTF asset policy were asked about; retain their answers before
those slices.

## Remaining catalogue families

Repeated entries in the catalogue refer to the same implementation. Each row
remains open until its named variants have implementations or explicit presets,
contracts and relevant verification. Existing partial foundations are recorded
without treating them as complete families.

| ID | Family / named variants | State or dependency |
| --- | --- | --- |
| U1 | Polarity, DC blocker, one-pole highpass | Some source processing exists; shared effects profiles pending. |
| U2 | Pan, balance, width, mid/side | Static source pan/balance exists; effects matrix profiles pending. |
| U3 | Channel mapper, swap, sum/difference | Pending explicit matrix processor/layout contracts. |
| U4 | Mixer, linear/equal-power crossfader, mute/solo | Wet/dry exists; explicit multi-input profiles pending. |
| U5 | Envelope follower | Pending reusable detector/control output. |
| U6 | Gate and expander, including sidechains | Pending P2 detector machinery, hold and hysteresis contracts. |
| D1 | Multiple fixed taps with gain/pan | Single tap exists; multitap/routing profile pending. |
| D2 | Feedback delay with damping | Feedback exists; damping/profile verification pending. |
| D3 | Ping-pong / cross-feedback delay | Pending explicit feedback matrices. |
| D4 | Doubling, ensemble, ADT | Pending decorrelated multi-read profiles over P1. |
| D5 | Karplus-style resonant delay | Pending tuned-loop stability/decay contract. |
| D6 | Convolution | Pending shared FFT, IR assets, partition/latency/tail contract. |
| N1 | Hard clip, soft clip, waveshaping, wavefolding | P4 establishes the oversampled core; folding remains a distinct curve/profile. |
| N2 | Tape/tube-style drive | Pending explicit mathematical coloration profiles over P4 and filters. Hardware emulation is not implied. |
| N3 | Bit crusher and sample-rate reducer | Pending quantization, rounding/dither and held-sample clock contracts. |
| N4 | Rectifier, harmonic enhancer, octave divider | Pending nonlinear profiles and octave edge/detection state. |
| N5 | Limiter | Pending bounded look-ahead, ceiling and latency compensation. |
| N6 | Transient shaper | Pending follower/detector machinery. |
| N7 | De-esser | Pending detector filtering and dynamic gain. |
| N8 | Dynamic EQ / multiband compression | Pending crossover/reconstruction and per-band detectors. |
| T1 | Vibrato / tremolo as effects | Gain/modulated delay foundations exist; named effect profiles pending. |
| T2 | Frequency shifter | Pending analytic/quadrature profile; also a spectral-appendix candidate. |
| T3 | Reverse, scrub, stutter | Pending bounded capture and traversal/trigger contracts. |
| T4 | Looper | Pending prepared recording/replay memory and host transport boundary. |
| F1 | Spectral freeze | Pending FFT framing, phase evolution, capture/release and latency. |
| F2 | Spectral blur | Pending FFT framing and bounded time/frequency smoothing. |
| F3 | Vocoder / cross-synthesis | Pending carrier/modulator analysis, band mapping and input-end contract. |
| F4 | Graphic, parametric and shelving EQ | Resonant kernels exist; explicit EQ gain/response profiles pending. |
| F5 | Tilt EQ / tone control | Pending complementary/shelving profiles. |
| F6 | Comb, notch bank, tuned resonator bank | Single notch exists; bank/tuning/decay profiles pending. |
| F7 | Formant / vowel filter | Pending resonator-bank profiles/presets. |
| F8 | Auto-wah / envelope filter | Pending follower-to-cutoff integration. |
| F9 | Dynamic notch / spectral suppression | Pending detector-driven filtering/masking. |
| M1 | Ring / cross modulation and generated-carrier amplitude modulation | Multiply exists; generated carrier/DC-removal/depth profiles pending. |
| M2 | Auto-pan / spatial LFO | Pending matrix processor plus enveLFO routes. |
| M3 | Step sequencer, rhythmic/trance gate | Pending prepared patterns and supplied transport timing. |
| M4 | Sidechain pump, audio-triggered and event-triggered | Pending follower/event-to-envelope integration; forms remain explicit. |
| M5 | Sample-and-hold modulation | Shared enveLFO execution exists; audio/control capture profile pending. |
| M6 | Follower-controlled gain/filter/delay/drive | Pending reusable audio-derived modulation sources. |
| S1 | Stereo panner, width, rotation, binaural balance | Pending explicit channel matrices and pan laws. |
| S2 | Surround up/down-mix | Named stereo/5.1/7.1 layouts and explicit matrices approved; processors pending. |
| S3 | Ambisonic encode / rotate / decode | First-order ACN/SN3D approved; processors pending. |
| S4 | Binaural / HRTF | Pending directional assets, interpolation, convolution and head-control contract. |
| S5 | Doppler / distance | Pending host-supplied trajectory, fractional delay, gain/filter mapping. |
| S6 | Early reflections | Pending sparse spatial multitap profile. |
| R1 | Noise gate, hum/DC remover, click reducer, denoiser | Pending explicit conservative repair profiles; share U1/U6 where applicable. |
| R2 | Vinyl, tape, radio, telephone, speaker simulation | Pending declared compositions/presets, adding state only where required. |
| R3 | Glitch, buffer repeat, dropout, slice rearrangement | Pending bounded capture and deterministic scheduling. |
| R4 | Granular / particle delay | Granulator history exists; named delay profiles pending. |
| R5 | Amplitude, pitch, shimmer and resonator freeze | Pending distinct capture/retention/release profiles. |
| A1 | Spectral morph | Pending F3/FFT infrastructure and morph contract. |
| A2 | Keyed reverb / ducked delay | Pending P2/P3 and explicit detector-driven wet/feedback controls. |
| A3 | Audio-driven filter / pitch tracker | Pending analysis outputs, confidence and fallback policy; no implied MIDI extraction. |
| A4 | Correlation, stereo linking, phase scope | Pending bounded analysis/metering outputs. |
| F10 | Spectral delay | Pending bounded per-band/bin histories and phase reconstruction. |

## Completion evidence

For each profile, record its schema/runtime commits, numerical and audio oracles,
NumPy/native coverage, input-end/bypass/partition/snapshot behavior, fixed resource
bounds and relevant heap/performance measurements. Preserve limits on optional
libraries, supplied assets and physical/device qualification. Do not count an
unimplemented row as complete because its supporting primitive exists.

## Shared native renderer work

The persistent Rust DSP core now lives in `src/effects/runtime.rs`, separate from
the source/callback owner in `src/live.rs`. The callback owner consumes that same
core; no DSP algorithm changed in the extraction. The 78 existing effect and live
regressions pass, with Rust build and lint checks clean.

The offline owner is now connected to the core, replacing the
old whole-graph and separate granular paths. Portable ramps/generators retain
their Python representation; DSP state has explicit JSON export/import. The
109-test focused audit passes with zero Rust heap operations in 4,118 successful
and ten failed callbacks. Cross-backend replay, native exact replay and native
release/drain/stop are covered. The full ordinary suite, including the explicit
grain cap, passed 3,210 tests with 21 skipped. Six sequential finite CPU
measurements recorded zero misses; their
outliers and limits are retained in the implementation report.

The user approved an explicit `maximum_duration_seconds` cap for granulation.
uFor commit `08010e5` validates authored duration, actions and modulation domains;
native slots reserve that output duration independently of source history.
Slow duration automation, portable snapshots and callback failure/recovery are
covered in the migration. See [the implementation contract](../doc/persistent-effects.md).
This foundation precedes the eight-line reverb and later stateful FX.

## Additional work beyond the prompt

None.
