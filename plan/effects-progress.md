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
| P6 | Rubber Band pitch / time | Fixed 48 kHz R3 offline preparation, linked channels, shifted formants, fixed settings and ties-to-even duration. Existing finite realtime/live adapters retain explicit timing/storage boundaries. | Complete for the approved profile. Unsafe bindings isolated in `ae2c160`; source/licence packaging in `099df03`. Full feature-enabled suite: 3,281 passed, no skips. All 29 focused tests also pass against the wheel built from its source distribution. Conversion scratch is reused; these adapters remain outside the effect graph callback guarantee. See [the contract and evidence](../doc/rubberband-effects.md). |

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
matrices for other layouts, were also approved on 2026-11-06. On 2026-11-08,
the user approved RealFFT backed by RustFFT, with plans and scratch prepared
outside callbacks, as the shared FFT foundation. External IR/HRTF assets remain
caller-supplied and explicitly declared, with caller-decoded audio following the
sampler ownership pattern. Tests use generated fixtures; no automatic downloads
or bundled third-party asset library. The user also approved fixed IRs and
uniform power-of-two partitions (default 256 frames, with matching declared
latency), arbitrary callback sizes and complete FIR tails. Convolution uses
explicit named input/output matrix kernels and is always wet; existing matrices,
Mixer and Crossfader nodes provide dry routing and warm mix/bypass with graph
alignment. Graph-local audio assets/timebases reuse uFor types. Effect preparation
accepts caller-decoded arrays by asset ID, validates dimensions and matching
sample rates, copies audio immutably and fingerprints it for snapshots.

## Remaining catalogue families

Repeated entries in the catalogue refer to the same implementation. Each row
remains open until its named variants have implementations or explicit presets,
contracts and relevant verification. Existing partial foundations are recorded
without treating them as complete families.

| ID | Family / named variants | State or dependency |
| --- | --- | --- |
| U1 | Polarity, DC blocker, one-pole highpass | Complete selected profiles: shared 20 Hz Highpass/DC removal and matrix polarity inversion. uFor `365fc7e`, `8a484fd`; enge 3,277 passed / 29 skipped. Zero callback heap operations and zero timing misses. See [utility profiles](../doc/utility-effects.md). |
| U2 | Pan, balance, width, mid/side | Complete static matrix profiles: equal-power mono pan, attenuation-only balance, width 0–2 and orthonormal M/S encode/decode. Dynamic auto-pan remains M2. See [utility profiles](../doc/utility-effects.md). |
| U3 | Channel mapper, swap, sum/difference | Complete always-applied named-channel matrices, unequal widths, explicit swap/sum/difference and compensated accumulation. No implicit clipping, normalization, wet/dry or bypass. See [utility profiles](../doc/utility-effects.md). |
| U4 | Mixer, linear/equal-power crossfader, mute/solo | Complete. Atomic API-pin integration approved. Schema `0ac37c7`, uFor 1,521 tests passed. Enge 3,289 passed / 29 skipped; 54-test audit records zero heap operations in 2,757 successful and 11 failed callbacks. Nine ordinary timing runs have zero misses. Authored port counts, per-input gain, continuously retimed selector fades and both explicit crossfader laws are covered. See [mixing profiles](../doc/mixer-effects.md). |
| U5 | Envelope follower | Complete. Atomic API-pin integration approved. Shared schema `df1ed1f`, uFor 1,524 tests passed. Enge 3,293 passed / 29 skipped; 114-test audit records zero heap operations in 5,138 successful and 15 failed callbacks. Three ordinary timing runs have zero misses. Same-frame acyclic analysis, independent contribution alignment, portable/opaque replay and failure recovery pass. See [follower profile](../doc/follower-effects.md). |
| U6 | Gate and expander, including sidechains | Complete. Shared schema `4d51cfa`, uFor 1,526 tests passed. Enge 3,312 passed / 29 skipped; 133-test audit records zero heap operations in 5,894 successful and 17 failed callbacks. Six ordinary timing runs have zero misses. Separate pin baseline: 3,293 passed / 29 skipped; pin `e23c878`. Linked peak detection, sidechains, startup, current-setting hold, hard-knee expansion, controls and replay pass. See [dynamics profiles](../doc/dynamics-effects.md). |
| D1 | Multiple fixed taps with gain/pan | Complete. Atomic runtime/API-pin integration approved. Helper `2415d82` composes existing taps, matrices and Mixer into stereo output with static pan/balance. Final instrumented enge suite: 3,328 passed / 29 skipped; 15,539 successful and 32 failed callbacks have zero Rust heap operations. Three ordinary multitap timing runs have zero misses. See [delay profiles](../doc/delay-profiles.md). |
| D2 | Feedback delay with damping | Complete. Atomic runtime/API-pin integration approved. Schema `a8fdc72`, uFor 1,530 tests passed. Optional normalized low-pass on delayed reads before feedback gain, disabled by default; state, controls, tails, replay and failure recovery pass. Same full-suite/heap evidence as D1. Initial 64/256-frame timing stalls did not recur in five repeats; cause remains unresolved and device deadlines remain unqualified. See [delay profiles](../doc/delay-profiles.md). |
| D3 | Ping-pong / cross-feedback delay | Complete. MatrixDelay shares TapDelay DSP and controls, with one delay time and a static square feedback matrix. Rows are destinations and columns sources; absolute row sums are at most one. Schema `391a6ba` / `76b86a1`, uFor 1,538 tests passed. Enge full audit: 3,343 passed / 29 skipped, with zero Rust heap operations in 16,066 successful and 33 failed callbacks. Pin `7bc06a9` baseline: 3,328 passed / 29 skipped. Three ordinary timing runs have zero misses. See [delay profiles](../doc/delay-profiles.md). |
| D4 | Doubling, ensemble, ADT | Complete selected compositions. Helper/recipes `1307c96`, uFor 1,545 tests passed; separate pin `6f174dc`. Authored ModulatedDelay voices feed existing stereo placement and Mixer nodes, with explicit bounds, rates, phases and gain. Seven mono/stereo/control regressions pass; shared enveLFO can replace periodic motion. Enge full audit: 3,350 passed / 29 skipped, with zero Rust heap operations in 16,584 successful and 33 failed callbacks. Three ordinary four-voice ensemble timing runs have zero misses. No tape-transport model. See [the profiles](../doc/modulated-multitap.md). |
| D5 | Karplus-style resonant delay | Complete selected profile. Fixed frequency/damping, integer delay plus all-pass tuning with damping phase compensation, frequency at most quarter-rate, nominal undamped T60 and dynamic decay/mix/bypass. Schema `436b230`, uFor 1,546 tests passed; separate pin `6239b36` baseline: 3,350 passed / 29 skipped. Final enge audit: 3,376 passed / 29 skipped, zero Rust heap operations in 17,115 successful and 37 failed callbacks. Includes a 20-second nominal decay transfer case and filter-only tails. Three ordinary timing runs have zero misses. See [the profile](../doc/resonant-delay.md). |
| D6 | Convolution | Complete selected profile. Schema `43c565d`, uFor 1,567 tests passed. Fixed caller-decoded IRs, local asset/timebase declarations, immutable PCM and snapshot fingerprints; always-wet named matrix kernels, uniform power-of-two partitions (default 256), equal latency and complete FIR tails. Shared RealFFT/RustFFT plans/scratch are prepared outside callbacks. Enge full audit: 3,398 passed / 29 skipped, zero Rust heap operations in 17,885 successful and 39 failed callbacks. Twenty-two focused cases cover direct FIR references, routing, arbitrary callbacks, portable/opaque replay and failure recovery. Three ordinary one-second-IR timing runs have zero misses. See [the profile](../doc/convolution-effects.md). |
| N1 | Hard clip, soft clip, waveshaping, wavefolding | Complete selected curves. Triangle folding extends P4's 4x oversampled Waveshaper with the existing drive, zero-preserving bias, output gain and 48-frame latency. Curve/coefficient selection stays fixed at preparation. Shared schema `7a51fb8`; runtime and qualification are documented in [the nonlinear profiles](../doc/nonlinear-effects.md). |
| N2 | Tape/tube-style drive | Complete selected mathematical compositions. Shared helper `5738c05` composes 20 Hz Highpass, soft Waveshaper and a two-pole lowpass: tape uses bias 0 / 8 kHz; tube uses bias 0.25 / 12 kHz. Both default to drive 2 and unity output gain. No hardware-emulation, hysteresis or automatic normalization claim. See [the nonlinear profiles](../doc/nonlinear-effects.md). |
| N3 | Bit crusher and sample-rate reducer | Complete selected deterministic profile. Shared definition `5738c05`: 1–24 integer bits, ties away from zero, signed endpoint clamp, no dither; 1–65536-frame holds with bits/length latched at capture. Warm bypass preserves the clock, input end completes its current interval, and snapshots preserve held values and remaining frames. See [the nonlinear profiles](../doc/nonlinear-effects.md). |
| N4 | Rectifier, harmonic enhancer, octave divider | Complete selected profiles. Shared schema/helper `b69f30b`: oversampled half/full-wave curves; high-band exciter composition; per-channel Schmitt octave-down square with peak-envelope scaling. Across the audit and corrected tool-test invocation, 3,452 enge tests pass with 29 optional skips; zero Rust heap operations in 20,277 successful and 51 failed callbacks. All twelve ordinary timing measurements are retained, including 25 misses in the half-wave 64-frame run on the shared machine. See [the profiles](../doc/rectifier-enhancer-divider.md). |
| N5 | Limiter | Approved linked sample-peak profile: default -1 dBFS ceiling, 5 ms fixed look-ahead, instantaneous reduction and 50 ms release. Shared schema/API `6009862` passes 1,591 uFor tests; latency calculation now takes an explicit sample rate. Enge integration and qualification pending. |
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
