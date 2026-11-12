# Oversampled waveshaper and graph latency

`ufor.audio_effects.Waveshaper` replaces `SoftClip` and `soft_clip`, without an
alias. The old normalized, stateless tanh curve has been removed from both
renderers. Existing callers use the new `waveshaper` definition and its explicit
`soft`, `hard` or `fold` curve, positive drive, finite bias and output gain in dB.

The nonlinear stage computes `f(drive*x + bias) - f(bias)`, using tanh for soft
saturation, clamping to [-1, 1] for hard clipping, or the period-four triangle
transfer for folding. This permits asymmetric
distortion while preserving zero input at zero. Output gain applies after
downsampling. It does not normalize individual blocks or signals.

The fixed 4× profile uses two cascaded 2× upsamplers and two 2× downsamplers,
each with the same normalized 65-tap Blackman-windowed half-band FIR. Upsamplers
insert zeros and scale filtered output by two; downsamplers retain even outputs.
The initial phase is zero throughout. The implementation follows the numerical
specification in uFor `d9da8f6`.

Audio latency is 48 input frames, including the dry and bypass paths. FIRs keep
running while bypassed. Each channel reserves four 65-frame histories and a
48-frame dry history. A source ending drains the finite cascade for 96 frames;
silence alone does not end a source. Native snapshots and portable JSON retain
histories, write positions and the drain countdown.

`PreparedEffects.latency_frames` reports the graph output latency. At each node,
incoming audio and end markers are delayed to the greatest upstream latency.
Resolved parameters and bypass controls are compensated too. Waveshaper drive
and bias additionally wait 24 frames to the nonlinear stage; output gain, mix
and bypass wait 48 frames to the output stage. Other processors' controls wait
their upstream audio latency. One resolved drive/bias value is held across four
oversamples. Control histories begin at authored values, with bypass enabled.
Audio, control and end-marker compensation buffers are allocated at preparation
and included in snapshots. Granular freeze/unfreeze commands wait for their
upstream audio too; the authored target and pending command history are distinct
from the currently sounding frozen state. Stop remains an immediate terminal
action on its exact frame. The output keeps the requested frame count; a host
can use the reported latency when aligning recordings and other graphs.

The persistent native core owns the same implementation for offline and live
rendering. The NumPy reference uses vector FIR evaluation. Regressions also use
an independent full-convolution oracle built from sinc and Blackman functions,
including explicit control timing. Tests cover all three curves, bias, wet/dry gain,
alias reduction, independent branch delays, mid-buffer JSON replay, shared
enveLFO and tempo-driven Patch controls, rejected snapshots and callback failure
recovery. Audio regressions write paired 48 kHz WAVs of at least one second.

The following figures record the original hard/soft qualification. The
[nonlinear extensions report](nonlinear-effects.md) records folding,
tape/tube compositions and bitcrushing separately.

The final 124-test allocation audit includes pending granular freeze commands
behind upstream latency. It recorded zero Rust allocations, reallocations or
frees in 5,008 successful callbacks and 12 failed callbacks. Native snapshot
rejection checks cover FIR, audio-alignment and control-compensation storage.
The overflow regression uses a single-frame callback to verify that non-finite
upstream output is rejected before entering retained FIR history. Processor input,
FIR outputs and the wet output have explicit finite-value checks.
The final ordinary suite passed 3,237 tests with 21 skipped. Ruff, formatting,
Python type checks, pyupgrade, Rust formatting and Clippy checks are clean.

After all test jobs finished, three sequential ordinary-build measurements used
48 kHz, 16 synth voices, the Waveshaper profile and 64 effect actions per block.
Each requested two seconds and measured whole blocks. Times are microseconds;
median/p99/maximum exclude the first block, while misses include it.

| Frames | Median | p99 | Maximum | First | Budget | Misses |
| --- | --- | --- | --- | --- | --- | --- |
| 64 | 367.4 | 440.5 | 531.4 | 678.6 | 1,333.3 | 0 / 1,500 |
| 256 | 1,156.6 | 1,250.0 | 1,280.3 | 1,434.1 | 5,333.3 | 0 / 375 |
| 1,024 | 4,296.5 | 4,477.5 | 4,484.6 | 4,530.8 | 21,333.3 | 0 / 93 |

Reproduce with `scripts/benchmark_live.py --waveshaper --seconds '2 s'
--voices 16 --effect-actions 64 --block-frames FRAMES`. These finite CPU
measurements perform no device I/O and do not qualify physical-device deadlines.

The benchmark CLI accepts unit strings such as `--seconds '1 s'` through reccy's
Pint-backed parser and retains its positive whole-second duration requirement.
Its separate Pint dependency declaration does not change the locked package
versions. A one-voice/one-action parsing check used the instrumented build while
test workers were active: median 169.3, p99 10,476.0, maximum 28,522.1 and first
269.2 microseconds, with 33 misses in 750 callbacks at a 1,333.3-microsecond
budget. These contended timings are retained separately from ordinary-build
profile qualification.

## Additional work beyond the prompt

None.
