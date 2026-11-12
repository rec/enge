# Matrix convolution

`Convolution` is a fixed, always-wet FIR matrix. Each kernel names its input
channel, output channel, audio asset and asset channel. Sparse channel pairs
contribute zero; declared pairs are summed without clipping, normalization or
trimming. Input and output layouts can differ. Stereo, true-stereo and other
explicit matrices use the same processor.

## Preparation and assets

Graphs declare local `assets` and `timebases` using the existing uFor types.
Call `effects.prepare(graph, 48000, audio={"room": decoded_audio})`, where each
decoded array has dtype float64 and shape `(declared_frames, declared_channels)`.
Supply exactly the referenced asset IDs. IRs must be finite and nonempty, and
their declared sample rates must match the render rate. Preparation does not
open paths, download audio or resample it. All tests use generated impulses.

Prepared PCM and NumPy impulse spectra have immutable byte owners. Changing the
original decoding array cannot change a prepared graph. SHA-256 content digests
bind portable snapshots to the decoded assets. Native snapshots also bind kernel
routes, extents, partition sizes and the actual kernel-channel content.

Preparation builds uniform frequency-domain partitions and retained FFT plans,
scratch and history. The native implementation uses RealFFT 3.5.1, backed by
RustFFT 6.4.1. Forward transforms are unscaled and inverse transforms divide by
their transform length, matching NumPy. The two implementations maintain the
same overlap-add and partition-history state. Native spectral sums use compensated
accumulation. See [RealFFT's API](https://docs.rs/realfft/3.5.1/realfft/) and its
[prepared-scratch transform](https://docs.rs/realfft/3.5.1/realfft/trait.RealToComplex.html#tymethod.process_with_scratch).

## Latency, routing and endings

`partition_frames` is a positive power of two, default 256. A partition of B
frames uses transforms of 2B frames and declares exactly B frames of latency.
The default adds 5.333 ms at 48 kHz. Callback lengths can cross partition
boundaries and need not divide or equal B. Changing IRs or partition size requires
preparation again; this profile has no IR replacement or per-node controls.

Use existing channel matrices and Mixer/Crossfader nodes for dry routing and
wet/dry controls. Graph alignment delays earlier audio, controls and end markers
to the latest dependency. A Crossfader can select the aligned dry branch while
convolution continues running, providing warm bypass without a second bypass
definition inside the FIR processor.

Input endings drain the complete declared FIR extent, including trailing zeros.
If input ends at exclusive frame T and the longest kernel has L frames, the
last possible output is at `T + L + B - 2`; retirement starts on the following
empty frame. No amplitude threshold shortens the FIR. Explicit graph stop still
cuts processing immediately. Failures silence the caller's whole block and
latch failure until restore/reset.

## Snapshots and verification

Portable JSON snapshots retain the partial input partition, queued output,
overlap, complex input-spectrum history, partition fill/slot and tail clock.
Restore checks shape, finite values, real DC/Nyquist bins and clock bounds before
replacing state. Native opaque snapshots retain the same state and prepared
impulse identity. Both formats support replay between arbitrary partition
boundaries; JSON also supports switching NumPy/native backends.

The shared schema/specification is uFor `43c565d`, with 1,567 tests passing.
Twenty-two focused enge tests pass. Their 48 kHz WAV regressions cover direct
time-domain convolution, true-stereo and sparse non-square kernels, IR lengths
on both sides of partition boundaries, B=1, dry-path alignment, one-second IRs,
callbacks of 1/17/255/256/257/997 frames, replay after input end, the final FIR
sample and trailing-zero extent. Preparation, corrupt-state rejection, immutable
ownership, asset identity, failed-block silence and recovery are also checked.
The expanded live/persistent-effects suite passes 94 tests.

The full instrumented suite passes 3,398 tests with 29 optional Rubber Band skips.
It records zero Rust allocations, reallocations or frees in 17,885 successful
and 39 failed callbacks. Rust tests, Clippy, formatting, Ruff, type checking and
pyupgrade pass.
Preparation, Python bridges, host admission and snapshot conversion are outside
the callback heap guarantee. Device deadlines require separate host/device
qualification.

## CPU timing

Three sequential ordinary-build runs each render two seconds at 48 kHz, with
16 synth voices, the existing gain/filter chain and 64 effect actions per
callback. Convolution uses a generated one-second true-stereo IR with four
independent channels, 256-frame partitions, and the deterministic seed/envelope
in `scripts/benchmark_live.py`. Times are microseconds. Median/p99/maximum omit
the first block; misses include it. Host admission and native processing are
both inside the measured interval.

| Frames | Median | p99 | Maximum | First | Budget | Misses |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 64 | 395.8 | 1,057.5 | 1,160.3 | 1,000.0 | 1,333.3 | 0 / 1,500 |
| 256 | 1,416.2 | 1,538.5 | 1,563.1 | 1,736.0 | 5,333.3 | 0 / 375 |
| 1,024 | 5,165.6 | 5,538.6 | 5,646.1 | 5,697.5 | 21,333.3 | 0 / 93 |

Reproduce with `scripts/benchmark_live.py --convolution --seconds '2 s'`
`--voices 16 --effect-actions 64` and the listed `--block-frames`. FFT work
occurs at partition boundaries, so smaller callbacks have a higher peak/median
ratio. These finite measurements qualify this machine and profile without
device I/O; they do not establish deadlines for arbitrary IR lengths, matrix
sizes, partition sizes or hosts.

## Additional work beyond the prompt

None.
