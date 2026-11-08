# Doubling, ensemble and ADT

These profiles compose existing ModulatedDelay, channel-matrix and Mixer nodes.
uFor's `modulated_multitap_graph` accepts a nonempty list of `ModulatedTap`
voices. Each voice holds an existing `ModulatedDelay` in `delay`, plus static
`gain_db` and `pan`. Voice names must be unique. The graph preserves all authored
delay bounds, waveform, per-channel phases, feedback, mix and tail settings.
It adds no new DSP implementation or snapshot type.

Mono input uses equal-power placement into named left/right output. Named
left/right stereo input uses attenuation-only balance. The dry branch maps mono
to both channels at minus 3 dB each or passes stereo unchanged. Dry gain is
explicit and defaults to 0 dB. Summation has no normalization or clipping.

For graph `ensemble` and voice `one`, the delay node is `ensemble-tap-one`, its
placement node is `ensemble-pan-one`, and mixer `ensemble` exposes input
`tap-one`. Base delay, depth, rate, feedback, mix and bypass use existing delay
targets. Mixer gain and mute/solo retain their existing controls. Musical delay
adds no graph alignment latency.

Voice count and maximum delays fix storage at preparation. Distinct authored
rates, phases, delays and placement decorrelate the reads deterministically.
There is no random drift or tape-transport model. For shared enveLFO motion, set
the relevant voice's internal depth to zero and bind its base delay to an
existing generator source. Independent signed routes can move voices in
opposite directions while advancing one shared generator clock. Declared
modulation domains must remain within each prepared delay span.

The named recipes in uFor's `conformance/modulated-multitap.json` use a 50 ms
maximum per voice, sine motion, unity wet mix and zero feedback:

| Recipe | Wet voices | Base delays (ms) | Depths (ms) | Rates (Hz) | Initial mono phases | Voice gains (dB) | Pan/balance | Dry gain (dB) |
| --- | ---: | --- | --- | --- | --- | --- | --- | ---: |
| Doubling | 2 | 12, 18 | 0.8, 1.2 | 0.41, 0.67 | 0, 0.37 | -9, -9 | -0.8, 0.8 | 0 |
| Ensemble | 4 | 15, 19, 23, 27 | 2, 2.5, 3, 1.8 | 0.37, 0.53, 0.71, 0.89 | 0.05, 0.27, 0.51, 0.78 | -12 each | -1, -0.35, 0.35, 1 | -3 |
| ADT | 1 | 23 | 3 | 0.31 | 0.23 | -6 | 0.5 | -3 |

These are explicit authoring recipes. Stereo qualification uses the listed
phase for the left channel and that phase plus 0.37 cycles, modulo one, for the
right. The helper accepts authored values rather than selecting hidden presets.

Input endings let every read history drain before the mixer ends. All histories,
oscillator phases, gain ramps, generator state and selector state use existing
portable JSON and native opaque snapshots. NumPy, native offline and native live
rendering share the same graph and lifecycle.

## Verification

The uFor definition, helper, specification and recipes are committed in
`1307c96`; 1,545 tests pass with static checks clean.

Seven one-second 48 kHz WAV regressions compare each recipe in mono and stereo
against independent moving fractional reads, placement and dB-gain formulas.
They exercise base-delay ramps, mixer-gain ramps, explicit input endings,
cross-backend JSON replay and native callback replay. The seventh case replaces
both internal sweeps with oppositely signed shared enveLFO routes and compares
them against an independent square-wave delay trajectory. All seven pass on the
ordinary build. No renderer source change is required for this composition.

The full instrumented suite passes 3,350 tests with 29 optional Rubber Band
skips. It records zero Rust allocations, reallocations or frees in 16,584
successful and 33 failed callbacks. Preparation, Python bridges, host admission
and snapshot conversion remain outside this callback audit. The unchanged
renderer and pre-existing tests plus these seven new cases also qualify the
separate dependency pin `6f174dc`. Ruff, formatting, type checking, pyupgrade and
diff checks pass.

## CPU timing

After all test jobs finished and the ordinary extension was restored, three
sequential runs used the four-voice stereo ensemble, 16 synth voices, 64
delay/gain actions per callback and two seconds of 48 kHz audio. Times are
microseconds; median/p99/maximum exclude the first block, while misses include
it. The timed interval includes host admission and native processing.

| Frames | Median | p99 | Maximum | First | Budget | Misses |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 64 | 433.2 | 573.6 | 1,204.6 | 785.7 | 1,333.3 | 0 / 1,500 |
| 256 | 1,107.9 | 1,275.7 | 1,355.8 | 1,393.8 | 5,333.3 | 0 / 375 |
| 1,024 | 3,857.2 | 3,992.6 | 4,070.9 | 4,124.7 | 21,333.3 | 0 / 93 |

Reproduce with `scripts/benchmark_live.py --delay ensemble --seconds '2 s'`
`--voices 16 --effect-actions 64` and the listed `--block-frames`. This covers
this development machine and authored profile, without device I/O. It does not
establish device deadlines or timing for arbitrary voice counts and settings.

## Additional work beyond the prompt

None.
