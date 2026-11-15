# Prepared vibrato and tremolo

T1 adds named graph-authoring helpers in uFor: `tremolo_graph` and `vibrato_graph`.
Both expand into existing processors and one shared enveLFO sine source. Enge
adds no DSP implementation, generator clock, portable schema variant or snapshot
type. The helpers and default recipes are in uFor `6362b75`; enveLFO `041b7dc`
only aligns its required Python dependency pin. Rust dependencies and runtime
source are unchanged.

## Tremolo

The helper composes a same-layout zero ChannelMatrix `<name>-silence` and a
linear Crossfader `<name>` with silence first and original audio second. The
crossfader's position is driven by shared sine motion:

    gain(t) = 1 - depth/2 + (depth/2) * sin(2*pi*(rate_hz*t + phase))

Depth is in [0,1]. Zero depth leaves audio unchanged; full depth reaches silence
at the sine minima. Gain never boosts above unity and there is no normalization.
Defaults are depth 0.5, rate 5 Hz and phase 0.25 cycles, starting at unity.
Channel names and layout are preserved. Every channel receives the same gain,
preserving the stereo image. No channel geometry is inferred.

These are always-applied routing nodes, with no internal bypass. A separate
outer dry blend can provide bypass. Input end and stop finish immediately;
there is no audio tail or intrinsic latency. Coefficients and graph buffers are
bounded by the authored channel count, including the zero matrix's square array.

## Vibrato

The helper uses one fully wet ModulatedDelay `<name>`, with zero feedback and
internal oscillator depth zero. Shared enveLFO drives base_delay_seconds:

    delay(t) = base_delay_seconds + depth_seconds * sin(2*pi*(rate_hz*t + phase))
    read_delay_frames = max(1, sample_rate * delay(t))

Defaults are base 5 ms, depth 0.5 ms, prepared maximum 10 ms, rate 5 Hz and phase
0 cycles. Require base greater than depth and base+depth no greater than the
maximum. Reads use the existing linear fractional interpolation. All channels
share the same moving read trajectory. There is no dry blend or feedback in the
prepared recipe, distinguishing this vibrato from chorus.

Depth is a delay-time excursion. Pitch movement depends on both depth and rate;
there is no exact symmetric cents promise. Extreme settings retain the existing
moving-delay behavior. Zero depth retains the explicit delay bias. The bias is
musical delay and produces startup silence; it is not declared graph alignment
latency, so graph latency stays zero. Input end drains the existing bounded
history; its capacity follows the prepared maximum and channel count. Mix and
warm bypass keep the history and shared motion advancing. Stop ends immediately.

## Controls and continuity

Rate, depth and initial phase are preparation settings for these named profiles.
Rate is finite and nonnegative in Hz; zero holds the initial phase. Phase is in
cycles [0,1). This profile provides no named live rate/depth controls. Expanded
processors retain ordinary graph actions; changing those nodes keeps their
existing meanings and must respect the authored modulation domains.

Each graph instance owns its shared sine source, `<name>-motion`, starting at
graph frame zero. The source uses seconds-clock voice scope, meaning that graph
instance. Generator state, phase, delay history, action ramps and ending state
use existing portable JSON and native opaque snapshots. Generators continue
through bypass and tails. No new clock or transport ownership is introduced.

## Verification

The isolated uFor suite passed 1,800 tests with one optional skip, and enveLFO
passed 188 tests with one optional skip. Nine one-second 48 kHz audio cases
compare tremolo against the independent gain formula and vibrato against direct
fractional source reads. They cover mono/stereo layouts, zero/default/full and
fractional depth, initial phase and rate, portable cross-backend and opaque
callback replay, retained delay bias, warm bypass, input endings and stop.
All nine pass. Their detached DSP audit records 666 successful calls with zero
allocations, reallocations or frees. Whole native methods record 1,332 allocations
and matching frees from safe NumPy borrowing, separately from detached DSP.

The full suite passed 3,537 tests with 29 optional skips. Detached DSP recorded
23,067 successful and 61 failed calls with zero allocations, reallocations or
frees in both groups. Ruff, formatting, type checks and pyupgrade pass. Native
and Python renderer source remain identical to the previously qualified
`672b44e`; fresh wheels include the required dependency metadata.

Both boundaries follow [the performance contract](performance-fixes.md). The
containing native method recorded 23,067 successful calls with 38,363 allocations,
zero reallocations and 38,363 frees. Its 64 failed calls recorded 196 allocations,
one reallocation and 130 frees, including admission errors and Python error
construction. Error cleanup may occur after the measured method returns.
Whole-method allocations are reported; only detached DSP is required zero-heap.

After tests, six sequential ordinary release runs requested two seconds with
16 synth voices and 64 existing gain/filter actions per callback. Each used one
selected default motion profile. All observations are retained, with no selected
repeats. Times are microseconds:

| Profile | Frames | Median | p99 | Maximum | First | Budget | Misses |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Tremolo | 64 | 463.3 | 552.5 | 585.1 | 736.9 | 1333.3 | 0/1500 |
| Tremolo | 256 | 1117.0 | 1222.8 | 1244.2 | 1349.2 | 5333.3 | 0/375 |
| Tremolo | 1024 | 3669.4 | 3790.0 | 3795.5 | 3785.6 | 21333.3 | 0/93 |
| Vibrato | 64 | 464.8 | 915.8 | 34843.4 | 706.2 | 1333.3 | 12/1500 |
| Vibrato | 256 | 1112.0 | 19231.7 | 37672.5 | 1297.9 | 5333.3 | 32/375 |
| Vibrato | 1024 | 3704.8 | 23071.0 | 25579.6 | 3809.9 | 21333.3 | 2/93 |

Vibrato's medians are comparable to tremolo's, while its observed long stalls
produce misses at all three sizes. Their cause is not qualified by these finite
runs. The agreed shared-machine policy retains timing misses without treating
them as blockers for functional/allocation qualification. No audio-device
deadline guarantee is established.

Raw reports are `build/verification/{tremolo,vibrato}-timing-{64,256,1024}.log`.
The full audit is `vibrato-tremolo-full.xml`, with summary
`vibrato-tremolo-full-heap.json` in the same directory. The nine-case focused
report is `vibrato-tremolo-focused.xml`.

## Reproduction

Install declared pins without editable dependencies. Build ordinary release and
`allocation-audit` feature wheels with the existing Maturin workflow, install
the selected wheel and extract its `enge/_native*.so` into the checkout for
source-tree tests. Use `TMPDIR="$PWD/build/tmp"`, pytest `--basetemp` under
`build/verification`, `--require-callback-no-allocations` and a JUnit report.
Summarize with `scripts/summarize_callback_audit.py --reports REPORT`.

For ordinary timing, run `scripts/benchmark_live.py --motion tremolo` and
`--motion vibrato` with `--seconds '2 s' --voices 16 --effect-actions 64`, once
each at block sizes 64, 256 and 1024. These compose the selected default recipe
after the existing gain/filter workload. Run sequentially after tests and retain
every result. Finite CPU observations do not establish audio-device deadlines.

Generated wheels, logs, reports and regression audio under `build/` are disposable.
All development source and reproduction procedures are tracked. The helper work
uses `/Users/tom/code/ufor-effect-profiles`, an isolated detached worktree needed
to preserve concurrent uFor modulation changes and its separate staging area.

## Additional work beyond the prompt

None. The required helper definitions, dependency pins and profile qualification
implement the approved prepared compositions.
