# Reverse, scrub and stutter

T3 implements the shared `CapturedBuffer` profile in NumPy and the persistent
native effect graph. It provides triggered reverse, literal-position scrub and
stutter over recently recorded input, with independent named channels and shared
traversal. It adds no library dependency or intrinsic graph latency. Window
lookback is an intentional playback effect, rather than a delay imposed on dry
audio.

## Preparation and actions

Choose `mode='reverse'`, `'scrub'` or `'stutter'`. The fixed `window_seconds`
defaults to 0.25 and accepts positive durations up to two seconds. Duration rounds
to output frames with nearest ties-to-even; the result must contain at least
129 frames so the fixed 64-frame fades fit. At 48 kHz, the default window is
12,000 frames and the minimum is about 2.69 ms. Window and mode are preparation
settings. Scrub position, ordinary mix and warm bypass use existing controls.

`CaptureAction(tick=..., ordinal=..., processor=...)` captures the most recent
complete window before its aligned execution frame. Recording that frame comes
after capture, so block boundaries cannot change which samples are selected.
History must already be complete and the audio input must still be live.
`CaptureReleaseAction` returns captured playback to dry and is a lifecycle action
for batch admission. Stop silences the complete graph immediately.

Commands follow upstream latency to the processor's input timeline, preserving
frame and ordinal order. The per-processor FIFO reserves
`maximum_pending_actions`, default 64, during preparation. Queue exhaustion is
an explicit error. Pending commands survive JSON and native opaque snapshots,
including commands received before a snapshot but scheduled to execute after it.

Preparation reserves the recording ring and two captured regions. Recording
continues while playback is active, mixed out or bypassed. Recapture copies a
fresh chronological window into a free region. It rejects when both regions are
occupied by a crossfade. The copy performs work proportional to the authored
window and channel count, with no storage allocation in detached native DSP.
The two-second limit bounds capture work, but does not make it constant-time.

## Traversal and transitions

Before capture, wet output follows dry input. A capture linearly fades the
current dry/captured mixture into its new region over 64 frames. Recapture fades
the existing reader out while the new reader advances. Release retimes both
current reader gains toward zero over 64 frames, including during a capture
crossfade. An idle release is harmless. Recording and readers remain warm under
ordinary mix and bypass; bypass neither releases nor resets captured playback.

Reverse reads newest to oldest at one source frame per output frame. Its last
64 frames apply a separate linear tail envelope, reaching zero at the oldest
sample and blending back to dry. Release or recapture cannot extend that natural
ending. The last audible release sample is emitted before retirement.

Stutter has a 64-frame tail/head overlap seam and repeats every
`window_frames-64` frames. At cycle offset p below 64, blend captured tail sample
`period+p` with head sample p, using head weight p/64. Other offsets read p
directly. Capture begins at cycle offset zero, the start of the tail/head seam.
The default 250 ms window therefore repeats every 11,936 frames, approximately
248.67 ms at 48 kHz. This shortening is part of the selected profile.

Scrub maps position 0 to the oldest sample and 1 to the newest, with linear
interpolation. Position uses the ratio unit and accepts per-frame scoped
modulation, including enveLFO, and explicit parameter ramps. There is no autonomous
playback clock. Holding a position holds that sample value, potentially DC;
Highpass can remove it when desired. Abrupt position changes may click. Position
is rejected as an authored nonzero value or control target in the other modes.

## Endings, state and failures

When its aligned audio input ends, the processor stops recording and rejects
captures, including captures on that first ended frame. A live or decaying
follower control does not keep the audio recording input live. Already active
reverse finishes; stutter and scrub remain held until release or stop. Such a
held graph intentionally does not drain on input end, even when fully bypassed.
Once released, it retires after the complete fade. Capture after an offline graph
has retired is rejected too.

Snapshots retain the ring, chronological write/count state, both captured
regions, cursors, gain ramps and pending command FIFO, alongside existing graph
and modulation state. Native snapshot cloning/restoration preserves the reserved
FIFO capacity and reuses prepared history storage. A subsequent capture after
restoring an empty queue remains allocation-free in detached DSP.

Native callback errors silence that output block and latch failure until a valid
snapshot is restored. Focused regressions cover insufficient history, capture
after input end, occupied regions, queue exhaustion and successful recovery.
Snapshot/JSON operations themselves remain outside the callback processing
contract, as in the existing effects.

## Verification and reproduction

The shared uFor contract is `d9a0a5c`, following the foundation in `1b056bb`;
enveLFO `0ac7b57` only aligns its required Python metadata pin. The isolated uFor
suite passes 2,092 tests with one optional skip; enveLFO passes 188 with one
optional skip. Concurrent uFor development remains separate in its original
checkout. The existing `/Users/tom/code/ufor-effect-profiles` worktree contains
this contract's committed source and its non-editable test environment.

Audio regressions use 48 kHz WAV pairs, one second per ordinary case and three
seconds for the maximum-window case. Independent whole-array traversal and gain
equations cover all modes, overlap seams, fresh recapture, release retiming,
scoped scrub motion, warm bypass, aligned commands, portable/opaque replay,
pending commands at snapshots, held endings, stop and maximum capture storage.
Additional follower cases distinguish audio end from analysis-control lifetime.
All 23 focused cases pass. The final full suite passes 3,570 tests with 29 optional
skips. Ruff, formatting, type checks, pyupgrade, Cargo tests and Clippy pass.

The full detached DSP audit records 24,703 successful and 68 failed calls with
zero allocations, reallocations or frees. Its containing native method records
24,703 successful calls with 41,635 allocations and matching frees, and no
reallocations. Its 71 failed calls record 219 allocations, one reallocation and
146 frees, including admission errors and Python error construction. Error
cleanup can occur after the measured method returns. Whole-method counts remain
reported separately; only detached DSP is required zero-heap. The 23 focused
cases record 969 successful and six failed detached DSP calls, also zero-heap.

Twelve sequential ordinary release runs after tests retain every observation,
without selected repeats. All have zero misses. Each includes 16 synth voices
and the configured total of 64 effect actions per callback. Times are microseconds:

| Mode/window | Frames | Median | p99 | Maximum | First | Budget | Misses |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Reverse, 250 ms | 64 | 436.4 | 573.6 | 898.5 | 739.2 | 1333.3 | 0/1500 |
| Reverse, 250 ms | 256 | 954.2 | 1296.1 | 2269.5 | 1447.7 | 5333.3 | 0/375 |
| Reverse, 250 ms | 1024 | 2996.2 | 3264.0 | 3299.0 | 3349.6 | 21333.3 | 0/93 |
| Scrub, 250 ms | 64 | 403.4 | 537.1 | 923.2 | 658.6 | 1333.3 | 0/1500 |
| Scrub, 250 ms | 256 | 912.2 | 1057.6 | 1441.5 | 1190.9 | 5333.3 | 0/375 |
| Scrub, 250 ms | 1024 | 2996.9 | 3273.8 | 3313.3 | 3270.9 | 21333.3 | 0/93 |
| Stutter, 250 ms | 64 | 439.1 | 567.6 | 895.0 | 700.4 | 1333.3 | 0/1500 |
| Stutter, 250 ms | 256 | 958.1 | 1062.5 | 1253.4 | 1224.1 | 5333.3 | 0/375 |
| Stutter, 250 ms | 1024 | 3008.1 | 3210.2 | 3437.8 | 3222.8 | 21333.3 | 0/93 |
| Stutter, 2 s | 64 | 441.0 | 582.7 | 1097.0 | 726.6 | 1333.3 | 0/3000 |
| Stutter, 2 s | 256 | 951.8 | 1111.5 | 1364.6 | 1222.8 | 5333.3 | 0/750 |
| Stutter, 2 s | 1024 | 2979.4 | 3304.3 | 3719.6 | 3266.9 | 21333.3 | 0/187 |

These finite CPU observations do not establish audio-device deadline guarantees.
Reports under `build/verification` are `captured-buffer-qualified-full.xml`,
`captured-buffer-qualified-full-heap.json`, `captured-buffer-focused-qualified.xml`,
`captured-buffer-focused-qualified-heap.json`,
`captured-buffer-{reverse,scrub,stutter}-timing-{64,256,1024}.log` and
`captured-buffer-maximum-timing-{64,256,1024}.log`. Earlier diagnostic logs are
retained; the qualified full report is the final state after the follower-end
regressions and their correction.

Install declared pins without editable dependencies. Build `allocation-audit`
and ordinary release wheels with the existing Maturin workflow, install the
selected wheel and extract its `enge/_native*.so` into the checkout for tests.
Use `TMPDIR="$PWD/build/tmp"`, pytest `--basetemp` under `build/verification`,
`--require-callback-no-allocations` and JUnit output. The focused files are
`test/test_captured_buffer.py` and its `captured_buffer_oracle.py` helper.
Summarize with `scripts/summarize_callback_audit.py --reports REPORT`.

For ordinary CPU timing, use `scripts/benchmark_live.py --capture MODE
--seconds '2 s' --voices 16 --effect-actions 64`, for each mode and block sizes
64, 256 and 1024. The workload includes capture, fresh recapture and release on a
four-window cycle, replacing scalar actions to retain the configured total
action budget. Scrub also automates position. Measure the largest capture copy
with `--capture stutter --capture-window-seconds '2 s' --seconds '4 s'` at those
three sizes. That records two seconds before the first capture. Run sequentially
after tests, retain every observation and apply the agreed shared-machine policy
without claiming audio-device deadlines. Whole-call and detached DSP allocation
measurements remain separate under [the performance contract](performance-fixes.md).

Wheels, logs, WAVs and reports under `build/` are disposable generated outputs.
All development source and reproduction instructions are tracked.

## Additional work beyond the prompt

None. The bounded command FIFO and capture/release API are the approved extensions
needed for these three modes.
