# Rubber Band pitch and time processing

Rubber Band is an optional preparation/finite-stream adapter. Enable the native
`rubberband` feature as described in the README. `crates/rubberband` contains the
C boundary; the main native crate forbids unsafe code independently of this
feature. The allocation-audit allocator remains its separate existing exception.

## Completed offline profile

`enge._native.rubberband_stretch(audio, ratio, pitch)` prepares finite audio at
48,000 Hz. Input is a contiguous float64 `(frames, channels)` NumPy array;
layout conversion uses planar float32, matching Rubber Band's C API. Samples
must be finite and representable as float32. Frame/channel counts must fit the
C API's unsigned integers. Empty input and durations rounding to zero frames
are rejected.

`ratio` is output duration divided by source duration. `pitch` is output pitch
divided by source pitch. Both must be positive and finite. The fixed profile
uses the R3/Finer engine, linked channels, shifted formants and one processing
thread. R3 owns its transient behavior; no R2 transient option is exposed.
All settings are fixed for the prepared segment. There is no mid-segment
automation, additional mode selector or graph processor variant.

The adapter studies the complete input before processing it, in chunks of at
most 4,096 frames and within the library's process-size limit. The second pass
also retrieves output after each chunk. The final study/process chunks carry
the final flag. Rubber Band's offline mode compensates startup padding/delay.
The returned array is ready for ordinary prepared-sample playback, with no
additional streaming latency imposed by this adapter.

Output contains exactly `round(source_frames * ratio)` frames, using nearest
rounding with ties to even. Any extra library rounding frame is removed at the
end; an output shortfall raises an error rather than adding fabricated silence.
Output duration must fit NumPy's addressable array size. There is no further
tail beyond this prepared boundary. Preparation owns the complete output and
may allocate memory and take substantial CPU time.

## Existing finite realtime and live profiles

`RubberBandRealtimeStretcher` accepts bounded input blocks, uses the existing
R2 realtime profile, supplies preferred startup padding, and removes the
reported output delay. A short final block is accepted. After the final call,
further calls fail. Its planar input/output conversion memory and pointer
tables are allocated at construction and reused; output arrays still allocate.

`enge.rubberband.RealtimeRubberBand` supplies a fixed output clock around that
stretcher. Source length, block size, ratio, pitch and queued-output limit are
fixed at construction. Startup is
`ceil(max(0, source_frames * (1-ratio))) + start_delay + 4*block_frames`.
The audible window lasts `round(source_frames * ratio)` frames. Its last output
block is zero-filled after that boundary. Shortfalls inside the window are
counted in `underrun_frames`; overflow of the queued-output limit closes the
segment and raises an error. Completion releases queued output and the native
stretcher. Invalid input is rejected before advancing the source position, so
the same clock block can be retried with valid data.

`RubberBandLiveShifter` uses the library's fixed block size and reported startup
delay. It preserves block length while shifting pitch, with fixed settings and
reused planar conversion memory. It does not remove startup delay automatically.

These Python/NumPy adapters allocate returned arrays and, for finite streaming,
queued output. The queue cap bounds retained output frames; it does not bound
Rubber Band's internal memory or the transient output returned by one call.
They are outside the prepared effects graph's allocation-free callback guarantee.
There is no portable snapshot of Rubber Band's opaque internal state, graph
bypass/routing automation, or claimed hardware deadline qualification.

## Distribution and reference audit

The locked `rubberband-sys` 0.1.4 crate bundles Rubber Band 4.0.0 and generates
bindings from its C API 3.0 header. Its build compiles the bundled single-file
C++ source into a static library; Apple targets link Accelerate. It needs a C++
compiler and libclang rather than a separately installed Rubber Band library.
No external dependency version changed in this work.

Source distributions include the unchanged registry source archive and notices
under `third_party`. Wheels include the upstream licence and explanatory notice.
The optional feature links GPL code; see
[the distribution notice](../third_party/RUBBERBAND-NOTICE.md).

Reference documentation:

- [Rubber Band C API](https://breakfastquay.com/rubberband/code-doc/rubberband-c_8h_source.html)
- [Stretcher modes, timing and processing](https://breakfastquay.com/rubberband/code-doc/classRubberBand_1_1RubberBandStretcher.html)

## Verification

The focused suite covers pitch and duration independently, linked antiphase
stereo, startup/delay handling, final drain, finite clock boundaries and queue
overflow, half-frame rounding, invalid float conversion and retry, and completed
segment rejection. Audio regressions write WAVs at 48 kHz lasting at least one
second. Frequency checks compare against authored tones rather than another
implementation of Rubber Band. Exact external-library waveform identity is not
claimed.

All 29 focused tests pass both in the editable build and against an isolated
installation of the feature-enabled wheel built from the source distribution.
The source archive contains the internal crate and locked upstream source;
the wheel contains the upstream licence and notice. Feature-enabled workspace
Clippy, core-only Clippy, formatting, Ruff, type checking, pyupgrade and lockfile
checks pass.
The full feature-enabled suite passed 3,281 tests with no skips on 2026-11-07.

The repository workflow has Linux, macOS and Windows jobs for both feature
profiles, requiring the Rubber Band tests when enabled. Local verification is
on macOS; that does not establish the result of unrun platform jobs.

## Additional work beyond the prompt

None.
