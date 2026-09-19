# Control evaluation optimization 1

## Scope and commits

Replace per-sample Python control/modulation evaluation with NumPy array
operations, and expose a configurable control evaluation interval. Preserve the
independent uFor scalar semantics as the test oracle. Commit this plan first,
commit implementation and permanent documentation separately, then delete this
file in a final separate commit. No dependency or uFor format changes are needed.

## Exact vectorization

Generate constant controls and held linear control ramps as arrays. Resolve
rational event anchors and ramp endpoints before float conversion, including
large frame positions. Apply the existing route knot rules with vectorized
indexing, preserving step-at-knot behavior, route sorting, weighted neutral
values, additive summation, ordered multiplication, and domain errors. Do not
construct Pydantic models or Fractions per sample. Keep event-time ramp updates
and scalar validation outside the numerical block.

Vectorize NumPy LFO waveforms using the exact discontinuity spans already prepared
for native rendering. NumPy evaluates its own waveform arithmetic, while Rust
continues to execute its existing independent kernel. Existing amplitude
envelopes are already vectorized and remain unchanged. Shared sources in an
instrument render span should be evaluated once, not once per voice; keep any
cache bounded to that span and invalidate it on control/context changes or restore.

## Configurable interval

Add positive integer `control_interval`, default 1, to all four offline engines,
the control renderer, and offline MIDI/Bach entry points. Store it in snapshots
and reject mismatches, including silent snapshots. It is independent of the
audio buffer size and cannot delay note/control events.

Interval 1 keeps full-resolution control semantics. Larger intervals specify the
maximum spacing of sampled curved control signals: evaluate sine LFO knots and
linearly interpolate their values. Anchor knots to the LFO event state, not each
render call; derive knot positions with integer arithmetic. This approximation
must be independent of audio block partitions and JSON continuation.

Do not approximate what is already inexpensive and exact: constants, linear
ramps, envelope attack/release, activation delay/fade, square LFO edges, and
triangle corners retain their current sample boundaries. Route mappings and
domain validation operate on the resulting full-resolution arrays, so step
routes are never interpolated across their discontinuities. This interval is
not a general decimation of the final parameter arrays. Sine LFOs at or above
the chosen control-rate Nyquist frequency retain full-rate evaluation to avoid
introducing control-rate aliasing. FM operator modulation/feedback stays at
audio rate. No extra smoothing is added.

Evaluate future knots only from the current LFO state; never anticipate queued
rate/reset events. A new event replaces that trajectory at its addressed sample.
Larger intervals intentionally alter curved modulation and cannot promise
inaudibility. Default sound must remain within existing regression tolerances.

## Validation and measurement

- Compare vectorized control ramps and routes against uFor's scalar evaluator,
  including interrupted ramps, same-frame changes, fractional endpoints, step
  and linear maps, cancellation, weighted add/multiply, and invalid domains.
- Prove the optimized path does not call scalar evaluators per sample.
- Run existing synth, sampler, FM, noise, and LFO regressions on both backends.
  Keep WAV regressions at 48 kHz and at least one second long.
- Compare intervals 1/64/1024 against independent interpolation expectations;
  verify default parity, error bounds for slow sine modulation, events,
  discontinuities, arbitrary audio partitions, and snapshots.
- Re-run the four-voice noise benchmark at the same settings as the previous
  measurement. Also measure control/LFO evaluation separately and record results
  in permanent documentation. Do not infer callback safety from offline timing.
- Run pytest, Ruff, formatting, ty, pyupgrade, Cargo checks where applicable, and
  diff checks. Report unrelated existing failures separately.

## Additional work beyond the prompt

None.
