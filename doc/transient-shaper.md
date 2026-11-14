# Transient shaper

The selected N6 profile uses uFor `eac3270` and enveLFO `6d6bfe9`. It reuses
the existing linked peak followers, scoped controls, enveLFO routes, warm
bypass and NumPy/native effect graph. The detector is shared across all named channels; audio keeps its
channel layout and receives one common gain. There is no intrinsic latency.

## Detector and shaping law

Take the greatest absolute input sample across channels on each frame. Both
followers start at zero and use the existing exponential law:
`level += (peak-level) * (1-exp(-1/(sample_rate*seconds)))`.
Choose the attack time when peak exceeds level, otherwise release.

| Follower | Attack | Release |
| --- | --- | --- |
| Fast | 1 ms | 10 ms |
| Slow | 10 ms | 100 ms |

Levels below 1e-30 become zero. Define
`contrast = (fast-slow) / max(fast, slow, 1e-12)` and
`gain_db = attack_gain_db*max(contrast,0) + sustain_gain_db*max(-contrast,0)`.
Multiply every input channel by `10^(gain_db/20)`. The normalization floor
reduces the response at extremely quiet levels. Contrast stays in [-1,1],
and only one gain contribution is active at a time.

`attack_gain_db` and `sustain_gain_db` each range from -24 through +24 dB,
with neutral zero defaults. Mix defaults to one. Positive attack emphasizes
rising transients; negative attack softens them. Positive sustain emphasizes
decaying sound; negative sustain shortens its apparent decay. A steady input
returns toward unity gain as the followers converge. Sustain is not a separate
gain for an indefinitely held steady level.

Detector times and floors are fixed in this profile. Both gain controls and
mix accept ordinary ramps, bindings and shared enveLFO modulation. Existing
graph machinery compensates upstream latency for audio, controls and bypass.
There is no extra gain smoother, look-ahead, automatic normalization or output
ceiling. The neutral profile preserves the input exactly. Non-finite input or
output is an error; native callback errors silence the whole block and latch
failure until restore.

## Lifecycle and snapshots

Bypass continues running both followers and uses the existing 64-frame fade
contract. Input end retires on that frame and clears both levels: detector
memory generates no audio tail. Stop is immediate. Portable snapshots retain
two finite, nonnegative scalar levels, and validate them before replacing the
renderer. Native opaque snapshots retain the same state. Native processing
uses prepared storage and the existing fixed-size callback error path.

Captured exponential envelopes remain supported by NumPy effects. Native
effect preparation rejects captured stages in either the body or release,
including envelopes declared inside a Patch. Their shared Rust continuation
currently allocates during callbacks, so accepting them would break the
native effect allocation contract. Ordinary envelope and LFO modulation
remain supported. This boundary does not change native synth support.

## Verification

uFor's complete suite passes 1,608 tests and enveLFO's passes 217 tests. The
independent oracle solves the follower exponentials in closed form over
piecewise-constant peak intervals. It covers
positive/negative shaping, channel leadership changes, exact neutral gain,
quiet normalization and the state floor. Other regressions cover gain ramps,
upstream compensation, warm bypass, immediate stop/end, cross-backend JSON
and native opaque replay, invalid controls, output overflow and malformed
snapshot rejection. Audio regressions write paired 48 kHz WAVs lasting at
least one second. Scoped enveLFO gain modulation and generator release are
covered. A captured-envelope regression verifies NumPy modulation and portable
replay against the analytical curve, plus explicit native preparation rejection.
The final combined-state suite passes 3,487 tests with 29 optional skips.
The allocation audit records zero allocations, reallocations or frees in
21,185 successful and 57 failed callbacks. Python lint, formatting, type checks
and pyupgrade, plus Rust formatting, Clippy and unit tests pass.
The complete report is `build/verification/n6-integrated-full.xml`, with its
heap summary in `build/verification/n6-integrated-heap.json`.

Use the same instrumented wheel for the installed package and source checkout.
With one current-platform wheel in the output directory, POSIX-shell commands
are:

```sh
mkdir -p build/tmp build/wheels/n6-integrated-audit build/verification
TMPDIR="$PWD/build/tmp" uv sync --locked --no-editable
TMPDIR="$PWD/build/tmp" uv run --with maturin --no-sync maturin build \
  --release --locked --features allocation-audit --out build/wheels/n6-integrated-audit
uv pip install --no-deps --reinstall build/wheels/n6-integrated-audit/enge-*.whl
unzip -o -j build/wheels/n6-integrated-audit/enge-*.whl 'enge/_native*' -d enge
TMPDIR="$PWD/build/tmp" uv run --no-sync python -m pytest \
  --require-callback-no-allocations \
  --basetemp=build/verification/n6-integrated-full-temp \
  --junitxml=build/verification/n6-integrated-full.xml
uv run --no-sync python scripts/summarize_callback_audit.py \
  --reports build/verification/n6-integrated-full.xml
```

For ordinary CPU timing, rebuild without `allocation-audit`, install/extract
that wheel and use `scripts/benchmark_live.py --nonlinear transient`. This
benchmark selects attack +12 dB and sustain -6 dB, alongside the common
16-voice, 64-action chain. CPU measurements exclude audio-device I/O.

### Ordinary CPU measurements, 2026-11-14

Sequential ordinary-release runs requested two seconds at 48 kHz, with the
settings above. The 1024-frame run renders 93 whole callbacks. Times below
are microseconds. Median/p99/max exclude the first callback; misses include it.

| Frames | Median | p99 | Maximum | First | Budget | Misses |
| --- | --- | --- | --- | --- | --- | --- |
| 64 | 422.7 | 513.2 | 560.8 | 736.7 | 1333.3 | 0/1500 |
| 256 | 1170.2 | 1444.8 | 2021.6 | 1507.5 | 5333.3 | 0/375 |
| 1024 | 3774.0 | 3876.2 | 3897.1 | 4212.8 | 21333.3 | 0/93 |

These combined-state runs have zero misses. Logs are
`build/verification/n6-integrated-transient-{64,256,1024}.log`. These are
shared-machine CPU measurements, without an audio-device deadline
qualification. Earlier pre-integration reports remain as historical evidence.

Implementation, oracle and verification source are tracked. Reports, wheels
and regression audio under `build/` are disposable generated artifacts.

## Additional work beyond the prompt

The captured-envelope workstream landed on main as `af2c476` during this slice.
This integration preserves its synth implementation and voice-amplitude
rejection policy. The user approved an explicit native-effect boundary after
the captured-envelope regression measured 48,086 allocations and frees in 74
successful callbacks. Fixing shared captured-envelope storage is a separate
slice; no parallel envelope renderer is introduced here. The matching enveLFO
dependency-only pin update keeps its uFor revision aligned with enge's.
