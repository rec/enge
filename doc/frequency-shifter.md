# Frequency shifting

T2 adds `audio_effects.FrequencyShifter` to both offline backends and the shared
native live graph. It preserves named channels independently, with one shared
oscillator phase. It translates frequencies by an additive number of hertz,
rather than multiplying pitch or producing both ring-modulator sidebands.
No FFT or new dependency is needed.

## Profile and controls

The fixed 1,025-tap Blackman-windowed Hilbert FIR has 512 frames of intrinsic
latency, about 10.67 ms at 48 kHz. This is declared graph latency: dry paths and
parallel graph branches align with the FIR real branch. Coefficients are prepared
once. The native kernel pairs antisymmetric taps, requiring 256 products per
channel per frame instead of evaluating all 1,025 taps.

For odd n, with n from 0 through 1024:

    h[n] = 2/(pi*(n-512)) *
           (0.42 - 0.5*cos(2*pi*n/1024) + 0.08*cos(4*pi*n/1024))

Even taps are exactly zero. No coefficient normalization is applied. At frame t:

    real = input[t-512]
    quadrature = sum(h[n] * input[t-n])
    wet = real*cos(2*pi*phase) - quadrature*sin(2*pi*phase)
    phase = (phase + shift_hz/sample_rate) modulo 1

Phase and history start at zero. `shift_hz` defaults to zero and accepts
[-2000,2000] Hz, including ramps and per-frame scoped modulation. Positive shifts
translate upward. Automation changes the phase increment without resetting
phase. Returning to zero holds the accumulated phase, so it need not restore an
exact delayed identity; initial zero shift does. Use mix or bypass for dry audio.
The shift applies on the current output frame after upstream alignment.

Scoped modulation uses the explicit `hz_offset` unit, a signed frequency
difference. Absolute `hz` targets retain their positive-domain validation.
This unit is defined in both shared Python and Rust modulation contracts, with
common quantity and round-trip conformance cases.

The FIR approximates a Hilbert transform away from DC and Nyquist. At 48 kHz,
the response test bounds unwanted-sideband amplitude below 0.001 of the wanted
sideband, at least 60 dB rejection, over a 2,001-point grid from 200 Hz through
23.8 kHz. This is a finite measured grid, not a continuous-band proof. The
normalized transition regions scale with sample rate. Near the boundaries,
rejection weakens. Negative translated frequencies reflect at DC; translated
frequencies above Nyquist alias. There are no protective filters or feedback.

Mix defaults fully wet. Mix and warm bypass blend against the real branch,
delayed by 512 frames. Their commands receive the same intrinsic delay; shift
commands control the output oscillator directly. FIR history and phase continue
through bypass. Explicit input end supplies zeros and drains 1,024 history
frames. Stop silences immediately. Storage is fixed by authored channel count.

Portable JSON and native opaque snapshots retain history, write index,
oscillator phase, remaining tail and existing graph/control state. Invalid native
shift actions fail with fixed-size errors, silence the callback and latch failure;
restoring a saved state enables deterministic replay.

## Verification

Ten one-second 48 kHz audio regressions cover both signed shift limits, zero
shift, independent stereo histories, scoped motion through zero, parameter ramps,
warm bypass, Nyquist aliasing, graph/internal dry alignment, finite tails, stop,
failed-callback recovery and cross-backend/opaque replay. The oracle uses
whole-signal convolution and accurately summed oscillator phase rather than the
streaming Rust paired-tap FIR. WAV pairs are written before comparison.

uFor passes 1,973 tests with one optional skip, including portable signed-unit
cases. Its Rust suite and Clippy pass. enveLFO passes 188 tests with one optional
skip after aligning its required Python uFor pin. Its runtime is unchanged.

The full enge suite passes 3,547 tests with 29 optional skips. Ruff, formatting,
type checks, pyupgrade, Cargo tests and Clippy pass. The required shared pins are
uFor `9304264` and enveLFO `c649304`; the initial processor definition is `7ccbf25`.

The full detached DSP audit records 23,734 successful and 62 failed calls with
zero allocations, reallocations or frees. The containing native method records
23,734 successful calls with 39,697 allocations and matching frees, and no
reallocations. Its 65 failed calls record 199 allocations, one reallocation and
132 frees, including admission errors and Python error construction. Error
cleanup may occur after the measured method returns. Whole-method allocations
remain reported separately; only detached DSP is required zero-heap. The focused
ten cases record 667 successful and one failed detached DSP call, also zero-heap.

Three sequential ordinary release runs after the full suite requested two
seconds with 16 synth voices and 64 existing gain/filter actions per callback,
plus the stereo +250 Hz shifter. All observations are retained without selected
repeats. Times are microseconds:

| Frames | Median | p99 | Maximum | First | Budget | Misses |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 64 | 474.3 | 757.4 | 1755.6 | 967.8 | 1333.3 | 2/1500 |
| 256 | 1086.2 | 1245.0 | 1471.9 | 1306.0 | 5333.3 | 0/375 |
| 1024 | 3566.9 | 3733.5 | 3815.8 | 3947.8 | 21333.3 | 0/93 |

The agreed shared-machine policy retains the two 64-frame timing misses without
treating them as blockers for functional/allocation qualification. Their cause
is not established. Finite CPU observations do not establish an audio-device
deadline guarantee.

Reports are under `build/verification`: `frequency-shifter-full.xml`,
`frequency-shifter-full-heap.json`, `frequency-shifter-final-focused.xml`,
`frequency-shifter-focused-heap.json` and
`frequency-shifter-timing-{64,256,1024}.log`. Shared-library checks are
`frequency-offset-ufor-final.log`, `frequency-offset-rust-full.log`,
`frequency-offset-rust-clippy.log` and `frequency-offset-envelfo.log`.

## Reproduction

Install the declared pins without editable dependencies. Build ordinary and
`allocation-audit` release wheels with the existing Maturin workflow; install the
selected wheel and extract its `enge/_native*.so` into the checkout for tests.
Use `TMPDIR="$PWD/build/tmp"` and pytest `--basetemp` under `build/verification`.
The focused test is `test/test_frequency_shifter.py`; audit runs use
`--require-callback-no-allocations` and JUnit output. Summarize reports with
`scripts/summarize_callback_audit.py --reports REPORT`.

For timing, use `scripts/benchmark_live.py --motion frequency-shift
--seconds '2 s' --voices 16 --effect-actions 64` at block sizes 64, 256 and 1024.
This adds a stereo +250 Hz shifter to the existing synth/gain/filter workload.
Run sequentially after tests and retain all results. Whole native method and
detached DSP allocation measurements remain separate, following
[the performance contract](performance-fixes.md).

Generated wheels, WAVs, logs and reports under `build/` are disposable. All source,
regression logic and reproduction instructions are tracked.

## Additional work beyond the prompt

None. The signed frequency-offset unit is the approved contract extension needed
for frequency-shift modulation.
