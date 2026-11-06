# Eight-line algorithmic reverb

`ufor.audio_effects.Reverb` renders through the independent NumPy reference and
the persistent Rust DSP core shared by native offline and callback owners.
The portable definition and numerical profile were introduced in uFor
`9c2336f`; the enge dependency pin also includes the explicit granulator duration
cap in `08010e5`.

The input is mono or named left/right stereo. Output is always named left/right.
Mono maps to both channels at `1/sqrt(2)` before dry mixing and wet injection;
stereo passes through that mapping unchanged. Mix and bypass use the same linear
wet/dry rule as the other effects. The network continues processing while
bypassed. There is no additional algorithmic latency: predelay and network
propagation are intentional parts of the wet response.

Predelay accepts zero and fractional frames. Its stereo ring contains
`ceil(maximum_predelay_seconds * sample_rate) + 1` frames. The current mapped
input is written before a predelay read, so zero selects the current input.
Retiming shares the fixed-tap delay's continuous read-head mixture and bounded
capacity of `retime_fade_frames + 1`. Existing retiring heads finish on schedule;
each new destination takes the mixture's remaining gain.

The feedback network has eight scalar delay lines with lengths from
`ufor.audio_effects.reverb_delay_frames`. At 48 kHz they contain 1,426, 1,781,
1,973, 2,098, 2,549, 2,962, 3,422 and 3,826 frames. Each frame reads all lines,
forms the specified stereo output, applies one-pole damping, and feeds the
Householder matrix `I - 11ᵀ/4`. Each line has gain
`10 ** (-3 * line_frames / (sample_rate * decay_seconds))`.
The signed injection/output matrices follow uFor's numerical specification.
Decaying network and damping values below `state_floor` become zero. Predelay
retains the raw mapped input.

Decay, damping, predelay and mix support prepared actions and shared enveLFO
routes. Coefficients must remain representable below unity; an invalid native
callback fails with a fixed-size Rust error, silences the output block and can
recover from a snapshot.

Input end drains predelay, network and damping state below `tail_threshold`.
Maintained counts track non-silent history values, avoiding full-history scans
inside callbacks. Stop silences its exact frame and later blocks. JSON snapshots
contain the explicit predelay heads/history, network lines/write indices,
damping and tail count. Native imports validate these before replacing state.
Opaque live snapshots preserve the same DSP plus live controls and lifecycle.

## Verification

The focused regression suite covers mono/stereo impulse responses, zero and
half-frame predelay, exact dry mapping, dense retiming across snapshot boundaries,
bypass, damping/decay ramps, end/stop, shared enveLFO, cross-backend JSON replay,
exact native replay and coefficient failure/recovery. An independent first-echo
calculation checks the injection and output matrices. Every audio regression
writes paired 48 kHz WAVs lasting one second.

The 64-test allocation audit, including the existing persistent-state and delay
regressions, recorded zero Rust allocations, reallocations or frees in 2,039
successful callbacks and four failed callbacks.
The full ordinary suite passed 3,222 tests with 21 skipped. Ruff, formatting,
Python type checks, pyupgrade, Rust formatting and Clippy checks are clean.

After the test jobs finished, three sequential ordinary-build callback
measurements used 48 kHz audio, 16 synth voices, reverb and 64 effect actions per
block for two seconds. Times are microseconds. Median/p99/maximum exclude the
first block; deadline misses include it.

| Frames | Median | p99 | Maximum | First | Budget | Misses |
| --- | --- | --- | --- | --- | --- | --- |
| 64 | 281.9 | 315.5 | 409.0 | 545.8 | 1,333.3 | 0 / 1,500 |
| 256 | 838.4 | 953.7 | 1,120.4 | 1,236.1 | 5,333.3 | 0 / 375 |
| 1,024 | 3,032.2 | 3,165.0 | 3,174.2 | 3,374.2 | 21,333.3 | 0 / 93 |

Reproduce with `scripts/benchmark_live.py --reverb --seconds 2 --voices 16
--effect-actions 64 --block-frames FRAMES`. This finite CPU harness performs no
device I/O and does not establish physical-device deadlines.

## Additional work beyond the prompt

None.
