# Shared-sweep six-stage phaser

`ufor.audio_effects.Phaser` retains its input channels and uses six identical
first-order all-pass stages, with a one-frame feedback loop. It has no internal
LFO clock. Bind its bipolar `sweep` target to shared enveLFO or Patch output,
or supply ordinary parameter actions. Unbound sweep defaults to zero.

Frequency is `center_hz * 2 ** (depth_cents * sweep / 1200)`. Depth uses the
existing musical-cents unit, so `"2 octaves"` normalizes to 2,400 cents. The
default center is 800 Hz and default depth is 2,400 cents. Sweep is in [-1, 1]
and feedback is in [-0.95, 0.95]. Center, depth, sweep, feedback and mix all
accept automation and shared generator routes.

The implementation evaluates frequency in log2 space. It explicitly rejects
resolved frequencies at or above Nyquist and coefficients not representable
strictly between -1 and 1. Preparation checks the authored settings; each
resolved row must remain valid during rendering. There is no implicit clamp.

For coefficient `a = (tan(pi*f/rate)-1)/(tan(pi*f/rate)+1)`, each stage uses
the normalized lattice pair `y = a*x + b*memory` and
`new_memory = b*x-a*memory`, with `b = sqrt((1-a)*(1+a))`. Both expressions use
the old memory. This orthogonal update preserves the stage's energy balance
under rapid frequency changes. At fixed frequency it has the all-pass transfer
`(a+z**-1)/(1+a*z**-1)`. The six-stage output is fed back on the following frame.

Mix uses the common linear dry/wet rule, default 0.5. Bypass leaves the six
stages and feedback warm. The profile adds zero algorithmic latency; its phase
response and feedback delay are intentional DSP behavior. If placed behind a
latent processor, the graph's common audio, control and end-marker compensation
applies. Stop remains immediate at its exact frame.

Each channel retains six stage memories and one previous wet value. Preparation
allocates native storage before callbacks. Values below `state_floor` become
zero in retained memories. After input end, rendering drains until every stage
and feedback value is below `tail_threshold`. JSON and opaque native snapshots
retain those memories and the common controls, lifecycle and compensation
state. Invalid native snapshot shapes are rejected before replacing state.

The persistent Rust core serves native offline and callback rendering; NumPy
has an independent vector realization. The tests check analytical all-pass and
closed-loop frequency responses, both feedback signs, a fast shared-enveLFO
sweep against an impulse-energy bound, warm bypass, tail drain, parameter
automation and shared generator routes, JSON and opaque replay, upstream
Waveshaper compensation, and native coefficient-error recovery. Audio
regressions write paired one-second 48 kHz
WAV files.

The focused allocation audit passed 139 tests. Its 5,684 successful callbacks
and 15 failed callbacks recorded zero allocations, reallocations or frees.
Python preparation, offline bridging and snapshot conversion remain outside
that callback guarantee. These checks do not qualify audio-device deadlines.
The full ordinary suite passed 3,252 tests with 21 optional Rubber Band skips;
uFor passed 1,516 tests. Ruff, formatting, type checking, pyupgrade and Rust
formatting/lint checks passed.

After the test jobs finished, three sequential ordinary-build measurements used
two seconds of 48 kHz audio, 16 voices and 64 effect actions per callback. Times
are microseconds. Median/p99/maximum exclude the first block; misses include it.

| Frames | Median | p99 | Maximum | First | Budget | Misses |
| --- | --- | --- | --- | --- | --- | --- |
| 64 | 287.2 | 336.9 | 403.8 | 647.1 | 1,333.3 | 0 / 1,500 |
| 256 | 792.4 | 820.3 | 832.6 | 1,025.1 | 5,333.3 | 0 / 375 |
| 1,024 | 2,863.2 | 2,917.9 | 2,920.2 | 3,107.3 | 21,333.3 | 0 / 93 |

Reproduce with `scripts/benchmark_live.py --phaser --seconds '2 s' --voices 16
--effect-actions 64 --block-frames 64`, selecting the other listed block sizes
for the remaining rows. These are finite CPU measurements on the development
machine.

For a `phase` processor, a shared sine sweep can use:

```json
{
  "envelfos": {"motion": {"kind": "lfo", "shape": "sine", "rate": 0.5}},
  "bindings": [{"kind": "envelfo", "name": "motion", "reference": "motion"}],
  "modulation": {
    "sources": [{"name": "motion", "scope": "voice", "minimum": -1, "maximum": 1}],
    "parameters": [{
      "target": {"name": "phase", "parameter": "sweep"},
      "unit": "ratio", "scope": "voice", "minimum": -1, "maximum": 1, "default": 0
    }],
    "routes": [{
      "name": "motion", "source": "motion",
      "target": {"name": "phase", "parameter": "sweep"},
      "unit": "ratio", "operation": "add",
      "points": [{"input": -1, "amount": -1}, {"input": 1, "amount": 1}]
    }]
  }
}
```

## Additional work beyond the prompt

None.
