# enveLFO modulation of effects

Effect graphs use the same `envelfos`, `bindings`, `event_connections` and
`modulation` definitions as synth voices. Targets address an effect processor
by name and one of its existing automatable parameters. This applies to gain,
soft clipping, multiply mix, filters, granulators, tap delays and modulated delays.

Each graph owns one generator instance. Generator/source/parameter scope is
`voice`, meaning that graph instance, even for a master or host attachment.
Note-key, velocity and instrument control bindings are not effect bindings;
ordinary effect parameter actions remain available for host automation.
Definitions and ranges are checked by uFor; execution uses the existing shared
enveLFO machinery in Python and Rust.

The separate finite Rubber Band adapters retain their fixed pitch/ratio and
output-length contracts. They are not processors in this effect graph.

For example, add these fields to a graph with a `movement` modulated-delay
processor whose base delay is 0.005 seconds and maximum delay is 0.01 seconds:

```json
{
  "envelfos": {
    "sweep": {"kind": "lfo", "shape": "sine", "rate": "1/2"}
  },
  "bindings": [
    {"kind": "envelfo", "name": "sweep", "reference": "sweep"}
  ],
  "modulation": {
    "sources": [
      {"name": "sweep", "scope": "voice", "minimum": -1, "maximum": 1}
    ],
    "parameters": [{
      "target": {"name": "movement", "parameter": "base_delay_seconds"},
      "scope": "voice", "unit": "seconds",
      "minimum": 0.004, "maximum": 0.006, "default": 0.005
    }],
    "routes": [{
      "name": "sweep", "source": "sweep",
      "target": {"name": "movement", "parameter": "base_delay_seconds"},
      "unit": "seconds", "operation": "add",
      "points": [
        {"input": -1, "amount": -0.001},
        {"input": 1, "amount": 0.001}
      ]
    }]
  }
}
```

Set that processor's `depth_seconds` to zero to use enveLFO for its delay-time
motion. Its built-in oscillator then contributes no motion. A nonzero internal
depth can be composed with the externally modulated base, within the declared
delay span. Reads now use the [fractional delay and tap-retiming profile](delay-effects.md);
channel phase offsets retain their existing meaning.

Routes can also drive delay depth/rate/feedback, clipping drive, gain, wet/dry
mix, filter `<filter-name>-cutoff_hz`/`<filter-name>-q`, and the existing granular
parameters. Named envelopes, staged generators, SampleHold and Patch outputs use
the shared executor rather than effect-specific oscillator implementations.
Parameter actions update the base beneath the routes: additive contributions
are added to that current base, then multiplicative contributions are applied.
The result must remain within the declared parameter domain and prepared storage.
Native live filter-cutoff modulation requires its entire declared range to stay
between the filter's minimum frequency and Nyquist-ratio limit. Preparation
rejects ranges needing NumPy's boundary clamping instead of rendering them
differently.

Generators start at graph frame zero and continue through bypass and audio tails.
Ending the main input releases their envelopes/stages; stopping the graph stops
rendering. Effect instances are independent. A native live graph retains the
existing action profile, which excludes input-end and stop actions.

Beat-clock generators require a caller-owned `TempoMap` passed to
`effects.prepare(graph, sample_rate, tempo_map)`. The prepared clock is used by
both backends. A different tempo map fails snapshot restoration. Python effect
snapshots retain generator/Patch state as JSON; native live snapshots retain
their bounded Rust state. Preparation allocates scratch and event capacity
before callback execution.

## Verification

The full instrumented suite passed **3,150 tests, with 21 skipped**. The
[callback audit](live-callback-audit.md) measured zero Rust heap operations in
8,062 successful and eight failed callbacks. Fifteen focused regressions cover
the effect integrations, automation, clocks, release/tails, snapshots and
explicit native filter limits. Audio comparisons write paired 48 kHz WAVs of
at least one second. These are software checks, not device-deadline qualification.

## Additional work beyond the prompt

None.
