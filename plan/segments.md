# Human-facing segments

Replace the separate authored envelope segments and mix-automation knots with
one readable segment notation in ufor, consumed by enge and recs. This is a
schema and renderer migration, not implementation of the broader motion system.

```toml
initial = 1.0
segments = [
  { duration = "200 ms", to = 0.0 },
]
```

An envelope uses the same segment type for release:

```toml
release = [{ duration = "200 ms", to = 0.0 }]
```

Release starts from the current envelope value. Mix automation additionally
declares its target and start position; those remain separate from the shape.
Existing control-clip placement and source offsets retain their meaning.

## 1. Shared types and notation in ufor

- Add one shared duration parser and serializer. Accept explicit units such as
  `"200 ms"`, `"0.2 s"`, and `"1/4 beat"`; parse decimal and fractional numbers
  exactly, without a floating-point intermediate. Frame durations require a
  declared timebase. Beats require a resolvable musical clock; reject them in
  execution paths without one. Do not add tempo synchronization as part of this
  change.
- Move the reusable segment definition out of `ufor/envelope.py`. A segment has
  `duration` and `to`, with linear interpolation by default. Share an explicit
  interpolation contract that preserves existing held, curved-envelope, and
  equal-power gain behavior. Keep restrictions on the consuming quantity:
  logical gates use held segments; equal-power interpolation applies to gains.
- Use this type for `Envelope.segments`, `Envelope.release`, and `Curve.segments`.
  Replace `Segment.target` with `Segment.to` everywhere. Preserve envelope
  polarity, hold, retrigger, and release semantics.
- Change `ufor/automation.py`'s authored `TimelineCurve` from `knots` to a start
  position, `initial`, and `segments`. Convert each old knot interval into a
  segment whose duration is the interval and whose destination is the next
  knot's value. A single-knot curve becomes a constant with no segments.
  Preserve target units, additive/multiplicative routing, the pre-start value,
  and holding the final value after completion.

Apply this grammar to authored segment durations first. Do not indiscriminately
change every rational field: oscillator rates, absolute positions, and durations
are different types.

## 2. Keep exact timing inside the engines

Resolve units against the declared timebase during preparation. Accumulate
segment boundaries exactly; do not round each duration separately, which would
make long sequences drift. Convert absolute boundaries using the existing
consumer's documented sampling rule. Preserve fractional envelope boundaries
and recs' sample/tick placement behavior rather than silently making their
rounding rules identical.

Zero-duration envelope segments remain instantaneous changes. A held segment
keeps its entry value until its endpoint, then adopts `to`. Ordered simultaneous
instantaneous segments resolve in authored order. Release captures the value at
the effective release time, including minimum-hold behavior. Prepared arrays and
native state stay numeric; Rust callbacks never parse unit strings.

## 3. Update consumers and authored files together

- In enge, update `synth.py`'s envelope sampling and runtime preparation, FM and
  noise envelope consumers, native preparation helpers, and affected snapshots
  or bindings that refer to segment fields. Preserve the existing DSP formulas.
- In recs, update `recs/edit/commands.py` to emit segment-based automation and
  `recs/edit/automation.py` to consume prepared segments. Keep interpolation in
  agreement with ufor's scalar reference evaluator. Do not build a second mix
  renderer or change arrangement placement/routing.
- Update ufor's codecs and affected public parameter references, then repository
  presets, examples, fixtures, documentation, and code constructing these types.
  Serialization should emit explicit units and `to` consistently.
- Replace the old authored forms completely. Do not retain `target`/`to` aliases
  or parallel knot/segment authoring paths. Inventory affected persisted scores
  and snapshots before implementation; explicitly decide the format-version
  impact and whether a one-time converter is needed. A converter, if required,
  is a migration step, not a second runtime schema.

## 4. Verification and delivery

Verify equivalent duration spellings, exact fractional timing, invalid units,
missing clock bindings, and serialization round trips. Compare old behavior with
the new representation for linear/curved releases, held steps, equal-power mix
fades, single-value automation, and nonzero clip/source offsets. Cover zero-time
segments, fractional boundaries, long sequences, block partitions, and snapshot
continuation. Audio regressions use WAV files at 48 kHz lasting at least one
second.

Implement in coordinated ufor, enge, and recs commits, with each repository's
required checks passing. Completion means authors use one segment notation while
existing sound, placement, interpolation, and exact timing remain unchanged.

## Additional work beyond the prompt

None. This change adds the plan only.
