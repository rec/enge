# Pint units throughout enge and uFor

## Goal

Accept strings representing Pint quantities wherever an authored numerical value
has units. Replace project-specific unit parsing with one Pint-backed path, rather
than adding Pint beside the existing suffix parsers.

Musicians should be able to write `"20 ms"`, `"440 Hz"`, `"-6 dB"`, or
`"1/3 beat"` without knowing the renderer's internal representation. Preparation
must still produce the same bounded, numerical Python and native runtime data.

This document is a plan, not authorization to implement the migration or change
dependencies. It covers enge, its uFor document models, and the shared parsing
boundary used by their consumers. It does not propose a general repository cleanup.

## Current position

- uFor's base types generally declare implicit units around strict numerical
  magnitudes, such as seconds and frequency.
- `ufor/segments.py` parses a small set of duration suffixes itself and preserves
  exact rational seconds or beats.
- `ufor/motion.py` separately recognizes a Cycle rate's `Hz` suffix.
- Other portable models, including arpeggiation declarations, have additional
  musical-time parsing. An inventory must find these rather than assuming the two
  visible parsers are the complete list.
- Modulation declares semantic units, but many defaults, limits, route amounts,
  and graph values are plain numbers whose units depend on another declaration.
- recs already uses `reccy/configuration/units.py`: Pint-backed Pydantic types,
  Decimal magnitudes, authored provenance, and separate authored/runtime dumps.
  Its quantity regex only accepts a number and a single alphabetic unit; it also
  has a separate clock-string parser. Its dump helpers reject custom serializers,
  which uFor already uses. Reusing this implementation unchanged is insufficient.

Inspect current shared code at implementation time. Do not copy its parser into
uFor, nor apply its clock syntax or serialization restrictions without a concrete
requirement.

## Public input contract

1. A unit-bearing field accepts its existing numerical input in its existing
   canonical unit, and additionally accepts a Pint quantity string.
2. Convert compatible quantities into the field's canonical unit before applying
   its existing range and integer/exactness constraints. `"20 ms"` and `0.02`
   therefore mean the same thing in a seconds field.
3. Unitless numeric strings already accepted for exact rational coordinates retain
   that meaning. A bare numeric string must never choose a different clock.
4. A quantity with the wrong dimension or semantic family is an error. Report the
   field, expected unit, and supplied quantity clearly.
5. Reject booleans as quantities, nonfinite magnitudes, malformed expressions,
   unknown units, and conversions that violate a field's precision contract.
6. Keep current field names and canonical numerical units. This migration does not
   rename fields ending in `_seconds`, `_hz`, `_db`, or `_frames`.
7. Document the supported Pint quantity syntax, including prefixes, plural names,
   scientific notation, rational quantities, and compound units where meaningful.
   Do not introduce another project-specific expression grammar or an `eval` path.

Pint provides quantity parsing, unit definitions, and incompatible-dimension
errors. Domain validation remains our responsibility.
[Pint tutorial](https://pint.readthedocs.io/en/latest/getting/tutorial.html)

## Inventory and coverage

Build a checked inventory of fields, their canonical units, their source of unit
metadata, and their normalization boundary. Include nested models, collections,
defaults, aliases, public API arguments, and CLI inputs, not just scalar fields
with unit suffixes in their names.

| Family | Examples | Canonical treatment |
| --- | --- | --- |
| Physical time | delays, fades, stage durations, smoothing, grain history | Seconds; exact fractions for exact-time models |
| Musical time | beat durations, positions, quantization, tempo | Exact beats or declared tempo coordinates |
| Frequency and rate | tuning, filters, Cycles, grain density | Hz or the field's declared rate unit |
| Gain and pitch | dB gain, cents offsets, pitch ratios | Existing logarithmic coordinate or ratio |
| Phase and angles | radians, Cycle phase, phase offsets | Declared radians or fraction of a turn |
| Sample-grid time | frame offsets, crossfades, capacities, timebases | Integer frames when the field requires integers |
| Electrical and declared signal units | volts, seconds-valued signals | Unit declared by the port or parameter |
| Contextual values | parameter defaults/limits, route points, graph constants | Unit derived from the relevant declaration |
| Compound units | Slew rates, beats per minute, cycles per beat | Declared compound unit, not inferred from the name |

Audit audio effects, sampling/selection/variation, tuning, metadata, playback,
Motions, control timelines, modulation, and preparation/configuration APIs.
For integer frequency or storage fields that are actually present, retain their
integer requirements after conversion.

Not every number is a quantity. IDs, MIDI channels and keys, velocity codes,
hardware register values, operator algorithms, repeat counts, event counters,
selectors, and collection lengths remain their existing numerical types. A
Yamaha envelope `rate` code is not Hz merely because its name says rate.

Dimensionless mathematical expressions and protocol readers are not unit parsers.
Keep the pitch-ratio expression language and protocol-defined Scala/SysEx/MIDI
interpretation where needed. Convert actual quantities at their model boundary;
do not change external byte formats or reinterpret protocol codes through Pint.

## One shared authoring boundary

Use a single registry and shared conversion implementation for a given document
validation context. Define musical units and aliases in Pint definitions, not
string suffix branches. Registry construction must be deterministic and must not
silently override conflicting definitions.
[Pint unit definitions](https://pint.readthedocs.io/en/stable/advanced/defining.html)

Before implementation, decide ownership by inspecting reccy's dependency graph:

- Reuse its generic unit machinery if uFor can depend on it without pulling in
  unrelated host, audio, device, or service dependencies.
- Otherwise agree on a minimal shared ownership boundary before changing the
  architecture. Do not create a new project or introduce duplicate parsers by
  default. Preserve recs' existing behavior unless its migration is separately
  authorized.

Pydantic validators normalize inputs to the existing scalar or Fraction types.
Schemas must advertise quantity strings as well as numerical inputs. Contextual
unit checks belong at the model level that knows the declaration, rather than
guessing units in a generic float validator.

Do not pass Quantity objects, registries, parsing, or authored provenance into
render callbacks or native code. Rust receives validated magnitudes, integer
frames, or the existing exact-time representation. There is no second Rust unit
parser. Standalone document validation must remain usable without an audio stack;
verify that the chosen installation path does not require NumPy.

## Exact time and contextual conversion

Exact musical scheduling is a requirement, not an optional optimization.
Prove the registry's magnitude policy with rational and decimal quantities before
selecting it: `"1/3 s"`, `"0.1 ms"`, prefixes, and compound rational rates must
normalize without a binary-float round trip. Do not use `Fraction(float_value)`
as a substitute for exact parsing or silently approximate unsupported input.
Pint exposes configurable registry and magnitude behavior; verify the concrete
policy against the selected release.
[Pint registry API](https://pint.readthedocs.io/en/stable/api/base.html)

Define beats and sample frames as distinct contextual units, not aliases for
seconds. There is no global fixed beat-to-second or frame-to-second conversion:

- Beat conversion uses the declared clock and tempo map, including transport
  running/stopped state and seeks. A tempo change does not rewrite an authored
  beat duration into a fixed seconds duration.
- Frame conversion requires the applicable sample rate/timebase. Do not bake one
  renderer's sample rate into a process-global registry.
- An exact integer frame field rejects fractional results unless that specific
  API already defines a rounding policy.
- For the proposed signal delay, a positive physical duration becomes a buffer
  length by rounding up at preparation, with at least one frame. Keep that rule
  local to delay preparation, not the generic quantity converter. Beat-based
  signal delay remains a separate design decision, not part of this migration.

Explicit quantity units select seconds versus beats only where the existing
model permits either. Declared clocks and units must otherwise agree with input.
Cycle rates in Hz and cycles per beat likewise retain their different clocks.

## Semantic units need more than dimensionality

Pint considers radians dimensionless, and frequency conversions can involve
turn-versus-radian factors. Validate phase, angle, and cyclic frequency semantics
explicitly. Test a full turn and one cycle per second; never let dimensionality
alone silently equate radians, ratios, normalized controls, or logical values.
[Pint angular frequency](https://pint.readthedocs.io/en/stable/user/angular_frequency.html)

Pint's decibel conversion describes a power ratio. Audio amplitude gain commonly
uses `10 ** (db / 20)` instead. Accept dB quantities as logarithmic coordinates
and normalize their scale without inadvertently performing Pint's power-ratio
conversion on amplitude fields. Keep the existing DSP conversion at its current
boundary. Musical cents and octaves likewise require tested pitch-ratio semantics
and unambiguous definitions, not a collision with unrelated unit names.
[Pint logarithmic units](https://pint.readthedocs.io/en/stable/user/log_units.html)

Normalized, logical, ratio, and full-scale signal families remain distinct where
the schema distinguishes them. They do not become interchangeable just because
their underlying physical dimension is dimensionless.

For routes, additive amounts use the destination signal unit; multiplicative
amounts use ratios. For Patch nodes, propagate declared units into constants,
thresholds, Quantize steps/origins, Latch initial values, and Slew rates. Reject
incompatible connections; do not invent a physical unit for an undeclared signal.

## Serialization and consumer boundaries

Retain the current normalized numerical document/runtime representation where it
exists, and existing exact duration serialization where needed to distinguish
beats from seconds. Equivalent authored units must prepare identically and must
not produce different runtime cache identities merely because of spelling.

Do not add provenance wrappers or a new export mode solely for this migration.
If existing authored-string preservation is reused, prove it works with uFor's
custom serializers, exact fractions, copying, and revalidation. State clearly
whether an export preserves spelling or emits canonical values.

Apply the same input contract to files and public validation entry points. CLI
adapters should call that shared conversion, not parse suffixes independently.
Prepared arrays, packed event records, native structures, and external protocol
messages remain numerical. Audit consumers in enge and sibling projects before
changing shared uFor behavior; coordinate their updates rather than leaving
duplicate readers or stale parser imports.

## Implementation sequence

Each slice must be independently tested and committed. Dependency/lock changes
use separate commits from implementation changes.

1. Complete the inventory and decide shared ownership. Check the supported Pint
   release, dependency footprint, exact rational parsing, semantic definitions,
   and existing recs behavior. Resolve any architectural decision before coding.
2. Add the shared registry/conversion and Pydantic input types. Cover physical
   units, exact quantities, contextual units, error reporting, and schemas first.
3. Migrate fixed-unit uFor fields and replace duration/Hz/beat suffix parsing.
   Preserve clock distinctions, integer constraints, canonical values, and exact
   serialization. Remove superseded parsers in the same slice as their callers.
4. Migrate contextual parameter, modulation, and Patch fields, including collection
   elements and compound rate units. Add dimensional and semantic routing checks.
5. Update enge's authoring and preparation boundaries and affected consumers.
   Keep callbacks/native paths quantity-free. Coordinate any sibling changes and
   commit each affected project separately.
6. Complete a repository search for remaining project-specific unit parsers;
   classify legitimate protocol/math parsing explicitly. Update examples, public
   schemas, and documentation, and demonstrate the musician-facing delay syntax.

Do not claim completion after only adding convenient aliases for seconds and Hz.
Every inventory entry must be migrated or explicitly identified as non-quantity
data, and no alternate bespoke unit parser may remain for migrated values.

## Verification and acceptance

- Existing numerical documents retain their meaning and prepared results.
- Compatible strings cover seconds/ms/minutes, Hz/kHz, rational beats, declared
  rates, dB, cents, radians/turns, and contextual signal units.
- Wrong dimensions, semantic mismatches, undefined units, nonfinite quantities,
  fractional integer results, and missing timebase/tempo context fail clearly.
- Exact fractions survive validation, serialization, revalidation, tempo changes,
  seeks, and block partitioning without rounding drift.
- Equivalent units produce identical prepared records; Python/native parity and
  callback allocation constraints remain intact.
- Tests include nested collections, declared parameter defaults/limits, route
  operations, and graph values, not only top-level fields.
- Pure Python document validation is tested in an installation without NumPy or
  audio dependencies. Run Linux/Windows/macOS release verification under the
  existing release-only cross-platform policy, not on every commit.
- Any digital-audio regressions write at least one second of 48 kHz WAV output.
  Keep parser and model tests focused and numerical where no audio is involved.
- The public documentation states canonical units, contextual conversion rules,
  supported quantity syntax, and serialization behavior.

## Additional work beyond the prompt

None. This change only records the plan; implementation, dependency changes,
shared ownership changes, and sibling-project migrations remain future work.
