# MIDI performance binding conversion

`enge.performance.MidiConverter` is a caller-driven MIDI 1 adapter. It accepts
decoded meedee messages and emits uFor `Trigger`, `Release` and `ControlChange`
values. It opens no files, devices or audio streams and starts no threads.
The existing `enge.midi.read_midi()` importer is unchanged; connecting that
importer or a live host to this converter is a subsequent integration step.

## Preparation and reference ownership

Construct one converter per independent input source. Supply a
`PerformanceBindingScore` and its resolved `SynthInstrumentScore` or
`SampleInstrumentScore`. Synth, FM and noise share the synth instrument model.
The host resolves score references, verifies any hashes and ensures that the
supplied instrument is the referenced target. This adapter does not introduce
another score loader or infer identity from document names.
The host must serialize feed, loss, finish and restart calls on each instance;
the converter does not add threads or locks.

For each referenced tuning, supply its resolved uFor `Tuning` body in `tunings`,
keyed by `ScoreReference.key`: a relative path, or `selector:` followed by the
selector. Missing referenced tunings fail preparation. With no tuning reference,
notes use 12-tone equal temperament with A4 at 440 Hz. Finite tunings do not wrap
or fall back to equal temperament when a key is unavailable.

Preparation snapshots the binding and used tunings, and checks every CC/bend
target and output range against the instrument's control declarations. It does
not synthesize initial controller observations. Defaults remain the engine's
responsibility. Input channels are 1–16; output parts are the binding's named
parts rather than channel numbers. Several channels can share a part, while
their note-pairing queues remain separate.

## Feeding messages

```python
from enge.performance import MidiConverter

converter = MidiConverter(
    source="keyboard-a",
    binding=binding,
    instrument=instrument,
    tunings=resolved_tunings,
)
events = converter.feed(message, tick=destination_tick, ordinal=input_ordinal)
```

The host supplies integer destination ticks and nonnegative integer ordinals.
Pairs `(tick, ordinal)` must increase strictly, including ignored messages;
ordinals need only disambiguate equal ticks. Ticks can be negative for preroll.
No timestamps are rounded, clocks guessed or units converted. The host owns any
native-clock mapping and conversion to instrument frames.

Every nonzero NoteOn produces a fresh trigger identity with the source prefix.
Source names follow uFor identifier rules and must distinguish converter
instances whose outputs could be combined, including recreated instances.
`restart()` keeps the instance's ID counter, so abandoned identities are not
reused. NoteOff and zero-velocity NoteOn release the oldest or newest held
identity as selected by the binding, independently for each channel/key.
Velocity populates `Trigger.velocity`; no named `velocity` control is invented.

CC values follow the binding's declared linear ranges. Optional pitch bend uses
the explicit minimum/center/maximum mapping in
[uFor's contract](../../ufor/doc/performance-binding.md), reaching the endpoints
exactly. It emits a named part control and does not modify trigger pitch itself.
Unselected channels, unmapped CCs and unconfigured bend have no effect. Program
changes, pressure, timing messages and SysEx have no conversion in this source
profile. They return an empty list; the host may process them separately.
Sustain is only a mapped control, not an implicit converter note-lifetime rule.
UMP messages and compound commands are rejected explicitly; a shared `Message`
base does not make them primitive MIDI 1 input.

## Failure and restart

The default `max_held_notes=4096` bounds held identities across all selected
channels. A host may choose a different positive integer. Released identities
leave the budget even if an engine is still sounding a sustained voice or tail.
This is not an engine voice limit or a total-memory/CPU guarantee; definitions,
identifier lengths, integer sizes and application-retained outputs also cost
resources. There is no growing registry of sources or previously emitted events.

Overflow, unmatched releases, unavailable tuning notes, out-of-order input,
MIDI System Reset and unsupported message families raise `ValueError`, discard
pairing state and require an explicit `restart()`. Later conversion attempts
raise `RuntimeError` until then.
Ordinary argument validation rejects invalid values before consuming the input.
When a transport/decoder reports loss, the host must call `report_loss()`;
missing MIDI traffic is not independently detectable here.

`finish()` verifies that all observed notes were released and stops the instance
until restart; unreleased notes are an error. Neither failure, loss, finish nor
restart emits releases or restores controls. The host retains responsibility
for downstream voices and musical recovery. Never treat restarting observation
as evidence that an instrument is silent or synchronized.

This slice does not connect live transport, acquire an audio clock, render audio,
add an interface or replace the file importer's existing musical policy.
