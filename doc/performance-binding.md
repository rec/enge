# MIDI performance binding conversion

`enge.performance.MidiConverter` is a caller-driven MIDI 1 adapter. It accepts
decoded meedee messages and emits uFor `Trigger`, `Release` and `ControlChange`
values. It opens no files, devices or audio streams and starts no threads.
`enge.midi.read_midi()` supplies file timing to configured converters. Live
transport and audio-clock integration remain host responsibilities.

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

## MIDI-file import and rendering

`read_midi(path, *, converters=[...], sample_rate=48000)` requires at least one
ready converter. `ready` is true on construction or after an explicit restart;
an instance already fed messages or finished is not ready. Read-only `source`
and `parts` describe the converter's configured routing. Source names and part
names must be distinct across converters. Channels may overlap across converters
for deliberate layering; channels sharing a part within one converter remain
supported. Each converter retains its own held-note budget.

The importer checks arguments and routing before opening the file. It supports
synchronous PPQN SMF types 0/1, merges tracks stably, accumulates tempo with exact
rational arithmetic and rounds absolute times to frames, ties to even. Metadata
and track ends retain their timing effects, including trailing silence. SysEx
and escape file records are rejected: an escape record could contain arbitrary
MIDI traffic rather than a decoded SysEx message. Type 2 and SMPTE remain
unsupported by this adapter. The existing meedee reader defaults apply: 1 MiB
per record, 128 tracks and 8 MiB of merge lookahead.

Every primitive message is fed to each converter. Its binding chooses notes,
tuning, pairing and CC/bend semantics. Unmapped controls and pressure are ignored,
replacing the former importer's hard-coded rejection policy. Positive bend now
reaches its declared maximum at +8191 rather than falling slightly short.

`MidiPerformance.parts` maps binding names to event lists, including configured
parts with no events. No channel numbers are inferred as part names. The importer
finishes every converter on success. A failed file read, conversion, finish or
interruption invalidates all participating converters and returns no partial
performance. Argument/routing rejection before reading leaves their state alone.
There is no implicit restart, retry or rollback of previously emitted identities.

Prepare an instrument once with `Patch.prepare_score(sample_rate)` or supply an
equivalent resolved `(score, assets)` pair. Use the score to configure its
converter and pass that same pair to `render_midi`, keyed by the part name.
The Bach example resolves selectors from its fixed in-memory instrument map;
there is no implicit library discovery or score-file loading.

The renderer requires one instrument per configured part, matching sample rates
and the stereo `left`, `right` layout. It selects oscillator/FM/noise from the
resolved voice type or uses the sampler for sample scores. Existing engine
preparation rejects unsupported mixed-engine scores. Output retains the 250 ms
tail budget; a voice still active afterward fails rendering before publication.
Presets consume `Trigger.velocity` through uFor's velocity event binding, not a
synthetic named control. Offline oscillator, FM and noise rendering retain onset
velocity in their shared control-context snapshots; sampler slot velocities
continue to use the sampler's resolved onset values. Persistent oscillator/FM/
noise renderers retain their existing supported binding profile.

## Caller-polled live input

`enge.midi.LiveInput(connection, converters=[...], message_limit=128)` wraps a
caller-owned, queue-mode MIDI 1.0 `meedee.midi.TypedInputConnection`. UMP is
unsupported by this converter and fails explicitly. Configure fresh or
explicitly restarted converters with distinct source names and destination parts,
as for file import. Use one adapter per independent input connection. The host
opens and closes the connection, selects meedee's isolation mode and must be its
only queue consumer. Do not share its converters with another active adapter or
file import. The adapter adds no queue, thread or audio-device ownership.

Call `drain(frame=destination_frame)` on one host control thread. It polls at most
`message_limit` typed items without waiting for MIDI, including ignored messages,
and returns a dictionary of named parts with their ordered uFor events. Frames
must be integers and must not decrease; equal frames retain input order through
increasing ordinals across calls. Every message in a drain receives that frame.
Backend timestamps are not mapped to an audio clock. This is block-quantized
scheduling, not a hard real-time audio callback API.

The adapter checks receive-drop counts, connection failure and closure before and
after receiving. Any observed loss, decoder diagnostic, conversion failure or
interruption invalidates all participating converters, permanently stops the
adapter and discards that drain's result. In isolated mode drop updates are
asynchronous: detection cannot retract events returned by an earlier drain.
Silence alone does not establish device health. Previously returned events and
musical cleanup remain the host's responsibility.

`stop()` is idempotent and invalidates conversion state. It sends no messages,
closes no ports and generates no releases. Recovery requires a fresh healthy
connection, fresh or explicitly restarted converters, and a new adapter. Do not
reuse queued events from a failed session. The adapter does not produce sound or
add an audio-device host.

For supervised macOS qualification using only temporary test-created virtual
endpoints in both process modes, run explicitly:

```bash
uv run --frozen pytest -q -s -n 0 test/qualification/midi_input.py
```

Each native case has a 45-second outer deadline. The checks cover note/control/
bend conversion, release pairing, host-frame ordering, caller-owned close and
receive loss followed by silence. They do not qualify physical devices or other
operating systems.
