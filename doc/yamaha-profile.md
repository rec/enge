# Yamaha offline compatibility profile

The DX7, TX81Z and WT11 profiles are supported as provisional offline software
models. This is the accepted Yamaha delivery boundary following the user's
2026-11-05 decision to close this part of enge. Further hardware fidelity and
performance-mode expansion are deferred.

## Supported use

Use `YamahaDX7Voice`, `YamahaTX81ZVoice` or `YamahaWT11Voice` in a uFor
`SynthInstrumentScore`, prepare native performance actions with
`ufor.synth_trace.prepare`, and render through
`enge.yamaha_instrument.OfflineYamaha(enge.yamaha_instrument.prepare(score))`.
The adapter provides `advance(actions, start, end)`, `snapshot()` and
`restore(snapshot)`. MIDI-file rendering selects this adapter for Yamaha scores.
Yamaha audio rendering uses NumPy; these profiles do not provide a native/live
callback backend.

DX7 requires its explicit voice payload; TX81Z requires VCED and ACED; WT11
requires VCED, ACED and ACED2. Pairing is explicit. The schemas retain original
bytes, while rendering validates the implemented parameter domains. General FM
and Yamaha templates retain separate definitions and state.

The implemented profile includes operator algorithms/waveforms and envelopes,
shared LFO clocks, fixed/ratio frequency modes, tuning, declared controller
routes, gain/filters, minimum hold, continuous fades, external generators,
caller-owned tempo maps and JSON continuation. TX81Z/WT11 include the documented
provisional mono/glide behavior. WT11 includes its pitch-envelope projection,
aftertouch LFO pitch/amplitude assignments and aftertouch EG bias.

WT11 EG bias reuses the measured TX81Z curve per operator. For combined breath
and aftertouch it uses one baseline attenuation and the stronger controller
relief before the existing TL floor and foot-volume processing. This combination
and the WT11 transfer curve are explicit provisional choices. Omitted pressure
means zero pressure, including when an EG-bias assignment is enabled.

## Known limits

- TX81Z/WT11 breath pitch bias is unsupported when active. The offline adapter
  rejects nonneutral bias with a declared breath route; direct notes reject
  positive breath input with nonneutral bias. Stored bytes remain inspectable.
- WT11 aftertouch pitch bias must remain neutral (50). Nonneutral assignments
  are rejected at note/instrument admission.
- Performance edit buffers and banks can be inspected, but there is no complete
  hardware performance allocator or automatic playback of PCED/PCED2 settings.
  Existing explicitly authored TX81Z LFO/DAC groups remain available.
- Hardware utility, physical switch and other compatibility bytes can be
  preserved without being executed. Portable sustain/control bindings and
  explicit host routing define the delivered behavior; importing bytes does
  not imply a complete device emulator.
- WT11 performance DSP and breath/lip/off LFO-control settings are not executed
  by the voice profile. No optional effects catalogue is selected implicitly.
- Some complete-note comparisons differ from their software references.
  DX7 envelope/block behavior, four-operator fixed-frequency/shift boundaries,
  WT11 pitch projection and controller/clock calibration retain the documented
  provisional models. Hardware equivalence, listening qualification and live
  device deadlines are not claimed.

Snapshots preserve model definitions and owned render/control state, not future
caller-provided input rows. Direct note callers must resupply those rows when
replaying; instrument snapshots retain the scoped controlling state.

## Evidence and future work

See [reference reports](yamaha-reference/README.md), the
[WT11 model report](yamaha-reference/wt11-pitch.md) and
[closeout record](../plan/yamaha-completion.md). Reports retain their original
passing/failing comparisons and qualification limits. Deferred features require
a new explicit task; they are not continuing obligations of this closeout.
