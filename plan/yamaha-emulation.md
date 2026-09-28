# Yamaha-compatible FM

Implement a dedicated Yamaha renderer for the DX7 and TX81Z instead of
projecting their patches into the general FM voice profile. The general graph
remains useful for authored enge patches, but it cannot represent Yamaha's
rate/level envelopes, key scaling, fixed-frequency operators, pitch envelope,
or feedback arithmetic.

## DX7

- Decode each validated VCED or packed bank voice into six Yamaha operators,
  the selected one of 32 algorithms, carrier mix, and delayed feedback path.
- Preserve Yamaha operator numbering, frequency mode, coarse/fine ratio,
  detune, level scaling, velocity sensitivity, and oscillator sync.
- Render the four-rate/four-level amplitude and pitch envelopes in a dedicated
  fixed-point-compatible state model. Compare its generated output against
  known DX7-compatible renderers using named patches and 48 kHz WAV fixtures.
- Keep the lossless SysEx objects as the import boundary. A malformed dump must
  remain inspectable and never become a playable voice.

## TX81Z / WT11

- Pair VCED and ACED edit-buffer messages explicitly before conversion.
- Decode its four-operator, eight-algorithm routing, frequency modes, feedback,
  and ACED waveform selections into the same Yamaha renderer family.
- Preserve the TX81Z waveform definitions rather than mapping them to enge's
  generic sine, square, and triangle waveforms.

## Delivery boundaries

- The compatibility renderer is offline first. Do not claim callback-safe live
  Yamaha emulation until its fixed state and allocation bounds are measured.
- The general FM graph and Yamaha profiles have separate schema tags, prepared
  state, snapshots, and regressions. Neither is an alias for the other.
- Add a small set of public imported-patch fixtures only when their licensing
  permits redistribution. Tests may construct parameter bytes directly.

## Additional work beyond the prompt

None.
