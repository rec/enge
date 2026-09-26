# Four-operator FM profile

## Scope

Replace the fixed two-operator renderer with a portable graph profile supporting
four or six named operators, selectable operator waveforms, and one audible
carrier. Current-sample modulation edges form an acyclic directed graph.
Each edge contributes its index times the source's envelope-scaled output to the
destination phase. The carrier's envelope ends the voice.

Feedback edges are separate from the acyclic graph. Every feedback edge reads
the source output from the preceding output sample, has one sample of delay, and
is stored independently in snapshots. Feedback may target any operator. A patch
must reject a same-sample cycle rather than choosing an implicit ordering.

## Work

1. Define the profile in uFor with stable operator, edge, carrier, feedback, and
   waveform identifiers. Validate supported operator counts, endpoint existence,
   a unique carrier, acyclicity after removing delayed edges, finite nonnegative
   indices, and one-sample feedback delay.
2. Replace the fixed `VoiceRenderer` state with graph-owned operator phase,
   envelope, waveform, topological-order, and per-edge feedback-history state.
   Preserve two-operator patch behavior as the graph profile's smallest form.
3. Define waveform phase conventions and output ranges for sine, triangle, and
   square before adding Yamaha compatibility mappings. Waveform choice is latched
   at onset; ratio, tuning, level, and edge indices remain live targets.
4. Add scalar topology vectors for a four-operator chain, fan-in, a six-operator
   graph, multiple delayed feedback edges, reordered declarations, invalid cycles,
   partitions, and restore.
5. Add independent NumPy and Rust kernels. Both retain phases and compensation
   for every operator plus one history sample per delayed edge.
6. Integrate prepared actions, controls, filters, routes, snapshots, persistent
   runtime admission, and WAV/FLAC regressions.

## Superseded initial sequence

1. Define the profile in uFor with stable operator, edge, carrier, and feedback
   identifiers. Validate operator count, endpoint existence, a unique carrier,
   acyclicity after removing delayed edges, finite nonnegative indices, and
   one-sample feedback delay.
2. Add scalar topology vectors for a chain, fan-in, multiple delayed feedback
   edges, reordered operator declarations, invalid cycles, partitions, and
   restore.
3. Add separate NumPy and Rust kernels. Both retain phases and compensation for
   four operators plus one history sample per delayed edge. They evaluate the
   validated topological order, then advance all phases.
4. Integrate the existing prepared action/control/filter/routing path. Ratios,
   tuning, carrier level, and every edge index receive stable modulation targets;
   graph structure, carrier selection, authored phases, and edge delay remain
   latched at onset.
5. Add shared conformance, one-second 48 kHz WAV regression, and a native versus
   NumPy comparison. Do not claim alias-free output; oversampling is later work.

## Additional work beyond the prompt

None.
