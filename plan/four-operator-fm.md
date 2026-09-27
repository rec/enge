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

1. Replace the fixed `VoiceRenderer` state with graph-owned operator phase,
   envelope, waveform, topological-order, and per-edge feedback-history state.
   Preserve two-operator patch behavior as the graph profile's smallest form.
2. Define waveform phase conventions and output ranges for sine, triangle, and
   square before adding Yamaha compatibility mappings. Waveform choice is latched
   at onset; ratio, tuning, level, and edge indices remain live targets.
3. Add scalar topology vectors for a four-operator chain, fan-in, a six-operator
   graph, multiple delayed feedback edges, reordered declarations, invalid cycles,
   partitions, and restore.
4. Integrate prepared actions, controls, filters, routes, snapshots, persistent
   runtime admission, and WAV/FLAC regressions.

## Additional work beyond the prompt

None.
