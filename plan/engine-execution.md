# Remaining engine execution work

## Control integration

Addressed LFO and Envelfo playback actions already have offline and persistent
coverage, including snapshots, named envelopes and stages. Shared execution now
belongs to enveLFO; enge supplies audio observation and voice lifecycle policy.
Do not treat the original event-integration proposal as wholly unimplemented.

Persistent route preparation still requires direct two-point linear maps.
Amplitude, named and staged envelope adapters now preserve authored curves in
NumPy/native block and persistent rendering. Regression coverage includes large
positive/negative curves, held segments, release, partitions and continuation.
Extend remaining route mappings using existing portable contracts.

## Sampler coverage

The offline NumPy/native sampler and its Python persistent adapter already
support control routes, generators, filters, pan laws and key/velocity bindings.
Remaining preparation restrictions include layer crossfades, delayed/offset
starts and fade retirement. Audit each
against current code before editing because separate work may land concurrently.

Native live registration currently rejects controls, generators and filters.
The user approved extending that native sampler with the existing native
control/filter machinery on 2026-11-02, preserving bounded callback storage.

## Dynamic filters

The complete per-voice filter chain defaults to `before_amplitude` and can use
`after_amplitude`. Amplitude includes the envelope, prepared gain, and changing
amplitude controls. Pan/channel routing follows both stages. Sampler slot/group
filters precede instrument filters and active chains must agree on their order.

NumPy and Rust renderers carry this choice through preparation and snapshots.
FM keeps envelope-scaled operator signals for modulation and feedback while
placing the audible carrier envelope and level in the chosen amplitude stage.
Source exhaustion or envelope completion discards filter state without an extra
tail.

## Live execution

The Rust LiveRuntime already owns mixing, supported effects, generated sources
and static sample playback. Python still performs action admission/submission;
the higher-level LiveEngine also has allocating orchestration paths. Native
hosts retain device ownership. See [the callback audit](../doc/live-callback-audit.md)
for the measured boundary; offline conformance is not device qualification.

## Additional work beyond the prompt

None.
