# Remaining engine execution work

## Control integration

Addressed LFO and Envelfo playback actions already have offline and persistent
coverage, including snapshots, named envelopes and stages. Shared execution now
belongs to enveLFO; enge supplies audio observation and voice lifecycle policy.
Do not treat the original event-integration proposal as wholly unimplemented.

Persistent routes now accept piecewise linear, step and positive exponential
maps with endpoint clamping. Native prepared tables also cover Patch outputs;
LFO activation weights apply after mapping. The existing one-route-per-source
and one-route-per-target admission restriction remains.
Amplitude, named and staged envelope adapters now preserve authored curves in
NumPy/native block and persistent rendering. Regression coverage includes large
positive/negative curves, held segments, release, partitions and continuation.
Multiple-route composition remains a separate parity gap.

## Sampler coverage

The offline NumPy/native sampler and its Python persistent adapter already
support control routes, generators, filters, pan laws and key/velocity bindings.
Static key/velocity crossfades now use onset-latched linear or equal-power gains
in offline and native live rendering, including zero-weight voice ownership and
snapshot continuation. Remaining preparation restrictions include control-driven
layer crossfades. Fade retirement now works offline and in native live playback;
repeated requests retime continuously from the current fade gain. Earlier source or
envelope completion still ends the voice. Audit remaining restrictions
against current code before editing because separate work may land concurrently.

Signed alignment and resolved random offsets now work offline and in native live
playback, including snapshots. Random offsets follow playback direction; alignment
adds to the source index. Out-of-slice positions exhaust immediately without
clamping or changing slice/loop coordinates, as decided by the user.

Delayed starts now combine slot and resolved variation delays, rounding up to the
next output frame with traversal and envelopes beginning at age zero there.
uFor's explicit `trigger_release` observation cancels pending start-triggered
voices independently of sustain and one-shot retirement. Release-triggered
voices keep their own timer. Pending starts survive offline/native live snapshots
and consume bounded persistent voice capacity. Retirement before delayed onset,
including a fade choke, cancels the pending voice without sounding. At or after
onset, fade retirement applies the normal continuous ramp.

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
