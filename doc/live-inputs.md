# Named inputs in live effect graphs

`LiveEngine` and `NativeLiveEngine` route the graph's declared `MainStream`,
`HostStream` and `InstrumentStream` inputs. Main means the sum of all prepared
sources. An instrument `pre_effect` tap means that named source, including all
slots of a sampler. Source processing, such as its voice filters, has already
run at this point. Instrument `post_effect` taps are rejected because these
owners do not yet have per-instrument effect attachments.

All persistent sources currently share one sample rate and channel layout.
Main and instrument input declarations must match that layout. Each host input
declares its own channel list and may have a different width. The graph output
layout is reported by the live owner's `channels`; callers allocate output
arrays using that layout rather than assuming the source channel count.
Existing processors preserve their connected input layout. This routing slice
does not add matrix conversion or spatial processors.

Supply host audio to `advance` or `advance_into` as the `host_inputs` argument:
a dictionary keyed by `HostStream.stream`, not by the graph input ID. Every
required stream must cover the exact render interval as a finite float64 array
of shape `(frames, declared_channels)`. Unexpected or missing streams fail
admission. A stream may feed multiple graph inputs of the same width. No host
audio persists implicitly into the next call. Host blocks are validated before
source actions are prepared or submitted.

Native graph preparation receives explicit widths for every graph input and
processor output. It checks each connection's width and ordering. Input routes
refer to the main mix, a host block, or a prepared group of native source indices.
The sampler group contains all of its slot runtimes, so an instrument tap is not
accidentally limited to its first slot.

Preparation allocates input buffers for the maximum callback frame count and
maximum port width, per-frame node values, processor state, and an output buffer
with the declared output width. Source audio is accumulated into the connected
input buffers during rendering. Host audio is validated and copied into its
prepared buffers before the native callback body and heap-audit scope. Scratch
is reused for shorter blocks without changing its capacity. The Python wrappers
still prepare action arrays outside the native callback.

Snapshots retain processor state and input-route/width identity. Host blocks
and scratch are transient: after restoring a snapshot, the caller supplies the
host audio for the replayed interval again. A rejected native host block leaves
queued actions available for a subsequent valid block.

Regression tests in `test/test_live_inputs.py` write paired 48 kHz WAVs lasting
one second. They compare main and individual-source routing against independent
dry renders, cover a two-slot sampler tap, mono host output from stereo sources,
snapshot replay, and invalid-block admission. The existing delay, generator,
filter and native-owner regressions also exercise the extended graph API.
The full ordinary suite passed 3,166 tests with 21 skipped. The 65-test focused
instrumented run recorded zero Rust allocations,
reallocations or frees in 2,759 successful and seven failed callbacks. Host
admission failures occur before that audit scope. These measurements qualify the
tested native paths, not device deadlines or arbitrary host work.

Input-end/stop actions, per-instrument attachments, channel-conversion matrices
and spatial DSP remain separate roadmap work.

## Additional work beyond the prompt

None.
