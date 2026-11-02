# FM synthesis status

The general graph renderer is implemented in NumPy and Rust for two through six
operators, with sine, triangle and square waveforms, current-sample acyclic
modulation and independently retained delayed feedback. Prepared actions,
controls, filters, routing, partitions and snapshots have regression coverage.
Persistent FM uses graph state too; replacing fixed two-operator state is no
longer an open task.

[four-operator-fm.md](four-operator-fm.md) records the implemented graph contract.
Remaining shared runtime restrictions belong in
[engine-execution.md](engine-execution.md). Yamaha compatibility uses a dedicated
renderer and is tracked in [yamaha-emulation.md](yamaha-emulation.md).

PyTorch work was explicitly excluded and is not a future implementation task.

## Additional work beyond the prompt

None.
