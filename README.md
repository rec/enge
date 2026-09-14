# Enge

Enge is the engine-only implementation boundary for Ufor instruments. It has no
device, file, MIDI, OSC, GUI, or application dependency. Hosts prepare Ufor
scores and traces, advance this engine at exact output-frame boundaries, and own
transport and encoding.

The first profile renders canonical oscillator synth voices. Sample traversal,
non-instant envelopes, modulation, processing, and live hosting remain explicit
future profiles rather than implicit approximations.
