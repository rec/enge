# Optional Rubber Band distribution

The `rubberband` native feature statically links Rubber Band Library 4.0.0,
copyright 2007-2024 Particular Programs Ltd., through `rubberband-sys` 0.1.4.
Rubber Band is distributed under GPL-2.0-or-later; see `RUBBERBAND-COPYING`.
Enge's own source remains under its existing MIT licence. The optional combined
binary includes GPL code. No commercial Rubber Band licence is supplied here.

Source distributions include the original registry archive
`third_party/rubberband-sys-0.1.4.crate`, containing the Rust binding build script,
Rubber Band source, and its bundled third-party sources and notices. This is
the unchanged source archive used by the locked dependency, not an additional
build implementation. Cargo continues to resolve it through `Cargo.lock`.

SHA-256: `1f5da3be376569632abdaa5d772cf0bfe2d80d618c61072df0cbde38e215044e`.

Upstream source and licence information:

- <https://breakfastquay.com/rubberband/>
- <https://crates.io/crates/rubberband-sys/0.1.4>

The feature-disabled extension does not link Rubber Band. These notices are
included in both build profiles so the optional feature's terms remain visible.
