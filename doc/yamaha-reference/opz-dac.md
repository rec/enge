# TX81Z DAC model and integration boundary

This audit uses the same pinned ymfm revision as the
[operator-core comparison](opz-core.md):
`81aec25ccbb98f4873a255f7551ac4dadac59b4a`. It qualifies source arithmetic,
not measured TX81Z hardware output. The
[YM2414 generate implementation](https://github.com/aaronsgiles/ymfm/blob/81aec25ccbb98f4873a255f7551ac4dadac59b4a/src/ymfm_opz.cpp)
explicitly assumes YM2151-like output because the YM2414 output is uncertain.

## Reference output sequence

The reference clocks all chip channels, renders their combined integer output,
then calls `output->roundtrip_fp()`. Channel rendering uses output shift zero
and a 32767 clip limit. DAC conversion therefore belongs after hardware-channel
summation, before any external gain, filters or routing. The floating-point
representation here is a signed mantissa/exponent DAC code, not IEEE float.

The [executable check](check-ymfm-dac.cc) exhaustively compares the optimized
`roundtrip_fp` helper with `decode_fp(encode_fp(value))` for all signed 16-bit
inputs. All **65,536 values match**. It then probes overflow and mixing order;
the complete [captured output](opz-dac-check.txt) is retained.

From the repository root, build with `YMFM` pointing to the pinned source folder:

```sh
mkdir -p build
c++ -std=c++17 -O2 -I"$YMFM" doc/yamaha-reference/check-ymfm-dac.cc -o build/check-ymfm-dac
build/check-ymfm-dac
```

The executable exits nonzero if an in-range comparison fails or output cannot
be written. Arithmetic is tested directly without rendering an audio stream.

## Positive-overflow discrepancy

Both helpers quantize representable inputs by clearing low bits according to
magnitude. Negative values use arithmetic truncation, not rounding toward zero.
The optimized helper clips positive overflow directly to 32767, whereas
encode/decode clips to the largest representable mantissa, decoded as 32704.

| Input | Helper used by chip generation | Explicit encode/decode |
| ---: | ---: | ---: |
| -32769 | -32768 | -32768 |
| -513 | -514 | -514 |
| 513 | 512 | 512 |
| 32703 | 32640 | 32640 |
| 32704 | 32704 | 32704 |
| 32767 | 32704 | 32704 |
| 32768 | 32767 | 32704 |
| 65536 | 32767 | 32704 |

For a provisional model matching this pinned chip renderer, use the optimized
helper's actual overflow behavior. Do not replace it with encode/decode while
claiming exact reference equality. This recommendation does not establish which
behavior describes the physical DAC.

## Why conversion cannot be per note

| Voice A | Voice B | Convert after summing | Sum separately converted voices |
| ---: | ---: | ---: | ---: |
| 32767 | 1 | 32767 | 32705 |
| 20000 | 20000 | 32767 | 39936 |
| 1023 | -511 | 512 | 511 |

The last case differs even without overflow. Applying conversion independently
inside `TX81ZRenderer.render` would therefore implement a different signal path.
Applying it to `OfflineYamaha`'s final output would also be wrong: that output
already includes per-voice portable gain, filters, fades and routing, and may
mix DX7 and TX81Z templates. Its output buses are not hardware chip channels.

The note renderer continues to return raw carrier sums. uFor's
`SynthInstrument.tx81z_outputs` now supplies explicit output ownership. Each
group names its TX81Z templates; every active instance contributes to the same
raw sum, including voices from different parts. Group members omit individual
routes and host gain/filter settings. Their hardware controls and tuning remain
per voice. Group gain, polarity, filters and routes apply after conversion.
Ungrouped templates retain raw note output; they do not acquire an inferred DAC.

Enge's `OfflineYamaha` implements this boundary using `yamaha.tx81z_dac`.
Raw note samples convert back to chip units by multiplying by 8192. The decoded
DAC result divides by 32768 for signed 16-bit full scale. Host retirement fades
scale individual raw contributions; the combined sum rounds to integer chip
units with ties to even before conversion. Fades are a host lifecycle extension,
not a claim about the hardware's envelopes. Group filters persist when no voices
remain, and their state participates in JSON snapshot continuation.

The check executable optionally accepts a binary output path and captures all
65,536 decoded signed-16-bit inputs as native-endian `int32_t` values:

```sh
mkdir -p build
build/check-ymfm-dac build/opz-dac.i32
```

The little-endian capture is stored without value changes as the `decoded`
array in `test/data/tx81z-opz-dac.npz`. Regression tests compare every code,
overflow and half-unit fade rounding. One-second 48 kHz WAV regressions cover
simultaneous-note summation across parts, both clipping signs, post-DAC gain and
filters, separate output groups, fades, filter tails and partitioned JSON replay.

The fixture SHA-256 is
`a413756f2afd76c51300252b87226b1c97ade9b87f45423fa898661885cedce2`.

This qualifies the explicit output boundary against a provisional software
model. It does not establish native-clock sampling, analog output filtering,
physical hardware equivalence, the physical eight-note allocator or shared
multitimbral LFO behavior.
