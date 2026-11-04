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

Build with `YMFM` pointing to the pinned source folder:

```sh
c++ -std=c++17 -O2 -I"$YMFM" check-ymfm-dac.cc -o /tmp/check-ymfm-dac
/tmp/check-ymfm-dac
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

The existing note renderer should continue to return raw carrier sums. A shared
chip-output stage needs explicit ownership of the voices belonging to a chip,
then raw-channel summation, the provisional DAC conversion, chip-level portable
processing and routing. Inferring chip membership from MIDI part, template name
or an output channel would silently conflate distinct contracts. The shared
chip/output-group contract and its adapter integration remain implementation
work; this audit adds no unused runtime converter and does not mark DAC support
complete.

Required follow-up qualification includes simultaneous-note summation,
positive/negative clipping, post-DAC host gain ordering, distinct chip groups,
partitioned rendering and snapshot replay. This audit does not establish
native-clock sampling, analog output filtering or physical hardware equivalence.
