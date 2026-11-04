import pytest
from ufor import tx81z
from ufor.dx7 import DX7Voice

from enge.synth import EngineError
from enge.yamaha import DX7Renderer, TX81ZRenderer


@pytest.mark.parametrize("operator", range(4))
@pytest.mark.parametrize("source", ["voice", "additional"])
def test_tx81z_rejects_nonboolean_operator_flags_without_changing_dumps(
    operator: int, source: str
) -> None:
    raw = bytearray(93)
    raw[62] = 24
    extra = bytearray(23)
    if source == "voice":
        raw[operator * 13 + 8] = 2
    else:
        extra[operator * 5] = 2
    voice = tx81z.TX81ZVoice(data=bytes(raw))
    additional = tx81z.TX81ZAdditional(data=bytes(extra))
    with pytest.raises(EngineError, match="TX81Z .* must be 0 or 1"):
        TX81ZRenderer(voice, additional, 69, 100)
    assert voice.data == bytes(raw)
    assert additional.data == bytes(extra)


@pytest.mark.parametrize(
    "packed,index,value",
    [
        (False, 136, 2),
        (False, 137, 100),
        (False, 138, 100),
        (False, 139, 100),
        (False, 140, 100),
        (False, 141, 2),
        (False, 142, 6),
        (False, 143, 8),
        (False, 144, 49),
        (False, 17, 2),
        (False, 122, 2),
        (True, 112, 100),
        (True, 113, 100),
        (True, 114, 100),
        (True, 115, 100),
        (True, 116, 12),
        (True, 117, 49),
    ],
)
def test_dx7_rejects_invalid_globals_without_changing_the_dump(
    packed: bool, index: int, value: int
) -> None:
    data = bytearray(128 if packed else 155)
    data[index] = value
    voice = DX7Voice(data=bytes(data))
    with pytest.raises(EngineError, match="DX7 (parameter|oscillator mode)"):
        DX7Renderer(voice, 69, 100)
    assert voice.data == bytes(data)
    assert len(voice.title) == 10


@pytest.mark.parametrize("packed", [False, True])
def test_dx7_accepts_global_boundaries_and_sample_and_hold(packed: bool) -> None:
    data = bytearray(128 if packed else 155)
    if packed:
        data[111] = 7 | (1 << 3)
        data[112:116] = bytes([99] * 4)
        data[116] = 1 | (5 << 1) | (7 << 4)
        data[117] = 48
    else:
        data[135:145] = bytes([7, 1, 99, 99, 99, 99, 1, 5, 7, 48])
    voice = DX7Voice(data=bytes(data))
    renderer = DX7Renderer(voice, 69, 100)
    assert renderer.snapshot().voice.data == bytes(data)
