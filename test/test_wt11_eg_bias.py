from pathlib import Path

import numpy as np
import pytest
from test_synth import check_audio
from test_tx81z_frequency import opz_frequency
from test_tx81z_integer import native_carrier
from test_wt11 import definition
from ufor import tx81z, wt11

from enge import yamaha
from enge.synth import EngineError


def renderer(
    sensitivity: int, output: int, foot: bool, breath: int = 0, aftertouch: int = 99
) -> yamaha.WT11Renderer:
    data = bytearray(93)
    for index in range(0, 52, 13):
        data[index], data[index + 3], data[index + 4] = 31, 15, 15
        data[index + 7], data[index + 11], data[index + 12] = sensitivity, 4, 3
    data[49], data[52], data[62] = output, 7, 24
    data[67], data[75], data[76] = 99 if foot else 0, 50, breath
    data[87:93] = bytes([99, 99, 99, 50, 50, 50])
    return yamaha.WT11Renderer(
        wt11.WT11Voice(data=bytes(data)),
        wt11.WT11Additional(data=bytes(23)),
        wt11.WT11Additional2(data=bytes([0, 0, 50, aftertouch, 1, 2, 3, 4, 5, 6])),
        69,
        127,
    )


@pytest.mark.parametrize("sensitivity,output", [(7, 99), (0, 99), (7, 0)])
@pytest.mark.parametrize("foot", [False, True])
def test_wt11_aftertouch_eg_bias_uses_measured_curve_before_tl_saturation(
    tmp_path: Path, sensitivity: int, output: int, foot: bool
) -> None:
    note = renderer(sensitivity, output, foot)
    codes = np.repeat([0, 1, 2, 3, 4, 5, 6, 7, 127], [5000] * 8 + [8000])
    bias = np.repeat([127, 112, 96, 86, 80, 74, 70, 67, 0], [5000] * 8 + [8000])
    if sensitivity == 0:
        bias = np.zeros(48000)
    gain = 2 ** (
        -np.minimum(127, (16 if output else 127) + bias + (15 if foot else 0)) / 8
    )
    expected = native_carrier(opz_frequency(), gain)
    expected += 3 * native_carrier(opz_frequency(), 2 ** (-127 / 8))
    actual = note.render(
        48000,
        aftertouch=(codes + 0.25 * (codes < 127)) / 127,
        foot=np.full(48000, 64 / 127),
    )
    check_audio(tmp_path / "aftertouch-eg.wav", actual, expected[:, None])


@pytest.mark.parametrize("foot", [False, True])
def test_wt11_combines_strongest_eg_relief_from_one_operator_baseline(
    tmp_path: Path, foot: bool
) -> None:
    note = renderer(7, 99, foot, breath=99, aftertouch=50)
    breath = np.repeat([0, 1, 64, 127], 12000) / 127
    pressure = np.repeat([0, 127, 64, 0], 12000) / 127
    # Measured bias 99 at those breath codes is 127/112/16/0;
    # measured bias 50 at those pressure codes is 16/0/6/16.
    # One baseline 127 minus strongest relief gives 127/111/16/0.
    bias = np.repeat([127, 111, 16, 0], 12000)
    gain = 2 ** (-np.minimum(127, 16 + bias + (15 if foot else 0)) / 8)
    expected = native_carrier(opz_frequency(), gain)
    expected += 3 * native_carrier(opz_frequency(), 2 ** (-127 / 8))
    actual = note.render(
        48000, breath=breath, aftertouch=pressure, foot=np.full(48000, 64 / 127)
    )
    check_audio(tmp_path / "combined-eg.wav", actual, expected[:, None])


@pytest.mark.parametrize("released", [False, True])
def test_wt11_eg_bias_and_controller_state_survive_partitioned_snapshot_replay(
    tmp_path: Path, released: bool
) -> None:
    note = renderer(7, 99, True, breath=50)
    note.render(177, aftertouch=np.full(177, 0.4))
    if released:
        note.release()
    saved = yamaha.WT11Snapshot.model_validate_json(note.snapshot().model_dump_json())
    serialized = saved.model_dump_json()
    pressure = np.linspace(0, 1, 48000)
    breath = np.linspace(1, 0, 48000)
    expected = note.render(48000, aftertouch=pressure, breath=breath)
    note.restore(saved)
    actual = np.concatenate(
        [
            note.render(997, aftertouch=pressure[:997], breath=breath[:997]),
            note.render(47003, aftertouch=pressure[997:], breath=breath[997:]),
        ]
    )
    assert saved.model_dump_json() == serialized
    check_audio(tmp_path / "eg-replay.wav", actual, expected)


def test_wt11_assigned_eg_bias_with_omitted_pressure_uses_zero_pressure(
    tmp_path: Path,
) -> None:
    implicit = renderer(7, 99, False)
    explicit = renderer(7, 99, False)
    check_audio(
        tmp_path / "eg-default.wav",
        implicit.render(48000),
        explicit.render(48000, aftertouch=np.zeros(48000)),
    )


@pytest.mark.parametrize("control", ["aftertouch", "breath"])
def test_wt11_rejected_eg_controller_preserves_note_state(control: str) -> None:
    note = renderer(7, 99, False, breath=50)
    saved = note.snapshot()
    with pytest.raises(EngineError):
        if control == "aftertouch":
            note.render(100, aftertouch=np.full(100, -0.1))
        else:
            note.render(100, breath=np.full(100, -0.1))
    assert note.snapshot() == saved


@pytest.mark.parametrize("amount", [20, 99])
def test_wt11_eg_bias_preserves_each_reordered_operator_sensitivity(
    tmp_path: Path, amount: int
) -> None:
    patch, additional, _, original, original_extra = definition()
    data, reference_data = bytearray(patch.data), bytearray(original.data)
    for number, sensitivity in enumerate((0, 2, 5, 7), 1):
        source = {1: 3, 2: 1, 3: 2, 4: 0}[number]
        data[source * 13 + 7] = sensitivity
        reference_data[(4 - number) * 13 + 7] = sensitivity
    reference_data[76] = amount
    note = yamaha.WT11Renderer(
        wt11.WT11Voice(data=bytes(data)),
        additional,
        wt11.WT11Additional2(data=bytes([0, 0, 50, amount, 0, 0, 0, 0, 0, 0])),
        69,
        100,
    )
    reference = yamaha.TX81ZRenderer(
        tx81z.TX81ZVoice(data=bytes(reference_data)), original_extra, 69, 100
    )
    positions = np.linspace(0, 1, 48000)
    check_audio(
        tmp_path / "eg-operator-identity.wav",
        note.render(48000, aftertouch=positions),
        reference.render(48000, breath=positions),
    )
