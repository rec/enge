from pathlib import Path

import numpy as np
import pytest
from test_synth import check_audio
from test_yamaha import voice
from test_yamaha_instrument import instrument
from ufor import tx81z, wt11
from ufor.dx7 import DX7Voice
from ufor.synth import YamahaTX81ZVoice

from enge import yamaha
from enge.synth import EngineError


def definition(
    algorithm: int = 8,
) -> tuple[
    wt11.WT11Voice,
    wt11.WT11Additional,
    wt11.WT11Additional2,
    tx81z.TX81ZVoice,
    tx81z.TX81ZAdditional,
]:
    template = instrument("tx81z").body.voices[0]
    assert isinstance(template, YamahaTX81ZVoice)
    data = bytearray(template.patch.data)
    extra = bytearray(template.additional.data)
    data[52] = algorithm - 1
    data[53] = 5
    data[56:58] = bytes(2)
    data[75] = 50
    data[87:93] = bytes([99, 99, 99, 50, 50, 50])
    for index in range(4):
        data[index * 13 + 10] = 50 + index * 7
        data[index * 13 + 11] = 4 + index
        extra[index * 5 + 3] = index + 1
    reference_voice = tx81z.TX81ZVoice(data=bytes(data))
    reference_additional = tx81z.TX81ZAdditional(data=bytes(extra))
    for index, source in enumerate((0, 2, 1, 3)):
        data[index * 13 : index * 13 + 13] = reference_voice.data[
            source * 13 : source * 13 + 13
        ]
        extra[index * 5 : index * 5 + 5] = reference_additional.data[
            source * 5 : source * 5 + 5
        ]
    return (
        wt11.WT11Voice(data=bytes(data)),
        wt11.WT11Additional(data=bytes(extra)),
        wt11.WT11Additional2(data=bytes([0, 0, 50, 0, 1, 2, 3, 4, 5, 6])),
        reference_voice,
        reference_additional,
    )


@pytest.mark.parametrize("algorithm", range(1, 9))
@pytest.mark.parametrize("start_frame", [0, 7])
def test_wt11_neutral_pitch_preserves_opz_audio_and_operator_identity(
    tmp_path: Path, algorithm: int, start_frame: int
) -> None:
    patch, additional, additional2, reference_voice, reference_extra = definition(
        algorithm
    )
    renderer = yamaha.WT11Renderer(
        patch, additional, additional2, 69, 100, start_frame=start_frame
    )
    reference = yamaha.TX81ZRenderer(
        reference_voice, reference_extra, 69, 100, start_frame=start_frame
    )
    check_audio(
        tmp_path / "neutral.wav", renderer.render(48000), reference.render(48000)
    )


@pytest.mark.parametrize("release_at", [117, 24000])
@pytest.mark.parametrize("sample_rate", [24000, 48000, 96000])
def test_wt11_pitch_projection_releases_and_restores_with_mixed_operator_modes(
    tmp_path: Path,
    release_at: int,
    sample_rate: int,
) -> None:
    patch, additional, additional2, reference_voice, reference_extra = definition()
    data = bytearray(patch.data)
    data[87:93] = bytes([50, 99, 50, 82, 40, 50])
    patch = wt11.WT11Voice(data=bytes(data))
    extra = bytearray(additional.data)
    extra[5] = 1  # WT11's second wire block is OP2.
    extra[6] = 2
    additional = wt11.WT11Additional(data=bytes(extra))
    extra = bytearray(reference_extra.data)
    extra[10] = 1  # The TX81Z view places OP2 in its third block.
    extra[11] = 2
    reference_extra = tx81z.TX81ZAdditional(data=bytes(extra))
    renderer = yamaha.WT11Renderer(patch, additional, additional2, 69, 100, sample_rate)
    reference = yamaha.TX81ZRenderer(
        reference_voice, reference_extra, 69, 100, sample_rate
    )
    projected = bytearray(voice().data)
    projected[126:134] = bytes([60, 98, 99, 60, 82, 40, 40, 50])
    envelope = yamaha.DX7Renderer(
        DX7Voice(data=bytes(projected)), 69, 100, sample_rate
    ).pitch_envelope
    split = release_at * sample_rate // 48000
    tuning = np.linspace(-17, 23, sample_rate)
    cents = np.array([1200 * np.log2(envelope.sample()) for _ in range(split)])
    expected = [reference.render(split, tuning_cents=cents + tuning[:split])]
    actual = [renderer.render(split, tuning_cents=tuning[:split])]
    if release_at == 24000:
        assert renderer.pitch_envelope.stage == 3
    renderer.release()
    reference.release()
    envelope.release()
    snapshot = yamaha.WT11Snapshot.model_validate_json(
        renderer.snapshot().model_dump_json()
    )
    saved = snapshot.model_dump_json()
    cents = np.array(
        [1200 * np.log2(envelope.sample()) for _ in range(sample_rate - split)]
    )
    expected.append(
        reference.render(sample_rate - split, tuning_cents=cents + tuning[split:])
    )
    renderer.render(sample_rate - split, tuning_cents=tuning[split:])
    renderer.restore(snapshot)
    first = min(997, sample_rate - split)
    actual.append(renderer.render(first, tuning_cents=tuning[split : split + first]))
    actual.append(
        renderer.render(
            sample_rate - split - first, tuning_cents=tuning[split + first :]
        )
    )
    assert renderer.pitch_envelope.level == 0
    assert snapshot.model_dump_json() == saved
    assert snapshot.wt11_voice.data == patch.data
    assert snapshot.wt11_additional.data == additional.data
    assert snapshot.wt11_additional2.data == additional2.data
    actual_audio, expected_audio = np.concatenate(actual), np.concatenate(expected)
    if sample_rate == 24000:
        actual_audio = np.repeat(actual_audio, 2, axis=0)
        expected_audio = np.repeat(expected_audio, 2, axis=0)
    elif sample_rate == 96000:
        actual_audio, expected_audio = actual_audio[::2], expected_audio[::2]
    check_audio(tmp_path / "pitch.wav", actual_audio, expected_audio)


@pytest.mark.parametrize(
    "field", ["voice", "additional2", "pitch_rates", "pitch_rate", "legacy"]
)
def test_wt11_rejects_foreign_snapshots_before_changing_state(field: str) -> None:
    patch, additional, additional2, _, _ = definition()
    renderer = yamaha.WT11Renderer(patch, additional, additional2, 69, 100)
    saved = renderer.snapshot()
    foreign = saved.model_copy(deep=True)
    if field == "voice":
        data = bytearray(patch.data)
        data[75] = 100
        foreign = foreign.model_copy(
            update={"wt11_voice": wt11.WT11Voice(data=bytes(data))}
        )
    elif field == "additional2":
        foreign = foreign.model_copy(
            update={
                "wt11_additional2": wt11.WT11Additional2(
                    data=bytes([0, 0, 50, 0, 6, 5, 4, 3, 2, 1])
                )
            }
        )
    elif field == "pitch_rates":
        foreign.pitch_envelope.rates[0] = 0
    elif field == "pitch_rate":
        foreign.pitch_envelope.sample_rate = 96000
    else:
        foreign = yamaha.TX81ZSnapshot.model_validate(
            saved.model_dump(
                exclude={
                    "wt11_voice",
                    "wt11_additional",
                    "wt11_additional2",
                    "pitch_envelope",
                }
            )
        )
    with pytest.raises(EngineError, match="different note definition"):
        renderer.restore(foreign)
    assert renderer.snapshot() == saved


@pytest.mark.parametrize("controller", ["tuning", "foot", "breath"])
def test_wt11_failed_control_rows_do_not_advance_pitch_or_chip_state(
    controller: str,
) -> None:
    patch, additional, additional2, _, _ = definition()
    data = bytearray(patch.data)
    data[75] = 100
    renderer = yamaha.WT11Renderer(
        wt11.WT11Voice(data=bytes(data)), additional, additional2, 69, 100
    )
    saved = renderer.snapshot()
    with pytest.raises(EngineError):
        if controller == "tuning":
            renderer.render(100, tuning_cents=np.zeros(99))
        elif controller == "foot":
            renderer.render(100, foot=np.full(100, -1.0))
        else:
            renderer.render(100, breath=np.ones(100))
    assert renderer.snapshot() == saved


@pytest.mark.parametrize("index,value", [(0, 1), (1, 1), (2, 0), (3, 1)])
def test_wt11_rejects_unimplemented_aftertouch_assignments(
    index: int, value: int
) -> None:
    patch, additional, additional2, _, _ = definition()
    data = bytearray(additional2.data)
    data[index] = value
    with pytest.raises(EngineError, match="aftertouch assignments"):
        yamaha.WT11Renderer(
            patch, additional, wt11.WT11Additional2(data=bytes(data)), 69, 100
        )
