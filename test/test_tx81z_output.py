from pathlib import Path

import numpy as np
import pytest
from test_synth import check_audio
from test_yamaha_instrument import instrument
from ufor import synth_trace
from ufor.events import Trigger
from ufor.instrument_trace import RetirementCause, VoiceRetirement
from ufor.synth import SynthInstrumentScore

from enge import filters, yamaha, yamaha_instrument


def reference_dac(raw: np.ndarray) -> np.ndarray:
    with np.load(Path(__file__).parent / "data/tx81z-opz-dac.npz") as reference:
        table = reference["decoded"]
    units = np.rint(raw * 8192).astype(np.int64)
    result = table[np.clip(units, -32768, 32767) + 32768]
    return np.where(units > 32767, 32767, result) / 32768


def test_dac_matches_every_reference_code_and_overflow(tmp_path: Path) -> None:
    units = np.arange(-32768, 32768)
    units = np.concatenate(
        [units, [-65536, -32769, 32768, 65536, -513.5, -512.5, 512.5, 513.5]]
    )
    raw = units[:, None] / 8192
    actual = yamaha.tx81z_dac(raw)
    expected = reference_dac(raw)
    check_audio(tmp_path / "dac-codes.wav", actual, expected)
    np.testing.assert_array_equal(actual, expected)


@pytest.mark.parametrize("value", [np.nan, np.inf, -np.inf])
def test_dac_rejects_nonfinite_audio(value: float) -> None:
    with pytest.raises(ValueError, match="finite"):
        yamaha.tx81z_dac(np.array([[value]]))


def output_document(grouped: bool, split_groups: bool = False) -> SynthInstrumentScore:
    raw = instrument("tx81z").model_dump(mode="json")
    template = raw["body"]["voices"][0]
    data = bytearray.fromhex(template["patch"])
    for i in range(0, 52, 13):
        data[i + 10] = 99
    data[56:58] = bytes(2)
    template["patch"] = data.hex()
    template["channels"] = [{"input": "mono", "output": "left", "gain": 1}]
    raw["body"]["voices"].append({**template, "name": "second"})
    if grouped:
        for v in raw["body"]["voices"]:
            v["channels"] = []
        names = [["yamaha"], ["second"]] if split_groups else [["yamaha", "second"]]
        raw["body"]["tx81z_outputs"] = [
            {
                "name": f"chip-{i}",
                "voices": n,
                "processing": {
                    "volume_db": -6,
                    "filters": [
                        {"name": "low", "response": "lowpass", "cutoff_hz": 1800}
                    ],
                },
                "channels": [{"input": "mono", "output": "left", "gain": 0.5}],
            }
            for i, n in enumerate(names)
        ]
    return SynthInstrumentScore.model_validate(raw)


def test_dac_holds_the_last_chip_output_after_voice_retirement(tmp_path: Path) -> None:
    raw = output_document(True).model_dump(mode="json")
    raw["timebases"][0]["rate"]["numerator"] = 192000
    raw["body"]["tx81z_outputs"][0]["processing"] = {}
    document = SynthInstrumentScore.model_validate(raw)
    actions = synth_trace.prepare(
        document.body,
        [Trigger(tick=0, ordinal=0, part="main", trigger_id="note", key=69)],
        seed=0,
    ).actions
    renderer = yamaha_instrument.OfflineYamaha(yamaha_instrument.prepare(document))
    head = renderer.advance(actions, 0, 4)
    stops = [
        VoiceRetirement(
            tick=4,
            ordinal=i,
            voice_id=a.voice_id,
            cause=RetirementCause.transport_stop,
            action="stop",
        )
        for i, a in enumerate(actions)
        if isinstance(a, synth_trace.VoiceStart)
    ]
    middle = renderer.advance(stops, 4, 5)
    saved = yamaha_instrument.YamahaSnapshot.model_validate_json(
        renderer.snapshot().model_dump_json()
    )
    assert not saved.voices and head[3, 0] != 0
    tail = renderer.advance([], 5, 192000)
    expected = np.zeros((192000, 2))
    expected[3:6] = head[3]
    actual = np.concatenate([head, middle, tail])
    np.testing.assert_array_equal(actual, expected)
    check_audio(tmp_path / "held-chip-dac.wav", actual[3::4], expected[3::4])
    renderer.restore(saved)
    np.testing.assert_array_equal(renderer.advance([], 5, 192000), tail)


@pytest.mark.parametrize(
    "split_groups,fade", [(False, False), (True, False), (False, True)]
)
def test_group_dac_precedes_host_processing_and_restores_filter_tails(
    tmp_path: Path, split_groups: bool, fade: bool
) -> None:
    document = output_document(True, split_groups)
    events = [
        Trigger(tick=0, ordinal=i, part=f"part-{i}", trigger_id=f"note-{i}", key=69)
        for i in range(4)
    ]
    actions = synth_trace.prepare(document.body, events, seed=0).actions
    starts = [a for a in actions if isinstance(a, synth_trace.VoiceStart)]
    actions.extend(
        VoiceRetirement(
            tick=2000,
            ordinal=i,
            voice_id=a.voice_id,
            cause=RetirementCause.transport_stop,
            action="fade" if fade else "stop",
            fade_seconds=0.01 if fade else None,
        )
        for i, a in enumerate(starts)
    )
    raw_document = output_document(False)
    raw_actions = synth_trace.prepare(raw_document.body, events, seed=0).actions
    raw_actions.extend(a for a in actions if isinstance(a, VoiceRetirement))
    raw = yamaha_instrument.OfflineYamaha(
        yamaha_instrument.prepare(raw_document)
    ).advance(raw_actions, 0, 48000)[:, :1]
    assert np.max(raw) > 4
    assert np.min(raw) < -4
    groups = 2 if split_groups else 1
    decoded = reference_dac(raw / groups) * groups
    settings = document.body.tx81z_outputs[0].processing
    expected, _ = filters.filter_samples(
        settings.filters,
        filters.initial_states(settings.filters, 1),
        decoded,
        filters.parameters(settings.filters, 48000, 48000),
        48000,
    )
    expected = np.column_stack(
        [expected[:, 0] * 10 ** (-6 / 20) * 0.5, np.zeros(48000)]
    )
    renderer = yamaha_instrument.OfflineYamaha(yamaha_instrument.prepare(document))
    actual = renderer.advance(actions, 0, 48000)
    check_audio(tmp_path / "grouped-output.wav", actual, expected)
    assert np.any(actual[2000:2100])
    split = yamaha_instrument.OfflineYamaha(yamaha_instrument.prepare(document))
    boundary = 2501 if fade else 2001
    head = split.advance(actions, 0, boundary)
    saved = yamaha_instrument.YamahaSnapshot.model_validate_json(
        split.snapshot().model_dump_json()
    )
    assert not saved.voices
    split.restore(saved)
    tail = split.advance([], boundary, 48000)
    np.testing.assert_array_equal(np.concatenate([head, tail]), actual)
