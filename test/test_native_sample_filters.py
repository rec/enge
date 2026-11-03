from pathlib import Path

import numpy as np
import pytest
from test_dynamic_synth import onset
from test_sample_instrument import sample_score
from test_synth import check_audio
from ufor.events import Release
from ufor.instrument_trace import RetirementCause, VoiceRetirement
from ufor.samples import instrument, trace

from enge import live, sample_instrument


@pytest.mark.parametrize("order", ["before_amplitude", "after_amplitude"])
@pytest.mark.parametrize("response", ["lowpass", "highpass", "bandpass", "notch"])
def test_native_sample_filters_preserve_channels_voice_reuse_and_snapshots(
    tmp_path: Path, order: str, response: str
) -> None:
    raw = sample_score(frames=30000).model_dump(mode="json")
    raw["body"]["settings"] = {
        "processing": {
            "volume_db": -3,
            "filter_order": order,
            "filters": [{"name": "common", "response": "highpass", "cutoff_hz": 80}],
        },
        "envelope": {
            "segments": [{"duration": "0.04 s", "to": 1}],
            "release": [{"duration": "0.125 s", "to": 0}],
        },
    }
    raw["body"]["slots"][0]["processing"] = {
        "volume_db": -2,
        "filter_order": order,
        "filters": [
            {
                "name": "slot",
                "response": response,
                "cutoff_hz": 1320,
                "q": 0.8,
                "stages": 2,
            }
        ],
    }
    document = instrument.SampleInstrumentScore.model_validate(raw)
    time = np.arange(30000) / 48000
    definition = sample_instrument.prepare(
        document,
        {
            "asset": np.column_stack(
                [
                    0.1 * np.sin(2 * np.pi * 440 * time),
                    0.2 * np.cos(2 * np.pi * 1130 * time),
                ]
            )
        },
    )
    events = [
        onset(t, n, pitch=440).model_copy(update={"controls": {}})
        for t, n in [(0, "first"), (4000, "second"), (12000, "third")]
    ]
    events.extend(
        [
            Release(tick=6000, ordinal=0, part="main", trigger_id="first"),
            Release(tick=18000, ordinal=0, part="main", trigger_id="second"),
        ]
    )
    actions = trace.prepare(document.body, events, seed=0).actions
    actions.append(
        VoiceRetirement(
            tick=8000,
            ordinal=0,
            voice_id=next(
                a.voice_id
                for a in actions
                if isinstance(a, trace.VoiceStart) and a.trigger_id == "first"
            ),
            cause=RetirementCause.choke,
            action="stop",
        )
    )
    expected = sample_instrument.OfflineSampler(definition).advance(actions, 0, 48000)
    source = sample_instrument.PersistentSampler(definition, voices=2)
    engine = live.NativeLiveEngine({"sample": source}, 48000)
    actual = np.empty((48000, 2))
    engine.advance_into(
        {"sample": [a for a in actions if a.tick < 13001]}, [], 0, 13001, actual[:13001]
    )
    saved = engine.snapshot()
    later = [a for a in actions if a.tick >= 13001]
    engine.advance_into({"sample": later}, [], 13001, 48000, actual[13001:])
    check_audio(tmp_path / "native-sample-filters.wav", actual, expected)
    assert not np.any(actual[42000:])
    assert not source.native_voices
    engine.restore(saved)
    replay = np.empty((34999, 2))
    for start in range(13001, 48000, 997):
        end = min(48000, start + 997)
        engine.advance_into(
            {"sample": [a for a in later if start <= a.tick < end]},
            [],
            start,
            end,
            replay[start - 13001 : end - 13001],
        )
    np.testing.assert_array_equal(replay, actual[13001:])
