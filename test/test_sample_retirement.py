from pathlib import Path

import numpy as np
import pytest
from test_dynamic_synth import onset
from test_sample_instrument import sample_score
from test_synth import check_audio
from ufor.instrument_trace import RetirementCause, VoiceRetirement
from ufor.samples import instrument, trace

from enge import live, sample_instrument


@pytest.mark.parametrize("ending", ["fade", "stop", "release", "exhaustion"])
def test_fades_retime_continuously_and_restore_without_extending_source_lifetime(
    tmp_path: Path, ending: str
) -> None:
    frames = 20000 if ending == "exhaustion" else 96000
    raw = sample_score(frames=frames).model_dump(mode="json")
    slot = raw["body"]["slots"][0]
    slot.update(
        choke_group="self",
        chokes=[{"group": "self", "mode": "fade", "fade_seconds": 0.5}],
    )
    document = instrument.SampleInstrumentScore.model_validate(raw)
    definition = sample_instrument.prepare(
        document, {"asset": np.tile([0.2, 0.1], (frames, 1))}
    )
    actions = trace.prepare(
        document.body, [onset(pitch=440).model_copy(update={"controls": {}})], seed=0
    ).actions
    voice = next(a.voice_id for a in actions if isinstance(a, trace.VoiceStart))
    actions.extend(
        VoiceRetirement(
            tick=t,
            ordinal=0,
            voice_id=voice,
            cause=RetirementCause.choke,
            action="fade",
            fade_seconds=d,
        )
        for t, d in [(12000, 0.5), (18000, 0.125)]
    )
    if ending in ("stop", "release"):
        actions.append(
            VoiceRetirement(
                tick=20000,
                ordinal=0,
                voice_id=voice,
                cause=RetirementCause.choke,
                action=ending,
            )
        )
    gain = np.ones(48000)
    gain[12000:18000] = 1 - np.arange(6000) / 24000
    gain[18000:] = 0.75 * np.maximum(0, 1 - np.arange(30000) / 6000)
    if ending != "fade":
        gain[20000:] = 0
    expected = gain[:, None] * [0.2, 0.15]
    first_actions = [a for a in actions if a.tick < 19001]
    last_actions = [a for a in actions if a.tick >= 19001]
    for backend in ("numpy", "native"):
        renderer = sample_instrument.OfflineSampler(definition, backend)
        first = renderer.advance(first_actions, 0, 19001)
        restored = sample_instrument.OfflineSampler(definition, backend)
        restored.restore(renderer.snapshot())
        tail = renderer.advance(last_actions, 19001, 48000)
        check_audio(
            tmp_path / f"retirement-{backend}.wav",
            np.concatenate([first, tail]),
            expected,
        )
        np.testing.assert_array_equal(
            restored.advance(last_actions, 19001, 48000), tail
        )
        assert renderer.snapshot().voices == []
    source = sample_instrument.PersistentSampler(definition)
    engine = live.NativeLiveEngine({"sample": source}, 30000)
    actual = np.empty((48000, 2))
    engine.advance_into({"sample": first_actions}, [], 0, 19001, actual[:19001])
    saved = engine.snapshot()
    engine.advance_into({"sample": last_actions}, [], 19001, 48000, actual[19001:])
    check_audio(tmp_path / "retirement-live.wav", actual, expected)
    assert source.native_voices == {}
    engine.restore(saved)
    replay = np.empty((28999, 2))
    engine.advance_into({"sample": last_actions}, [], 19001, 48000, replay)
    np.testing.assert_array_equal(replay, actual[19001:])
