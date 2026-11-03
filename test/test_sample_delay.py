from pathlib import Path

import numpy as np
import pytest
from test_dynamic_synth import onset
from test_sample_instrument import sample_score
from test_synth import check_audio
from ufor.events import Release
from ufor.instrument_trace import RetirementCause, VoiceRetirement
from ufor.samples import instrument, trace
from ufor.samples.variation import ResolvedVariation

from enge import live, sample_instrument


@pytest.mark.parametrize("release_at", [3000, 6001, 9000])
@pytest.mark.parametrize("mode", ["one_shot", "while_held"])
@pytest.mark.parametrize("sustain", [False, True])
def test_pending_starts_cancel_on_physical_release_and_restore(
    tmp_path: Path, release_at: int, mode: str, sustain: bool
) -> None:
    raw = sample_score(frames=12000).model_dump(mode="json")
    raw["body"]["settings"].update(
        controls={"sustain": {"default": int(sustain)}},
        sustain={"control": "sustain"},
        envelope={
            "segments": [{"duration": "0.01 s", "to": 1}],
            "release": [{"duration": 0, "to": 0}],
        },
    )
    slot = raw["body"]["slots"][0]
    slot["playback"] = {"mode": mode, "start_delay_seconds": 0.125}
    slot["variation"] = {"delay_seconds": 0.00002}
    release_slot = {**slot, "name": "release", "trigger": "release"}
    release_slot["playback"] = {"mode": "one_shot", "start_delay_seconds": 0.125}
    raw["body"]["slots"].append(release_slot)
    document = instrument.SampleInstrumentScore.model_validate(raw)
    data = np.arange(12000) / 48000
    shaped = data * np.minimum(1, np.arange(12000) / 480)
    definition = sample_instrument.prepare(
        document, {"asset": np.column_stack([data, np.zeros(12000)])}
    )
    trigger = onset(pitch=440).model_copy(update={"controls": {}})
    actions = trace.prepare(
        document.body,
        [
            trigger,
            Release(
                tick=release_at,
                ordinal=0,
                part=trigger.part,
                trigger_id=trigger.trigger_id,
            ),
        ],
        seed=0,
    ).actions
    actions = [
        a.model_copy(update={"variation": ResolvedVariation(delay_seconds=0.00001)})
        if isinstance(a, trace.VoiceStart)
        else a
        for a in actions
    ]
    expected = np.zeros((48000, 2))
    if release_at >= 6001:
        count = 12000 if mode == "one_shot" or sustain else release_at - 6001
        expected[6001 : 6001 + count] += shaped[:count, None] * [1, 0.25]
    start = release_at + 6001
    expected[start : start + 12000] += shaped[:, None] * [1, 0.25]
    for backend in ("numpy", "native", "persistent"):
        renderer = (
            sample_instrument.PersistentSampler(definition)
            if backend == "persistent"
            else sample_instrument.OfflineSampler(definition, backend)
        )
        first = renderer.advance([a for a in actions if a.tick < 2000], 0, 2000)
        saved = renderer.snapshot()
        tail_actions = [a for a in actions if a.tick >= 2000]
        tail = renderer.advance(tail_actions, 2000, 48000)
        check_audio(
            tmp_path / f"delay-{backend}.wav", np.concatenate([first, tail]), expected
        )
        renderer.restore(saved)
        np.testing.assert_array_equal(renderer.advance(tail_actions, 2000, 48000), tail)
    engine = live.NativeLiveEngine(
        {"sample": sample_instrument.PersistentSampler(definition)}, 48000
    )
    actual = np.empty((48000, 2))
    engine.advance_into(
        {"sample": [a for a in actions if a.tick < 2000]}, [], 0, 2000, actual[:2000]
    )
    saved = engine.snapshot()
    tail_actions = [a for a in actions if a.tick >= 2000]
    engine.advance_into({"sample": tail_actions}, [], 2000, 48000, actual[2000:])
    check_audio(tmp_path / "delay-live.wav", actual, expected)
    engine.restore(saved)
    replay = np.empty((46000, 2))
    engine.advance_into({"sample": tail_actions}, [], 2000, 48000, replay)
    np.testing.assert_array_equal(replay, actual[2000:])


@pytest.mark.parametrize("choke_at", [3000, 6000, 9000])
def test_fade_chokes_cancel_pending_starts_but_fade_started_voices(
    tmp_path: Path, choke_at: int
) -> None:
    raw = sample_score(frames=48000).model_dump(mode="json")
    raw["body"]["slots"][0]["playback"]["start_delay_seconds"] = 0.125
    document = instrument.SampleInstrumentScore.model_validate(raw)
    definition = sample_instrument.prepare(
        document, {"asset": np.tile([0.2, 0.1], (48000, 1))}
    )
    actions = trace.prepare(
        document.body, [onset(pitch=440).model_copy(update={"controls": {}})], seed=0
    ).actions
    retirement = VoiceRetirement(
        tick=choke_at,
        ordinal=0,
        voice_id=next(a.voice_id for a in actions if isinstance(a, trace.VoiceStart)),
        cause=RetirementCause.choke,
        action="fade",
        fade_seconds=0.125,
    )
    gain = np.zeros(48000)
    if choke_at >= 6000:
        gain[6000:choke_at] = 1
        gain[choke_at:] = np.maximum(0, 1 - np.arange(48000 - choke_at) / 6000)
    expected = gain[:, None] * [0.2, 0.15]
    for backend in ("numpy", "native", "persistent"):
        renderer = (
            sample_instrument.PersistentSampler(definition, voices=1)
            if backend == "persistent"
            else sample_instrument.OfflineSampler(definition, backend)
        )
        first = renderer.advance(actions, 0, 2000)
        saved = renderer.snapshot()
        tail = renderer.advance([retirement], 2000, 48000)
        check_audio(
            tmp_path / f"choke-{backend}.wav", np.concatenate([first, tail]), expected
        )
        assert not renderer.pending and not renderer.voices
        renderer.restore(saved)
        np.testing.assert_array_equal(renderer.advance([retirement], 2000, 48000), tail)
    source = sample_instrument.PersistentSampler(definition, voices=1)
    engine = live.NativeLiveEngine({"sample": source}, 48000)
    actual = np.empty((48000, 2))
    engine.advance_into({"sample": actions}, [], 0, 2000, actual[:2000])
    saved = engine.snapshot()
    engine.advance_into({"sample": [retirement]}, [], 2000, 48000, actual[2000:])
    check_audio(tmp_path / "choke-live.wav", actual, expected)
    assert not source.native_voices
    engine.restore(saved)
    replay = np.empty((46000, 2))
    engine.advance_into({"sample": [retirement]}, [], 2000, 48000, replay)
    np.testing.assert_array_equal(replay, actual[2000:])
