from pathlib import Path

import numpy as np
import pytest
from test_dynamic_synth import onset
from test_sample_instrument import sample_score
from test_synth import check_audio
from ufor.samples import instrument, trace
from ufor.samples.variation import ResolvedVariation

from enge import live, sample_instrument


@pytest.mark.parametrize("direction", ["forward", "backward"])
@pytest.mark.parametrize("offset", [0, 3, 99])
@pytest.mark.parametrize("alignment", [-2, 0, 2])
def test_offsets_skip_in_playback_order_and_exhaust_outside_slice(
    tmp_path: Path, direction: str, offset: int, alignment: int
) -> None:
    raw = sample_score(frames=16).model_dump(mode="json")
    raw["body"]["slots"][0].update(
        playback={"direction": direction}, variation={"offset_frames": 99}
    )
    raw["body"]["settings"]["selections"] = [{"name": "takes", "mode": "cycle"}]
    raw["body"]["slots"][0].update(
        selection="takes", take="one", microphone="close", alignment_frames=alignment
    )
    document = instrument.SampleInstrumentScore.model_validate(raw)
    data = np.arange(16, dtype=float) / 16
    definition = sample_instrument.prepare(
        document, {"asset": np.column_stack([data, np.zeros(16)])}
    )
    actions = trace.prepare(
        document.body, [onset(pitch=440).model_copy(update={"controls": {}})], seed=0
    ).actions
    actions = [
        a.model_copy(update={"variation": ResolvedVariation(offset_frames=offset)})
        if isinstance(a, trace.VoiceStart)
        else a
        for a in actions
    ]
    index = (15 - offset if direction == "backward" else offset) + alignment
    values = (
        np.empty(0)
        if not 0 <= index < 16
        else data[: index + 1][::-1]
        if direction == "backward"
        else data[index:]
    )
    expected = np.zeros((48000, 2))
    expected[: len(values)] = values[:, None] * [1, 0.25]
    for backend in ("numpy", "native"):
        renderer = sample_instrument.OfflineSampler(definition, backend)
        first = renderer.advance(actions, 0, 3)
        restored = sample_instrument.OfflineSampler(definition, backend)
        restored.restore(renderer.snapshot())
        tail = renderer.advance([], 3, 48000)
        check_audio(
            tmp_path / f"offset-{backend}.wav", np.concatenate([first, tail]), expected
        )
        np.testing.assert_array_equal(restored.advance([], 3, 48000), tail)
        assert renderer.snapshot().voices == []
    engine = live.NativeLiveEngine(
        {"sample": sample_instrument.PersistentSampler(definition)}, 48000
    )
    actual = np.empty((48000, 2))
    engine.advance_into({"sample": actions}, [], 0, 3, actual[:3])
    saved = engine.snapshot()
    engine.advance_into({"sample": []}, [], 3, 48000, actual[3:])
    check_audio(tmp_path / "offset-live.wav", actual, expected)
    engine.restore(saved)
    replay = np.empty((47997, 2))
    engine.advance_into({"sample": []}, [], 3, 48000, replay)
    np.testing.assert_array_equal(replay, actual[3:])
