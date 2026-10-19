from pathlib import Path
from typing import Literal

import numpy as np
import pytest
from test_dynamic_synth import onset
from test_sample_instrument import sample_score
from test_synth import check_audio
from ufor.samples import instrument, trace

from enge import live, sample_instrument


@pytest.mark.parametrize("law", ["native", "sfizz"])
@pytest.mark.parametrize("stereo", [False, True])
@pytest.mark.parametrize("position", [-1.0, -0.5, 0.0, 0.5, 1.0])
def test_pan_laws_render_absolute_gains_once(
    backend: Literal["numpy", "native"],
    tmp_path: Path,
    law: str,
    stereo: bool,
    position: float,
) -> None:
    document, audio = pan_score(law, stereo, position)
    prepared = sample_instrument.prepare(document, {"asset": audio})
    actions = trace.prepare(
        document.body, [onset(pitch=440).model_copy(update={"controls": {}})], seed=0
    ).actions
    renderer = sample_instrument.OfflineSampler(prepared, backend)
    first = renderer.advance(actions, 0, 24000)
    snapshot = renderer.snapshot()
    second = renderer.advance([], 24000, 48000)
    renderer.restore(snapshot)
    np.testing.assert_array_equal(renderer.advance([], 24000, 48000), second)
    gains = PAN_GAINS[law, stereo][[-1.0, -0.5, 0.0, 0.5, 1.0].index(position)]
    expected = (
        np.tile(([0.125, -0.0625] if stereo else [0.125, 0.125]), (48000, 1)) * gains
    )
    check_audio(tmp_path / "pan.wav", np.concatenate([first, second]), expected)


@pytest.mark.parametrize("law", ["native", "sfizz"])
@pytest.mark.parametrize("stereo", [False, True])
def test_persistent_and_native_live_pan_match_offline(
    tmp_path: Path, law: str, stereo: bool
) -> None:
    document, audio = pan_score(law, stereo, 0.5)
    prepared = sample_instrument.prepare(document, {"asset": audio})
    actions = trace.prepare(
        document.body, [onset(pitch=440).model_copy(update={"controls": {}})], seed=0
    ).actions
    expected = sample_instrument.OfflineSampler(prepared, "native").advance(
        actions, 0, 48000
    )
    persistent = sample_instrument.PersistentSampler(prepared, voices=2)
    check_audio(
        tmp_path / "persistent-pan.wav", persistent.advance(actions, 0, 48000), expected
    )
    engine = live.NativeLiveEngine(
        {"sample": sample_instrument.PersistentSampler(prepared, voices=2)}, 997
    )
    output = np.empty_like(expected)
    for start in range(0, 48000, 997):
        end = min(start + 997, 48000)
        engine.advance_into(
            {"sample": [a for a in actions if start <= a.tick < end]},
            [],
            start,
            end,
            output[start:end],
        )
    check_audio(tmp_path / "native-live-pan.wav", output, expected)


def pan_score(
    law: str, stereo: bool, position: float
) -> tuple[instrument.SampleInstrumentScore, np.ndarray]:
    raw = sample_score(frames=48000).model_dump(mode="json")
    channels = ["left", "right"] if stereo else ["mono"]
    raw["assets"][0]["audio"]["channels"] = channels
    raw["body"]["pan_law"] = law
    parameter = "stereo_balance" if stereo else "pan"
    raw["body"]["settings"]["processing"] = {parameter: position / 2}
    raw["body"]["groups"] = [{"name": "pan", "processing": {parameter: position / 2}}]
    slot = raw["body"]["slots"][0]
    slot["group"] = "pan"
    slot.pop("processing")
    slot["channels"] = (
        [{"input": c, "output": c, "gain": 1} for c in channels]
        if stereo
        else [
            {"input": "mono", "output": c, "gain": 2**-0.5} for c in ["left", "right"]
        ]
    )
    audio = np.tile([0.125, -0.0625] if stereo else [0.125], (48000, 1))
    return instrument.SampleInstrumentScore.model_validate(raw), audio


PAN_GAINS = {
    ("native", False): [
        [1, 0],
        [0.9238795325112867, 0.3826834323650898],
        [2**-0.5, 2**-0.5],
        [0.3826834323650898, 0.9238795325112867],
        [0, 1],
    ],
    ("native", True): [[1, 0], [1, 0.5], [1, 1], [0.5, 1], [0, 1]],
    ("sfizz", False): [
        [1.4125375446227544, 0],
        [1.3050145263807111, 0.5405547159207919],
        [0.9988148764833452, 0.9988148764833452],
        [0.5405547159207919, 1.3050145263807111],
        [0, 1.4125375446227544],
    ],
    ("sfizz", True): [
        [1.4108635131604637, 0],
        [1.3034679229759207, 0.5399140918149152],
        [0.9976311574844398, 0.9976311574844398],
        [0.5399140918149152, 1.3034679229759207],
        [0, 1.4108635131604637],
    ],
}
