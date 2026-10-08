from pathlib import Path

import numpy as np
import pytest
from test_dynamic_synth import onset
from test_lfo_instrument import lfo_score
from test_synth import check_audio
from ufor import synth_trace
from ufor.synth import SynthInstrumentScore

from enge import synth


@pytest.mark.parametrize("backend", ["numpy", "native", "persistent"])
@pytest.mark.parametrize("scope", ["voice", "part", "instrument"])
@pytest.mark.parametrize(
    "step,origin,output",
    [(0.5, 0.0, "steps"), (0.75, 0.125, "level"), (1.5, 0.0, "level")],
)
def test_quantize_grid_matches_across_clocks_partitions_and_snapshots(
    tmp_path: Path, backend: str, scope: str, step: float, origin: float, output: str
) -> None:
    raw = lfo_score("synth", scope).model_dump(mode="json")
    voice = raw["body"]["voices"][0]
    voice["motions"]["motion"]["body"] = {
        "kind": "patch",
        "motions": {
            "wave": {"kind": "cycle", "shape": "triangle", "rate": "2"},
            "steps": {
                "kind": "quantize",
                "input": "wave",
                "step": step,
                "origin": origin,
            },
            "shared": {"kind": "sum", "inputs": ["steps", "steps"]},
            "level": {
                "kind": "affine",
                "input": "shared",
                "scale": 0.125,
                "offset": 0.5,
            },
        },
        "outputs": {"value": output},
    }
    voice["bindings"][0]["output"] = "value"
    minimum, maximum = (-2, 2) if output == "steps" else (0, 1)
    voice["modulation"]["sources"][0].update(minimum=minimum, maximum=maximum)
    voice["modulation"]["parameters"][0]["maximum"] = 1
    voice["modulation"]["routes"][0]["points"] = [
        {"input": minimum, "amount": 0},
        {"input": maximum, "amount": 1},
    ]
    document = SynthInstrumentScore.model_validate(raw)
    events = [onset(6000, pitch=0.1).model_copy(update={"controls": {}})]
    actions = synth_trace.prepare(document.body, events, seed=5).actions
    raw["body"]["voices"][0].update(motions={}, bindings=[], modulation={})
    baseline = SynthInstrumentScore.model_validate(raw)
    plain = synth.OfflineSynth(synth.prepare(baseline)).advance(
        synth_trace.prepare(baseline.body, events, seed=5).actions, 0, 48000
    )
    frames = np.arange(48000) - (6000 if scope == "voice" else 0)
    phase = (frames % 24000) / 24000
    values = np.where(phase < 0.5, 4 * phase - 1, 3 - 4 * phase)
    grid = origin + step * np.arange(-4, 5)
    boundaries = (grid[:-1] + grid[1:]) / 2
    quantized = grid[np.searchsorted(boundaries, values, side="right")]
    if step == 1.5:
        assert quantized.max() > 1
        assert quantized.min() < -1
    expected = plain * (0.5 + 0.25 * quantized)[:, None]
    definition = synth.prepare(document)
    renderer = (
        synth.PersistentSynth(definition, voices=1)
        if backend == "persistent"
        else synth.OfflineSynth(definition, backend)
    )
    whole = renderer.advance(actions, 0, 48000)
    renderer = (
        synth.PersistentSynth(definition, voices=1)
        if backend == "persistent"
        else synth.OfflineSynth(definition, backend)
    )
    first = renderer.advance(actions, 0, 13003)
    snapshot = renderer.snapshot()
    middle = renderer.advance([], 13003, 17317)
    renderer.restore(snapshot)
    np.testing.assert_array_equal(renderer.advance([], 13003, 17317), middle)
    tail = renderer.advance([], 17317, 48000)
    partitioned = np.vstack((first, middle, tail))
    np.testing.assert_allclose(partitioned, whole, atol=1e-10, rtol=1e-9)
    check_audio(
        tmp_path / f"quantize-{scope}-{step}-{origin}-{backend}.wav", whole, expected
    )
