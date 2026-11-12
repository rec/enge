from pathlib import Path

import numpy as np
import pytest
from test_persistent_effects import callback_audio
from test_synth import check_audio
from ufor import audio_effects

from enge import _native, effects


def convolution_graph(frames: int, partition: int = 256) -> audio_effects.EffectGraph:
    return audio_effects.EffectGraph.model_validate(
        {
            "scope": "master",
            "timebases": [{"name": "ir", "rate": {"numerator": 48000}}],
            "assets": [
                {
                    "name": "ir",
                    "location": {"kind": "relative_file", "path": "ir.wav"},
                    "encoding": "wav",
                    "content": {},
                    "audio": {
                        "timebase": "ir",
                        "channels": ["ll", "lr", "rl", "rr"],
                        "frames": frames,
                    },
                }
            ],
            "inputs": [
                {
                    "name": "main",
                    "channels": ["left", "right"],
                    "source": {"kind": "main"},
                }
            ],
            "processors": [
                {
                    "kind": "convolution",
                    "name": "room",
                    "partition_frames": partition,
                    "input_channels": ["left", "right"],
                    "output_channels": ["left", "right"],
                    "kernels": [
                        {
                            "input_channel": i,
                            "output_channel": o,
                            "asset": "ir",
                            "channel": c,
                        }
                        for i, o, c in [
                            ("left", "left", "ll"),
                            ("left", "right", "lr"),
                            ("right", "left", "rl"),
                            ("right", "right", "rr"),
                        ]
                    ],
                }
            ],
            "connections": [
                {
                    "processor": "room",
                    "port": "input",
                    "source": {"kind": "input", "input": "main"},
                }
            ],
            "output": {"kind": "processor", "processor": "room"},
            "maximum_block_frames": 48000,
        }
    )


def direct_convolution(
    source: np.ndarray, impulse: np.ndarray, partition: int
) -> np.ndarray:
    expected = np.zeros_like(source)
    for i, o, c in [(0, 0, 0), (0, 1, 1), (1, 0, 2), (1, 1, 3)]:
        expected[partition:, o] += np.convolve(source[:, i], impulse[:, c])[
            : len(source) - partition
        ]
    return expected


@pytest.mark.parametrize(
    "length,partition", [(1, 1), (255, 256), (256, 256), (257, 256), (513, 128)]
)
def test_matrix_fir_matches_direct_convolution_and_replays_partial_partitions(
    tmp_path: Path,
    length: int,
    partition: int,
) -> None:
    rng = np.random.default_rng(981)
    impulse = rng.uniform(-0.02, 0.02, (length, 4))
    source = rng.uniform(-0.1, 0.1, (48000, 2))
    source[24003:] = 0
    prepared = effects.prepare(
        convolution_graph(length, partition), 48000, audio={"ir": impulse}
    )
    assert prepared.latency_frames == partition
    expected = direct_convolution(source, impulse, partition)
    actions = [audio_effects.InputEndAction(tick=24003, ordinal=0, input="main")]
    for origin, target in [("numpy", "native"), ("native", "numpy")]:
        renderer = effects.OfflineEffects(prepared, origin)
        first = renderer.advance({"main": source[:23928]}, [], 0, 23928)
        snapshot = effects.EffectSnapshot.model_validate_json(
            renderer.snapshot().model_dump_json()
        )
        restored = effects.OfflineEffects(prepared, target)
        restored.restore(snapshot)
        last = restored.advance({"main": source[23928:]}, actions, 23928, 48000)
        assert restored.drained
        check_audio(
            tmp_path / f"{origin}-{target}.wav", np.vstack([first, last]), expected
        )
    actual, replay, drained = callback_audio(prepared, {"main": source}, actions)
    assert drained
    check_audio(tmp_path / "callback.wav", actual, expected)
    check_audio(tmp_path / "replay.wav", replay, actual)


def test_sparse_non_square_kernels_and_dry_routing_share_partition_latency(
    tmp_path: Path,
) -> None:
    raw = convolution_graph(257).model_dump(mode="json")
    raw["processors"][0].update(
        output_channels=["mono"],
        kernels=[
            {
                "input_channel": "left",
                "output_channel": "mono",
                "asset": "ir",
                "channel": "ll",
            },
        ],
    )
    raw["processors"].extend(
        [
            {
                "kind": "matrix",
                "name": "dry",
                "input_channels": ["left", "right"],
                "output_channels": ["mono"],
                "coefficients": [[0.5, 0.5]],
            },
            {"kind": "crossfader", "name": "blend", "position": 0.25, "law": "linear"},
        ]
    )
    raw["connections"].extend(
        [
            {
                "processor": "dry",
                "port": "input",
                "source": {"kind": "input", "input": "main"},
            },
            {
                "processor": "blend",
                "port": "first",
                "source": {"kind": "processor", "processor": "dry"},
            },
            {
                "processor": "blend",
                "port": "second",
                "source": {"kind": "processor", "processor": "room"},
            },
        ]
    )
    raw["output"] = {"kind": "processor", "processor": "blend"}
    graph = audio_effects.EffectGraph.model_validate(raw)
    impulse = np.zeros((257, 4))
    impulse[[0, 256], 0] = [0.4, -0.2]
    source = np.zeros((48000, 2))
    source[:23000] = [0.2, -0.1]
    prepared = effects.prepare(graph, 48000, audio={"ir": impulse})
    expected = np.zeros((48000, 1))
    expected[256:, 0] = (
        0.75 * source[:-256].mean(axis=1)
        + 0.25 * np.convolve(source[:, 0], impulse[:, 0])[:47744]
    )
    actions = [audio_effects.InputEndAction(tick=23000, ordinal=0, input="main")]
    for backend in ["numpy", "native"]:
        actual = effects.process_audio(
            prepared, {"main": source}, actions, backend, 997
        )
        check_audio(tmp_path / f"{backend}.wav", actual, expected)
    actual, replay, drained = callback_audio(prepared, {"main": source}, actions)
    assert drained
    check_audio(tmp_path / "callback.wav", actual, expected)
    check_audio(tmp_path / "replay.wav", replay, actual)


@pytest.mark.parametrize("trailing_zeros", [False, True])
def test_final_fir_sample_is_retained_and_trailing_zeros_do_not_shorten_extent(
    tmp_path: Path,
    trailing_zeros: bool,
) -> None:
    impulse = np.zeros((513, 4))
    impulse[0 if trailing_zeros else -1, [0, 3]] = 0.5
    source = np.zeros((48000, 2))
    source[7] = [0.2, -0.1]
    prepared = effects.prepare(convolution_graph(513), 48000, audio={"ir": impulse})
    end = 8
    empty = end + 513 + 256 - 1
    expected = direct_convolution(source, impulse, 256)
    for backend in ["numpy", "native"]:
        renderer = effects.OfflineEffects(prepared, backend)
        first = renderer.advance(
            {"main": source[:empty]},
            [audio_effects.InputEndAction(tick=end, ordinal=0, input="main")],
            0,
            empty,
        )
        assert not renderer.drained
        last = renderer.advance({"main": source[empty:]}, [], empty, 48000)
        assert renderer.drained
        check_audio(tmp_path / f"{backend}.wav", np.vstack([first, last]), expected)


def test_long_impulse_drains_across_arbitrary_callbacks_and_replays_after_input_end(
    tmp_path: Path,
) -> None:
    impulse = np.zeros((48000, 4))
    impulse[[0, 255, 256, 24000, 47999]] = [0.2, -0.1, 0.03, 0.17]
    source = np.zeros((96000, 2))
    source[[0, 7]] = [0.2, -0.1]
    expected = np.zeros_like(source)
    for at in [0, 7]:
        expected[256 + at : 256 + at + 48000, 0] += (
            source[at, 0] * impulse[:, 0] + source[at, 1] * impulse[:, 2]
        )
        expected[256 + at : 256 + at + 48000, 1] += (
            source[at, 0] * impulse[:, 1] + source[at, 1] * impulse[:, 3]
        )
    prepared = effects.prepare(convolution_graph(48000), 48000, audio={"ir": impulse})
    actions = [audio_effects.InputEndAction(tick=8, ordinal=0, input="main")]
    renderer = effects.OfflineEffects(prepared)
    first = renderer.advance({"main": source[:12345]}, actions, 0, 12345)
    snapshot = effects.EffectSnapshot.model_validate_json(
        renderer.snapshot().model_dump_json()
    )
    restored = effects.OfflineEffects(prepared, "native")
    restored.restore(snapshot)
    after = restored.advance({"main": source[12345:48000]}, [], 12345, 48000)
    last = restored.advance({"main": source[48000:]}, [], 48000, 96000)
    assert restored.drained
    check_audio(tmp_path / "portable.wav", np.vstack([first, after, last]), expected)
    runtime = _native.LiveRuntime.empty(48000, 2, 997, 64)
    runtime.set_effect_graph(
        *effects.native_live_graph(prepared), 2, input_routes=[[-2]]
    )
    actual = np.empty_like(source)
    start = 0
    sizes = [1, 17, 255, 256, 257, 997]
    iteration = 0
    while start < len(source):
        end = min(start + sizes[iteration % len(sizes)], len(source))
        batch = [a for a in actions if start <= a.tick < end]
        if batch:
            runtime.submit_effects(
                effects.native_live_actions(prepared, batch, start, end)
            )
        runtime.process_into(actual[start:end], [source[start:end]])
        start = end
        iteration += 1
    assert runtime.effects_drained
    check_audio(tmp_path / "callbacks.wav", actual, expected)


@pytest.mark.parametrize("failure", ["fill", "shape", "spectrum", "tail"])
def test_portable_snapshot_rejects_corrupt_convolution_state_without_mutation(
    failure: str,
) -> None:
    prepared = effects.prepare(
        convolution_graph(17), 48000, audio={"ir": np.zeros((17, 4))}
    )
    renderer = effects.OfflineEffects(prepared, "native")
    original = renderer.snapshot()
    broken = original.model_copy(deep=True)
    state = broken.processors["room"].convolution
    assert state is not None
    if failure == "fill":
        state.fill = 256
    elif failure == "shape":
        state.overlap.pop()
    elif failure == "spectrum":
        state.spectra_imag[0][0][0] = 1.0
    else:
        state.tail_remaining = 1
    with pytest.raises(ValueError):
        renderer.restore(broken)
    assert renderer.snapshot() == original


def test_preparation_owns_immutable_ir_and_snapshots_bind_its_contents() -> None:
    impulse = np.ones((17, 4), dtype=np.float64)
    graph = convolution_graph(17)
    prepared = effects.prepare(graph, 48000, audio={"ir": impulse})
    impulse.fill(0)
    assert np.all(prepared.audio["ir"] == 1)
    with pytest.raises(ValueError):
        prepared.audio["ir"].setflags(write=True)
    other = effects.prepare(graph, 48000, audio={"ir": impulse})
    with pytest.raises(ValueError, match="different prepared"):
        effects.OfflineEffects(other).restore(
            effects.OfflineEffects(prepared).snapshot()
        )
    runtime = _native.LiveRuntime.empty(48000, 2, 997, 64)
    runtime.set_effect_graph(
        *effects.native_live_graph(prepared), 2, input_routes=[[-2]]
    )
    wrong = _native.LiveRuntime.empty(48000, 2, 997, 64)
    wrong.set_effect_graph(*effects.native_live_graph(other), 2, input_routes=[[-2]])
    with pytest.raises(ValueError):
        wrong.restore(runtime.snapshot())


@pytest.mark.parametrize(
    "failure", ["missing", "extra", "extent", "dtype", "nonfinite", "rate"]
)
def test_impulse_preparation_rejects_mismatched_decoding(failure: str) -> None:
    graph = convolution_graph(17)
    audio = {"ir": np.zeros((17, 4))}
    if failure == "missing":
        audio = {}
    elif failure == "extra":
        audio["unused"] = audio["ir"]
    elif failure == "extent":
        audio["ir"] = np.zeros((18, 4))
    elif failure == "dtype":
        audio["ir"] = audio["ir"].astype(np.float32)
    elif failure == "nonfinite":
        audio["ir"][0, 0] = np.nan
    else:
        raw = graph.model_dump(mode="json")
        raw["timebases"][0]["rate"]["numerator"] = 44100
        graph = audio_effects.EffectGraph.model_validate(raw)
    with pytest.raises(ValueError):
        effects.prepare(graph, 48000, audio=audio)


@pytest.mark.parametrize("failure", ["action", "overflow"])
def test_convolution_callback_failures_silence_and_restore(
    tmp_path: Path, failure: str
) -> None:
    impulse = np.full((257, 4), 0.01)
    prepared = effects.prepare(convolution_graph(257), 48000, audio={"ir": impulse})
    runtime = _native.LiveRuntime.empty(48000, 2, 48000, 64)
    runtime.set_effect_graph(
        *effects.native_live_graph(prepared), 2, input_routes=[[-2]]
    )
    saved = runtime.snapshot()
    source = np.full((48000, 2), 1e308 if failure == "overflow" else 0.1)
    if failure == "action":
        runtime.submit_effects(np.asarray([[12003, 0, 0, 1, 0, 0]], dtype=np.float64))
    actual = np.ones_like(source)
    with pytest.raises(ValueError):
        runtime.process_into(actual, [source])
    assert runtime.failed()
    check_audio(tmp_path / "failed.wav", actual, np.zeros_like(actual))
    runtime.restore(saved)
    source.fill(0.1)
    runtime.process_into(actual, [source])
    check_audio(
        tmp_path / "recovered.wav", actual, direct_convolution(source, impulse, 256)
    )
