from math import cos, pi, sin
from pathlib import Path

import numpy as np
import pytest
from test_persistent_effects import callback_audio
from test_synth import check_audio
from ufor import audio_effects

from enge import _native, effects


def matrix_graph(
    processors: list[audio_effects.ChannelMatrix | audio_effects.Waveshaper],
) -> audio_effects.EffectGraph:
    first = processors[0]
    channels = (
        first.input_channels
        if isinstance(first, audio_effects.ChannelMatrix)
        else ["left", "right"]
    )
    return effects.serial_graph(
        audio_effects.AttachmentScope.master,
        audio_effects.GraphInput(
            name="main", channels=channels, source=audio_effects.MainStream()
        ),
        processors,
        48000,
    )


@pytest.mark.parametrize(
    "kind",
    [
        "polarity",
        "pan",
        "balance",
        "width",
        "mid_side",
        "swap",
        "sum_difference",
        "expand",
        "cancellation",
    ],
)
def test_named_routing_presets_match_explicit_channel_transforms(
    tmp_path: Path, kind: str
) -> None:
    time = np.arange(48000) / 48000
    source = np.column_stack([0.3 * np.sin(2 * pi * f * time) for f in (220, 330)])
    source[32000:] = 0
    processors = []
    if kind == "polarity":
        processors = [audio_effects.polarity_matrix("route", ["left", "right"])]
        expected = -source
    elif kind == "pan":
        source = source[:, :1].copy()
        processors = [audio_effects.pan_matrix("route", 0.25)]
        expected = source * [cos(5 * pi / 16), sin(5 * pi / 16)]
    elif kind == "balance":
        processors = [audio_effects.balance_matrix("route", -0.25)]
        expected = source * [1, 0.75]
    elif kind == "width":
        processors = [audio_effects.width_matrix("route", 1.5)]
        mid = (source[:, 0] + source[:, 1]) / 2
        side = (source[:, 0] - source[:, 1]) / 2
        expected = np.column_stack([mid + 1.5 * side, mid - 1.5 * side])
    elif kind == "mid_side":
        processors = [
            audio_effects.mid_side_matrix("encode"),
            audio_effects.mid_side_matrix("decode", decode=True),
        ]
        expected = source
    elif kind == "swap":
        processors = [
            audio_effects.ChannelMatrix(
                name="route",
                input_channels=["left", "right"],
                output_channels=["left", "right"],
                coefficients=[[0, 1], [1, 0]],
            )
        ]
        expected = source[:, ::-1].copy()
    elif kind == "sum_difference":
        processors = [
            audio_effects.ChannelMatrix(
                name="route",
                input_channels=["left", "right"],
                output_channels=["sum", "difference"],
                coefficients=[[1, 1], [1, -1]],
            )
        ]
        expected = np.column_stack(
            [source[:, 0] + source[:, 1], source[:, 0] - source[:, 1]]
        )
    elif kind == "expand":
        processors = [
            audio_effects.ChannelMatrix(
                name="route",
                input_channels=["left", "right"],
                output_channels=["left", "right", "mono"],
                coefficients=[[1, 0], [0, 1], [0.5, 0.5]],
            )
        ]
        expected = np.column_stack([source, np.mean(source, axis=1)])
    else:
        middle = source[:, 0].copy()
        source = np.column_stack([np.full(48000, 1e16), middle, np.full(48000, -1e16)])
        source[32000:] = 0
        processors = [
            audio_effects.ChannelMatrix(
                name="route",
                input_channels=["a", "b", "c"],
                output_channels=["mono"],
                coefficients=[[1, 1, 1]],
            )
        ]
        expected = middle[:, None]
    prepared = effects.prepare(matrix_graph(processors), 48000)
    assert prepared.latency_frames == 0
    actions = [audio_effects.InputEndAction(tick=32000, ordinal=0, input="main")]
    for backend in ("numpy", "native"):
        renderer = effects.OfflineEffects(prepared, backend)
        first = renderer.advance({"main": source[:23928]}, [], 0, 23928)
        saved = effects.EffectSnapshot.model_validate_json(
            renderer.snapshot().model_dump_json()
        )
        renderer.restore(saved)
        last = renderer.advance({"main": source[23928:]}, actions, 23928, 48000)
        assert renderer.drained
        check_audio(
            tmp_path / f"{kind}-{backend}.wav", np.vstack([first, last]), expected
        )
    callback, replay, drained = callback_audio(prepared, {"main": source}, actions)
    assert drained
    check_audio(tmp_path / f"{kind}-callback.wav", callback, expected)
    check_audio(tmp_path / f"{kind}-replay.wav", replay, callback)


@pytest.mark.parametrize("failure", ["product", "action"])
def test_matrix_callback_failure_zeros_the_block_and_restores(
    tmp_path: Path, failure: str
) -> None:
    matrix = audio_effects.ChannelMatrix(
        name="route",
        input_channels=["left", "right"],
        output_channels=["left", "right"],
        coefficients=[[2, 0], [0, 2]],
    )
    prepared = effects.prepare(matrix_graph([matrix]), 48000)
    runtime = _native.LiveRuntime.empty(48000, 2, 48000, 64)
    runtime.set_effect_graph(
        *effects.native_live_graph(prepared), 2, input_routes=[[-2]]
    )
    saved = runtime.snapshot()
    source = np.full((48000, 2), 1e308 if failure == "product" else 0.1)
    if failure == "action":
        runtime.submit_effects(np.asarray([[12003, 0, 0, 1, 0, 0]], dtype=np.float64))
    actual = np.ones_like(source)
    with pytest.raises(ValueError, match="[Mm]atri|[Rr]outing"):
        runtime.process_into(actual, [source])
    assert runtime.failed()
    check_audio(tmp_path / "failed.wav", actual, np.zeros_like(actual))
    runtime.restore(saved)
    source.fill(0.1)
    runtime.process_into(actual, [source])
    check_audio(tmp_path / "recovered.wav", actual, source * 2)


def test_channel_count_change_preserves_upstream_latency_and_snapshots(
    tmp_path: Path,
) -> None:
    matrix = audio_effects.ChannelMatrix(
        name="route",
        input_channels=["left", "right"],
        output_channels=["left", "right", "mono"],
        coefficients=[[1, 0], [0, 1], [0.5, 0.5]],
    )
    prepared = effects.prepare(
        matrix_graph([audio_effects.Waveshaper(name="latency", mix=0), matrix]), 48000
    )
    assert prepared.latency_frames == 48
    source = np.column_stack(
        [0.3 * np.sin(np.arange(48000) * v / 48000) for v in (631, 997)]
    )
    source[32000:] = 0
    delayed = np.zeros_like(source)
    delayed[48:] = source[:-48]
    expected = np.column_stack([delayed, np.mean(delayed, axis=1)])
    actions = [audio_effects.InputEndAction(tick=32000, ordinal=0, input="main")]
    for origin, target in (
        ("numpy", "native"),
        ("native", "numpy"),
        ("native", "native"),
    ):
        renderer = effects.OfflineEffects(prepared, origin)
        first = renderer.advance({"main": source[:23928]}, [], 0, 23928)
        saved = effects.EffectSnapshot.model_validate_json(
            renderer.snapshot().model_dump_json()
        )
        restored = effects.OfflineEffects(prepared, target)
        restored.restore(saved)
        last = restored.advance({"main": source[23928:]}, actions, 23928, 48000)
        assert restored.drained
        check_audio(
            tmp_path / f"latency-{origin}-{target}.wav",
            np.vstack([first, last]),
            expected,
        )
    callback, replay, drained = callback_audio(prepared, {"main": source}, actions)
    assert drained
    check_audio(tmp_path / "latency-callback.wav", callback, expected)
    check_audio(tmp_path / "latency-replay.wav", replay, callback)


def test_native_matrix_snapshot_requires_matching_coefficients(tmp_path: Path) -> None:
    runtimes = []
    for gain in (1, 2):
        matrix = audio_effects.ChannelMatrix(
            name="route",
            input_channels=["left", "right"],
            output_channels=["left", "right"],
            coefficients=[[gain, 0], [0, gain]],
        )
        prepared = effects.prepare(matrix_graph([matrix]), 48000)
        runtime = _native.LiveRuntime.empty(48000, 2, 48000, 64)
        runtime.set_effect_graph(
            *effects.native_live_graph(prepared), 2, input_routes=[[-2]]
        )
        runtimes.append(runtime)
    with pytest.raises(ValueError, match="[Ss]napshot"):
        runtimes[1].restore(runtimes[0].snapshot())
    source = np.full((48000, 2), 0.1)
    actual = np.empty_like(source)
    runtimes[1].process_into(actual, [source])
    check_audio(tmp_path / "unchanged-definition.wav", actual, source * 2)
