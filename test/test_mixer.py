from math import pi
from pathlib import Path

import numpy as np
import pytest
from test_persistent_effects import callback_audio
from test_synth import check_audio
from ufor import audio_effects

from enge import _native, effects
from enge.synth import EngineError


def mixer_graph() -> audio_effects.EffectGraph:
    inputs = [
        audio_effects.GraphInput(
            name=n,
            channels=["left", "right"],
            source=audio_effects.MainStream()
            if n == "main"
            else audio_effects.HostStream(stream=n),
        )
        for n in ("main", "b", "c", "d")
    ]
    return audio_effects.EffectGraph(
        scope="master",
        inputs=inputs,
        processors=[
            audio_effects.Mixer(
                name="sum",
                inputs=[
                    audio_effects.MixerInput(name=n, gain_db=-6 if n == "d" else 0)
                    for n in ("a", "b", "c", "d")
                ],
            )
        ],
        connections=[
            audio_effects.Connection(
                processor="sum", port=p, source=audio_effects.InputSource(input=n)
            )
            for p, n in zip(("a", "b", "c", "d"), ("main", "b", "c", "d"), strict=True)
        ],
        output=audio_effects.ProcessorSource(processor="sum"),
        maximum_block_frames=48000,
    )


def crossfader_graph(law: str) -> audio_effects.EffectGraph:
    raw = mixer_graph().model_dump()
    raw["inputs"] = raw["inputs"][:2]
    raw["processors"] = [
        {"kind": "crossfader", "name": "sum", "law": law, "position": 0}
    ]
    raw["connections"] = [
        {"processor": "sum", "port": p, "source": {"kind": "input", "input": n}}
        for p, n in [("first", "main"), ("second", "b")]
    ]
    return audio_effects.EffectGraph.model_validate(raw)


def test_four_input_mixer_retimes_mute_solo_and_replays_pending_fades(
    tmp_path: Path,
) -> None:
    prepared = effects.prepare(mixer_graph(), 48000)
    frames = np.arange(48000)
    inputs = {
        n: np.column_stack(
            [a * np.sin(2 * pi * (f + c * 37) * frames / 48000) for c in range(2)]
        )
        for n, a, f in [
            ("main", 0.2, 220),
            ("b", 0.1, 330),
            ("c", 0.05, 550),
            ("d", 0.15, 770),
        ]
    }
    endings = {"main": 30000, "b": 32000, "c": 36000, "d": 40000}
    for name, end in endings.items():
        inputs[name][end:] = 0
    changes = [
        (10000, "b-solo", 1),
        (10016, "b-muted", 1),
        (10032, "b-solo", 0),
        (10048, "b-muted", 0),
        (23900, "a-muted", 1),
        (23920, "d-solo", 1),
        (23940, "d-solo", 0),
        (23960, "a-muted", 0),
    ]
    actions: list[audio_effects.EffectAction] = [
        audio_effects.ParameterAction(
            tick=t, ordinal=0, processor="sum", parameter=p, value=v
        )
        for t, p, v in changes
    ]
    actions.extend(
        [
            audio_effects.ParameterAction(
                tick=15003,
                ordinal=0,
                processor="sum",
                parameter="c-gain_db",
                value="-12 dB",
                duration_frames=512,
            ),
            audio_effects.ParameterAction(
                tick=23600,
                ordinal=0,
                processor="sum",
                parameter="d-gain_db",
                value="-3 dB",
                duration_frames=1200,
            ),
        ]
    )
    actions.extend(
        audio_effects.InputEndAction(tick=t, ordinal=0, input=n)
        for n, t in endings.items()
    )
    actions.sort(key=lambda a: (a.tick, a.ordinal))
    early = (
        1
        - 0.5 * np.clip((frames - 10000) / 32, 0, 1)
        + 0.5 * np.clip((frames - 10032) / 64, 0, 1)
    )
    early_b = (
        1
        - 0.5 * np.clip((frames - 10016) / 32, 0, 1)
        + 0.5 * np.clip((frames - 10048) / 64, 0, 1)
    )
    late_a = (
        1
        - (60 / 64) * np.clip((frames - 23900) / 60, 0, 1)
        + (60 / 64) * np.clip((frames - 23960) / 64, 0, 1)
    )
    late_bc = (
        1
        - (20 / 64) * np.clip((frames - 23920) / 20, 0, 1)
        + (20 / 64) * np.clip((frames - 23940) / 64, 0, 1)
    )
    weights = {
        "main": early + late_a - 1,
        "b": early_b + late_bc - 1,
        "c": (early + late_bc - 1)
        * 10 ** (-12 * np.clip((frames - 15003) / 512, 0, 1) / 20),
        "d": early * 10 ** ((-6 + 3 * np.clip((frames - 23600) / 1200, 0, 1)) / 20),
    }
    expected = sum(inputs[n] * weights[n][:, None] for n in inputs)
    for origin, target in (
        ("numpy", "native"),
        ("native", "numpy"),
        ("native", "native"),
    ):
        renderer = effects.OfflineEffects(prepared, origin)
        first = renderer.advance(
            {n: a[:23928] for n, a in inputs.items()},
            [a for a in actions if a.tick < 23928],
            0,
            23928,
        )
        saved = effects.EffectSnapshot.model_validate_json(
            renderer.snapshot().model_dump_json()
        )
        restored = effects.OfflineEffects(prepared, target)
        restored.restore(saved)
        if target == origin == "native":
            invalid = saved.model_copy(deep=True)
            assert invalid.processors["sum"].mixer is not None
            invalid.processors["sum"].mixer.fades[0].remaining = 65
            with pytest.raises(ValueError, match="mixer fade"):
                restored.restore(invalid)
            assert restored.snapshot() == saved
        last = restored.advance(
            {n: a[23928:] for n, a in inputs.items()},
            [a for a in actions if a.tick >= 23928],
            23928,
            48000,
        )
        assert restored.drained
        check_audio(
            tmp_path / f"four-input-{origin}-{target}.wav",
            np.vstack([first, last]),
            expected,
        )
    callback, replay, drained = callback_audio(prepared, inputs, actions)
    assert drained
    check_audio(tmp_path / "four-input-callback.wav", callback, expected)
    check_audio(tmp_path / "four-input-replay.wav", replay, callback)


@pytest.mark.parametrize("law", ["linear", "equal_power"])
def test_crossfader_ramps_preserve_law_and_independent_input_endings(
    tmp_path: Path, law: str
) -> None:
    prepared = effects.prepare(crossfader_graph(law), 48000)
    frames = np.arange(48000)
    inputs = {
        "main": np.column_stack(
            [0.2 * np.cos(2 * pi * f * frames / 48000) for f in (220, 330)]
        ),
        "b": np.column_stack(
            [0.3 * np.sin(2 * pi * f * frames / 48000) for f in (440, 550)]
        ),
    }
    inputs["main"][26000:] = 0
    inputs["b"][36000:] = 0
    position = np.clip((frames - 10003) / 1024, 0, 1) - 0.75 * np.clip(
        (frames - 23800) / 1024, 0, 1
    )
    position[35000:] = 1
    left, right = 1 - position, position
    if law == "equal_power":
        left, right = np.cos(pi * position / 2), np.sin(pi * position / 2)
        left[position == 1] = 0
        right[position == 0] = 0
    expected = inputs["main"] * left[:, None] + inputs["b"] * right[:, None]
    actions: list[audio_effects.EffectAction] = [
        audio_effects.ParameterAction(
            tick=10003,
            ordinal=0,
            processor="sum",
            parameter="position",
            value=1,
            duration_frames=1024,
        ),
        audio_effects.ParameterAction(
            tick=23800,
            ordinal=0,
            processor="sum",
            parameter="position",
            value=0.25,
            duration_frames=1024,
        ),
        audio_effects.ParameterAction(
            tick=35000, ordinal=0, processor="sum", parameter="position", value=1
        ),
        audio_effects.InputEndAction(tick=26000, ordinal=0, input="main"),
        audio_effects.InputEndAction(tick=36000, ordinal=0, input="b"),
    ]
    actions.sort(key=lambda a: (a.tick, a.ordinal))
    for backend in ("numpy", "native"):
        renderer = effects.OfflineEffects(prepared, backend)
        first = renderer.advance(
            {n: a[:23928] for n, a in inputs.items()},
            [a for a in actions if a.tick < 23928],
            0,
            23928,
        )
        saved = effects.EffectSnapshot.model_validate_json(
            renderer.snapshot().model_dump_json()
        )
        renderer.restore(saved)
        last = renderer.advance(
            {n: a[23928:] for n, a in inputs.items()},
            [a for a in actions if a.tick >= 23928],
            23928,
            48000,
        )
        assert renderer.drained
        check_audio(
            tmp_path / f"fade-{law}-{backend}.wav", np.vstack([first, last]), expected
        )
    callback, replay, drained = callback_audio(prepared, inputs, actions)
    assert drained
    check_audio(tmp_path / f"fade-{law}-callback.wav", callback, expected)
    check_audio(tmp_path / f"fade-{law}-replay.wav", replay, callback)


@pytest.mark.parametrize("failure", ["selector", "gain", "crossfade"])
def test_mixer_and_crossfader_callback_failures_restore_without_partial_output(
    tmp_path: Path, failure: str
) -> None:
    graph = crossfader_graph("linear") if failure == "crossfade" else mixer_graph()
    prepared = effects.prepare(graph, 48000)
    runtime = _native.LiveRuntime.empty(48000, 2, 48000, 64)
    runtime.set_effect_graph(
        *effects.native_live_graph(prepared),
        2,
        input_routes=[[-2] for _ in graph.inputs],
    )
    saved = runtime.snapshot()
    parameter, bad = {
        "selector": ("b-muted", 0.5),
        "gain": ("b-gain_db", 1e308),
        "crossfade": ("position", 2),
    }[failure]
    action = audio_effects.ParameterAction(
        tick=12003, ordinal=0, processor="sum", parameter=parameter, value=0
    )
    encoded = effects.native_live_actions(prepared, [action], 0, 48000)
    encoded[0, 4] = bad
    runtime.submit_effects(encoded)
    source = np.full((48000, 2), 0.1)
    actual = np.ones_like(source)
    with pytest.raises(ValueError, match="[Mm]ixer|[Cc]rossfader"):
        runtime.process_into(actual, [source for _ in graph.inputs])
    assert runtime.failed()
    check_audio(tmp_path / "failed.wav", actual, np.zeros_like(actual))
    runtime.restore(saved)
    runtime.process_into(actual, [source for _ in graph.inputs])
    expected = source if failure == "crossfade" else source * (3 + 10 ** (-6 / 20))
    check_audio(tmp_path / "recovered.wav", actual, expected)


@pytest.mark.parametrize("kind", ["mixer", "crossfader"])
def test_mixer_gain_and_crossfade_use_the_shared_envelfo_clock(
    tmp_path: Path, kind: str
) -> None:
    raw = (
        mixer_graph() if kind == "mixer" else crossfader_graph("equal_power")
    ).model_dump(mode="json")
    if kind == "crossfader":
        raw["processors"][0]["position"] = 0.5
    parameter, default, depth, unit = (
        ("b-gain_db", 0, 6, "db")
        if kind == "mixer"
        else ("position", 0.5, 0.5, "ratio")
    )
    raw.update(
        envelfos={"motion": {"kind": "lfo", "shape": "square", "rate": 2}},
        bindings=[{"kind": "envelfo", "name": "motion", "reference": "motion"}],
        modulation={
            "sources": [
                {"name": "motion", "scope": "voice", "minimum": -1, "maximum": 1}
            ],
            "parameters": [
                {
                    "target": {"name": "sum", "parameter": parameter},
                    "unit": unit,
                    "scope": "voice",
                    "minimum": default - depth,
                    "maximum": default + depth,
                    "default": default,
                }
            ],
            "routes": [
                {
                    "name": "motion",
                    "source": "motion",
                    "target": {"name": "sum", "parameter": parameter},
                    "unit": unit,
                    "operation": "add",
                    "points": [
                        {"input": -1, "amount": -depth},
                        {"input": 1, "amount": depth},
                    ],
                }
            ],
        },
    )
    graph = audio_effects.EffectGraph.model_validate(raw)
    prepared = effects.prepare(graph, 48000)
    frames = np.arange(48000)
    inputs = {
        i.name: np.column_stack(
            [
                0.1
                * (p + 1)
                * np.sin(2 * pi * (220 + 110 * p + 37 * c) * frames / 48000)
                for c in range(2)
            ]
        )
        for p, i in enumerate(graph.inputs)
    }
    positive = (frames // 12000) % 2 == 0
    if kind == "mixer":
        expected = (
            inputs["main"]
            + inputs["c"]
            + inputs["d"] * 10 ** (-6 / 20)
            + inputs["b"] * 10 ** (np.where(positive, 6, -6)[:, None] / 20)
        )
    else:
        expected = np.where(positive[:, None], inputs["b"], inputs["main"])
    for backend in ("numpy", "native"):
        actual = effects.OfflineEffects(prepared, backend).advance(inputs, [], 0, 48000)
        check_audio(tmp_path / f"lfo-{kind}-{backend}.wav", actual, expected)
    callback, replay, drained = callback_audio(prepared, inputs, [])
    assert not drained
    check_audio(tmp_path / f"lfo-{kind}-callback.wav", callback, expected)
    check_audio(tmp_path / f"lfo-{kind}-replay.wav", replay, callback)


def test_four_mixer_ports_align_latency_controls_endings_and_json_state(
    tmp_path: Path,
) -> None:
    raw = mixer_graph().model_dump()
    raw["processors"].insert(
        0, audio_effects.Waveshaper(name="latency", mix=0).model_dump()
    )
    raw["connections"][0]["source"] = {"kind": "processor", "processor": "latency"}
    raw["connections"].append(
        {
            "processor": "latency",
            "port": "input",
            "source": {"kind": "input", "input": "main"},
        }
    )
    prepared = effects.prepare(audio_effects.EffectGraph.model_validate(raw), 48000)
    assert prepared.latency_frames == 48
    frames = np.arange(48000)
    inputs = {
        n: np.column_stack(
            [
                0.1
                * (i + 1)
                * np.sin(2 * pi * (220 + 110 * i + 37 * c) * frames / 48000)
                for c in range(2)
            ]
        )
        for i, n in enumerate(("main", "b", "c", "d"))
    }
    delayed = {}
    for name, source in inputs.items():
        source[30000:] = 0
        delayed[name] = np.zeros_like(source)
        delayed[name][48:] = source[:-48]
    gain = 10 ** (-6 * np.clip((frames - 23948) / 400, 0, 1) / 20)
    expected = (
        delayed["main"]
        + delayed["b"] * gain[:, None]
        + delayed["c"]
        + delayed["d"] * 10 ** (-6 / 20)
    )
    actions: list[audio_effects.EffectAction] = [
        audio_effects.ParameterAction(
            tick=23900,
            ordinal=0,
            processor="sum",
            parameter="b-gain_db",
            value="-6 dB",
            duration_frames=400,
        )
    ]
    actions.extend(
        audio_effects.InputEndAction(tick=30000, ordinal=i, input=n)
        for i, n in enumerate(inputs)
    )
    for origin, target in (
        ("numpy", "native"),
        ("native", "numpy"),
        ("native", "native"),
    ):
        renderer = effects.OfflineEffects(prepared, origin)
        first = renderer.advance(
            {n: a[:23928] for n, a in inputs.items()}, [actions[0]], 0, 23928
        )
        saved = effects.EffectSnapshot.model_validate_json(
            renderer.snapshot().model_dump_json()
        )
        restored = effects.OfflineEffects(prepared, target)
        restored.restore(saved)
        last = restored.advance(
            {n: a[23928:] for n, a in inputs.items()}, actions[1:], 23928, 48000
        )
        assert restored.drained
        check_audio(
            tmp_path / f"aligned-{origin}-{target}.wav",
            np.vstack([first, last]),
            expected,
        )
    callback, replay, drained = callback_audio(prepared, inputs, actions)
    assert drained
    check_audio(tmp_path / "aligned-callback.wav", callback, expected)
    check_audio(tmp_path / "aligned-replay.wav", replay, callback)


def test_mixer_preserves_small_signals_under_multichannel_cancellation(
    tmp_path: Path,
) -> None:
    prepared = effects.prepare(mixer_graph(), 48000)
    middle = np.column_stack(
        [0.2 * np.sin(2 * pi * f * np.arange(48000) / 48000) for f in (220, 330)]
    )
    inputs = {
        "main": np.full((48000, 2), 1e16),
        "b": middle.copy(),
        "c": np.full((48000, 2), -1e16),
        "d": np.zeros((48000, 2)),
    }
    for backend in ("numpy", "native"):
        actual = effects.OfflineEffects(prepared, backend).advance(inputs, [], 0, 48000)
        check_audio(tmp_path / f"cancellation-{backend}.wav", actual, middle)
    callback, replay, _ = callback_audio(prepared, inputs, [])
    check_audio(tmp_path / "cancellation-callback.wav", callback, middle)
    check_audio(tmp_path / "cancellation-replay.wav", replay, callback)


def test_mixer_native_snapshots_reject_renamed_ports(tmp_path: Path) -> None:
    runtimes = []
    for renamed in (False, True):
        raw = mixer_graph().model_dump()
        if renamed:
            for definition in raw["processors"][0]["inputs"]:
                definition["name"] = "renamed_" + definition["name"]
            for connection in raw["connections"]:
                connection["port"] = "renamed_" + connection["port"]
        prepared = effects.prepare(audio_effects.EffectGraph.model_validate(raw), 48000)
        runtime = _native.LiveRuntime.empty(48000, 2, 48000, 64)
        runtime.set_effect_graph(
            *effects.native_live_graph(prepared),
            2,
            input_routes=[[-2] for _ in prepared.definition.inputs],
        )
        runtimes.append(runtime)
    with pytest.raises(ValueError, match="[Ss]napshot"):
        runtimes[1].restore(runtimes[0].snapshot())
    source = np.full((48000, 2), 0.1)
    actual = np.empty_like(source)
    runtimes[1].process_into(actual, [source] * 4)
    check_audio(tmp_path / "renamed-ports.wav", actual, source * (3 + 10 ** (-6 / 20)))


def test_mixer_rejects_inexact_native_fade_durations() -> None:
    raw = mixer_graph().model_dump()
    raw["processors"][0]["switch_fade_frames"] = 2**53 + 1
    with pytest.raises(EngineError, match="fade duration"):
        effects.prepare(audio_effects.EffectGraph.model_validate(raw), 48000)
