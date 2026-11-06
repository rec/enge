from pathlib import Path

import numpy as np
import pytest
from test_effects import graph_input
from test_live import native_oscillator_runtime
from test_noise import trigger
from test_sample_instrument import sample_score
from test_synth import check_audio, score
from ufor import audio_effects
from ufor.samples import instrument, trace
from ufor.synth_trace import prepare as prepare_trace

from enge import _native, effects, live, sample_instrument, synth


@pytest.mark.parametrize("owner", ["python", "native"])
@pytest.mark.parametrize("tap", [False, True])
def test_live_routes_main_or_individual_source_with_named_host_input(
    tmp_path: Path, owner: str, tap: bool
) -> None:
    definition = synth.prepare(score())
    graph = audio_effects.EffectGraph(
        scope=audio_effects.AttachmentScope.master,
        inputs=[
            graph_input(),
            graph_input("side", audio_effects.HostStream(stream="external")),
            graph_input(
                "selected",
                audio_effects.InstrumentStream(instrument="one", tap="pre_effect"),
            ),
        ],
        processors=[audio_effects.Multiply(name="ring")],
        connections=[
            audio_effects.Connection(
                processor="ring",
                port="carrier",
                source=audio_effects.InputSource(input="selected" if tap else "main"),
            ),
            audio_effects.Connection(
                processor="ring",
                port="modulator",
                source=audio_effects.InputSource(input="side"),
            ),
        ],
        output=audio_effects.ProcessorSource(processor="ring"),
        maximum_block_frames=997,
    )
    prepared = effects.prepare(graph, 48000)
    sources = {n: synth.PersistentSynth(definition, voices=2) for n in ("one", "two")}
    engine = (
        live.NativeLiveEngine(sources, 997, prepared)
        if owner == "native"
        else live.LiveEngine(sources, effects.OfflineEffects(prepared))
    )
    actions = {
        n: list(
            prepare_trace(
                score().body,
                [trigger().model_copy(update={"pitch_hz": p})],
                seed=0,
            ).actions
        )
        for n, p in (("one", 220), ("two", 330))
    }
    dry = {
        n: synth.OfflineSynth(definition, "native").advance(a, 0, 48000)
        for n, a in actions.items()
    }
    time = np.arange(48000) / 48000
    host = np.column_stack(
        (np.sin(2 * np.pi * 7 * time), np.cos(2 * np.pi * 11 * time))
    )
    expected = (dry["one"] if tap else dry["one"] + dry["two"]) * host
    actual = np.empty_like(expected)
    for start in range(0, 48000, 997):
        end = min(start + 997, 48000)
        engine.advance_into(
            {n: [a for a in v if start <= a.tick < end] for n, v in actions.items()},
            [],
            start,
            end,
            actual[start:end],
            {"external": host[start:end]},
        )
        if end == 23928:
            saved = engine.snapshot()
    engine.restore(saved)
    replay = actual.copy()
    for start in range(23928, 48000, 997):
        end = min(start + 997, 48000)
        engine.advance_into(
            {n: [] for n in actions},
            [],
            start,
            end,
            replay[start:end],
            {"external": host[start:end]},
        )
    check_audio(tmp_path / "routes.wav", actual, expected)
    check_audio(tmp_path / "replay.wav", replay, actual)


@pytest.mark.parametrize("owner", ["python", "native"])
def test_live_host_output_declares_its_own_layout(tmp_path: Path, owner: str) -> None:
    graph = audio_effects.EffectGraph(
        scope=audio_effects.AttachmentScope.master,
        inputs=[
            audio_effects.GraphInput(
                name="detector",
                channels=["mono"],
                source=audio_effects.HostStream(stream="microphone"),
            ),
            graph_input(),
        ],
        processors=[audio_effects.Gain(name="trim", gain_db=-6)],
        connections=[
            audio_effects.Connection(
                processor="trim",
                port="input",
                source=audio_effects.InputSource(input="detector"),
            )
        ],
        output=audio_effects.ProcessorSource(processor="trim"),
        maximum_block_frames=997,
    )
    prepared = effects.prepare(graph, 48000)
    sources = {"tone": synth.PersistentSynth(synth.prepare(score()), voices=1)}
    engine = (
        live.NativeLiveEngine(sources, 997, prepared)
        if owner == "native"
        else live.LiveEngine(sources, effects.OfflineEffects(prepared))
    )
    assert engine.channels == ["mono"]
    host = np.sin(2 * np.pi * np.arange(48000)[:, None] * 220 / 48000)
    actual = np.empty_like(host)
    for start in range(0, 48000, 997):
        end = min(start + 997, 48000)
        engine.advance_into(
            {"tone": []},
            [],
            start,
            end,
            actual[start:end],
            {"microphone": host[start:end]},
        )
    check_audio(tmp_path / "mono.wav", actual, host * 10 ** (-6 / 20))


@pytest.mark.parametrize("invalid", ["missing", "width", "nonfinite"])
def test_invalid_native_host_block_preserves_queued_actions(
    tmp_path: Path, invalid: str
) -> None:
    graph = audio_effects.EffectGraph(
        scope=audio_effects.AttachmentScope.master,
        inputs=[
            graph_input(),
            graph_input("host", audio_effects.HostStream(stream="external")),
        ],
        processors=[audio_effects.Multiply(name="ring")],
        connections=[
            audio_effects.Connection(
                processor="ring", port=p, source=audio_effects.InputSource(input=i)
            )
            for p, i in (("carrier", "main"), ("modulator", "host"))
        ],
        output=audio_effects.ProcessorSource(processor="ring"),
        maximum_block_frames=48000,
    )
    runtime = _native.LiveRuntime([native_oscillator_runtime()], 48000, 64, 2)
    runtime.set_effect_graph(
        *effects.native_live_graph(effects.prepare(graph, 48000)),
        2,
        input_routes=[[-1], [-2]],
    )
    onset = np.array([[0, 0, 0, 220, 0.2, 0]], dtype=np.float64)
    runtime.submit(0, onset)
    bad = (
        []
        if invalid == "missing"
        else [np.ones((48000, 1 if invalid == "width" else 2))]
    )
    if invalid == "nonfinite":
        bad[0][123, 0] = np.nan
    actual = np.empty((48000, 2))
    with pytest.raises(ValueError, match="[Hh]ost input"):
        runtime.process_into(actual, bad)
    assert not runtime.failed()
    host = np.full((48000, 2), 0.5)
    runtime.process_into(actual, [host])
    expected = native_oscillator_runtime().process_actions(48000, 0, 1, 0, onset) * host
    check_audio(tmp_path / "admission.wav", actual, expected)


def test_native_instrument_tap_sums_all_sample_slots(tmp_path: Path) -> None:
    raw = sample_score().model_dump(mode="json")
    raw["body"]["slots"].append({**raw["body"]["slots"][0], "name": "second"})
    document = instrument.SampleInstrumentScore.model_validate(raw)
    time = np.arange(96000) / 48000
    asset = np.column_stack((np.sin(2 * np.pi * 220 * time), np.zeros(96000)))
    prepared = sample_instrument.prepare(document, {"asset": asset})
    actions = list(
        trace.prepare(
            document.body,
            [trigger().model_copy(update={"pitch_hz": 440})],
            seed=0,
        ).actions
    )
    graph = audio_effects.EffectGraph(
        scope=audio_effects.AttachmentScope.master,
        inputs=[
            graph_input(),
            graph_input(
                "selected",
                audio_effects.InstrumentStream(instrument="sample", tap="pre_effect"),
            ),
        ],
        processors=[audio_effects.Gain(name="trim", gain_db=-3)],
        connections=[
            audio_effects.Connection(
                processor="trim",
                port="input",
                source=audio_effects.InputSource(input="selected"),
            )
        ],
        output=audio_effects.ProcessorSource(processor="trim"),
        maximum_block_frames=997,
    )
    engine = live.NativeLiveEngine(
        {
            "tone": synth.PersistentSynth(synth.prepare(score()), voices=1),
            "sample": sample_instrument.PersistentSampler(prepared, voices=1),
        },
        997,
        effects.prepare(graph, 48000),
    )
    tone = list(
        prepare_trace(
            score().body,
            [trigger().model_copy(update={"pitch_hz": 330})],
            seed=0,
        ).actions
    )
    actual = np.empty((48000, 2))
    for start in range(0, 48000, 997):
        end = min(start + 997, 48000)
        engine.advance_into(
            {
                "tone": [a for a in tone if start <= a.tick < end],
                "sample": [a for a in actions if start <= a.tick < end],
            },
            [],
            start,
            end,
            actual[start:end],
        )
    expected = sample_instrument.OfflineSampler(prepared, "native").advance(
        actions, 0, 48000
    ) * 10 ** (-3 / 20)
    check_audio(tmp_path / "sample-tap.wav", actual, expected)
