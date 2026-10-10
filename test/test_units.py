from pathlib import Path
from typing import Literal

import numpy as np
import pytest
import test_fm
import test_synth
from test_dynamic_synth import onset
from test_effects import graph_input
from test_motion_latch import latch_score
from test_synth import check_audio
from ufor import audio_effects, instrument_trace, synth_trace
from ufor.events import PerformanceEvent, Trigger
from ufor.samples import instrument, trace
from ufor.synth import SynthInstrumentScore

from enge import effects, fm, presets, sample_instrument, synth


@pytest.mark.parametrize("engine", ["synth", "fm"])
@pytest.mark.parametrize("backend", ["numpy", "native", "persistent"])
def test_quantity_authored_instruments_match_canonical_audio(
    tmp_path: Path, engine: str, backend: Literal["numpy", "native", "persistent"]
) -> None:
    original = test_fm.score() if engine == "fm" else test_synth.score()
    raw = original.model_dump(mode="json")
    raw["timebases"][0]["rate"]["numerator"] = "48 kHz"
    voice = raw["body"]["voices"][0]
    voice["mapping"]["reference_pitch_hz"] = "0.44 kHz"
    voice["processing"]["volume_db"] = "0 dB"
    voice["processing"]["tuning_cents"] = "1 semitone"
    voice["minimum_hold_seconds"] = "1/3 ms"
    envelopes = (
        [o["envelope"] for o in voice["fm"]["operators"]]
        if engine == "fm"
        else [voice["envelope"]]
    )
    for envelope in envelopes:
        envelope["segments"] = [{"duration": "10 ms", "to": 1}]
        envelope["release"] = [{"duration": "20 ms", "to": 0}]
    authored = SynthInstrumentScore.model_validate(raw)
    # Revalidation of the normalized document exercises the public export boundary.
    canonical = SynthInstrumentScore.model_validate(authored.model_dump(mode="json"))
    assert authored == canonical
    event = Trigger.model_validate(
        {
            "tick": "0 tick",
            "ordinal": 0,
            "part": "main",
            "trigger_id": "note",
            "key": 69,
            "pitch_hz": "0.44 kHz",
        }
    )
    numeric_event = event.model_copy(update={"tick": 0, "pitch_hz": 440.0})
    actions: list[instrument_trace.TraceAction] = list(
        synth_trace.prepare(authored.body, [event], seed=0).actions
    )
    expected_actions: list[instrument_trace.TraceAction] = list(
        synth_trace.prepare(canonical.body, [numeric_event], seed=0).actions
    )
    assert actions == expected_actions
    if engine == "fm":
        definition = fm.prepare(authored)
        renderer = (
            fm.PersistentFM(definition, voices=1)
            if backend == "persistent"
            else fm.OfflineFM(definition, backend)
        )
        reference = fm.OfflineFM(fm.prepare(canonical))
    else:
        definition = synth.prepare(authored)
        renderer = (
            synth.PersistentSynth(definition, voices=1)
            if backend == "persistent"
            else synth.OfflineSynth(definition, backend)
        )
        reference = synth.OfflineSynth(synth.prepare(canonical))
    assert definition.sample_rate == 48000
    assert type(definition.sample_rate) is int
    check_audio(
        tmp_path / f"units-{engine}-{backend}.wav",
        renderer.advance(actions, 0, 48000),
        reference.advance(expected_actions, 0, 48000),
    )


@pytest.mark.parametrize("backend", ["numpy", "native", "persistent"])
def test_quantity_authored_motion_preserves_fractional_event_delivery(
    tmp_path: Path, backend: Literal["numpy", "native", "persistent"]
) -> None:
    canonical = latch_score("cycle", 1)
    raw = canonical.model_dump(mode="json")
    patch = raw["body"]["voices"][0]["motions"]["gesture"]["body"]
    patch["motions"]["clock"]["rate"] = "8 Hz"
    patch["motions"]["wave"]["rate"] = "2 Hz"
    patch["events"][0]["delay"] = "12001/192 ms"
    patch["events"][1]["delay"] = "31.25 ms"
    authored = SynthInstrumentScore.model_validate(raw)
    assert authored == canonical
    events: list[PerformanceEvent] = [
        onset(0, pitch=0.1).model_copy(update={"controls": {}})
    ]
    actions: list[instrument_trace.TraceAction] = list(
        synth_trace.prepare(authored.body, events, seed=81).actions
    )
    definition = synth.prepare(authored)
    renderer = (
        synth.PersistentSynth(definition, voices=1)
        if backend == "persistent"
        else synth.OfflineSynth(definition, backend)
    )
    first = renderer.advance(actions, 0, 9000)
    snapshot = renderer.snapshot()
    tail = renderer.advance([], 9000, 48000)
    if isinstance(renderer, synth.PersistentSynth):
        assert isinstance(snapshot, synth.PersistentSynthSnapshot)
        renderer.restore(snapshot)
    else:
        assert isinstance(snapshot, synth.SynthSnapshot)
        renderer.restore(snapshot)
    np.testing.assert_array_equal(renderer.advance([], 9000, 48000), tail)
    reference = synth.OfflineSynth(synth.prepare(canonical))
    check_audio(
        tmp_path / f"units-motion-{backend}.wav",
        np.vstack((first, tail)),
        reference.advance(actions, 0, 48000),
    )


@pytest.mark.parametrize("backend", ["numpy", "native"])
def test_quantity_authored_effects_keep_amplitude_decibels_and_automation(
    tmp_path: Path, backend: Literal["numpy", "native"]
) -> None:
    graph = effects.serial_graph(
        audio_effects.AttachmentScope.voice,
        graph_input(),
        [
            audio_effects.Gain.model_validate({"name": "trim", "gain_db": "-6 dB"}),
            audio_effects.TapDelay.model_validate(
                {
                    "name": "echo",
                    "delay_seconds": "1 ms",
                    "maximum_delay_seconds": "10 ms",
                    "feedback": 0,
                }
            ),
        ],
        48000,
    )
    action = audio_effects.ParameterAction.model_validate(
        {
            "tick": "24000 frame",
            "ordinal": 0,
            "processor": "echo",
            "parameter": "delay_seconds",
            "value": "2 ms",
            "duration_frames": "64 frame",
        }
    )
    canonical = audio_effects.EffectGraph.model_validate(graph.model_dump(mode="json"))
    numeric_action = action.model_copy(update={"value": 0.002})
    source = np.ones((48000, 2))
    actual = effects.OfflineEffects(effects.prepare(graph, 48000), backend).advance(
        {"main": source}, [action], 0, 48000
    )
    expected = effects.OfflineEffects(effects.prepare(canonical, 48000)).advance(
        {"main": source}, [numeric_action], 0, 48000
    )
    assert actual[1000, 0] == pytest.approx(10 ** (-6 / 20))
    check_audio(tmp_path / f"units-effects-{backend}.wav", actual, expected)


@pytest.mark.parametrize("backend", ["numpy", "native", "persistent"])
def test_quantity_authored_sampler_preserves_asset_identity_and_traversal(
    tmp_path: Path, backend: Literal["numpy", "native", "persistent"]
) -> None:
    document, assets = presets.Patch(engine="sample").prepare_score(48000)
    assert isinstance(document, instrument.SampleInstrumentScore)
    raw = document.model_dump(mode="json")
    raw["timebases"][0]["rate"]["numerator"] = "48 kHz"
    raw["assets"][0]["audio"]["frames"] = "48000 frame"
    raw["assets"][0]["content"]["byte_length"] = "375 KiB"
    raw["body"]["slices"][0]["end_frame"] = "48000 frame"
    raw["body"]["slices"][0]["loop"].update(
        start_frame="0 frame", end_frame="48000 frame"
    )
    voice = raw["body"]["slots"][0]
    voice["mapping"]["reference_pitch_hz"] = "0.22 kHz"
    voice["processing"]["volume_db"] = "-6 dB"
    voice["processing"]["tuning_cents"] = "1 semitone"
    for parameter in voice["modulation"]["parameters"]:
        if parameter["target"]["parameter"] == "tuning_cents":
            parameter["default"] = "1 semitone"
    authored = instrument.SampleInstrumentScore.model_validate(raw)
    canonical = instrument.SampleInstrumentScore.model_validate(
        authored.model_dump(mode="json")
    )
    assert authored.assets[0].content == document.assets[0].content
    events: list[PerformanceEvent] = [
        onset(0, pitch=220).model_copy(update={"controls": {}})
    ]
    actions: list[instrument_trace.TraceAction] = list(
        trace.prepare(authored.body, events, seed=0).actions
    )
    definition = sample_instrument.prepare(authored, assets)
    renderer = (
        sample_instrument.PersistentSampler(definition, voices=1)
        if backend == "persistent"
        else sample_instrument.OfflineSampler(definition, backend)
    )
    reference = sample_instrument.OfflineSampler(
        sample_instrument.prepare(canonical, assets)
    )
    check_audio(
        tmp_path / f"units-sampler-{backend}.wav",
        renderer.advance(actions, 0, 48000),
        reference.advance(actions, 0, 48000),
    )
