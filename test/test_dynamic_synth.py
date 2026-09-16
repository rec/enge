from fractions import Fraction
from itertools import pairwise
from pathlib import Path
from typing import Literal

import numpy as np
import pytest
from test_synth import check_audio, score
from ufor import synth_trace
from ufor.envelope import Envelope, Segment
from ufor.events import ControlChange, Release, Trigger
from ufor.instrument_trace import TraceAction, VoiceRetirement
from ufor.synth import SynthInstrumentScore

from enge.synth import EngineError, OfflineSynth, SynthSnapshot, prepare


def dynamic_score(
    scope: str = "trigger", smoothing: str = "1/200", waveform: str = "sine"
) -> SynthInstrumentScore:
    raw = score().model_dump(mode="json")
    raw["body"]["controls"] = {"gain": {"default": 0.5}, "bend": {"default": 0}}
    voice = raw["body"]["voices"][0]
    voice["oscillator"]["waveform"] = waveform
    voice["modulation"] = {
        "parameters": [
            {
                "target": {"name": "processing", "parameter": "amplitude"},
                "unit": "ratio",
                "scope": "voice",
                "minimum": 0,
                "maximum": 1,
                "default": 1,
            },
            {
                "target": {"name": "processing", "parameter": "tuning_cents"},
                "unit": "cents",
                "scope": "voice",
                "minimum": 0,
                "maximum": 1200,
                "default": 0,
            },
        ],
        "sources": [
            {"name": n, "scope": scope, "minimum": 0, "maximum": 1}
            for n in ("gain", "bend")
        ],
        "routes": [
            {
                "name": "gain",
                "source": "gain",
                "target": {"name": "processing", "parameter": "amplitude"},
                "operation": "multiply",
                "unit": "ratio",
                "points": [{"input": 0, "amount": 0}, {"input": 1, "amount": 1}],
            },
            {
                "name": "bend",
                "source": "bend",
                "target": {"name": "processing", "parameter": "tuning_cents"},
                "operation": "add",
                "unit": "cents",
                "points": [{"input": 0, "amount": 0}, {"input": 1, "amount": 1200}],
            },
        ],
    }
    voice["bindings"] = [
        {
            "name": n,
            "kind": "control",
            "control": n,
            "smoothing": smoothing if n == "gain" else "0",
        }
        for n in ("gain", "bend")
    ]
    return SynthInstrumentScore.model_validate(raw)


def onset(
    frame: int = 0,
    trigger_id: str = "note",
    part: str = "main",
    pitch: float = 100,
    gain: float = 0.5,
) -> Trigger:
    return Trigger(
        tick=frame,
        ordinal=0,
        part=part,
        trigger_id=trigger_id,
        key=69,
        pitch_hz=pitch,
        controls={"gain": gain},
    )


def change(
    frame: int,
    value: float,
    control: str = "gain",
    ordinal: int = 1,
    scope: str = "trigger",
    part: str = "main",
    trigger_id: str = "note",
) -> ControlChange:
    return ControlChange.model_validate(
        {
            "tick": frame,
            "ordinal": ordinal,
            "control": control,
            "value": value,
            "scope": scope,
            "part": None if scope == "instrument" else part,
            "trigger_id": trigger_id if scope == "trigger" else None,
        }
    )


def render(
    document: SynthInstrumentScore,
    actions: list[TraceAction],
    backend: Literal["numpy", "native"],
    block: int = 48000,
) -> tuple[np.ndarray, SynthSnapshot]:
    renderer = OfflineSynth(prepare(document), backend)
    boundaries = (
        sorted(
            {
                0,
                48000,
                *range(0, 48000, block),
                119,
                120,
                121,
                239,
                240,
                241,
                270,
                301,
                24001,
            }
        )
        if block != 48000
        else [0, 48000]
    )
    chunks: list[np.ndarray] = []
    for start, end in pairwise(boundaries):
        chunks.append(
            renderer.advance([a for a in actions if start <= a.tick < end], start, end)
        )
        if block != 48000 and end in (121, 270, 24001):
            saved = renderer.snapshot()
            restored = OfflineSynth(prepare(document), backend)
            restored.restore(SynthSnapshot.model_validate_json(saved.model_dump_json()))
            # Advancing the original cannot mutate the saved or restored state.
            renderer.advance([], end, end + 1)
            assert restored.snapshot() == saved
            renderer = restored
    return np.concatenate(chunks), renderer.snapshot()


def test_pitch_steps_preserve_phase_and_gain_ramps_interrupt(
    tmp_path: Path, backend: Literal["numpy", "native"]
) -> None:
    document = dynamic_score()
    actions = synth_trace.prepare(
        document.body,
        [onset(gain=0), change(0, 1), change(120, 1, "bend", 0), change(120, 0)],
        seed=0,
    ).actions
    actual = OfflineSynth(prepare(document), backend).advance(actions, 0, 48000)
    frames = np.arange(48000)
    phase = (np.minimum(frames, 120) * 100 + np.maximum(frames - 120, 0) * 200) / 48000
    gain = np.where(
        frames < 120, frames / 240, 0.5 * np.maximum(0, 1 - (frames - 120) / 240)
    )
    expected = np.zeros_like(actual)
    expected[:, 0] = np.sin(2 * np.pi * phase) * gain
    check_audio(tmp_path / "pitch-and-gain.wav", actual, expected)
    assert actual[120, 0] == pytest.approx(0.5, rel=0, abs=1e-12)
    assert np.all(actual[360:] == 0)


def test_square_pitch_steps_switch_on_the_exact_sample(
    tmp_path: Path, backend: Literal["numpy", "native"]
) -> None:
    document = dynamic_score(waveform="square")
    actions = synth_trace.prepare(
        document.body, [onset(gain=1), change(120, 1, "bend")], seed=0
    ).actions
    actual, _ = render(document, actions, backend)
    frames = np.arange(48000)
    position = (
        np.minimum(frames, 120) * 100 + np.maximum(frames - 120, 0) * 200
    ) % 48000
    expected = np.zeros_like(actual)
    expected[:, 0] = np.where(position < 24000, 1, -1)
    check_audio(tmp_path / "square-boundary.wav", actual, expected)


@pytest.mark.parametrize("scope", ["part", "instrument"])
def test_controls_continue_while_silent_and_are_shared_at_voice_start(
    scope: str, tmp_path: Path, backend: Literal["numpy", "native"]
) -> None:
    document = dynamic_score(scope=scope, waveform="square")
    events = [
        change(0, 1, scope=scope),
        onset(120, "a", gain=0, pitch=0.1),
        onset(180, "b", gain=0, pitch=0.1),
        onset(180, "c", "other", pitch=0.1).model_copy(update={"ordinal": 1}),
    ]
    actions = synth_trace.prepare(document.body, events, seed=0).actions
    actual, _ = render(document, actions, backend)
    frames = np.arange(48000)
    trajectory = 0.5 + 0.5 * np.minimum(frames / 240, 1)
    expected = np.zeros_like(actual)
    expected[:, 0] = (frames >= 120) * trajectory + (frames >= 180) * trajectory
    expected[:, 0] += (frames >= 180) * (trajectory if scope == "instrument" else 0.5)
    check_audio(tmp_path / "shared-context.wav", actual, expected)


def test_reused_trigger_does_not_reassign_an_old_release_tail(
    tmp_path: Path, backend: Literal["numpy", "native"]
) -> None:
    document = dynamic_score(waveform="square")
    raw = document.model_dump(mode="json")
    raw["body"]["voices"][0]["envelope"]["release"] = [{"duration": "1", "target": 0}]
    document = SynthInstrumentScore.model_validate(raw)
    events = [
        onset(pitch=0.1, gain=0.2),
        change(60, 1),
        Release(tick=120, ordinal=0, part="main", trigger_id="note"),
        onset(240, pitch=0.1, gain=0.8),
        change(240, 0.4),
    ]
    actual, snapshot = render(
        document,
        synth_trace.prepare(document.body, events, seed=0).actions,
        backend,
        block=128,
    )
    frames = np.arange(48000)
    old_gain = 0.2 + 0.8 * np.clip((frames - 60) / 240, 0, 1)
    old_envelope = np.minimum(1, 1 - (frames - 120) / 48000)
    new_gain = 0.8 - 0.4 * np.clip((frames - 240) / 240, 0, 1)
    expected = np.zeros_like(actual)
    expected[:, 0] = old_gain * old_envelope + (frames >= 240) * new_gain
    check_audio(tmp_path / "reused-trigger.wav", actual, expected)
    assert len(snapshot.voices) == 2
    assert snapshot.voices[0].sources != snapshot.voices[1].sources


@pytest.fixture(scope="module", params=["sine", "square", "triangle"])
def partition_case(
    request: pytest.FixtureRequest,
    backend: Literal["numpy", "native"],
) -> tuple[SynthInstrumentScore, list[TraceAction], np.ndarray, SynthSnapshot]:
    document = dynamic_score(smoothing="1/32000", waveform=request.param)
    raw = document.model_dump(mode="json")
    voice = raw["body"]["voices"][0]
    voice["bindings"][1]["smoothing"] = "1/200"
    voice["minimum_hold_seconds"] = "1/2"
    voice["envelope"]["release"] = [{"duration": "1/4", "target": 0.25}]
    document = SynthInstrumentScore.model_validate(raw)
    events = [
        onset(pitch=100.25),
        change(119, 0.9),
        change(120, 0.1),
        change(120, 0.7, "bend", 2),
        change(241, 0.8),
        change(270, 0.2, "bend"),
        Release(tick=18000, ordinal=0, part="main", trigger_id="note"),
        change(25000, 0.5, "bend"),
        change(26000, 0.3),
    ]
    actions = synth_trace.prepare(document.body, events, seed=0).actions
    actual, snapshot = render(document, actions, backend)
    return document, actions, actual, snapshot


@pytest.mark.parametrize("block", [64, 128, 256, 1024, 997])
def test_dynamic_render_is_partition_invariant_and_restorable(
    partition_case: tuple[
        SynthInstrumentScore, list[TraceAction], np.ndarray, SynthSnapshot
    ],
    block: int,
    tmp_path: Path,
    backend: Literal["numpy", "native"],
) -> None:
    document, actions, expected, snapshot = partition_case
    actual, state = render(document, actions, backend, block)
    check_audio(tmp_path / "partition.wav", actual, expected)
    assert state == snapshot
    assert np.all(actual[36000:] == 0)


def test_bindings_with_different_smoothing_keep_separate_trajectories(
    tmp_path: Path,
    backend: Literal["numpy", "native"],
) -> None:
    raw = dynamic_score(scope="part", waveform="square").model_dump(mode="json")
    voice = raw["body"]["voices"][0]
    voice["modulation"]["sources"].append(
        {"name": "fast", "scope": "part", "minimum": 0, "maximum": 1}
    )
    route = dict(voice["modulation"]["routes"][0], name="fast", source="fast")
    voice["modulation"]["routes"].append(route)
    voice["bindings"].append(
        {"name": "fast", "kind": "control", "control": "gain", "smoothing": "1/400"}
    )
    document = SynthInstrumentScore.model_validate(raw)
    actions = synth_trace.prepare(
        document.body, [change(0, 1, scope="part"), onset(60, pitch=0.1)], seed=0
    ).actions
    actual, _ = render(document, actions, backend)
    frames = np.arange(48000)
    expected = np.zeros_like(actual)
    expected[:, 0] = (
        (frames >= 60)
        * (0.5 + 0.5 * np.minimum(frames / 240, 1))
        * (0.5 + 0.5 * np.minimum(frames / 120, 1))
    )
    check_audio(tmp_path / "two-smoothing-times.wav", actual, expected)


def test_equal_frame_control_order_and_stop_are_exact(
    tmp_path: Path, backend: Literal["numpy", "native"]
) -> None:
    document = dynamic_score(smoothing="0", waveform="square")
    events = [
        change(0, 0),
        onset(0, gain=0.75).model_copy(update={"ordinal": 2}),
        change(0, 0.25, ordinal=3),
        change(0, 0.5, ordinal=4),
    ]
    actions = synth_trace.prepare(document.body, events, seed=0).actions
    actions.append(
        VoiceRetirement(
            tick=120,
            ordinal=0,
            voice_id="voice-0",
            cause="transport_stop",
            action="stop",
        )
    )
    actual, snapshot = render(document, actions, backend)
    expected = np.zeros_like(actual)
    expected[:120, 0] = 0.5
    check_audio(tmp_path / "same-frame-stop.wav", actual, expected)
    assert snapshot.voices == []


def test_fractional_release_boundary_and_nonzero_terminal_level(
    tmp_path: Path, backend: Literal["numpy", "native"]
) -> None:
    document = dynamic_score(waveform="square")
    raw = document.model_dump(mode="json")
    raw["body"]["voices"][0]["minimum_hold_seconds"] = "1/32000"
    raw["body"]["voices"][0]["envelope"] = Envelope(
        segments=[Segment(duration=Fraction(0), target=1)],
        release=[Segment(duration=Fraction(1, 32000), target=0.5)],
    ).model_dump(mode="json")
    document = SynthInstrumentScore.model_validate(raw)
    events = [
        onset(pitch=0.1, gain=1),
        Release(tick=0, ordinal=1, part="main", trigger_id="note"),
    ]
    actual, state = render(
        document, synth_trace.prepare(document.body, events, seed=0).actions, backend
    )
    expected = np.zeros_like(actual)
    expected[:3, 0] = [1, 1, 5 / 6]
    check_audio(tmp_path / "fractional-release.wav", actual, expected)
    assert state.voices == []


def test_restore_rejects_a_different_definition(
    backend: Literal["numpy", "native"],
) -> None:
    first = OfflineSynth(prepare(dynamic_score()), backend)
    second = OfflineSynth(prepare(dynamic_score(smoothing="0")), backend)
    with pytest.raises(EngineError, match="different prepared synth"):
        second.restore(first.snapshot())


def test_static_tuning_and_hz_offset_are_applied_once(
    tmp_path: Path, backend: Literal["numpy", "native"]
) -> None:
    raw = dynamic_score().model_dump(mode="json")
    voice = raw["body"]["voices"][0]
    voice["frequency_offset_hz"] = 5
    voice["processing"]["tuning_cents"] = 600
    voice["modulation"]["parameters"][1].update(default=600, maximum=1800)
    document = SynthInstrumentScore.model_validate(raw)
    event = onset(pitch=440, gain=1).model_copy(
        update={"controls": {"gain": 1, "bend": 0.5}}
    )
    actions = synth_trace.prepare(document.body, [event], seed=0).actions
    actual, state = render(document, actions, backend)
    expected = np.zeros_like(actual)
    expected[:, 0] = np.sin(2 * np.pi * np.arange(48000) * 890 / 48000)
    check_audio(tmp_path / "pitch-composition.wav", actual, expected)
    assert state.voices[0].frequency_hz == 445


@pytest.mark.parametrize("feature", ["pan", "generator", "curve", "fade"])
def test_preparation_rejects_unsupported_features(feature: str) -> None:
    raw = dynamic_score().model_dump(mode="json")
    voice = raw["body"]["voices"][0]
    if feature == "pan":
        voice["processing"]["pan"] = 0.5
    elif feature == "generator":
        voice["lfos"] = {"vibrato": {"rate": "1", "clock": "beats"}}
    elif feature == "curve":
        voice["envelope"]["segments"][0]["curve"] = 5
    else:
        voice["choke_group"] = "self"
        voice["chokes"] = [{"group": "self", "mode": "fade", "fade_seconds": 0.1}]
    document = SynthInstrumentScore.model_validate(raw)
    with pytest.raises(EngineError):
        prepare(document)


def test_unknown_action_is_not_silently_ignored(
    backend: Literal["numpy", "native"],
) -> None:
    with pytest.raises(EngineError, match="Unsupported synth action"):
        OfflineSynth(prepare(dynamic_score()), backend).advance(
            [TraceAction(tick=0, ordinal=0)], 0, 1
        )
