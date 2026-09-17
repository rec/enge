import math
from pathlib import Path
from typing import Literal

import numpy as np
import pytest
from test_synth import check_audio
from test_synth_demo import publish_flac
from ufor import synth_trace
from ufor.events import ControlChange, Release, Trigger
from ufor.instrument_trace import VoiceRetirement
from ufor.synth import SynthInstrumentScore

from enge import fm, synth


def score() -> SynthInstrumentScore:
    return SynthInstrumentScore.model_validate_json(
        (Path(__file__).parents[1] / "conformance/fm-instrument.json").read_text()
    )


def trigger(frame: int = 0, name: str = "note", pitch: float = 220) -> Trigger:
    return Trigger(
        tick=frame, ordinal=0, part="main", trigger_id=name, key=57, pitch_hz=pitch
    )


def change(frame: int, value: float) -> ControlChange:
    return ControlChange(
        tick=frame,
        ordinal=1,
        control="color",
        value=value,
        scope="trigger",
        part="main",
        trigger_id="note",
    )


@pytest.mark.parametrize("index", [0, 2, 8])
def test_fm_matches_closed_form_and_named_operator_order(
    tmp_path: Path, index: float, backend: Literal["numpy", "native"]
) -> None:
    raw = score().model_dump()
    voice = raw["body"]["voices"][0]
    voice["modulation"] = {}
    voice["bindings"] = []
    voice["frequency_offset_hz"] = 10
    voice["processing"]["tuning_cents"] = 1200
    voice["fm"]["connection"]["index"] = index
    voice["fm"]["operators"][0]["phase_cycles"] = 0.125
    voice["fm"]["operators"][1]["phase_cycles"] = 0.25
    voice["fm"]["operators"].reverse()
    document = SynthInstrumentScore.model_validate(raw)
    actions = synth_trace.prepare(document.body, [trigger()], seed=0).actions
    actual = fm.OfflineFM(fm.prepare(document), backend).advance(actions, 0, 48000)
    t = np.arange(48000) / 48000
    modulator = np.sin(2 * np.pi * (920 * t + 0.125))
    carrier = 0.2 * np.sin(2 * np.pi * (460 * t + 0.25) + index * modulator)
    expected = carrier[:, None] * np.array([[1, 0.5]])
    check_audio(tmp_path / "fm-static.wav", actual, expected)


def test_fm_feedback_control_smoothing_and_release_match_independent_recurrence(
    tmp_path: Path,
    backend: Literal["numpy", "native"],
    pytestconfig: pytest.Config,
) -> None:
    raw = score().model_dump()
    raw["body"]["voices"][0]["fm"]["feedback"] = 0.4
    document = SynthInstrumentScore.model_validate(raw)
    events = [
        trigger(),
        change(12000, 1),
        change(13200, 0.25),
        Release(tick=36000, ordinal=0, part="main", trigger_id="note"),
    ]
    actions = synth_trace.prepare(document.body, events, seed=0).actions
    engine = fm.OfflineFM(fm.prepare(document), backend)
    actual = engine.advance(actions, 0, 48000)
    expected = np.zeros((48000, 2))
    previous = 0.0
    for i in range(40800):
        color = (
            0
            if i < 12000
            else (
                min((i - 12000) / 2400, 1)
                if i < 13200
                else 0.5 - 0.25 * min((i - 13200) / 2400, 1)
            )
        )
        em = max(0, 1 - max(0, i - 36000) / 2400)
        ec = max(0, 1 - max(0, i - 36000) / 4800)
        previous = em * math.sin(
            2 * math.pi * (i * 440 % 48000) / 48000 + 0.4 * previous
        )
        value = (
            0.2
            * ec
            * math.sin(
                2 * math.pi * (i * 220 % 48000) / 48000 + (2 + 4 * color) * previous
            )
        )
        expected[i] = value, value * 0.5
    check_audio(tmp_path / "fm-demo.wav", actual, expected)
    assert engine.snapshot().voices == []
    publish_flac(
        tmp_path / "fm-demo-actual.wav",
        pytestconfig.cache.mkdir("audio") / f"fm-demo-{backend}.flac",
    )


@pytest.mark.parametrize("block", [64, 128, 256, 997, 1024])
def test_fm_partitions_and_json_restore_preserve_feedback_and_release(
    tmp_path: Path,
    backend: Literal["numpy", "native"],
    block: int,
) -> None:
    raw = score().model_dump()
    raw["body"]["voices"][0]["fm"]["feedback"] = 0.7
    document = SynthInstrumentScore.model_validate(raw)
    events = [
        trigger(),
        change(12001, 0.75),
        trigger(17003, "second", 330),
        Release(tick=35003, ordinal=0, part="main", trigger_id="note"),
    ]
    actions = synth_trace.prepare(document.body, events, seed=0).actions
    definition = fm.prepare(document)
    whole = fm.OfflineFM(definition, backend)
    expected = whole.advance(actions, 0, 48000)
    partitioned = fm.OfflineFM(definition, backend)
    actual = np.empty_like(expected)
    for start in range(0, 48000, block):
        end = min(48000, start + block)
        actual[start:end] = partitioned.advance(
            [a for a in actions if start <= a.tick < end], start, end
        )
        snapshot = fm.FMSnapshot.model_validate_json(
            partitioned.snapshot().model_dump_json()
        )
        partitioned = fm.OfflineFM(definition, backend)
        partitioned.restore(snapshot)
    check_audio(tmp_path / "fm-partitions.wav", actual, expected)
    assert partitioned.snapshot() == whole.snapshot()


def test_fm_minimum_hold_and_duplicate_release_preserve_carrier_lifetime(
    tmp_path: Path,
    backend: Literal["numpy", "native"],
) -> None:
    raw = score().model_dump()
    voice = raw["body"]["voices"][0]
    voice["minimum_hold_seconds"] = "1/2"
    # Modulator ends immediately even with a nonzero terminal envelope value.
    voice["fm"]["operators"][0]["envelope"]["release"] = [{"duration": 0, "target": 1}]
    document = SynthInstrumentScore.model_validate(raw)
    actions = synth_trace.prepare(document.body, [trigger()], seed=0).actions
    actions += [
        VoiceRetirement(
            tick=i,
            ordinal=0,
            voice_id="voice-0",
            cause="physical_release",
            action="release",
        )
        for i in (100, 101)
    ]
    engine = fm.OfflineFM(fm.prepare(document), backend)
    actual = engine.advance(actions, 0, 48000)
    t = np.arange(48000)
    modulator = np.where(t < 24000, np.sin(2 * np.pi * t * 440 / 48000), 0)
    envelope = np.clip(1 - (t - 24000) / 4800, 0, 1)
    carrier = 0.2 * envelope * np.sin(2 * np.pi * t * 220 / 48000 + 2 * modulator)
    check_audio(tmp_path / "fm-minimum-hold.wav", actual, carrier[:, None] * [[1, 0.5]])
    assert engine.snapshot().voices == []


def test_fm_rejects_wrong_source_profile_and_snapshot() -> None:
    document = score()
    with pytest.raises(synth.EngineError, match="oscillator voice"):
        synth.prepare(document)
    prepared = fm.prepare(document)
    snapshot = fm.OfflineFM(prepared).snapshot()
    raw = document.model_dump()
    raw["body"]["voices"][0]["fm"]["feedback"] = 1
    other = fm.OfflineFM(fm.prepare(SynthInstrumentScore.model_validate(raw)))
    with pytest.raises(synth.EngineError, match="different prepared"):
        other.restore(snapshot)


@pytest.mark.parametrize(
    "name,parameter,unit,default,amount",
    [
        ("operator-modulator", "ratio", "ratio", 2, 0.5),
        ("operator-carrier", "tuning_cents", "cents", 0, 1200),
        ("processing", "tuning_cents", "cents", 0, 1200),
        ("processing", "amplitude", "ratio", 1, 0.5),
        ("fm", "feedback", "radians", 0, 0.3),
        ("fm", "carrier_level", "ratio", 0.2, 0.2),
    ],
)
def test_fm_live_targets_preserve_phase_and_use_declared_units(
    tmp_path: Path,
    backend: Literal["numpy", "native"],
    name: str,
    parameter: str,
    unit: str,
    default: float,
    amount: float,
) -> None:
    raw = score().model_dump()
    voice = raw["body"]["voices"][0]
    target = {"name": name, "parameter": parameter}
    voice["modulation"]["parameters"] = [
        {
            "target": target,
            "unit": unit,
            "scope": "voice",
            "minimum": default,
            "maximum": default + amount,
            "default": default,
        }
    ]
    voice["modulation"]["routes"][0].update(
        target=target,
        unit=unit,
        points=[{"input": 0, "amount": 0}, {"input": 1, "amount": amount}],
    )
    voice["bindings"][0]["smoothing"] = 0
    document = SynthInstrumentScore.model_validate(raw)
    actions = synth_trace.prepare(
        document.body, [trigger(), change(12001, 1)], seed=0
    ).actions
    actual = fm.OfflineFM(fm.prepare(document), backend).advance(actions, 0, 48000)
    expected = np.empty_like(actual)
    pm = pc = previous = 0.0
    for i in range(48000):
        value = default + (amount if i >= 12001 else 0)
        ratio = value if parameter == "ratio" else 2
        multiplier = 2 ** (value / 1200) if parameter == "tuning_cents" else 1
        feedback = value if parameter == "feedback" else 0
        level = value if parameter == "carrier_level" else 0.2
        gain = value if parameter == "amplitude" else 1
        previous = math.sin(2 * math.pi * pm / 48000 + feedback * previous)
        sample = gain * level * math.sin(2 * math.pi * pc / 48000 + 2 * previous)
        expected[i] = sample, sample * 0.5
        pm = (pm + 220 * ratio * (multiplier if name == "processing" else 1)) % 48000
        pc = (pc + 220 * multiplier) % 48000
    check_audio(tmp_path / "fm-live-target.wav", actual, expected)


def test_fm_lfo_filter_and_immediate_stop_share_existing_processing(
    tmp_path: Path,
    backend: Literal["numpy", "native"],
) -> None:
    from test_filters import matrix_filter

    raw = score().model_dump()
    voice = raw["body"]["voices"][0]
    voice["lfos"] = {"color": {"waveform": "sine", "rate": 2}}
    voice["modulation"]["sources"][0].update(scope="voice", minimum=-1)
    voice["modulation"]["routes"][0]["points"] = [
        {"input": -1, "amount": -1},
        {"input": 1, "amount": 1},
    ]
    voice["bindings"] = [{"name": "color", "kind": "lfo", "reference": "color"}]
    voice["processing"]["filters"] = [
        {"name": "tone", "response": "lowpass", "cutoff_hz": 1800, "q": 0.7}
    ]
    document = SynthInstrumentScore.model_validate(raw)
    actions = synth_trace.prepare(document.body, [trigger()], seed=0).actions
    actions.append(
        VoiceRetirement(
            tick=36000,
            ordinal=0,
            voice_id="voice-0",
            cause="transport_stop",
            action="stop",
        )
    )
    engine = fm.OfflineFM(fm.prepare(document), backend)
    actual = engine.advance(actions, 0, 48000)
    t = np.arange(36000) / 48000
    index = 2 + np.sin(2 * np.pi * 2 * t)
    audio = 0.2 * np.sin(2 * np.pi * 220 * t + index * np.sin(2 * np.pi * 440 * t))
    definitions = document.body.voices[0].processing.filters
    values = np.tile([[[1800, 0.7]]], (36000, 1, 1))
    expected = np.zeros_like(actual)
    expected[:36000] = matrix_filter(definitions, audio[:, None], values) * [[1, 0.5]]
    check_audio(tmp_path / "fm-lfo-filter.wav", actual, expected)
    assert engine.snapshot().voices == []


def test_fm_snapshot_rejects_backend_mismatch() -> None:
    definition = fm.prepare(score())
    reference = fm.OfflineFM(definition)
    native = fm.OfflineFM(definition, "native")
    with pytest.raises(synth.EngineError, match="different FM backend"):
        native.restore(reference.snapshot())
    with pytest.raises(synth.EngineError, match="Unknown FM backend"):
        fm.OfflineFM(definition, "invalid")  # type: ignore[arg-type]


def test_native_fm_preserves_sound_and_state_without_python_dsp(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from enge import filters

    raw = score().model_dump()
    raw["body"]["voices"][0]["fm"]["feedback"] = 0.7
    raw["body"]["voices"][0]["processing"]["filters"] = [
        {"name": "tone", "response": "lowpass", "cutoff_hz": 1800, "q": 0.7}
    ]
    document = SynthInstrumentScore.model_validate(raw)
    actions = synth_trace.prepare(
        document.body, [trigger(pitch=223.17), change(12001, 0.75)], seed=0
    ).actions
    definition = fm.prepare(document)
    reference = fm.OfflineFM(definition)
    expected = reference.advance(actions, 0, 48000)

    def forbidden(*args: object, **kwargs: object) -> None:
        raise AssertionError("Native FM called Python DSP")

    monkeypatch.setattr(fm, "fm_samples", forbidden)
    monkeypatch.setattr(synth, "envelope_samples", forbidden)
    monkeypatch.setattr(synth, "route_samples", forbidden)
    monkeypatch.setattr(filters, "filter_samples", forbidden)
    native = fm.OfflineFM(definition, "native")
    actual = native.advance(actions, 0, 48000)
    check_audio(tmp_path / "fm-native-parity.wav", actual, expected)
    left = reference.snapshot().voices[0].renderer
    right = native.snapshot().voices[0].renderer
    np.testing.assert_allclose(right.phases, left.phases, atol=1e-10, rtol=1e-9)
    assert right.previous_modulator == pytest.approx(left.previous_modulator, abs=1e-10)
    np.testing.assert_allclose(
        right.filter_states[0].integrators,
        left.filter_states[0].integrators,
        atol=1e-10,
        rtol=1e-9,
    )
    with pytest.raises(synth.EngineError, match="different FM backend"):
        native.restore(reference.snapshot().model_copy(update={"backend": "native"}))


def test_native_fm_owns_strided_inputs(tmp_path: Path) -> None:
    from enge import _native

    parameters = np.tile([440, 220, 2, 0.4, 0.2], (96000, 1))[::2]
    phases = np.zeros((2, 4))[:, ::2]
    gains = np.ones(96000)[::2]
    routes = np.array([1, 0, 0.5, 0])[::2]
    spans = np.array([[0, 48000, 1, 0, 0, 1, 0]], dtype=float)
    memory = np.zeros((0, 0, 2))
    values = np.zeros((48000, 0, 2))
    copies = [x.copy() for x in (parameters, phases, gains, routes, spans)]
    audio, state, history, _ = _native.render_fm(
        48000,
        parameters,
        phases,
        0,
        gains,
        spans,
        spans,
        48000,
        routes,
        ([], memory, values),
    )
    expected, expected_state, expected_history = fm.fm_samples(
        parameters[:, :2],
        np.ones((48000, 2)),
        parameters[:, 2],
        parameters[:, 3],
        parameters[:, 4],
        phases,
        np.array([0.0]),
        48000,
    )
    check_audio(tmp_path / "fm-owned.wav", audio, expected * routes)
    np.testing.assert_allclose(state, expected_state, atol=1e-10)
    assert history == pytest.approx(expected_history[0], abs=1e-10)
    for original, copy in zip(
        (parameters, phases, gains, routes, spans), copies, strict=True
    ):
        np.testing.assert_array_equal(original, copy)
        assert not np.shares_memory(audio, original)
        assert not np.shares_memory(state, original)


@pytest.mark.parametrize(
    "case",
    ["rate", "shape", "frequency", "history", "phases", "gains", "routes", "release"],
)
def test_native_fm_rejects_invalid_inputs(case: str) -> None:
    from enge import _native

    parameters = np.array([[440, 220, 2, 0.4, 0.2]])
    phases = np.zeros((2, 2))
    gains = np.ones(1)
    routes = np.ones(1)
    spans = np.array([[0, 1, 1, 0, 0, 1, 0]], dtype=float)
    if case == "shape":
        parameters = parameters[:, :4]
    elif case == "frequency":
        parameters[0, 0] = 0
    elif case == "phases":
        phases[0, 0] = np.nan
    elif case == "gains":
        gains[0] = -1
    elif case == "routes":
        routes[0] = np.inf
    with pytest.raises(ValueError, match="Invalid FM"):
        _native.render_fm(
            0 if case == "rate" else 48000,
            parameters,
            phases,
            np.nan if case == "history" else 0,
            gains,
            spans,
            spans,
            2 if case == "release" else 1,
            routes,
            ([], np.zeros((0, 0, 2)), np.zeros((1, 0, 2))),
        )
