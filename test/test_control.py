from fractions import Fraction
from pathlib import Path
from typing import Literal
from unittest.mock import Mock

import numpy as np
import pytest
from test_synth import check_audio
from ufor import lfo, modulation
from ufor.samples import controls

from enge import control
from enge.lfo import lfo_samples


def test_control_ramps_match_scalar_queries_and_interrupted_events() -> None:
    declaration = controls.ControlDeclaration()
    state = controls.initial_control(
        declaration, Fraction(2**53 + 37, 48000), Fraction(4801, 96000)
    )
    state = controls.control_event(
        declaration, state, controls.ControlValueEvent(at=state.at, ordinal=0, value=1)
    )
    origin = 2**53 + 37
    for start, frames in [(origin, 1301), (origin + 1301, 3000)]:
        expected = [
            controls.control_at(state, Fraction(start + i, 48000))
            for i in range(frames)
        ]
        np.testing.assert_allclose(
            control.control_samples(state, start, frames, 48000),
            expected,
            atol=1e-15,
            rtol=1e-14,
        )
        state = controls.control_event(
            declaration,
            state,
            controls.ControlValueEvent(
                at=Fraction(start + frames, 48000), ordinal=0, value=0.3
            ),
        )
    # Same-frame interruption captures the current ramp value, not its target.
    state = controls.control_event(
        declaration,
        state,
        controls.ControlValueEvent(at=state.at, ordinal=1, value=0.8),
    )
    np.testing.assert_allclose(
        control.control_samples(state, origin + 4301, 4800, 48000),
        [
            controls.control_at(state, Fraction(origin + 4301 + i, 48000))
            for i in range(4800)
        ],
        atol=1e-15,
    )


@pytest.mark.parametrize("interpolation", ["linear", "step"])
def test_array_modulation_matches_scalar_routes_and_weights(interpolation: str) -> None:
    target = {"name": "processing", "parameter": "amplitude"}
    definition = modulation.Modulation.model_validate(
        {
            "sources": [
                {"name": "a", "scope": "voice", "minimum": -1, "maximum": 1},
                {"name": "b", "scope": "voice", "minimum": -1, "maximum": 1},
            ],
            "parameters": [
                {
                    "target": target,
                    "unit": "ratio",
                    "scope": "voice",
                    "minimum": -10,
                    "maximum": 10,
                    "default": 1,
                }
            ],
            "routes": [
                {
                    "name": "offset",
                    "source": "a",
                    "target": target,
                    "operation": "add",
                    "unit": "ratio",
                    "interpolation": interpolation,
                    "points": [
                        {"input": -0.5, "amount": -1},
                        {"input": 0, "amount": 0.25},
                        {"input": 0.5, "amount": 1},
                    ],
                },
                {
                    "name": "scale",
                    "source": "b",
                    "target": target,
                    "operation": "multiply",
                    "unit": "ratio",
                    "points": [{"input": -1, "amount": 0}, {"input": 1, "amount": 2}],
                },
            ],
        }
    )
    signals = {
        "a": np.column_stack((np.linspace(-1, 1, 48001), np.linspace(0, 1, 48001))),
        "b": np.column_stack((np.linspace(1, -1, 48001), np.full(48001, 0.6))),
    }
    expected = [
        modulation.evaluate(
            definition,
            {
                n: modulation.SourceValue(value=s[i, 0], weight=s[i, 1])
                for n, s in signals.items()
            },
        )[0].value
        for i in range(48001)
    ]
    actual = control.modulation_samples(definition, signals, 48001)[
        "processing", "amplitude"
    ]
    np.testing.assert_allclose(actual, expected, atol=1e-14, rtol=1e-14)
    reversed_routes = definition.model_copy(
        update={"routes": list(reversed(definition.routes))}
    )
    np.testing.assert_array_equal(
        control.modulation_samples(reversed_routes, signals, 48001)[
            "processing", "amplitude"
        ],
        actual,
    )
    bad = {n: s.copy() for n, s in signals.items()}
    bad["a"][3, 0] = 1.01
    with pytest.raises(ValueError, match="source a"):
        control.modulation_samples(definition, bad, 48001)
    bad["a"][3, 0] = 0
    bad["a"][3, 1] = -1
    with pytest.raises(ValueError, match="weight"):
        control.modulation_samples(definition, bad, 48001)
    with pytest.raises(ValueError, match="missing"):
        control.modulation_samples(definition, {"a": signals["a"]}, 48001)


def test_additive_routes_retain_small_terms_under_cancellation() -> None:
    target = {"name": "processing", "parameter": "amplitude"}
    definition = modulation.Modulation.model_validate(
        {
            "sources": [{"name": "a", "scope": "voice", "minimum": 0, "maximum": 1}],
            "parameters": [
                {
                    "target": target,
                    "unit": "ratio",
                    "scope": "voice",
                    "minimum": 0,
                    "maximum": 2,
                    "default": 1,
                }
            ],
            "routes": [
                {
                    "name": f"r{i}",
                    "source": "a",
                    "target": target,
                    "operation": "add",
                    "unit": "ratio",
                    "points": [{"input": 0, "amount": v}],
                }
                for i, v in enumerate([1e16, 1, -1e16])
            ],
        }
    )
    actual = control.modulation_samples(definition, {"a": np.ones((10, 2))}, 10)
    np.testing.assert_array_equal(actual["processing", "amplitude"], np.full(10, 2))
    narrow = definition.model_copy(
        update={
            "parameters": [definition.parameters[0].model_copy(update={"maximum": 1})]
        }
    )
    with pytest.raises(ValueError, match="modulated"):
        control.modulation_samples(narrow, {"a": np.ones((10, 2))}, 10)


@pytest.mark.parametrize("interval", [1, 64, 1024])
def test_lfo_interval_matches_event_anchored_interpolation(
    tmp_path: Path, backend: Literal["numpy", "native"], interval: int
) -> None:
    definition = lfo.LFO(
        rate=5, phase=Fraction(1, 7), delay=Fraction(3, 96000), fade_in=Fraction(1, 10)
    )
    state = lfo.initial_lfo(definition, Fraction(37, 48000))
    actual = lfo_samples(definition, state, 37, 48000, 48000, backend, interval)
    knots = np.arange(0, 48000 + interval, interval)
    values = np.sin(2 * np.pi * (1 / 7 + knots * 5 / 48000))
    expected = np.column_stack(
        (
            np.interp(np.arange(48000), knots, values),
            np.clip((np.arange(48000) - 1.5) / 4800, 0, 1),
        )
    )
    check_audio(tmp_path / "lfo-control-interval.wav", actual, expected)
    split = np.concatenate(
        [
            lfo_samples(
                definition,
                lfo.LFOState.model_validate_json(state.model_dump_json()),
                37 + i,
                min(997, 48000 - i),
                48000,
                backend,
                interval,
            )
            for i in range(0, 48000, 997)
        ]
    )
    np.testing.assert_allclose(split, actual, atol=1e-10, rtol=1e-9)
    exact = np.sin(2 * np.pi * (1 / 7 + np.arange(48000) * 5 / 48000))
    bound = (2 * np.pi * 5 * interval / 48000) ** 2 / 8
    assert np.max(np.abs(actual[:, 0] - exact)) <= bound + 1e-12


@pytest.mark.parametrize(
    "waveform,rate", [("square", 5), ("triangle", 5), ("sine", 1000)]
)
def test_control_interval_preserves_edges_and_fast_lfos(
    tmp_path: Path, backend: Literal["numpy", "native"], waveform: str, rate: int
) -> None:
    definition = lfo.LFO(waveform=waveform, rate=rate, duty_cycle=Fraction(1, 3))
    state = lfo.initial_lfo(definition, Fraction(0))
    expected = lfo_samples(definition, state, 0, 48000, 48000, backend)
    actual = lfo_samples(definition, state, 0, 48000, 48000, backend, 1024)
    check_audio(tmp_path / "lfo-control-edges.wav", actual, expected)


def test_block_evaluation_does_not_call_scalar_models(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from test_noise import score, trigger
    from ufor import synth_trace

    from enge import noise

    document = score()
    actions = synth_trace.prepare(document.body, [trigger()], seed=0).actions

    def forbidden(*args: object, **kwargs: object) -> None:
        raise AssertionError("Called scalar control evaluator")

    monkeypatch.setattr(controls, "control_at", forbidden)
    monkeypatch.setattr(modulation, "evaluate", forbidden)
    monkeypatch.setattr(lfo, "lfo_at", forbidden)
    engine = noise.OfflineNoise(noise.prepare(document), "native")
    engine.advance(actions, 0, 48000)
    definition = lfo.LFO(rate=3)
    lfo_samples(definition, lfo.initial_lfo(definition, Fraction(0)), 0, 48000, 48000)


@pytest.mark.parametrize("kind", ["synth", "sampler", "fm", "noise"])
def test_engine_control_intervals_survive_events_partitions_and_restore(
    tmp_path: Path, backend: Literal["numpy", "native"], kind: str
) -> None:
    import test_fm
    import test_noise
    from test_lfo_instrument import lfo_score, lfo_settings
    from ufor import synth_trace
    from ufor.events import Release, Trigger
    from ufor.samples import instrument, trace
    from ufor.synth import SynthInstrumentScore

    from enge import fm, noise, sample_instrument, synth

    if kind in ("synth", "sampler"):
        document = lfo_score(kind)
    else:
        raw = (test_fm.score() if kind == "fm" else test_noise.score()).model_dump()
        settings = lfo_settings().model_dump()
        raw["body"]["voices"][0].update(
            {n: settings[n] for n in ("lfos", "modulation", "bindings")}
        )
        document = SynthInstrumentScore.model_validate(raw)
    events = [
        Trigger(
            tick=37, ordinal=0, part="main", trigger_id="note", key=60, pitch_hz=220
        ),
        Release(tick=35003, ordinal=0, part="main", trigger_id="note"),
    ]
    if isinstance(document, instrument.SampleInstrumentScore):
        prepared = sample_instrument.prepare(document, {"asset": np.ones((96000, 2))})
        actions = trace.prepare(document.body, events, seed=0).actions
        engine = sample_instrument.OfflineSampler(
            prepared, backend, control_interval=64
        )
    else:
        actions = synth_trace.prepare(document.body, events, seed=0).actions
        if kind == "synth":
            prepared = synth.prepare(document)
            engine = synth.OfflineSynth(prepared, backend, control_interval=64)
        elif kind == "fm":
            prepared = fm.prepare(document)
            engine = fm.OfflineFM(prepared, backend, control_interval=64)
        else:
            prepared = noise.prepare(document)
            engine = noise.OfflineNoise(prepared, backend, control_interval=64)
    saved = engine.snapshot()
    other = type(engine)(prepared, backend, control_interval=1024)
    with pytest.raises(synth.EngineError, match="control interval"):
        other.restore(saved)
    for invalid in (0, -1, True, 1.5):
        with pytest.raises(synth.EngineError, match="positive integer"):
            type(engine)(prepared, backend, control_interval=invalid)
    expected = engine.advance(actions, 0, 48000)
    final = engine.snapshot()
    engine.restore(saved)
    actual = np.empty_like(expected)
    for start in range(0, 48000, 997):
        end = min(start + 997, 48000)
        actual[start:end] = engine.advance(
            [a for a in actions if start <= a.tick < end], start, end
        )
        snapshot = type(saved).model_validate_json(engine.snapshot().model_dump_json())
        engine = type(engine)(prepared, backend, control_interval=64)
        engine.restore(snapshot)
    check_audio(tmp_path / "instrument-control-interval.wav", actual, expected)
    assert engine.snapshot() == final


def test_control_cache_preserves_ownership_and_observes_same_frame_changes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from test_noise import score
    from ufor.instrument_trace import ControlObservation

    from enge.synth import ControlRenderer

    raw = score().model_dump()
    raw["body"]["voices"][0]["modulation"]["sources"][0]["scope"] = "instrument"
    document = type(score()).model_validate(raw)
    settings = document.body.voices[0]
    renderer = ControlRenderer(48000, document.body.controls, [settings])
    # An instrument context can be evaluated directly using its source index.
    sources = {"color": 0}
    evaluator = Mock(wraps=control.modulation_samples)
    monkeypatch.setattr(control, "modulation_samples", evaluator)
    first = renderer.values(settings, sources, 0, 48000)
    first["filter-tone", "cutoff_hz"][:] = 0
    second = renderer.values(settings, sources, 0, 48000)
    assert evaluator.call_count == 1
    assert np.all(second["filter-tone", "cutoff_hz"] == 1200)
    renderer.apply(
        ControlObservation(
            tick=0, ordinal=0, control="color", value=1, scope="instrument"
        )
    )
    raised = renderer.values(settings, sources, 0, 48000)
    assert raised["filter-tone", "cutoff_hz"][-1] == 8000
    renderer.apply(
        ControlObservation(
            tick=0, ordinal=1, control="color", value=0.5, scope="instrument"
        )
    )
    interrupted = renderer.values(settings, sources, 0, 48000)
    assert interrupted["filter-tone", "cutoff_hz"][-1] == 4600
    assert evaluator.call_count == 3


def test_interpolated_lfo_rate_and_reset_events_do_not_anticipate_changes(
    tmp_path: Path, backend: Literal["numpy", "native"]
) -> None:
    definition = lfo.LFO(rate=3, phase=Fraction(1, 7))
    state = lfo.initial_lfo(definition, Fraction(0))
    actual = []
    expected = []
    for start, end, event in [
        (0, 10003, None),
        (10003, 23011, "rate"),
        (23011, 48000, "reset"),
    ]:
        if event is not None:
            state = lfo.lfo_event(
                definition,
                state,
                lfo.LFOEvent(
                    at=Fraction(start, 48000),
                    ordinal=0,
                    action=event,
                    rate=Fraction(5) if event == "rate" else None,
                ),
            )
        actual.append(
            lfo_samples(definition, state, start, end - start, 48000, backend, 64)
        )
        knots = list(range(start, end + 64, 64))
        values = [
            lfo.lfo_at(definition, state, Fraction(i, 48000)).value for i in knots
        ]
        expected.append(
            np.column_stack(
                (np.interp(np.arange(start, end), knots, values), np.ones(end - start))
            )
        )
    check_audio(
        tmp_path / "lfo-interval-events.wav",
        np.concatenate(actual),
        np.concatenate(expected),
    )


def test_step_route_changes_on_the_same_sample_across_partitions() -> None:
    target = {"name": "processing", "parameter": "amplitude"}
    definition = modulation.Modulation.model_validate(
        {
            "sources": [{"name": "a", "scope": "voice", "minimum": 0, "maximum": 1}],
            "parameters": [
                {
                    "target": target,
                    "unit": "ratio",
                    "scope": "voice",
                    "minimum": 0,
                    "maximum": 1,
                    "default": 0,
                }
            ],
            "routes": [
                {
                    "name": "step",
                    "source": "a",
                    "target": target,
                    "operation": "add",
                    "unit": "ratio",
                    "interpolation": "step",
                    "points": [{"input": 0, "amount": 0}, {"input": 0.5, "amount": 1}],
                }
            ],
        }
    )
    origin = 2**53 + 37
    state = controls.ControlState(
        at=Fraction(origin, 48000), value=0, target=1, smoothing=Fraction(1, 10)
    )
    pieces = []
    for i in range(0, 48000, 997):
        count = min(997, 48000 - i)
        signal = control.control_samples(state, origin + i, count, 48000)
        pieces.append(
            control.modulation_samples(
                definition, {"a": np.column_stack((signal, np.ones(count)))}, count
            )["processing", "amplitude"]
        )
    np.testing.assert_array_equal(
        np.concatenate(pieces), (np.arange(48000) >= 2400).astype(float)
    )
