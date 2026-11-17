"""Native route layout, host metadata and bounded preparation."""

import json
from pathlib import Path

import numpy as np
import pytest
from test_live import native_oscillator_runtime
from test_synth import check_audio

from enge import _native


@pytest.mark.parametrize("scope", ["instrument", "part", "trigger"])
def test_routes_keep_sorted_slots_exact_smoothing_and_latched_events(
    scope: str,
) -> None:
    context = control_context(scope)
    for name, kind, maximum in [("a-key", "key", 127), ("b-velocity", "velocity", 1)]:
        context["modulation"]["sources"].append(
            {"name": name, "scope": "voice", "minimum": 0, "maximum": maximum}
        )
        context["bindings"].append({"kind": kind, "name": name})
        context["modulation"]["routes"].append(
            {**context["modulation"]["routes"][0], "name": name, "source": name}
        )
    prepared = _native.prepare_modulation(
        48000, json.dumps(context), [], TARGETS, np.empty((0, 7))
    )
    np.testing.assert_array_equal(
        prepared[0],
        [
            [0, 3, 0, 1, 0, 127],
            [0, 3, 0, 1, 0, 1],
            [0.25, ["instrument", "part", "trigger"].index(scope), 0, 1, 0, 1],
        ],
    )
    assert prepared[1] == [(0, 1), (0, 1), (48000, 7)]
    assert prepared[2] == [1, 0, 2]
    assert prepared[3] == {"gain": [2]}
    assert prepared[4] == ["voice", "voice", scope]
    assert prepared[5] == ["", "", "gain"]
    assert prepared[13] == {"voice", scope}
    assert prepared[19] == {0: "key", 1: "velocity"}
    assert prepared[22] == [("processing", "amplitude", 0)]


def test_route_budget_combines_declarations_and_host_targets() -> None:
    text = json.dumps(control_context("instrument"))
    _native.prepare_modulation(
        48000, text, [], TARGETS, np.empty((0, 7)), max_records=10
    )
    with pytest.raises(ValueError):
        _native.prepare_modulation(
            48000, text, [], TARGETS, np.empty((0, 7)), max_records=9
        )
    context = control_context("instrument")
    context["modulation"]["routes"][0]["points"] = [
        {"input": i / 10000, "amount": i / 10000} for i in range(10001)
    ]
    text = json.dumps(context)
    with pytest.raises(ValueError):
        _native.prepare_modulation(48000, text, [], TARGETS, np.empty((0, 7)))
    prepared = _native.prepare_modulation(
        48000, text, [], TARGETS, np.empty((0, 7)), max_records=10009
    )
    assert len(prepared[18][0][0][1]) == 10001


@pytest.mark.parametrize(
    "failure", ["target", "duplicate_index", "default", "smoothing"]
)
def test_route_preparation_rejects_invalid_host_facts_and_frame_overflow(
    failure: str,
) -> None:
    targets = TARGETS
    context = control_context("instrument")
    if failure == "target":
        targets = [("other", "amplitude", 0, 1, 0, 2)]
    elif failure == "duplicate_index":
        targets = TARGETS + [("other", "amplitude", 0, 1, 0, 2)]
    elif failure == "default":
        targets = [("processing", "amplitude", 0, float("nan"), 0, 2)]
    else:
        context["bindings"][0]["smoothing"] = str(2**63 - 1)
    with pytest.raises(ValueError):
        _native.prepare_modulation(
            48000, json.dumps(context), [], targets, np.empty((0, 7))
        )


def test_expanded_generator_slots_share_the_preparation_budget() -> None:
    context = control_context("voice")
    context["controls"] = {}
    context["bindings"] = [
        {"kind": "envelfo", "name": "z-control", "reference": "shape"}
    ]
    documents = [
        (
            "shape",
            json.dumps(
                {
                    "kind": "envelope",
                    "segments": [{"duration": "0 s", "to": 0} for _ in range(2000)],
                }
            ),
        )
    ]
    route = context["modulation"]["routes"][0]
    context["modulation"]["routes"] = [
        {**route, "name": f"route-{i}"} for i in range(5)
    ]
    with pytest.raises(ValueError):
        _native.prepare_modulation(
            48000, json.dumps(context), documents, TARGETS, np.empty((0, 7))
        )
    prepared = _native.prepare_modulation(
        48000,
        json.dumps(context),
        documents,
        TARGETS,
        np.empty((0, 7)),
        max_records=13000,
    )
    assert prepared[12] == {"shape": [0, 1, 2, 3, 4]}
    assert len(prepared[10]) == 5


def test_generator_source_maps_keep_route_order_instead_of_sorting_root_names() -> None:
    context = control_context("instrument")
    context["controls"] = {}
    context["modulation"]["sources"] = [
        {"name": n, "scope": "instrument", "minimum": -1, "maximum": 1}
        for n in ("a-first", "b-second", "c-shared")
    ]
    context["bindings"] = [
        {"kind": "envelfo", "name": n, "reference": r}
        for n, r in [
            ("a-first", "z-root"),
            ("b-second", "a-root"),
            ("c-shared", "z-root"),
        ]
    ]
    route = context["modulation"]["routes"][0]
    context["modulation"]["routes"] = [
        {**route, "name": n, "source": n} for n in ("c-shared", "b-second", "a-first")
    ]
    documents = [
        (n, json.dumps({"kind": "lfo", "scope": "instrument", "rate": "1"}))
        for n in ("a-root", "z-root")
    ]
    prepared = _native.prepare_modulation(
        48000, json.dumps(context), documents, TARGETS, np.empty((0, 7))
    )
    assert list(prepared[8]) == ["z-root", "a-root"]
    assert prepared[8] == {"z-root": [0, 2], "a-root": [1]}


@pytest.mark.parametrize("failure", ["mapping", "sum"])
def test_shared_arithmetic_failures_zero_the_callback_without_heap_operations(
    tmp_path: Path, failure: str
) -> None:
    count = 1 if failure == "mapping" else 2
    documents = [json.dumps({"kind": "lfo", "rate": "1"})] * count
    bank = _native.GeneratorBank(
        48000, 1, 2, np.array([[0, 0, 0, 1]] * count, dtype=float), documents, [], []
    )
    source = native_oscillator_runtime(bank)
    points = [(-1e308, 0.25), (1e308, 0.75)] if failure == "mapping" else [(0, 1e308)]
    source.set_route_maps(1, [(0, points)] * count)
    runtime = _native.LiveRuntime([source], 48000, 64, 4)
    runtime.submit(0, np.array([[0, 0, 0, 220, 0.2, 0]], dtype=float))
    actual = np.ones((48000, 2))
    with pytest.raises(ValueError, match="non-finite scalar arithmetic"):
        runtime.process_into(actual)
    assert runtime.failed()
    check_audio(tmp_path / "failed-math.wav", actual, np.zeros_like(actual))


def control_context(scope: str) -> dict:
    target = {"name": "processing", "parameter": "amplitude"}
    return {
        "modulation": {
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
            "sources": [
                {"name": "z-control", "scope": scope, "minimum": 0, "maximum": 1}
            ],
            "routes": [
                {
                    "name": "gain",
                    "source": "z-control",
                    "target": target,
                    "operation": "multiply",
                    "unit": "ratio",
                    "points": [{"input": 0, "amount": 0}, {"input": 1, "amount": 1}],
                }
            ],
        },
        "bindings": [
            {
                "kind": "control",
                "name": "z-control",
                "control": "gain",
                "smoothing": "1/7 s",
            }
        ],
        "controls": {"gain": {"default": 0.25}},
        "event_connections": [],
    }


TARGETS = [("processing", "amplitude", 0, 0.5, -10, 10)]
