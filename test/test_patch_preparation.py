"""Shared Patch/binding validation at the native preparation boundary."""

import json
from pathlib import Path

import numpy as np
import pytest
from test_live import native_oscillator_runtime
from test_synth import check_audio

from enge import _native


@pytest.mark.parametrize(
    "failure", ["unknown_field", "unknown_output", "reference", "missing_controls"]
)
def test_binding_preparation_rejects_incomplete_or_unresolved_declarations(
    failure: str,
) -> None:
    context = json.loads(binding_context())
    documents = patch_documents()
    if failure == "unknown_field":
        context["extra"] = True
    elif failure == "unknown_output":
        context["bindings"][0]["output"] = "missing"
    elif failure == "reference":
        documents[0] = (
            "gesture",
            json.dumps({"kind": "reference", "score": {"path": "future.toml"}}),
        )
    else:
        del context["controls"]
    with pytest.raises(ValueError):
        _native.prepare_modulation(
            48000,
            json.dumps(context),
            documents,
            [(n, p, i, 1, 0, 2) for n, p, i in TARGETS],
        )


def test_follower_requirements_cannot_become_an_executable_bank() -> None:
    context = json.dumps(
        {
            "modulation": {
                "sources": [
                    {"name": "level", "scope": "voice", "minimum": 0, "maximum": 1}
                ]
            },
            "bindings": [
                {"kind": "follower", "name": "level", "processor": "detector"}
            ],
            "controls": {},
            "event_connections": [],
        }
    )
    with pytest.raises(ValueError, match="Unresolved"):
        _native.prepare_modulation(48000, context, [], [])


def test_patch_preparation_retains_the_host_layout_and_rejects_invalid_orders() -> None:
    bank = generator_bank()
    nodes, mappings = bank.configure_patches(
        binding_context(), patch_documents(), SLOTS, TARGETS, ORDERS
    )
    assert nodes == [(("gesture", "first"), 0), (("gesture", "second"), 1)]
    assert mappings == [(0, [(-1.0, 0.25), (1.0, 0.75)])]
    with pytest.raises(ValueError, match="dependencies"):
        bank.configure_patches(
            binding_context(),
            patch_documents(),
            SLOTS,
            TARGETS,
            [("gesture", ["first", "clock", "second"])],
        )


def test_patch_budget_combines_declarations_and_host_metadata() -> None:
    bank = generator_bank()
    # Eight context records, nine generator records and six layout references.
    bank.configure_patches(
        binding_context(), patch_documents(), SLOTS, TARGETS, ORDERS, max_records=23
    )
    with pytest.raises(ValueError):
        bank.configure_patches(
            binding_context(), patch_documents(), SLOTS, TARGETS, ORDERS, max_records=22
        )
    documents = patch_documents()
    documents.append(
        (
            "spare",
            json.dumps(
                {"kind": "envelope", "segments": [{"duration": "0 s", "to": 0}] * 10000}
            ),
        )
    )
    with pytest.raises(ValueError):
        bank.configure_patches(binding_context(), documents, SLOTS, TARGETS, ORDERS)
    bank.configure_patches(
        binding_context(), documents, SLOTS, TARGETS, ORDERS, max_records=10024
    )


def test_patch_budget_override_reaches_embedded_modulation() -> None:
    context = json.loads(binding_context())
    context["modulation"]["routes"][0]["points"] = [
        {"input": i / 5000 - 1, "amount": 0.25 + i / 20000} for i in range(10001)
    ]
    context = json.dumps(context)
    bank = generator_bank()
    with pytest.raises(ValueError):
        bank.configure_patches(context, patch_documents(), SLOTS, TARGETS, ORDERS)
    bank.configure_patches(
        context, patch_documents(), SLOTS, TARGETS, ORDERS, max_records=10022
    )


def test_failed_patch_configuration_preserves_the_previous_bank(tmp_path: Path) -> None:
    bank = generator_bank(instrument=True)
    _, mappings = bank.configure_patches(
        binding_context(), patch_documents(), SLOTS, TARGETS, ORDERS
    )
    documents = patch_documents()
    parent = json.loads(documents[0][1])
    parent["envelfos"]["first"] = {
        "kind": "threshold",
        "input": "clock",
        "lower": -0.25,
        "upper": 0.25,
    }
    parent["events"] = [
        {"source": "clock.pulse", "target": "second", "action": "capture"}
    ]
    documents[0] = ("gesture", json.dumps(parent))
    # The declared marker source is voice-scoped; this bank's actual slot is an
    # instrument LFO. Native event-profile checks must reject it without mutation.
    with pytest.raises(ValueError, match="Patch event connections"):
        bank.configure_patches(binding_context(), documents, SLOTS, TARGETS, ORDERS)
    runtime = native_oscillator_runtime(bank)
    runtime.set_route_maps(1, [(0, [(-1.0, -1.0), (1.0, 1.0)])])
    runtime.set_route_maps(4, mappings)
    actual = runtime.process_actions(
        48000, 0, 1, 0, np.array([[0, 0, 0, 220, 0.2, 0]], dtype=float)
    )
    expected = np.sin(2 * np.pi * np.arange(48000) * 220 / 48000)[:, None] * [0.1, 0.05]
    check_audio(tmp_path / "unchanged-patch-bank.wav", actual, expected)


def binding_context() -> str:
    target = {"name": "processing", "parameter": "amplitude"}
    return json.dumps(
        {
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
                    {"name": "shape", "scope": "voice", "minimum": -1, "maximum": 1}
                ],
                "routes": [
                    {
                        "name": "gain",
                        "source": "shape",
                        "target": target,
                        "operation": "multiply",
                        "unit": "ratio",
                        "points": [
                            {"input": -1, "amount": 0.25},
                            {"input": 1, "amount": 0.75},
                        ],
                    }
                ],
            },
            "bindings": [
                {
                    "kind": "envelfo",
                    "name": "shape",
                    "reference": "gesture",
                    "output": "first",
                }
            ],
            "controls": {},
            "event_connections": [],
        }
    )


def patch_documents() -> list[tuple[str, str]]:
    clock = {
        "kind": "lfo",
        "rate": "1",
        "markers": [{"name": "pulse", "position": "1/4"}],
    }
    return [
        (
            "gesture",
            json.dumps(
                {
                    "kind": "patch",
                    "envelfos": {
                        "clock": clock,
                        "second": {"kind": "latch", "input": "clock"},
                        "first": {"kind": "latch", "input": "clock"},
                    },
                    "outputs": {"second": "second", "first": "first"},
                }
            ),
        ),
        ("patch-gesture-clock", json.dumps(clock)),
    ]


def generator_bank(instrument: bool = False) -> _native.GeneratorBank:
    clock = json.loads(patch_documents()[1][1])
    if instrument:
        clock.update(scope="instrument", markers=[])
    return _native.GeneratorBank(
        48000, 1, 2, np.array([[0, 0, 0, 0]], dtype=float), [json.dumps(clock)], [], []
    )


SLOTS = ([("patch-gesture-clock", 0)], [], [])
TARGETS = [("processing", "amplitude", 0)]
ORDERS = [("gesture", ["clock", "first", "second"])]
