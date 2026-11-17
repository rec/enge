"""Canonical Stages definitions crossing the native preparation boundary."""

import json

import numpy as np
import pytest

from enge import _native


@pytest.mark.parametrize(
    "child",
    [
        {"kind": "envelope", "initial": "current", "segments": [{"duration": None}]},
        {
            "kind": "envelope",
            "playback": "loop",
            "segments": [{"duration": "1 s", "to": True}],
            "repeat_count": 2**63,
        },
        {
            "kind": "envelope",
            "segments": [{"duration": "9223372036854775807 s", "to": 1}],
        },
        {
            "kind": "envelope",
            "clock": "seconds",
            "segments": [{"duration": "1 s", "to": 1}],
        },
        {"kind": "lfo", "rate": {"parameter": "speed"}},
        {"kind": "lfo", "rate": 1, "phase": True},
    ],
)
def test_stages_validate_children_and_exact_frame_conversion(
    child: dict[str, object],
) -> None:
    with pytest.raises(ValueError):
        generator_bank().set_staged_envelfos([(stages_document(child), 0, 1)], [])


@pytest.mark.parametrize("parameter,operation", [(1, 1), (0, 2)])
def test_stages_validate_route_indices(parameter: int, operation: int) -> None:
    with pytest.raises(ValueError):
        generator_bank().set_staged_envelfos(
            [(stages_document({"kind": "lfo", "rate": 1}), parameter, operation)], []
        )


def test_stages_reject_unknown_transition_targets() -> None:
    definition = json.loads(stages_document({"kind": "lfo", "rate": 1}))
    definition["transitions"] = [
        {
            "from": ["active"],
            "event": "note_off",
            "action": {"kind": "enter", "stage": "missing"},
        }
    ]
    with pytest.raises(ValueError):
        generator_bank().set_staged_envelfos([(json.dumps(definition), 0, 1)], [])


def test_stages_budget_includes_connections_and_supports_larger_graphs() -> None:
    bank = generator_bank()
    child = {"kind": "envelope", "segments": [{"duration": "1 s", "to": 1}]}
    definition = stages_document(child)
    connection = (0, "stage.done", 0, "next", 1, 0, 2**53, 0, (0, 1))
    # Root, route slot, Stages body, stage entry, child, segment and connection.
    bank.set_staged_envelfos([(definition, 0, 1)], [connection], max_records=7)
    with pytest.raises(ValueError):
        bank.set_staged_envelfos([(definition, 0, 1)], [connection], max_records=6)
    child["segments"] = [{"duration": "1 s", "to": 1}] * 10000
    large = stages_document(child)
    with pytest.raises(ValueError):
        bank.set_staged_envelfos([(large, 0, 1)], [])
    bank.set_staged_envelfos([(large, 0, 1)], [], max_records=10005)


def test_noninitial_current_value_can_hold_indefinitely() -> None:
    definition = json.loads(stages_document({"kind": "lfo", "rate": 1}))
    definition["stages"].append(
        {
            "name": "held",
            "envelfo": {
                "kind": "envelope",
                "initial": "current",
                "segments": [{"duration": None}],
            },
        }
    )
    definition["transitions"] = [
        {
            "from": ["active"],
            "event": "note_off",
            "action": {"kind": "enter", "stage": "held"},
        }
    ]
    generator_bank().set_staged_envelfos([(json.dumps(definition), 0, 1)], [])


def generator_bank() -> _native.GeneratorBank:
    return _native.GeneratorBank(48000, 1, 1, np.empty((0, 4)), [], [], [])


def stages_document(child: dict[str, object]) -> str:
    return json.dumps(
        {
            "kind": "stages",
            "initial_stage": "active",
            "stages": [{"name": "active", "envelfo": child}],
        }
    )
