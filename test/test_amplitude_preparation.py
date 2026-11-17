"""Voice-amplitude definitions at the preparation-only native boundary."""

import json

import numpy as np
import pytest
from ufor.envelfo import Envelope
from ufor.segments import Segment

from enge import _native


@pytest.mark.parametrize(
    "changes",
    [
        {"kind": "lfo", "rate": 1},
        {"scope": "instrument"},
        {"clock": "beats", "segments": [{"duration": "1 beat", "to": 1}]},
        {"initial": "current"},
        {"initial": True},
        {"start": "event"},
        {"hold": False},
        {"playback": "loop"},
        {"markers": [{"name": "middle", "position": "1/2"}]},
        {"release": []},
        {"segments": []},
        {"segments": [{"duration": "9223372036854775807 s", "to": 1}]},
        {"segments": [{"duration": "1/9223372036854775808 s", "to": 1}]},
        {
            "segments": [
                {
                    "kind": "exponential_approach",
                    "asymptote": 1,
                    "time_constant": 0.1,
                    "threshold": 0.9,
                    "transition_to": 1,
                }
            ]
        },
    ],
)
def test_amplitude_constructors_reject_invalid_or_unsupported_profiles(
    changes: dict[str, object],
) -> None:
    definition = Envelope(
        segments=[Segment(duration=0, to=1)],
        release=[Segment(duration="1/4 s", to=0)],
    ).model_dump(mode="json")
    with pytest.raises(ValueError):
        amplitude_runtime("synth", [json.dumps(definition | changes)])


@pytest.mark.parametrize("kind", ["fm", "fm_graph"])
def test_fm_validates_every_operator_amplitude(kind: str) -> None:
    definition = Envelope(
        segments=[Segment(duration=0, to=1)],
        release=[Segment(duration=0, to=0)],
    ).model_dump_json()
    with pytest.raises(ValueError):
        amplitude_runtime(kind, [definition, "{}"])


@pytest.mark.parametrize("kind", ["synth", "noise", "fm", "fm_graph", "sampler"])
def test_amplitude_budget_counts_all_operator_definitions_and_can_be_raised(
    kind: str,
) -> None:
    definition = Envelope(
        segments=[Segment(duration=0, to=1)],
        release=[Segment(duration=0, to=0)],
    ).model_dump_json()
    # Root plus a slot, definition, onset segment and release segment per operator.
    count = 9 if kind in ("fm", "fm_graph") else 5
    amplitude_runtime(kind, [definition], count)
    with pytest.raises(ValueError):
        amplitude_runtime(kind, [definition], count - 1)
    large = Envelope(
        segments=[Segment(duration=0, to=1)] * 5000,
        release=[Segment(duration=0, to=0)],
    ).model_dump_json()
    if kind in ("fm", "fm_graph"):
        with pytest.raises(ValueError):
            amplitude_runtime(kind, [large])
        amplitude_runtime(kind, [large], 10007)
    else:
        amplitude_runtime(kind, [large])


def amplitude_runtime(
    kind: str, documents: list[str], max_records: int = 10000
) -> _native.SynthRuntime | _native.LiveRuntime:
    routes = np.ones((1, 1))
    controls = np.empty((0, 6))
    generators = _native.GeneratorBank(48000, 1, 2, np.empty((0, 4)), [], [], [])
    filters = np.empty((0, 7))
    parameters = [1, 0, 1e300, 0, -120000, 120000]
    if kind == "sampler":
        runtime = _native.LiveRuntime.empty(48000, 1, 997, 64)
        runtime.add_sample(
            _native.SampleBuffer(np.ones((48000, 1))),
            (0, 48000, False, None),
            (0, 0, 0, 1, False, False, False, None, None, 0, None),
            48000,
            routes,
            documents[0],
            0,
            1,
            4,
            ([], np.empty((0, 1, 2)), np.empty((1, 0, 2))),
            False,
            max_records=max_records,
        )
        return runtime
    if kind == "fm":
        return _native.SynthRuntime.fm(
            48000,
            routes,
            documents[0],
            documents[-1],
            [0, 0],
            (0, 1),
            controls,
            [],
            generators,
            filters,
            parameters,
            1,
            max_records=max_records,
        )
    if kind == "fm_graph":
        return _native.SynthRuntime.fm_graph(
            48000,
            routes,
            [0, 0],
            [documents[0], documents[-1]],
            [0, 0],
            [(0, 1), (0, 1)],
            [],
            [0, 1],
            1,
            0,
            (0, 1),
            controls,
            [],
            generators,
            filters,
            parameters,
            1,
            max_records=max_records,
        )
    if kind == "noise":
        return _native.SynthRuntime.noise(
            48000,
            routes,
            documents[0],
            (0, 1),
            controls,
            [],
            generators,
            filters,
            parameters,
            1,
            max_records=max_records,
        )
    return _native.SynthRuntime(
        48000,
        0,
        0.5,
        routes,
        documents[0],
        (0, 1),
        controls,
        [],
        generators,
        filters,
        parameters,
        1,
        max_records=max_records,
    )
