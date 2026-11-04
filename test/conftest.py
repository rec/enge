from collections.abc import Callable, Iterator
from functools import cache
from typing import Literal

import numpy as np
import pytest
from test_dynamic_synth import onset
from test_lfo_instrument import lfo_score
from ufor import synth_trace
from ufor.synth import SynthInstrumentScore

from enge import _native, synth


def pytest_addoption(parser: pytest.Parser) -> None:
    parser.addoption(
        "--require-callback-no-allocations",
        action="store_true",
        help="Require allocation-audit and zero callback heap operations.",
    )
    parser.addoption(
        "--require-rubberband",
        action="store_true",
        help="Fail Rubber Band tests if the optional native feature was not built.",
    )


@pytest.fixture(autouse=True)
def callback_heap_audit(request: pytest.FixtureRequest) -> Iterator[None]:
    if not request.config.getoption("--require-callback-no-allocations"):
        yield
        return
    audit = getattr(_native, "take_callback_audit", None)
    if audit is None:
        pytest.fail("Rebuild enge with the allocation-audit Cargo feature")
    audit()
    yield
    successful, failed = audit()
    request.node.user_properties.append(
        ("native_callback_heap", {"successful": successful, "failed": failed})
    )
    assert list(successful[1:]) == [0, 0, 0], (
        "Successful callbacks used the Rust heap "
        f"(calls, allocs, reallocs, frees): {successful}"
    )
    assert list(failed[1:]) == [0, 0, 0], (
        "Failed callbacks used the Rust heap "
        f"(calls, allocs, reallocs, frees): {failed}"
    )


@pytest.fixture(scope="session", params=["numpy", "native"])
def backend(request: pytest.FixtureRequest) -> Literal["numpy", "native"]:
    return request.param


@pytest.fixture(scope="session")
def envelfo_baseline() -> Callable[[int], np.ndarray]:
    """Share read-only square-wave reference audio for Envelfo routing tests."""

    @cache
    def render(start_frame: int) -> np.ndarray:
        raw = lfo_score("synth").model_dump(mode="json")
        raw["body"]["voices"][0].update(envelfos={}, bindings=[], modulation={})
        document = SynthInstrumentScore.model_validate(raw)
        events = [onset(start_frame, pitch=0.1).model_copy(update={"controls": {}})]
        actions = synth_trace.prepare(document.body, events, seed=0).actions
        audio = synth.OfflineSynth(synth.prepare(document)).advance(actions, 0, 48000)
        audio.setflags(write=False)
        return audio

    return render
