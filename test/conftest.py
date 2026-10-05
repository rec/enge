from typing import Literal

import pytest


def pytest_addoption(parser: pytest.Parser) -> None:
    parser.addoption(
        "--require-rubberband",
        action="store_true",
        help="Fail Rubber Band tests if the optional native feature was not built.",
    )


@pytest.fixture(scope="session", params=["numpy", "native"])
def backend(request: pytest.FixtureRequest) -> Literal["numpy", "native"]:
    return request.param
