from typing import Literal

import pytest


@pytest.fixture(scope="session", params=["numpy", "native"])
def backend(request: pytest.FixtureRequest) -> Literal["numpy", "native"]:
    return request.param
