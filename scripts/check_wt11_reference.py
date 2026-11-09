"""Check all legal WT11 pitch rates against the pinned DXconvert source."""

from ast import Assign, Name, literal_eval, parse
from hashlib import sha256
from pathlib import Path
from typing import cast

import tyro
from pydantic import BaseModel


class Options(BaseModel, frozen=True):
    reference: Path


def rate_table(path: Path, name: str) -> list[int]:
    for node in parse(path.read_text()).body:
        if isinstance(node, Assign) and any(
            isinstance(t, Name) and t.id == name for t in node.targets
        ):
            values: object = literal_eval(node.value)
            if not isinstance(values, (list, tuple)) or any(
                type(v) is not int for v in values
            ):
                raise ValueError(f"{name} must be a literal integer table in {path}")
            return cast(list[int], list(values))
    raise ValueError(f"Missing table {name} in {path}")


def compare_rates(reference: list[int], actual: list[int]) -> None:
    if len(actual) != 100 or len(reference) < 100:
        raise ValueError(
            "WT11 requires exactly 100 actual and at least 100 reference rates"
        )
    differences = [str(i) for i in range(100) if actual[i] != reference[i]]
    if differences:
        raise ValueError("WT11 mappings differ at rates: " + ", ".join(differences))


def check(reference: Path) -> None:
    if sha256(reference.read_bytes()).hexdigest() != REFERENCE_SHA256:
        raise ValueError(
            "Reference must match DXconvert revision " + REFERENCE_REVISION
        )
    actual = Path(__file__).resolve().parents[1] / "enge/yamaha.py"
    compare_rates(rate_table(reference, "pr"), rate_table(actual, "_WT11_PITCH_RATES"))


REFERENCE_REVISION = "89aaf6cf48ff2eec7e11e82bb92e315b2e6e6621"
REFERENCE_SHA256 = "55c6e636ee45bed46d208ae10b1d1fabf786d8adf8804280864165c987ee6c3e"


if __name__ == "__main__":
    check(tyro.cli(Options).reference)
    print("All 100 legal WT11 rate mappings match the pinned DXconvert table.")
