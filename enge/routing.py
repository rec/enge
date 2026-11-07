"""Always-applied matrix routing with scalar compensated accumulation."""

from math import fsum

import numpy as np


def process(coefficients: list[list[float]], source: np.ndarray) -> np.ndarray:
    try:
        output = np.asarray(
            [
                fsum(c * float(x) for c, x in zip(r, source, strict=True))
                for r in coefficients
            ]
        )
    except OverflowError as error:
        raise ValueError("Non-finite routing matrix sum") from error
    if not np.all(np.isfinite(output)):
        raise ValueError("Non-finite routing matrix output")
    return output
