"""Cut-only parallel bell using the existing normalized band-pass and dynamics."""

from math import isfinite

import numpy as np
from ufor.audio_effects import DynamicEQ
from ufor.samples.processing import FilterResponse, ResonantFilter

from . import compressor, filters
from .filters import FilterState


def band_definition(definition: DynamicEQ) -> ResonantFilter:
    return ResonantFilter(
        name="band",
        response=FilterResponse.bandpass,
        cutoff_hz=definition.center_hz,
        q=definition.q,
    )


def validate_state(states: list[FilterState], reduction: float, channels: int) -> None:
    memory = np.asarray([s.integrators for s in states])
    if (
        memory.shape != (1, channels, 2)
        or not np.all(np.isfinite(memory))
        or not isfinite(reduction)
        or reduction > 0
    ):
        raise ValueError("Invalid dynamic EQ snapshot")


def process(
    definition: DynamicEQ,
    band: ResonantFilter,
    states: list[FilterState],
    reduction: float,
    source: np.ndarray,
    values: dict[str, float],
    rate: int,
    ended: bool,
) -> tuple[np.ndarray, list[FilterState], float, bool]:
    parameters = filters.parameters(
        [band], rate, 1, np.array([[[values["center_hz"], values["q"]]]])
    )
    audio, states = filters.filter_samples(
        [band], states, source[None, :], parameters, rate
    )
    states = [
        s.model_copy(
            update={
                "integrators": [
                    [0 if abs(v) < definition.state_floor else v for v in r]
                    for r in s.integrators
                ]
            }
        )
        for s in states
    ]
    gain, reduction = compressor.gain(
        float(np.max(np.abs(audio[0]))),
        [
            values[p]
            for p in (
                "threshold_db",
                "ratio",
                "knee_db",
                "attack_seconds",
                "release_seconds",
            )
        ]
        + [0.0],
        reduction,
        rate,
        -values["maximum_cut_db"],
    )
    wet = source + (gain - 1) * audio[0]
    drained = ended and all(
        abs(v) < definition.tail_threshold
        for s in states
        for r in s.integrators
        for v in r
    )
    return (np.zeros_like(source) if drained else wet), states, reduction, drained
