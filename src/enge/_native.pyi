from typing import Literal

import numpy as np

def render_filters(
    samples: np.ndarray,
    rate: float,
    filters: tuple[list[int], np.ndarray, np.ndarray],
) -> tuple[np.ndarray, np.ndarray]: ...
def render_lfo(
    waveform: int, phases: np.ndarray, envelope: np.ndarray, frames: int
) -> np.ndarray: ...
def render(
    waveform: int,
    duty: float,
    rate: float,
    prepared_gain: float,
    frequencies: np.ndarray,
    states: np.ndarray,
    gains: np.ndarray,
    envelope: np.ndarray,
    routes: np.ndarray,
    frames: int,
    filters: tuple[list[int], np.ndarray, np.ndarray] | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]: ...

class SampleBuffer:
    def __init__(self, samples: np.ndarray) -> None: ...

type SampleState = tuple[
    int, float, float, Literal[-1, 1], bool, bool, bool, int | None
]

def render_sample(
    buffer: SampleBuffer,
    selection: tuple[int, int, bool, tuple[int, int, int, bool] | None],
    state: SampleState,
    frame: int,
    rate: float,
    steps: np.ndarray,
    release: tuple[int, bool, float, float] | None,
    prepared_gain: float,
    gains: np.ndarray,
    envelope: np.ndarray,
    routes: np.ndarray,
    frames: int,
    filters: tuple[list[int], np.ndarray, np.ndarray] | None = None,
) -> tuple[np.ndarray, SampleState, np.ndarray]: ...
