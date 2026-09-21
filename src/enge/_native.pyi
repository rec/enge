from typing import Literal

import numpy as np

class SynthRuntimeSnapshot: ...

class SynthRuntime:
    def __init__(
        self,
        rate: float,
        waveform: int,
        duty: float,
        routes: np.ndarray,
        initial: float,
        attack: np.ndarray,
        release: np.ndarray,
        minimum_hold_frames: float,
    ) -> None: ...
    def process(
        self, frames: int, cutoff_hz: float, q: float, gain_db: float
    ) -> np.ndarray: ...
    def process_actions(
        self,
        frames: int,
        cutoff_hz: float,
        q: float,
        gain_db: float,
        actions: np.ndarray,
    ) -> np.ndarray: ...
    def process_into(
        self, output: np.ndarray, cutoff_hz: float, q: float, gain_db: float
    ) -> None: ...
    def process_actions_into(
        self,
        output: np.ndarray,
        cutoff_hz: float,
        q: float,
        gain_db: float,
        actions: np.ndarray,
        action_count: int,
    ) -> None: ...
    def active_slots(self) -> list[bool]: ...
    def snapshot(self) -> SynthRuntimeSnapshot: ...
    def restore(self, snapshot: SynthRuntimeSnapshot) -> None: ...

def render_effects(
    inputs: np.ndarray,
    kinds: list[int],
    sources: np.ndarray,
    parameters: np.ndarray,
    output_source: int,
    rate: float,
    filter_inputs: list[tuple[list[int], np.ndarray, np.ndarray] | None],
    state_floors: list[float],
) -> tuple[np.ndarray, list[np.ndarray]]: ...
def render_granulator(
    samples: np.ndarray,
    parameters: np.ndarray,
    start_frame: int,
    rate: int,
    history_frames: int,
    maximum_grains: int,
    history: np.ndarray,
    history_start: int,
    phase: float,
    counter: int,
    grain_samples: list[np.ndarray],
    grain_indices: list[int],
    grain_gains: list[float],
) -> tuple[
    np.ndarray,
    np.ndarray,
    int,
    float,
    int,
    list[np.ndarray],
    list[int],
    list[float],
]: ...
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
def render_fm(
    rate: float,
    parameters: np.ndarray,
    phases: np.ndarray,
    history: float,
    gains: np.ndarray,
    modulator: np.ndarray,
    carrier: np.ndarray,
    modulator_frames: int,
    routes: np.ndarray,
    filter_inputs: tuple[list[int], np.ndarray, np.ndarray],
) -> tuple[np.ndarray, np.ndarray, float, np.ndarray]: ...
def render_noise(
    key: int,
    start: int,
    rate: float,
    gains: np.ndarray,
    envelope: np.ndarray,
    routes: np.ndarray,
    filter_inputs: tuple[list[int], np.ndarray, np.ndarray],
) -> tuple[np.ndarray, np.ndarray]: ...
