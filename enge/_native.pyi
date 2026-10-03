from typing import Literal

import numpy as np

class SynthRuntimeSnapshot: ...

class RubberBandLiveShifter:
    def __init__(self, sample_rate: int, channels: int, pitch: float) -> None: ...
    @property
    def block_size(self) -> int: ...
    @property
    def start_delay(self) -> int: ...
    def shift(self, audio: np.ndarray) -> np.ndarray: ...

class RubberBandRealtimeStretcher:
    def __init__(
        self,
        sample_rate: int,
        channels: int,
        ratio: float,
        pitch: float,
        max_block_frames: int,
    ) -> None: ...
    @property
    def start_delay(self) -> int: ...
    def process(self, audio: np.ndarray, final_block: bool) -> np.ndarray: ...

def rubberband_stretch(audio: np.ndarray, ratio: float, pitch: float) -> np.ndarray: ...

class ActionQueue:
    def __init__(self, batch_capacity: int) -> None: ...
    def push(self, actions: np.ndarray) -> None: ...
    def drain_into(self, output: np.ndarray) -> int: ...
    def queued_batches(self) -> int: ...

class LiveRuntimeSnapshot: ...

class LiveRuntime:
    def __init__(
        self,
        runtimes: list[SynthRuntime],
        maximum_block_frames: int,
        action_capacity: int,
        batch_capacity: int,
    ) -> None: ...
    @staticmethod
    def empty(
        rate: float,
        channels: int,
        maximum_block_frames: int,
        action_capacity: int,
    ) -> LiveRuntime: ...
    def set_effect_graph(
        self,
        kinds: list[int],
        sources: np.ndarray,
        parameters: np.ndarray,
        output_source: int,
        filters: np.ndarray,
        granulators: np.ndarray,
        tap_delays: np.ndarray,
        modulated_delays: np.ndarray,
        batch_capacity: int,
    ) -> None: ...
    def submit_effects(self, actions: np.ndarray) -> None: ...
    def active_slots(self, source: int) -> list[bool]: ...
    def active_contexts(self, source: int) -> list[bool]: ...
    def add_sample(
        self,
        buffer: SampleBuffer,
        selection: tuple[int, int, bool, tuple[int, int, int, bool] | None],
        state: tuple[int, float, float, int, bool, bool, bool, int | None],
        rate: float,
        routes: np.ndarray,
        initial: float,
        attack: np.ndarray,
        release: np.ndarray,
        minimum_hold_frames: float,
        voices: int,
        batch_capacity: int,
    ) -> int: ...
    def submit(self, source: int, actions: np.ndarray) -> None: ...
    def process_into(self, output: np.ndarray) -> None: ...
    def snapshot(self) -> LiveRuntimeSnapshot: ...
    def restore(self, snapshot: LiveRuntimeSnapshot) -> None: ...
    def failed(self) -> bool: ...
    def reset_failure(self) -> None: ...

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
        controls: np.ndarray,
        control_smoothing: list[tuple[int, int]],
        lfos: np.ndarray,
        lfo_rationals: list[tuple[int, int]],
        filters: np.ndarray,
        parameters: list[float],
        context_capacity: int,
    ) -> None: ...
    @staticmethod
    def fm(
        rate: float,
        routes: np.ndarray,
        carrier_initial: float,
        carrier_attack: np.ndarray,
        carrier_release: np.ndarray,
        mod_initial: float,
        mod_attack: np.ndarray,
        mod_release: np.ndarray,
        phase_offsets: list[float],
        minimum_hold_frames: float,
        controls: np.ndarray,
        control_smoothing: list[tuple[int, int]],
        lfos: np.ndarray,
        lfo_rationals: list[tuple[int, int]],
        filters: np.ndarray,
        parameters: list[float],
        context_capacity: int,
    ) -> SynthRuntime: ...
    @staticmethod
    def fm_graph(
        rate: float,
        routes: np.ndarray,
        carrier_initial: float,
        carrier_attack: np.ndarray,
        carrier_release: np.ndarray,
        waveforms: list[int],
        initials: list[float],
        attacks: list[list[tuple[float, float]]],
        releases: list[list[tuple[float, float]]],
        phase_offsets: list[float],
        parameters: list[tuple[int, int]],
        edges: list[tuple[int, int, bool, int]],
        order: list[int],
        carrier: int,
        carrier_parameter: int,
        minimum_hold_frames: float,
        controls: np.ndarray,
        control_smoothing: list[tuple[int, int]],
        lfos: np.ndarray,
        lfo_rationals: list[tuple[int, int]],
        filters: np.ndarray,
        parameter_definitions: list[float],
        context_capacity: int,
    ) -> SynthRuntime: ...
    @staticmethod
    def noise(
        rate: float,
        routes: np.ndarray,
        initial: float,
        attack: np.ndarray,
        release: np.ndarray,
        minimum_hold_frames: float,
        controls: np.ndarray,
        control_smoothing: list[tuple[int, int]],
        lfos: np.ndarray,
        lfo_rationals: list[tuple[int, int]],
        filters: np.ndarray,
        parameters: list[float],
        context_capacity: int,
    ) -> SynthRuntime: ...
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
    def set_named_envelopes(
        self,
        initials: list[float],
        attacks: list[list[tuple[float, float]]],
        releases: list[list[tuple[float, float]]],
        parameters: list[
            tuple[int, int, float, float, bool, int, float, float, int | None]
        ],
    ) -> None: ...
    def set_beat_clock(
        self, points: list[tuple[float, float, float, float]]
    ) -> None: ...
    def set_beat_named_envelopes(self, indices: list[int]) -> None: ...
    def set_staged_motions(
        self,
        definitions: list[
            tuple[
                list[
                    tuple[
                        int,
                        float,
                        bool,
                        list[tuple[float, float]],
                        list[float],
                        list[tuple[float, str]],
                        int,
                        float,
                        float,
                        int | None,
                    ]
                ],
                list[tuple[int, str, int]],
                int,
                int | None,
                int,
                float,
                float,
                bool,
            ]
        ],
        connections: list[tuple[int, str, int, str]],
    ) -> None: ...
    def set_motion_state_owners(
        self, lfos: list[int], envelopes: list[int], stages: list[int]
    ) -> None: ...
    def active_slots(self) -> list[bool]: ...
    def active_contexts(self) -> list[bool]: ...
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
    freeze_crossfade_frames: int,
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
def render_graph_fm(
    rate: float,
    frequencies: np.ndarray,
    envelopes: np.ndarray,
    waveforms: np.ndarray,
    indices: np.ndarray,
    edges: np.ndarray,
    carrier: int,
    levels: np.ndarray,
    phases: np.ndarray,
    history: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]: ...
def render_noise(
    key: int,
    start: int,
    rate: float,
    gains: np.ndarray,
    envelope: np.ndarray,
    routes: np.ndarray,
    filter_inputs: tuple[list[int], np.ndarray, np.ndarray],
) -> tuple[np.ndarray, np.ndarray]: ...
