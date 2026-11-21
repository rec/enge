from collections.abc import Sequence
from typing import Literal

import numpy as np

from .convolution import NativeConvolutionPlan
from .effect_controls import EffectControlPlan

class SynthRuntimeSnapshot: ...

class PreparedEffectGraph:
    def __init__(self, document: str, *, max_records: int = 10000) -> None: ...
    def topology(
        self, sample_rate: int
    ) -> tuple[list[str], dict[str, list[str]], list[str], dict[str, int]]: ...
    def validate_actions(self, documents: list[str]) -> None: ...

def prepare_effect_controls(
    document: str,
    processors: list[tuple[int, str]],
    slots: list[tuple[int, int]],
    followers: list[tuple[str, int]],
    *,
    max_records: int = 10000,
) -> EffectControlPlan: ...

class PreparedTuning:
    def __init__(self, document: str) -> None: ...
    def frequency_hz(self, note: int) -> float: ...

class PreparedFm:
    def __init__(
        self, document: str, rate: float, *, max_records: int = 10000
    ) -> None: ...

def prepare_modulation(
    sample_rate: float,
    context: str,
    generators: list[tuple[str, str]],
    targets: Sequence[tuple[str, str, int, float | None, float, float]],
    *,
    sound_settings: str | None = None,
    fm: PreparedFm | None = None,
    max_records: int = 10000,
) -> tuple[
    np.ndarray,
    list[tuple[int, int]],
    list[float],
    dict[str, list[int]],
    list[str],
    list[str],
    np.ndarray,
    list[str],
    dict[str, list[int]],
    dict[str, list[int]],
    list[str],
    list[tuple[int, int, bool]],
    dict[str, list[int]],
    set[str],
    np.ndarray,
    list[tuple[str, int | None, int]],
    dict[str, list[int]],
    list[tuple[int, str, int, str, int, int, int, int, tuple[int, int]]],
    list[list[tuple[int, list[tuple[float, float]]]]],
    dict[int, str],
    str,
    list[tuple[str, str]],
    list[tuple[str, str, int]],
]: ...

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

class EffectRuntime:
    def __init__(
        self,
        kinds: list[int],
        sources: np.ndarray,
        parameters: np.ndarray,
        output_source: int,
        filters: np.ndarray,
        granulators: np.ndarray,
        tap_delays: np.ndarray,
        modulated_delays: np.ndarray,
        widths: list[int],
        main_input: int,
        tail_thresholds: list[float],
        port_names: list[list[str]],
        control_plan: EffectControlPlan,
        impulses: NativeConvolutionPlan,
        rate: float,
    ) -> None: ...
    def render_frame(
        self,
        frame: int,
        inputs: list[np.ndarray],
        parameters: np.ndarray,
        ended: list[bool],
        frozen: list[bool],
        contributions: list[float],
    ) -> tuple[list[float], bool]: ...
    def export_state(self) -> list[dict[str, object]]: ...
    def enqueue_capture(self, node: int, frame: int, release: bool) -> None: ...
    def import_state(self, values: list[dict[str, object]], frame: int) -> None: ...

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
        widths: list[int],
        main_input: int,
        tail_thresholds: list[float],
        port_names: list[list[str]],
        control_plan: EffectControlPlan,
        impulses: NativeConvolutionPlan,
        batch_capacity: int,
        input_routes: list[list[int]] | None = None,
    ) -> None: ...
    def submit_effects(self, actions: np.ndarray) -> None: ...
    @property
    def effects_drained(self) -> bool: ...
    def set_effect_generators(
        self,
        generators: GeneratorBank,
        maps: list[list[tuple[int, list[tuple[float, float]]]]],
        targets: list[int],
    ) -> None: ...
    def active_slots(self, source: int) -> list[bool]: ...
    def active_contexts(self, source: int) -> list[bool]: ...
    def set_sample_controls(
        self,
        source: int,
        controls: np.ndarray,
        smoothing: list[tuple[int, int]],
        parameters: list[float],
        maps: list[tuple[int, list[tuple[float, float]]]],
        context_capacity: int,
        crossfades: list[tuple[bool, bool]],
        filters: list[tuple[int, int, int, bool, float, float]],
    ) -> None: ...
    def set_sample_generators(
        self,
        source: int,
        generators: GeneratorBank,
        maps: list[list[tuple[int, list[tuple[float, float]]]]],
    ) -> None: ...
    def add_sample(
        self,
        traversal: SampleTraversal,
        routes: np.ndarray,
        envelope: str,
        minimum_hold_frames: float,
        voices: int,
        batch_capacity: int,
        filters: tuple[list[int], np.ndarray, np.ndarray],
        filter_after_amplitude: bool,
        *,
        max_records: int = 10000,
    ) -> int: ...
    def submit(self, source: int, actions: np.ndarray) -> None: ...
    def process_into(
        self, output: np.ndarray, inputs: Sequence[np.ndarray] | None = ...
    ) -> None: ...
    def snapshot(self) -> LiveRuntimeSnapshot: ...
    def restore(self, snapshot: LiveRuntimeSnapshot) -> None: ...
    def failed(self) -> bool: ...

class GeneratorBank:
    def append(
        self, other: GeneratorBank, parameters: list[int]
    ) -> tuple[int, int, int, int]: ...
    def __init__(
        self,
        rate: float,
        slots: int,
        parameter_count: int,
        lfo_routes: np.ndarray,
        lfo_documents: list[str],
        envelope_documents: list[str],
        envelope_routes: list[tuple[int, int, bool]],
        *,
        max_records: int = 10000,
    ) -> None: ...
    def set_beat_clock(self, document: str, *, max_records: int = 10000) -> None: ...
    def set_staged_envelfos(
        self,
        inputs: list[tuple[str, int | None, int]],
        connections: list[
            tuple[int, str, int, str, int, int, int, int, tuple[int, int]]
        ],
        *,
        max_records: int = 10000,
    ) -> None: ...
    def set_sample_hold_keys(self, definitions: list[tuple[int, int]]) -> None: ...
    def set_envelfo_state_owners(
        self, lfos: list[int], envelopes: list[int], stages: list[int]
    ) -> None: ...
    def configure_patches(
        self,
        context: str,
        generators: list[tuple[str, str]],
        slots: tuple[
            list[tuple[str, int]], list[tuple[str, int]], list[tuple[str, int]]
        ],
        targets: list[tuple[str, str, int]],
        orders: list[tuple[str, list[str]]],
        *,
        max_records: int = 10000,
    ) -> tuple[
        list[tuple[tuple[str, str], int]],
        list[tuple[int, list[tuple[float, float]]]],
    ]: ...

class SynthRuntime:
    def set_route_maps(
        self, kind: int, maps: list[tuple[int, list[tuple[float, float]]]]
    ) -> None: ...
    def __init__(
        self,
        rate: float,
        waveform: int,
        duty: float,
        routes: np.ndarray,
        envelope: str,
        minimum_hold_frames: tuple[int, int],
        controls: np.ndarray,
        control_smoothing: list[tuple[int, int]],
        generators: GeneratorBank,
        filters: np.ndarray,
        parameters: list[float],
        context_capacity: int,
        filter_after_amplitude: bool = False,
        *,
        max_records: int = 10000,
    ) -> None: ...
    @staticmethod
    def fm(
        rate: float,
        routes: np.ndarray,
        carrier_envelope: str,
        mod_envelope: str,
        phase_offsets: list[float],
        minimum_hold_frames: tuple[int, int],
        controls: np.ndarray,
        control_smoothing: list[tuple[int, int]],
        generators: GeneratorBank,
        filters: np.ndarray,
        parameters: list[float],
        context_capacity: int,
        filter_after_amplitude: bool = False,
        *,
        max_records: int = 10000,
    ) -> SynthRuntime: ...
    @staticmethod
    def fm_graph(
        routes: np.ndarray,
        graph: PreparedFm,
        minimum_hold_frames: tuple[int, int],
        controls: np.ndarray,
        control_smoothing: list[tuple[int, int]],
        generators: GeneratorBank,
        filters: np.ndarray,
        parameter_definitions: list[float],
        context_capacity: int,
        filter_after_amplitude: bool = False,
    ) -> SynthRuntime: ...
    @staticmethod
    def noise(
        rate: float,
        routes: np.ndarray,
        envelope: str,
        minimum_hold_frames: tuple[int, int],
        controls: np.ndarray,
        control_smoothing: list[tuple[int, int]],
        generators: GeneratorBank,
        filters: np.ndarray,
        parameters: list[float],
        context_capacity: int,
        filter_after_amplitude: bool = False,
        *,
        max_records: int = 10000,
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
    def active_slots(self) -> list[bool]: ...
    def active_contexts(self) -> list[bool]: ...
    def snapshot(self) -> SynthRuntimeSnapshot: ...
    def restore(self, snapshot: SynthRuntimeSnapshot) -> None: ...

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
    filter_after_amplitude: bool = False,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]: ...

class SampleBuffer:
    def __init__(self, samples: np.ndarray) -> None: ...

class SamplePitchMapping:
    def __init__(self, document: str) -> None: ...
    def pitch_ratio(
        self, pitch_hz: float | None, tuning_cents: float, variation_cents: float
    ) -> float: ...

class SampleCrossfades:
    def __init__(self, documents: list[str]) -> None: ...
    def onset_gain(self, key: int | None, velocity: float) -> float: ...
    def control_routes(self) -> list[tuple[float, float, bool, bool]]: ...
    def control_weights(self, index: int, values: np.ndarray) -> np.ndarray: ...

def plan_voice_retirements(document: str) -> str: ...
def plan_voice_release(document: str) -> str: ...

class SampleEligibility:
    def __init__(self, document: str, *, max_records: int = 10000) -> None: ...
    def matching_slots(self, document: str) -> list[str]: ...

def prepare_channel_routes(
    inputs: list[str],
    outputs: list[str],
    routes: list[tuple[str, str, float]],
    spatial: tuple[str, float, float] | None = None,
) -> list[list[float]]: ...
def tuned_frequencies(
    pitch_hz: float,
    tuning: np.ndarray,
    operator: tuple[np.ndarray, np.ndarray] | None = None,
) -> np.ndarray: ...

class PreparedOscillator:
    def __init__(self, document: str) -> None: ...
    def shape(self) -> tuple[int, float]: ...
    def gain(self, note: int) -> float: ...

class PreparedVoiceEligibility:
    def __init__(self, document: str, *, max_records: int = 10000) -> None: ...
    def matching_voices(self, trigger: str, key: int, velocity: float) -> list[str]: ...

def schedule_envelfo_commands(
    tempo: str,
    sample_rate: int,
    events: list[tuple[int, int, tuple[int, int] | None]],
    *,
    max_records: int = 10000,
) -> list[tuple[int, int, int]]: ...
def prepare_voice_start(
    seed: int,
    voice_id: str,
    pitch_hz: float | None,
    frequency_offset_hz: float,
    is_noise: bool,
) -> tuple[float | None, int, int | None]: ...
def resolve_sample_slot_settings(slot: str, group: str | None = None) -> str: ...
def advance_sample_selection(
    mode: str, seed: int, document: str
) -> tuple[str, str]: ...
def draw_sample_range(
    seed: int, part: str, trigger_id: str | None, tick: int, ordinal: int
) -> float: ...
def resolve_sample_variation(
    document: str,
    seed: int,
    part: str,
    trigger_id: str | None,
    tick: int,
    ordinal: int,
    identity: str,
) -> str: ...

class PreparedSlotPlayback:
    def __init__(self, document: str, defaults: str) -> None: ...
    def resolved(self) -> tuple[str, int, float]: ...
    def delay_frames(self, variation: str, sample_rate: int) -> int: ...

class SampleTraversal:
    def __init__(
        self,
        buffer: SampleBuffer,
        slice: str,
        playback: str,
        rate: float,
        additional_plays: int = 0,
        end_fade_seconds: float = 0,
    ) -> None: ...

type SampleState = tuple[
    int,
    float,
    float,
    Literal[-1, 1],
    bool,
    bool,
    bool,
    int | None,
    int | None,
    int,
    int | None,
]

def render_sample(
    traversal: SampleTraversal,
    state: SampleState,
    frame: int,
    steps: np.ndarray,
    release: tuple[int, bool, float, float] | None,
    prepared_gain: float,
    gains: np.ndarray,
    envelope: np.ndarray,
    routes: np.ndarray,
    frames: int,
    filters: tuple[list[int], np.ndarray, np.ndarray] | None = None,
    filter_after_amplitude: bool = False,
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
    carrier_amplitude: bool = True,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]: ...
def render_noise(
    key: int,
    start: int,
    rate: float,
    gains: np.ndarray,
    envelope: np.ndarray,
    routes: np.ndarray,
    filter_inputs: tuple[list[int], np.ndarray, np.ndarray],
    filter_after_amplitude: bool = False,
) -> tuple[np.ndarray, np.ndarray]: ...
