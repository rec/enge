//! Persistent oscillator runtime with sample-accurate voice actions.

use numpy::ndarray::{Array2, Array3, ArrayViewMut2};
use numpy::{IntoPyArray, PyArray2, PyReadonlyArray2, PyReadwriteArray2, PyUntypedArrayMethods};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use std::collections::VecDeque;
use std::f64::consts::{PI, TAU};

#[derive(Clone, PartialEq)]
pub(crate) struct Segment {
    pub(crate) frames: f64,
    pub(crate) target: f64,
}

#[derive(Clone, PartialEq)]
struct ControlDefinition {
    default: f64,
    smoothing_numerator: u64,
    smoothing_denominator: u64,
    scope: usize,
    parameter: usize,
    operation: usize,
    minimum: f64,
    maximum: f64,
    intercept: f64,
    slope: f64,
}

#[derive(Clone)]
struct ControlState {
    start: f64,
    target: f64,
    elapsed: usize,
}

#[derive(Clone, PartialEq)]
struct LfoDefinition {
    scope: usize,
    waveform: usize,
    parameter: usize,
    operation: usize,
    intercept: f64,
    slope: f64,
    duty: (u64, u64),
    rate: (u64, u64),
    phase: (u64, u64),
    delay: (u64, u64),
    fade: (u64, u64),
    beat_clock: bool,
    transport_position: bool,
}

#[derive(Clone)]
struct LfoEventState {
    at: usize,
    age: f64,
    phase: f64,
    rate: f64,
    direction: f64,
    paused: bool,
}

#[derive(Clone, PartialEq)]
struct BeatPoint {
    frame: f64,
    beat: f64,
    song_beat: f64,
    rate: f64,
}

#[derive(Clone, PartialEq)]
struct NamedEnvelopeDefinition {
    sample_hold: Option<(u64, f64, f64)>,
    initial: f64,
    attack: Vec<Segment>,
    release: Vec<Segment>,
    parameter: usize,
    operation: usize,
    intercept: f64,
    slope: f64,
    release_with_voice: bool,
    playback: usize,
    loop_start: f64,
    loop_end: f64,
    repeat_count: Option<usize>,
    beat_clock: bool,
    event_started: bool,
}

#[derive(Clone)]
struct NamedContourState {
    random_state: u64,
    playback: PlaybackState,
    start_value: f64,
    released: bool,
    pending_release: Option<(f64, f64)>,
    traversals: usize,
    complete_coordinate: Option<f64>,
    idle: bool,
}

#[derive(Clone, PartialEq)]
struct PatchEventConnection {
    threshold: bool,
    source: usize,
    marker: f64,
    destination: usize,
    cue: Option<String>,
    order: usize,
    every: usize,
    offset: usize,
    probability: u64,
    key: u64,
    delay: f64,
}

#[derive(Clone, PartialEq)]
struct PatchStageStartConnection {
    source: usize,
    port: String,
    destination: usize,
    order: usize,
    every: usize,
    offset: usize,
    probability: u64,
    key: u64,
    delay: f64,
}

#[derive(Clone)]
struct ContourStart {
    at: f64,
    order: usize,
    voice: usize,
    destination: usize,
}

#[derive(Clone, PartialEq)]
struct StageDefinition {
    kind: u8,
    playback: usize,
    loop_start: f64,
    loop_end: f64,
    repeat_count: Option<usize>,
    initial: f64,
    current_initial: bool,
    segments: Vec<Segment>,
    cycle: Vec<f64>,
    markers: Vec<(f64, String)>,
}

#[derive(Clone, PartialEq)]
struct StagedMotionDefinition {
    stages: Vec<StageDefinition>,
    transitions: Vec<(usize, String, usize)>,
    initial_stage: usize,
    parameter: Option<usize>,
    operation: usize,
    intercept: f64,
    slope: f64,
    beat_clock: bool,
}

#[derive(Clone)]
struct StagedMotionState {
    stage: usize,
    playback: PlaybackState,
    entry_value: f64,
    completed: bool,
    complete_value: f64,
    cursor_at: f64,
    cursor_order: usize,
    traversals: usize,
}

#[derive(Clone)]
struct PlaybackState {
    at: f64,
    coordinate: f64,
    rate: f64,
    direction: f64,
    paused: bool,
    age: f64,
}

impl PlaybackState {
    fn coordinate_at(&self, at: f64) -> f64 {
        self.coordinate
            + if self.paused {
                0.0
            } else {
                self.direction * self.rate * (at - self.at)
            }
    }

    fn age_at(&self, at: f64) -> f64 {
        self.age + if self.paused { 0.0 } else { at - self.at }
    }

    fn advance(&mut self, at: f64) {
        self.coordinate = self.coordinate_at(at);
        self.age = self.age_at(at);
        self.at = at;
    }

    fn command(&mut self, kind: usize, position: f64, at: f64) {
        self.advance(at);
        match kind {
            0 => self.paused = true,
            1 => self.paused = false,
            2 => self.direction = -self.direction,
            3 => self.coordinate = position,
            4 => self.coordinate += position,
            _ => unreachable!(),
        }
    }
}

#[derive(Clone, PartialEq)]
struct StagedConnection {
    source: usize,
    port: String,
    destination: usize,
    cue: String,
    every: usize,
    offset: usize,
    probability: u64,
    key: u64,
    delay: f64,
}

#[derive(Clone)]
struct StagedEvent {
    at: f64,
    voice: usize,
    source: usize,
    port: String,
}

#[derive(Clone, Copy)]
enum DelayedConnection {
    StagedCue(usize),
    PatchEvent(usize),
    StageStart(usize),
}

#[derive(Clone)]
struct DelayedEvent {
    at: f64,
    connection: DelayedConnection,
}

type PatchEventInput = (
    usize,
    f64,
    usize,
    Option<String>,
    usize,
    usize,
    usize,
    u64,
    u64,
    f64,
);
type StageConnectionInput = (usize, String, usize, String, usize, usize, u64, u64, f64);
type StageStartInput = (usize, String, usize, usize, usize, usize, u64, u64, f64);

type MotionTransformInput = (u8, Vec<(u8, usize, f64, f64)>, f64, f64, Option<f64>);
type MotionTransformRoute = (usize, usize, usize, f64, f64);

type StageInput = (
    u8,
    f64,
    bool,
    Vec<(f64, f64)>,
    Vec<f64>,
    Vec<(f64, String)>,
    usize,
    f64,
    f64,
    Option<usize>,
);
type NamedContourInput = (
    usize,
    usize,
    f64,
    f64,
    bool,
    usize,
    f64,
    f64,
    Option<usize>,
    bool,
);
type StagedMotionInput = (
    Vec<StageInput>,
    Vec<(usize, String, usize)>,
    usize,
    Option<usize>,
    usize,
    f64,
    f64,
    bool,
);

#[derive(Clone, PartialEq)]
struct FilterDefinition {
    response: usize,
    stages: usize,
    boundary: usize,
    minimum_hz: f64,
    maximum_hz: f64,
    cutoff_parameter: usize,
    q_parameter: usize,
}

#[derive(Clone, PartialEq)]
struct GraphOperator {
    waveform: u8,
    initial: f64,
    attack: Vec<Segment>,
    release: Vec<Segment>,
    phase_offset: f64,
    ratio_parameter: usize,
    tuning_parameter: usize,
}

#[derive(Clone, PartialEq)]
struct GraphEdge {
    source: usize,
    destination: usize,
    delayed: bool,
    parameter: usize,
}

#[pyclass(frozen, skip_from_py_object)]
#[derive(Clone)]
pub struct SynthRuntimeSnapshot {
    motion_transform_states: Vec<Option<f64>>,
    motion_transforms: Vec<MotionTransformInput>,
    motion_transform_routes: Vec<MotionTransformRoute>,
    rate: f64,
    waveform: u8,
    source_kind: u8,
    duty: f64,
    mod_initial: f64,
    mod_attack: Vec<Segment>,
    mod_release: Vec<Segment>,
    fm_phase_offsets: [f64; 2],
    graph_operators: Vec<GraphOperator>,
    graph_edges: Vec<GraphEdge>,
    graph_order: Vec<usize>,
    graph_carrier: usize,
    graph_carrier_parameter: usize,
    initial: f64,
    attack: Vec<Segment>,
    release: Vec<Segment>,
    minimum_hold_frames: f64,
    routes: Array2<f64>,
    control_definitions: Vec<ControlDefinition>,
    control_states: Vec<ControlState>,
    lfo_definitions: Vec<LfoDefinition>,
    lfo_owners: Vec<usize>,
    lfo_event_states: Vec<Option<LfoEventState>>,
    voice_lfo_event_states: Vec<Option<LfoEventState>>,
    named_envelopes: Vec<NamedEnvelopeDefinition>,
    named_owners: Vec<usize>,
    patch_events: Vec<PatchEventConnection>,
    patch_stage_starts: Vec<PatchStageStartConnection>,
    pending_contour_starts: Vec<ContourStart>,
    delayed_events: Vec<Vec<DelayedEvent>>,
    event_counts: Vec<usize>,
    motion_keys: Vec<u64>,
    event_random: Vec<u64>,
    beat_points: Vec<BeatPoint>,
    staged_motions: Vec<StagedMotionDefinition>,
    staged_owners: Vec<usize>,
    staged_states: Vec<StagedMotionState>,
    staged_connections: Vec<StagedConnection>,
    staged_pending: VecDeque<StagedEvent>,
    filter_definitions: Vec<FilterDefinition>,
    context_kinds: Vec<usize>,
    context_states: Vec<ControlState>,
    parameter_definitions: Vec<f64>,
    frequencies: Vec<f64>,
    phases: Vec<f64>,
    errors: Vec<f64>,
    mod_phases: Vec<f64>,
    mod_errors: Vec<f64>,
    previous_modulators: Vec<f64>,
    mod_release_levels: Vec<f64>,
    named_states: Vec<NamedContourState>,
    graph_phases: Vec<f64>,
    graph_errors: Vec<f64>,
    graph_outputs: Vec<f64>,
    graph_history: Vec<f64>,
    graph_release_levels: Vec<f64>,
    noise_keys: Vec<u64>,
    noise_counters: Vec<u64>,
    gains: Vec<f64>,
    active: Vec<bool>,
    ages: Vec<usize>,
    release_frames: Vec<Option<f64>>,
    release_levels: Vec<f64>,
    frequency_steps: Vec<f64>,
    frequency_remaining: Vec<usize>,
    gain_steps: Vec<f64>,
    gain_remaining: Vec<usize>,
    voice_part_contexts: Vec<usize>,
    voice_trigger_contexts: Vec<usize>,
    filter_states: Array2<f64>,
    voice_filter_states: Array3<f64>,
    frame: usize,
}

#[pyclass(skip_from_py_object)]
#[derive(Clone)]
pub struct SynthRuntime {
    motion_transforms: Vec<MotionTransformInput>,
    motion_transform_routes: Vec<MotionTransformRoute>,
    motion_transform_values: Vec<f64>,
    motion_transform_states: Vec<Option<f64>>,
    rate: f64,
    filter_after_amplitude: bool,
    waveform: u8,
    source_kind: u8,
    duty: f64,
    mod_initial: f64,
    mod_attack: Vec<Segment>,
    mod_release: Vec<Segment>,
    fm_phase_offsets: [f64; 2],
    graph_operators: Vec<GraphOperator>,
    graph_edges: Vec<GraphEdge>,
    graph_order: Vec<usize>,
    graph_carrier: usize,
    graph_carrier_parameter: usize,
    initial: f64,
    attack: Vec<Segment>,
    release: Vec<Segment>,
    minimum_hold_frames: f64,
    routes: Array2<f64>,
    control_definitions: Vec<ControlDefinition>,
    control_states: Vec<ControlState>,
    lfo_definitions: Vec<LfoDefinition>,
    lfo_owners: Vec<usize>,
    lfo_event_states: Vec<Option<LfoEventState>>,
    voice_lfo_event_states: Vec<Option<LfoEventState>>,
    named_envelopes: Vec<NamedEnvelopeDefinition>,
    named_owners: Vec<usize>,
    patch_events: Vec<PatchEventConnection>,
    patch_stage_starts: Vec<PatchStageStartConnection>,
    pending_contour_starts: Vec<ContourStart>,
    delayed_events: Vec<Vec<DelayedEvent>>,
    event_counts: Vec<usize>,
    motion_keys: Vec<u64>,
    event_random: Vec<u64>,
    beat_points: Vec<BeatPoint>,
    staged_motions: Vec<StagedMotionDefinition>,
    staged_owners: Vec<usize>,
    staged_states: Vec<StagedMotionState>,
    staged_connections: Vec<StagedConnection>,
    staged_pending: VecDeque<StagedEvent>,
    filter_definitions: Vec<FilterDefinition>,
    context_kinds: Vec<usize>,
    context_states: Vec<ControlState>,
    parameter_definitions: Vec<f64>,
    frequencies: Vec<f64>,
    phases: Vec<f64>,
    errors: Vec<f64>,
    mod_phases: Vec<f64>,
    mod_errors: Vec<f64>,
    previous_modulators: Vec<f64>,
    mod_release_levels: Vec<f64>,
    named_states: Vec<NamedContourState>,
    graph_phases: Vec<f64>,
    graph_errors: Vec<f64>,
    graph_outputs: Vec<f64>,
    graph_history: Vec<f64>,
    graph_release_levels: Vec<f64>,
    noise_keys: Vec<u64>,
    noise_counters: Vec<u64>,
    gains: Vec<f64>,
    active: Vec<bool>,
    ages: Vec<usize>,
    release_frames: Vec<Option<f64>>,
    release_levels: Vec<f64>,
    frequency_steps: Vec<f64>,
    frequency_remaining: Vec<usize>,
    gain_steps: Vec<f64>,
    gain_remaining: Vec<usize>,
    voice_part_contexts: Vec<usize>,
    voice_trigger_contexts: Vec<usize>,
    filter_states: Array2<f64>,
    voice_filter_states: Array3<f64>,
    frame: usize,
}

#[pymethods]
impl SynthRuntime {
    #[new]
    #[allow(clippy::too_many_arguments)]
    #[pyo3(signature = (rate, waveform, duty, routes, initial, attack, release, minimum_hold_frames, controls, control_smoothing, lfos, lfo_rationals, filters, parameters, context_capacity, filter_after_amplitude=false))]
    fn new(
        rate: f64,
        waveform: u8,
        duty: f64,
        routes: PyReadonlyArray2<'_, f64>,
        initial: f64,
        attack: PyReadonlyArray2<'_, f64>,
        release: PyReadonlyArray2<'_, f64>,
        minimum_hold_frames: f64,
        controls: PyReadonlyArray2<'_, f64>,
        control_smoothing: Vec<(u64, u64)>,
        lfos: PyReadonlyArray2<'_, f64>,
        lfo_rationals: Vec<(u64, u64)>,
        filters: PyReadonlyArray2<'_, f64>,
        parameters: Vec<f64>,
        context_capacity: usize,
        filter_after_amplitude: bool,
    ) -> PyResult<Self> {
        let attack = segments(attack)?;
        let release = segments(release)?;
        let (control_definitions, control_states) =
            control_definitions(controls, &control_smoothing)?;
        let lfo_definitions = lfo_definitions(lfos, &lfo_rationals)?;
        let filter_definitions = filter_definitions(filters)?;
        if !rate.is_finite()
            || rate <= 0.0
            || waveform > 2
            || !duty.is_finite()
            || !(0.0..=1.0).contains(&duty)
            || routes.shape()[0] == 0
            || routes.shape()[1] == 0
            || routes.as_array().iter().any(|v| !v.is_finite())
            || !initial.is_finite()
            || !minimum_hold_frames.is_finite()
            || minimum_hold_frames < 0.0
            || parameters.len() < 6
            || parameters.len() % 3 != 0
            || parameters.iter().any(|v| !v.is_finite())
            || parameters
                .chunks_exact(3)
                .any(|values| values[1] > values[0] || values[0] > values[2])
            || filter_definitions.iter().any(|definition| {
                definition.cutoff_parameter >= parameters.len() / 3
                    || definition.q_parameter >= parameters.len() / 3
            })
            || control_definitions
                .iter()
                .any(|definition| definition.parameter >= parameters.len() / 3)
            || lfo_definitions
                .iter()
                .any(|definition| definition.parameter >= parameters.len() / 3)
            || context_capacity == 0
            || (!lfo_definitions.is_empty() && rate != rate as u64 as f64)
            || lfo_definitions.iter().any(|definition| {
                !definition.beat_clock && lfo_phase_denominator(definition, rate as u64).is_none()
            })
        {
            return Err(PyValueError::new_err("Invalid synth runtime definition"));
        }
        let slots = routes.shape()[0];
        let channels = routes.shape()[1];
        let filter_stages = filter_definitions.iter().map(|v| v.stages).sum();
        let lfo_count = lfo_definitions.len();
        let context_states = (0..context_capacity)
            .flat_map(|_| control_states.clone())
            .collect();
        Ok(Self {
            motion_transforms: Vec::new(),
            motion_transform_routes: Vec::new(),
            motion_transform_values: Vec::new(),
            motion_transform_states: Vec::new(),
            rate,
            waveform,
            source_kind: 0,
            filter_after_amplitude,
            duty,
            mod_initial: 0.0,
            mod_attack: Vec::new(),
            mod_release: Vec::new(),
            fm_phase_offsets: [0.0; 2],
            graph_operators: Vec::new(),
            graph_edges: Vec::new(),
            graph_order: Vec::new(),
            graph_carrier: 0,
            graph_carrier_parameter: 0,
            initial,
            attack,
            release,
            minimum_hold_frames,
            routes: routes.as_array().to_owned(),
            control_definitions,
            control_states,
            lfo_definitions,
            lfo_owners: (0..lfo_count).collect(),
            lfo_event_states: vec![None; lfo_count],
            voice_lfo_event_states: vec![None; slots * lfo_count],
            named_envelopes: Vec::new(),
            named_owners: Vec::new(),
            patch_events: Vec::new(),
            patch_stage_starts: Vec::new(),
            pending_contour_starts: Vec::new(),
            delayed_events: (0..slots).map(|_| Vec::new()).collect(),
            event_counts: Vec::new(),
            motion_keys: vec![0; slots],
            event_random: Vec::new(),
            beat_points: Vec::new(),
            staged_motions: Vec::new(),
            staged_owners: Vec::new(),
            staged_states: Vec::new(),
            staged_connections: Vec::new(),
            staged_pending: VecDeque::new(),
            filter_definitions,
            context_kinds: vec![0; context_capacity],
            context_states,
            parameter_definitions: parameters,
            frequencies: vec![1.0; slots],
            phases: vec![0.0; slots],
            errors: vec![0.0; slots],
            mod_phases: vec![0.0; slots],
            mod_errors: vec![0.0; slots],
            previous_modulators: vec![0.0; slots],
            mod_release_levels: vec![0.0; slots],
            named_states: Vec::new(),
            graph_phases: Vec::new(),
            graph_errors: Vec::new(),
            graph_outputs: Vec::new(),
            graph_history: Vec::new(),
            graph_release_levels: Vec::new(),
            noise_keys: vec![0; slots],
            noise_counters: vec![0; slots],
            gains: vec![0.0; slots],
            active: vec![false; slots],
            ages: vec![0; slots],
            release_frames: vec![None; slots],
            release_levels: vec![0.0; slots],
            frequency_steps: vec![0.0; slots],
            frequency_remaining: vec![0; slots],
            gain_steps: vec![0.0; slots],
            gain_remaining: vec![0; slots],
            voice_part_contexts: vec![usize::MAX; slots],
            voice_trigger_contexts: vec![usize::MAX; slots],
            filter_states: Array2::zeros((channels, 2)),
            voice_filter_states: Array3::zeros((slots, filter_stages, 2)),
            frame: 0,
        })
    }

    #[staticmethod]
    #[allow(clippy::too_many_arguments)]
    #[pyo3(signature = (rate, routes, carrier_initial, carrier_attack, carrier_release, mod_initial, mod_attack, mod_release, phase_offsets, minimum_hold_frames, controls, control_smoothing, lfos, lfo_rationals, filters, parameters, context_capacity, filter_after_amplitude=false))]
    fn fm(
        rate: f64,
        routes: PyReadonlyArray2<'_, f64>,
        carrier_initial: f64,
        carrier_attack: PyReadonlyArray2<'_, f64>,
        carrier_release: PyReadonlyArray2<'_, f64>,
        mod_initial: f64,
        mod_attack: PyReadonlyArray2<'_, f64>,
        mod_release: PyReadonlyArray2<'_, f64>,
        phase_offsets: Vec<f64>,
        minimum_hold_frames: f64,
        controls: PyReadonlyArray2<'_, f64>,
        control_smoothing: Vec<(u64, u64)>,
        lfos: PyReadonlyArray2<'_, f64>,
        lfo_rationals: Vec<(u64, u64)>,
        filters: PyReadonlyArray2<'_, f64>,
        parameters: Vec<f64>,
        context_capacity: usize,
        filter_after_amplitude: bool,
    ) -> PyResult<Self> {
        if phase_offsets.len() != 2
            || phase_offsets.iter().any(|v| !v.is_finite())
            || !mod_initial.is_finite()
        {
            return Err(PyValueError::new_err("Invalid FM runtime definition"));
        }
        let mod_attack = segments(mod_attack)?;
        let mod_release = segments(mod_release)?;
        let mut runtime = Self::new(
            rate,
            0,
            0.5,
            routes,
            carrier_initial,
            carrier_attack,
            carrier_release,
            minimum_hold_frames,
            controls,
            control_smoothing,
            lfos,
            lfo_rationals,
            filters,
            parameters,
            context_capacity,
            filter_after_amplitude,
        )?;
        runtime.source_kind = 1;
        runtime.mod_initial = mod_initial;
        runtime.mod_attack = mod_attack;
        runtime.mod_release = mod_release;
        runtime.fm_phase_offsets = [phase_offsets[0], phase_offsets[1]];
        Ok(runtime)
    }

    #[staticmethod]
    #[allow(clippy::too_many_arguments)]
    #[pyo3(signature = (rate, routes, carrier_initial, carrier_attack, carrier_release, waveforms, initials, attacks, releases, phase_offsets, parameters, edges, order, carrier, carrier_parameter, minimum_hold_frames, controls, control_smoothing, lfos, lfo_rationals, filters, parameter_definitions, context_capacity, filter_after_amplitude=false))]
    fn fm_graph(
        rate: f64,
        routes: PyReadonlyArray2<'_, f64>,
        carrier_initial: f64,
        carrier_attack: PyReadonlyArray2<'_, f64>,
        carrier_release: PyReadonlyArray2<'_, f64>,
        waveforms: Vec<u8>,
        initials: Vec<f64>,
        attacks: Vec<Vec<(f64, f64)>>,
        releases: Vec<Vec<(f64, f64)>>,
        phase_offsets: Vec<f64>,
        parameters: Vec<(usize, usize)>,
        edges: Vec<(usize, usize, bool, usize)>,
        order: Vec<usize>,
        carrier: usize,
        carrier_parameter: usize,
        minimum_hold_frames: f64,
        controls: PyReadonlyArray2<'_, f64>,
        control_smoothing: Vec<(u64, u64)>,
        lfos: PyReadonlyArray2<'_, f64>,
        lfo_rationals: Vec<(u64, u64)>,
        filters: PyReadonlyArray2<'_, f64>,
        parameter_definitions: Vec<f64>,
        context_capacity: usize,
        filter_after_amplitude: bool,
    ) -> PyResult<Self> {
        let operators = waveforms.len();
        if !(2..=6).contains(&operators)
            || initials.len() != operators
            || attacks.len() != operators
            || releases.len() != operators
            || phase_offsets.len() != operators
            || parameters.len() != operators
            || carrier >= operators
            || carrier_parameter >= parameter_definitions.len() / 3
            || waveforms.iter().any(|v| *v > 2)
            || initials
                .iter()
                .chain(&phase_offsets)
                .any(|v| !v.is_finite())
            || order.len() != operators
            || order.iter().any(|v| *v >= operators)
        {
            return Err(PyValueError::new_err("Invalid graph FM runtime definition"));
        }
        let mut positions = vec![usize::MAX; operators];
        for (position, operator) in order.iter().enumerate() {
            if positions[*operator] != usize::MAX {
                return Err(PyValueError::new_err("Invalid graph FM execution order"));
            }
            positions[*operator] = position;
        }
        let graph_operators: Vec<GraphOperator> = waveforms
            .into_iter()
            .zip(initials)
            .zip(attacks)
            .zip(releases)
            .zip(phase_offsets)
            .zip(parameters)
            .map(
                |(((((waveform, initial), attack), release), phase_offset), parameters)| {
                    if attack.iter().chain(&release).any(|(frames, target)| {
                        !frames.is_finite() || *frames < 0.0 || !target.is_finite()
                    }) || parameters.0 >= parameter_definitions.len() / 3
                        || parameters.1 >= parameter_definitions.len() / 3
                    {
                        return Err(PyValueError::new_err("Invalid graph FM operator"));
                    }
                    Ok(GraphOperator {
                        waveform,
                        initial,
                        attack: attack
                            .into_iter()
                            .map(|(frames, target)| Segment { frames, target })
                            .collect(),
                        release: release
                            .into_iter()
                            .map(|(frames, target)| Segment { frames, target })
                            .collect(),
                        phase_offset,
                        ratio_parameter: parameters.0,
                        tuning_parameter: parameters.1,
                    })
                },
            )
            .collect::<PyResult<_>>()?;
        let graph_edges: Vec<GraphEdge> = edges
            .into_iter()
            .map(|(source, destination, delayed, parameter)| {
                if source >= operators
                    || destination >= operators
                    || parameter >= parameter_definitions.len() / 3
                {
                    return Err(PyValueError::new_err("Invalid graph FM edge"));
                }
                if !delayed && positions[source] >= positions[destination] {
                    return Err(PyValueError::new_err("Invalid graph FM execution order"));
                }
                Ok(GraphEdge {
                    source,
                    destination,
                    delayed,
                    parameter,
                })
            })
            .collect::<PyResult<_>>()?;
        let mut runtime = Self::new(
            rate,
            0,
            0.5,
            routes,
            carrier_initial,
            carrier_attack,
            carrier_release,
            minimum_hold_frames,
            controls,
            control_smoothing,
            lfos,
            lfo_rationals,
            filters,
            parameter_definitions,
            context_capacity,
            filter_after_amplitude,
        )?;
        let slots = runtime.frequencies.len();
        runtime.source_kind = 3;
        runtime.graph_operators = graph_operators;
        runtime.graph_edges = graph_edges;
        runtime.graph_order = order;
        runtime.graph_carrier = carrier;
        runtime.graph_carrier_parameter = carrier_parameter;
        runtime.graph_phases = vec![0.0; slots * operators];
        runtime.graph_errors = vec![0.0; slots * operators];
        runtime.graph_outputs = vec![0.0; slots * operators];
        runtime.graph_history = vec![0.0; slots * runtime.graph_edges.len()];
        runtime.graph_release_levels = vec![0.0; slots * operators];
        Ok(runtime)
    }

    fn set_named_envelopes(
        &mut self,
        initials: Vec<f64>,
        attacks: Vec<Vec<(f64, f64)>>,
        releases: Vec<Vec<(f64, f64)>>,
        parameters: Vec<NamedContourInput>,
    ) -> PyResult<()> {
        let count = initials.len();
        if attacks.len() != count || releases.len() != count || parameters.len() != count {
            return Err(PyValueError::new_err("Invalid named envelope definitions"));
        }
        let definitions: Vec<NamedEnvelopeDefinition> = initials
            .into_iter()
            .zip(attacks)
            .zip(releases)
            .zip(parameters)
            .map(
                |(
                    ((initial, attack), release),
                    (
                        parameter,
                        operation,
                        intercept,
                        slope,
                        release_with_voice,
                        playback,
                        loop_start,
                        loop_end,
                        repeat_count,
                        event_started,
                    ),
                )| {
                    if !initial.is_finite()
                        || attack.iter().chain(&release).any(|(frames, target)| {
                            !frames.is_finite() || *frames < 0.0 || !target.is_finite()
                        })
                        || parameter >= self.parameter_definitions.len() / 3
                        || operation > 1
                        || playback > 2
                        || !(0.0..1.0).contains(&loop_start)
                        || !(loop_start..=1.0).contains(&loop_end)
                        || loop_start == loop_end
                        || repeat_count.is_some_and(|count| count == 0 || playback == 0)
                        || !intercept.is_finite()
                        || !slope.is_finite()
                    {
                        return Err(PyValueError::new_err("Invalid named envelope definition"));
                    }
                    Ok(NamedEnvelopeDefinition {
                        sample_hold: None,
                        initial,
                        attack: attack
                            .into_iter()
                            .map(|(frames, target)| Segment { frames, target })
                            .collect(),
                        release: release
                            .into_iter()
                            .map(|(frames, target)| Segment { frames, target })
                            .collect(),
                        parameter,
                        operation,
                        intercept,
                        slope,
                        release_with_voice,
                        playback,
                        loop_start,
                        loop_end,
                        repeat_count,
                        beat_clock: false,
                        event_started,
                    })
                },
            )
            .collect::<PyResult<_>>()?;
        let states = self.frequencies.len() * count;
        self.named_envelopes = definitions;
        self.named_owners = (0..count).collect();
        self.named_states = self
            .named_envelopes
            .iter()
            .cycle()
            .take(states)
            .map(|definition| named_contour_initial(definition, 0.0))
            .collect();
        Ok(())
    }

    fn set_beat_clock(&mut self, points: Vec<(f64, f64, f64, f64)>) -> PyResult<()> {
        if self.frame != 0
            || points.is_empty()
            || points[0].0 != 0.0
            || points.iter().any(|(frame, beat, song_beat, rate)| {
                !frame.is_finite()
                    || !beat.is_finite()
                    || !song_beat.is_finite()
                    || !rate.is_finite()
                    || *beat < 0.0
                    || *rate < 0.0
            })
            || points.windows(2).any(|pair| pair[1].0 <= pair[0].0)
        {
            return Err(PyValueError::new_err("Invalid native beat clock"));
        }
        self.beat_points = points
            .into_iter()
            .map(|(frame, beat, song_beat, rate)| BeatPoint {
                frame,
                beat,
                song_beat,
                rate,
            })
            .collect();
        Ok(())
    }

    fn set_beat_named_envelopes(&mut self, indices: Vec<usize>) -> PyResult<()> {
        if self.frame != 0
            || self.beat_points.is_empty()
            || indices
                .iter()
                .any(|index| *index >= self.named_envelopes.len())
        {
            return Err(PyValueError::new_err("Invalid beat-clock named Motion"));
        }
        for index in indices {
            self.named_envelopes[index].beat_clock = true;
        }
        Ok(())
    }

    fn set_staged_motions(
        &mut self,
        inputs: Vec<StagedMotionInput>,
        connections: Vec<StageConnectionInput>,
    ) -> PyResult<()> {
        let mut definitions = Vec::with_capacity(inputs.len());
        for (
            stages,
            transitions,
            initial_stage,
            parameter,
            operation,
            intercept,
            slope,
            beat_clock,
        ) in inputs
        {
            if stages.is_empty()
                || initial_stage >= stages.len()
                || (beat_clock && self.beat_points.is_empty())
                || parameter.is_some_and(|p| p >= self.parameter_definitions.len() / 3)
                || operation > 1
                || !intercept.is_finite()
                || !slope.is_finite()
                || transitions.iter().any(|(source, event, target)| {
                    *source >= stages.len() || event.is_empty() || *target > stages.len()
                })
            {
                return Err(PyValueError::new_err("Invalid staged Motion definition"));
            }
            let stages: Vec<StageDefinition> = stages
                .into_iter()
                .map(
                    |(
                        kind,
                        initial,
                        current_initial,
                        segments,
                        cycle,
                        markers,
                        playback,
                        loop_start,
                        loop_end,
                        repeat_count,
                    )| {
                        if kind > 2
                            || playback > 2
                            || (kind != 1 && playback != 0)
                            || !(0.0..1.0).contains(&loop_start)
                            || !(loop_start..=1.0).contains(&loop_end)
                            || loop_start == loop_end
                            || repeat_count
                                .is_some_and(|count| count == 0 || kind != 1 || playback == 0)
                            || !initial.is_finite()
                            || (kind != 1 && current_initial)
                            || segments.iter().any(|(frames, target)| {
                                !frames.is_finite() || *frames < 0.0 || !target.is_finite()
                            })
                            || (kind == 1
                                && segments.iter().map(|(frames, _)| frames).sum::<f64>() <= 0.0)
                            || (kind != 1 && !segments.is_empty())
                            || (kind == 2
                                && (cycle.len() != 8
                                    || cycle.iter().any(|value| !value.is_finite())
                                    || !(0.0..=2.0).contains(&cycle[0])
                                    || cycle[0].fract() != 0.0
                                    || cycle[1] < 0.0
                                    || !(0.0..1.0).contains(&cycle[2])
                                    || !(0.0..=1.0).contains(&cycle[3])
                                    || cycle[4] < 0.0
                                    || cycle[5] < 0.0
                                    || cycle[7] < 0.0
                                    || cycle[6] - cycle[7] < -1.0
                                    || cycle[6] + cycle[7] > 1.0))
                            || (kind != 2 && !cycle.is_empty())
                            || (kind == 0 && !markers.is_empty())
                            || markers.iter().any(|(position, name)| {
                                !position.is_finite()
                                    || !(0.0..=1.0).contains(position)
                                    || (kind == 2 && *position == 1.0)
                                    || name.is_empty()
                            })
                        {
                            return Err(PyValueError::new_err("Invalid staged Motion stage"));
                        }
                        Ok(StageDefinition {
                            kind,
                            playback,
                            loop_start,
                            loop_end,
                            repeat_count,
                            initial,
                            current_initial,
                            segments: segments
                                .into_iter()
                                .map(|(frames, target)| Segment { frames, target })
                                .collect(),
                            cycle,
                            markers,
                        })
                    },
                )
                .collect::<PyResult<_>>()?;
            if stages[initial_stage].current_initial {
                return Err(PyValueError::new_err(
                    "Initial staged Motion stage cannot capture current",
                ));
            }
            definitions.push(StagedMotionDefinition {
                stages,
                transitions,
                initial_stage,
                parameter,
                operation,
                intercept,
                slope,
                beat_clock,
            });
        }
        if connections.iter().any(
            |(source, port, destination, cue, every, _, probability, _, delay)| {
                *source >= definitions.len()
                    || *destination >= definitions.len()
                    || port.is_empty()
                    || cue.is_empty()
                    || *every == 0
                    || *probability > (1 << 53)
                    || !delay.is_finite()
                    || *delay < 0.0
            },
        ) {
            return Err(PyValueError::new_err("Invalid staged Motion connection"));
        }
        self.staged_states = (0..self.frequencies.len())
            .flat_map(|_| {
                definitions.iter().map(|definition| StagedMotionState {
                    stage: definition.initial_stage,
                    playback: stage_playback(
                        &definition.stages[definition.initial_stage],
                        0.0,
                        false,
                        1.0,
                    ),
                    entry_value: definition.stages[definition.initial_stage].initial,
                    completed: false,
                    complete_value: 0.0,
                    cursor_at: 0.0,
                    cursor_order: usize::MAX,
                    traversals: 0,
                })
            })
            .collect();
        self.staged_motions = definitions;
        self.staged_owners = (0..self.staged_motions.len()).collect();
        self.staged_connections = connections
            .into_iter()
            .map(
                |(source, port, destination, cue, every, offset, probability, key, delay)| {
                    StagedConnection {
                        source,
                        port,
                        destination,
                        cue,
                        every,
                        offset,
                        probability,
                        key,
                        delay,
                    }
                },
            )
            .collect();
        self.staged_pending.clear();
        self.reset_event_counts();
        Ok(())
    }

    fn set_sample_hold_motions(
        &mut self,
        definitions: Vec<(usize, u64, f64, f64)>,
    ) -> PyResult<()> {
        if self.frame != 0
            || definitions.iter().any(|(source, _, minimum, maximum)| {
                *source >= self.named_envelopes.len()
                    || !minimum.is_finite()
                    || !maximum.is_finite()
                    || !(-1.0..=1.0).contains(minimum)
                    || !(-1.0..=1.0).contains(maximum)
                    || minimum > maximum
            })
        {
            return Err(PyValueError::new_err("Invalid sample-and-hold definition"));
        }
        for (source, key, minimum, maximum) in definitions {
            self.named_envelopes[source].sample_hold = Some((key, minimum, maximum));
        }
        Ok(())
    }

    fn set_motion_state_owners(
        &mut self,
        lfos: Vec<usize>,
        envelopes: Vec<usize>,
        stages: Vec<usize>,
    ) -> PyResult<()> {
        if self.frame != 0
            || lfos.len() != self.lfo_definitions.len()
            || envelopes.len() != self.named_envelopes.len()
            || stages.len() != self.staged_motions.len()
            || !valid_motion_owners(&lfos)
            || !valid_motion_owners(&envelopes)
            || !valid_motion_owners(&stages)
        {
            return Err(PyValueError::new_err("Invalid shared Motion state owners"));
        }
        self.lfo_owners = lfos;
        self.named_owners = envelopes;
        self.staged_owners = stages;
        Ok(())
    }

    fn set_motion_transforms(
        &mut self,
        nodes: Vec<MotionTransformInput>,
        routes: Vec<MotionTransformRoute>,
    ) -> PyResult<()> {
        if self.frame != 0
            || nodes
                .iter()
                .enumerate()
                .any(|(node, (kind, inputs, scale, offset, initial))| {
                    *kind > 4
                        || !scale.is_finite()
                        || !offset.is_finite()
                        || inputs.len() < if *kind >= 2 { 1 } else { 2 }
                        || (*kind >= 2 && inputs.len() != 1)
                        || (*kind == 3 && scale >= offset)
                        || (*kind == 4 && (*scale < 0.0 || *offset < 0.0))
                        || initial.is_some_and(|value| *kind != 4 || !value.is_finite())
                        || inputs.iter().any(|(kind, index, offset, scale)| {
                            !offset.is_finite()
                                || !scale.is_finite()
                                || match kind {
                                    0 => *index >= self.lfo_definitions.len(),
                                    1 => *index >= self.named_envelopes.len(),
                                    2 => *index >= self.staged_motions.len(),
                                    3 => *index >= node,
                                    _ => true,
                                }
                        })
                })
            || routes
                .iter()
                .any(|(node, parameter, operation, intercept, slope)| {
                    *node >= nodes.len()
                        || *parameter >= self.parameter_definitions.len() / 3
                        || *operation > 1
                        || !intercept.is_finite()
                        || !slope.is_finite()
                })
        {
            return Err(PyValueError::new_err("Invalid Motion transform graph"));
        }
        self.motion_transform_values = vec![0.0; self.frequencies.len() * nodes.len()];
        self.motion_transform_states = vec![None; self.motion_transform_values.len()];
        self.motion_transforms = nodes;
        self.motion_transform_routes = routes;
        Ok(())
    }

    fn set_patch_events(&mut self, connections: Vec<PatchEventInput>) -> PyResult<()> {
        if self.frame != 0
            || connections.iter().any(
                |(source, marker, destination, cue, _, every, _, probability, _, delay)| {
                    *every == 0
                        || *probability > (1 << 53)
                        || !delay.is_finite()
                        || *delay < 0.0
                        || *source >= self.lfo_definitions.len()
                        || self.lfo_definitions[*source].scope != 3
                        || self.lfo_definitions[*source].rate.0 == 0
                        || self.lfo_definitions[*source].beat_clock
                        || self.lfo_definitions[*source].transport_position
                        || !marker.is_finite()
                        || !(0.0..1.0).contains(marker)
                        || if cue.is_some() {
                            *destination >= self.staged_motions.len()
                        } else {
                            *destination >= self.named_envelopes.len()
                                || !self.named_envelopes[*destination].event_started
                        }
                },
            )
        {
            return Err(PyValueError::new_err("Invalid Patch event connections"));
        }
        self.patch_events = connections
            .into_iter()
            .map(
                |(
                    source,
                    marker,
                    destination,
                    cue,
                    order,
                    every,
                    offset,
                    probability,
                    key,
                    delay,
                )| {
                    PatchEventConnection {
                        threshold: false,
                        source: self.lfo_owners[source],
                        marker,
                        destination: if cue.is_some() {
                            self.staged_owners[destination]
                        } else {
                            self.named_owners[destination]
                        },
                        cue,
                        order,
                        every,
                        offset,
                        probability,
                        key,
                        delay,
                    }
                },
            )
            .collect();
        self.reset_event_counts();
        Ok(())
    }

    fn set_patch_stage_starts(&mut self, connections: Vec<StageStartInput>) -> PyResult<()> {
        if self.frame != 0
            || connections.iter().any(
                |(source, port, destination, _, every, _, probability, _, delay)| {
                    *source >= self.staged_motions.len()
                        || *every == 0
                        || *probability > (1 << 53)
                        || !delay.is_finite()
                        || *delay < 0.0
                        || port.is_empty()
                        || *destination >= self.named_envelopes.len()
                        || !self.named_envelopes[*destination].event_started
                },
            )
        {
            return Err(PyValueError::new_err(
                "Invalid Patch stage start connections",
            ));
        }
        self.patch_stage_starts = connections
            .into_iter()
            .map(
                |(source, port, destination, order, every, offset, probability, key, delay)| {
                    PatchStageStartConnection {
                        source: self.staged_owners[source],
                        port,
                        destination: self.named_owners[destination],
                        order,
                        every,
                        offset,
                        probability,
                        key,
                        delay,
                    }
                },
            )
            .collect();
        self.reset_event_counts();
        Ok(())
    }

    fn set_threshold_events(&mut self, connections: Vec<PatchEventInput>) -> PyResult<()> {
        if self.frame != 0
            || connections.iter().any(
                |(source, marker, destination, cue, _, every, _, probability, _, delay)| {
                    *source >= self.motion_transforms.len()
                        || self.motion_transforms[*source].0 != 3
                        || (*marker != 0.0 && *marker != 1.0)
                        || *every == 0
                        || *probability > (1 << 53)
                        || !delay.is_finite()
                        || *delay < 1.0
                        || if cue.is_some() {
                            *destination >= self.staged_motions.len()
                        } else {
                            *destination >= self.named_envelopes.len()
                                || !self.named_envelopes[*destination].event_started
                        }
                },
            )
        {
            return Err(PyValueError::new_err("Invalid threshold event connections"));
        }
        self.patch_events.extend(connections.into_iter().map(
            |(source, marker, destination, cue, order, every, offset, probability, key, delay)| {
                PatchEventConnection {
                    threshold: true,
                    source,
                    marker,
                    destination: if cue.is_some() {
                        self.staged_owners[destination]
                    } else {
                        self.named_owners[destination]
                    },
                    cue,
                    order,
                    every,
                    offset,
                    probability,
                    key,
                    delay,
                }
            },
        ));
        self.reset_event_counts();
        Ok(())
    }

    #[staticmethod]
    #[allow(clippy::too_many_arguments)]
    #[pyo3(signature = (rate, routes, initial, attack, release, minimum_hold_frames, controls, control_smoothing, lfos, lfo_rationals, filters, parameters, context_capacity, filter_after_amplitude=false))]
    fn noise(
        rate: f64,
        routes: PyReadonlyArray2<'_, f64>,
        initial: f64,
        attack: PyReadonlyArray2<'_, f64>,
        release: PyReadonlyArray2<'_, f64>,
        minimum_hold_frames: f64,
        controls: PyReadonlyArray2<'_, f64>,
        control_smoothing: Vec<(u64, u64)>,
        lfos: PyReadonlyArray2<'_, f64>,
        lfo_rationals: Vec<(u64, u64)>,
        filters: PyReadonlyArray2<'_, f64>,
        parameters: Vec<f64>,
        context_capacity: usize,
        filter_after_amplitude: bool,
    ) -> PyResult<Self> {
        let mut runtime = Self::new(
            rate,
            0,
            0.5,
            routes,
            initial,
            attack,
            release,
            minimum_hold_frames,
            controls,
            control_smoothing,
            lfos,
            lfo_rationals,
            filters,
            parameters,
            context_capacity,
            filter_after_amplitude,
        )?;
        runtime.source_kind = 2;
        Ok(runtime)
    }

    fn process<'py>(
        &mut self,
        py: Python<'py>,
        frames: usize,
        cutoff_hz: f64,
        q: f64,
        gain_db: f64,
    ) -> PyResult<Bound<'py, PyArray2<f64>>> {
        self.render_owned(py, frames, cutoff_hz, q, gain_db, &[])
    }

    fn process_actions<'py>(
        &mut self,
        py: Python<'py>,
        frames: usize,
        cutoff_hz: f64,
        q: f64,
        gain_db: f64,
        actions: PyReadonlyArray2<'_, f64>,
    ) -> PyResult<Bound<'py, PyArray2<f64>>> {
        if actions.shape()[1] != 6 || actions.as_array().iter().any(|v| !v.is_finite()) {
            return Err(PyValueError::new_err("Invalid synth runtime actions"));
        }
        self.render_owned(py, frames, cutoff_hz, q, gain_db, actions.as_slice()?)
    }

    fn process_into(
        &mut self,
        mut output: PyReadwriteArray2<'_, f64>,
        cutoff_hz: f64,
        q: f64,
        gain_db: f64,
    ) -> PyResult<()> {
        if !output.is_c_contiguous() {
            return Err(PyValueError::new_err(
                "Synth runtime output must be C-contiguous",
            ));
        }
        self.render_into(output.as_array_mut(), cutoff_hz, q, gain_db, &[])
    }

    fn process_actions_into(
        &mut self,
        mut output: PyReadwriteArray2<'_, f64>,
        cutoff_hz: f64,
        q: f64,
        gain_db: f64,
        actions: PyReadonlyArray2<'_, f64>,
        action_count: usize,
    ) -> PyResult<()> {
        if !output.is_c_contiguous() || actions.shape()[1] != 6 || action_count > actions.shape()[0]
        {
            return Err(PyValueError::new_err(
                "Invalid synth runtime output or actions",
            ));
        }
        let actions = &actions.as_slice()?[..action_count * 6];
        if actions.iter().any(|v| !v.is_finite()) {
            return Err(PyValueError::new_err("Invalid synth runtime actions"));
        }
        self.render_into(output.as_array_mut(), cutoff_hz, q, gain_db, actions)
    }

    pub(crate) fn active_slots(&self) -> Vec<bool> {
        self.active.clone()
    }

    pub(crate) fn active_contexts(&self) -> Vec<bool> {
        self.context_kinds.iter().map(|kind| *kind != 0).collect()
    }

    pub(crate) fn snapshot(&self) -> SynthRuntimeSnapshot {
        SynthRuntimeSnapshot {
            motion_transform_states: self.motion_transform_states.clone(),
            motion_transforms: self.motion_transforms.clone(),
            motion_transform_routes: self.motion_transform_routes.clone(),
            rate: self.rate,
            waveform: self.waveform,
            source_kind: self.source_kind,
            duty: self.duty,
            mod_initial: self.mod_initial,
            mod_attack: self.mod_attack.clone(),
            mod_release: self.mod_release.clone(),
            fm_phase_offsets: self.fm_phase_offsets,
            graph_operators: self.graph_operators.clone(),
            graph_edges: self.graph_edges.clone(),
            graph_order: self.graph_order.clone(),
            graph_carrier: self.graph_carrier,
            graph_carrier_parameter: self.graph_carrier_parameter,
            initial: self.initial,
            attack: self.attack.clone(),
            release: self.release.clone(),
            minimum_hold_frames: self.minimum_hold_frames,
            routes: self.routes.clone(),
            control_definitions: self.control_definitions.clone(),
            control_states: self.control_states.clone(),
            lfo_definitions: self.lfo_definitions.clone(),
            lfo_owners: self.lfo_owners.clone(),
            lfo_event_states: self.lfo_event_states.clone(),
            voice_lfo_event_states: self.voice_lfo_event_states.clone(),
            named_envelopes: self.named_envelopes.clone(),
            named_owners: self.named_owners.clone(),
            patch_events: self.patch_events.clone(),
            patch_stage_starts: self.patch_stage_starts.clone(),
            pending_contour_starts: self.pending_contour_starts.clone(),
            delayed_events: self.delayed_events.clone(),
            event_counts: self.event_counts.clone(),
            motion_keys: self.motion_keys.clone(),
            event_random: self.event_random.clone(),
            beat_points: self.beat_points.clone(),
            staged_motions: self.staged_motions.clone(),
            staged_owners: self.staged_owners.clone(),
            staged_states: self.staged_states.clone(),
            staged_connections: self.staged_connections.clone(),
            staged_pending: self.staged_pending.clone(),
            filter_definitions: self.filter_definitions.clone(),
            context_kinds: self.context_kinds.clone(),
            context_states: self.context_states.clone(),
            parameter_definitions: self.parameter_definitions.clone(),
            frequencies: self.frequencies.clone(),
            phases: self.phases.clone(),
            errors: self.errors.clone(),
            mod_phases: self.mod_phases.clone(),
            mod_errors: self.mod_errors.clone(),
            previous_modulators: self.previous_modulators.clone(),
            mod_release_levels: self.mod_release_levels.clone(),
            named_states: self.named_states.clone(),
            graph_phases: self.graph_phases.clone(),
            graph_errors: self.graph_errors.clone(),
            graph_outputs: self.graph_outputs.clone(),
            graph_history: self.graph_history.clone(),
            graph_release_levels: self.graph_release_levels.clone(),
            noise_keys: self.noise_keys.clone(),
            noise_counters: self.noise_counters.clone(),
            gains: self.gains.clone(),
            active: self.active.clone(),
            ages: self.ages.clone(),
            release_frames: self.release_frames.clone(),
            release_levels: self.release_levels.clone(),
            frequency_steps: self.frequency_steps.clone(),
            frequency_remaining: self.frequency_remaining.clone(),
            gain_steps: self.gain_steps.clone(),
            gain_remaining: self.gain_remaining.clone(),
            voice_part_contexts: self.voice_part_contexts.clone(),
            voice_trigger_contexts: self.voice_trigger_contexts.clone(),
            filter_states: self.filter_states.clone(),
            voice_filter_states: self.voice_filter_states.clone(),
            frame: self.frame,
        }
    }

    pub(crate) fn restore(&mut self, snapshot: &SynthRuntimeSnapshot) -> PyResult<()> {
        self.validate_snapshot(snapshot)?;
        self.motion_transform_states
            .clone_from(&snapshot.motion_transform_states);
        self.frequencies.clone_from(&snapshot.frequencies);
        self.phases.clone_from(&snapshot.phases);
        self.errors.clone_from(&snapshot.errors);
        self.mod_phases.clone_from(&snapshot.mod_phases);
        self.mod_errors.clone_from(&snapshot.mod_errors);
        self.previous_modulators
            .clone_from(&snapshot.previous_modulators);
        self.mod_release_levels
            .clone_from(&snapshot.mod_release_levels);
        self.named_states.clone_from(&snapshot.named_states);
        self.staged_states.clone_from(&snapshot.staged_states);
        self.staged_pending.clone_from(&snapshot.staged_pending);
        self.pending_contour_starts
            .clone_from(&snapshot.pending_contour_starts);
        for (queue, saved) in self.delayed_events.iter_mut().zip(&snapshot.delayed_events) {
            queue.clone_from(saved);
        }
        self.event_counts.clone_from(&snapshot.event_counts);
        self.motion_keys.clone_from(&snapshot.motion_keys);
        self.event_random.clone_from(&snapshot.event_random);
        self.graph_phases.clone_from(&snapshot.graph_phases);
        self.graph_errors.clone_from(&snapshot.graph_errors);
        self.graph_outputs.clone_from(&snapshot.graph_outputs);
        self.graph_history.clone_from(&snapshot.graph_history);
        self.graph_release_levels
            .clone_from(&snapshot.graph_release_levels);
        self.noise_keys.clone_from(&snapshot.noise_keys);
        self.noise_counters.clone_from(&snapshot.noise_counters);
        self.gains.clone_from(&snapshot.gains);
        self.active.clone_from(&snapshot.active);
        self.ages.clone_from(&snapshot.ages);
        self.release_frames.clone_from(&snapshot.release_frames);
        self.release_levels.clone_from(&snapshot.release_levels);
        self.frequency_steps.clone_from(&snapshot.frequency_steps);
        self.frequency_remaining
            .clone_from(&snapshot.frequency_remaining);
        self.gain_steps.clone_from(&snapshot.gain_steps);
        self.gain_remaining.clone_from(&snapshot.gain_remaining);
        self.filter_states.clone_from(&snapshot.filter_states);
        self.voice_filter_states
            .clone_from(&snapshot.voice_filter_states);
        self.frame = snapshot.frame;
        self.control_states.clone_from(&snapshot.control_states);
        self.lfo_event_states.clone_from(&snapshot.lfo_event_states);
        self.voice_lfo_event_states
            .clone_from(&snapshot.voice_lfo_event_states);
        self.context_kinds.clone_from(&snapshot.context_kinds);
        self.context_states.clone_from(&snapshot.context_states);
        self.voice_part_contexts
            .clone_from(&snapshot.voice_part_contexts);
        self.voice_trigger_contexts
            .clone_from(&snapshot.voice_trigger_contexts);
        Ok(())
    }
}

impl SynthRuntime {
    fn reset_event_counts(&mut self) {
        let delayed = self
            .staged_connections
            .iter()
            .map(|c| c.delay)
            .chain(self.patch_events.iter().map(|c| c.delay))
            .chain(self.patch_stage_starts.iter().map(|c| c.delay))
            .any(|delay| delay > 0.0);
        if delayed {
            for queue in &mut self.delayed_events {
                if queue.capacity() < 4096 {
                    queue.reserve_exact(4096 - queue.len());
                }
            }
        }
        self.event_counts = vec![
            0;
            self.frequencies.len()
                * (self.staged_connections.len()
                    + self.patch_events.len()
                    + self.patch_stage_starts.len())
        ];
        self.event_random = vec![0; self.event_counts.len()];
        for voice in 0..self.frequencies.len() {
            self.reset_voice_event_counts(voice);
        }
    }

    fn reset_voice_event_counts(&mut self, voice: usize) {
        let nodes = self.motion_transforms.len();
        self.motion_transform_states[voice * nodes..(voice + 1) * nodes].fill(None);
        self.delayed_events[voice].clear();
        let connections =
            self.staged_connections.len() + self.patch_events.len() + self.patch_stage_starts.len();
        let offsets = self
            .staged_connections
            .iter()
            .map(|c| c.offset)
            .chain(self.patch_events.iter().map(|c| c.offset))
            .chain(self.patch_stage_starts.iter().map(|c| c.offset));
        for (count, offset) in self.event_counts[voice * connections..(voice + 1) * connections]
            .iter_mut()
            .zip(offsets)
        {
            *count = offset;
        }
        let keys = self
            .staged_connections
            .iter()
            .map(|c| c.key)
            .chain(self.patch_events.iter().map(|c| c.key))
            .chain(self.patch_stage_starts.iter().map(|c| c.key));
        for (state, key) in self.event_random[voice * connections..(voice + 1) * connections]
            .iter_mut()
            .zip(keys)
        {
            *state = self.motion_keys[voice] ^ key;
        }
    }

    pub(crate) fn validate_snapshot(&self, snapshot: &SynthRuntimeSnapshot) -> PyResult<()> {
        if self.rate != snapshot.rate
            || self.waveform != snapshot.waveform
            || self.source_kind != snapshot.source_kind
            || self.duty != snapshot.duty
            || self.mod_initial != snapshot.mod_initial
            || self.mod_attack != snapshot.mod_attack
            || self.mod_release != snapshot.mod_release
            || self.fm_phase_offsets != snapshot.fm_phase_offsets
            || self.graph_operators != snapshot.graph_operators
            || self.graph_edges != snapshot.graph_edges
            || self.graph_order != snapshot.graph_order
            || self.graph_carrier != snapshot.graph_carrier
            || self.graph_carrier_parameter != snapshot.graph_carrier_parameter
            || self.initial != snapshot.initial
            || self.attack != snapshot.attack
            || self.release != snapshot.release
            || self.minimum_hold_frames != snapshot.minimum_hold_frames
            || self.routes != snapshot.routes
            || self.control_definitions != snapshot.control_definitions
            || self.lfo_definitions != snapshot.lfo_definitions
            || self.lfo_owners != snapshot.lfo_owners
            || self.named_envelopes != snapshot.named_envelopes
            || self.named_owners != snapshot.named_owners
            || self.motion_transforms != snapshot.motion_transforms
            || self.motion_transform_routes != snapshot.motion_transform_routes
            || self.patch_events != snapshot.patch_events
            || self.patch_stage_starts != snapshot.patch_stage_starts
            || self.beat_points != snapshot.beat_points
            || self.staged_motions != snapshot.staged_motions
            || self.staged_owners != snapshot.staged_owners
            || self.staged_connections != snapshot.staged_connections
            || self.filter_definitions != snapshot.filter_definitions
            || self.parameter_definitions != snapshot.parameter_definitions
            || self.context_kinds.len() != snapshot.context_kinds.len()
        {
            return Err(PyValueError::new_err(
                "Snapshot belongs to a different synth runtime",
            ));
        }
        Ok(())
    }

    fn local_beat_at(&self, frame: f64) -> f64 {
        let index = self
            .beat_points
            .partition_point(|point| point.frame <= frame);
        let point = &self.beat_points[index - 1];
        point.beat + (frame - point.frame) * point.rate
    }

    fn song_beat_at(&self, frame: f64) -> f64 {
        let index = self
            .beat_points
            .partition_point(|point| point.frame <= frame);
        let point = &self.beat_points[index - 1];
        point.song_beat + (frame - point.frame) * point.rate
    }

    fn frame_at_local_beat(&self, beat: f64) -> f64 {
        for (index, point) in self.beat_points.iter().enumerate() {
            let next = self.beat_points.get(index + 1);
            if point.rate > 0.0 && beat >= point.beat {
                let frame = point.frame + (beat - point.beat) / point.rate;
                if next.is_none_or(|following| frame <= following.frame) {
                    return frame;
                }
            }
        }
        f64::INFINITY
    }

    fn staged_time(&self, definition: &StagedMotionDefinition, frame: f64) -> f64 {
        if definition.beat_clock {
            self.local_beat_at(frame)
        } else {
            frame
        }
    }

    fn named_time(&self, definition: &NamedEnvelopeDefinition, frame: f64) -> f64 {
        if definition.beat_clock {
            self.local_beat_at(frame)
        } else {
            frame
        }
    }

    fn lfo_elapsed(&self, definition: &LfoDefinition, from: usize, to: usize) -> f64 {
        if definition.beat_clock {
            self.local_beat_at(to as f64) - self.local_beat_at(from as f64)
        } else {
            (to - from) as f64
        }
    }

    fn lfo_divisor(&self, definition: &LfoDefinition) -> f64 {
        if definition.beat_clock {
            1.0
        } else {
            self.rate
        }
    }

    pub(crate) fn channels(&self) -> usize {
        self.routes.ncols()
    }

    pub(crate) fn rate(&self) -> f64 {
        self.rate
    }

    pub(crate) fn frame(&self) -> usize {
        self.frame
    }

    fn render_owned<'py>(
        &mut self,
        py: Python<'py>,
        frames: usize,
        cutoff_hz: f64,
        q: f64,
        gain_db: f64,
        actions: &[f64],
    ) -> PyResult<Bound<'py, PyArray2<f64>>> {
        let mut output = Array2::zeros((frames, self.routes.ncols()));
        self.render_into(output.view_mut(), cutoff_hz, q, gain_db, actions)?;
        Ok(output.into_pyarray(py))
    }

    pub(crate) fn render_into(
        &mut self,
        mut output: ArrayViewMut2<'_, f64>,
        cutoff_hz: f64,
        q: f64,
        gain_db: f64,
        actions: &[f64],
    ) -> PyResult<()> {
        let frames = output.nrows();
        if frames == 0
            || output.ncols() != self.routes.ncols()
            || (self.beat_points.is_empty()
                && self
                    .lfo_definitions
                    .iter()
                    .any(|definition| definition.beat_clock))
            || !cutoff_hz.is_finite()
            || cutoff_hz < 0.0
            || cutoff_hz >= self.rate / 2.0
            || !q.is_finite()
            || q <= 0.0
            || !gain_db.is_finite()
            || actions.len() % 6 != 0
            || actions.chunks_exact(6).any(|a| {
                let offset = a[0] as usize;
                a[0] != offset as f64 || offset >= frames
            })
            || actions
                .chunks_exact(6)
                .zip(actions.chunks_exact(6).skip(1))
                .any(|(a, b)| b[0] < a[0])
        {
            return Err(PyValueError::new_err("Invalid synth runtime block"));
        }
        let channels = self.routes.ncols();
        let gain = 10_f64.powf(gain_db / 20.0);
        let filter = cutoff_hz > 0.0;
        let g = (PI * cutoff_hz / self.rate).tan();
        let a1 = if q < 1.0 {
            q / (q * (1.0 + g * g) + g)
        } else {
            1.0 / (1.0 + g * (g + 1.0 / q))
        };
        let a2 = g * a1;
        let a3 = g * a2;
        output.fill(0.0);
        let mut action = 0;
        for frame in 0..frames {
            self.advance_patch_events(false)?;
            self.advance_staged(self.frame as f64, false, true)?;
            while action < actions.len() && actions[action] as usize == frame {
                self.apply_action(&actions[action..action + 6], frames)?;
                action += 6;
            }
            self.dispatch_staged_events()?;
            self.advance_patch_events(true)?;
            self.advance_staged(self.frame as f64, true, true)?;
            self.dispatch_staged_events()?;
            self.flush_contour_starts()?;
            for voice in 0..self.frequencies.len() {
                if !self.active[voice] {
                    continue;
                }
                self.evaluate_motion_transforms(voice)?;
                let amplitude = self.parameter(voice, 0)?
                    * if self.source_kind == 3 {
                        self.parameter(voice, self.graph_carrier_parameter)?
                    } else if self.source_kind == 1 {
                        self.parameter(voice, 8)?
                    } else {
                        1.0
                    };
                let age = self.ages[voice] as f64;
                let level = if let Some(release_frame) = self.release_frames[voice] {
                    let end = release_frame + total_frames(&self.release);
                    if age >= end.ceil() {
                        self.active[voice] = false;
                        self.delayed_events[voice].clear();
                        continue;
                    }
                    if age >= release_frame.ceil() {
                        envelope_value(
                            self.release_levels[voice],
                            &self.release,
                            age - release_frame,
                        )
                    } else {
                        envelope_value(self.initial, &self.attack, age)
                    }
                } else {
                    envelope_value(self.initial, &self.attack, age)
                };
                let wave = if self.source_kind == 0 {
                    let angle = TAU * self.phases[voice] / self.rate;
                    let phase = (angle / TAU).rem_euclid(1.0);
                    match self.waveform {
                        0 => angle.sin(),
                        1 => {
                            if phase < self.duty {
                                1.0
                            } else {
                                -1.0
                            }
                        }
                        _ if self.duty == 0.0 => 1.0 - 2.0 * phase,
                        _ if self.duty == 1.0 => 2.0 * phase - 1.0,
                        _ if phase < self.duty => 2.0 * phase / self.duty - 1.0,
                        _ => (1.0 + self.duty - 2.0 * phase) / (1.0 - self.duty),
                    }
                } else if self.source_kind == 1 {
                    let mod_level = if let Some(release_frame) = self.release_frames[voice] {
                        let end = release_frame + total_frames(&self.mod_release);
                        if age >= end.ceil() {
                            0.0
                        } else if age >= release_frame.ceil() {
                            envelope_value(
                                self.mod_release_levels[voice],
                                &self.mod_release,
                                age - release_frame,
                            )
                        } else {
                            envelope_value(self.mod_initial, &self.mod_attack, age)
                        }
                    } else {
                        envelope_value(self.mod_initial, &self.mod_attack, age)
                    };
                    let process_tuning = self.parameter(voice, 1)?;
                    let mod_frequency = self.frequencies[voice]
                        * self.parameter(voice, 2)?
                        * 2_f64.powf((process_tuning + self.parameter(voice, 3)?) / 1200.0);
                    let carrier_frequency = self.frequencies[voice]
                        * self.parameter(voice, 4)?
                        * 2_f64.powf((process_tuning + self.parameter(voice, 5)?) / 1200.0);
                    let modulator = mod_level
                        * (TAU * self.mod_phases[voice] / self.rate
                            + self.parameter(voice, 7)? * self.previous_modulators[voice])
                            .sin();
                    let carrier = (TAU * self.phases[voice] / self.rate
                        + self.parameter(voice, 6)? * modulator)
                        .sin();
                    self.previous_modulators[voice] = modulator;
                    let increment = mod_frequency - self.mod_errors[voice];
                    let total = self.mod_phases[voice] + increment;
                    self.mod_errors[voice] = (total - self.mod_phases[voice]) - increment;
                    self.mod_phases[voice] = total.rem_euclid(self.rate);
                    let increment = carrier_frequency - self.errors[voice];
                    let total = self.phases[voice] + increment;
                    self.errors[voice] = (total - self.phases[voice]) - increment;
                    self.phases[voice] = total.rem_euclid(self.rate);
                    carrier
                } else if self.source_kind == 3 {
                    let operators = self.graph_operators.len();
                    let edges = self.graph_edges.len();
                    let operator_base = voice * operators;
                    let mut carrier_sample = 0.0;
                    let edge_base = voice * edges;
                    let process_tuning = self.parameter(voice, 1)?;
                    for operator in &self.graph_order {
                        let definition = &self.graph_operators[*operator];
                        let level = if let Some(release_frame) = self.release_frames[voice] {
                            let end = release_frame + total_frames(&definition.release);
                            if age >= end.ceil() {
                                0.0
                            } else if age >= release_frame.ceil() {
                                envelope_value(
                                    self.graph_release_levels[operator_base + *operator],
                                    &definition.release,
                                    age - release_frame,
                                )
                            } else {
                                envelope_value(definition.initial, &definition.attack, age)
                            }
                        } else {
                            envelope_value(definition.initial, &definition.attack, age)
                        };
                        let mut offset = 0.0;
                        for (index, edge) in self.graph_edges.iter().enumerate() {
                            if edge.destination == *operator {
                                offset += self.parameter(voice, edge.parameter)?
                                    * if edge.delayed {
                                        self.graph_history[edge_base + index]
                                    } else {
                                        self.graph_outputs[operator_base + edge.source]
                                    };
                            }
                        }
                        let angle =
                            TAU * self.graph_phases[operator_base + *operator] / self.rate + offset;
                        let phase = (angle / TAU).rem_euclid(1.0);
                        let value = match definition.waveform {
                            0 => angle.sin(),
                            1 => {
                                if phase < 0.5 {
                                    1.0
                                } else {
                                    -1.0
                                }
                            }
                            _ => {
                                if phase < 0.5 {
                                    4.0 * phase - 1.0
                                } else {
                                    3.0 - 4.0 * phase
                                }
                            }
                        };
                        self.graph_outputs[operator_base + *operator] = level * value;
                        if *operator == self.graph_carrier {
                            carrier_sample = value;
                        }
                    }
                    if let Some(release_frame) = self.release_frames[voice] {
                        let carrier = &self.graph_operators[self.graph_carrier];
                        if age >= (release_frame + total_frames(&carrier.release)).ceil() {
                            self.active[voice] = false;
                            self.delayed_events[voice].clear();
                            continue;
                        }
                    }
                    let carrier = carrier_sample;
                    for (index, edge) in self.graph_edges.iter().enumerate() {
                        self.graph_history[edge_base + index] =
                            self.graph_outputs[operator_base + edge.source];
                    }
                    for operator in 0..operators {
                        let definition = &self.graph_operators[operator];
                        let frequency = self.frequencies[voice]
                            * self.parameter(voice, definition.ratio_parameter)?
                            * 2_f64.powf(
                                (process_tuning
                                    + self.parameter(voice, definition.tuning_parameter)?)
                                    / 1200.0,
                            );
                        let index = operator_base + operator;
                        let increment = frequency - self.graph_errors[index];
                        let total = self.graph_phases[index] + increment;
                        self.graph_errors[index] = (total - self.graph_phases[index]) - increment;
                        self.graph_phases[index] = total.rem_euclid(self.rate);
                    }
                    carrier
                } else {
                    let index = self.noise_counters[voice];
                    let mut word = self.noise_keys[voice]
                        .wrapping_add(index.wrapping_add(1).wrapping_mul(0x9e3779b97f4a7c15));
                    word = (word ^ (word >> 30)).wrapping_mul(0xbf58476d1ce4e5b9);
                    word = (word ^ (word >> 27)).wrapping_mul(0x94d049bb133111eb);
                    word ^= word >> 31;
                    self.noise_counters[voice] = index.checked_add(1).ok_or_else(|| {
                        PyValueError::new_err("Persistent noise counter exhausted")
                    })?;
                    2.0 * ((word >> 11) as f64 / 9007199254740992.0) - 1.0
                };
                let sample = if self.filter_after_amplitude {
                    self.filter_voice(voice, wave * self.gains[voice] * level * amplitude)?
                } else {
                    self.filter_voice(voice, wave)? * self.gains[voice] * level * amplitude
                };
                if self.source_kind == 0 {
                    let tuning = 2_f64.powf(self.parameter(voice, 1)? / 1200.0);
                    let increment = self.frequencies[voice] * tuning - self.errors[voice];
                    let total = self.phases[voice] + increment;
                    self.errors[voice] = (total - self.phases[voice]) - increment;
                    self.phases[voice] = total.rem_euclid(self.rate);
                }
                for channel in 0..channels {
                    output[[frame, channel]] += sample * self.routes[[voice, channel]];
                }
                self.ages[voice] += 1;
                advance_ramp(
                    &mut self.frequencies[voice],
                    self.frequency_steps[voice],
                    &mut self.frequency_remaining[voice],
                );
                advance_ramp(
                    &mut self.gains[voice],
                    self.gain_steps[voice],
                    &mut self.gain_remaining[voice],
                );
            }
            for (source, state) in self.control_states.iter_mut().enumerate() {
                if self.control_definitions[source].scope == 0 {
                    state.elapsed = state.elapsed.saturating_add(1);
                }
            }
            for context in 0..self.context_kinds.len() {
                let kind = self.context_kinds[context];
                if kind == 0 {
                    continue;
                }
                for source in 0..self.control_definitions.len() {
                    if self.control_definitions[source].scope == kind {
                        let state = &mut self.context_states
                            [context * self.control_definitions.len() + source];
                        state.elapsed = state.elapsed.saturating_add(1);
                    }
                }
            }
            for channel in 0..channels {
                if filter {
                    let input = output[[frame, channel]];
                    let s1 = self.filter_states[[channel, 0]];
                    let s2 = self.filter_states[[channel, 1]];
                    let v3 = input - s2;
                    let v1 = a1 * s1 + a2 * v3;
                    let v2 = s2 + a2 * s1 + a3 * v3;
                    self.filter_states[[channel, 0]] = 2.0 * v1 - s1;
                    self.filter_states[[channel, 1]] = 2.0 * v2 - s2;
                    output[[frame, channel]] = v2 * gain;
                } else {
                    output[[frame, channel]] *= gain;
                }
            }
            self.frame = self
                .frame
                .checked_add(1)
                .ok_or_else(|| PyValueError::new_err("Synth runtime frame overflow"))?;
        }
        self.retire_trigger_contexts();
        Ok(())
    }

    fn apply_action(&mut self, action: &[f64], frames: usize) -> PyResult<()> {
        let offset = action[0] as usize;
        let kind = action[1] as usize;
        let voice = action[2] as usize;
        if action[0] != offset as f64
            || offset >= frames
            || action[1] != kind as f64
            || action[2] != voice as f64
        {
            return Err(PyValueError::new_err("Invalid synth runtime action"));
        }
        if (kind <= 3 || kind == 5 || (9..=12).contains(&kind)) && voice >= self.frequencies.len() {
            return Err(PyValueError::new_err("Invalid synth runtime voice"));
        }
        match kind {
            12 => {
                if action[3] != action[3] as u32 as f64
                    || action[4] != action[4] as u32 as f64
                    || action[5] != 0.0
                {
                    return Err(PyValueError::new_err("Invalid Motion stream key"));
                }
                self.motion_keys[voice] = action[3] as u64 | ((action[4] as u64) << 32);
            }
            0 => {
                if self.source_kind != 2 && (action[3] <= 0.0 || action[4] < 0.0)
                    || self.source_kind == 2
                        && (action[3] != action[3] as u32 as f64
                            || action[4] != action[4] as u32 as f64
                            || action[5] < 0.0)
                    || self.active[voice]
                {
                    return Err(PyValueError::new_err("Invalid voice start"));
                }
                self.active[voice] = true;
                self.reset_voice_event_counts(voice);
                let lfo_base = voice * self.lfo_definitions.len();
                self.voice_lfo_event_states[lfo_base..lfo_base + self.lfo_definitions.len()]
                    .fill(None);
                for (source, definition) in self.staged_motions.iter().enumerate() {
                    let index = voice * self.staged_motions.len() + source;
                    let at = self.staged_time(definition, self.frame as f64);
                    self.staged_states[index] = StagedMotionState {
                        stage: definition.initial_stage,
                        playback: stage_playback(
                            &definition.stages[definition.initial_stage],
                            at,
                            false,
                            1.0,
                        ),
                        entry_value: definition.stages[definition.initial_stage].initial,
                        completed: false,
                        complete_value: 0.0,
                        cursor_at: at,
                        cursor_order: usize::MAX,
                        traversals: 0,
                    };
                    if staged_transition(definition, &mut self.staged_states[index], "note_on", at)
                    {
                        self.staged_pending.push_back(StagedEvent {
                            at: self.frame as f64,
                            voice,
                            source,
                            port: "done".to_owned(),
                        });
                    }
                }
                self.phases[voice] = action[5].rem_euclid(self.rate);
                self.errors[voice] = 0.0;
                self.mod_phases[voice] =
                    (self.fm_phase_offsets[0] * self.rate).rem_euclid(self.rate);
                if self.source_kind == 1 {
                    self.phases[voice] =
                        (self.fm_phase_offsets[1] * self.rate).rem_euclid(self.rate);
                }
                self.mod_errors[voice] = 0.0;
                self.previous_modulators[voice] = 0.0;
                if self.source_kind == 3 {
                    let operators = self.graph_operators.len();
                    let edges = self.graph_edges.len();
                    let operator_base = voice * operators;
                    for operator in 0..operators {
                        let index = operator_base + operator;
                        self.graph_phases[index] = (self.graph_operators[operator].phase_offset
                            * self.rate)
                            .rem_euclid(self.rate);
                        self.graph_errors[index] = 0.0;
                        self.graph_outputs[index] = 0.0;
                        self.graph_release_levels[index] = 0.0;
                    }
                    self.graph_history[voice * edges..(voice + 1) * edges].fill(0.0);
                }
                let envelope_base = voice * self.named_envelopes.len();
                for (index, definition) in self.named_envelopes.iter().enumerate() {
                    self.named_states[envelope_base + index] = named_contour_initial(
                        definition,
                        self.named_time(definition, self.frame as f64),
                    );
                    if let Some((key, _, _)) = definition.sample_hold {
                        let state = &mut self.named_states[envelope_base + index];
                        state.random_state = self.motion_keys[voice] ^ key;
                        sample_hold_draw(definition, state);
                    }
                }
                if self.source_kind == 2 {
                    self.noise_keys[voice] = action[3] as u64 | ((action[4] as u64) << 32);
                    self.noise_counters[voice] = 0;
                    self.frequencies[voice] = 1.0;
                    self.gains[voice] = action[5];
                } else {
                    self.frequencies[voice] = action[3];
                    self.gains[voice] = action[4];
                }
                self.ages[voice] = 0;
                self.release_frames[voice] = None;
                self.voice_part_contexts[voice] = usize::MAX;
                self.voice_trigger_contexts[voice] = usize::MAX;
                self.frequency_remaining[voice] = 0;
                self.gain_remaining[voice] = 0;
                for stage in 0..self.voice_filter_states.shape()[1] {
                    self.voice_filter_states[[voice, stage, 0]] = 0.0;
                    self.voice_filter_states[[voice, stage, 1]] = 0.0;
                }
            }
            1 => {
                if self.active[voice] && self.release_frames[voice].is_none() {
                    for (source, definition) in self.staged_motions.iter().enumerate() {
                        let index = voice * self.staged_motions.len() + source;
                        let at = self.staged_time(definition, self.frame as f64);
                        if staged_transition(
                            definition,
                            &mut self.staged_states[index],
                            "note_off",
                            at,
                        ) {
                            self.staged_pending.push_back(StagedEvent {
                                at: self.frame as f64,
                                voice,
                                source,
                                port: "done".to_owned(),
                            });
                        }
                    }
                    let release_frame = (self.ages[voice] as f64).max(self.minimum_hold_frames);
                    self.release_levels[voice] =
                        envelope_value(self.initial, &self.attack, release_frame);
                    self.mod_release_levels[voice] =
                        envelope_value(self.mod_initial, &self.mod_attack, release_frame);
                    if self.source_kind == 3 {
                        let operator_base = voice * self.graph_operators.len();
                        for (operator, definition) in self.graph_operators.iter().enumerate() {
                            self.graph_release_levels[operator_base + operator] = envelope_value(
                                definition.initial,
                                &definition.attack,
                                release_frame,
                            );
                        }
                    }
                    let envelope_base = voice * self.named_envelopes.len();
                    for (index, definition) in self.named_envelopes.iter().enumerate() {
                        if definition.release.is_empty() {
                            continue;
                        }
                        let named_release_at = if definition.release_with_voice {
                            self.frame as f64 + release_frame - self.ages[voice] as f64
                        } else {
                            self.frame as f64
                        };
                        self.named_states[envelope_base + index].pending_release = Some((
                            named_release_at,
                            self.named_time(definition, named_release_at),
                        ));
                    }
                    self.release_frames[voice] = Some(release_frame);
                }
            }
            2 => {
                self.active[voice] = false;
                self.delayed_events[voice].clear();
                let base = voice * self.named_envelopes.len();
                for state in &mut self.named_states[base..base + self.named_envelopes.len()] {
                    state.pending_release = None;
                }
            }
            3 => {
                let duration = action[5] as usize;
                if action[3] <= 0.0 || action[4] < 0.0 || action[5] != duration as f64 {
                    return Err(PyValueError::new_err("Invalid voice control"));
                }
                self.frequency_steps[voice] =
                    (action[3] - self.frequencies[voice]) / duration.max(1) as f64;
                self.frequency_remaining[voice] = duration;
                self.gain_steps[voice] = (action[4] - self.gains[voice]) / duration.max(1) as f64;
                self.gain_remaining[voice] = duration;
                if duration == 0 {
                    self.frequencies[voice] = action[3];
                    self.gains[voice] = action[4];
                }
            }
            4 => {
                let Some(definition) = self.control_definitions.get(voice).cloned() else {
                    return Err(PyValueError::new_err("Invalid synth runtime control"));
                };
                if action[3] < definition.minimum || action[3] > definition.maximum {
                    return Err(PyValueError::new_err("Invalid synth control value"));
                }
                let context = decode_context(action[4], self.context_kinds.len())?;
                let state = self.control_state_mut(voice, context)?;
                state.start = control_value(&definition, state);
                state.target = action[3];
                state.elapsed = 0;
            }
            5 => {
                let part = decode_context(action[3], self.context_kinds.len())?;
                let trigger = decode_context(action[4], self.context_kinds.len())?;
                if part.is_some_and(|context| self.context_kinds[context] != 1)
                    || trigger.is_some_and(|context| self.context_kinds[context] != 2)
                {
                    return Err(PyValueError::new_err("Invalid voice control contexts"));
                }
                self.voice_part_contexts[voice] = part.unwrap_or(usize::MAX);
                self.voice_trigger_contexts[voice] = trigger.unwrap_or(usize::MAX);
            }
            6 => {
                if voice >= self.context_kinds.len() || self.context_kinds[voice] != 0 {
                    return Err(PyValueError::new_err("Invalid synth control context"));
                }
                let context_kind = action[3] as usize;
                if action[3] != context_kind as f64 || !(1..=2).contains(&context_kind) {
                    return Err(PyValueError::new_err("Invalid synth context kind"));
                }
                self.context_kinds[voice] = context_kind;
                let sources = self.control_definitions.len();
                for source in 0..sources {
                    let default = self.control_definitions[source].default;
                    self.context_states[voice * sources + source] = ControlState {
                        start: default,
                        target: default,
                        elapsed: 0,
                    };
                }
            }
            7 => {
                let source = action[3] as usize;
                if voice >= self.context_kinds.len()
                    || action[3] != source as f64
                    || source >= self.control_definitions.len()
                    || action[4] < self.control_definitions[source].minimum
                    || action[4] > self.control_definitions[source].maximum
                    || self.context_kinds[voice] != self.control_definitions[source].scope
                {
                    return Err(PyValueError::new_err("Invalid synth context value"));
                }
                let index = voice * self.control_definitions.len() + source;
                self.context_states[index].start = action[4];
                self.context_states[index].target = action[4];
                self.context_states[index].elapsed = 0;
            }
            8 => {
                let lfo = voice;
                let action_kind = action[3] as usize;
                if lfo >= self.lfo_definitions.len()
                    || action[3] != action_kind as f64
                    || action_kind > 6
                    || !action[4].is_finite()
                    || (action_kind != 1
                        && action_kind != 5
                        && action_kind != 6
                        && action[4] != 0.0)
                    || (action_kind == 1 && action[4] < 0.0)
                    || (action_kind == 5 && !(0.0..=1.0).contains(&action[4]))
                {
                    return Err(PyValueError::new_err("Invalid synth LFO action"));
                }
                let definition = &self.lfo_definitions[lfo];
                if definition.scope != 0 || definition.transport_position {
                    return Err(PyValueError::new_err("Invalid synth LFO scope"));
                }
                let state = self.lfo_event_states[lfo].clone().unwrap_or(LfoEventState {
                    at: 0,
                    age: 0.0,
                    phase: definition.phase.0 as f64 / definition.phase.1 as f64,
                    rate: definition.rate.0 as f64 / definition.rate.1 as f64,
                    direction: 1.0,
                    paused: false,
                });
                let elapsed = if state.paused {
                    0.0
                } else {
                    self.lfo_elapsed(definition, state.at, self.frame)
                };
                let phase = state.phase
                    + state.direction * state.rate * elapsed / self.lfo_divisor(definition);
                self.lfo_event_states[lfo] = Some(LfoEventState {
                    at: self.frame,
                    age: if action_kind == 0 {
                        0.0
                    } else {
                        state.age + elapsed
                    },
                    phase: if action_kind == 0 {
                        definition.phase.0 as f64 / definition.phase.1 as f64
                    } else if action_kind == 5 {
                        action[4]
                    } else if action_kind == 6 {
                        phase + action[4]
                    } else {
                        phase
                    },
                    rate: if action_kind == 1 {
                        action[4]
                    } else {
                        state.rate
                    },
                    direction: if action_kind == 4 {
                        -state.direction
                    } else {
                        state.direction
                    },
                    paused: if action_kind == 2 {
                        true
                    } else if action_kind == 3 {
                        false
                    } else {
                        state.paused
                    },
                });
            }
            9 => {
                let source = action[3] as usize;
                let command = action[4] as usize;
                if !self.active[voice]
                    || action[3] != source as f64
                    || source >= self.named_envelopes.len()
                    || action[4] != command as f64
                    || command > 4
                    || (command == 3 && !(0.0..=1.0).contains(&action[5]))
                    || (command == 4 && !action[5].is_finite())
                    || (command != 3 && command != 4 && action[5] != 0.0)
                {
                    return Err(PyValueError::new_err("Invalid named Motion action"));
                }
                let index = voice * self.named_envelopes.len() + source;
                let definition = &self.named_envelopes[source];
                if definition.sample_hold.is_some() {
                    return Ok(());
                }
                let at = self.named_time(definition, self.frame as f64);
                let mut state =
                    named_contour_settled(definition, &self.named_states[index], self.frame as f64);
                if let Some(count) = definition.repeat_count.filter(|_| !state.released) {
                    if state.complete_coordinate.is_none() {
                        let (crossed, boundary) = repeat_crossings(
                            &state.playback,
                            at,
                            definition.loop_start,
                            definition.loop_end,
                            count - state.traversals,
                        );
                        state.traversals += crossed;
                        if let Some(boundary) = boundary {
                            state.playback.advance(at);
                            state.playback.coordinate = boundary;
                            state.complete_coordinate = Some(repeat_boundary_coordinate(
                                definition.playback,
                                boundary,
                                state.playback.direction,
                                definition.loop_start,
                                definition.loop_end,
                            ));
                        }
                    }
                }
                state.playback.command(command, action[5], at);
                if command == 4 && (state.released || definition.playback == 0) {
                    state.playback.coordinate = state.playback.coordinate.clamp(0.0, 1.0);
                }
                self.named_states[index] = state;
            }
            10 => {
                let source = action[3] as usize;
                let command = action[4] as usize;
                if !self.active[voice]
                    || action[3] != source as f64
                    || source >= self.staged_motions.len()
                    || action[4] != command as f64
                    || command > 4
                    || (command == 3 && !(0.0..=1.0).contains(&action[5]))
                    || (command == 4 && !action[5].is_finite())
                    || (command != 3 && command != 4 && action[5] != 0.0)
                {
                    return Err(PyValueError::new_err("Invalid staged Motion action"));
                }
                let index = voice * self.staged_motions.len() + source;
                let at = self.staged_time(&self.staged_motions[source], self.frame as f64);
                let state = &mut self.staged_states[index];
                if (command == 3 || command == 4)
                    && self.staged_motions[source].stages[state.stage].kind == 0
                {
                    return Err(PyValueError::new_err("Hold stage cannot change position"));
                }
                state.playback.command(command, action[5], at);
                if command == 4
                    && self.staged_motions[source].stages[state.stage].kind == 1
                    && self.staged_motions[source].stages[state.stage].playback == 0
                {
                    state.playback.coordinate = state.playback.coordinate.clamp(0.0, 1.0);
                }
                state.cursor_at = at;
                state.cursor_order = usize::MAX;
            }
            11 => {
                let source = action[3] as usize;
                let command = action[4] as usize;
                if !self.active[voice]
                    || action[3] != source as f64
                    || source >= self.lfo_definitions.len()
                    || self.lfo_definitions[source].scope != 3
                    || self.lfo_definitions[source].transport_position
                    || action[4] != command as f64
                    || command > 4
                    || (command == 3 && !(0.0..=1.0).contains(&action[5]))
                    || (command == 4 && !action[5].is_finite())
                    || (command != 3 && command != 4 && action[5] != 0.0)
                {
                    return Err(PyValueError::new_err("Invalid voice Cycle action"));
                }
                let definition = &self.lfo_definitions[source];
                let index = voice * self.lfo_definitions.len() + source;
                let state = self.voice_lfo_event_states[index]
                    .clone()
                    .unwrap_or(LfoEventState {
                        at: self.frame - self.ages[voice],
                        age: 0.0,
                        phase: definition.phase.0 as f64 / definition.phase.1 as f64,
                        rate: definition.rate.0 as f64 / definition.rate.1 as f64,
                        direction: 1.0,
                        paused: false,
                    });
                let elapsed = if state.paused {
                    0.0
                } else {
                    self.lfo_elapsed(definition, state.at, self.frame)
                };
                let phase = state.phase
                    + state.direction * state.rate * elapsed / self.lfo_divisor(definition);
                self.voice_lfo_event_states[index] = Some(LfoEventState {
                    at: self.frame,
                    age: state.age + elapsed,
                    phase: if command == 3 {
                        action[5]
                    } else if command == 4 {
                        phase + action[5]
                    } else {
                        phase
                    },
                    rate: state.rate,
                    direction: if command == 2 {
                        -state.direction
                    } else {
                        state.direction
                    },
                    paused: if command == 0 {
                        true
                    } else if command == 1 {
                        false
                    } else {
                        state.paused
                    },
                });
            }
            _ => return Err(PyValueError::new_err("Unknown synth runtime action")),
        }
        Ok(())
    }

    fn advance_staged(
        &mut self,
        limit_frame: f64,
        inclusive: bool,
        delayed_inclusive: bool,
    ) -> PyResult<()> {
        let mut count = 0;
        let mut batch_at = None;
        loop {
            let mut next: Option<f64> = None;
            let mut events = Vec::new();
            for voice in 0..self.frequencies.len() {
                if !self.active[voice] {
                    continue;
                }
                for (source, definition) in self.staged_motions.iter().enumerate() {
                    if self.staged_owners[source] != source {
                        continue;
                    }
                    let state = &self.staged_states[voice * self.staged_motions.len() + source];
                    let limit = self.staged_time(definition, limit_frame);
                    if let Some((stage_at, order, port)) =
                        next_staged_event(definition, state, limit, inclusive)
                    {
                        let at = if definition.beat_clock {
                            self.frame_at_local_beat(stage_at)
                        } else {
                            stage_at
                        };
                        if next.is_none_or(|current| at < current) {
                            next = Some(at);
                            events.clear();
                        }
                        if next == Some(at) {
                            events.push((voice, source, stage_at, order, port));
                        }
                    }
                }
            }
            let delayed_at = self
                .delayed_events
                .iter()
                .flatten()
                .filter(|event| {
                    event.at < limit_frame
                        || (inclusive && delayed_inclusive && event.at == limit_frame)
                })
                .map(|event| event.at)
                .min_by(f64::total_cmp);
            if let Some(at) = delayed_at {
                if next.is_none_or(|current| at < current) {
                    next = Some(at);
                    events.clear();
                }
            }
            if batch_at.is_some() && batch_at != next {
                self.dispatch_staged_events()?;
                batch_at = None;
                continue;
            }
            let Some(at) = next else {
                break;
            };
            batch_at = Some(at);
            if events.is_empty() {
                self.dispatch_staged_events()?;
                count += self.deliver_delayed_events(at)?;
                self.dispatch_staged_events()?;
                batch_at = None;
                if count > 4096 {
                    return Err(PyValueError::new_err(
                        "Delayed Motion event capacity exceeded",
                    ));
                }
                continue;
            }
            for (voice, source, stage_at, order, port) in events {
                if count == 4096 {
                    return Err(PyValueError::new_err(
                        "Staged Motion event capacity exceeded",
                    ));
                }
                let definition = &self.staged_motions[source];
                let state = &mut self.staged_states[voice * self.staged_motions.len() + source];
                state.cursor_at = stage_at;
                state.cursor_order = order;
                state.playback.advance(stage_at);
                let stage_index = state.stage;
                let stage = &definition.stages[state.stage];
                if port == "done" {
                    if stage.repeat_count.is_none() {
                        state.playback.coordinate = 1.0;
                    }
                } else if stage.kind == 1 {
                    if stage.loop_start != 0.0 || stage.loop_end != 1.0 {
                        if port == "cycle" || port == "turned" {
                            let width = stage.loop_end - stage.loop_start;
                            state.playback.coordinate = stage.loop_end
                                + ((state.playback.coordinate - stage.loop_end) / width).round()
                                    * width;
                        }
                    } else if port == "cycle" || port == "turned" {
                        state.playback.coordinate = state.playback.coordinate.round();
                    } else if stage.playback == 0 {
                        state.playback.coordinate = stage.markers[order].0;
                    } else {
                        let marker = stage.markers[order].0;
                        let period = if stage.playback == 1 { 1.0 } else { 2.0 };
                        let outward = ((state.playback.coordinate - marker) / period).round()
                            * period
                            + marker;
                        if stage.playback == 1 {
                            state.playback.coordinate = outward;
                        } else {
                            let returning = ((state.playback.coordinate - (2.0 - marker)) / period)
                                .round()
                                * period
                                + 2.0
                                - marker;
                            state.playback.coordinate = if (outward - state.playback.coordinate)
                                .abs()
                                < (returning - state.playback.coordinate).abs()
                            {
                                outward
                            } else {
                                returning
                            };
                        }
                    }
                } else if stage.kind == 2 {
                    let marker = stage.markers[order].0;
                    state.playback.coordinate =
                        (state.playback.coordinate - marker).round() + marker;
                }
                if stage.repeat_count.is_some()
                    && port
                        == if stage.playback == 1 {
                            "cycle"
                        } else {
                            "turned"
                        }
                {
                    state.traversals += 1;
                }
                let final_repeat = stage.repeat_count.is_some_and(|count| {
                    state.traversals == count
                        && port
                            == if stage.playback == 1 {
                                "cycle"
                            } else {
                                "turned"
                            }
                });
                let returning = final_repeat
                    && stage.playback == 2
                    && (((state.playback.coordinate - stage.loop_end)
                        / (stage.loop_end - stage.loop_start))
                        .round()
                        .rem_euclid(2.0)
                        == 1.0);
                let finished =
                    staged_transition(definition, state, &format!("stage.{port}"), stage_at);
                self.staged_pending.push_back(StagedEvent {
                    at,
                    voice,
                    source,
                    port: if port == "done" {
                        "stage.done".to_owned()
                    } else {
                        port
                    },
                });
                if final_repeat {
                    if returning {
                        self.staged_pending.push_back(StagedEvent {
                            at,
                            voice,
                            source,
                            port: "cycle".to_owned(),
                        });
                    }
                    self.staged_pending.push_back(StagedEvent {
                        at,
                        voice,
                        source,
                        port: "stage.done".to_owned(),
                    });
                    if !finished && state.stage == stage_index {
                        if staged_transition(definition, state, "stage.done", stage_at) {
                            self.staged_pending.push_back(StagedEvent {
                                at,
                                voice,
                                source,
                                port: "done".to_owned(),
                            });
                        } else if state.stage == stage_index {
                            state.complete_value = staged_value(definition, state, stage_at);
                            state.completed = true;
                        }
                    }
                }
                if finished {
                    self.staged_pending.push_back(StagedEvent {
                        at,
                        voice,
                        source,
                        port: "done".to_owned(),
                    });
                }
                count += 1;
            }
        }
        self.dispatch_staged_events()?;
        Ok(())
    }

    fn dispatch_staged_events(&mut self) -> PyResult<()> {
        let mut count = 0;
        while let Some(event) = self.staged_pending.pop_front() {
            let connections = self.staged_connections.len()
                + self.patch_events.len()
                + self.patch_stage_starts.len();
            for (index, connection) in self.staged_connections.iter().enumerate() {
                if connection.source != event.source || connection.port != event.port {
                    continue;
                }
                if !forward_event(
                    &mut self.event_counts[event.voice * connections + index],
                    &mut self.event_random[event.voice * connections + index],
                    connection.every,
                    connection.probability,
                ) {
                    continue;
                }
                if count == 4096 {
                    return Err(PyValueError::new_err(
                        "Staged Motion event capacity exceeded",
                    ));
                }
                if connection.delay > 0.0 {
                    queue_delayed_event(
                        &mut self.delayed_events[event.voice],
                        event.at,
                        connection.delay,
                        DelayedConnection::StagedCue(index),
                    )?;
                    count += 1;
                    continue;
                }
                let definition = &self.staged_motions[connection.destination];
                let at = self.staged_time(definition, event.at);
                let state = &mut self.staged_states
                    [event.voice * self.staged_motions.len() + connection.destination];
                if staged_transition(definition, state, &format!("cue.{}", connection.cue), at) {
                    self.staged_pending.push_back(StagedEvent {
                        at: event.at,
                        voice: event.voice,
                        source: connection.destination,
                        port: "done".to_owned(),
                    });
                }
                count += 1;
            }
            for (index, connection) in self.patch_stage_starts.iter().enumerate() {
                if connection.source != event.source || connection.port != event.port {
                    continue;
                }
                if !forward_event(
                    &mut self.event_counts[event.voice * connections
                        + self.staged_connections.len()
                        + self.patch_events.len()
                        + index],
                    &mut self.event_random[event.voice * connections
                        + self.staged_connections.len()
                        + self.patch_events.len()
                        + index],
                    connection.every,
                    connection.probability,
                ) {
                    continue;
                }
                if count == 4096 {
                    return Err(PyValueError::new_err(
                        "Staged Motion event capacity exceeded",
                    ));
                }
                if connection.delay > 0.0 {
                    queue_delayed_event(
                        &mut self.delayed_events[event.voice],
                        event.at,
                        connection.delay,
                        DelayedConnection::StageStart(index),
                    )?;
                    count += 1;
                    continue;
                }
                self.pending_contour_starts.push(ContourStart {
                    at: event.at,
                    order: connection.order,
                    voice: event.voice,
                    destination: connection.destination,
                });
                count += 1;
            }
        }
        Ok(())
    }

    fn advance_patch_events(&mut self, at_boundary: bool) -> PyResult<()> {
        if self.patch_events.is_empty() {
            return Ok(());
        }
        let mut events = Vec::new();
        for voice in 0..self.frequencies.len() {
            if !self.active[voice] || self.ages[voice] == 0 {
                continue;
            }
            let age = self.ages[voice] as f64;
            for (order, connection) in self.patch_events.iter().enumerate() {
                if connection.threshold {
                    continue;
                }
                let definition = &self.lfo_definitions[connection.source];
                let rate = definition.rate.0 as f64 / definition.rate.1 as f64;
                let phase = definition.phase.0 as f64 / definition.phase.1 as f64;
                let before = ((phase + rate * (age - 1.0) / self.rate) - connection.marker).floor();
                let after = ((phase + rate * age / self.rate) - connection.marker + 1e-12).floor();
                if !before.is_finite()
                    || !after.is_finite()
                    || before.abs() >= i64::MAX as f64
                    || after.abs() >= i64::MAX as f64
                    || after - before > (4096 - events.len()) as f64
                {
                    return Err(PyValueError::new_err("Patch event capacity exceeded"));
                }
                let first = before as i64 + 1;
                let last = after as i64;
                for turn in first..=last {
                    let event_age = (turn as f64 + connection.marker - phase) * self.rate / rate;
                    events.push((self.frame as f64 - (age - event_age), order, voice));
                }
            }
        }
        events.sort_by(|a, b| a.partial_cmp(b).expect("finite Patch event time"));
        let mut previous_at = None;
        for (at, order, voice) in events {
            if (at >= self.frame as f64) != at_boundary {
                continue;
            }
            if previous_at != Some(at) {
                self.dispatch_staged_events()?;
                self.advance_staged(at, true, false)?;
                previous_at = Some(at);
            }
            let connection = &self.patch_events[order];
            let connections = self.staged_connections.len()
                + self.patch_events.len()
                + self.patch_stage_starts.len();
            if !forward_event(
                &mut self.event_counts[voice * connections + self.staged_connections.len() + order],
                &mut self.event_random[voice * connections + self.staged_connections.len() + order],
                connection.every,
                connection.probability,
            ) {
                continue;
            }
            let destination = connection.destination;
            if connection.delay > 0.0 {
                queue_delayed_event(
                    &mut self.delayed_events[voice],
                    at,
                    connection.delay,
                    DelayedConnection::PatchEvent(order),
                )?;
                continue;
            }
            if let Some(cue) = &connection.cue {
                let definition = &self.staged_motions[destination];
                let stage_at = self.staged_time(definition, at);
                let state =
                    &mut self.staged_states[voice * self.staged_motions.len() + destination];
                if staged_transition(definition, state, &format!("cue.{cue}"), stage_at) {
                    self.staged_pending.push_back(StagedEvent {
                        at,
                        voice,
                        source: destination,
                        port: "done".to_owned(),
                    });
                }
                continue;
            }
            self.pending_contour_starts.push(ContourStart {
                at,
                order: connection.order,
                voice,
                destination,
            });
        }
        self.dispatch_staged_events()?;
        Ok(())
    }

    fn deliver_delayed_events(&mut self, at: f64) -> PyResult<usize> {
        let mut count = 0;
        for voice in 0..self.delayed_events.len() {
            while let Some(index) = self.delayed_events[voice]
                .iter()
                .position(|event| event.at == at)
            {
                let event = self.delayed_events[voice].remove(index);
                count += 1;
                let (destination, cue, order) = match event.connection {
                    DelayedConnection::StagedCue(index) => {
                        let connection = &self.staged_connections[index];
                        (connection.destination, Some(&connection.cue), index)
                    }
                    DelayedConnection::PatchEvent(index) => {
                        let connection = &self.patch_events[index];
                        (
                            connection.destination,
                            connection.cue.as_ref(),
                            connection.order,
                        )
                    }
                    DelayedConnection::StageStart(index) => {
                        let connection = &self.patch_stage_starts[index];
                        (connection.destination, None, connection.order)
                    }
                };
                if let Some(cue) = cue {
                    let definition = &self.staged_motions[destination];
                    let stage_at = self.staged_time(definition, at);
                    let state =
                        &mut self.staged_states[voice * self.staged_motions.len() + destination];
                    if staged_transition(definition, state, &format!("cue.{cue}"), stage_at) {
                        self.staged_pending.push_back(StagedEvent {
                            at,
                            voice,
                            source: destination,
                            port: "done".to_owned(),
                        });
                    }
                } else {
                    self.pending_contour_starts.push(ContourStart {
                        at,
                        voice,
                        destination,
                        order,
                    });
                }
                self.dispatch_staged_events()?;
            }
        }
        Ok(count)
    }

    fn flush_contour_starts(&mut self) -> PyResult<()> {
        if self.pending_contour_starts.len() > 4096 {
            return Err(PyValueError::new_err("Patch event capacity exceeded"));
        }
        let mut starts = std::mem::take(&mut self.pending_contour_starts);
        starts.sort_by(|a, b| {
            a.at.total_cmp(&b.at)
                .then_with(|| a.order.cmp(&b.order))
                .then_with(|| a.voice.cmp(&b.voice))
        });
        for event in starts {
            let definition = &self.named_envelopes[event.destination];
            let index = event.voice * self.named_envelopes.len() + event.destination;
            if definition.sample_hold.is_some() {
                sample_hold_draw(definition, &mut self.named_states[index]);
                continue;
            }
            let previous = &self.named_states[index];
            let start_value = if previous.idle {
                definition.initial
            } else {
                named_contour_value(definition, previous, event.at, event.at)
            };
            let mut state = named_contour_initial(definition, event.at);
            state.idle = false;
            state.start_value = start_value;
            self.named_states[index] = state;
        }
        Ok(())
    }

    fn lfo_signal(&self, voice: usize, source: usize) -> PyResult<(f64, f64)> {
        let definition = &self.lfo_definitions[source];
        let owner = self.lfo_owners[source];
        let elapsed = match definition.scope {
            0 => self.frame,
            1 => {
                self.context_kind(self.voice_part_contexts[voice], 1)?;
                self.frame
            }
            _ => self.ages[voice],
        };
        let event_state = if definition.scope == 3 {
            &self.voice_lfo_event_states[voice * self.lfo_definitions.len() + owner]
        } else {
            &self.lfo_event_states[owner]
        };
        let (value, weight) = if definition.transport_position {
            let state = LfoEventState {
                at: 0,
                age: 0.0,
                phase: definition.phase.0 as f64 / definition.phase.1 as f64,
                rate: definition.rate.0 as f64 / definition.rate.1 as f64,
                direction: 1.0,
                paused: false,
            };
            lfo_event_value(
                definition,
                &state,
                self.song_beat_at(self.frame as f64),
                1.0,
            )
        } else if let Some(state) = event_state {
            lfo_event_value(
                definition,
                state,
                if state.paused {
                    0.0
                } else {
                    self.lfo_elapsed(definition, state.at, self.frame)
                },
                self.lfo_divisor(definition),
            )
        } else if definition.beat_clock {
            let start = if definition.scope == 3 {
                self.frame - self.ages[voice]
            } else {
                0
            };
            let state = LfoEventState {
                at: start,
                age: 0.0,
                phase: definition.phase.0 as f64 / definition.phase.1 as f64,
                rate: definition.rate.0 as f64 / definition.rate.1 as f64,
                direction: 1.0,
                paused: false,
            };
            lfo_event_value(
                definition,
                &state,
                self.lfo_elapsed(definition, start, self.frame),
                1.0,
            )
        } else {
            lfo_value(definition, elapsed, self.rate as u64)?
        };
        Ok((value, weight))
    }

    fn evaluate_motion_transforms(&mut self, voice: usize) -> PyResult<()> {
        let base = voice * self.motion_transforms.len();
        for (node, (kind, inputs, scale, offset, initial)) in
            self.motion_transforms.iter().enumerate()
        {
            let mut output = if *kind == 1 { 1.0 } else { 0.0 };
            for (source_kind, source, center, depth) in inputs {
                let value = match source_kind {
                    0 => {
                        let (value, weight) = self.lfo_signal(voice, *source)?;
                        (center + depth * value) * weight
                    }
                    1 => {
                        let definition = &self.named_envelopes[*source];
                        named_contour_value(
                            definition,
                            &self.named_states
                                [voice * self.named_envelopes.len() + self.named_owners[*source]],
                            self.named_time(definition, self.frame as f64),
                            self.frame as f64,
                        )
                    }
                    2 => {
                        let definition = &self.staged_motions[*source];
                        staged_value(
                            definition,
                            &self.staged_states
                                [voice * self.staged_motions.len() + self.staged_owners[*source]],
                            self.staged_time(definition, self.frame as f64),
                        )
                    }
                    _ => self.motion_transform_values[base + source],
                };
                if *kind == 1 {
                    output *= value;
                } else {
                    output += value;
                }
            }
            self.motion_transform_values[base + node] = if *kind == 4 {
                let value = match self.motion_transform_states[base + node] {
                    None => initial.unwrap_or(output),
                    Some(previous) if output >= previous => output.min(previous + scale),
                    Some(previous) => output.max(previous - offset),
                };
                self.motion_transform_states[base + node] = Some(value);
                value
            } else if *kind == 3 {
                let previous = self.motion_transform_states[base + node];
                let value = match previous {
                    None => {
                        if output >= *offset {
                            1.0
                        } else {
                            0.0
                        }
                    }
                    Some(0.0) if output >= *offset => 1.0,
                    Some(1.0) if output <= *scale => 0.0,
                    Some(value) => value,
                };
                self.motion_transform_states[base + node] = Some(value);
                if previous.is_some_and(|old| old != value) {
                    let connections = self.staged_connections.len()
                        + self.patch_events.len()
                        + self.patch_stage_starts.len();
                    for (order, connection) in self.patch_events.iter().enumerate() {
                        if !connection.threshold
                            || connection.source != node
                            || connection.marker != value
                        {
                            continue;
                        }
                        let index = voice * connections + self.staged_connections.len() + order;
                        if forward_event(
                            &mut self.event_counts[index],
                            &mut self.event_random[index],
                            connection.every,
                            connection.probability,
                        ) {
                            queue_delayed_event(
                                &mut self.delayed_events[voice],
                                self.frame as f64,
                                connection.delay,
                                DelayedConnection::PatchEvent(order),
                            )?;
                        }
                    }
                }
                value
            } else if *kind == 2 {
                offset + scale * output
            } else {
                output
            };
        }
        Ok(())
    }

    fn parameter(&self, voice: usize, parameter: usize) -> PyResult<f64> {
        let mut addition = 0.0;
        let mut product = 1.0;
        for (source, definition) in self.control_definitions.iter().enumerate() {
            if definition.parameter != parameter {
                continue;
            }
            let state = match definition.scope {
                0 => &self.control_states[source],
                1 => self.context_state(self.voice_part_contexts[voice], source, 1)?,
                _ => self.context_state(self.voice_trigger_contexts[voice], source, 2)?,
            };
            let amount = definition.intercept + definition.slope * control_value(definition, state);
            if definition.operation == 0 {
                addition += amount;
            } else {
                product *= amount;
            }
        }
        for (source, definition) in self.lfo_definitions.iter().enumerate() {
            if definition.parameter != parameter {
                continue;
            }
            let (value, weight) = self.lfo_signal(voice, source)?;
            let amount = definition.intercept + definition.slope * value;
            if definition.operation == 0 {
                addition += weight * amount;
            } else {
                product *= 1.0 + weight * (amount - 1.0);
            }
        }
        let envelope_base = voice * self.named_envelopes.len();
        for (source, definition) in self.named_envelopes.iter().enumerate() {
            if definition.parameter != parameter {
                continue;
            }
            let value = named_contour_value(
                definition,
                &self.named_states[envelope_base + self.named_owners[source]],
                self.named_time(definition, self.frame as f64),
                self.frame as f64,
            );
            let amount = definition.intercept + definition.slope * value;
            if definition.operation == 0 {
                addition += amount;
            } else {
                product *= amount;
            }
        }
        for (source, definition) in self.staged_motions.iter().enumerate() {
            if definition.parameter != Some(parameter) {
                continue;
            }
            let state =
                &self.staged_states[voice * self.staged_motions.len() + self.staged_owners[source]];
            let value = staged_value(
                definition,
                state,
                self.staged_time(definition, self.frame as f64),
            );
            let amount = definition.intercept + definition.slope * value;
            if definition.operation == 0 {
                addition += amount;
            } else {
                product *= amount;
            }
        }
        for (source, target, operation, intercept, slope) in &self.motion_transform_routes {
            if *target != parameter {
                continue;
            }
            let amount = intercept
                + slope
                    * self.motion_transform_values[voice * self.motion_transforms.len() + source];
            if *operation == 0 {
                addition += amount;
            } else {
                product *= amount;
            }
        }
        let definition = &self.parameter_definitions[parameter * 3..parameter * 3 + 3];
        let value = (definition[0] + addition) * product;
        if !value.is_finite() || value < definition[1] || value > definition[2] {
            return Err(PyValueError::new_err(
                "Modulated persistent synth parameter is outside its domain",
            ));
        }
        Ok(value)
    }

    fn filter_voice(&mut self, voice: usize, mut sample: f64) -> PyResult<f64> {
        let mut stage = 0;
        for filter_index in 0..self.filter_definitions.len() {
            let filter = self.filter_definitions[filter_index].clone();
            let mut cutoff = self.parameter(voice, filter.cutoff_parameter)?;
            let q = self.parameter(voice, filter.q_parameter)?;
            if q <= 0.0 {
                return Err(PyValueError::new_err("Invalid persistent synth filter Q"));
            }
            if cutoff < filter.minimum_hz || cutoff > filter.maximum_hz {
                if filter.boundary == 0 {
                    return Err(PyValueError::new_err(
                        "Invalid persistent synth filter cutoff",
                    ));
                }
                cutoff = cutoff.clamp(filter.minimum_hz, filter.maximum_hz);
            }
            let g = (PI * cutoff / self.rate).tan();
            let d = q * (1.0 + g * g) + g;
            let a1 = if q < 1.0 {
                q / d
            } else {
                1.0 / (1.0 + g * (g + 1.0 / q))
            };
            let a2 = g * a1;
            let a3 = g * a2;
            for _ in 0..filter.stages {
                let s1 = self.voice_filter_states[[voice, stage, 0]];
                let s2 = self.voice_filter_states[[voice, stage, 1]];
                let v3 = sample - s2;
                let v1 = a1 * s1 + a2 * v3;
                let v2 = s2 + a2 * s1 + a3 * v3;
                let band = if q < 1.0 {
                    s1 / d + (g / d) * v3
                } else {
                    v1 / q
                };
                sample = match filter.response {
                    0 => v2,
                    1 => sample - band - v2,
                    2 => band,
                    _ => sample - band,
                };
                self.voice_filter_states[[voice, stage, 0]] = 2.0 * v1 - s1;
                self.voice_filter_states[[voice, stage, 1]] = 2.0 * v2 - s2;
                if !sample.is_finite()
                    || !self.voice_filter_states[[voice, stage, 0]].is_finite()
                    || !self.voice_filter_states[[voice, stage, 1]].is_finite()
                {
                    return Err(PyValueError::new_err(
                        "Non-finite persistent synth filter output or state",
                    ));
                }
                stage += 1;
            }
        }
        Ok(sample)
    }

    fn context_state(&self, context: usize, source: usize, kind: usize) -> PyResult<&ControlState> {
        self.context_kind(context, kind)?;
        Ok(&self.context_states[context * self.control_definitions.len() + source])
    }

    fn context_kind(&self, context: usize, kind: usize) -> PyResult<()> {
        if context >= self.context_kinds.len() || self.context_kinds[context] != kind {
            return Err(PyValueError::new_err(
                "Voice is missing its control context",
            ));
        }
        Ok(())
    }

    fn control_state_mut(
        &mut self,
        source: usize,
        context: Option<usize>,
    ) -> PyResult<&mut ControlState> {
        let scope = self.control_definitions[source].scope;
        if scope == 0 && context.is_none() {
            return Ok(&mut self.control_states[source]);
        }
        let Some(context) = context else {
            return Err(PyValueError::new_err(
                "Control action is missing its context",
            ));
        };
        if self.context_kinds[context] != scope {
            return Err(PyValueError::new_err(
                "Control action has the wrong context",
            ));
        }
        Ok(&mut self.context_states[context * self.control_definitions.len() + source])
    }

    fn retire_trigger_contexts(&mut self) {
        for context in 0..self.context_kinds.len() {
            if self.context_kinds[context] == 2
                && !self
                    .active
                    .iter()
                    .enumerate()
                    .any(|(voice, active)| *active && self.voice_trigger_contexts[voice] == context)
            {
                self.context_kinds[context] = 0;
            }
        }
    }
}

fn decode_context(value: f64, capacity: usize) -> PyResult<Option<usize>> {
    if value == -1.0 {
        return Ok(None);
    }
    let context = value as usize;
    if value != context as f64 || context >= capacity {
        return Err(PyValueError::new_err("Invalid synth control context"));
    }
    Ok(Some(context))
}

fn control_definitions(
    values: PyReadonlyArray2<'_, f64>,
    smoothing: &[(u64, u64)],
) -> PyResult<(Vec<ControlDefinition>, Vec<ControlState>)> {
    if values.shape()[1] != 8
        || values.shape()[0] != smoothing.len()
        || smoothing.iter().any(|(_, denominator)| *denominator == 0)
        || values.as_array().iter().any(|v| !v.is_finite())
    {
        return Err(PyValueError::new_err("Invalid synth runtime controls"));
    }
    let mut definitions = Vec::with_capacity(values.shape()[0]);
    let mut states = Vec::with_capacity(values.shape()[0]);
    for (row, (smoothing_numerator, smoothing_denominator)) in
        values.as_array().rows().into_iter().zip(smoothing)
    {
        let scope = row[1] as usize;
        let parameter = row[2] as usize;
        let operation = row[3] as usize;
        if row[1] != scope as f64
            || scope > 2
            || row[2] != parameter as f64
            || row[3] != operation as f64
            || operation > 1
            || row[4] > row[5]
            || row[0] < row[4]
            || row[0] > row[5]
        {
            return Err(PyValueError::new_err("Invalid synth runtime control"));
        }
        definitions.push(ControlDefinition {
            default: row[0],
            smoothing_numerator: *smoothing_numerator,
            smoothing_denominator: *smoothing_denominator,
            scope,
            parameter,
            operation,
            minimum: row[4],
            maximum: row[5],
            intercept: row[6],
            slope: row[7],
        });
        states.push(ControlState {
            start: row[0],
            target: row[0],
            elapsed: 0,
        });
    }
    Ok((definitions, states))
}

fn control_value(definition: &ControlDefinition, state: &ControlState) -> f64 {
    if state.elapsed as u128 * definition.smoothing_denominator as u128
        >= definition.smoothing_numerator as u128
    {
        state.target
    } else {
        state.start
            + (state.target - state.start)
                * state.elapsed as f64
                * definition.smoothing_denominator as f64
                / definition.smoothing_numerator as f64
    }
}

fn lfo_definitions(
    values: PyReadonlyArray2<'_, f64>,
    rationals: &[(u64, u64)],
) -> PyResult<Vec<LfoDefinition>> {
    if values.shape()[1] != 7
        || rationals.len() != values.shape()[0] * 5
        || rationals.iter().any(|(_, denominator)| *denominator == 0)
        || values.as_array().iter().any(|v| !v.is_finite())
    {
        return Err(PyValueError::new_err("Invalid synth runtime LFOs"));
    }
    let mut definitions = Vec::with_capacity(values.shape()[0]);
    for (index, row) in values.as_array().rows().into_iter().enumerate() {
        let scope = row[0] as usize;
        let waveform = row[1] as usize;
        let parameter = row[2] as usize;
        let operation = row[3] as usize;
        let exact = &rationals[index * 5..index * 5 + 5];
        if row[0] != scope as f64
            || ![0, 1, 3].contains(&scope)
            || row[1] != waveform as f64
            || waveform > 2
            || row[2] != parameter as f64
            || row[3] != operation as f64
            || operation > 1
            || ![0.0, 1.0, 2.0].contains(&row[6])
            || exact[0].0 > exact[0].1
            || exact[2].0 >= exact[2].1
        {
            return Err(PyValueError::new_err("Invalid synth runtime LFO"));
        }
        definitions.push(LfoDefinition {
            scope,
            waveform,
            parameter,
            operation,
            intercept: row[4],
            slope: row[5],
            duty: exact[0],
            rate: exact[1],
            phase: exact[2],
            delay: exact[3],
            fade: exact[4],
            beat_clock: row[6] != 0.0,
            transport_position: row[6] == 2.0,
        });
    }
    Ok(definitions)
}

fn filter_definitions(values: PyReadonlyArray2<'_, f64>) -> PyResult<Vec<FilterDefinition>> {
    if values.shape()[1] != 7 || values.as_array().iter().any(|v| !v.is_finite()) {
        return Err(PyValueError::new_err("Invalid synth runtime filters"));
    }
    values
        .as_array()
        .rows()
        .into_iter()
        .map(|row| {
            let response = row[0] as usize;
            let stages = row[1] as usize;
            let boundary = row[2] as usize;
            let cutoff_parameter = row[5] as usize;
            let q_parameter = row[6] as usize;
            if row[0] != response as f64
                || response > 3
                || row[1] != stages as f64
                || !(1..=2).contains(&stages)
                || row[2] != boundary as f64
                || boundary > 1
                || row[3] <= 0.0
                || row[3] >= row[4]
                || row[5] != cutoff_parameter as f64
                || row[6] != q_parameter as f64
            {
                return Err(PyValueError::new_err("Invalid synth runtime filter"));
            }
            Ok(FilterDefinition {
                response,
                stages,
                boundary,
                minimum_hz: row[3],
                maximum_hz: row[4],
                cutoff_parameter,
                q_parameter,
            })
        })
        .collect()
}

fn lfo_value(definition: &LfoDefinition, elapsed: usize, sample_rate: u64) -> PyResult<(f64, f64)> {
    let Some(phase_denominator) = lfo_phase_denominator(definition, sample_rate) else {
        return Err(PyValueError::new_err("Invalid synth runtime LFO phase"));
    };
    let base = modular_multiply(
        modular_multiply(
            definition.phase.0 as u128,
            definition.rate.1 as u128,
            phase_denominator,
        ),
        sample_rate as u128,
        phase_denominator,
    );
    let step = modular_multiply(
        definition.rate.0 as u128,
        definition.phase.1 as u128,
        phase_denominator,
    );
    let phase_numerator = modular_add(
        base,
        modular_multiply(elapsed as u128, step, phase_denominator),
        phase_denominator,
    );
    let rising = phase_numerator * (definition.duty.1 as u128)
        < definition.duty.0 as u128 * phase_denominator;
    let phase = phase_numerator as f64 / phase_denominator as f64;
    let value = match definition.waveform {
        0 => (TAU * phase).sin(),
        1 => {
            if rising {
                1.0
            } else {
                -1.0
            }
        }
        _ if definition.duty.0 == 0 => 1.0 - 2.0 * phase,
        _ if definition.duty.0 == definition.duty.1 => 2.0 * phase - 1.0,
        _ if rising => 2.0 * phase / (definition.duty.0 as f64 / definition.duty.1 as f64) - 1.0,
        _ => {
            let duty = definition.duty.0 as f64 / definition.duty.1 as f64;
            (1.0 + duty - 2.0 * phase) / (1.0 - duty)
        }
    };
    let delayed =
        (elapsed as u128) < (definition.delay.0 as u128).div_ceil(definition.delay.1 as u128);
    let weight = if delayed {
        0.0
    } else if definition.fade.0 == 0 {
        1.0
    } else {
        ((elapsed as f64 - definition.delay.0 as f64 / definition.delay.1 as f64)
            / (definition.fade.0 as f64 / definition.fade.1 as f64))
            .clamp(0.0, 1.0)
    };
    Ok((value.clamp(-1.0, 1.0), weight))
}

fn lfo_event_value(
    definition: &LfoDefinition,
    state: &LfoEventState,
    elapsed: f64,
    divisor: f64,
) -> (f64, f64) {
    let phase = (state.phase + state.direction * state.rate * elapsed / divisor).rem_euclid(1.0);
    let duty = definition.duty.0 as f64 / definition.duty.1 as f64;
    let value = match definition.waveform {
        0 => (TAU * phase).sin(),
        1 => {
            if phase < duty {
                1.0
            } else {
                -1.0
            }
        }
        _ if duty == 0.0 => 1.0 - 2.0 * phase,
        _ if duty == 1.0 => 2.0 * phase - 1.0,
        _ if phase < duty => 2.0 * phase / duty - 1.0,
        _ => (1.0 + duty - 2.0 * phase) / (1.0 - duty),
    };
    let age = state.age + elapsed;
    let delay = definition.delay.0 as f64 / definition.delay.1 as f64;
    let fade = definition.fade.0 as f64 / definition.fade.1 as f64;
    let weight = if age < delay {
        0.0
    } else if fade == 0.0 {
        1.0
    } else {
        ((age - delay) / fade).clamp(0.0, 1.0)
    };
    (value, weight)
}

fn lfo_phase_denominator(definition: &LfoDefinition, sample_rate: u64) -> Option<u128> {
    let denominator = (definition.phase.1 as u128)
        .checked_mul(definition.rate.1 as u128)?
        .checked_mul(sample_rate as u128)?;
    denominator.checked_mul(definition.duty.0.max(definition.duty.1) as u128)?;
    Some(denominator)
}

fn modular_add(first: u128, second: u128, modulus: u128) -> u128 {
    if first >= modulus - second {
        first - (modulus - second)
    } else {
        first + second
    }
}

fn modular_multiply(mut first: u128, mut second: u128, modulus: u128) -> u128 {
    first %= modulus;
    let mut result = 0;
    while second != 0 {
        if second & 1 != 0 {
            result = modular_add(result, first, modulus);
        }
        first = modular_add(first, first, modulus);
        second >>= 1;
    }
    result
}

pub(crate) fn segments(values: PyReadonlyArray2<'_, f64>) -> PyResult<Vec<Segment>> {
    if values.shape()[1] != 2
        || values.as_array().iter().any(|v| !v.is_finite())
        || values.as_array().rows().into_iter().any(|r| r[0] < 0.0)
    {
        return Err(PyValueError::new_err("Invalid synth runtime envelope"));
    }
    Ok(values
        .as_array()
        .rows()
        .into_iter()
        .map(|r| Segment {
            frames: r[0],
            target: r[1],
        })
        .collect())
}

pub(crate) fn total_frames(segments: &[Segment]) -> f64 {
    segments.iter().map(|s| s.frames).sum()
}

pub(crate) fn envelope_value(initial: f64, segments: &[Segment], elapsed: f64) -> f64 {
    let mut boundary = 0.0;
    let mut entry = initial;
    for segment in segments {
        let end = boundary + segment.frames;
        if elapsed < end {
            return entry + (segment.target - entry) * (elapsed - boundary) / segment.frames;
        }
        boundary = end;
        entry = segment.target;
    }
    entry
}

fn named_contour_initial(definition: &NamedEnvelopeDefinition, at: f64) -> NamedContourState {
    let duration = total_frames(&definition.attack);
    NamedContourState {
        random_state: 0,
        playback: PlaybackState {
            at,
            coordinate: if duration == 0.0 { 1.0 } else { 0.0 },
            rate: if duration == 0.0 { 0.0 } else { 1.0 / duration },
            direction: 1.0,
            paused: false,
            age: 0.0,
        },
        start_value: definition.initial,
        released: false,
        pending_release: None,
        traversals: 0,
        complete_coordinate: None,
        idle: definition.event_started,
    }
}

fn named_contour_settled(
    definition: &NamedEnvelopeDefinition,
    state: &NamedContourState,
    frame: f64,
) -> NamedContourState {
    let mut settled = state.clone();
    if let Some((_, release_at)) = state
        .pending_release
        .filter(|(release_frame, _)| *release_frame <= frame)
    {
        let duration = total_frames(&definition.attack);
        let value = envelope_value(
            state.start_value,
            &definition.attack,
            named_contour_coordinate(definition, state, release_at) * duration,
        );
        let release_duration = total_frames(&definition.release);
        settled.start_value = value;
        settled.released = true;
        settled.pending_release = None;
        settled.traversals = 0;
        settled.complete_coordinate = None;
        settled.playback = PlaybackState {
            at: release_at,
            coordinate: if release_duration == 0.0 { 1.0 } else { 0.0 },
            rate: if release_duration == 0.0 {
                0.0
            } else {
                1.0 / release_duration
            },
            direction: state.playback.direction,
            paused: state.playback.paused,
            age: 0.0,
        };
    }
    settled
}

fn named_contour_value(
    definition: &NamedEnvelopeDefinition,
    state: &NamedContourState,
    at: f64,
    frame: f64,
) -> f64 {
    if definition.sample_hold.is_some() {
        return state.start_value;
    }
    if state.idle {
        return definition.initial;
    }
    let settled = named_contour_settled(definition, state, frame);
    let segments = if settled.released {
        &definition.release
    } else {
        &definition.attack
    };
    let duration = total_frames(segments);
    envelope_value(
        settled.start_value,
        segments,
        named_contour_coordinate(definition, &settled, at) * duration,
    )
}

fn repeat_crossings(
    playback: &PlaybackState,
    at: f64,
    loop_start: f64,
    loop_end: f64,
    remaining: usize,
) -> (usize, Option<f64>) {
    let width = loop_end - loop_start;
    let coordinate = playback.coordinate;
    let target = playback.coordinate_at(at);
    if playback.paused || playback.rate == 0.0 || target == coordinate {
        return (0, None);
    }
    let named = loop_start != 0.0 || loop_end != 1.0;
    let forward = playback.direction > 0.0;
    let first = if forward {
        if named {
            loop_end + (((coordinate - loop_end) / width).floor() + 1.0).max(0.0) * width
        } else {
            coordinate.floor() + 1.0
        }
    } else if named {
        loop_end + (((coordinate - loop_end) / width).ceil() - 1.0) * width
    } else {
        coordinate.ceil() - 1.0
    };
    if !forward && named && first < loop_end {
        return (0, None);
    }
    let crossed = if forward {
        (((target - first) / width + 1e-8).floor() + 1.0).max(0.0) as usize
    } else {
        (((first - target) / width + 1e-8).floor() + 1.0).max(0.0) as usize
    };
    let boundary = if crossed >= remaining {
        Some(first + (remaining - 1) as f64 * width * playback.direction)
    } else {
        None
    };
    (crossed.min(remaining), boundary)
}

fn repeat_boundary_coordinate(
    playback: usize,
    boundary: f64,
    direction: f64,
    loop_start: f64,
    loop_end: f64,
) -> f64 {
    if playback == 1 {
        if direction > 0.0 {
            loop_end
        } else {
            loop_start
        }
    } else {
        contour_coordinate(boundary, playback, loop_start, loop_end)
    }
}

fn named_contour_coordinate(
    definition: &NamedEnvelopeDefinition,
    state: &NamedContourState,
    at: f64,
) -> f64 {
    if let Some(coordinate) = state.complete_coordinate {
        return coordinate;
    }
    if !state.released {
        if let Some(count) = definition.repeat_count {
            let (_, boundary) = repeat_crossings(
                &state.playback,
                at,
                definition.loop_start,
                definition.loop_end,
                count - state.traversals,
            );
            if let Some(boundary) = boundary {
                return repeat_boundary_coordinate(
                    definition.playback,
                    boundary,
                    state.playback.direction,
                    definition.loop_start,
                    definition.loop_end,
                );
            }
        }
    }
    contour_coordinate(
        state.playback.coordinate_at(at),
        if state.released {
            0
        } else {
            definition.playback
        },
        definition.loop_start,
        definition.loop_end,
    )
}

fn contour_coordinate(coordinate: f64, playback: usize, start: f64, end: f64) -> f64 {
    if start != 0.0 || end != 1.0 {
        if coordinate < end {
            return coordinate.max(0.0);
        }
        let width = end - start;
        if playback == 1 {
            return start + (coordinate - end).rem_euclid(width);
        }
        if playback == 2 {
            let position = (coordinate - end).rem_euclid(2.0 * width);
            return if position <= width {
                end - position
            } else {
                start + position - width
            };
        }
    }
    match playback {
        0 => coordinate.clamp(0.0, 1.0),
        1 => coordinate.rem_euclid(1.0),
        2 => {
            let position = coordinate.rem_euclid(2.0);
            if position <= 1.0 {
                position
            } else {
                2.0 - position
            }
        }
        _ => unreachable!(),
    }
}

fn staged_value(definition: &StagedMotionDefinition, state: &StagedMotionState, at: f64) -> f64 {
    if state.completed {
        return state.complete_value;
    }
    let stage = &definition.stages[state.stage];
    match stage.kind {
        0 => stage.initial,
        1 => {
            let duration = total_frames(&stage.segments);
            let coordinate = state.playback.coordinate_at(at);
            let bounded = if stage
                .repeat_count
                .is_some_and(|count| state.traversals >= count)
            {
                repeat_boundary_coordinate(
                    stage.playback,
                    coordinate,
                    state.playback.direction,
                    stage.loop_start,
                    stage.loop_end,
                )
            } else {
                contour_coordinate(coordinate, stage.playback, stage.loop_start, stage.loop_end)
            };
            envelope_value(state.entry_value, &stage.segments, bounded * duration)
        }
        _ => {
            let cycle = &stage.cycle;
            let phase = state.playback.coordinate_at(at).rem_euclid(1.0);
            let age = state.playback.age_at(at);
            let weight = if age < cycle[4] {
                0.0
            } else if cycle[5] == 0.0 {
                1.0
            } else {
                ((age - cycle[4]) / cycle[5]).clamp(0.0, 1.0)
            };
            let shape = match cycle[0] as usize {
                0 => (TAU * phase).sin(),
                1 => {
                    if phase < cycle[3] {
                        1.0
                    } else {
                        -1.0
                    }
                }
                _ if phase < cycle[3] => 2.0 * phase / cycle[3] - 1.0,
                _ => (1.0 + cycle[3] - 2.0 * phase) / (1.0 - cycle[3]),
            };
            cycle[6] + cycle[7] * weight * shape
        }
    }
}

fn stage_playback(stage: &StageDefinition, at: f64, paused: bool, direction: f64) -> PlaybackState {
    let (coordinate, rate) = match stage.kind {
        0 => (0.0, 0.0),
        1 => (0.0, 1.0 / total_frames(&stage.segments)),
        _ => (stage.cycle[2], stage.cycle[1]),
    };
    PlaybackState {
        at,
        coordinate,
        rate,
        direction,
        paused,
        age: 0.0,
    }
}

fn staged_event_time(playback: &PlaybackState, target: f64) -> f64 {
    let at = playback.at + (target - playback.coordinate).abs() / playback.rate;
    let frame = at.round();
    if (at - frame).abs() < 1e-8 { frame } else { at }
}

fn next_staged_event(
    definition: &StagedMotionDefinition,
    state: &StagedMotionState,
    limit: f64,
    inclusive: bool,
) -> Option<(f64, usize, String)> {
    if state.completed {
        return None;
    }
    let stage = &definition.stages[state.stage];
    let playback = &state.playback;
    if playback.paused || playback.rate == 0.0 {
        return None;
    }
    let start = playback.coordinate;
    let forward = playback.direction > 0.0;
    let mut candidates = Vec::new();
    if stage.kind == 1 && stage.playback == 0 {
        for (order, (position, name)) in stage.markers.iter().enumerate() {
            if (if forward {
                *position > start
            } else {
                *position < start
            }) || (*position == start
                && state.cursor_order != usize::MAX
                && if forward {
                    order > state.cursor_order
                } else {
                    order < state.cursor_order
                })
            {
                candidates.push((staged_event_time(playback, *position), order, name.clone()));
            }
        }
        if forward
            && (start < 1.0
                || (start == 1.0
                    && state.cursor_order != usize::MAX
                    && state.cursor_order < stage.markers.len()))
        {
            candidates.push((
                staged_event_time(playback, 1.0),
                stage.markers.len(),
                "done".to_owned(),
            ));
        }
    } else if stage.kind == 1 && (stage.loop_start != 0.0 || stage.loop_end != 1.0) {
        let width = stage.loop_end - stage.loop_start;
        let period = if stage.playback == 1 {
            width
        } else {
            2.0 * width
        };
        for (order, (position, name)) in stage.markers.iter().enumerate() {
            let mut targets = vec![*position];
            let mut bases = Vec::new();
            if *position >= stage.loop_start {
                if stage.playback == 1 {
                    bases.push(stage.loop_end + position - stage.loop_start);
                } else {
                    if *position < stage.loop_end {
                        bases.push(2.0 * stage.loop_end - position);
                    }
                    if *position > stage.loop_start {
                        bases.push(stage.loop_end + width + position - stage.loop_start);
                    }
                }
            }
            for base in bases {
                let relative = (start - base) / period;
                let turn = if forward {
                    (relative.floor() + 1.0).max(0.0)
                } else {
                    relative.ceil() - 1.0
                };
                if turn >= 0.0 {
                    targets.push(base + turn * period);
                }
                if start >= base && (relative - relative.round()).abs() < 1e-8 {
                    targets.push(start);
                }
            }
            for target in targets {
                if (if forward {
                    target > start
                } else {
                    target < start
                }) || ((target - start).abs() < 1e-8
                    && state.cursor_order != usize::MAX
                    && if forward {
                        order > state.cursor_order
                    } else {
                        state.cursor_order < stage.markers.len() && order < state.cursor_order
                    })
                {
                    candidates.push((staged_event_time(playback, target), order, name.clone()));
                }
            }
        }
        let order = stage.markers.len();
        let turn = if forward {
            ((start - stage.loop_end) / width).floor() + 1.0
        } else {
            ((start - stage.loop_end) / width).ceil() - 1.0
        };
        let turn = if forward { turn.max(0.0) } else { turn };
        if turn >= 0.0 {
            let target = stage.loop_end + turn * width;
            candidates.push((
                staged_event_time(playback, target),
                order,
                if stage.playback == 1 {
                    "cycle"
                } else {
                    "turned"
                }
                .to_owned(),
            ));
            if stage.playback == 2 && turn.rem_euclid(2.0) == 1.0 {
                candidates.push((
                    staged_event_time(playback, target),
                    order + 1,
                    "cycle".to_owned(),
                ));
            }
        }
        let relative = (start - stage.loop_end) / width;
        if start + 1e-8 >= stage.loop_end && (relative - relative.round()).abs() < 1e-8 {
            if state.cursor_order < order {
                candidates.push((
                    playback.at,
                    order,
                    if stage.playback == 1 {
                        "cycle"
                    } else {
                        "turned"
                    }
                    .to_owned(),
                ));
            }
            if stage.playback == 2
                && relative.round().rem_euclid(2.0) == 1.0
                && state.cursor_order == order
            {
                candidates.push((playback.at, order + 1, "cycle".to_owned()));
            }
        }
    } else if stage.kind == 1 {
        let period = if stage.playback == 1 { 1.0 } else { 2.0 };
        for (order, (position, name)) in stage.markers.iter().enumerate() {
            let offsets = if stage.playback == 1 {
                vec![*position]
            } else {
                vec![*position, 2.0 - position]
            };
            for offset in offsets {
                let relative = (start - offset) / period;
                let turn = if forward {
                    relative.floor() + 1.0
                } else {
                    relative.ceil() - 1.0
                };
                let target = turn * period + offset;
                candidates.push((staged_event_time(playback, target), order, name.clone()));
                if state.cursor_order != usize::MAX
                    && (relative - relative.round()).abs() < 1e-8
                    && if forward {
                        order > state.cursor_order
                    } else {
                        order < state.cursor_order
                    }
                {
                    candidates.push((playback.at, order, name.clone()));
                }
            }
        }
        let boundary = if forward {
            start.floor() + 1.0
        } else {
            start.ceil() - 1.0
        };
        let order = stage.markers.len();
        candidates.push((
            staged_event_time(playback, boundary),
            order,
            if stage.playback == 1 {
                "cycle"
            } else {
                "turned"
            }
            .to_owned(),
        ));
        if stage.playback == 2 {
            if boundary.rem_euclid(2.0) == 0.0 {
                candidates.push((
                    staged_event_time(playback, boundary),
                    order + 1,
                    "cycle".to_owned(),
                ));
            }
            if start == start.round() && start.rem_euclid(2.0) == 0.0 && state.cursor_order == order
            {
                candidates.push((playback.at, order + 1, "cycle".to_owned()));
            }
        }
    } else if stage.kind == 2 {
        for (order, (position, name)) in stage.markers.iter().enumerate() {
            let offset = start - position;
            let turn = if forward {
                offset.floor() + 1.0
            } else {
                offset.ceil() - 1.0
            };
            let target = turn + position;
            candidates.push((staged_event_time(playback, target), order, name.clone()));
            if state.cursor_order != usize::MAX
                && offset.fract() == 0.0
                && if forward {
                    order > state.cursor_order
                } else {
                    order < state.cursor_order
                }
            {
                candidates.push((playback.at, order, name.clone()));
            }
        }
    }
    candidates
        .into_iter()
        .filter(|(at, _, _)| if inclusive { *at <= limit } else { *at < limit })
        .min_by(|a, b| {
            a.0.total_cmp(&b.0).then_with(|| {
                let order = |candidate: &(f64, usize, String)| {
                    if forward || candidate.2 == "turned" || candidate.2 == "cycle" {
                        candidate.1 as i64
                    } else {
                        -(candidate.1 as i64)
                    }
                };
                order(a).cmp(&order(b))
            })
        })
}

fn staged_transition(
    definition: &StagedMotionDefinition,
    state: &mut StagedMotionState,
    event: &str,
    at: f64,
) -> bool {
    if state.completed {
        return false;
    }
    let Some((_, _, target)) = definition
        .transitions
        .iter()
        .find(|(source, kind, _)| *source == state.stage && *kind == event)
    else {
        return false;
    };
    let value = staged_value(definition, state, at);
    if *target == definition.stages.len() {
        state.completed = true;
        state.complete_value = value;
        true
    } else {
        let stage = &definition.stages[*target];
        *state = StagedMotionState {
            stage: *target,
            playback: stage_playback(stage, at, state.playback.paused, state.playback.direction),
            entry_value: if stage.current_initial {
                value
            } else {
                stage.initial
            },
            completed: false,
            complete_value: 0.0,
            cursor_at: at,
            cursor_order: usize::MAX,
            traversals: 0,
        };
        false
    }
}

fn advance_ramp(value: &mut f64, step: f64, remaining: &mut usize) {
    if *remaining > 0 {
        *value += step;
        *remaining -= 1;
    }
}

fn valid_motion_owners(owners: &[usize]) -> bool {
    owners
        .iter()
        .enumerate()
        .all(|(index, owner)| *owner <= index && owners[*owner] == *owner)
}

fn queue_delayed_event(
    queue: &mut Vec<DelayedEvent>,
    at: f64,
    delay: f64,
    connection: DelayedConnection,
) -> PyResult<()> {
    if queue.len() == 4096 {
        return Err(PyValueError::new_err(
            "Delayed Motion event capacity exceeded",
        ));
    }
    let due = at + delay;
    if !due.is_finite() || due <= at {
        return Err(PyValueError::new_err(
            "Delayed Motion event time is not representable",
        ));
    }
    queue.push(DelayedEvent {
        at: due,
        connection,
    });
    Ok(())
}

fn forward_event(count: &mut usize, state: &mut u64, every: usize, probability: u64) -> bool {
    let forward = *count == 0;
    *count = if forward { every - 1 } else { *count - 1 };
    if !forward || probability == 0 {
        return false;
    }
    if probability == (1 << 53) {
        return true;
    }
    motion_random_word(state) >> 11 < probability
}

fn sample_hold_draw(definition: &NamedEnvelopeDefinition, state: &mut NamedContourState) {
    let (_, minimum, maximum) = definition.sample_hold.unwrap();
    let word = motion_random_word(&mut state.random_state);
    state.start_value =
        minimum + (maximum - minimum) * ((word >> 11) as f64 / (1_u64 << 53) as f64);
}

fn motion_random_word(state: &mut u64) -> u64 {
    *state = state.wrapping_add(0x9e3779b97f4a7c15);
    let mut word = (*state ^ (*state >> 30)).wrapping_mul(0xbf58476d1ce4e5b9);
    word = (word ^ (word >> 27)).wrapping_mul(0x94d049bb133111eb);
    word ^= word >> 31;
    word
}
