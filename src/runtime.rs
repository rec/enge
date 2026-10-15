//! Persistent oscillator runtime with sample-accurate voice actions.

use crate::amplitude::{Amplitude, AmplitudeSegments, AmplitudeState, frame_position, frame_time};
use envelfo_core::envelope::{
    Envelope, Initial, Marker, Playback, Retrigger, Segment as CoreSegment, Start,
};
use envelfo_core::lfo::Command;
use envelfo_core::patch::{ChildCommand, PatchState};
use envelfo_core::time::Time;
use envelfo_core::{lfo, stages};
use numpy::ndarray::{Array2, Array3, ArrayViewMut2};
use numpy::{IntoPyArray, PyArray2, PyReadonlyArray2, PyReadwriteArray2, PyUntypedArrayMethods};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use std::f64::consts::{PI, TAU};

use patch_host::{PatchHost, clock_at};

mod patch_host;

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
    body: lfo::LFO,
    parameter: usize,
    operation: usize,
    intercept: f64,
    slope: f64,
}

#[derive(Clone)]
struct LfoPlayback {
    core: lfo::LFOState,
    ordinal: u64,
}

#[derive(Clone, PartialEq)]
struct BeatPoint {
    song_beat: Time,
    clock_frame: Time,
    clock_beat: Time,
    clock_rate: Time,
}

#[derive(Clone, PartialEq)]
struct NamedEnvelopeDefinition {
    sample_hold: Option<(u64, f64, f64)>,
    body: Envelope,
    parameter: usize,
    operation: usize,
    intercept: f64,
    slope: f64,
    release_with_voice: bool,
    beat_clock: bool,
    event_started: bool,
}

#[derive(Clone, PartialEq)]
struct PatchEventConnection {
    threshold: bool,
    capture: bool,
    source: usize,
    marker: Time,
    destination: usize,
    cue: Option<String>,
    every: usize,
    offset: usize,
    probability: u64,
    key: u64,
    delay: Time,
}

#[derive(Clone, PartialEq)]
struct PatchStageStartConnection {
    capture: bool,
    source: usize,
    port: String,
    destination: usize,
    every: usize,
    offset: usize,
    probability: u64,
    key: u64,
    delay: Time,
}

#[derive(Clone, PartialEq)]
struct StagedEnvelfoDefinition {
    body: stages::Stages,
    parameter: Option<usize>,
    operation: usize,
    intercept: f64,
    slope: f64,
    beat_clock: bool,
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
    delay: Time,
}

type PatchEventInput = (
    usize,
    (i64, i64),
    usize,
    Option<String>,
    usize,
    usize,
    usize,
    u64,
    u64,
    (i64, i64),
    bool,
);
type StageConnectionInput = (
    usize,
    String,
    usize,
    String,
    usize,
    usize,
    u64,
    u64,
    (i64, i64),
);
type StageStartInput = (
    usize,
    String,
    usize,
    usize,
    usize,
    usize,
    u64,
    u64,
    (i64, i64),
    bool,
);

type EnvelfoTransformInput = (u8, Vec<(u8, usize, f64, f64)>, f64, f64, Option<f64>);
type EnvelfoTransformRoute = (usize, usize, usize, f64, f64);

type StageInput = (
    u8,
    f64,
    bool,
    AmplitudeSegments,
    Option<(usize, Vec<(i64, i64)>, f64, f64)>,
    Vec<((i64, i64), String)>,
    usize,
    (i64, i64),
    (i64, i64),
    Option<u64>,
);
type NamedEnvelopeInput = (
    usize,
    usize,
    f64,
    f64,
    bool,
    usize,
    (i64, i64),
    (i64, i64),
    Option<u64>,
    bool,
    (bool, usize, bool),
);
type BeatClockInput = ((i64, i64), (i64, i64), (i64, i64), (i64, i64));
type StagedEnvelfoInput = (
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
    patch_host: Option<PatchHost>,
    patch_states: Vec<Option<PatchState>>,
    pending_releases: Vec<Vec<Option<Time>>>,
    envelfo_transforms: Vec<EnvelfoTransformInput>,
    envelfo_transform_routes: Vec<EnvelfoTransformRoute>,
    rate: f64,
    waveform: u8,
    source_kind: u8,
    duty: f64,
    fm_phase_offsets: [f64; 2],
    graph_operators: Vec<GraphOperator>,
    graph_edges: Vec<GraphEdge>,
    graph_order: Vec<usize>,
    graph_carrier: usize,
    graph_carrier_parameter: usize,
    minimum_hold_frames: Time,
    routes: Array2<f64>,
    control_definitions: Vec<ControlDefinition>,
    control_states: Vec<ControlState>,
    lfo_definitions: Vec<LfoDefinition>,
    lfo_owners: Vec<usize>,
    lfo_event_states: Vec<LfoPlayback>,
    named_envelopes: Vec<NamedEnvelopeDefinition>,
    named_owners: Vec<usize>,
    patch_events: Vec<PatchEventConnection>,
    patch_stage_starts: Vec<PatchStageStartConnection>,
    envelfo_keys: Vec<u64>,
    beat_points: Vec<BeatPoint>,
    staged_envelfos: Vec<StagedEnvelfoDefinition>,
    staged_owners: Vec<usize>,
    staged_connections: Vec<StagedConnection>,
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
    graph_phases: Vec<f64>,
    graph_errors: Vec<f64>,
    graph_outputs: Vec<f64>,
    graph_history: Vec<f64>,
    noise_keys: Vec<u64>,
    noise_counters: Vec<u64>,
    gains: Vec<f64>,
    active: Vec<bool>,
    ages: Vec<usize>,
    release_frames: Vec<Option<Time>>,
    frequency_steps: Vec<f64>,
    frequency_remaining: Vec<usize>,
    gain_steps: Vec<f64>,
    gain_remaining: Vec<usize>,
    voice_part_contexts: Vec<usize>,
    voice_trigger_contexts: Vec<usize>,
    filter_states: Array2<f64>,
    voice_filter_states: Array3<f64>,
    frame: usize,
    amplitude_definitions: Vec<Amplitude>,
    amplitude_states: Vec<Vec<AmplitudeState>>,
}

#[pyclass(skip_from_py_object)]
#[derive(Clone)]
pub struct SynthRuntime {
    patch_host: Option<PatchHost>,
    patch_states: Vec<Option<PatchState>>,
    pending_releases: Vec<Vec<Option<Time>>>,
    envelfo_transforms: Vec<EnvelfoTransformInput>,
    envelfo_transform_routes: Vec<EnvelfoTransformRoute>,
    envelfo_transform_values: Vec<f64>,
    rate: f64,
    filter_after_amplitude: bool,
    waveform: u8,
    source_kind: u8,
    duty: f64,
    fm_phase_offsets: [f64; 2],
    graph_operators: Vec<GraphOperator>,
    graph_edges: Vec<GraphEdge>,
    graph_order: Vec<usize>,
    graph_carrier: usize,
    graph_carrier_parameter: usize,
    minimum_hold_frames: Time,
    routes: Array2<f64>,
    control_definitions: Vec<ControlDefinition>,
    control_states: Vec<ControlState>,
    lfo_definitions: Vec<LfoDefinition>,
    lfo_owners: Vec<usize>,
    lfo_event_states: Vec<LfoPlayback>,
    named_envelopes: Vec<NamedEnvelopeDefinition>,
    named_owners: Vec<usize>,
    patch_events: Vec<PatchEventConnection>,
    patch_stage_starts: Vec<PatchStageStartConnection>,
    envelfo_keys: Vec<u64>,
    beat_points: Vec<BeatPoint>,
    staged_envelfos: Vec<StagedEnvelfoDefinition>,
    staged_owners: Vec<usize>,
    staged_connections: Vec<StagedConnection>,
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
    graph_phases: Vec<f64>,
    graph_errors: Vec<f64>,
    graph_outputs: Vec<f64>,
    graph_history: Vec<f64>,
    noise_keys: Vec<u64>,
    noise_counters: Vec<u64>,
    gains: Vec<f64>,
    active: Vec<bool>,
    ages: Vec<usize>,
    release_frames: Vec<Option<Time>>,
    frequency_steps: Vec<f64>,
    frequency_remaining: Vec<usize>,
    gain_steps: Vec<f64>,
    gain_remaining: Vec<usize>,
    voice_part_contexts: Vec<usize>,
    voice_trigger_contexts: Vec<usize>,
    filter_states: Array2<f64>,
    voice_filter_states: Array3<f64>,
    frame: usize,
    amplitude_definitions: Vec<Amplitude>,
    amplitude_states: Vec<Vec<AmplitudeState>>,
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
        attack: AmplitudeSegments,
        release: AmplitudeSegments,
        minimum_hold_frames: (i64, i64),
        controls: PyReadonlyArray2<'_, f64>,
        control_smoothing: Vec<(u64, u64)>,
        lfos: PyReadonlyArray2<'_, f64>,
        lfo_rationals: Vec<(u64, u64)>,
        filters: PyReadonlyArray2<'_, f64>,
        parameters: Vec<f64>,
        context_capacity: usize,
        filter_after_amplitude: bool,
    ) -> PyResult<Self> {
        let minimum_hold_frames = ratio_time(minimum_hold_frames)?;
        let amplitude = Amplitude::new(initial, attack, release)?;
        let (control_definitions, control_states) =
            control_definitions(controls, &control_smoothing)?;
        let lfo_definitions = lfo_definitions(lfos, &lfo_rationals, rate)?;
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
            || minimum_hold_frames < Time::ZERO
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
        {
            return Err(PyValueError::new_err("Invalid synth runtime definition"));
        }
        let slots = routes.shape()[0];
        let channels = routes.shape()[1];
        let filter_stages = filter_definitions.iter().map(|v| v.stages).sum();
        let lfo_count = lfo_definitions.len();
        let lfo_states = lfo_definitions
            .iter()
            .map(|d| initial_lfo(d, Time::ZERO))
            .collect::<PyResult<Vec<_>>>()?;
        let context_states = (0..context_capacity)
            .flat_map(|_| control_states.clone())
            .collect();
        let amplitude_states = (0..slots)
            .map(|_| Ok(vec![amplitude.start()?]))
            .collect::<PyResult<_>>()?;
        Ok(Self {
            patch_host: None,
            patch_states: vec![None; slots],
            pending_releases: vec![vec![]; slots],
            envelfo_transforms: Vec::new(),
            envelfo_transform_routes: Vec::new(),
            envelfo_transform_values: Vec::new(),
            rate,
            waveform,
            source_kind: 0,
            amplitude_definitions: vec![amplitude],
            amplitude_states,
            filter_after_amplitude,
            duty,
            fm_phase_offsets: [0.0; 2],
            graph_operators: Vec::new(),
            graph_edges: Vec::new(),
            graph_order: Vec::new(),
            graph_carrier: 0,
            graph_carrier_parameter: 0,
            minimum_hold_frames,
            routes: routes.as_array().to_owned(),
            control_definitions,
            control_states,
            lfo_definitions,
            lfo_owners: (0..lfo_count).collect(),
            lfo_event_states: lfo_states,
            named_envelopes: Vec::new(),
            named_owners: Vec::new(),
            patch_events: Vec::new(),
            patch_stage_starts: Vec::new(),
            envelfo_keys: vec![0; slots],
            beat_points: Vec::new(),
            staged_envelfos: Vec::new(),
            staged_owners: Vec::new(),
            staged_connections: Vec::new(),
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
            graph_phases: Vec::new(),
            graph_errors: Vec::new(),
            graph_outputs: Vec::new(),
            graph_history: Vec::new(),
            noise_keys: vec![0; slots],
            noise_counters: vec![0; slots],
            gains: vec![0.0; slots],
            active: vec![false; slots],
            ages: vec![0; slots],
            release_frames: vec![None; slots],
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
        carrier_attack: AmplitudeSegments,
        carrier_release: AmplitudeSegments,
        mod_initial: f64,
        mod_attack: AmplitudeSegments,
        mod_release: AmplitudeSegments,
        phase_offsets: Vec<f64>,
        minimum_hold_frames: (i64, i64),
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
        let mod_envelope = Amplitude::new(mod_initial, mod_attack, mod_release)?;
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
        runtime.amplitude_definitions.push(mod_envelope);
        runtime.amplitude_states = (0..runtime.frequencies.len())
            .map(|_| {
                runtime
                    .amplitude_definitions
                    .iter()
                    .map(Amplitude::start)
                    .collect()
            })
            .collect::<PyResult<_>>()?;
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
        carrier_attack: AmplitudeSegments,
        carrier_release: AmplitudeSegments,
        waveforms: Vec<u8>,
        initials: Vec<f64>,
        attacks: Vec<AmplitudeSegments>,
        releases: Vec<AmplitudeSegments>,
        phase_offsets: Vec<f64>,
        parameters: Vec<(usize, usize)>,
        edges: Vec<(usize, usize, bool, usize)>,
        order: Vec<usize>,
        carrier: usize,
        carrier_parameter: usize,
        minimum_hold_frames: (i64, i64),
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
        let mut amplitude_definitions = Vec::with_capacity(operators);
        let graph_operators: Vec<GraphOperator> = waveforms
            .into_iter()
            .zip(initials)
            .zip(attacks)
            .zip(releases)
            .zip(phase_offsets)
            .zip(parameters)
            .map(
                |(((((waveform, initial), attack), release), phase_offset), parameters)| {
                    if parameters.0 >= parameter_definitions.len() / 3
                        || parameters.1 >= parameter_definitions.len() / 3
                    {
                        return Err(PyValueError::new_err("Invalid graph FM operator"));
                    }
                    amplitude_definitions.push(Amplitude::new(initial, attack, release)?);
                    Ok(GraphOperator {
                        waveform,
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
        runtime.amplitude_definitions = amplitude_definitions;
        runtime.amplitude_states = (0..slots)
            .map(|_| {
                runtime
                    .amplitude_definitions
                    .iter()
                    .map(Amplitude::start)
                    .collect()
            })
            .collect::<PyResult<_>>()?;
        Ok(runtime)
    }

    fn set_named_envelopes(
        &mut self,
        initials: Vec<f64>,
        attacks: Vec<AmplitudeSegments>,
        releases: Vec<AmplitudeSegments>,
        parameters: Vec<NamedEnvelopeInput>,
    ) -> PyResult<()> {
        let count = initials.len();
        if attacks.len() != count || releases.len() != count || parameters.len() != count {
            return Err(PyValueError::new_err("Invalid named envelope definitions"));
        }
        let definitions = initials
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
                        (hold, retrigger, bipolar),
                    ),
                )| {
                    if parameter >= self.parameter_definitions.len() / 3
                        || operation > 1
                        || !intercept.is_finite()
                        || !slope.is_finite()
                    {
                        return Err(PyValueError::new_err("Invalid named envelope definition"));
                    }
                    let convert = |parts: AmplitudeSegments| -> PyResult<Vec<CoreSegment>> {
                        parts
                            .into_iter()
                            .map(|(duration, to)| {
                                Ok(CoreSegment {
                                    duration: duration
                                        .map(|(n, d)| Time::new(n, d))
                                        .transpose()
                                        .map_err(core_error)?,
                                    to,
                                    curve: 0.0,
                                })
                            })
                            .collect()
                    };
                    let start = Time::new(loop_start.0, loop_start.1).map_err(core_error)?;
                    let end = Time::new(loop_end.0, loop_end.1).map_err(core_error)?;
                    let bounds = if start == Time::ZERO && end == Time::ONE {
                        None
                    } else {
                        Some((start, end))
                    };
                    let mut segments = convert(attack)?;
                    // SampleHold shares the host route slots but does not execute this body.
                    if segments.is_empty() {
                        segments.push(CoreSegment {
                            duration: Some(Time::ZERO),
                            to: initial,
                            curve: 0.0,
                        });
                    }
                    let body = Envelope {
                        initial: Initial::Value(initial),
                        segments,
                        release: convert(release)?,
                        bipolar,
                        hold,
                        retrigger: match retrigger {
                            0 => Retrigger::Current,
                            1 => Retrigger::Initial,
                            2 => Retrigger::Ignore,
                            _ => {
                                return Err(PyValueError::new_err(
                                    "Invalid named envelope retrigger",
                                ));
                            }
                        },
                        start: if event_started {
                            Start::Event
                        } else {
                            Start::Activation
                        },
                        playback: match playback {
                            0 => Playback::Once,
                            1 => Playback::Loop,
                            2 => Playback::PingPong,
                            _ => {
                                return Err(PyValueError::new_err(
                                    "Invalid named envelope playback",
                                ));
                            }
                        },
                        markers: bounds
                            .map(|(start, end)| {
                                vec![
                                    Marker {
                                        name: "loop_start".into(),
                                        position: start,
                                    },
                                    Marker {
                                        name: "loop_end".into(),
                                        position: end,
                                    },
                                ]
                            })
                            .unwrap_or_default(),
                        loop_bounds: bounds,
                        repeat_count,
                    };
                    body.validate().map_err(core_error)?;
                    Ok(NamedEnvelopeDefinition {
                        sample_hold: None,
                        body,
                        parameter,
                        operation,
                        intercept,
                        slope,
                        release_with_voice,
                        beat_clock: false,
                        event_started,
                    })
                },
            )
            .collect::<PyResult<Vec<_>>>()?;
        self.named_envelopes = definitions;
        self.named_owners = (0..count).collect();
        self.pending_releases = vec![vec![None; count]; self.frequencies.len()];
        Ok(())
    }

    fn set_beat_clock(&mut self, points: Vec<BeatClockInput>) -> PyResult<()> {
        let points = points
            .into_iter()
            .map(|(frame, beat, song, rate)| {
                let clock_frame = Time::new(frame.0, frame.1).map_err(core_error)?;
                let clock_beat = Time::new(beat.0, beat.1).map_err(core_error)?;
                let clock_rate = Time::new(rate.0, rate.1).map_err(core_error)?;
                let song_beat = Time::new(song.0, song.1).map_err(core_error)?;
                Ok(BeatPoint {
                    song_beat,
                    clock_frame,
                    clock_beat,
                    clock_rate,
                })
            })
            .collect::<PyResult<Vec<_>>>()?;
        if self.frame != 0
            || points.is_empty()
            || points[0].clock_frame != Time::ZERO
            || points
                .iter()
                .any(|p| p.clock_beat < Time::ZERO || p.clock_rate < Time::ZERO)
            || points
                .windows(2)
                .any(|p| p[1].clock_frame <= p[0].clock_frame)
        {
            return Err(PyValueError::new_err("Invalid native beat clock"));
        }
        self.beat_points = points;
        Ok(())
    }

    fn set_beat_named_envelopes(&mut self, indices: Vec<usize>) -> PyResult<()> {
        if self.frame != 0
            || self.beat_points.is_empty()
            || indices
                .iter()
                .any(|index| *index >= self.named_envelopes.len())
        {
            return Err(PyValueError::new_err("Invalid beat-clock named Envelfo"));
        }
        for index in indices {
            self.named_envelopes[index].beat_clock = true;
        }
        Ok(())
    }

    fn set_staged_envelfos(
        &mut self,
        inputs: Vec<StagedEnvelfoInput>,
        connections: Vec<StageConnectionInput>,
    ) -> PyResult<()> {
        let mut definitions = Vec::with_capacity(inputs.len());
        for (
            inputs,
            transitions,
            initial_stage,
            parameter,
            operation,
            intercept,
            slope,
            beat_clock,
        ) in inputs
        {
            if inputs.is_empty()
                || initial_stage >= inputs.len()
                || (beat_clock && self.beat_points.is_empty())
                || parameter.is_some_and(|p| p >= self.parameter_definitions.len() / 3)
                || operation > 1
                || !intercept.is_finite()
                || !slope.is_finite()
                || transitions
                    .iter()
                    .any(|(source, _, target)| *source >= inputs.len() || *target > inputs.len())
            {
                return Err(PyValueError::new_err("Invalid staged Envelfo definition"));
            }
            let mut bodies = Vec::with_capacity(inputs.len());
            for (
                index,
                (kind, initial, current, parts, lfo, markers, playback, start, end, repeats),
            ) in inputs.into_iter().enumerate()
            {
                let markers = markers
                    .into_iter()
                    .map(|(position, name)| {
                        Ok(Marker {
                            name,
                            position: ratio_time(position)?,
                        })
                    })
                    .collect::<PyResult<Vec<_>>>()?;
                let body = match kind {
                    1 => {
                        if lfo.is_some() {
                            return Err(PyValueError::new_err("Invalid staged Envelope"));
                        }
                        let start = ratio_time(start)?;
                        let end = ratio_time(end)?;
                        stages::Body::Envelope(Envelope {
                            initial: if current {
                                Initial::Current
                            } else {
                                Initial::Value(initial)
                            },
                            segments: parts
                                .into_iter()
                                .map(|(duration, to)| {
                                    Ok(CoreSegment {
                                        duration: duration.map(ratio_time).transpose()?,
                                        to,
                                        curve: 0.0,
                                    })
                                })
                                .collect::<PyResult<_>>()?,
                            release: vec![],
                            bipolar: true,
                            hold: true,
                            retrigger: Retrigger::Current,
                            start: Start::Activation,
                            playback: match playback {
                                0 => Playback::Once,
                                1 => Playback::Loop,
                                2 => Playback::PingPong,
                                _ => {
                                    return Err(PyValueError::new_err(
                                        "Invalid staged Envelope playback",
                                    ));
                                }
                            },
                            loop_bounds: if start == Time::ZERO && end == Time::ONE {
                                None
                            } else {
                                Some((start, end))
                            },
                            repeat_count: repeats,
                            markers: markers.clone(),
                        })
                    }
                    2 => {
                        if current || !parts.is_empty() || playback != 0 || repeats.is_some() {
                            return Err(PyValueError::new_err("Invalid staged LFO"));
                        }
                        let Some((shape, times, center, depth)) = lfo else {
                            return Err(PyValueError::new_err("Invalid staged LFO"));
                        };
                        let times = times
                            .into_iter()
                            .map(ratio_time)
                            .collect::<PyResult<Vec<_>>>()?;
                        if times.len() != 5 {
                            return Err(PyValueError::new_err("Invalid staged LFO timing"));
                        }
                        let mut body = lfo::LFO::new(times[0]).map_err(core_error)?;
                        body.shape = match shape {
                            0 => lfo::Shape::Sine,
                            1 => lfo::Shape::Square,
                            2 => lfo::Shape::Triangle,
                            _ => return Err(PyValueError::new_err("Invalid staged LFO shape")),
                        };
                        body.phase = times[1];
                        body.duty = times[2];
                        body.delay = times[3];
                        body.fade = times[4];
                        body.center = center;
                        body.depth = depth;
                        stages::Body::LFO(body)
                    }
                    _ => return Err(PyValueError::new_err("Invalid staged Envelfo stage")),
                };
                bodies.push(stages::Stage {
                    name: index.to_string(),
                    markers: if kind == 2 { markers } else { vec![] },
                    body,
                });
            }
            let count = bodies.len();
            let body = stages::Stages {
                initial_stage: initial_stage.to_string(),
                stages: bodies,
                transitions: transitions
                    .into_iter()
                    .map(|(source, event, target)| stages::Transition {
                        from: vec![source.to_string()],
                        event,
                        action: if target == count {
                            stages::Action::Finish
                        } else {
                            stages::Action::Enter(target.to_string())
                        },
                    })
                    .collect(),
            };
            body.validate().map_err(core_error)?;
            definitions.push(StagedEnvelfoDefinition {
                body,
                parameter,
                operation,
                intercept,
                slope,
                beat_clock,
            });
        }
        let connections = connections
            .into_iter()
            .map(
                |(source, port, destination, cue, every, offset, probability, key, delay)| {
                    let delay = ratio_time(delay)?;
                    if source >= definitions.len()
                        || destination >= definitions.len()
                        || port.is_empty()
                        || cue.is_empty()
                        || every == 0
                        || probability > (1 << 53)
                        || delay < Time::ZERO
                    {
                        return Err(PyValueError::new_err("Invalid staged Envelfo connection"));
                    }
                    Ok(StagedConnection {
                        source,
                        port,
                        destination,
                        cue,
                        every,
                        offset,
                        probability,
                        key,
                        delay,
                    })
                },
            )
            .collect::<PyResult<_>>()?;
        self.staged_envelfos = definitions;
        self.staged_owners = (0..self.staged_envelfos.len()).collect();
        self.staged_connections = connections;
        Ok(())
    }

    fn set_sample_hold_envelfos(
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

    fn set_envelfo_state_owners(
        &mut self,
        lfos: Vec<usize>,
        envelopes: Vec<usize>,
        stages: Vec<usize>,
    ) -> PyResult<()> {
        if self.frame != 0
            || lfos.len() != self.lfo_definitions.len()
            || envelopes.len() != self.named_envelopes.len()
            || stages.len() != self.staged_envelfos.len()
            || !valid_envelfo_owners(&lfos)
            || !valid_envelfo_owners(&envelopes)
            || !valid_envelfo_owners(&stages)
        {
            return Err(PyValueError::new_err("Invalid shared Envelfo state owners"));
        }
        self.lfo_owners = lfos;
        self.named_owners = envelopes;
        self.staged_owners = stages;
        Ok(())
    }

    fn set_envelfo_transforms(
        &mut self,
        nodes: Vec<EnvelfoTransformInput>,
        routes: Vec<EnvelfoTransformRoute>,
    ) -> PyResult<()> {
        if self.frame != 0
            || nodes
                .iter()
                .enumerate()
                .any(|(node, (kind, inputs, scale, offset, initial))| {
                    *kind > 6
                        || !scale.is_finite()
                        || !offset.is_finite()
                        || inputs.len() < if *kind >= 2 { 1 } else { 2 }
                        || (*kind >= 2 && inputs.len() != 1)
                        || (*kind == 3 && scale >= offset)
                        || (*kind == 4 && (*scale < 0.0 || *offset < 0.0))
                        || (*kind == 5 && *scale <= 0.0)
                        || initial.is_some_and(|value| *kind != 4 || !value.is_finite())
                        || inputs.iter().any(|(kind, index, offset, scale)| {
                            !offset.is_finite()
                                || !scale.is_finite()
                                || match kind {
                                    0 => *index >= self.lfo_definitions.len(),
                                    1 => *index >= self.named_envelopes.len(),
                                    2 => *index >= self.staged_envelfos.len(),
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
            return Err(PyValueError::new_err("Invalid Envelfo transform graph"));
        }
        self.envelfo_transform_values = vec![0.0; self.frequencies.len() * nodes.len()];
        self.envelfo_transforms = nodes;
        self.envelfo_transform_routes = routes;
        Ok(())
    }

    fn set_patch_events(&mut self, connections: Vec<PatchEventInput>) -> PyResult<()> {
        if self.frame != 0
            || connections.iter().any(
                |(
                    source,
                    marker,
                    destination,
                    cue,
                    _,
                    every,
                    _,
                    probability,
                    _,
                    delay,
                    capture,
                )| {
                    *every == 0
                        || *probability > (1 << 53)
                        || ratio_time(*delay).is_err()
                        || ratio_time(*delay).is_ok_and(|d| d < Time::ZERO)
                        || *source >= self.lfo_definitions.len()
                        || self.lfo_definitions[*source].scope != 3
                        || self.lfo_definitions[*source].body.rate == Time::ZERO
                        || self.lfo_definitions[*source].body.clock == lfo::Clock::Beats
                        || self.lfo_definitions[*source].body.driver == lfo::Driver::Transport
                        || ratio_time(*marker).is_err()
                        || ratio_time(*marker).is_ok_and(|m| m < Time::ZERO || m >= Time::ONE)
                        || if *capture {
                            cue.is_some()
                                || *destination >= self.envelfo_transforms.len()
                                || self.envelfo_transforms[*destination].0 != 6
                        } else if cue.is_some() {
                            *destination >= self.staged_envelfos.len()
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
                    _order,
                    every,
                    offset,
                    probability,
                    key,
                    delay,
                    capture,
                )| {
                    Ok(PatchEventConnection {
                        threshold: false,
                        capture,
                        source: self.lfo_owners[source],
                        marker: ratio_time(marker)?,
                        destination: if capture {
                            destination
                        } else if cue.is_some() {
                            self.staged_owners[destination]
                        } else {
                            self.named_owners[destination]
                        },
                        cue,
                        every,
                        offset,
                        probability,
                        key,
                        delay: ratio_time(delay)?,
                    })
                },
            )
            .collect::<PyResult<_>>()?;
        Ok(())
    }

    fn set_patch_stage_starts(&mut self, connections: Vec<StageStartInput>) -> PyResult<()> {
        if self.frame != 0
            || connections.iter().any(
                |(source, port, destination, _, every, _, probability, _, delay, capture)| {
                    *source >= self.staged_envelfos.len()
                        || *every == 0
                        || *probability > (1 << 53)
                        || ratio_time(*delay).is_err()
                        || ratio_time(*delay).is_ok_and(|d| d < Time::ZERO)
                        || port.is_empty()
                        || if *capture {
                            *destination >= self.envelfo_transforms.len()
                                || self.envelfo_transforms[*destination].0 != 6
                        } else {
                            *destination >= self.named_envelopes.len()
                                || !self.named_envelopes[*destination].event_started
                        }
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
                |(
                    source,
                    port,
                    destination,
                    _order,
                    every,
                    offset,
                    probability,
                    key,
                    delay,
                    capture,
                )| {
                    Ok(PatchStageStartConnection {
                        capture,
                        source: self.staged_owners[source],
                        port,
                        destination: if capture {
                            destination
                        } else {
                            self.named_owners[destination]
                        },
                        every,
                        offset,
                        probability,
                        key,
                        delay: ratio_time(delay)?,
                    })
                },
            )
            .collect::<PyResult<_>>()?;
        Ok(())
    }

    fn set_threshold_events(&mut self, connections: Vec<PatchEventInput>) -> PyResult<()> {
        if self.frame != 0
            || connections.iter().any(
                |(
                    source,
                    marker,
                    destination,
                    cue,
                    _,
                    every,
                    _,
                    probability,
                    _,
                    delay,
                    capture,
                )| {
                    *source >= self.envelfo_transforms.len()
                        || self.envelfo_transforms[*source].0 != 3
                        || (*marker != (0, 1) && *marker != (1, 1))
                        || *every == 0
                        || *probability > (1 << 53)
                        || ratio_time(*delay).is_err()
                        || ratio_time(*delay).is_ok_and(|d| d < Time::ONE)
                        || if *capture {
                            cue.is_some()
                                || *destination >= self.envelfo_transforms.len()
                                || self.envelfo_transforms[*destination].0 != 6
                        } else if cue.is_some() {
                            *destination >= self.staged_envelfos.len()
                        } else {
                            *destination >= self.named_envelopes.len()
                                || !self.named_envelopes[*destination].event_started
                        }
                },
            )
        {
            return Err(PyValueError::new_err("Invalid threshold event connections"));
        }
        let definitions = connections
            .into_iter()
            .map(
                |(
                    source,
                    marker,
                    destination,
                    cue,
                    _order,
                    every,
                    offset,
                    probability,
                    key,
                    delay,
                    capture,
                )| {
                    Ok(PatchEventConnection {
                        threshold: true,
                        capture,
                        source,
                        marker: ratio_time(marker)?,
                        destination: if capture {
                            destination
                        } else if cue.is_some() {
                            self.staged_owners[destination]
                        } else {
                            self.named_owners[destination]
                        },
                        cue,
                        every,
                        offset,
                        probability,
                        key,
                        delay: ratio_time(delay)?,
                    })
                },
            )
            .collect::<PyResult<Vec<_>>>()?;
        self.patch_events.extend(definitions);
        Ok(())
    }

    #[staticmethod]
    #[allow(clippy::too_many_arguments)]
    #[pyo3(signature = (rate, routes, initial, attack, release, minimum_hold_frames, controls, control_smoothing, lfos, lfo_rationals, filters, parameters, context_capacity, filter_after_amplitude=false))]
    fn noise(
        rate: f64,
        routes: PyReadonlyArray2<'_, f64>,
        initial: f64,
        attack: AmplitudeSegments,
        release: AmplitudeSegments,
        minimum_hold_frames: (i64, i64),
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
            patch_host: self.patch_host.clone(),
            patch_states: self.patch_states.clone(),
            pending_releases: self.pending_releases.clone(),
            amplitude_definitions: self.amplitude_definitions.clone(),
            amplitude_states: self.amplitude_states.clone(),
            envelfo_transforms: self.envelfo_transforms.clone(),
            envelfo_transform_routes: self.envelfo_transform_routes.clone(),
            rate: self.rate,
            waveform: self.waveform,
            source_kind: self.source_kind,
            duty: self.duty,
            fm_phase_offsets: self.fm_phase_offsets,
            graph_operators: self.graph_operators.clone(),
            graph_edges: self.graph_edges.clone(),
            graph_order: self.graph_order.clone(),
            graph_carrier: self.graph_carrier,
            graph_carrier_parameter: self.graph_carrier_parameter,
            minimum_hold_frames: self.minimum_hold_frames,
            routes: self.routes.clone(),
            control_definitions: self.control_definitions.clone(),
            control_states: self.control_states.clone(),
            lfo_definitions: self.lfo_definitions.clone(),
            lfo_owners: self.lfo_owners.clone(),
            lfo_event_states: self.lfo_event_states.clone(),
            named_envelopes: self.named_envelopes.clone(),
            named_owners: self.named_owners.clone(),
            patch_events: self.patch_events.clone(),
            patch_stage_starts: self.patch_stage_starts.clone(),
            envelfo_keys: self.envelfo_keys.clone(),
            beat_points: self.beat_points.clone(),
            staged_envelfos: self.staged_envelfos.clone(),
            staged_owners: self.staged_owners.clone(),
            staged_connections: self.staged_connections.clone(),
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
            graph_phases: self.graph_phases.clone(),
            graph_errors: self.graph_errors.clone(),
            graph_outputs: self.graph_outputs.clone(),
            graph_history: self.graph_history.clone(),
            noise_keys: self.noise_keys.clone(),
            noise_counters: self.noise_counters.clone(),
            gains: self.gains.clone(),
            active: self.active.clone(),
            ages: self.ages.clone(),
            release_frames: self.release_frames.clone(),
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
        self.patch_host.clone_from(&snapshot.patch_host);
        self.patch_states.clone_from(&snapshot.patch_states);
        self.pending_releases.clone_from(&snapshot.pending_releases);
        self.amplitude_states.clone_from(&snapshot.amplitude_states);
        self.frequencies.clone_from(&snapshot.frequencies);
        self.phases.clone_from(&snapshot.phases);
        self.errors.clone_from(&snapshot.errors);
        self.mod_phases.clone_from(&snapshot.mod_phases);
        self.mod_errors.clone_from(&snapshot.mod_errors);
        self.previous_modulators
            .clone_from(&snapshot.previous_modulators);
        self.envelfo_keys.clone_from(&snapshot.envelfo_keys);
        self.graph_phases.clone_from(&snapshot.graph_phases);
        self.graph_errors.clone_from(&snapshot.graph_errors);
        self.graph_outputs.clone_from(&snapshot.graph_outputs);
        self.graph_history.clone_from(&snapshot.graph_history);
        self.noise_keys.clone_from(&snapshot.noise_keys);
        self.noise_counters.clone_from(&snapshot.noise_counters);
        self.gains.clone_from(&snapshot.gains);
        self.active.clone_from(&snapshot.active);
        self.ages.clone_from(&snapshot.ages);
        self.release_frames.clone_from(&snapshot.release_frames);
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
    pub(crate) fn validate_snapshot(&self, snapshot: &SynthRuntimeSnapshot) -> PyResult<()> {
        if self.rate != snapshot.rate
            || self.waveform != snapshot.waveform
            || self.source_kind != snapshot.source_kind
            || self.amplitude_definitions != snapshot.amplitude_definitions
            || self.duty != snapshot.duty
            || self.fm_phase_offsets != snapshot.fm_phase_offsets
            || self.graph_operators != snapshot.graph_operators
            || self.graph_edges != snapshot.graph_edges
            || self.graph_order != snapshot.graph_order
            || self.graph_carrier != snapshot.graph_carrier
            || self.graph_carrier_parameter != snapshot.graph_carrier_parameter
            || self.minimum_hold_frames != snapshot.minimum_hold_frames
            || self.routes != snapshot.routes
            || self.control_definitions != snapshot.control_definitions
            || self.lfo_definitions != snapshot.lfo_definitions
            || self.lfo_owners != snapshot.lfo_owners
            || self.named_envelopes != snapshot.named_envelopes
            || self.named_owners != snapshot.named_owners
            || self.envelfo_transforms != snapshot.envelfo_transforms
            || self.envelfo_transform_routes != snapshot.envelfo_transform_routes
            || self.patch_events != snapshot.patch_events
            || self.patch_stage_starts != snapshot.patch_stage_starts
            || self.beat_points != snapshot.beat_points
            || self.staged_envelfos != snapshot.staged_envelfos
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

    fn lfo_time(&self, definition: &LfoDefinition, frame: Time) -> PyResult<Time> {
        if definition.body.driver == lfo::Driver::Transport {
            clock_at(&self.beat_points, 2, frame).map_err(core_error)
        } else if definition.body.clock == lfo::Clock::Beats {
            clock_at(&self.beat_points, 1, frame).map_err(core_error)
        } else {
            Ok(frame)
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
        self.prepare_patch()?;
        let frames = output.nrows();
        if frames == 0
            || output.ncols() != self.routes.ncols()
            || (self.beat_points.is_empty()
                && self
                    .lfo_definitions
                    .iter()
                    .any(|definition| definition.body.clock == lfo::Clock::Beats))
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
            if action < actions.len() && actions[action] as usize == frame {
                self.settle_patches(false)?;
            }
            while action < actions.len() && actions[action] as usize == frame {
                self.apply_action(&actions[action..action + 6], frames)?;
                action += 6;
            }
            self.settle_patches(true)?;
            for voice in 0..self.frequencies.len() {
                if !self.active[voice] {
                    continue;
                }
                self.observe_patch(voice)?;
                let amplitude = self.parameter(voice, 0)?
                    * if self.source_kind == 3 {
                        self.parameter(voice, self.graph_carrier_parameter)?
                    } else if self.source_kind == 1 {
                        self.parameter(voice, 8)?
                    } else {
                        1.0
                    };
                for (definition, state) in self
                    .amplitude_definitions
                    .iter()
                    .zip(&mut self.amplitude_states[voice])
                {
                    definition.settle(state, self.release_frames[voice], self.ages[voice])?;
                }
                let definition = &self.amplitude_definitions[self.graph_carrier];
                let state = &self.amplitude_states[voice][self.graph_carrier];
                let observed = definition.value(state, self.ages[voice])?;
                if definition.complete(state, &observed) {
                    self.active[voice] = false;
                    self.patch_states[voice] = None;
                    self.pending_releases[voice].fill(None);
                    continue;
                }
                let level = observed.value;
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
                    let definition = &self.amplitude_definitions[1];
                    let state = &self.amplitude_states[voice][1];
                    let value = definition.value(state, self.ages[voice])?;
                    let mod_level = if definition.complete(state, &value) {
                        0.0
                    } else {
                        value.value
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
                        let amplitude = &self.amplitude_definitions[*operator];
                        let state = &self.amplitude_states[voice][*operator];
                        let observed = amplitude.value(state, self.ages[voice])?;
                        let level = if amplitude.complete(state, &observed) {
                            0.0
                        } else {
                            observed.value
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
        if (kind <= 3 || kind == 5 || (9..=13).contains(&kind)) && voice >= self.frequencies.len() {
            return Err(PyValueError::new_err("Invalid synth runtime voice"));
        }
        match kind {
            13 => {
                let source = action[3] as usize;
                let command = action[4] as usize;
                if action[3] != source as f64
                    || source >= self.amplitude_definitions.len()
                    || action[4] != command as f64
                    || command > 5
                    || !action[5].is_finite()
                    || (command == 3 && !(0.0..=1.0).contains(&action[5]))
                    || (command != 3 && command != 4 && action[5] != 0.0)
                {
                    return Err(PyValueError::new_err("Invalid amplitude command"));
                }
                if !self.active[voice] {
                    return Ok(());
                }
                let definition = &self.amplitude_definitions[source];
                let state = &mut self.amplitude_states[voice][source];
                definition.settle(state, self.release_frames[voice], self.ages[voice])?;
                definition.command(state, self.ages[voice], command, action[5])?;
            }
            12 => {
                if action[3] != action[3] as u32 as f64
                    || action[4] != action[4] as u32 as f64
                    || action[5] != 0.0
                {
                    return Err(PyValueError::new_err("Invalid Envelfo stream key"));
                }
                self.envelfo_keys[voice] = action[3] as u64 | ((action[4] as u64) << 32);
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
                self.start_patch(voice)?;
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
                    }
                    self.graph_history[voice * edges..(voice + 1) * edges].fill(0.0);
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
                for (definition, state) in self
                    .amplitude_definitions
                    .iter()
                    .zip(&mut self.amplitude_states[voice])
                {
                    *state = definition.start()?;
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
                    for source in 0..self.staged_envelfos.len() {
                        if self.staged_owners[source] == source {
                            self.patch_command(
                                voice,
                                2,
                                source,
                                frame_position(self.frame)?,
                                ChildCommand::Playback(Command::NoteOff),
                            )?;
                        }
                    }
                    let age = frame_position(self.ages[voice])?;
                    let frame = frame_position(self.frame)?;
                    let release_frame = age.max(self.minimum_hold_frames);
                    for (index, definition) in self.named_envelopes.iter().enumerate() {
                        if definition.body.release.is_empty() {
                            continue;
                        }
                        let named_release_at = if definition.release_with_voice {
                            frame
                                .checked_add(release_frame.checked_sub(age).map_err(core_error)?)
                                .map_err(core_error)?
                        } else {
                            frame
                        };
                        if self.named_owners[index] == index {
                            self.pending_releases[voice][index] = Some(named_release_at);
                        }
                    }
                    self.release_frames[voice] = Some(release_frame);
                }
            }
            2 => {
                self.active[voice] = false;
                self.patch_states[voice] = None;
                self.pending_releases[voice].fill(None);
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
                if definition.scope != 0 || definition.body.driver == lfo::Driver::Transport {
                    return Err(PyValueError::new_err("Invalid synth LFO scope"));
                }
                let at = self.lfo_time(definition, frame_position(self.frame)?)?;
                let command = match action_kind {
                    0 => Command::Reset,
                    1 => Command::Rate(if definition.body.clock == lfo::Clock::Beats {
                        control_time(action[4])?
                    } else {
                        control_time(action[4])?
                            .checked_div(frame_time(self.rate)?)
                            .map_err(core_error)?
                    }),
                    2 => Command::Pause,
                    3 => Command::Resume,
                    4 => Command::Reverse,
                    5 => Command::Seek(control_time(action[4])?),
                    6 => Command::Shift(control_time(action[4])?),
                    _ => unreachable!(),
                };
                lfo_command(definition, &mut self.lfo_event_states[lfo], at, command)?;
            }
            9 => {
                let source = action[3] as usize;
                let command = action[4] as usize;
                if !self.active[voice]
                    || action[3] != source as f64
                    || source >= self.named_envelopes.len()
                    || action[4] != command as f64
                    || command > 5
                    || (command == 3 && !(0.0..=1.0).contains(&action[5]))
                    || (command == 4 && !action[5].is_finite())
                    || (command != 3 && command != 4 && action[5] != 0.0)
                {
                    return Err(PyValueError::new_err("Invalid named Envelfo action"));
                }
                if self.named_envelopes[source].sample_hold.is_some() {
                    return Ok(());
                }
                let command = match command {
                    0 => ChildCommand::Playback(Command::Pause),
                    1 => ChildCommand::Playback(Command::Resume),
                    2 => ChildCommand::Playback(Command::Reverse),
                    3 => ChildCommand::Playback(Command::Seek(frame_time(action[5])?)),
                    4 => ChildCommand::Playback(Command::Shift(frame_time(action[5])?)),
                    5 => ChildCommand::Advance,
                    _ => unreachable!(),
                };
                self.patch_command(voice, 1, source, frame_position(self.frame)?, command)?;
            }

            10 => {
                let source = action[3] as usize;
                let command = action[4] as usize;
                if !self.active[voice]
                    || action[3] != source as f64
                    || source >= self.staged_envelfos.len()
                    || action[4] != command as f64
                    || command > 5
                    || (command == 3 && !(0.0..=1.0).contains(&action[5]))
                    || (command == 4 && !action[5].is_finite())
                    || (command != 3 && command != 4 && action[5] != 0.0)
                {
                    return Err(PyValueError::new_err("Invalid staged Envelfo action"));
                }
                let command = match command {
                    0 => ChildCommand::Playback(Command::Pause),
                    1 => ChildCommand::Playback(Command::Resume),
                    2 => ChildCommand::Playback(Command::Reverse),
                    3 => ChildCommand::Playback(Command::Seek(frame_time(action[5])?)),
                    4 => ChildCommand::Playback(Command::Shift(frame_time(action[5])?)),
                    5 => ChildCommand::Advance,
                    _ => unreachable!(),
                };
                self.patch_command(voice, 2, source, frame_position(self.frame)?, command)?;
            }
            11 => {
                let source = action[3] as usize;
                let command = action[4] as usize;
                if !self.active[voice]
                    || action[3] != source as f64
                    || source >= self.lfo_definitions.len()
                    || self.lfo_definitions[source].scope != 3
                    || self.lfo_definitions[source].body.driver == lfo::Driver::Transport
                    || action[4] != command as f64
                    || command > 4
                    || (command == 3 && !(0.0..=1.0).contains(&action[5]))
                    || (command == 4 && !action[5].is_finite())
                    || (command != 3 && command != 4 && action[5] != 0.0)
                {
                    return Err(PyValueError::new_err("Invalid voice LFO action"));
                }
                let command = match command {
                    0 => Command::Pause,
                    1 => Command::Resume,
                    2 => Command::Reverse,
                    3 => Command::Seek(control_time(action[5])?),
                    4 => Command::Shift(control_time(action[5])?),
                    _ => unreachable!(),
                };
                self.patch_command(
                    voice,
                    0,
                    source,
                    frame_position(self.frame)?,
                    ChildCommand::Playback(command),
                )?;
            }
            _ => return Err(PyValueError::new_err("Unknown synth runtime action")),
        }
        Ok(())
    }

    fn lfo_signal(&self, voice: usize, source: usize) -> PyResult<(f64, f64)> {
        let definition = &self.lfo_definitions[source];
        if definition.scope == 1 {
            self.context_kind(self.voice_part_contexts[voice], 1)?;
        }
        let value = if definition.scope == 3 {
            self.patch_value(voice, 0, source)?
        } else {
            let state = &self.lfo_event_states[self.lfo_owners[source]];
            definition
                .body
                .value(
                    &state.core,
                    self.lfo_time(definition, frame_position(self.frame)?)?,
                )
                .map_err(core_error)?
        };
        Ok((value.value, value.weight))
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
        for (source, definition) in self.named_envelopes.iter().enumerate() {
            if definition.parameter != parameter {
                continue;
            }
            let value = self.patch_value(voice, 1, source)?.value;
            let amount = definition.intercept + definition.slope * value;
            if definition.operation == 0 {
                addition += amount;
            } else {
                product *= amount;
            }
        }
        for (source, definition) in self.staged_envelfos.iter().enumerate() {
            if definition.parameter != Some(parameter) {
                continue;
            }
            let value = self.patch_value(voice, 2, source)?.value;
            let amount = definition.intercept + definition.slope * value;
            if definition.operation == 0 {
                addition += amount;
            } else {
                product *= amount;
            }
        }
        for (source, target, operation, intercept, slope) in &self.envelfo_transform_routes {
            if *target != parameter {
                continue;
            }
            let amount = intercept
                + slope
                    * self.envelfo_transform_values[voice * self.envelfo_transforms.len() + source];
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
    sample_rate: f64,
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
        let exact = exact
            .iter()
            .map(|(n, d)| {
                Time::new(
                    i64::try_from(*n).map_err(|_| core_error(envelfo_core::Error::Overflow))?,
                    i64::try_from(*d).map_err(|_| core_error(envelfo_core::Error::Overflow))?,
                )
                .map_err(core_error)
            })
            .collect::<PyResult<Vec<_>>>()?;
        let beat_clock = row[6] != 0.0;
        let mut body = lfo::LFO::new(if beat_clock {
            exact[1]
        } else {
            exact[1]
                .checked_div(frame_time(sample_rate)?)
                .map_err(core_error)?
        })
        .map_err(core_error)?;
        body.shape = match waveform {
            0 => lfo::Shape::Sine,
            1 => lfo::Shape::Square,
            _ => lfo::Shape::Triangle,
        };
        body.duty = exact[0];
        body.phase = exact[2];
        body.delay = exact[3];
        body.fade = exact[4];
        body.clock = if beat_clock {
            lfo::Clock::Beats
        } else {
            lfo::Clock::Seconds
        };
        body.reset = if scope == 3 {
            lfo::Reset::Trigger
        } else {
            lfo::Reset::Free
        };
        if row[6] == 2.0 {
            body.driver = lfo::Driver::Transport;
            body.reset = lfo::Reset::Transport;
        }
        body.validate().map_err(core_error)?;
        definitions.push(LfoDefinition {
            scope,
            body,
            parameter,
            operation,
            intercept: row[4],
            slope: row[5],
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

fn initial_lfo(definition: &LfoDefinition, at: Time) -> PyResult<LfoPlayback> {
    Ok(LfoPlayback {
        core: definition.body.initial(at).map_err(core_error)?,
        ordinal: 0,
    })
}

fn lfo_command(
    definition: &LfoDefinition,
    state: &mut LfoPlayback,
    at: Time,
    command: Command,
) -> PyResult<()> {
    let ordinal = state
        .ordinal
        .checked_add(1)
        .ok_or_else(|| core_error(envelfo_core::Error::Overflow))?;
    definition
        .body
        .command(&mut state.core, at, ordinal, command)
        .map_err(core_error)?;
    state.ordinal = ordinal;
    Ok(())
}

fn control_time(value: f64) -> PyResult<Time> {
    // Public numeric commands use the same decimal value as Python Fraction(str(value)).
    let text = value.to_string();
    let (whole, fraction) = text.split_once('.').unwrap_or((&text, ""));
    let denominator = 10_i64
        .checked_pow(fraction.len() as u32)
        .ok_or_else(|| core_error(envelfo_core::Error::Overflow))?;
    let digits = format!("{whole}{fraction}");
    let numerator = digits
        .parse::<i64>()
        .map_err(|_| core_error(envelfo_core::Error::Overflow))?;
    Time::new(numerator, denominator).map_err(core_error)
}

fn core_error(error: envelfo_core::Error) -> PyErr {
    PyValueError::new_err(format!("Native EnveLFO: {error}"))
}

fn ratio_time(value: (i64, i64)) -> PyResult<Time> {
    Time::new(value.0, value.1).map_err(core_error)
}

fn advance_ramp(value: &mut f64, step: f64, remaining: &mut usize) {
    if *remaining > 0 {
        *value += step;
        *remaining -= 1;
    }
}

fn valid_envelfo_owners(owners: &[usize]) -> bool {
    owners
        .iter()
        .enumerate()
        .all(|(index, owner)| *owner <= index && owners[*owner] == *owner)
}
