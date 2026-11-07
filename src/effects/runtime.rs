//! Persistent DSP state and processing shared by effect graph owners.

mod highpass;
mod matrix;
mod phaser;
mod reverb;
mod state;
mod waveshaper;

use crate::amplitude::frame_time;
use crate::compressor;
use crate::error::{NativeError, NativeResult};
use crate::queue::{ActionBatch, BATCH_ACTIONS};
use crate::runtime::controls::ControlBank;
use crate::runtime::generators::GeneratorBank;
use crate::runtime::{RouteMap, sum_partials};
use highpass::Highpass;
use matrix::RoutingMatrix;
use numpy::ndarray::{Array2, Array3, Axis};
use numpy::{PyReadonlyArray1, PyReadonlyArray2, PyUntypedArrayMethods};
use phaser::Phaser;
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::PyDict;
use reverb::Reverb;
use rtrb::{Consumer, Producer, RingBuffer};
use std::f64::consts::PI;
use waveshaper::Waveshaper;

#[pyclass(unsendable)]
pub(crate) struct EffectRuntime {
    kinds: Vec<u8>,
    rate: f64,
    main_input: usize,
    tail_thresholds: Vec<f64>,
    ended: Vec<bool>,
    pub(crate) stopped: bool,
    pub(crate) widths: Vec<usize>,
    pub(crate) input_routes: Vec<Vec<i64>>,
    pub(crate) inputs: Array3<f64>,
    sources: Array2<i64>,
    output_source: usize,
    pub(crate) parameters: Array2<f64>,
    resolved: Array2<f64>,
    pub(crate) generators: Option<EffectGenerators>,
    targets: Array2<f64>,
    steps: Array2<f64>,
    remaining: Array2<usize>,
    filter_rows: Vec<EffectFilter>,
    filter_states: Array3<f64>,
    granulators: Vec<Option<LiveGranulator>>,
    tap_delays: Vec<Option<LiveTapDelay>>,
    modulated_delays: Vec<Option<LiveModulatedDelay>>,
    reductions: Vec<f64>,
    reverbs: Vec<Option<Reverb>>,
    waveshapers: Vec<Option<Waveshaper>>,
    phasers: Vec<Option<Phaser>>,
    highpasses: Vec<Option<Highpass>>,
    matrices: Vec<Option<RoutingMatrix>>,
    alignment: Vec<[Option<Alignment>; 2]>,
    port_values: Array3<f64>,
    port_ended: Array2<bool>,
    dry: Vec<f64>,
    control_delays: Vec<Vec<Option<ControlDelay>>>,
    values: Array2<f64>,
    wet: Vec<f64>,
    pub(crate) producer: Producer<ActionBatch>,
    pub(crate) consumer: Consumer<ActionBatch>,
    pub(crate) action_scratch: Vec<f64>,
    pub(crate) action_count: usize,
}

#[derive(Clone, PartialEq)]
struct EffectFilter {
    node: usize,
    response: u8,
    parameter: usize,
    state_floor: f64,
    minimum: f64,
    maximum: f64,
    clamp: bool,
}

#[derive(Clone)]
pub(crate) struct EffectSnapshot {
    kinds: Vec<u8>,
    rate: f64,
    main_input: usize,
    tail_thresholds: Vec<f64>,
    ended: Vec<bool>,
    stopped: bool,
    widths: Vec<usize>,
    input_routes: Vec<Vec<i64>>,
    sources: Array2<i64>,
    output_source: usize,
    parameters: Array2<f64>,
    generators: Option<EffectGenerators>,
    targets: Array2<f64>,
    steps: Array2<f64>,
    remaining: Array2<usize>,
    filter_rows: Vec<EffectFilter>,
    filter_states: Array3<f64>,
    granulators: Vec<Option<LiveGranulator>>,
    tap_delays: Vec<Option<LiveTapDelay>>,
    modulated_delays: Vec<Option<LiveModulatedDelay>>,
    reductions: Vec<f64>,
    reverbs: Vec<Option<Reverb>>,
    waveshapers: Vec<Option<Waveshaper>>,
    phasers: Vec<Option<Phaser>>,
    highpasses: Vec<Option<Highpass>>,
    alignment: Vec<[Option<Alignment>; 2]>,
    control_delays: Vec<Vec<Option<ControlDelay>>>,
}

#[derive(Clone)]
struct ControlDelay {
    history: Vec<f64>,
    write: usize,
}

#[derive(Clone)]
struct Alignment {
    history: Array2<f64>,
    write: usize,
    remaining: usize,
}

#[derive(Clone)]
pub(crate) struct EffectGenerators {
    pub(crate) bank: GeneratorBank,
    pub(crate) controls: ControlBank,
    pub(crate) maps: Vec<Vec<RouteMap>>,
    pub(crate) targets: Vec<(usize, usize, f64, f64)>,
    pub(crate) partials: Vec<f64>,
}

#[derive(Clone)]
struct LiveGranulator {
    history: Array2<f64>,
    history_start: i64,
    history_len: usize,
    history_write: usize,
    grains: Vec<LiveGrain>,
    phase: f64,
    counter: u64,
    frame: usize,
    frozen: bool,
    frozen_start: i64,
    order: Vec<usize>,
    active_count: usize,
    freeze_crossfade_frames: usize,
    maximum_duration_seconds: f64,
}

#[derive(Clone)]
struct LiveTapDelay {
    history: Array2<f64>,
    write: usize,
    heads: Vec<DelayHead>,
    current: usize,
    fade_frames: usize,
    state_floor: f64,
    non_silent: usize,
}

#[derive(Clone)]
struct DelayHead {
    delay: f64,
    gain: f64,
    step: f64,
    remaining: usize,
    active: bool,
}

#[derive(Clone)]
struct LiveModulatedDelay {
    history: Array2<f64>,
    write: usize,
    phase: f64,
    waveform: u8,
    phase_offsets: Vec<f64>,
    state_floor: f64,
    non_silent: usize,
}

#[derive(Clone)]
struct LiveGrain {
    samples: Array2<f64>,
    len: usize,
    index: usize,
    gain: f64,
    active: bool,
}

#[pymethods]
impl EffectRuntime {
    #[new]
    #[allow(clippy::too_many_arguments)]
    fn py_new(
        kinds: Vec<u8>,
        sources: PyReadonlyArray2<'_, i64>,
        parameters: PyReadonlyArray2<'_, f64>,
        output_source: usize,
        filters: PyReadonlyArray2<'_, f64>,
        granulators: PyReadonlyArray2<'_, i64>,
        tap_delays: PyReadonlyArray2<'_, i64>,
        modulated_delays: PyReadonlyArray2<'_, f64>,
        widths: Vec<usize>,
        main_input: usize,
        tail_thresholds: Vec<f64>,
        rate: f64,
    ) -> PyResult<Self> {
        let inputs = widths
            .len()
            .checked_sub(kinds.len())
            .ok_or_else(|| PyValueError::new_err("Invalid native effect port widths"))?;
        Self::new(
            kinds,
            sources,
            parameters,
            output_source,
            filters,
            granulators,
            tap_delays,
            modulated_delays,
            rate,
            widths,
            main_input,
            tail_thresholds,
            vec![vec![-2]; inputs],
            1,
            BATCH_ACTIONS,
            1,
        )
    }

    fn render_frame(
        &mut self,
        inputs: Vec<PyReadonlyArray1<'_, f64>>,
        parameters: PyReadonlyArray2<'_, f64>,
        ended: Vec<bool>,
        frozen: Vec<bool>,
    ) -> PyResult<(Vec<f64>, bool)> {
        if inputs.len() != self.input_routes.len()
            || ended.len() != inputs.len()
            || frozen.len() != self.kinds.len()
            || parameters.shape() != self.parameters.shape()
            || parameters.as_array().iter().any(|value| !value.is_finite())
            || inputs.iter().enumerate().any(|(input, values)| {
                values.len() != self.widths[input]
                    || values.as_array().iter().any(|value| !value.is_finite())
            })
            || frozen
                .iter()
                .enumerate()
                .any(|(node, frozen)| *frozen && self.kinds[node] != 3)
        {
            return Err(PyValueError::new_err("Invalid native effect frame"));
        }
        self.resolved.assign(&parameters.as_array());
        for (input, values) in inputs.iter().enumerate() {
            self.ended[input] = ended[input];
            for (channel, value) in values.as_array().iter().enumerate() {
                self.values[[input, channel]] = if ended[input] { 0.0 } else { *value };
            }
        }
        for (node, frozen) in frozen.into_iter().enumerate() {
            if self.kinds[node] == 3 {
                self.resolved[[node, 9]] = f64::from(frozen);
            }
        }
        self.process_nodes(self.rate)?;
        let width = self.widths[self.output_source];
        let drained = self.ended[self.output_source];
        let output = (0..width)
            .map(|channel| {
                if drained {
                    0.0
                } else {
                    self.values[[self.output_source, channel]]
                }
            })
            .collect::<Vec<_>>();
        if output.iter().any(|value| !value.is_finite()) {
            return Err(PyValueError::new_err("Non-finite native effect output"));
        }
        Ok((output, drained))
    }

    fn export_state<'py>(&self, py: Python<'py>) -> PyResult<Vec<Bound<'py, PyDict>>> {
        self.export_processors(py)
    }

    fn import_state(&mut self, values: Vec<Bound<'_, PyDict>>, frame: usize) -> PyResult<()> {
        self.import_processors(values, frame)
    }
}

impl EffectRuntime {
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn new(
        kinds: Vec<u8>,
        sources: PyReadonlyArray2<'_, i64>,
        parameters: PyReadonlyArray2<'_, f64>,
        output_source: usize,
        filters: PyReadonlyArray2<'_, f64>,
        granulators: PyReadonlyArray2<'_, i64>,
        tap_delays: PyReadonlyArray2<'_, i64>,
        modulated_delays: PyReadonlyArray2<'_, f64>,
        rate: f64,
        widths: Vec<usize>,
        main_input: usize,
        tail_thresholds: Vec<f64>,
        input_routes: Vec<Vec<i64>>,
        maximum_block_frames: usize,
        action_capacity: usize,
        batch_capacity: usize,
    ) -> PyResult<Self> {
        let nodes = kinds.len();
        let inputs = input_routes.len();
        let channels = widths.iter().copied().max().unwrap_or(0);
        if kinds.iter().any(|v| *v > 11)
            || !rate.is_finite()
            || rate <= 0.0
            || maximum_block_frames == 0
            || action_capacity < BATCH_ACTIONS
            || batch_capacity == 0
            || inputs == 0
            || main_input >= inputs
            || tail_thresholds.len() != nodes
            || tail_thresholds.iter().enumerate().any(|(node, value)| {
                !value.is_finite()
                    || if matches!(kinds[node], 2 | 5 | 6 | 8 | 9 | 10) {
                        *value <= 0.0
                    } else {
                        *value != 0.0
                    }
            })
            || widths.len() != inputs + nodes
            || widths.contains(&0)
            || sources.shape() != [nodes, 2]
            || parameters.shape()[0] != nodes
            || parameters.shape()[1] < 3
            || parameters.as_array().iter().any(|v| !v.is_finite())
            || parameters
                .as_array()
                .column(1)
                .iter()
                .any(|v| !(0.0..=1.0).contains(v))
            || parameters
                .as_array()
                .column(2)
                .iter()
                .any(|v| !(0.0..=1.0).contains(v))
            || output_source >= inputs + nodes
            || filters.shape()[1] != 7
            || filters.as_array().iter().any(|v| !v.is_finite())
            || granulators.shape()[1] != 4
            || tap_delays.shape()[1] != 3
            || modulated_delays.shape()[1] != channels + 3
            || modulated_delays.as_array().iter().any(|v| !v.is_finite())
        {
            return Err(PyValueError::new_err("Invalid live effect graph"));
        }
        let sources = sources.as_array().to_owned();
        for node in 0..nodes {
            let ports = if matches!(kinds[node], 1 | 7) { 2 } else { 1 };
            for port in 0..ports {
                let source = sources[[node, port]];
                if source < 0 || source as usize >= node + inputs {
                    return Err(PyValueError::new_err(
                        "Effect source must precede its consumer",
                    ));
                }
                if kinds[node] == 8 {
                    if !matches!(widths[source as usize], 1 | 2) || widths[node + inputs] != 2 {
                        return Err(PyValueError::new_err("Invalid reverb channel widths"));
                    }
                } else if kinds[node] != 11
                    && !(kinds[node] == 7 && port == 1)
                    && widths[source as usize] != widths[node + inputs]
                {
                    return Err(PyValueError::new_err(
                        "Effect input and output widths differ",
                    ));
                }
            }
            if kinds[node] == 7 {
                if parameters.shape()[1] < 8 {
                    return Err(PyValueError::new_err("Missing compressor parameters"));
                }
                let values = parameters.as_array();
                compressor::gain(
                    0.0,
                    [
                        values[[node, 0]],
                        values[[node, 3]],
                        values[[node, 4]],
                        values[[node, 5]],
                        values[[node, 6]],
                        values[[node, 7]],
                    ],
                    0.0,
                    rate,
                )?;
            }
        }
        let mut latencies = vec![0; nodes + inputs];
        let mut alignment = Vec::with_capacity(nodes);
        let mut waveshapers = (0..nodes).map(|_| None).collect::<Vec<_>>();
        for node in 0..nodes {
            let ports = if matches!(kinds[node], 1 | 7) { 2 } else { 1 };
            let arrival = (0..ports)
                .map(|port| latencies[sources[[node, port]] as usize])
                .max()
                .unwrap_or(0);
            let mut delays = std::array::from_fn(|_| None);
            for (port, delay) in delays.iter_mut().enumerate().take(ports) {
                let source = sources[[node, port]] as usize;
                let frames = arrival - latencies[source];
                if frames > 0 {
                    *delay = Some(Alignment {
                        history: Array2::zeros((frames, widths[source])),
                        write: 0,
                        remaining: frames,
                    });
                }
            }
            alignment.push(delays);
            latencies[node + inputs] =
                arrival
                    .checked_add(if kinds[node] == 4 { 48 } else { 0 })
                    .ok_or_else(|| PyValueError::new_err("Effect latency overflow"))?;
            if kinds[node] == 4 {
                let values = parameters.as_array();
                if parameters.shape()[1] < 71
                    || values[[node, 0]] <= 0.0
                    || !matches!(values[[node, 5]], 0.0 | 1.0)
                {
                    return Err(PyValueError::new_err("Invalid waveshaper profile"));
                }
                let coefficients = std::array::from_fn(|i| values[[node, 6 + i]]);
                waveshapers[node] = Some(Waveshaper::new(
                    widths[node + inputs],
                    values[[node, 5]] == 1.0,
                    coefficients,
                ));
            }
        }
        let mut filter_rows = Vec::with_capacity(filters.shape()[0]);
        for row in filters.as_array().rows() {
            let node = row[0] as usize;
            let response = row[1] as u8;
            let parameter = row[2] as usize;
            if row[0] != node as f64
                || node >= nodes
                || kinds[node] != 2
                || row[1] != response as f64
                || response > 3
                || row[2] != parameter as f64
                || parameter + 1 >= parameters.shape()[1]
                || row[3] < 0.0
                || row[3] >= tail_thresholds[node]
                || row[4] <= 0.0
                || row[5] < row[4]
                || row[5] >= rate / 2.0
                || !matches!(row[6], 0.0 | 1.0)
            {
                return Err(PyValueError::new_err("Invalid live effect filter"));
            }
            filter_rows.push(EffectFilter {
                node,
                response,
                parameter,
                state_floor: row[3],
                minimum: row[4],
                maximum: row[5],
                clamp: row[6] == 1.0,
            });
        }
        let parameters = parameters.as_array().to_owned();
        let mut matrices = (0..nodes).map(|_| None).collect::<Vec<_>>();
        for node in 0..nodes {
            if kinds[node] == 11 {
                let columns = widths[sources[[node, 0]] as usize];
                let rows = widths[node + inputs];
                let size = rows
                    .checked_mul(columns)
                    .and_then(|size| size.checked_add(3))
                    .ok_or_else(|| PyValueError::new_err("Invalid routing matrix size"))?;
                if parameters.ncols() < size
                    || parameters[[node, 1]] != 1.0
                    || parameters[[node, 2]] != 1.0
                {
                    return Err(PyValueError::new_err("Invalid routing matrix profile"));
                }
                matrices[node] = Some(RoutingMatrix::new(Array2::from_shape_fn(
                    (rows, columns),
                    |(row, column)| parameters[[node, 3 + row * columns + column]],
                )));
            }
        }
        let mut highpasses = (0..nodes).map(|_| None).collect::<Vec<_>>();
        for node in 0..nodes {
            if kinds[node] == 10 {
                if parameters.ncols() < 4
                    || parameters[[node, 3]] < 0.0
                    || parameters[[node, 3]] >= tail_thresholds[node]
                {
                    return Err(PyValueError::new_err("Invalid highpass profile"));
                }
                highpass::coefficient(parameters[[node, 0]], rate)?;
                highpasses[node] =
                    Some(Highpass::new(widths[node + inputs], parameters[[node, 3]]));
            }
        }
        let mut phasers = (0..nodes).map(|_| None).collect::<Vec<_>>();
        for node in 0..nodes {
            if kinds[node] == 9 {
                if parameters.ncols() < 7
                    || parameters[[node, 6]] < 0.0
                    || parameters[[node, 6]] >= tail_thresholds[node]
                    || !(-0.95..=0.95).contains(&parameters[[node, 5]])
                {
                    return Err(PyValueError::new_err("Invalid phaser profile"));
                }
                phaser::coefficient(
                    parameters[[node, 0]],
                    parameters[[node, 3]],
                    parameters[[node, 4]],
                    rate,
                )?;
                phasers[node] = Some(Phaser::new(widths[node + inputs], parameters[[node, 6]]));
            }
        }
        let mut prepared_granulators = (0..nodes).map(|_| None).collect::<Vec<_>>();
        for row in granulators.as_array().rows() {
            let node = usize::try_from(row[0])
                .map_err(|_| PyValueError::new_err("Invalid live granulator node"))?;
            let history_frames = usize::try_from(row[1])
                .map_err(|_| PyValueError::new_err("Invalid live granulator history"))?;
            let maximum_grains = usize::try_from(row[2])
                .map_err(|_| PyValueError::new_err("Invalid live granulator capacity"))?;
            let freeze_crossfade_frames = usize::try_from(row[3])
                .map_err(|_| PyValueError::new_err("Invalid live freeze crossfade"))?;
            if node >= nodes
                || kinds[node] != 3
                || history_frames < 2
                || maximum_grains == 0
                || freeze_crossfade_frames == 0
                || prepared_granulators[node].is_some()
                || parameters.ncols() < 10
            {
                return Err(PyValueError::new_err("Invalid live granulator"));
            }
            let duration_frames = parameters[[node, 8]] * rate;
            if !duration_frames.is_finite()
                || duration_frames <= 0.0
                || duration_frames >= usize::MAX as f64
                || parameters[[node, 8]] <= 0.0
                || parameters[[node, 3]] > parameters[[node, 8]]
            {
                return Err(PyValueError::new_err(
                    "Invalid native granulator duration capacity",
                ));
            }
            let maximum_grain_frames = duration_frames.round_ties_even().max(2.0) as usize;
            prepared_granulators[node] = Some(LiveGranulator::new(
                history_frames,
                maximum_grain_frames,
                maximum_grains,
                widths[node + inputs],
                freeze_crossfade_frames,
                parameters[[node, 8]],
            ));
        }
        if kinds
            .iter()
            .enumerate()
            .any(|(i, kind)| *kind == 3 && prepared_granulators[i].is_none())
        {
            return Err(PyValueError::new_err("Missing live granulator setup"));
        }
        let mut reverbs = (0..nodes).map(|_| None).collect::<Vec<_>>();
        for node in 0..nodes {
            if kinds[node] != 8 {
                continue;
            }
            if parameters.ncols() < 16
                || parameters[[node, 0]] < 0.0
                || parameters[[node, 0]] > parameters[[node, 6]]
                || parameters[[node, 6]] < 0.0
            {
                return Err(PyValueError::new_err("Invalid reverb predelay bounds"));
            }
            let mut lengths = [0; 8];
            for (i, length) in lengths.iter_mut().enumerate() {
                let value = parameters[[node, i + 8]];
                if value < 1.0 || value >= usize::MAX as f64 || value.fract() != 0.0 {
                    return Err(PyValueError::new_err("Invalid reverb delay line"));
                }
                *length = value as usize;
            }
            let reverb = Reverb::new(lengths);
            reverb.coefficients(parameters[[node, 3]], parameters[[node, 5]], rate)?;
            reverbs[node] = Some(reverb);
        }
        let mut prepared_tap_delays = (0..nodes).map(|_| None).collect::<Vec<_>>();
        for row in tap_delays.as_array().rows() {
            let node = usize::try_from(row[0])
                .map_err(|_| PyValueError::new_err("Invalid live tap delay node"))?;
            let history_frames = usize::try_from(row[1])
                .map_err(|_| PyValueError::new_err("Invalid live tap delay history"))?;
            let fade_frames = usize::try_from(row[2])
                .map_err(|_| PyValueError::new_err("Invalid live tap retime duration"))?;
            if node >= nodes
                || !matches!(kinds[node], 5 | 8)
                || history_frames == 0
                || prepared_tap_delays[node].is_some()
                || parameters.ncols() < 5
                || fade_frames == 0
                || fade_frames.checked_add(1).is_none()
                || parameters[[node, 4]] < 0.0
                || parameters[[node, 4]] >= tail_thresholds[node]
                || (kinds[node] == 8 && parameters[[node, 6]] * rate > (history_frames - 1) as f64)
            {
                return Err(PyValueError::new_err("Invalid live tap delay"));
            }
            prepared_tap_delays[node] = Some(LiveTapDelay {
                history: Array2::zeros((history_frames, widths[node + inputs])),
                write: 0,
                heads: (0..fade_frames + 1)
                    .map(|i| DelayHead {
                        delay: (parameters[[node, 0]] * rate).max(if kinds[node] == 8 {
                            0.0
                        } else {
                            1.0
                        }),
                        gain: if i == 0 { 1.0 } else { 0.0 },
                        step: 0.0,
                        remaining: 0,
                        active: i == 0,
                    })
                    .collect(),
                current: 0,
                fade_frames,
                state_floor: parameters[[node, 4]],
                non_silent: 0,
            });
        }
        if kinds
            .iter()
            .enumerate()
            .any(|(i, kind)| matches!(*kind, 5 | 8) && prepared_tap_delays[i].is_none())
        {
            return Err(PyValueError::new_err("Missing live tap delay setup"));
        }
        let mut prepared_modulated_delays = (0..nodes).map(|_| None).collect::<Vec<_>>();
        for row in modulated_delays.as_array().rows() {
            let node = row[0] as usize;
            let history_frames = row[1] as usize;
            let waveform = row[2] as u8;
            if row[0] != node as f64
                || node >= nodes
                || kinds[node] != 6
                || row[1] != history_frames as f64
                || history_frames == 0
                || row[2] != waveform as f64
                || waveform > 2
                || row.iter().skip(3).any(|v| !(0.0..1.0).contains(v))
                || prepared_modulated_delays[node].is_some()
                || parameters.ncols() < 7
                || parameters[[node, 6]] < 0.0
                || parameters[[node, 6]] >= tail_thresholds[node]
            {
                return Err(PyValueError::new_err("Invalid live modulated delay"));
            }
            prepared_modulated_delays[node] = Some(LiveModulatedDelay {
                history: Array2::zeros((history_frames, widths[node + inputs])),
                write: 0,
                phase: 0.0,
                waveform,
                phase_offsets: row.iter().skip(3).copied().collect(),
                state_floor: parameters[[node, 6]],
                non_silent: 0,
            });
        }
        if kinds
            .iter()
            .enumerate()
            .any(|(i, kind)| *kind == 6 && prepared_modulated_delays[i].is_none())
        {
            return Err(PyValueError::new_err("Missing live modulated delay setup"));
        }
        let control_delays = (0..nodes)
            .map(|node| {
                let arrival = latencies[node + inputs] - if kinds[node] == 4 { 48 } else { 0 };
                (0..parameters.ncols())
                    .map(|parameter| {
                        let dynamic = kinds[node] != 11
                            && (matches!(parameter, 1 | 2)
                                || match kinds[node] {
                                    0 => parameter == 0,
                                    2 => filter_rows.iter().any(|filter| {
                                        filter.node == node
                                            && (parameter == filter.parameter
                                                || parameter == filter.parameter + 1)
                                    }),
                                    3 => (3..=7).contains(&parameter) || parameter == 9,
                                    4 => matches!(parameter, 0 | 3 | 4),
                                    5 => matches!(parameter, 0 | 3),
                                    6 => matches!(parameter, 0 | 3..=5),
                                    7 => matches!(parameter, 0 | 3..=7),
                                    8 => matches!(parameter, 0 | 3 | 5),
                                    9 => matches!(parameter, 0 | 3..=5),
                                    10 => parameter == 0,
                                    _ => false,
                                });
                        let frames = arrival
                            + if kinds[node] == 4 {
                                if matches!(parameter, 0 | 3) { 24 } else { 48 }
                            } else {
                                0
                            };
                        if dynamic && frames > 0 {
                            Some(ControlDelay {
                                history: vec![parameters[[node, parameter]]; frames],
                                write: 0,
                            })
                        } else {
                            None
                        }
                    })
                    .collect()
            })
            .collect();
        let (producer, consumer) = RingBuffer::new(batch_capacity);
        Ok(Self {
            rate,
            main_input,
            tail_thresholds,
            ended: vec![false; nodes + inputs],
            stopped: false,
            reductions: vec![0.0; nodes],
            reverbs,
            waveshapers,
            phasers,
            highpasses,
            matrices,
            alignment,
            port_values: Array3::zeros((nodes, 2, channels)),
            port_ended: Array2::from_elem((nodes, 2), false),
            dry: vec![0.0; channels],
            control_delays,
            kinds,
            inputs: Array3::zeros((inputs, maximum_block_frames, channels)),
            input_routes,
            widths,
            sources,
            output_source,
            targets: parameters.clone(),
            resolved: parameters.clone(),
            generators: None,
            steps: Array2::zeros(parameters.raw_dim()),
            remaining: Array2::zeros(parameters.raw_dim()),
            parameters,
            filter_states: Array3::zeros((filter_rows.len(), channels, 2)),
            filter_rows,
            granulators: prepared_granulators,
            tap_delays: prepared_tap_delays,
            modulated_delays: prepared_modulated_delays,
            values: Array2::zeros((nodes + inputs, channels)),
            wet: vec![0.0; channels],
            producer,
            consumer,
            action_scratch: vec![0.0; action_capacity * 6],
            action_count: 0,
        })
    }

    pub(crate) fn process(
        &mut self,
        output: &mut Array2<f64>,
        frames: usize,
        start: usize,
    ) -> NativeResult<()> {
        let count = self.action_count;
        {
            let actions = &self.action_scratch[..count * 6];
            if actions.chunks_exact(6).any(|a| {
                let offset = a[0] as usize;
                a[0] != offset as f64 || offset >= frames
            }) || actions
                .chunks_exact(6)
                .zip(actions.chunks_exact(6).skip(1))
                .any(|(a, b)| b[0] < a[0])
            {
                return Err(NativeError::Message("Invalid live effect block"));
            }
        }
        let mut action = 0;
        for frame in 0..frames {
            let at = start
                .checked_add(frame)
                .ok_or(NativeError::Message("Effect frame overflow"))?;
            while action < count * 6 && self.action_scratch[action] as usize == frame {
                let values: [f64; 6] = self.action_scratch[action..action + 6]
                    .try_into()
                    .expect("validated action width");
                self.apply_action(&values, at)?;
                action += 6;
            }
            if self.stopped {
                self.advance_ramps();
                continue;
            }
            self.resolved.assign(&self.parameters);
            if let Some(generators) = &mut self.generators {
                generators.bank.settle_patches(&[true], at, true)?;
                generators.bank.observe_patch(0, at, &generators.controls)?;
                for (index, (node, parameter, minimum, maximum)) in
                    generators.targets.iter().enumerate()
                {
                    let mut count = 0;
                    let product = generators.bank.contributions(
                        0,
                        index,
                        1.0,
                        at,
                        &generators.controls,
                        &generators.maps,
                        &mut generators.partials,
                        &mut count,
                    )?;
                    let value = (self.parameters[[*node, *parameter]]
                        + sum_partials(&generators.partials[..count]))
                        * product;
                    if !value.is_finite() || value < *minimum || value > *maximum {
                        return Err(NativeError::Message(
                            "Modulated effect parameter is outside its domain",
                        ));
                    }
                    self.resolved[[*node, *parameter]] = value;
                }
            }
            let inputs = self.input_routes.len();
            for input in 0..inputs {
                for channel in 0..self.widths[input] {
                    self.values[[input, channel]] = if self.ended[input] {
                        0.0
                    } else {
                        self.inputs[[input, frame, channel]]
                    };
                }
            }
            self.process_nodes(self.rate)?;
            self.stopped = self.ended[self.output_source];
            for channel in 0..self.widths[self.output_source] {
                output[[frame, channel]] = if self.stopped {
                    0.0
                } else {
                    self.values[[self.output_source, channel]]
                };
            }
            self.advance_ramps();
        }
        self.action_count = 0;
        Ok(())
    }

    fn process_nodes(&mut self, rate: f64) -> NativeResult<()> {
        let inputs = self.input_routes.len();
        for node in 0..self.kinds.len() {
            for (parameter, delay) in self.control_delays[node].iter_mut().enumerate() {
                if let Some(delay) = delay {
                    std::mem::swap(
                        &mut self.resolved[[node, parameter]],
                        &mut delay.history[delay.write],
                    );
                    delay.write = (delay.write + 1) % delay.history.len();
                }
            }
            let channels = self.widths[node + inputs];
            let first = self.sources[[node, 0]] as usize;
            let ports = if matches!(self.kinds[node], 1 | 7) {
                2
            } else {
                1
            };
            for port in 0..ports {
                let source = self.sources[[node, port]] as usize;
                self.port_ended[[node, port]] = self.ended[source];
                if let Some(delay) = &mut self.alignment[node][port] {
                    self.port_ended[[node, port]] &= delay.remaining == 0;
                    for channel in 0..self.widths[source] {
                        self.port_values[[node, port, channel]] =
                            delay.history[[delay.write, channel]];
                        delay.history[[delay.write, channel]] = self.values[[source, channel]];
                    }
                    delay.write = (delay.write + 1) % delay.history.nrows();
                    delay.remaining = if self.ended[source] {
                        delay.remaining.saturating_sub(1)
                    } else {
                        delay.history.nrows()
                    };
                } else {
                    for channel in 0..self.widths[source] {
                        self.port_values[[node, port, channel]] = self.values[[source, channel]];
                    }
                }
            }
            let authored_mix = self.resolved[[node, 1]];
            let bypass = self.resolved[[node, 2]];
            if !(0.0..=1.0).contains(&authored_mix) || !(0.0..=1.0).contains(&bypass) {
                return Err(NativeError::Message("Invalid live effect mix"));
            }
            let mix = authored_mix * bypass;
            if self.kinds[node] == 11 && (authored_mix != 1.0 || bypass != 1.0) {
                return Err(NativeError::Message("Routing matrix has no mix or bypass"));
            }
            let mut waveshaper_drained = false;
            for channel in 0..channels {
                let value = if self.kinds[node] == 8 && self.widths[first] == 1 {
                    self.port_values[[node, 0, 0]] / std::f64::consts::SQRT_2
                } else {
                    self.port_values[[node, 0, channel]]
                };
                self.dry[channel] = value;
                self.wet[channel] = value;
            }
            match self.kinds[node] {
                0 => {
                    let gain = 10_f64.powf(self.resolved[[node, 0]] / 20.0);
                    for value in &mut self.wet[..channels] {
                        *value *= gain;
                    }
                }
                1 => {
                    for (channel, value) in self.wet[..channels].iter_mut().enumerate() {
                        *value *= self.port_values[[node, 1, channel]];
                    }
                }
                2 => self.process_filters(node, channels, rate)?,
                3 => {
                    let frozen = self.resolved[[node, 9]];
                    if !matches!(frozen, 0.0 | 1.0) {
                        return Err(NativeError::Message("Invalid granular freeze value"));
                    }
                    self.set_frozen(node, frozen == 1.0);
                    let parameters = [
                        self.resolved[[node, 3]],
                        self.resolved[[node, 4]],
                        self.resolved[[node, 5]],
                        self.resolved[[node, 6]],
                        self.resolved[[node, 7]],
                    ];
                    self.granulators[node]
                        .as_mut()
                        .expect("validated live granulator")
                        .process(
                            &mut self.wet[..channels],
                            parameters,
                            rate,
                            self.port_ended[[node, 0]],
                        )?;
                }
                4 => {
                    waveshaper_drained = self.waveshapers[node]
                        .as_mut()
                        .expect("prepared waveshaper")
                        .process(
                            &mut self.dry[..channels],
                            &mut self.wet[..channels],
                            self.resolved[[node, 0]],
                            self.resolved[[node, 3]],
                            self.resolved[[node, 4]],
                            self.port_ended[[node, 0]],
                        )?;
                }
                5 => self.process_tap_delay(node, channels, rate)?,
                6 => self.process_modulated_delay(node, channels, rate)?,
                7 => {
                    let detector = self.sources[[node, 1]] as usize;
                    let peak = (0..self.widths[detector])
                        .map(|channel| self.port_values[[node, 1, channel]].abs())
                        .fold(0.0, f64::max);
                    let (gain, reduction) = compressor::gain(
                        peak,
                        [
                            self.resolved[[node, 0]],
                            self.resolved[[node, 3]],
                            self.resolved[[node, 4]],
                            self.resolved[[node, 5]],
                            self.resolved[[node, 6]],
                            self.resolved[[node, 7]],
                        ],
                        self.reductions[node],
                        rate,
                    )?;
                    self.reductions[node] = reduction;
                    for value in &mut self.wet[..channels] {
                        *value *= gain;
                    }
                }
                8 => self.process_reverb(node, first, rate)?,
                9 => self.phasers[node]
                    .as_mut()
                    .expect("prepared phaser")
                    .process(
                        &mut self.wet[..channels],
                        [
                            self.resolved[[node, 0]],
                            self.resolved[[node, 3]],
                            self.resolved[[node, 4]],
                            self.resolved[[node, 5]],
                        ],
                        rate,
                    )?,
                10 => self.highpasses[node]
                    .as_mut()
                    .expect("prepared highpass")
                    .process(&mut self.wet[..channels], self.resolved[[node, 0]], rate)?,
                _ => self.matrices[node]
                    .as_mut()
                    .expect("prepared routing matrix")
                    .process(
                        self.port_values.index_axis(Axis(0), node).row(0),
                        &mut self.wet[..channels],
                    )?,
            }
            let drained = self.port_ended[[node, 0]]
                && match self.kinds[node] {
                    2 => self
                        .filter_rows
                        .iter()
                        .enumerate()
                        .filter(|(_, filter)| filter.node == node)
                        .all(|(stage, _)| {
                            (0..channels).all(|channel| {
                                (0..2).all(|state| {
                                    self.filter_states[[stage, channel, state]].abs()
                                        < self.tail_thresholds[node]
                                })
                            })
                        }),
                    3 => self.granulators[node]
                        .as_ref()
                        .is_some_and(|granular| !granular.frozen && granular.active_count == 0),
                    4 => waveshaper_drained,
                    5 => self.tap_delays[node]
                        .as_ref()
                        .is_some_and(|delay| delay.non_silent == 0),
                    6 => self.modulated_delays[node]
                        .as_ref()
                        .is_some_and(|delay| delay.non_silent == 0),
                    8 => {
                        let reverb = self.reverbs[node].as_ref().expect("prepared reverb");
                        self.tap_delays[node]
                            .as_ref()
                            .is_some_and(|delay| delay.non_silent == 0)
                            && reverb.non_silent == 0
                            && reverb
                                .damping
                                .iter()
                                .all(|value| value.abs() < self.tail_thresholds[node])
                    }
                    9 => {
                        let phaser = self.phasers[node].as_ref().expect("prepared phaser");
                        phaser
                            .stages
                            .iter()
                            .chain(phaser.feedback.iter())
                            .all(|value| value.abs() < self.tail_thresholds[node])
                    }
                    10 => self.highpasses[node]
                        .as_ref()
                        .expect("prepared highpass")
                        .memory
                        .iter()
                        .all(|value| value.abs() < self.tail_thresholds[node]),
                    _ => true,
                };
            self.ended[node + inputs] = drained;
            for channel in 0..channels {
                self.values[[node + inputs, channel]] = if drained {
                    0.0
                } else {
                    (1.0 - mix) * self.dry[channel] + mix * self.wet[channel]
                };
            }
        }
        Ok(())
    }

    fn apply_action(&mut self, action: &[f64], frame: usize) -> NativeResult<()> {
        let kind = action[1] as usize;
        let node = action[2] as usize;
        let parameter = action[3] as usize;
        let duration = action[5] as usize;
        if action[1] != kind as f64 || action[2] != node as f64 || action[5] != duration as f64 {
            return Err(NativeError::Message("Invalid live effect action"));
        }
        if kind == 2 {
            if node >= self.input_routes.len() || self.ended[node] || action[3..] != [0.0, 0.0, 0.0]
            {
                return Err(NativeError::Message("Invalid or repeated live input end"));
            }
            self.ended[node] = true;
            if node == self.main_input {
                if let Some(generators) = &mut self.generators {
                    generators
                        .bank
                        .release_voice(0, frame, frame, frame_time(0.0)?)?;
                }
            }
            return Ok(());
        }
        if kind == 3 {
            if action[2..] != [0.0, 0.0, 0.0, 0.0] {
                return Err(NativeError::Message("Invalid live effect stop"));
            }
            self.stopped = true;
            return Ok(());
        }
        if node >= self.parameters.nrows() {
            return Err(NativeError::Message("Invalid live effect node"));
        }
        if self.kinds[node] == 11 {
            return Err(NativeError::Message("Routing matrices have no actions"));
        }
        if kind == 1 {
            if self.kinds[node] != 3
                || action[3] != 0.0
                || !matches!(action[4], 0.0 | 1.0)
                || duration != 0
            {
                return Err(NativeError::Message("Invalid live granulator freeze"));
            }
            self.parameters[[node, 9]] = action[4];
            return Ok(());
        }
        if kind != 0 || action[3] != parameter as f64 || parameter >= self.parameters.ncols() {
            return Err(NativeError::Message("Invalid live effect parameter action"));
        }
        self.targets[[node, parameter]] = action[4];
        self.remaining[[node, parameter]] = duration;
        if duration == 0 {
            self.parameters[[node, parameter]] = action[4];
            self.steps[[node, parameter]] = 0.0;
        } else {
            self.steps[[node, parameter]] =
                (action[4] - self.parameters[[node, parameter]]) / duration as f64;
        }
        Ok(())
    }

    fn set_frozen(&mut self, node: usize, frozen: bool) {
        let granular = self.granulators[node]
            .as_mut()
            .expect("validated live granulator");
        if frozen && !granular.frozen {
            granular.frozen_start = granular.history_start;
        }
        granular.frozen = frozen;
    }

    fn advance_ramps(&mut self) {
        for node in 0..self.parameters.nrows() {
            for parameter in 0..self.parameters.ncols() {
                if self.remaining[[node, parameter]] == 0 {
                    continue;
                }
                self.remaining[[node, parameter]] -= 1;
                if self.remaining[[node, parameter]] == 0 {
                    self.parameters[[node, parameter]] = self.targets[[node, parameter]];
                    self.steps[[node, parameter]] = 0.0;
                } else {
                    self.parameters[[node, parameter]] += self.steps[[node, parameter]];
                }
            }
        }
    }

    fn process_filters(&mut self, node: usize, channels: usize, rate: f64) -> NativeResult<()> {
        for (stage, filter) in self.filter_rows.iter().enumerate() {
            if filter.node != node {
                continue;
            }
            let mut cutoff = self.resolved[[node, filter.parameter]];
            let q = self.resolved[[node, filter.parameter + 1]];
            if !cutoff.is_finite()
                || q <= 0.0
                || (!filter.clamp && (cutoff < filter.minimum || cutoff > filter.maximum))
            {
                return Err(NativeError::Message("Invalid live filter cutoff/Q"));
            }
            cutoff = cutoff.clamp(filter.minimum, filter.maximum);
            let g = (PI * cutoff / rate).tan();
            let d = q * (1.0 + g * g) + g;
            let a1 = if q < 1.0 {
                q / d
            } else {
                1.0 / (1.0 + g * (g + 1.0 / q))
            };
            let a2 = g * a1;
            let a3 = g * a2;
            for (channel, input) in self.wet[..channels].iter_mut().enumerate() {
                let (s1, s2) = (
                    self.filter_states[[stage, channel, 0]],
                    self.filter_states[[stage, channel, 1]],
                );
                let v3 = *input - s2;
                let v1 = a1 * s1 + a2 * v3;
                let v2 = s2 + a2 * s1 + a3 * v3;
                let band = if q < 1.0 {
                    s1 / d + (g / d) * v3
                } else {
                    v1 / q
                };
                *input = match filter.response {
                    0 => v2,
                    1 => *input - band - v2,
                    2 => band,
                    _ => *input - band,
                };
                self.filter_states[[stage, channel, 0]] = 2.0 * v1 - s1;
                self.filter_states[[stage, channel, 1]] = 2.0 * v2 - s2;
                if filter.state_floor > 0.0 {
                    for state in 0..2 {
                        if self.filter_states[[stage, channel, state]].abs() < filter.state_floor {
                            self.filter_states[[stage, channel, state]] = 0.0;
                        }
                    }
                }
                if !input.is_finite()
                    || !self.filter_states[[stage, channel, 0]].is_finite()
                    || !self.filter_states[[stage, channel, 1]].is_finite()
                {
                    return Err(NativeError::Message("Non-finite live filter state"));
                }
            }
        }
        Ok(())
    }

    fn process_tap_delay(&mut self, node: usize, channels: usize, rate: f64) -> NativeResult<()> {
        let delay_seconds = self.resolved[[node, 0]];
        let feedback = self.resolved[[node, 3]];
        let delay = self.tap_delays[node]
            .as_mut()
            .expect("validated live tap delay");
        let frames = delay.history.nrows();
        let delay_frames = (delay_seconds * rate).max(1.0);
        if !delay_seconds.is_finite()
            || delay_seconds <= 0.0
            || delay_frames > frames as f64
            || !feedback.is_finite()
            || !(0.0..1.0).contains(&feedback)
        {
            return Err(NativeError::Message("Invalid live tap delay parameters"));
        }
        delay.retime(delay_frames)?;
        for channel in 0..channels {
            let wet = delay
                .heads
                .iter()
                .filter(|h| h.active)
                .map(|h| h.gain * delay_read(&delay.history, delay.write, h.delay, channel))
                .sum::<f64>();
            let value = self.wet[channel];
            let stored = value + feedback * wet;
            let stored = if stored.abs() < delay.state_floor {
                0.0
            } else {
                stored
            };
            delay.non_silent -= usize::from(
                delay.history[[delay.write, channel]].abs() >= self.tail_thresholds[node],
            );
            delay.non_silent += usize::from(stored.abs() >= self.tail_thresholds[node]);
            delay.history[[delay.write, channel]] = stored;
            self.wet[channel] = wet;
        }
        delay.write = (delay.write + 1) % frames;
        delay.advance_heads();
        Ok(())
    }

    fn process_reverb(&mut self, node: usize, input: usize, rate: f64) -> NativeResult<()> {
        let delay = self.tap_delays[node]
            .as_mut()
            .expect("prepared reverb predelay");
        let seconds = self.resolved[[node, 0]];
        if !seconds.is_finite() || seconds < 0.0 || seconds > self.parameters[[node, 6]] {
            return Err(NativeError::Message("Invalid reverb predelay"));
        }
        delay.retime(seconds * rate)?;
        let threshold = self.tail_thresholds[node];
        let mut delayed = [0.0; 2];
        for (channel, sample) in delayed.iter_mut().enumerate() {
            let dry = if self.widths[input] == 1 {
                self.port_values[[node, 0, 0]] / std::f64::consts::SQRT_2
            } else {
                self.port_values[[node, 0, channel]]
            };
            delay.non_silent -=
                usize::from(delay.history[[delay.write, channel]].abs() >= threshold);
            delay.non_silent += usize::from(dry.abs() >= threshold);
            delay.history[[delay.write, channel]] = dry;
            *sample = delay
                .heads
                .iter()
                .filter(|head| head.active)
                .map(|head| {
                    head.gain * delay_read(&delay.history, delay.write, head.delay, channel)
                })
                .sum();
        }
        delay.write = (delay.write + 1) % delay.history.nrows();
        delay.advance_heads();
        let output = self.reverbs[node]
            .as_mut()
            .expect("prepared reverb")
            .process(
                delayed,
                self.resolved[[node, 3]],
                self.resolved[[node, 5]],
                rate,
                delay.state_floor,
                threshold,
            )?;
        self.wet[..2].copy_from_slice(&output);
        Ok(())
    }

    fn process_modulated_delay(
        &mut self,
        node: usize,
        channels: usize,
        rate: f64,
    ) -> NativeResult<()> {
        let base = self.resolved[[node, 0]];
        let depth = self.resolved[[node, 3]];
        let rate_hz = self.resolved[[node, 4]];
        let feedback = self.resolved[[node, 5]];
        let delay = self.modulated_delays[node]
            .as_mut()
            .expect("validated live modulated delay");
        let frames = delay.history.nrows();
        if !base.is_finite()
            || !depth.is_finite()
            || !rate_hz.is_finite()
            || base <= depth
            || depth < 0.0
            || rate_hz < 0.0
            || !feedback.is_finite()
            || !(0.0..1.0).contains(&feedback)
        {
            return Err(NativeError::Message(
                "Invalid live modulated delay parameters",
            ));
        }
        for channel in 0..channels {
            let phase = (delay.phase + delay.phase_offsets[channel]) % 1.0;
            let lfo = match delay.waveform {
                0 => (2.0 * PI * phase).sin(),
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
            let delay_frames = ((base + depth * lfo) * rate).max(1.0);
            if delay_frames > frames as f64 {
                return Err(NativeError::Message("Invalid live modulated delay read"));
            }
            let wet = delay_read(&delay.history, delay.write, delay_frames, channel);
            let stored = self.wet[channel] + feedback * wet;
            let stored = if stored.abs() < delay.state_floor {
                0.0
            } else {
                stored
            };
            delay.non_silent -= usize::from(
                delay.history[[delay.write, channel]].abs() >= self.tail_thresholds[node],
            );
            delay.non_silent += usize::from(stored.abs() >= self.tail_thresholds[node]);
            delay.history[[delay.write, channel]] = stored;
            self.wet[channel] = wet;
        }
        delay.write = (delay.write + 1) % frames;
        delay.phase = (delay.phase + rate_hz / rate) % 1.0;
        Ok(())
    }

    pub(crate) fn snapshot(&self) -> EffectSnapshot {
        EffectSnapshot {
            kinds: self.kinds.clone(),
            rate: self.rate,
            main_input: self.main_input,
            tail_thresholds: self.tail_thresholds.clone(),
            ended: self.ended.clone(),
            stopped: self.stopped,
            widths: self.widths.clone(),
            input_routes: self.input_routes.clone(),
            sources: self.sources.clone(),
            output_source: self.output_source,
            parameters: self.parameters.clone(),
            generators: self.generators.clone(),
            targets: self.targets.clone(),
            steps: self.steps.clone(),
            remaining: self.remaining.clone(),
            filter_rows: self.filter_rows.clone(),
            filter_states: self.filter_states.clone(),
            granulators: self.granulators.clone(),
            tap_delays: self.tap_delays.clone(),
            modulated_delays: self.modulated_delays.clone(),
            reductions: self.reductions.clone(),
            reverbs: self.reverbs.clone(),
            waveshapers: self.waveshapers.clone(),
            phasers: self.phasers.clone(),
            highpasses: self.highpasses.clone(),
            alignment: self.alignment.clone(),
            control_delays: self.control_delays.clone(),
        }
    }

    pub(crate) fn validate_snapshot(&self, snapshot: &EffectSnapshot) -> PyResult<()> {
        if self.kinds != snapshot.kinds
            || self.rate != snapshot.rate
            || self.main_input != snapshot.main_input
            || self.tail_thresholds != snapshot.tail_thresholds
            || self.widths != snapshot.widths
            || self.input_routes != snapshot.input_routes
            || match (&self.generators, &snapshot.generators) {
                (Some(a), Some(b)) => {
                    a.targets != b.targets || a.maps != b.maps || !a.bank.same_definition(&b.bank)
                }
                (None, None) => false,
                _ => true,
            }
            || self.sources != snapshot.sources
            || self.output_source != snapshot.output_source
            || self.filter_rows != snapshot.filter_rows
            || self.granulators.len() != snapshot.granulators.len()
            || self.tap_delays.len() != snapshot.tap_delays.len()
            || self.modulated_delays.len() != snapshot.modulated_delays.len()
            || self.highpasses.len() != snapshot.highpasses.len()
            || self
                .highpasses
                .iter()
                .zip(&snapshot.highpasses)
                .any(|(a, b)| match (a, b) {
                    (Some(a), Some(b)) => !a.same_capacity(b),
                    (None, None) => false,
                    _ => true,
                })
            || self
                .phasers
                .iter()
                .zip(&snapshot.phasers)
                .any(|(a, b)| match (a, b) {
                    (Some(a), Some(b)) => !a.same_capacity(b),
                    (None, None) => false,
                    _ => true,
                })
            || self
                .waveshapers
                .iter()
                .zip(&snapshot.waveshapers)
                .any(|(a, b)| match (a, b) {
                    (Some(a), Some(b)) => !a.same_capacity(b),
                    (None, None) => false,
                    _ => true,
                })
            || self
                .reverbs
                .iter()
                .zip(&snapshot.reverbs)
                .any(|(a, b)| match (a, b) {
                    (Some(a), Some(b)) => a
                        .lines
                        .iter()
                        .map(Vec::len)
                        .ne(b.lines.iter().map(Vec::len)),
                    (None, None) => false,
                    _ => true,
                })
            || self
                .granulators
                .iter()
                .zip(&snapshot.granulators)
                .any(|(a, b)| match (a, b) {
                    (Some(a), Some(b)) => !a.same_capacity(b),
                    (None, None) => false,
                    _ => true,
                })
            || self
                .modulated_delays
                .iter()
                .zip(&snapshot.modulated_delays)
                .any(|(a, b)| match (a, b) {
                    (Some(a), Some(b)) => {
                        a.history.raw_dim() != b.history.raw_dim()
                            || a.waveform != b.waveform
                            || a.phase_offsets != b.phase_offsets
                            || a.state_floor != b.state_floor
                    }
                    (None, None) => false,
                    _ => true,
                })
            || self
                .tap_delays
                .iter()
                .zip(&snapshot.tap_delays)
                .any(|(a, b)| match (a, b) {
                    (Some(a), Some(b)) => {
                        a.history.raw_dim() != b.history.raw_dim()
                            || a.heads.len() != b.heads.len()
                            || a.fade_frames != b.fade_frames
                            || a.state_floor != b.state_floor
                    }
                    (None, None) => false,
                    _ => true,
                })
            || self.parameters.raw_dim() != snapshot.parameters.raw_dim()
            || self.matrices.iter().enumerate().any(|(node, matrix)| {
                matrix
                    .as_ref()
                    .is_some_and(|matrix| !matrix.matches_parameters(snapshot.parameters.row(node)))
            })
            || self.filter_states.raw_dim() != snapshot.filter_states.raw_dim()
        {
            return Err(PyValueError::new_err(
                "Snapshot belongs to a different live effect graph",
            ));
        }
        Ok(())
    }

    pub(crate) fn restore(&mut self, snapshot: &EffectSnapshot) -> PyResult<()> {
        self.validate_snapshot(snapshot)?;
        self.parameters.assign(&snapshot.parameters);
        self.generators.clone_from(&snapshot.generators);
        self.targets.assign(&snapshot.targets);
        self.steps.assign(&snapshot.steps);
        self.remaining.assign(&snapshot.remaining);
        self.filter_states.assign(&snapshot.filter_states);
        self.granulators.clone_from(&snapshot.granulators);
        self.tap_delays.clone_from(&snapshot.tap_delays);
        self.modulated_delays.clone_from(&snapshot.modulated_delays);
        self.reductions.clone_from(&snapshot.reductions);
        self.reverbs.clone_from(&snapshot.reverbs);
        self.waveshapers.clone_from(&snapshot.waveshapers);
        self.phasers.clone_from(&snapshot.phasers);
        self.highpasses.clone_from(&snapshot.highpasses);
        self.alignment.clone_from(&snapshot.alignment);
        self.control_delays.clone_from(&snapshot.control_delays);
        self.ended.clone_from(&snapshot.ended);
        self.stopped = snapshot.stopped;
        Ok(())
    }
}

impl LiveTapDelay {
    fn retime(&mut self, frames: f64) -> NativeResult<()> {
        self.heads[self.current].gain = (1.0
            - self
                .heads
                .iter()
                .enumerate()
                .filter(|(i, head)| head.active && *i != self.current)
                .map(|(_, head)| head.gain)
                .sum::<f64>())
        .max(0.0);
        if self.heads[self.current].delay != frames {
            let index =
                self.heads
                    .iter()
                    .position(|head| !head.active)
                    .ok_or(NativeError::Message(
                        "Tap retiming exceeds prepared read-head capacity",
                    ))?;
            let current = &mut self.heads[self.current];
            current.remaining = self.fade_frames;
            current.step = current.gain / self.fade_frames as f64;
            self.heads[index] = DelayHead {
                delay: frames,
                gain: 0.0,
                step: 0.0,
                remaining: 0,
                active: true,
            };
            self.current = index;
        }
        Ok(())
    }

    fn advance_heads(&mut self) {
        for (index, head) in self.heads.iter_mut().enumerate() {
            if !head.active || index == self.current {
                continue;
            }
            head.remaining -= 1;
            if head.remaining == 0 {
                head.active = false;
                head.gain = 0.0;
            } else {
                head.gain = (head.gain - head.step).max(0.0);
            }
        }
    }
}

fn delay_read(history: &Array2<f64>, write: usize, delay: f64, channel: usize) -> f64 {
    let whole = delay.floor() as usize;
    let fraction = delay - whole as f64;
    let index = (write + history.nrows() - whole) % history.nrows();
    let older = (index + history.nrows() - 1) % history.nrows();
    (1.0 - fraction) * history[[index, channel]] + fraction * history[[older, channel]]
}

impl LiveGranulator {
    fn new(
        history_frames: usize,
        grain_frames: usize,
        maximum_grains: usize,
        channels: usize,
        freeze_crossfade_frames: usize,
        maximum_duration_seconds: f64,
    ) -> Self {
        Self {
            history: Array2::zeros((history_frames, channels)),
            history_start: 0,
            history_len: 0,
            history_write: 0,
            grains: (0..maximum_grains)
                .map(|_| LiveGrain {
                    samples: Array2::zeros((grain_frames, channels)),
                    len: 0,
                    index: 0,
                    gain: 0.0,
                    active: false,
                })
                .collect(),
            phase: 0.0,
            counter: 0,
            frame: 0,
            frozen: false,
            frozen_start: 0,
            order: vec![0; maximum_grains],
            active_count: 0,
            freeze_crossfade_frames,
            maximum_duration_seconds,
        }
    }

    fn same_capacity(&self, other: &Self) -> bool {
        self.history.raw_dim() == other.history.raw_dim()
            && self.grains.len() == other.grains.len()
            && self.order.len() == other.order.len()
            && self.freeze_crossfade_frames == other.freeze_crossfade_frames
            && self.maximum_duration_seconds == other.maximum_duration_seconds
            && self
                .grains
                .iter()
                .zip(&other.grains)
                .all(|(a, b)| a.samples.raw_dim() == b.samples.raw_dim())
    }

    fn process(
        &mut self,
        input: &mut [f64],
        parameters: [f64; 5],
        rate: f64,
        ended: bool,
    ) -> NativeResult<()> {
        let [duration_seconds, density, lookback, ratio, jitter_seconds] = parameters;
        if duration_seconds <= 0.0
            || duration_seconds > self.maximum_duration_seconds
            || density <= 0.0
            || density > rate
            || lookback < 0.0
            || ratio <= 0.0
            || jitter_seconds < 0.0
        {
            return Err(NativeError::Message("Invalid live granulator parameter"));
        }
        if !self.frozen && !ended {
            for (channel, value) in input.iter().enumerate() {
                self.history[[self.history_write, channel]] = *value;
            }
            self.history_write = (self.history_write + 1) % self.history.nrows();
            if self.history_len < self.history.nrows() {
                self.history_len += 1;
            } else {
                self.history_start += 1;
            }
        }
        self.phase += density / rate;
        if self.phase >= 1.0 {
            self.phase -= 1.0;
            let launch_counter = self.counter;
            self.counter = self.counter.wrapping_add(1);
            let slot = if !ended || self.frozen {
                self.grains.iter().position(|v| !v.active)
            } else {
                None
            };
            if let Some(slot) = slot {
                let duration = (duration_seconds * rate).round_ties_even().max(2.0) as usize;
                let end = self.frame as f64 - lookback * rate
                    + jitter_seconds * rate * granular_jitter(launch_counter);
                let start = end - (duration - 1) as f64 * ratio;
                if duration <= self.grains[slot].samples.nrows()
                    && self.capture(slot, start, ratio, duration)
                {
                    let grain = &mut self.grains[slot];
                    grain.len = duration;
                    grain.index = 0;
                    grain.gain = 1.0 / (density * duration_seconds).max(1.0).sqrt();
                    grain.active = true;
                    self.order[self.active_count] = slot;
                    self.active_count += 1;
                }
            }
        }
        input.fill(0.0);
        for index in 0..self.active_count {
            let grain = &mut self.grains[self.order[index]];
            let window = 0.5 - 0.5 * (2.0 * PI * grain.index as f64 / grain.len as f64).cos();
            for (channel, value) in input.iter_mut().enumerate() {
                *value += grain.samples[[grain.index, channel]] * window * grain.gain;
            }
            grain.index += 1;
            if grain.index == grain.len {
                grain.active = false;
            }
        }
        let mut active = 0;
        for index in 0..self.active_count {
            if self.grains[self.order[index]].active {
                self.order[active] = self.order[index];
                active += 1;
            }
        }
        self.active_count = active;
        self.frame = self
            .frame
            .checked_add(1)
            .ok_or(NativeError::Message("Live granulator frame overflow"))?;
        Ok(())
    }

    fn capture(&mut self, slot: usize, start: f64, ratio: f64, frames: usize) -> bool {
        if self.frozen && self.history_len < 2 {
            return false;
        }
        for i in 0..frames {
            let position = start + i as f64 * ratio;
            if self.frozen {
                let crossfade = self
                    .freeze_crossfade_frames
                    .min(self.history_len.saturating_sub(1) / 2);
                let period = self.history_len - crossfade;
                let offset = (position - self.history_start as f64).rem_euclid(period as f64);
                let lower = offset.floor() as usize;
                let upper = (lower + 1) % period;
                let fraction = offset - lower as f64;
                for channel in 0..self.history.ncols() {
                    let a = self.frozen_value(lower, period, crossfade, channel);
                    let b = self.frozen_value(upper, period, crossfade, channel);
                    self.grains[slot].samples[[i, channel]] = a + (b - a) * fraction;
                }
                continue;
            }
            let lower = position.floor() as i64;
            let upper = lower + 1;
            if lower < self.history_start || upper >= self.history_start + self.history_len as i64 {
                return false;
            }
            let fraction = position - lower as f64;
            let first = self.history_index(lower);
            let second = self.history_index(upper);
            for channel in 0..self.history.ncols() {
                let a = self.history[[first, channel]];
                let b = self.history[[second, channel]];
                self.grains[slot].samples[[i, channel]] = a + (b - a) * fraction;
            }
        }
        true
    }

    fn frozen_value(&self, offset: usize, period: usize, crossfade: usize, channel: usize) -> f64 {
        if offset >= crossfade {
            return self.history[[
                self.history_index(self.history_start + offset as i64),
                channel,
            ]];
        }
        let weight = offset as f64 / crossfade as f64;
        let outgoing = self.history[[
            self.history_index(self.history_start + period as i64 + offset as i64),
            channel,
        ]];
        let incoming = self.history[[
            self.history_index(self.history_start + offset as i64),
            channel,
        ]];
        (1.0 - weight) * outgoing + weight * incoming
    }

    fn history_index(&self, frame: i64) -> usize {
        let oldest =
            (self.history_write + self.history.nrows() - self.history_len) % self.history.nrows();
        (oldest + (frame - self.history_start) as usize) % self.history.nrows()
    }
}

fn granular_jitter(counter: u64) -> f64 {
    let mut value = counter.wrapping_add(0x9E37_79B9_7F4A_7C15);
    value = (value ^ (value >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    value = (value ^ (value >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    value ^= value >> 31;
    2.0 * ((value >> 11) as f64 / 2_f64.powi(53)) - 1.0
}
