//! Persistent DSP state and processing shared by effect graph owners.

use crate::compressor;
use crate::error::{NativeError, NativeResult};
use crate::queue::ActionBatch;
use crate::runtime::controls::ControlBank;
use crate::runtime::generators::GeneratorBank;
use crate::runtime::{RouteMap, sum_partials};
use numpy::ndarray::{Array2, Array3};
use numpy::{PyReadonlyArray2, PyUntypedArrayMethods};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use rtrb::{Consumer, Producer, RingBuffer};
use std::f64::consts::PI;

pub(crate) struct EffectRuntime {
    kinds: Vec<u8>,
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
}

#[derive(Clone)]
pub(crate) struct EffectSnapshot {
    kinds: Vec<u8>,
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
    freeze_crossfade_frames: usize,
}

#[derive(Clone)]
struct LiveTapDelay {
    history: Array2<f64>,
    write: usize,
    heads: Vec<DelayHead>,
    current: usize,
    fade_frames: usize,
    state_floor: f64,
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
}

#[derive(Clone)]
struct LiveGrain {
    samples: Array2<f64>,
    len: usize,
    index: usize,
    gain: f64,
    active: bool,
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
        input_routes: Vec<Vec<i64>>,
        maximum_block_frames: usize,
        action_capacity: usize,
        batch_capacity: usize,
    ) -> PyResult<Self> {
        let nodes = kinds.len();
        let inputs = input_routes.len();
        let channels = widths.iter().copied().max().unwrap_or(0);
        if kinds.iter().any(|v| *v > 7)
            || inputs == 0
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
            || filters.shape()[1] != 4
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
                if !(kinds[node] == 7 && port == 1)
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
            {
                return Err(PyValueError::new_err("Invalid live effect filter"));
            }
            filter_rows.push(EffectFilter {
                node,
                response,
                parameter,
                state_floor: row[3],
            });
        }
        let parameters = parameters.as_array().to_owned();
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
                || parameters.ncols() < 8
            {
                return Err(PyValueError::new_err("Invalid live granulator"));
            }
            prepared_granulators[node] = Some(LiveGranulator::new(
                history_frames,
                maximum_grains,
                widths[node + inputs],
                freeze_crossfade_frames,
            ));
        }
        if kinds
            .iter()
            .enumerate()
            .any(|(i, kind)| *kind == 3 && prepared_granulators[i].is_none())
        {
            return Err(PyValueError::new_err("Missing live granulator setup"));
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
                || kinds[node] != 5
                || history_frames == 0
                || prepared_tap_delays[node].is_some()
                || parameters.ncols() < 5
                || fade_frames == 0
                || fade_frames.checked_add(1).is_none()
                || parameters[[node, 4]] < 0.0
            {
                return Err(PyValueError::new_err("Invalid live tap delay"));
            }
            prepared_tap_delays[node] = Some(LiveTapDelay {
                history: Array2::zeros((history_frames, widths[node + inputs])),
                write: 0,
                heads: (0..fade_frames + 1)
                    .map(|i| DelayHead {
                        delay: (parameters[[node, 0]] * rate).max(1.0),
                        gain: if i == 0 { 1.0 } else { 0.0 },
                        step: 0.0,
                        remaining: 0,
                        active: i == 0,
                    })
                    .collect(),
                current: 0,
                fade_frames,
                state_floor: parameters[[node, 4]],
            });
        }
        if kinds
            .iter()
            .enumerate()
            .any(|(i, kind)| *kind == 5 && prepared_tap_delays[i].is_none())
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
            });
        }
        if kinds
            .iter()
            .enumerate()
            .any(|(i, kind)| *kind == 6 && prepared_modulated_delays[i].is_none())
        {
            return Err(PyValueError::new_err("Missing live modulated delay setup"));
        }
        let (producer, consumer) = RingBuffer::new(batch_capacity);
        Ok(Self {
            reductions: vec![0.0; nodes],
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
        rate: f64,
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
            while action < count * 6 && self.action_scratch[action] as usize == frame {
                let values: [f64; 6] = self.action_scratch[action..action + 6]
                    .try_into()
                    .expect("validated action width");
                self.apply_action(&values)?;
                action += 6;
            }
            self.resolved.assign(&self.parameters);
            if let Some(generators) = &mut self.generators {
                let at = start
                    .checked_add(frame)
                    .ok_or(NativeError::Message("Effect frame overflow"))?;
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
                    self.values[[input, channel]] = self.inputs[[input, frame, channel]];
                }
            }
            for node in 0..self.kinds.len() {
                let channels = self.widths[node + inputs];
                let first = self.sources[[node, 0]] as usize;
                let authored_mix = self.resolved[[node, 1]];
                let bypass = self.resolved[[node, 2]];
                if !(0.0..=1.0).contains(&authored_mix) || !(0.0..=1.0).contains(&bypass) {
                    return Err(NativeError::Message("Invalid live effect mix"));
                }
                let mix = authored_mix * bypass;
                for channel in 0..channels {
                    self.wet[channel] = self.values[[first, channel]];
                }
                match self.kinds[node] {
                    0 => {
                        let gain = 10_f64.powf(self.resolved[[node, 0]] / 20.0);
                        for value in &mut self.wet[..channels] {
                            *value *= gain;
                        }
                    }
                    1 => {
                        let second = self.sources[[node, 1]] as usize;
                        for (channel, value) in self.wet[..channels].iter_mut().enumerate() {
                            *value *= self.values[[second, channel]];
                        }
                    }
                    2 => self.process_filters(node, channels, rate)?,
                    3 => {
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
                            .process(&mut self.wet[..channels], parameters, rate)?;
                    }
                    4 => {
                        let drive = self.resolved[[node, 0]];
                        if drive <= 0.0 {
                            return Err(NativeError::Message("Invalid live soft clip drive"));
                        }
                        let normalization = drive.tanh();
                        for value in &mut self.wet[..channels] {
                            *value = (drive * *value).tanh() / normalization;
                        }
                    }
                    5 => self.process_tap_delay(node, channels, rate)?,
                    6 => self.process_modulated_delay(node, channels, rate)?,
                    _ => {
                        let detector = self.sources[[node, 1]] as usize;
                        let peak = (0..self.widths[detector])
                            .map(|channel| self.values[[detector, channel]].abs())
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
                }
                for channel in 0..channels {
                    self.values[[node + inputs, channel]] =
                        (1.0 - mix) * self.values[[first, channel]] + mix * self.wet[channel];
                }
            }
            for channel in 0..self.widths[self.output_source] {
                output[[frame, channel]] = self.values[[self.output_source, channel]];
            }
            self.advance_ramps();
        }
        self.action_count = 0;
        Ok(())
    }

    fn apply_action(&mut self, action: &[f64]) -> NativeResult<()> {
        let kind = action[1] as usize;
        let node = action[2] as usize;
        let parameter = action[3] as usize;
        let duration = action[5] as usize;
        if action[1] != kind as f64
            || action[2] != node as f64
            || node >= self.parameters.nrows()
            || action[5] != duration as f64
        {
            return Err(NativeError::Message("Invalid live effect action"));
        }
        if kind == 1 {
            if self.kinds[node] != 3
                || action[3] != 0.0
                || !matches!(action[4], 0.0 | 1.0)
                || duration != 0
            {
                return Err(NativeError::Message("Invalid live granulator freeze"));
            }
            self.granulators[node]
                .as_mut()
                .expect("validated live granulator")
                .frozen = action[4] == 1.0;
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
            let cutoff = self.resolved[[node, filter.parameter]];
            let q = self.resolved[[node, filter.parameter + 1]];
            if !cutoff.is_finite() || cutoff <= 0.0 || cutoff >= rate / 2.0 || q <= 0.0 {
                return Err(NativeError::Message("Invalid live filter cutoff/Q"));
            }
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
        delay.heads[delay.current].gain = (1.0
            - delay
                .heads
                .iter()
                .enumerate()
                .filter(|(i, h)| h.active && *i != delay.current)
                .map(|(_, h)| h.gain)
                .sum::<f64>())
        .max(0.0);
        if delay.heads[delay.current].delay != delay_frames {
            let index = delay
                .heads
                .iter()
                .position(|h| !h.active)
                .ok_or(NativeError::Message(
                    "Tap retiming exceeds prepared read-head capacity",
                ))?;
            let current = &mut delay.heads[delay.current];
            current.remaining = delay.fade_frames;
            current.step = current.gain / delay.fade_frames as f64;
            delay.heads[index] = DelayHead {
                delay: delay_frames,
                gain: 0.0,
                step: 0.0,
                remaining: 0,
                active: true,
            };
            delay.current = index;
        }
        for channel in 0..channels {
            let wet = delay
                .heads
                .iter()
                .filter(|h| h.active)
                .map(|h| h.gain * delay_read(&delay.history, delay.write, h.delay, channel))
                .sum::<f64>();
            let value = self.wet[channel];
            let stored = value + feedback * wet;
            delay.history[[delay.write, channel]] = if stored.abs() < delay.state_floor {
                0.0
            } else {
                stored
            };
            self.wet[channel] = wet;
        }
        delay.write = (delay.write + 1) % frames;
        for (index, head) in delay.heads.iter_mut().enumerate() {
            if !head.active || index == delay.current {
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
            delay.history[[delay.write, channel]] = if stored.abs() < delay.state_floor {
                0.0
            } else {
                stored
            };
            self.wet[channel] = wet;
        }
        delay.write = (delay.write + 1) % frames;
        delay.phase = (delay.phase + rate_hz / rate) % 1.0;
        Ok(())
    }

    pub(crate) fn snapshot(&self) -> EffectSnapshot {
        EffectSnapshot {
            kinds: self.kinds.clone(),
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
        }
    }

    pub(crate) fn validate_snapshot(&self, snapshot: &EffectSnapshot) -> PyResult<()> {
        if self.kinds != snapshot.kinds
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
        Ok(())
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
        maximum_grains: usize,
        channels: usize,
        freeze_crossfade_frames: usize,
    ) -> Self {
        Self {
            history: Array2::zeros((history_frames, channels)),
            history_start: 0,
            history_len: 0,
            history_write: 0,
            grains: (0..maximum_grains)
                .map(|_| LiveGrain {
                    samples: Array2::zeros((history_frames, channels)),
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
            freeze_crossfade_frames,
        }
    }

    fn same_capacity(&self, other: &Self) -> bool {
        self.history.raw_dim() == other.history.raw_dim()
            && self.grains.len() == other.grains.len()
            && self.freeze_crossfade_frames == other.freeze_crossfade_frames
            && self
                .grains
                .iter()
                .zip(&other.grains)
                .all(|(a, b)| a.samples.raw_dim() == b.samples.raw_dim())
    }

    fn process(&mut self, input: &mut [f64], parameters: [f64; 5], rate: f64) -> NativeResult<()> {
        let [duration_seconds, density, lookback, ratio, jitter_seconds] = parameters;
        if duration_seconds <= 0.0
            || density <= 0.0
            || density > rate
            || lookback < 0.0
            || ratio <= 0.0
            || jitter_seconds < 0.0
        {
            return Err(NativeError::Message("Invalid live granulator parameter"));
        }
        if !self.frozen {
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
            if let Some(slot) = self.grains.iter().position(|v| !v.active) {
                let duration = (duration_seconds * rate).round().max(2.0) as usize;
                let end = self.frame as f64 - lookback * rate
                    + jitter_seconds * rate * granular_jitter(launch_counter);
                let start = end - (duration - 1) as f64 * ratio;
                if duration <= self.history.nrows() && self.capture(slot, start, ratio, duration) {
                    let grain = &mut self.grains[slot];
                    grain.len = duration;
                    grain.index = 0;
                    grain.gain = 1.0 / (density * duration_seconds).max(1.0).sqrt();
                    grain.active = true;
                }
            }
        }
        input.fill(0.0);
        for grain in &mut self.grains {
            if !grain.active {
                continue;
            }
            let window = 0.5 - 0.5 * (2.0 * PI * grain.index as f64 / grain.len as f64).cos();
            for (channel, value) in input.iter_mut().enumerate() {
                *value += grain.samples[[grain.index, channel]] * window * grain.gain;
            }
            grain.index += 1;
            if grain.index == grain.len {
                grain.active = false;
            }
        }
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
