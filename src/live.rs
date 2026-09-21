//! One native owner for persistent generated sources and callback scratch.

use crate::queue::{ActionBatch, BATCH_ACTIONS};
use crate::runtime::{
    Segment, SynthRuntime, SynthRuntimeSnapshot, envelope_value, segments, total_frames,
};
use crate::sampler::{Cursor, SampleBuffer, Selection, State};
use numpy::ndarray::{Array2, Array3, ArrayViewMut2};
use numpy::{PyReadonlyArray2, PyReadwriteArray2, PyUntypedArrayMethods};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use rtrb::{Consumer, Producer, RingBuffer};
use std::f64::consts::PI;

struct Source {
    runtime: SourceRuntime,
    producer: Producer<ActionBatch>,
    consumer: Consumer<ActionBatch>,
    action_scratch: Vec<f64>,
    action_count: usize,
}

#[derive(Clone)]
enum SourceRuntime {
    Generated(Box<SynthRuntime>),
    Sample(Box<SampleRuntime>),
}

#[derive(Clone)]
struct SampleRuntime {
    buffer: SampleBuffer,
    initial_cursor: Cursor,
    routes: Array2<f64>,
    initial: f64,
    attack: Vec<Segment>,
    release: Vec<Segment>,
    minimum_hold_frames: f64,
    cursors: Vec<Cursor>,
    steps: Vec<f64>,
    gains: Vec<f64>,
    active: Vec<bool>,
    ages: Vec<usize>,
    release_frames: Vec<Option<f64>>,
    release_levels: Vec<f64>,
    source_values: Vec<f64>,
    frame: usize,
}

#[derive(Clone)]
enum SourceSnapshot {
    Generated(Box<SynthRuntimeSnapshot>),
    Sample(Box<SampleRuntime>),
}

struct EffectRuntime {
    kinds: Vec<u8>,
    sources: Array2<i64>,
    output_source: usize,
    parameters: Array2<f64>,
    targets: Array2<f64>,
    steps: Array2<f64>,
    remaining: Array2<usize>,
    filter_rows: Vec<EffectFilter>,
    filter_states: Array3<f64>,
    values: Array2<f64>,
    wet: Vec<f64>,
    producer: Producer<ActionBatch>,
    consumer: Consumer<ActionBatch>,
    action_scratch: Vec<f64>,
    action_count: usize,
}

#[derive(Clone, PartialEq)]
struct EffectFilter {
    node: usize,
    response: u8,
    parameter: usize,
    state_floor: f64,
}

#[derive(Clone)]
struct EffectSnapshot {
    kinds: Vec<u8>,
    sources: Array2<i64>,
    output_source: usize,
    parameters: Array2<f64>,
    targets: Array2<f64>,
    steps: Array2<f64>,
    remaining: Array2<usize>,
    filter_rows: Vec<EffectFilter>,
    filter_states: Array3<f64>,
}

#[pyclass(frozen, skip_from_py_object)]
#[derive(Clone)]
pub struct LiveRuntimeSnapshot {
    sources: Vec<SourceSnapshot>,
    effects: Option<EffectSnapshot>,
    frame: usize,
    failed: bool,
}

#[pyclass(unsendable)]
pub struct LiveRuntime {
    sources: Vec<Source>,
    source_scratch: Array2<f64>,
    output_scratch: Array2<f64>,
    maximum_block_frames: usize,
    channels: usize,
    action_capacity: usize,
    rate: f64,
    effects: Option<EffectRuntime>,
    frame: usize,
    failed: bool,
}

#[pymethods]
impl LiveRuntime {
    #[new]
    fn new(
        py: Python<'_>,
        runtimes: Vec<Py<SynthRuntime>>,
        maximum_block_frames: usize,
        action_capacity: usize,
        batch_capacity: usize,
    ) -> PyResult<Self> {
        if runtimes.is_empty()
            || maximum_block_frames == 0
            || action_capacity == 0
            || batch_capacity == 0
            || action_capacity < BATCH_ACTIONS
        {
            return Err(PyValueError::new_err("Invalid live runtime capacity"));
        }
        let runtimes = runtimes
            .iter()
            .map(|v| v.borrow(py).clone())
            .collect::<Vec<_>>();
        let channels = runtimes[0].channels();
        let rate = runtimes[0].rate();
        let frame = runtimes[0].frame();
        if runtimes
            .iter()
            .any(|v| v.channels() != channels || v.rate() != rate || v.frame() != frame)
        {
            return Err(PyValueError::new_err(
                "Live runtime sources must share rate, channels, and frame",
            ));
        }
        let sources = runtimes
            .into_iter()
            .map(|runtime| {
                let (producer, consumer) = RingBuffer::new(batch_capacity);
                Source {
                    runtime: SourceRuntime::Generated(Box::new(runtime)),
                    producer,
                    consumer,
                    action_scratch: vec![0.0; action_capacity * 6],
                    action_count: 0,
                }
            })
            .collect();
        Ok(Self {
            sources,
            source_scratch: Array2::zeros((maximum_block_frames, channels)),
            output_scratch: Array2::zeros((maximum_block_frames, channels)),
            maximum_block_frames,
            channels,
            action_capacity,
            rate,
            effects: None,
            frame,
            failed: false,
        })
    }

    #[allow(clippy::too_many_arguments)]
    fn set_effect_graph(
        &mut self,
        kinds: Vec<u8>,
        sources: PyReadonlyArray2<'_, i64>,
        parameters: PyReadonlyArray2<'_, f64>,
        output_source: usize,
        filters: PyReadonlyArray2<'_, f64>,
        batch_capacity: usize,
    ) -> PyResult<()> {
        if self.frame != 0 || self.effects.is_some() || batch_capacity == 0 {
            return Err(PyValueError::new_err("Invalid live effect graph setup"));
        }
        self.effects = Some(EffectRuntime::new(
            kinds,
            sources,
            parameters,
            output_source,
            filters,
            self.channels,
            self.action_capacity,
            batch_capacity,
        )?);
        Ok(())
    }

    fn submit_effects(&mut self, actions: PyReadonlyArray2<'_, f64>) -> PyResult<()> {
        let Some(effects) = &mut self.effects else {
            return Err(PyValueError::new_err("Live runtime has no effect graph"));
        };
        push_batch(&mut effects.producer, actions, "effect")
    }

    #[allow(clippy::too_many_arguments)]
    fn add_sample(
        &mut self,
        py: Python<'_>,
        buffer: Py<SampleBuffer>,
        selection: Selection,
        state: State,
        rate: f64,
        routes: PyReadonlyArray2<'_, f64>,
        initial: f64,
        attack: PyReadonlyArray2<'_, f64>,
        release: PyReadonlyArray2<'_, f64>,
        minimum_hold_frames: f64,
        voices: usize,
        batch_capacity: usize,
    ) -> PyResult<usize> {
        if self.frame != 0
            || voices == 0
            || batch_capacity == 0
            || rate != self.rate
            || routes.shape() != [buffer.borrow(py).audio.ncols(), self.channels]
            || routes.as_array().iter().any(|v| !v.is_finite())
            || !initial.is_finite()
            || !minimum_hold_frames.is_finite()
            || minimum_hold_frames < 0.0
        {
            return Err(PyValueError::new_err("Invalid live sample source"));
        }
        let buffer = buffer.borrow(py).clone();
        let mut cursor = Cursor::new(selection, rate, state);
        cursor.settle(0);
        if cursor.exhausted.is_some() {
            return Err(PyValueError::new_err("Live sample starts exhausted"));
        }
        let source_channels = buffer.audio.ncols();
        let runtime = SampleRuntime {
            buffer,
            initial_cursor: cursor,
            routes: routes.as_array().to_owned(),
            initial,
            attack: segments(attack)?,
            release: segments(release)?,
            minimum_hold_frames,
            cursors: vec![cursor; voices],
            steps: vec![1.0; voices],
            gains: vec![0.0; voices],
            active: vec![false; voices],
            ages: vec![0; voices],
            release_frames: vec![None; voices],
            release_levels: vec![0.0; voices],
            source_values: vec![0.0; source_channels],
            frame: 0,
        };
        let (producer, consumer) = RingBuffer::new(batch_capacity);
        self.sources.push(Source {
            runtime: SourceRuntime::Sample(Box::new(runtime)),
            producer,
            consumer,
            action_scratch: vec![0.0; self.action_capacity * 6],
            action_count: 0,
        });
        Ok(self.sources.len() - 1)
    }

    fn submit(&mut self, source: usize, actions: PyReadonlyArray2<'_, f64>) -> PyResult<()> {
        let Some(source) = self.sources.get_mut(source) else {
            return Err(PyValueError::new_err("Unknown live runtime source"));
        };
        push_batch(&mut source.producer, actions, "source")
    }

    fn process_into(
        &mut self,
        py: Python<'_>,
        mut output: PyReadwriteArray2<'_, f64>,
    ) -> PyResult<()> {
        if !output.is_c_contiguous()
            || output.shape()[0] == 0
            || output.shape()[0] > self.maximum_block_frames
            || output.shape()[1] != self.channels
        {
            return Err(PyValueError::new_err("Invalid live runtime output"));
        }
        let frames = output.shape()[0];
        let result = py.detach(|| self.process(frames));
        if let Err(error) = result {
            output.as_array_mut().fill(0.0);
            self.failed = true;
            return Err(error);
        }
        for (target, value) in output
            .as_array_mut()
            .iter_mut()
            .zip(self.output_scratch.iter())
        {
            *target = *value;
        }
        Ok(())
    }

    fn snapshot(&self) -> PyResult<LiveRuntimeSnapshot> {
        if self
            .sources
            .iter()
            .any(|v| v.consumer.slots() != 0 || v.action_count != 0)
            || self
                .effects
                .as_ref()
                .is_some_and(|v| v.consumer.slots() != 0 || v.action_count != 0)
        {
            return Err(PyValueError::new_err(
                "Cannot snapshot a live runtime with queued actions",
            ));
        }
        Ok(LiveRuntimeSnapshot {
            sources: self.sources.iter().map(|v| v.runtime.snapshot()).collect(),
            effects: self.effects.as_ref().map(EffectRuntime::snapshot),
            frame: self.frame,
            failed: self.failed,
        })
    }

    fn restore(&mut self, snapshot: &LiveRuntimeSnapshot) -> PyResult<()> {
        if snapshot.sources.len() != self.sources.len() {
            return Err(PyValueError::new_err(
                "Snapshot belongs to a different live runtime",
            ));
        }
        if snapshot.effects.is_some() != self.effects.is_some() {
            return Err(PyValueError::new_err(
                "Snapshot belongs to a different live effect graph",
            ));
        }
        for (source, state) in self.sources.iter_mut().zip(&snapshot.sources) {
            source.runtime.restore(state)?;
        }
        if let (Some(effects), Some(state)) = (&mut self.effects, &snapshot.effects) {
            effects.restore(state)?;
        }
        self.frame = snapshot.frame;
        self.failed = snapshot.failed;
        Ok(())
    }

    fn failed(&self) -> bool {
        self.failed
    }

    fn reset_failure(&mut self) {
        self.failed = false;
    }
}

impl LiveRuntime {
    fn process(&mut self, frames: usize) -> PyResult<()> {
        for value in self.output_scratch.iter_mut().take(frames * self.channels) {
            *value = 0.0;
        }
        if self.failed {
            return Ok(());
        }
        for source in &mut self.sources {
            source.action_count = drain_batches(
                &mut source.consumer,
                &mut source.action_scratch,
                self.action_capacity,
                "source",
            )?;
        }
        if let Some(effects) = &mut self.effects {
            effects.action_count = drain_batches(
                &mut effects.consumer,
                &mut effects.action_scratch,
                self.action_capacity,
                "effect",
            )?;
        }
        for source in &mut self.sources {
            let mut scratch = ArrayViewMut2::from_shape(
                (frames, self.channels),
                &mut self
                    .source_scratch
                    .as_slice_mut()
                    .expect("owned callback scratch")[..frames * self.channels],
            )
            .expect("validated callback scratch shape");
            source.runtime.render_into(
                scratch.view_mut(),
                &source.action_scratch[..source.action_count * 6],
            )?;
            source.action_count = 0;
            for (mixed, value) in self
                .output_scratch
                .iter_mut()
                .take(frames * self.channels)
                .zip(scratch.iter())
            {
                *mixed += *value;
            }
        }
        if let Some(effects) = &mut self.effects {
            effects.process(&mut self.output_scratch, frames, self.channels, self.rate)?;
        }
        if self
            .output_scratch
            .iter()
            .take(frames * self.channels)
            .any(|v| !v.is_finite())
        {
            return Err(PyValueError::new_err("Non-finite live runtime output"));
        }
        self.frame = self
            .frame
            .checked_add(frames)
            .ok_or_else(|| PyValueError::new_err("Live runtime frame overflow"))?;
        Ok(())
    }
}

impl EffectRuntime {
    #[allow(clippy::too_many_arguments)]
    fn new(
        kinds: Vec<u8>,
        sources: PyReadonlyArray2<'_, i64>,
        parameters: PyReadonlyArray2<'_, f64>,
        output_source: usize,
        filters: PyReadonlyArray2<'_, f64>,
        channels: usize,
        action_capacity: usize,
        batch_capacity: usize,
    ) -> PyResult<Self> {
        let nodes = kinds.len();
        if kinds.iter().any(|v| *v > 2)
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
            || output_source > nodes
            || filters.shape()[1] != 4
            || filters.as_array().iter().any(|v| !v.is_finite())
        {
            return Err(PyValueError::new_err("Invalid live effect graph"));
        }
        let sources = sources.as_array().to_owned();
        for node in 0..nodes {
            let ports = if kinds[node] == 1 { 2 } else { 1 };
            for port in 0..ports {
                let source = sources[[node, port]];
                if source < 0 || source as usize > node {
                    return Err(PyValueError::new_err(
                        "Effect source must precede its consumer",
                    ));
                }
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
        let (producer, consumer) = RingBuffer::new(batch_capacity);
        Ok(Self {
            kinds,
            sources,
            output_source,
            targets: parameters.clone(),
            steps: Array2::zeros(parameters.raw_dim()),
            remaining: Array2::zeros(parameters.raw_dim()),
            parameters,
            filter_states: Array3::zeros((filter_rows.len(), channels, 2)),
            filter_rows,
            values: Array2::zeros((nodes + 1, channels)),
            wet: vec![0.0; channels],
            producer,
            consumer,
            action_scratch: vec![0.0; action_capacity * 6],
            action_count: 0,
        })
    }

    fn process(
        &mut self,
        output: &mut Array2<f64>,
        frames: usize,
        channels: usize,
        rate: f64,
    ) -> PyResult<()> {
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
                return Err(PyValueError::new_err("Invalid live effect block"));
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
            for channel in 0..channels {
                self.values[[0, channel]] = output[[frame, channel]];
            }
            for node in 0..self.kinds.len() {
                let first = self.sources[[node, 0]] as usize;
                let authored_mix = self.parameters[[node, 1]];
                let bypass = self.parameters[[node, 2]];
                if !(0.0..=1.0).contains(&authored_mix) || !(0.0..=1.0).contains(&bypass) {
                    return Err(PyValueError::new_err("Invalid live effect mix"));
                }
                let mix = authored_mix * bypass;
                for channel in 0..channels {
                    self.wet[channel] = self.values[[first, channel]];
                }
                match self.kinds[node] {
                    0 => {
                        let gain = 10_f64.powf(self.parameters[[node, 0]] / 20.0);
                        for value in &mut self.wet {
                            *value *= gain;
                        }
                    }
                    1 => {
                        let second = self.sources[[node, 1]] as usize;
                        for (channel, value) in self.wet.iter_mut().enumerate() {
                            *value *= self.values[[second, channel]];
                        }
                    }
                    _ => self.process_filters(node, rate)?,
                }
                for channel in 0..channels {
                    self.values[[node + 1, channel]] =
                        (1.0 - mix) * self.values[[first, channel]] + mix * self.wet[channel];
                }
            }
            for channel in 0..channels {
                output[[frame, channel]] = self.values[[self.output_source, channel]];
            }
            self.advance_ramps();
        }
        self.action_count = 0;
        Ok(())
    }

    fn apply_action(&mut self, action: &[f64]) -> PyResult<()> {
        let kind = action[1] as usize;
        let node = action[2] as usize;
        let parameter = action[3] as usize;
        let duration = action[5] as usize;
        if action[1] != kind as f64
            || kind != 0
            || action[2] != node as f64
            || node >= self.parameters.nrows()
            || action[3] != parameter as f64
            || parameter >= self.parameters.ncols()
            || action[5] != duration as f64
        {
            return Err(PyValueError::new_err("Invalid live effect action"));
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

    fn process_filters(&mut self, node: usize, rate: f64) -> PyResult<()> {
        for (stage, filter) in self.filter_rows.iter().enumerate() {
            if filter.node != node {
                continue;
            }
            let cutoff = self.parameters[[node, filter.parameter]];
            let q = self.parameters[[node, filter.parameter + 1]];
            if !cutoff.is_finite() || cutoff <= 0.0 || cutoff >= rate / 2.0 || q <= 0.0 {
                return Err(PyValueError::new_err("Invalid live filter cutoff/Q"));
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
            for (channel, input) in self.wet.iter_mut().enumerate() {
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
                    return Err(PyValueError::new_err("Non-finite live filter state"));
                }
            }
        }
        Ok(())
    }

    fn snapshot(&self) -> EffectSnapshot {
        EffectSnapshot {
            kinds: self.kinds.clone(),
            sources: self.sources.clone(),
            output_source: self.output_source,
            parameters: self.parameters.clone(),
            targets: self.targets.clone(),
            steps: self.steps.clone(),
            remaining: self.remaining.clone(),
            filter_rows: self.filter_rows.clone(),
            filter_states: self.filter_states.clone(),
        }
    }

    fn restore(&mut self, snapshot: &EffectSnapshot) -> PyResult<()> {
        if self.kinds != snapshot.kinds
            || self.sources != snapshot.sources
            || self.output_source != snapshot.output_source
            || self.filter_rows != snapshot.filter_rows
            || self.parameters.raw_dim() != snapshot.parameters.raw_dim()
            || self.filter_states.raw_dim() != snapshot.filter_states.raw_dim()
        {
            return Err(PyValueError::new_err(
                "Snapshot belongs to a different live effect graph",
            ));
        }
        self.parameters.assign(&snapshot.parameters);
        self.targets.assign(&snapshot.targets);
        self.steps.assign(&snapshot.steps);
        self.remaining.assign(&snapshot.remaining);
        self.filter_states.assign(&snapshot.filter_states);
        Ok(())
    }
}

fn push_batch(
    producer: &mut Producer<ActionBatch>,
    actions: PyReadonlyArray2<'_, f64>,
    subject: &str,
) -> PyResult<()> {
    if actions.shape()[1] != 6
        || actions.shape()[0] == 0
        || actions.shape()[0] > BATCH_ACTIONS
        || actions.as_array().iter().any(|v| !v.is_finite())
    {
        return Err(PyValueError::new_err(format!(
            "Invalid live {subject} action batch"
        )));
    }
    let mut batch = ActionBatch {
        len: actions.shape()[0],
        actions: [[0.0; 6]; BATCH_ACTIONS],
    };
    for (target, values) in batch.actions.iter_mut().zip(actions.as_array().rows()) {
        target.copy_from_slice(
            values
                .as_slice()
                .ok_or_else(|| PyValueError::new_err("Action rows must be contiguous"))?,
        );
    }
    producer
        .push(batch)
        .map_err(|_| PyValueError::new_err(format!("Live {subject} action queue is full")))
}

fn drain_batches(
    consumer: &mut Consumer<ActionBatch>,
    scratch: &mut [f64],
    action_capacity: usize,
    subject: &str,
) -> PyResult<usize> {
    let batches = consumer.slots();
    let mut count = 0;
    for _ in 0..batches {
        let batch = consumer.pop().expect("captured queue boundary");
        if count + batch.len > action_capacity {
            return Err(PyValueError::new_err(format!(
                "Live {subject} actions exceed callback capacity"
            )));
        }
        for action in &batch.actions[..batch.len] {
            scratch[count * 6..count * 6 + 6].copy_from_slice(action);
            count += 1;
        }
    }
    Ok(count)
}

impl SourceRuntime {
    fn render_into(&mut self, output: ArrayViewMut2<'_, f64>, actions: &[f64]) -> PyResult<()> {
        match self {
            Self::Generated(runtime) => runtime.render_into(output, 0.0, 1.0, 0.0, actions),
            Self::Sample(runtime) => runtime.render_into(output, actions),
        }
    }

    fn snapshot(&self) -> SourceSnapshot {
        match self {
            Self::Generated(runtime) => SourceSnapshot::Generated(Box::new(runtime.snapshot())),
            Self::Sample(runtime) => SourceSnapshot::Sample(runtime.clone()),
        }
    }

    fn restore(&mut self, snapshot: &SourceSnapshot) -> PyResult<()> {
        match (self, snapshot) {
            (Self::Generated(runtime), SourceSnapshot::Generated(state)) => runtime.restore(state),
            (Self::Sample(runtime), SourceSnapshot::Sample(state)) => {
                if runtime.routes != state.routes
                    || runtime.initial != state.initial
                    || runtime.attack != state.attack
                    || runtime.release != state.release
                    || runtime.minimum_hold_frames != state.minimum_hold_frames
                    || runtime.cursors.len() != state.cursors.len()
                    || runtime.buffer.audio.as_ref() != state.buffer.audio.as_ref()
                {
                    return Err(PyValueError::new_err(
                        "Snapshot belongs to a different live sample source",
                    ));
                }
                runtime.clone_from(state);
                Ok(())
            }
            _ => Err(PyValueError::new_err(
                "Snapshot belongs to a different live source kind",
            )),
        }
    }
}

impl SampleRuntime {
    fn render_into(&mut self, mut output: ArrayViewMut2<'_, f64>, actions: &[f64]) -> PyResult<()> {
        let frames = output.nrows();
        if output.ncols() != self.routes.ncols()
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
            return Err(PyValueError::new_err("Invalid live sample block"));
        }
        output.fill(0.0);
        let mut action = 0;
        let channels = output.ncols();
        for offset in 0..frames {
            while action < actions.len() && actions[action] as usize == offset {
                self.apply_action(&actions[action..action + 6])?;
                action += 6;
            }
            for voice in 0..self.active.len() {
                if !self.active[voice] {
                    continue;
                }
                let age = self.ages[voice] as f64;
                let level = if let Some(release_frame) = self.release_frames[voice] {
                    if age >= release_frame.ceil() && !self.cursors[voice].released {
                        self.cursors[voice].release();
                    }
                    let end = release_frame + total_frames(&self.release);
                    if age >= end.ceil() {
                        self.active[voice] = false;
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
                let frame = i64::try_from(self.frame + offset)
                    .map_err(|_| PyValueError::new_err("Live sample frame overflow"))?;
                let cursor = &mut self.cursors[voice];
                cursor.settle(frame);
                if cursor.exhausted.is_some() {
                    self.active[voice] = false;
                    continue;
                }
                let mut following = *cursor;
                if cursor.position != 0.0 {
                    following.index += following.direction;
                    following.position = 0.0;
                    following.settle(frame);
                }
                for (channel, value) in self.source_values.iter_mut().enumerate() {
                    let first = cursor.read(&self.buffer.audio, channel);
                    let last = if following.exhausted.is_some() {
                        first
                    } else {
                        following.read(&self.buffer.audio, channel)
                    };
                    *value = if cursor.position == 0.0 {
                        first
                    } else {
                        first + (cursor.position / cursor.rate) * (last - first)
                    };
                }
                for channel in 0..channels {
                    let mixed = self
                        .source_values
                        .iter()
                        .enumerate()
                        .map(|(source, value)| value * self.routes[[source, channel]])
                        .sum::<f64>();
                    output[[offset, channel]] += mixed * level * self.gains[voice];
                }
                cursor.advance(self.steps[voice], frame + 1);
                self.ages[voice] += 1;
            }
        }
        self.frame = self
            .frame
            .checked_add(frames)
            .ok_or_else(|| PyValueError::new_err("Live sample frame overflow"))?;
        Ok(())
    }

    fn apply_action(&mut self, action: &[f64]) -> PyResult<()> {
        let kind = action[1] as usize;
        let voice = action[2] as usize;
        if action[1] != kind as f64 || action[2] != voice as f64 || voice >= self.active.len() {
            return Err(PyValueError::new_err("Invalid live sample action"));
        }
        match kind {
            0 => {
                if self.active[voice]
                    || !action[3].is_finite()
                    || action[3] <= 0.0
                    || !action[4].is_finite()
                    || action[4] < 0.0
                {
                    return Err(PyValueError::new_err("Invalid live sample start"));
                }
                self.active[voice] = true;
                self.cursors[voice] = self.initial_cursor;
                self.steps[voice] = action[3];
                self.gains[voice] = action[4];
                self.ages[voice] = 0;
                self.release_frames[voice] = None;
            }
            1 => {
                if self.active[voice] && self.release_frames[voice].is_none() {
                    let release_frame = (self.ages[voice] as f64).max(self.minimum_hold_frames);
                    self.release_levels[voice] =
                        envelope_value(self.initial, &self.attack, release_frame);
                    self.release_frames[voice] = Some(release_frame);
                }
            }
            2 => self.active[voice] = false,
            3 => {
                if !self.active[voice] || action[3] <= 0.0 || action[4] < 0.0 || action[5] != 0.0 {
                    return Err(PyValueError::new_err("Invalid live sample control"));
                }
                self.steps[voice] = action[3];
                self.gains[voice] = action[4];
            }
            _ => return Err(PyValueError::new_err("Unknown live sample action")),
        }
        Ok(())
    }
}
