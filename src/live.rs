//! One native owner for persistent generated sources and callback scratch.

use crate::queue::{ActionBatch, BATCH_ACTIONS};
use crate::runtime::{
    Segment, SynthRuntime, SynthRuntimeSnapshot, envelope_value, segments, total_frames,
};
use crate::sampler::{Cursor, SampleBuffer, Selection, State};
use numpy::ndarray::{Array2, ArrayViewMut2};
use numpy::{PyReadonlyArray2, PyReadwriteArray2, PyUntypedArrayMethods};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use rtrb::{Consumer, Producer, RingBuffer};

struct Source {
    runtime: SourceRuntime,
    producer: Producer<ActionBatch>,
    consumer: Consumer<ActionBatch>,
    action_scratch: Vec<f64>,
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

#[pyclass(frozen, skip_from_py_object)]
#[derive(Clone)]
pub struct LiveRuntimeSnapshot {
    sources: Vec<SourceSnapshot>,
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
    gain: f64,
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
        gain_db: f64,
    ) -> PyResult<Self> {
        if runtimes.is_empty()
            || maximum_block_frames == 0
            || action_capacity == 0
            || batch_capacity == 0
            || action_capacity < BATCH_ACTIONS
            || !gain_db.is_finite()
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
            gain: 10_f64.powf(gain_db / 20.0),
            frame,
            failed: false,
        })
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
        });
        Ok(self.sources.len() - 1)
    }

    fn submit(&mut self, source: usize, actions: PyReadonlyArray2<'_, f64>) -> PyResult<()> {
        let Some(source) = self.sources.get_mut(source) else {
            return Err(PyValueError::new_err("Unknown live runtime source"));
        };
        if actions.shape()[1] != 6
            || actions.shape()[0] == 0
            || actions.shape()[0] > BATCH_ACTIONS
            || actions.as_array().iter().any(|v| !v.is_finite())
        {
            return Err(PyValueError::new_err("Invalid live action batch"));
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
        source
            .producer
            .push(batch)
            .map_err(|_| PyValueError::new_err("Live action queue capacity exceeded"))
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
        if self.sources.iter().any(|v| v.consumer.slots() != 0) {
            return Err(PyValueError::new_err(
                "Cannot snapshot a live runtime with queued actions",
            ));
        }
        Ok(LiveRuntimeSnapshot {
            sources: self.sources.iter().map(|v| v.runtime.snapshot()).collect(),
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
        for (source, state) in self.sources.iter_mut().zip(&snapshot.sources) {
            source.runtime.restore(state)?;
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
            let batches = source.consumer.slots();
            let mut count = 0;
            for _ in 0..batches {
                let batch = source.consumer.pop().expect("captured queue boundary");
                if count + batch.len > self.action_capacity {
                    return Err(PyValueError::new_err(
                        "Live actions exceed callback capacity",
                    ));
                }
                for action in &batch.actions[..batch.len] {
                    source.action_scratch[count * 6..count * 6 + 6].copy_from_slice(action);
                    count += 1;
                }
            }
            let mut scratch = ArrayViewMut2::from_shape(
                (frames, self.channels),
                &mut self
                    .source_scratch
                    .as_slice_mut()
                    .expect("owned callback scratch")[..frames * self.channels],
            )
            .expect("validated callback scratch shape");
            source
                .runtime
                .render_into(scratch.view_mut(), &source.action_scratch[..count * 6])?;
            for (mixed, value) in self
                .output_scratch
                .iter_mut()
                .take(frames * self.channels)
                .zip(scratch.iter())
            {
                *mixed += *value;
            }
        }
        for value in self.output_scratch.iter_mut().take(frames * self.channels) {
            *value *= self.gain;
            if !value.is_finite() {
                return Err(PyValueError::new_err("Non-finite live runtime output"));
            }
        }
        self.frame = self
            .frame
            .checked_add(frames)
            .ok_or_else(|| PyValueError::new_err("Live runtime frame overflow"))?;
        Ok(())
    }
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
