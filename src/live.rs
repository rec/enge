//! One native owner for persistent generated sources and callback scratch.

use crate::queue::{ActionBatch, BATCH_ACTIONS};
use crate::runtime::{SynthRuntime, SynthRuntimeSnapshot};
use numpy::ndarray::{Array2, ArrayViewMut2};
use numpy::{PyReadonlyArray2, PyReadwriteArray2, PyUntypedArrayMethods};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use rtrb::{Consumer, Producer, RingBuffer};

struct Source {
    runtime: SynthRuntime,
    producer: Producer<ActionBatch>,
    consumer: Consumer<ActionBatch>,
    action_scratch: Vec<f64>,
}

#[pyclass(frozen, skip_from_py_object)]
#[derive(Clone)]
pub struct LiveRuntimeSnapshot {
    sources: Vec<SynthRuntimeSnapshot>,
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
                    runtime,
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
            gain: 10_f64.powf(gain_db / 20.0),
            frame,
            failed: false,
        })
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
            source.runtime.render_into(
                scratch.view_mut(),
                0.0,
                1.0,
                0.0,
                &source.action_scratch[..count * 6],
            )?;
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
