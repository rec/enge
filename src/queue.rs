//! Fixed-size action batches carried by a lock-free SPSC ring buffer.

use numpy::{PyReadonlyArray2, PyReadwriteArray2, PyUntypedArrayMethods};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use rtrb::{Consumer, Producer, RingBuffer};

pub(crate) const BATCH_ACTIONS: usize = 64;

#[derive(Clone)]
pub(crate) struct ActionBatch {
    pub(crate) len: usize,
    pub(crate) actions: [[f64; 6]; BATCH_ACTIONS],
}

#[pyclass(unsendable)]
pub struct ActionQueue {
    producer: Producer<ActionBatch>,
    consumer: Consumer<ActionBatch>,
    pending: Option<ActionBatch>,
}

#[pymethods]
impl ActionQueue {
    #[new]
    fn new(batch_capacity: usize) -> PyResult<Self> {
        if batch_capacity == 0 {
            return Err(PyValueError::new_err(
                "Action queue capacity must be positive",
            ));
        }
        let (producer, consumer) = RingBuffer::new(batch_capacity);
        Ok(Self {
            producer,
            consumer,
            pending: None,
        })
    }

    fn push(&mut self, actions: PyReadonlyArray2<'_, f64>) -> PyResult<()> {
        if actions.shape()[1] != 6
            || actions.shape()[0] == 0
            || actions.shape()[0] > BATCH_ACTIONS
            || actions.as_array().iter().any(|v| !v.is_finite())
        {
            return Err(PyValueError::new_err("Invalid action batch"));
        }
        let mut batch = ActionBatch {
            len: actions.shape()[0],
            actions: [[0.0; 6]; BATCH_ACTIONS],
        };
        for (target, source) in batch.actions.iter_mut().zip(actions.as_array().rows()) {
            target.copy_from_slice(
                source
                    .as_slice()
                    .ok_or_else(|| PyValueError::new_err("Action batch rows must be contiguous"))?,
            );
        }
        self.producer
            .push(batch)
            .map_err(|_| PyValueError::new_err("Action queue capacity exceeded"))
    }

    fn drain_into(&mut self, mut output: PyReadwriteArray2<'_, f64>) -> PyResult<usize> {
        if !output.is_c_contiguous() || output.shape()[1] != 6 {
            return Err(PyValueError::new_err("Invalid action queue output"));
        }
        let batches = self.consumer.slots() + usize::from(self.pending.is_some());
        let mut written = 0;
        for _ in 0..batches {
            let batch = self
                .pending
                .take()
                .or_else(|| self.consumer.pop().ok())
                .expect("captured queue boundary");
            if written + batch.len > output.shape()[0] {
                self.pending = Some(batch);
                break;
            }
            for action in &batch.actions[..batch.len] {
                for (column, value) in action.iter().enumerate() {
                    output.as_array_mut()[[written, column]] = *value;
                }
                written += 1;
            }
        }
        Ok(written)
    }

    fn queued_batches(&self) -> usize {
        self.consumer.slots() + usize::from(self.pending.is_some())
    }
}
