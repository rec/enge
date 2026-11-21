//! Portable tuning preparation for MIDI conversion.

use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use ufor::codec::{self, Format};
use ufor::tuning::Tuning;

#[pyclass(frozen, skip_from_py_object)]
pub(crate) struct PreparedTuning(Tuning);

#[pymethods]
impl PreparedTuning {
    #[new]
    fn new(document: &str) -> PyResult<Self> {
        codec::parse_tuning(document, Format::Json)
            .map(Self)
            .map_err(|error| PyValueError::new_err(error.to_string()))
    }

    fn frequency_hz(&self, note: i64) -> PyResult<f64> {
        self.0
            .evaluate(note)
            .map(|frequency| frequency.as_f64())
            .map_err(|error| PyValueError::new_err(error.to_string()))
    }
}
