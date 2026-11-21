//! Portable preparation for MIDI conversion and instrument traces.

use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use ufor::codec::{self, Format};
use ufor::samples::eligibility::VoiceEligibility;
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

#[pyclass(frozen, skip_from_py_object)]
pub(crate) struct PreparedVoiceEligibility(VoiceEligibility);

#[pymethods]
impl PreparedVoiceEligibility {
    #[new]
    #[pyo3(signature = (document, *, max_records=10_000))]
    fn new(document: &str, max_records: usize) -> PyResult<Self> {
        codec::parse_voice_eligibility(document, Format::Json, max_records)
            .map(Self)
            .map_err(|error| PyValueError::new_err(error.to_string()))
    }

    fn matching_voices(&self, trigger: &str, key: i64, velocity: f64) -> PyResult<Vec<String>> {
        let trigger = serde_json::from_value(serde_json::Value::String(trigger.to_owned()))
            .map_err(|error| PyValueError::new_err(error.to_string()))?;
        Ok(self
            .0
            .matching_voices(trigger, key, velocity)
            .map(|name| name.as_str().to_owned())
            .collect())
    }
}
