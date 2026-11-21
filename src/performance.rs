//! Portable preparation for MIDI conversion and instrument traces.

use numpy::{IntoPyArray, PyArray1, PyReadonlyArray1};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use ufor::codec::{self, Format};
use ufor::control::schedule::{EventCoordinate, schedule_commands};
use ufor::envelfo::{PreparationLimits, Waveform};
use ufor::identifier::Identifier;
use ufor::number::Rational;
use ufor::oscillator::Oscillator;
use ufor::samples::eligibility::VoiceEligibility;
use ufor::samples::routing::{ChannelRoute, channel_matrix, sampler_matrix};
use ufor::synth_pitch;
use ufor::tuning::Tuning;
use ufor::voice_start;

type CommandCoordinate = (i64, u64, Option<(i64, i64)>);
type OperatorTuning<'py> = (PyReadonlyArray1<'py, f64>, PyReadonlyArray1<'py, f64>);

#[pyfunction]
#[pyo3(signature = (inputs, outputs, routes, spatial=None))]
pub(crate) fn prepare_channel_routes(
    inputs: Vec<String>,
    outputs: Vec<String>,
    routes: Vec<(String, String, f64)>,
    spatial: Option<(&str, f64, f64)>,
) -> PyResult<Vec<Vec<f64>>> {
    let routes: Vec<_> = routes
        .into_iter()
        .map(|(input, output, gain)| ChannelRoute {
            input,
            output,
            gain,
        })
        .collect();
    let matrix = if let Some((law, pan, balance)) = spatial {
        let law = serde_json::from_value(serde_json::Value::String(law.to_owned()))
            .map_err(|error| PyValueError::new_err(error.to_string()))?;
        sampler_matrix(&inputs, &outputs, &routes, law, pan, balance)
    } else {
        channel_matrix(&inputs, &outputs, &routes)
    };
    matrix.map_err(|error| PyValueError::new_err(error.to_string()))
}

#[pyfunction]
#[pyo3(signature = (pitch_hz, tuning, operator=None))]
pub(crate) fn tuned_frequencies<'py>(
    py: Python<'py>,
    pitch_hz: f64,
    tuning: PyReadonlyArray1<'py, f64>,
    operator: Option<OperatorTuning<'py>>,
) -> PyResult<Bound<'py, PyArray1<f64>>> {
    let tuning = tuning.as_array();
    let frequencies = if let Some((ratios, cents)) = operator {
        let ratios = ratios.as_array();
        let cents = cents.as_array();
        if ratios.len() != tuning.len() || cents.len() != tuning.len() {
            return Err(PyValueError::new_err(
                "Mismatched FM operator tuning arrays",
            ));
        }
        tuning
            .iter()
            .zip(ratios.iter())
            .zip(cents.iter())
            .map(|((&tuning, &ratio), &cents)| {
                synth_pitch::operator_frequency(pitch_hz, ratio, tuning, cents)
            })
            .collect::<Result<Vec<_>, _>>()
    } else {
        tuning
            .iter()
            .map(|&cents| synth_pitch::frequency(pitch_hz, cents))
            .collect::<Result<Vec<_>, _>>()
    };
    Ok(frequencies
        .map_err(|error| PyValueError::new_err(error.to_string()))?
        .into_pyarray(py))
}

#[pyclass(frozen, skip_from_py_object)]
pub(crate) struct PreparedOscillator(Oscillator);

#[pymethods]
impl PreparedOscillator {
    #[new]
    fn new(document: &str) -> PyResult<Self> {
        codec::parse_oscillator(document, Format::Json)
            .map(Self)
            .map_err(|error| PyValueError::new_err(error.to_string()))
    }

    fn shape(&self) -> (u8, f64) {
        let waveform = match self.0.waveform() {
            Waveform::Sine => 0,
            Waveform::Square => 1,
            Waveform::Triangle => 2,
        };
        (waveform, self.0.duty_cycle_f64())
    }

    fn gain(&self, note: i64) -> PyResult<f64> {
        self.0
            .gain(note)
            .map_err(|error| PyValueError::new_err(error.to_string()))
    }
}

#[pyfunction]
#[pyo3(signature = (tempo, sample_rate, events, *, max_records=10_000))]
pub(crate) fn schedule_envelfo_commands(
    tempo: &str,
    sample_rate: i64,
    events: Vec<CommandCoordinate>,
    max_records: usize,
) -> PyResult<Vec<(usize, i64, u64)>> {
    let invalid = |error: ufor::error::Error| PyValueError::new_err(error.to_string());
    let limits = PreparationLimits::new(max_records).map_err(invalid)?;
    let tempo = codec::parse_tempo_map(tempo, Format::Json, limits).map_err(invalid)?;
    let events = events
        .into_iter()
        .map(|(tick, ordinal, division)| {
            Ok(EventCoordinate {
                tick,
                ordinal,
                quantize_beats: division.map(|(n, d)| Rational::new(n, d)).transpose()?,
            })
        })
        .collect::<Result<Vec<_>, ufor::error::Error>>()
        .map_err(invalid)?;
    Ok(schedule_commands(&tempo, sample_rate, &events)
        .map_err(invalid)?
        .into_iter()
        .map(|event| (event.index, event.tick, event.ordinal))
        .collect())
}

#[pyfunction]
#[pyo3(signature = (seed, voice_id, pitch_hz, frequency_offset_hz, is_noise))]
pub(crate) fn prepare_voice_start(
    seed: u64,
    voice_id: String,
    pitch_hz: Option<f64>,
    frequency_offset_hz: f64,
    is_noise: bool,
) -> PyResult<(Option<f64>, u64, Option<u64>)> {
    let invalid = |error: ufor::error::Error| PyValueError::new_err(error.to_string());
    let id = Identifier::new(voice_id).map_err(invalid)?;
    let values = voice_start::prepare_values(seed, &id, pitch_hz, frequency_offset_hz, is_noise)
        .map_err(invalid)?;
    Ok((values.pitch_hz, values.envelfo_key, values.noise_key))
}

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
