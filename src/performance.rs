//! Portable preparation for MIDI conversion and instrument traces.

use numpy::{IntoPyArray, PyArray1, PyReadonlyArray1};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::PyInt;
use std::collections::BTreeMap;
use ufor::codec::{self, Format};
use ufor::control::schedule::{EventCoordinate, schedule_commands};
use ufor::envelfo::{PreparationLimits, Waveform};
use ufor::identifier::Identifier;
use ufor::number::Rational;
use ufor::oscillator::{Oscillator, PhaseOrigin, phase_at_frame};
use ufor::performance_binding::{ControlValue, HeldNotes, MidiBindings, RepeatedKeyRelease};
use ufor::samples::controls::ControlDeclaration;
use ufor::samples::playback::{self, sample_step, tuned_pitch_ratio};
use ufor::samples::routing::{ChannelRoute, channel_matrix, sampler_matrix};
use ufor::synth::SynthInstrument;
use ufor::synth_pitch;
use ufor::time::{PpqnClock, duration_frames};
use ufor::voice_start;

type CommandCoordinate = (i64, u64, Option<(i64, i64)>);
type OperatorTuning<'py> = (PyReadonlyArray1<'py, f64>, PyReadonlyArray1<'py, f64>);

#[pyfunction]
pub(crate) fn prepare_hold_frames(seconds: &str, sample_rate: i64) -> PyResult<(i64, i64)> {
    ufor::quantity::exact(seconds, "seconds")
        .and_then(|seconds| duration_frames(seconds, sample_rate))
        .map(|frames| (frames.numerator(), frames.denominator()))
        .map_err(|error| PyValueError::new_err(error.to_string()))
}

#[pyclass(frozen, skip_from_py_object)]
pub(crate) struct PreparedEventValidator(BTreeMap<Identifier, ControlDeclaration>);

#[pymethods]
impl PreparedEventValidator {
    #[new]
    fn new(controls: &str) -> PyResult<Self> {
        let value = codec::parse_value(controls, Format::Json)
            .map_err(|error| PyValueError::new_err(error.to_string()))?;
        serde_json::from_value(value)
            .map(Self)
            .map_err(|error| PyValueError::new_err(error.to_string()))
    }

    fn validate(&self, payload: &str) -> PyResult<()> {
        validate_event_payload(payload, &self.0)
    }
}

fn validate_event_payload(
    payload: &str,
    controls: &BTreeMap<Identifier, ControlDeclaration>,
) -> PyResult<()> {
    codec::parse_performance_payload(payload, Format::Json)
        .and_then(|event| event.validate_controls(controls))
        .map_err(|error| PyValueError::new_err(error.to_string()))
}

#[pyclass(frozen, skip_from_py_object)]
pub(crate) struct PreparedMidiBindings(MidiBindings);

#[pyclass(skip_from_py_object)]
pub(crate) struct MidiHeldNotes(HeldNotes);

#[pymethods]
impl MidiHeldNotes {
    #[new]
    fn new(source: String, limit: usize) -> PyResult<Self> {
        Identifier::new(source)
            .and_then(|source| HeldNotes::new(source, limit))
            .map(Self)
            .map_err(|error| PyValueError::new_err(error.to_string()))
    }

    fn source(&self) -> &str {
        self.0.source().as_str()
    }

    fn stopped(&self) -> bool {
        self.0.stopped()
    }

    fn note_on(
        &mut self,
        channel: u8,
        key: u8,
        velocity: u8,
        bindings: &PreparedMidiBindings,
    ) -> PyResult<(String, f64, f64)> {
        self.0
            .note_on(channel, key, velocity, bindings.0.tuning(channel))
            .map(|note| (note.trigger_id, note.velocity, note.pitch_hz))
            .map_err(|error| PyValueError::new_err(error.to_string()))
    }

    fn note_off(&mut self, channel: u8, key: u8, oldest: bool) -> PyResult<String> {
        let policy = if oldest {
            RepeatedKeyRelease::Oldest
        } else {
            RepeatedKeyRelease::Newest
        };
        self.0
            .note_off(channel, key, policy)
            .map_err(|error| PyValueError::new_err(error.to_string()))
    }

    fn report_loss(&mut self) {
        self.0.report_loss();
    }

    fn restart(&mut self) {
        self.0.restart();
    }

    fn finish(&mut self) -> PyResult<()> {
        self.0
            .finish()
            .map_err(|error| PyValueError::new_err(error.to_string()))
    }
}

#[pymethods]
impl PreparedMidiBindings {
    #[new]
    fn new(score: &str, context: &str) -> PyResult<Self> {
        codec::parse_midi_bindings(score, context, Format::Json, PreparationLimits::default())
            .map(Self)
            .map_err(|error| PyValueError::new_err(error.to_string()))
    }

    fn parts(&self) -> Vec<&str> {
        self.0.parts().map(Identifier::as_str).collect()
    }

    fn note_route(&self, channel: u8) -> Option<(&str, bool)> {
        self.0
            .note_route(channel)
            .map(|(part, release)| (part.as_str(), release == RepeatedKeyRelease::Oldest))
    }

    fn control_change(
        &self,
        channel: u8,
        controller: u8,
        value: u8,
    ) -> PyResult<Option<(&str, &str, f64)>> {
        control_value(self.0.control_change(channel, controller, value))
    }

    fn pitch_bend(&self, channel: u8, value: i16) -> PyResult<Option<(&str, &str, f64)>> {
        control_value(self.0.pitch_bend(channel, value))
    }
}

fn control_value(
    result: Result<Option<ControlValue<'_>>, ufor::error::Error>,
) -> PyResult<Option<(&str, &str, f64)>> {
    result
        .map(|value| {
            value.map(|mapped| (mapped.part.as_str(), mapped.control.as_str(), mapped.value))
        })
        .map_err(|error| PyValueError::new_err(error.to_string()))
}

#[pyclass(skip_from_py_object)]
pub(crate) struct MidiFileClock(PpqnClock);

#[pymethods]
impl MidiFileClock {
    #[new]
    fn new(ticks_per_quarter: u16, sample_rate: i64) -> PyResult<Self> {
        PpqnClock::new(ticks_per_quarter, sample_rate)
            .map(Self)
            .map_err(|error| PyValueError::new_err(error.to_string()))
    }

    #[pyo3(signature = (tick, next_tempo=None))]
    fn advance(&mut self, tick: i64, next_tempo: Option<u32>) -> PyResult<i64> {
        self.0
            .advance(tick, next_tempo)
            .map_err(|error| PyValueError::new_err(error.to_string()))
    }
}

#[pyfunction]
pub(crate) fn integer_sample_rate(document: &str) -> PyResult<Option<i64>> {
    codec::parse_timebase(document, Format::Json)
        .map(|timebase| timebase.rate().integer_hz())
        .map_err(|error| PyValueError::new_err(error.to_string()))
}

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

#[pyfunction]
pub(crate) fn tuned_sample_ratios<'py>(
    py: Python<'py>,
    base: f64,
    tuning: PyReadonlyArray1<'py, f64>,
    layer: PyReadonlyArray1<'py, f64>,
) -> PyResult<Bound<'py, PyArray1<f64>>> {
    let tuning = tuning.as_array();
    let layer = layer.as_array();
    if tuning.len() != layer.len() {
        return Err(PyValueError::new_err("Mismatched sample tuning arrays"));
    }
    let ratios = tuning
        .iter()
        .zip(layer.iter())
        .map(|(&cents, &layer_cents)| tuned_pitch_ratio(base, cents, layer_cents))
        .collect::<Result<Vec<_>, _>>()
        .map_err(|error| PyValueError::new_err(error.to_string()))?;
    Ok(ratios.into_pyarray(py))
}

#[pyfunction]
pub(crate) fn sample_steps<'py>(
    py: Python<'py>,
    ratios: PyReadonlyArray1<'py, f64>,
    native_rate: f64,
) -> PyResult<Bound<'py, PyArray1<f64>>> {
    let steps = ratios
        .as_array()
        .iter()
        .map(|&ratio| sample_step(ratio, native_rate))
        .collect::<Result<Vec<_>, _>>()
        .map_err(|error| PyValueError::new_err(error.to_string()))?;
    Ok(steps.into_pyarray(py))
}

#[pyfunction]
pub(crate) fn prepare_sample_tuning_bounds(layers: Vec<(f64, String)>) -> PyResult<Vec<f64>> {
    layers
        .iter()
        .map(|(static_cents, document)| {
            let modulation = codec::parse_modulation(document, Format::Json)
                .map_err(|error| PyValueError::new_err(error.to_string()))?;
            Ok(playback::maximum_tuning_cents(
                *static_cents,
                modulation.parameters(),
            ))
        })
        .collect()
}

#[pyfunction]
pub(crate) fn maximum_sample_step(
    base: f64,
    native_rate: f64,
    common_cents: f64,
    layer_cents: f64,
) -> f64 {
    playback::maximum_sample_step(base, native_rate, common_cents, layer_cents)
}

#[pyclass(frozen, skip_from_py_object)]
pub(crate) struct PreparedOscillator(Oscillator);

#[pyfunction]
pub(crate) fn oscillator_phase(
    frame: &Bound<'_, PyAny>,
    frequency_hz: f64,
    sample_rate: i64,
) -> PyResult<f64> {
    let origin = if frame.is_instance_of::<PyInt>() {
        PhaseOrigin::Whole(frame.extract()?)
    } else {
        PhaseOrigin::Fractional(frame.extract()?)
    };
    phase_at_frame(origin, frequency_hz, sample_rate)
        .map_err(|error| PyValueError::new_err(error.to_string()))
}

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
pub(crate) struct PreparedVoiceEligibility(SynthInstrument);

#[pymethods]
impl PreparedVoiceEligibility {
    #[new]
    #[pyo3(signature = (document, *, max_records=10_000))]
    fn new(document: &str, max_records: usize) -> PyResult<Self> {
        PreparationLimits::new(max_records)
            .and_then(|limits| codec::parse_synth_instrument(document, Format::Json, limits))
            .map(Self)
            .map_err(|error| PyValueError::new_err(error.to_string()))
    }

    #[getter]
    fn sustain_pressed(&self) -> bool {
        self.0.sustain_pressed()
    }

    fn validate_event(&self, payload: &str) -> PyResult<()> {
        validate_event_payload(payload, self.0.controls())
    }

    fn matching_voices(&self, trigger: &str, key: i64, velocity: f64) -> PyResult<Vec<String>> {
        let trigger = serde_json::from_value(serde_json::Value::String(trigger.to_owned()))
            .map_err(|error| PyValueError::new_err(error.to_string()))?;
        Ok(self
            .0
            .voices()
            .matching_voices(trigger, key, velocity)
            .map(|name| name.as_str().to_owned())
            .collect())
    }
}
