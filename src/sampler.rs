//! Sample traversal uses pending boundary choices, exactly as the reference does.
use numpy::ndarray::Array2;
use numpy::{
    IntoPyArray, PyArray2, PyArray3, PyReadonlyArray1, PyReadonlyArray2, PyUntypedArrayMethods,
};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use std::{collections::BTreeMap, sync::Arc};
use ufor::codec::{self, Format};
use ufor::envelfo::PreparationLimits;
use ufor::identifier::Identifier;
use ufor::number::Rational;
use ufor::samples::crossfade::Crossfade;
use ufor::samples::cursor::{Cursor, CursorState};
use ufor::samples::eligibility::Eligibility;
use ufor::samples::inheritance::resolve_slot_settings;
use ufor::samples::playback::{Mapping, ResolvedSlotPlayback};
use ufor::samples::variation::DrawEvent;

#[pyfunction]
#[pyo3(signature = (document, decoded_shapes, *, max_records=10_000))]
pub(crate) fn prepare_sample_assets(
    document: &str,
    decoded_shapes: BTreeMap<String, Vec<usize>>,
    max_records: usize,
) -> PyResult<(i64, BTreeMap<String, i64>)> {
    let invalid = |error: ufor::error::Error| PyValueError::new_err(error.to_string());
    let limits = PreparationLimits::new(max_records).map_err(invalid)?;
    codec::parse_sample_inventory(document, Format::Json, limits)
        .and_then(|inventory| inventory.integer_rates(&decoded_shapes))
        .map_err(invalid)
}

#[pyfunction]
#[pyo3(signature = (slot, group=None))]
pub(crate) fn resolve_sample_slot_settings(slot: &str, group: Option<&str>) -> PyResult<String> {
    let resolved = resolve_slot_settings(slot, group)
        .map_err(|error| PyValueError::new_err(error.to_string()))?;
    serde_json::to_string(&resolved).map_err(|error| PyValueError::new_err(error.to_string()))
}

#[pyfunction]
#[pyo3(signature = (seed, part, trigger_id, tick, ordinal))]
pub(crate) fn draw_sample_range(
    seed: u64,
    part: String,
    trigger_id: Option<String>,
    tick: i64,
    ordinal: u64,
) -> PyResult<f64> {
    let event = sample_draw_event(seed, part, trigger_id, tick, ordinal)?;
    Ok(event.random_range_value())
}

#[pyfunction]
#[pyo3(signature = (document, seed, part, trigger_id, tick, ordinal, identity))]
pub(crate) fn resolve_sample_variation(
    document: &str,
    seed: u64,
    part: String,
    trigger_id: Option<String>,
    tick: i64,
    ordinal: u64,
    identity: String,
) -> PyResult<String> {
    let event = sample_draw_event(seed, part, trigger_id, tick, ordinal)?;
    let invalid = |error: ufor::error::Error| PyValueError::new_err(error.to_string());
    let variation = codec::parse_sample_variation(document, Format::Json).map_err(invalid)?;
    let identity = Identifier::new(identity).map_err(invalid)?;
    let variation = variation.resolve(&event, &identity).map_err(invalid)?;
    serde_json::to_string(&variation).map_err(|error| PyValueError::new_err(error.to_string()))
}

fn sample_draw_event(
    seed: u64,
    part: String,
    trigger_id: Option<String>,
    tick: i64,
    ordinal: u64,
) -> PyResult<DrawEvent> {
    let invalid = |error: ufor::error::Error| PyValueError::new_err(error.to_string());
    Ok(DrawEvent {
        seed,
        part: Identifier::new(part).map_err(invalid)?,
        trigger_id: trigger_id
            .map(Identifier::new)
            .transpose()
            .map_err(invalid)?,
        tick,
        ordinal,
    })
}

#[pyfunction]
pub(crate) fn advance_sample_selection(
    mode: &str,
    seed: u64,
    document: &str,
) -> PyResult<(String, String)> {
    let mode = serde_json::from_value(serde_json::Value::String(mode.to_owned()))
        .map_err(|error| PyValueError::new_err(error.to_string()))?;
    let invalid = |error: ufor::error::Error| PyValueError::new_err(error.to_string());
    let sequence =
        codec::parse_sample_selection_sequence(document, Format::Json).map_err(invalid)?;
    let (choice, next) = sequence.advance(mode, seed).map_err(invalid)?;
    let next =
        serde_json::to_string(&next).map_err(|error| PyValueError::new_err(error.to_string()))?;
    Ok((choice.into(), next))
}

// Integers stay integers across the binding, including frames beyond 2**53.
pub(crate) type State = (
    i64,
    f64,
    f64,
    i64,
    bool,
    bool,
    bool,
    Option<i64>,
    Option<u64>,
    u64,
    Option<i64>,
);

type RenderedSample<'py> = (Bound<'py, PyArray2<f64>>, State, Bound<'py, PyArray3<f64>>);

fn restore_cursor(cursor: Cursor, state: State, frame: i64) -> PyResult<Cursor> {
    let (
        index,
        position,
        error,
        direction,
        looping,
        overlap,
        released,
        exhausted,
        repeats_remaining,
        plays_remaining,
        end_fade_at,
    ) = state;
    cursor
        .resume(
            CursorState {
                index,
                position,
                error,
                direction,
                looping,
                overlap,
                released,
                exhausted,
                repeats_remaining,
                plays_remaining,
                end_fade_at,
            },
            frame,
        )
        .map_err(|error| PyValueError::new_err(error.to_string()))
}

#[pyclass(frozen, skip_from_py_object)]
#[derive(Clone)]
pub struct SampleBuffer {
    pub(crate) audio: Arc<Array2<f64>>,
}

#[pymethods]
impl SampleBuffer {
    #[new]
    fn new(samples: PyReadonlyArray2<'_, f64>) -> PyResult<Self> {
        if samples.shape().contains(&0) || samples.as_array().iter().any(|v| !v.is_finite()) {
            return Err(PyValueError::new_err("Invalid native sample audio"));
        }
        Ok(Self {
            audio: Arc::new(samples.as_array().to_owned()),
        })
    }
}

#[pyclass(frozen)]
pub(crate) struct SampleEligibility {
    slots: Eligibility,
}

#[pymethods]
impl SampleEligibility {
    #[new]
    #[pyo3(signature = (document, *, max_records=10_000))]
    fn new(document: &str, max_records: usize) -> PyResult<Self> {
        let slots = codec::parse_sample_eligibility(document, Format::Json, max_records)
            .map_err(|error| PyValueError::new_err(error.to_string()))?;
        Ok(Self { slots })
    }

    fn matching_slots(&self, document: &str) -> PyResult<Vec<String>> {
        let invalid = |error: ufor::error::Error| PyValueError::new_err(error.to_string());
        let context =
            codec::parse_sample_eligibility_context(document, Format::Json).map_err(invalid)?;
        Ok(self
            .slots
            .matching_slots(&context)
            .map_err(invalid)?
            .into_iter()
            .map(|name| name.as_str().to_owned())
            .collect())
    }
}

#[pyclass(frozen)]
pub(crate) struct SampleCrossfades {
    latched: Vec<Crossfade>,
    controls: Vec<Crossfade>,
}

#[pymethods]
impl SampleCrossfades {
    #[new]
    fn new(documents: Vec<String>) -> PyResult<Self> {
        let fades = documents
            .iter()
            .map(|text| {
                codec::parse_sample_crossfade(text, Format::Json)
                    .map_err(|error| PyValueError::new_err(error.to_string()))
            })
            .collect::<PyResult<Vec<_>>>()?;
        let (controls, latched) = fades
            .into_iter()
            .partition(|fade| fade.control_route().is_some());
        Ok(Self { latched, controls })
    }

    #[pyo3(signature = (key, velocity))]
    fn onset_gain(&self, key: Option<i64>, velocity: f64) -> PyResult<f64> {
        self.latched.iter().try_fold(1.0, |gain, fade| {
            let weight = fade
                .onset_weight(key, velocity)
                .map_err(|error| PyValueError::new_err(error.to_string()))?;
            Ok(gain * weight.expect("latched crossfade"))
        })
    }

    fn control_routes(&self) -> Vec<(f64, f64, bool, bool)> {
        self.controls
            .iter()
            .filter_map(Crossfade::control_route)
            .collect()
    }

    fn control_weights<'py>(
        &self,
        py: Python<'py>,
        index: usize,
        values: PyReadonlyArray1<'py, f64>,
    ) -> PyResult<Bound<'py, numpy::PyArray1<f64>>> {
        let fade = self
            .controls
            .get(index)
            .ok_or_else(|| PyValueError::new_err("Unknown control crossfade"))?;
        let weights = values
            .as_array()
            .iter()
            .map(|&value| {
                fade.control_weight(value)
                    .map_err(|error| PyValueError::new_err(error.to_string()))
            })
            .collect::<PyResult<Vec<_>>>()?;
        Ok(weights.into_pyarray(py))
    }
}

#[pyclass(frozen)]
pub(crate) struct PreparedSlotPlayback(ResolvedSlotPlayback);

#[pymethods]
impl PreparedSlotPlayback {
    #[new]
    fn new(document: &str, defaults: &str) -> PyResult<Self> {
        let invalid = |error: ufor::error::Error| PyValueError::new_err(error.to_string());
        let defaults = codec::parse_sample_playback(defaults, Format::Json).map_err(invalid)?;
        let slot = codec::parse_slot_playback(document, Format::Json).map_err(invalid)?;
        slot.resolve(defaults).map(Self).map_err(invalid)
    }

    fn resolved(&self) -> PyResult<(String, u64, f64)> {
        let playback = serde_json::to_string(&self.0.playback)
            .map_err(|error| PyValueError::new_err(error.to_string()))?;
        Ok((playback, self.0.additional_plays, self.0.end_fade_seconds))
    }

    fn delay_frames(&self, variation: &str, sample_rate: i64) -> PyResult<i64> {
        let invalid = |error: ufor::error::Error| PyValueError::new_err(error.to_string());
        let variation = ufor::quantity::exact(variation, "seconds").map_err(invalid)?;
        self.0.delay_frames(variation, sample_rate).map_err(invalid)
    }
}

#[pyclass(frozen)]
pub(crate) struct SamplePitchMapping(Mapping);

#[pymethods]
impl SamplePitchMapping {
    #[new]
    fn new(document: &str) -> PyResult<Self> {
        codec::parse_sample_mapping(document, Format::Json)
            .map(Self)
            .map_err(|error| PyValueError::new_err(error.to_string()))
    }

    #[pyo3(signature = (pitch_hz, tuning_cents, variation_cents))]
    fn pitch_ratio(
        &self,
        pitch_hz: Option<f64>,
        tuning_cents: f64,
        variation_cents: f64,
    ) -> PyResult<f64> {
        self.0
            .pitch_ratio(pitch_hz, tuning_cents, variation_cents)
            .map_err(|error| PyValueError::new_err(error.to_string()))
    }
}

#[pyclass(frozen)]
pub(crate) struct SampleTraversal {
    pub(crate) buffer: SampleBuffer,
    cursor: Cursor,
}

#[pymethods]
impl SampleTraversal {
    #[new]
    #[pyo3(signature = (buffer, slice, playback, rate, additional_plays=0, end_fade_seconds=0.0))]
    fn new(
        buffer: &SampleBuffer,
        slice: &str,
        playback: &str,
        rate: f64,
        additional_plays: u64,
        end_fade_seconds: f64,
    ) -> PyResult<Self> {
        let invalid = |error: ufor::error::Error| PyValueError::new_err(error.to_string());
        let slice = codec::parse_sample_slice(slice, Format::Json).map_err(invalid)?;
        let playback = codec::parse_sample_playback(playback, Format::Json).map_err(invalid)?;
        let frames = i64::try_from(buffer.audio.nrows())
            .map_err(|_| PyValueError::new_err("Decoded sample frame count exceeds i64"))?;
        let cursor = Cursor::new(
            &slice,
            playback,
            frames,
            rate,
            additional_plays,
            end_fade_seconds,
        )
        .map_err(invalid)?;
        Ok(Self {
            buffer: buffer.clone(),
            cursor,
        })
    }

    fn active_prefix(
        &self,
        state: State,
        frame: i64,
        release_now: bool,
        maximum_step: f64,
        requested: usize,
    ) -> PyResult<usize> {
        Ok(restore_cursor(self.cursor, state, frame)?.active_prefix(
            frame,
            release_now,
            maximum_step,
            requested,
        ))
    }
}

impl SampleTraversal {
    pub(crate) fn start(&self) -> Cursor {
        self.cursor
    }
}

#[pyfunction]
#[pyo3(signature = (traversal, state, frame, steps, release, prepared_gain, gains, envelope, routes, frames, filters=None, filter_after_amplitude=false))]
#[allow(clippy::too_many_arguments)]
pub(crate) fn render_sample<'py>(
    py: Python<'py>,
    traversal: &SampleTraversal,
    state: State,
    frame: i64,
    steps: PyReadonlyArray1<'py, f64>,
    release: Option<(usize, i64, i64)>,
    prepared_gain: f64,
    gains: PyReadonlyArray1<'py, f64>,
    envelope: PyReadonlyArray2<'py, f64>,
    routes: PyReadonlyArray2<'py, f64>,
    frames: usize,
    filters: Option<super::filters::Inputs<'py>>,
    filter_after_amplitude: bool,
) -> PyResult<RenderedSample<'py>> {
    let count = steps.len();
    let channels = routes.shape()[1];
    let SampleTraversal { buffer, cursor } = traversal;
    let rate = cursor.rate();
    if i64::try_from(count)
        .ok()
        .and_then(|n| frame.checked_add(n))
        .is_none()
        || steps.as_array().iter().any(|s| !s.is_finite() || *s <= 0.0)
        || frames < count
        || gains.len() != count
        || routes.shape()[0] != buffer.audio.ncols()
        || channels == 0
        || !steps.is_c_contiguous()
        || !gains.is_c_contiguous()
        || !routes.is_c_contiguous()
        || release.is_some_and(|(i, numerator, denominator)| {
            i >= count || numerator < 0 || numerator >= denominator
        })
    {
        return Err(PyValueError::new_err(
            "Invalid native sample settings, state, or buffers",
        ));
    }
    let initial = restore_cursor(*cursor, state, frame)?;
    let release = release
        .map(|(index, n, d)| Rational::new(n, d).map(|fraction| (index, fraction)))
        .transpose()
        .map_err(|error| PyValueError::new_err(error.to_string()))?;
    let envelope = super::envelope_spans(envelope, count)?;
    let steps = steps.as_slice()?.to_vec();
    let gains = gains.as_slice()?.to_vec();
    let routes = routes.as_slice()?.to_vec();
    let audio = Arc::clone(&buffer.audio);
    let mut filters = super::filters::FilterBank::new(filters, rate, count, audio.ncols())?;
    let (output, state, filter_states) = py.detach(move || -> PyResult<_> {
        let amplitude = super::envelope_amplitudes(&envelope, count);
        let mut cursor = initial;
        let mut output = Array2::zeros((frames, channels));
        let mut sources = vec![0.0; audio.ncols()];
        for (i, step) in steps.into_iter().enumerate() {
            if cursor.state().exhausted.is_some() {
                break;
            }
            let current = frame + i as i64;
            let fraction = release
                .filter(|(index, _)| *index == i)
                .map(|(_, value)| value);
            if fraction.is_some_and(|value| value.numerator() == 0) {
                cursor.release();
            }
            cursor.settle(current);
            if cursor.state().exhausted.is_some() {
                break;
            }
            let end_gain = cursor.end_gain(current, step);
            let Some(read) = cursor.read_plan(current) else {
                break;
            };
            for (c, source) in sources.iter_mut().enumerate() {
                *source = read.sample(|index| audio[[index as usize, c]]);
            }
            let gain = (amplitude[i] * prepared_gain) * gains[i];
            if filter_after_amplitude {
                for source in &mut sources {
                    *source *= gain;
                }
            }
            filters.process(i, &mut sources)?;
            for c in 0..channels {
                let mut mixed = 0.0;
                for (s, source) in sources.iter().enumerate() {
                    mixed += source * routes[s * channels + c];
                }
                output[[i, c]] = end_gain
                    * if filter_after_amplitude {
                        mixed
                    } else {
                        mixed * gain
                    };
            }
            let invalid = |_| PyValueError::new_err("Invalid native sample advance");
            if let Some(fraction) = fraction.filter(|value| value.numerator() != 0) {
                cursor
                    .advance_with_release(step, fraction, current + 1)
                    .map_err(invalid)?;
            } else {
                cursor.advance(step, current + 1).map_err(invalid)?;
            }
        }
        let cursor = cursor.state();
        Ok((
            output,
            (
                cursor.index,
                cursor.position,
                cursor.error,
                cursor.direction,
                cursor.looping,
                cursor.overlap,
                cursor.released,
                cursor.exhausted,
                cursor.repeats_remaining,
                cursor.plays_remaining,
                cursor.end_fade_at,
            ),
            filters.states,
        ))
    })?;
    Ok((
        output.into_pyarray(py),
        state,
        filter_states.into_pyarray(py),
    ))
}
