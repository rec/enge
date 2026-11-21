//! Sample traversal uses pending boundary choices, exactly as the reference does.
use numpy::ndarray::Array2;
use numpy::{
    IntoPyArray, PyArray2, PyArray3, PyReadonlyArray1, PyReadonlyArray2, PyUntypedArrayMethods,
};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use std::sync::Arc;
use ufor::codec::{self, Format};
use ufor::identifier::Identifier;
use ufor::samples::crossfade::Crossfade;
use ufor::samples::eligibility::Eligibility;
use ufor::samples::playback::{Direction, LoopMode, Mapping, ResolvedSlotPlayback};
use ufor::samples::variation::DrawEvent;

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
pub(crate) type Selection = (i64, i64, bool, Option<(i64, i64, i64, bool)>);
type RenderedSample<'py> = (Bound<'py, PyArray2<f64>>, State, Bound<'py, PyArray3<f64>>);

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
    selection: Selection,
    rate: f64,
    end_fade_frames: f64,
    initial_state: State,
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
        slice
            .validate_playback(playback, frames, additional_plays)
            .map_err(invalid)?;
        let end_fade_frames = end_fade_seconds * rate;
        if !rate.is_finite() || rate <= 0.0 || !end_fade_frames.is_finite() || end_fade_frames < 0.0
        {
            return Err(PyValueError::new_err("Invalid sample rate or end fade"));
        }
        let backward = playback.direction == Direction::Backward;
        let loop_settings = slice.loop_settings();
        let selection = (
            slice.start_frame(),
            slice.end_frame(),
            playback.direction == Direction::Mirror,
            loop_settings.map(|settings| {
                (
                    settings.start_frame(),
                    settings.end_frame(),
                    settings.crossfade_frames(),
                    settings.mode() == LoopMode::UntilRelease,
                )
            }),
        );
        let initial_state = (
            if backward {
                slice.end_frame() - 1
            } else {
                slice.start_frame()
            },
            0.0,
            0.0,
            if backward { -1 } else { 1 },
            loop_settings.is_some(),
            false,
            false,
            None,
            loop_settings.and_then(|settings| settings.repeat_count()),
            additional_plays,
            None,
        );
        Ok(Self {
            buffer: buffer.clone(),
            selection,
            rate,
            end_fade_frames,
            initial_state,
        })
    }
}

impl SampleTraversal {
    pub(crate) fn start(&self) -> Cursor {
        Cursor::new(
            self.selection,
            self.rate,
            self.initial_state,
            self.end_fade_frames,
        )
    }
}

#[derive(Clone, Copy)]
pub(crate) struct Cursor {
    pub(crate) selection: Selection,
    pub(crate) rate: f64,
    pub(crate) index: i64,
    pub(crate) position: f64,
    pub(crate) error: f64,
    pub(crate) direction: i64,
    pub(crate) looping: bool,
    pub(crate) overlap: bool,
    pub(crate) released: bool,
    pub(crate) exhausted: Option<i64>,
    pub(crate) repeats_remaining: Option<u64>,
    pub(crate) plays_remaining: u64,
    pub(crate) end_fade_at: Option<i64>,
    pub(crate) end_fade_frames: f64,
}

impl Cursor {
    pub(crate) fn new(selection: Selection, rate: f64, state: State, end_fade_frames: f64) -> Self {
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
        Self {
            selection,
            rate,
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
            end_fade_frames,
        }
    }

    fn remaining(&self) -> Option<f64> {
        let (start, end, mirror, loop_) = self.selection;
        let mut distance = if self.direction == 1 {
            end - self.index
        } else {
            self.index - start + 1
        } as f64;
        if let Some((a, b, overlap, _)) = loop_.filter(|_| self.looping || self.overlap) {
            let repeats = if self.looping {
                self.repeats_remaining?
            } else {
                0
            };
            let length = b - a;
            if mirror && repeats > 0 {
                let boundary = if self.direction == 1 { b - 1 } else { a };
                let direction = self.direction * if repeats % 2 == 0 { 1 } else { -1 };
                let last = if direction == -1 { b - 1 } else { a };
                let tail = if direction == 1 {
                    end - last
                } else {
                    last - start + 1
                };
                distance = (self.direction * (boundary - self.index)) as f64
                    + (repeats - 1) as f64 * (length - 1) as f64
                    + tail as f64;
            } else if !mirror {
                distance += (repeats as f64 + f64::from(self.overlap)) * (length - overlap) as f64;
            }
        } else if mirror && loop_.is_none() && self.direction == 1 {
            distance += (end - start - 1) as f64;
        }
        let length = end - start;
        distance += self.plays_remaining as f64
            * if mirror {
                2.0 * length as f64 - 1.0
            } else {
                length as f64
            };
        Some(distance - self.position / self.rate)
    }

    pub(crate) fn end_gain(&mut self, frame: i64, step: f64) -> f64 {
        if self.end_fade_frames == 0.0 {
            return 1.0;
        }
        if self.end_fade_at.is_none()
            && self
                .remaining()
                .is_some_and(|d| d * self.rate / step <= self.end_fade_frames)
        {
            self.end_fade_at = Some(frame);
        }
        let gain = self.end_fade_at.map_or(1.0, |at| {
            (1.0 - (frame - at) as f64 / self.end_fade_frames).max(0.0)
        });
        if gain == 0.0 {
            self.exhausted = Some(frame);
        }
        gain
    }

    pub(crate) fn release(&mut self) {
        self.released = true;
        if self.selection.3.is_some_and(|l| l.3) {
            self.looping = false;
        }
    }

    fn take_repeat(&mut self) -> bool {
        if let Some(remaining) = &mut self.repeats_remaining {
            if *remaining == 0 {
                self.looping = false;
                return false;
            }
            *remaining -= 1;
        }
        true
    }

    pub(crate) fn settle(&mut self, frame: i64) {
        if self.exhausted.is_some() {
            return;
        }
        let (start, end, mirror, loop_) = self.selection;
        if let Some((a, b, overlap, _)) = loop_ {
            if self.overlap {
                if self.direction == 1 && self.index == b {
                    self.index = a + overlap;
                    self.overlap = false;
                } else if self.direction == -1 && self.index == a - 1 {
                    self.index = b - 1 - overlap;
                    self.overlap = false;
                }
            } else if self.looping {
                if mirror {
                    if self.direction == 1 && self.index == b - 1 && self.take_repeat() {
                        self.direction = -1;
                    } else if self.direction == -1 && self.index == a && self.take_repeat() {
                        self.direction = 1;
                    }
                } else if overlap != 0 {
                    self.overlap = self.index
                        == if self.direction == 1 {
                            b - overlap
                        } else {
                            a + overlap - 1
                        }
                        && self.take_repeat();
                } else if self.direction == 1 && self.index == b && self.take_repeat() {
                    self.index = a;
                } else if self.direction == -1 && self.index == a - 1 && self.take_repeat() {
                    self.index = b - 1;
                }
            }
        } else if mirror && self.direction == 1 && self.index == end - 1 && end - start > 1 {
            self.direction = -1;
        }
        if !(start..end).contains(&self.index) {
            if self.plays_remaining > 0 {
                self.plays_remaining -= 1;
                self.index = if !mirror && self.direction == -1 {
                    end - 1
                } else {
                    start
                };
                if mirror {
                    self.direction = 1;
                }
                return;
            }
            self.exhausted = Some(frame);
            self.position = 0.0;
            self.error = 0.0;
        }
    }

    pub(crate) fn read(&self, audio: &Array2<f64>, channel: usize) -> f64 {
        let first = audio[[self.index as usize, channel]];
        if !self.overlap {
            return first;
        }
        let (a, b, overlap, _) = self.selection.3.expect("validated overlap");
        let offset = if self.direction == 1 {
            self.index - (b - overlap)
        } else {
            a + overlap - 1 - self.index
        };
        let incoming = if self.direction == 1 {
            a + offset
        } else {
            b - 1 - offset
        };
        let weight = offset as f64 / (overlap - 1) as f64;
        (1.0 - weight) * first + weight * audio[[incoming as usize, channel]]
    }

    fn boundary(&self) -> (i64, i64) {
        let (start, end, mirror, loop_) = self.selection;
        if let Some((a, b, overlap, _)) = loop_.filter(|_| self.looping || self.overlap) {
            let length = b - a;
            if mirror {
                return (
                    if self.direction == 1 { b - 1 } else { a },
                    if (a..b).contains(&self.index) {
                        2 * (length - 1)
                    } else {
                        0
                    },
                );
            }
            if self.direction == 1 {
                return (
                    b - if self.overlap { 0 } else { overlap },
                    if self.looping && (a + overlap..b).contains(&self.index) {
                        length - overlap
                    } else {
                        0
                    },
                );
            }
            return (
                a - 1 + if self.overlap { 0 } else { overlap },
                if self.looping && (a..b - overlap).contains(&self.index) {
                    length - overlap
                } else {
                    0
                },
            );
        }
        if mirror && loop_.is_none() && self.direction == 1 && end - start > 1 {
            return (end - 1, 0);
        }
        (if self.direction == 1 { end } else { start - 1 }, 0)
    }

    pub(crate) fn advance(&mut self, step: f64, frame: i64) {
        if self.exhausted.is_some() {
            return;
        }
        let increment = step - self.error;
        let total = self.position + increment;
        self.error = (total - self.position) - increment;
        // Python float divmod snaps the quotient after computing the remainder.
        // floor(total / rate) alone can disagree at a rounded integer boundary.
        let mut remainder = total % self.rate;
        let mut quotient = (total - remainder) / self.rate;
        if remainder < 0.0 {
            remainder += self.rate;
            quotient -= 1.0;
        }
        let mut count = quotient.floor();
        if quotient - count > 0.5 {
            count += 1.0;
        }
        self.position = remainder;
        // Keep small moves separate from a huge integer-valued float count.
        // This avoids both integer overflow and loss of entry-path steps above 2**53.
        let mut consumed = 0.0;
        let mut finite_count =
            (self.repeats_remaining.is_some() || self.plays_remaining > 0).then_some(count as u128);
        while finite_count.map_or(count > consumed, |c| c > 0) {
            if self.plays_remaining > 0 {
                let length = (self.selection.1 - self.selection.0) as u128;
                let period = if self.selection.2 {
                    2 * length - 1
                } else {
                    length
                };
                let left = finite_count.as_mut().expect("finite play count");
                let skipped = ((*left - 1) / period).min(self.plays_remaining as u128);
                self.plays_remaining -= skipped as u64;
                *left -= skipped * period;
            }
            let (boundary, period) = self.boundary();
            if period != 0 {
                if let Some(left) = &mut finite_count {
                    let traversals = if self.selection.2 { 2 } else { 1 };
                    let remaining = self.repeats_remaining.as_mut().expect("finite count");
                    let skipped =
                        ((*left - 1) / period as u128).min((*remaining / traversals) as u128);
                    *remaining -= skipped as u64 * traversals;
                    *left -= skipped * period as u128;
                    if skipped > 0 && *remaining == 0 {
                        self.looping = false;
                        continue;
                    }
                } else {
                    count = ((count % period as f64) - consumed).rem_euclid(period as f64);
                    if count == 0.0 {
                        count = period as f64;
                    }
                    consumed = 0.0;
                }
            }
            let distance = self.direction * (boundary - self.index);
            let moved = finite_count.map_or_else(
                || (count - consumed).min(distance as f64) as i64,
                |c| c.min(distance as u128) as i64,
            );
            self.index += self.direction * moved;
            consumed += moved as f64;
            if let Some(left) = &mut finite_count {
                *left -= moved as u128;
            }
            if finite_count.map_or(count > consumed, |c| c > 0) || self.position != 0.0 {
                self.settle(frame);
                if self.exhausted.is_some() {
                    return;
                }
            }
        }
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
    release: Option<(usize, bool, f64, f64)>,
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
    let SampleTraversal {
        buffer,
        selection,
        rate,
        end_fade_frames,
        ..
    } = traversal;
    let (selection, rate, end_fade_frames) = (*selection, *rate, *end_fade_frames);
    let (start, end, mirror, loop_) = selection;
    let (index, position, error, direction, looping, overlap, _, _, _, _, end_fade_at) = state;
    let valid_overlap = loop_.is_some_and(|(a, b, m, _)| {
        m >= 2
            && !mirror
            && if direction == 1 {
                (b - m..=b).contains(&index)
            } else {
                (a - 1..a + m).contains(&index)
            }
    });
    if end_fade_at.is_some_and(|at| at < 0 || at > frame)
        || (looping && loop_.is_none())
        || (overlap && !valid_overlap)
        || ![-1, 1].contains(&direction)
        || !(start - 1..=end).contains(&index)
        || !position.is_finite()
        || !(0.0..rate).contains(&position)
        || !error.is_finite()
        || frame < 0
        || i64::try_from(count)
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
        || release.is_some_and(|(i, _, a, b)| {
            i >= count || !a.is_finite() || a < 0.0 || !b.is_finite() || b < 0.0
        })
    {
        return Err(PyValueError::new_err(
            "Invalid native sample settings, state, or buffers",
        ));
    }
    let initial = Cursor::new(selection, rate, state, end_fade_frames);
    let mut settled = initial;
    settled.settle(frame);
    if settled.exhausted.is_none() && settled.direction * (settled.boundary().0 - settled.index) < 0
    {
        return Err(PyValueError::new_err(
            "Invalid native sample cursor direction",
        ));
    }
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
            if cursor.exhausted.is_some() {
                break;
            }
            let current = frame + i as i64;
            let split = release.filter(|r| r.0 == i);
            if split.is_some_and(|r| !r.1) {
                cursor.release();
            }
            cursor.settle(current);
            if cursor.exhausted.is_some() {
                break;
            }
            let end_gain = cursor.end_gain(current, step);
            if cursor.exhausted.is_some() {
                break;
            }
            let mut following = cursor;
            if cursor.position != 0.0 {
                following.index += following.direction;
                following.position = 0.0;
                following.settle(current);
            }
            for (c, source) in sources.iter_mut().enumerate() {
                let first = cursor.read(&audio, c);
                let last = if following.exhausted.is_some() {
                    first
                } else {
                    following.read(&audio, c)
                };
                *source = if cursor.position == 0.0 {
                    first
                } else {
                    first + (cursor.position / rate) * (last - first)
                };
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
            if let Some((_, _, before, after)) = split.filter(|r| r.1) {
                cursor.advance(before, current + 1);
                if cursor.exhausted.is_none() {
                    cursor.release();
                    cursor.settle(current + 1);
                    cursor.advance(after, current + 1);
                }
            } else {
                cursor.advance(step, current + 1);
            }
        }
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
