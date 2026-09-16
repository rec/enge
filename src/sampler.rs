//! Sample traversal uses pending boundary choices, exactly as the reference does.
use numpy::ndarray::Array2;
use numpy::{
    IntoPyArray, PyArray2, PyArray3, PyReadonlyArray1, PyReadonlyArray2, PyUntypedArrayMethods,
};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use std::sync::Arc;

// Integers stay integers across the binding, including frames beyond 2**53.
type State = (i64, f64, f64, i64, bool, bool, bool, Option<i64>);
type Selection = (i64, i64, bool, Option<(i64, i64, i64, bool)>);
type RenderedSample<'py> = (Bound<'py, PyArray2<f64>>, State, Bound<'py, PyArray3<f64>>);

#[pyclass(frozen)]
pub struct SampleBuffer {
    audio: Arc<Array2<f64>>,
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

#[derive(Clone, Copy)]
struct Cursor {
    selection: Selection,
    rate: f64,
    index: i64,
    position: f64,
    error: f64,
    direction: i64,
    looping: bool,
    overlap: bool,
    released: bool,
    exhausted: Option<i64>,
}

impl Cursor {
    fn release(&mut self) {
        self.released = true;
        if self.selection.3.is_some_and(|l| l.3) {
            self.looping = false;
        }
    }

    fn settle(&mut self, frame: i64) {
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
                    if self.direction == 1 && self.index == b - 1 {
                        self.direction = -1;
                    } else if self.direction == -1 && self.index == a {
                        self.direction = 1;
                    }
                } else if overlap != 0 {
                    self.overlap = self.index
                        == if self.direction == 1 {
                            b - overlap
                        } else {
                            a + overlap - 1
                        };
                } else if self.direction == 1 && self.index == b {
                    self.index = a;
                } else if self.direction == -1 && self.index == a - 1 {
                    self.index = b - 1;
                }
            }
        } else if mirror && self.direction == 1 && self.index == end - 1 && end - start > 1 {
            self.direction = -1;
        }
        if !(start..end).contains(&self.index) {
            self.exhausted = Some(frame);
            self.position = 0.0;
            self.error = 0.0;
        }
    }

    fn read(&self, audio: &Array2<f64>, channel: usize) -> f64 {
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

    fn advance(&mut self, step: f64, frame: i64) {
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
        while count > consumed {
            let (boundary, period) = self.boundary();
            if period != 0 {
                count = ((count % period as f64) - consumed).rem_euclid(period as f64);
                if count == 0.0 {
                    count = period as f64;
                }
                consumed = 0.0;
            }
            let distance = self.direction * (boundary - self.index);
            let moved = (count - consumed).min(distance as f64) as i64;
            self.index += self.direction * moved;
            consumed += moved as f64;
            if count > consumed || self.position != 0.0 {
                self.settle(frame);
                if self.exhausted.is_some() {
                    return;
                }
            }
        }
    }
}

#[pyfunction]
#[pyo3(signature = (buffer, selection, state, frame, rate, steps, release, prepared_gain, gains, envelope, routes, frames, filters=None))]
#[allow(clippy::too_many_arguments)]
pub fn render_sample<'py>(
    py: Python<'py>,
    buffer: &SampleBuffer,
    selection: Selection,
    state: State,
    frame: i64,
    rate: f64,
    steps: PyReadonlyArray1<'py, f64>,
    release: Option<(usize, bool, f64, f64)>,
    prepared_gain: f64,
    gains: PyReadonlyArray1<'py, f64>,
    envelope: PyReadonlyArray2<'py, f64>,
    routes: PyReadonlyArray2<'py, f64>,
    frames: usize,
    filters: Option<super::filters::Inputs<'py>>,
) -> PyResult<RenderedSample<'py>> {
    let count = steps.len();
    let channels = routes.shape()[1];
    let (start, end, mirror, loop_) = selection;
    let (index, position, error, direction, looping, overlap, released, exhausted) = state;
    if start < 0 || end <= start || end as u64 > buffer.audio.nrows() as u64 {
        return Err(PyValueError::new_err("Invalid native sample slice"));
    }
    let invalid_loop = loop_.is_some_and(|(a, b, m, _)| {
        a < start
            || a >= end
            || b <= a
            || b > end
            || b - a < 2
            || (m != 0 && (m < 2 || m >= (b - a + 1) / 2 || mirror))
    });
    let valid_overlap = !invalid_loop
        && loop_.is_some_and(|(a, b, m, _)| {
            m >= 2
                && !mirror
                && if direction == 1 {
                    (b - m..=b).contains(&index)
                } else {
                    (a - 1..a + m).contains(&index)
                }
        });
    if invalid_loop
        || (looping && loop_.is_none())
        || (overlap && !valid_overlap)
        || ![-1, 1].contains(&direction)
        || !(start - 1..=end).contains(&index)
        || !rate.is_finite()
        || rate <= 0.0
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
    let initial = Cursor {
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
    };
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
            filters.process(i, &mut sources)?;
            for c in 0..channels {
                let mut mixed = 0.0;
                for (s, source) in sources.iter().enumerate() {
                    mixed += source * routes[s * channels + c];
                }
                output[[i, c]] = mixed * ((amplitude[i] * prepared_gain) * gains[i]);
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
