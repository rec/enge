//! Deterministic bounded granular processing for the native effect backend.

use numpy::ndarray::Array2;
use numpy::{IntoPyArray, PyArray2, PyReadonlyArray2, PyUntypedArrayMethods};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use std::f64::consts::PI;

type RenderedGranulator<'py> = (
    Bound<'py, PyArray2<f64>>,
    Bound<'py, PyArray2<f64>>,
    i64,
    f64,
    u64,
    Vec<Bound<'py, PyArray2<f64>>>,
    Vec<usize>,
    Vec<f64>,
);

struct Grain {
    samples: Vec<Vec<f64>>,
    index: usize,
    gain: f64,
}

#[pyfunction]
#[allow(clippy::too_many_arguments)]
pub fn render_granulator<'py>(
    py: Python<'py>,
    samples: PyReadonlyArray2<'py, f64>,
    parameters: PyReadonlyArray2<'py, f64>,
    start_frame: usize,
    rate: usize,
    history_frames: usize,
    maximum_grains: usize,
    history: PyReadonlyArray2<'py, f64>,
    mut history_start: i64,
    mut phase: f64,
    mut counter: u64,
    grain_samples: Vec<PyReadonlyArray2<'py, f64>>,
    grain_indices: Vec<usize>,
    grain_gains: Vec<f64>,
) -> PyResult<RenderedGranulator<'py>> {
    let frames = samples.shape()[0];
    let channels = samples.shape()[1];
    if rate == 0
        || history_frames < 2
        || maximum_grains == 0
        || parameters.shape() != [frames, 6]
        || history.shape()[1] != channels
        || grain_samples.len() != grain_indices.len()
        || grain_samples.len() != grain_gains.len()
        || grain_samples.len() > maximum_grains
        || !phase.is_finite()
        || !(0.0..1.0).contains(&phase)
        || samples.as_array().iter().any(|v| !v.is_finite())
        || parameters.as_array().iter().any(|v| !v.is_finite())
    {
        return Err(PyValueError::new_err("Invalid native granulator buffers"));
    }
    let samples = samples.as_array().to_owned();
    let parameters = parameters.as_array().to_owned();
    let mut history = history
        .as_array()
        .rows()
        .into_iter()
        .map(|v| v.to_vec())
        .collect::<Vec<_>>();
    let mut grains = grain_samples
        .into_iter()
        .zip(grain_indices)
        .zip(grain_gains)
        .map(|((v, index), gain)| Grain {
            samples: v
                .as_array()
                .rows()
                .into_iter()
                .map(|row| row.to_vec())
                .collect(),
            index,
            gain,
        })
        .collect::<Vec<_>>();
    if history.len() > history_frames
        || grains.iter().any(|g| {
            g.samples.len() < 2
                || g.index >= g.samples.len()
                || g.samples.iter().any(|v| v.len() != channels)
                || !g.gain.is_finite()
        })
    {
        return Err(PyValueError::new_err("Invalid native granulator state"));
    }
    let (output, history, history_start, phase, counter, grains) =
        py.detach(move || -> PyResult<_> {
            let mut output = Array2::zeros((frames, channels));
            history.reserve(history_frames - history.len());
            for frame in 0..frames {
                history.push((0..channels).map(|c| samples[[frame, c]]).collect());
                if history.len() > history_frames {
                    history.remove(0);
                    history_start += 1;
                }
                let duration_seconds = parameters[[frame, 0]];
                let density = parameters[[frame, 1]];
                let lookback = parameters[[frame, 2]];
                let ratio = parameters[[frame, 3]];
                let jitter_seconds = parameters[[frame, 4]];
                let mix_value = parameters[[frame, 5]];
                if duration_seconds <= 0.0
                    || density <= 0.0
                    || density > rate as f64
                    || lookback < 0.0
                    || ratio <= 0.0
                    || jitter_seconds < 0.0
                    || !(0.0..=1.0).contains(&mix_value)
                {
                    return Err(PyValueError::new_err("Invalid native granulator parameter"));
                }
                phase += density / rate as f64;
                if phase >= 1.0 {
                    phase -= 1.0;
                    let launch_counter = counter;
                    counter += 1;
                    if grains.len() < maximum_grains {
                        let duration = (duration_seconds * rate as f64).round().max(2.0) as usize;
                        let end = (start_frame + frame) as f64 - lookback * rate as f64
                            + jitter_seconds * rate as f64 * jitter(launch_counter);
                        let start = end - (duration - 1) as f64 * ratio;
                        if let Some(captured) =
                            capture(&history, history_start, start, ratio, duration)
                        {
                            grains.push(Grain {
                                samples: captured,
                                index: 0,
                                gain: 1.0 / (density * duration_seconds).max(1.0).sqrt(),
                            });
                        }
                    }
                }
                let mut wet = vec![0.0; channels];
                for grain in &mut grains {
                    let window = 0.5
                        - 0.5 * (2.0 * PI * grain.index as f64 / grain.samples.len() as f64).cos();
                    for (channel, value) in wet.iter_mut().enumerate() {
                        *value += grain.samples[grain.index][channel] * window * grain.gain;
                    }
                    grain.index += 1;
                }
                grains.retain(|g| g.index < g.samples.len());
                for channel in 0..channels {
                    output[[frame, channel]] =
                        (1.0 - mix_value) * samples[[frame, channel]] + mix_value * wet[channel];
                }
            }
            Ok((output, history, history_start, phase, counter, grains))
        })?;
    let history = rows_array(history, channels)?;
    let mut rendered_grains = Vec::with_capacity(grains.len());
    let mut indices = Vec::with_capacity(grains.len());
    let mut gains = Vec::with_capacity(grains.len());
    for grain in grains {
        rendered_grains.push(rows_array(grain.samples, channels)?.into_pyarray(py));
        indices.push(grain.index);
        gains.push(grain.gain);
    }
    Ok((
        output.into_pyarray(py),
        history.into_pyarray(py),
        history_start,
        phase,
        counter,
        rendered_grains,
        indices,
        gains,
    ))
}

fn rows_array(values: Vec<Vec<f64>>, channels: usize) -> PyResult<Array2<f64>> {
    let frames = values.len();
    Array2::from_shape_vec((frames, channels), values.into_iter().flatten().collect())
        .map_err(|_| PyValueError::new_err("Invalid native granulator rows"))
}

fn capture(
    history: &[Vec<f64>],
    history_start: i64,
    start: f64,
    ratio: f64,
    frames: usize,
) -> Option<Vec<Vec<f64>>> {
    let mut result = Vec::with_capacity(frames);
    for i in 0..frames {
        let position = start + i as f64 * ratio;
        let lower = position.floor() as i64;
        let upper = lower + 1;
        if lower < history_start || upper >= history_start + history.len() as i64 {
            return None;
        }
        let fraction = position - lower as f64;
        result.push(
            history[(lower - history_start) as usize]
                .iter()
                .zip(&history[(upper - history_start) as usize])
                .map(|(a, b)| a + (b - a) * fraction)
                .collect(),
        );
    }
    Some(result)
}

fn jitter(counter: u64) -> f64 {
    let mut value = counter.wrapping_add(0x9E37_79B9_7F4A_7C15);
    value = (value ^ (value >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    value = (value ^ (value >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    value ^= value >> 31;
    2.0 * ((value >> 11) as f64 / 2_f64.powi(53)) - 1.0
}
