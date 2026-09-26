#![forbid(unsafe_code)]

mod effects;
mod filters;
mod fm;
mod granulator;
mod lfo;
mod live;
mod noise;
mod queue;
mod runtime;
mod sampler;

use numpy::ndarray::Array2;
use numpy::{
    IntoPyArray, PyArray2, PyArray3, PyReadonlyArray1, PyReadonlyArray2, PyUntypedArrayMethods,
};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use std::f64::consts::TAU;

type RenderedBlock<'py> = (
    Bound<'py, PyArray2<f64>>,
    Bound<'py, PyArray2<f64>>,
    Bound<'py, PyArray3<f64>>,
);

fn envelope_spans(envelope: PyReadonlyArray2<'_, f64>, count: usize) -> PyResult<Vec<f64>> {
    if envelope.shape()[1] != 7 || !envelope.is_c_contiguous() {
        return Err(PyValueError::new_err("Invalid native envelope layout"));
    }
    let spans = envelope.as_slice()?.to_vec();
    for span in spans.chunks_exact(7) {
        let (first, last) = (span[0], span[1]);
        if !first.is_finite()
            || !last.is_finite()
            || first < 0.0
            || last < first
            || last > count as f64
            || first != first.floor()
            || last != last.floor()
            || !span[5].is_finite()
            || span[5] <= 0.0
        {
            return Err(PyValueError::new_err("Invalid native envelope span"));
        }
    }
    Ok(spans)
}

fn envelope_amplitudes(spans: &[f64], count: usize) -> Vec<f64> {
    let mut amplitude = vec![0.0; count];
    for span in spans.chunks_exact(7) {
        for (i, value) in amplitude
            .iter_mut()
            .enumerate()
            .take(span[1] as usize)
            .skip(span[0] as usize)
        {
            let progress = span[4] + (i as f64 - span[6]) / span[5];
            *value = span[2] + span[3] * progress;
        }
    }
    amplitude
}

#[pyfunction]
#[pyo3(signature = (waveform, duty, rate, prepared_gain, frequencies, states, gains, envelope, routes, frames, filters=None))]
#[allow(clippy::too_many_arguments)]
fn render<'py>(
    py: Python<'py>,
    waveform: u8,
    duty: f64,
    rate: f64,
    prepared_gain: f64,
    frequencies: PyReadonlyArray2<'py, f64>,
    states: PyReadonlyArray2<'py, f64>,
    gains: PyReadonlyArray1<'py, f64>,
    envelope: PyReadonlyArray2<'py, f64>,
    routes: PyReadonlyArray2<'py, f64>,
    frames: usize,
    filters: Option<filters::Inputs<'py>>,
) -> PyResult<RenderedBlock<'py>> {
    let count = frequencies.shape()[0];
    let sources = frequencies.shape()[1];
    let channels = routes.shape()[1];
    if waveform > 2
        || !duty.is_finite()
        || !(0.0..=1.0).contains(&duty)
        || !rate.is_finite()
        || rate <= 0.0
        || states.shape() != [sources, 2]
        || gains.len() != count
        || envelope.shape()[1] != 7
        || routes.shape()[0] != sources
        || frames < count
        || !frequencies.is_c_contiguous()
        || !envelope.is_c_contiguous()
        || !routes.is_c_contiguous()
    {
        return Err(PyValueError::new_err(
            "Invalid native voice dimensions, layout, or oscillator settings",
        ));
    }
    let envelope = envelope_spans(envelope, count)?;
    // Own every working buffer before releasing the GIL. No borrowed Python
    // memory or Python callbacks cross into the numerical loop.
    let frequencies = frequencies.as_slice()?.to_vec();
    let gains = gains.as_slice()?.to_vec();
    let routes = routes.as_slice()?.to_vec();
    let states = states.as_array().to_owned();
    let mut filters = filters::FilterBank::new(filters, rate, count, sources)?;
    let (output, states, filter_states) = py.detach(move || -> PyResult<_> {
        let mut states = states;
        let mut output = Array2::zeros((frames, channels));
        let amplitude = envelope_amplitudes(&envelope, count);
        let mut waves = vec![0.0; sources];
        for i in 0..count {
            for s in 0..sources {
                let angle = TAU * (states[[s, 0]] / rate);
                // Retain the reference's angle-to-phase rounding at shape edges.
                let phase = (angle / TAU).rem_euclid(1.0);
                waves[s] = match waveform {
                    0 => angle.sin(),
                    1 => {
                        if phase < duty {
                            1.0
                        } else {
                            -1.0
                        }
                    }
                    _ if duty == 0.0 => 1.0 - 2.0 * phase,
                    _ if duty == 1.0 => 2.0 * phase - 1.0,
                    _ if phase < duty => 2.0 * phase / duty - 1.0,
                    _ => (1.0 + duty - 2.0 * phase) / (1.0 - duty),
                };
                let increment = frequencies[i * sources + s] - states[[s, 1]];
                let total = states[[s, 0]] + increment;
                states[[s, 1]] = (total - states[[s, 0]]) - increment;
                states[[s, 0]] = total.rem_euclid(rate);
            }
            filters.process(i, &mut waves)?;
            for c in 0..channels {
                let mut mixed = 0.0;
                for s in 0..sources {
                    mixed += waves[s] * routes[s * channels + c];
                }
                output[[i, c]] = mixed * ((amplitude[i] * prepared_gain) * gains[i]);
            }
        }
        Ok((output, states, filters.states))
    })?;
    Ok((
        output.into_pyarray(py),
        states.into_pyarray(py),
        filter_states.into_pyarray(py),
    ))
}

#[pymodule]
fn _native(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_function(wrap_pyfunction!(effects::render_effects, module)?)?;
    module.add_function(wrap_pyfunction!(filters::render_filters, module)?)?;
    module.add_function(wrap_pyfunction!(render, module)?)?;
    module.add_function(wrap_pyfunction!(fm::render_fm, module)?)?;
    module.add_function(wrap_pyfunction!(fm::render_four_operator_fm, module)?)?;
    module.add_function(wrap_pyfunction!(granulator::render_granulator, module)?)?;
    module.add_function(wrap_pyfunction!(noise::render_noise, module)?)?;
    module.add_class::<live::LiveRuntime>()?;
    module.add_class::<live::LiveRuntimeSnapshot>()?;
    module.add_class::<queue::ActionQueue>()?;
    module.add_class::<runtime::SynthRuntime>()?;
    module.add_class::<runtime::SynthRuntimeSnapshot>()?;
    module.add_class::<sampler::SampleBuffer>()?;
    module.add_function(wrap_pyfunction!(sampler::render_sample, module)?)?;
    module.add_function(wrap_pyfunction!(lfo::render_lfo, module)?)
}
