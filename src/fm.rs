use crate::{envelope_amplitudes, envelope_spans, filters};
use numpy::ndarray::Array2;
use numpy::{
    IntoPyArray, PyArray2, PyArray3, PyReadonlyArray1, PyReadonlyArray2, PyUntypedArrayMethods,
};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use std::f64::consts::TAU;

type RenderedFM<'py> = (
    Bound<'py, PyArray2<f64>>,
    Bound<'py, PyArray2<f64>>,
    f64,
    Bound<'py, PyArray3<f64>>,
);

#[pyfunction]
#[allow(clippy::too_many_arguments)]
pub fn render_fm<'py>(
    py: Python<'py>,
    rate: f64,
    parameters: PyReadonlyArray2<'py, f64>,
    phases: PyReadonlyArray2<'py, f64>,
    history: f64,
    gains: PyReadonlyArray1<'py, f64>,
    modulator: PyReadonlyArray2<'py, f64>,
    carrier: PyReadonlyArray2<'py, f64>,
    modulator_frames: usize,
    routes: PyReadonlyArray1<'py, f64>,
    filter_inputs: filters::Inputs<'py>,
) -> PyResult<RenderedFM<'py>> {
    let count = parameters.shape()[0];
    if !rate.is_finite()
        || rate <= 0.0
        || parameters.shape()[1] != 5
        || phases.shape() != [2, 2]
        || gains.len() != count
        || routes.is_empty()
        || modulator_frames > count
        || !history.is_finite()
    {
        return Err(PyValueError::new_err(
            "Invalid FM dimensions, rate, or history",
        ));
    }
    let parameters = parameters.as_array().to_owned();
    let mut phases = phases.as_array().to_owned();
    let gains: Vec<f64> = gains.as_array().iter().copied().collect();
    let routes: Vec<f64> = routes.as_array().iter().copied().collect();
    if parameters
        .indexed_iter()
        .any(|((_, j), x)| !x.is_finite() || if j < 2 { *x <= 0.0 } else { *x < 0.0 })
        || phases.iter().any(|x| !x.is_finite())
        || gains.iter().any(|x| !x.is_finite() || *x < 0.0)
        || routes.iter().any(|x| !x.is_finite())
    {
        return Err(PyValueError::new_err("Invalid FM parameters or state"));
    }
    let modulator = envelope_spans(modulator, count)?;
    let carrier = envelope_spans(carrier, count)?;
    let mut filters = filters::FilterBank::new(Some(filter_inputs), rate, count, 1)?;
    let (audio, phases, history, memory) = py.detach(move || -> PyResult<_> {
        let modulator = envelope_amplitudes(&modulator, count);
        let carrier = envelope_amplitudes(&carrier, count);
        let mut previous = history;
        let mut audio = Array2::zeros((count, routes.len()));
        for i in 0..count {
            let amplitude = if i < modulator_frames {
                modulator[i]
            } else {
                0.0
            };
            let value =
                amplitude * (TAU * phases[[0, 0]] / rate + parameters[[i, 3]] * previous).sin();
            let sample = parameters[[i, 4]]
                * carrier[i]
                * (TAU * phases[[1, 0]] / rate + parameters[[i, 2]] * value).sin();
            let mut source = [sample];
            for j in 0..2 {
                let increment = parameters[[i, j]] - phases[[j, 1]];
                let total = phases[[j, 0]] + increment;
                phases[[j, 1]] = (total - phases[[j, 0]]) - increment;
                phases[[j, 0]] = total.rem_euclid(rate);
            }
            previous = value;
            filters.process(i, &mut source)?;
            for (j, route) in routes.iter().enumerate() {
                audio[[i, j]] = (source[0] * gains[i]) * route;
            }
        }
        if audio.iter().chain(phases.iter()).any(|x| !x.is_finite()) || !previous.is_finite() {
            return Err(PyValueError::new_err("Non-finite FM output or state"));
        }
        Ok((audio, phases, previous, filters.states))
    })?;
    Ok((
        audio.into_pyarray(py),
        phases.into_pyarray(py),
        history,
        memory.into_pyarray(py),
    ))
}
