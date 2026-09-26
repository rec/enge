use crate::{envelope_amplitudes, envelope_spans, filters};
use numpy::ndarray::Array2;
use numpy::{
    IntoPyArray, PyArray1, PyArray2, PyArray3, PyReadonlyArray1, PyReadonlyArray2,
    PyUntypedArrayMethods,
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

#[pyfunction]
pub fn render_four_operator_fm<'py>(
    py: Python<'py>,
    rate: f64,
    frequencies: PyReadonlyArray2<'py, f64>,
    envelopes: PyReadonlyArray2<'py, f64>,
    indices: PyReadonlyArray2<'py, f64>,
    edges: PyReadonlyArray2<'py, f64>,
    carrier: usize,
    levels: PyReadonlyArray1<'py, f64>,
    phases: PyReadonlyArray2<'py, f64>,
    history: PyReadonlyArray1<'py, f64>,
) -> PyResult<(
    Bound<'py, PyArray2<f64>>,
    Bound<'py, PyArray2<f64>>,
    Bound<'py, PyArray1<f64>>,
)> {
    let count = frequencies.shape()[0];
    if !rate.is_finite()
        || rate <= 0.0
        || carrier >= 4
        || frequencies.shape() != [count, 4]
        || envelopes.shape() != [count, 4]
        || indices.shape() != [count, edges.shape()[0]]
        || edges.shape()[1] != 3
        || levels.len() != count
        || phases.shape() != [4, 2]
        || history.len() != edges.shape()[0]
    {
        return Err(PyValueError::new_err("Invalid four-operator FM layout"));
    }
    let frequencies = frequencies.as_array().to_owned();
    let envelopes = envelopes.as_array().to_owned();
    let indices = indices.as_array().to_owned();
    let edges = edges.as_array().to_owned();
    let levels: Vec<f64> = levels.as_array().iter().copied().collect();
    let mut phases = phases.as_array().to_owned();
    let mut history: Vec<f64> = history.as_array().iter().copied().collect();
    let (audio, phases, history) = py.detach(move || -> PyResult<_> {
        let mut order = Vec::new();
        let mut pending = [0usize; 4];
        for edge in edges.rows() {
            if edge[0] < 0.0
                || edge[0] >= 4.0
                || edge[1] < 0.0
                || edge[1] >= 4.0
                || edge[2] < 0.0
                || edge[2] > 1.0
            {
                return Err(PyValueError::new_err("Invalid FM edge"));
            }
            if edge[2] == 0.0 {
                pending[edge[1] as usize] += 1;
            }
        }
        let mut ready: Vec<usize> = (0..4).filter(|i| pending[*i] == 0).collect();
        while let Some(operator) = ready.pop() {
            order.push(operator);
            for edge in edges.rows() {
                if edge[2] == 0.0 && edge[0] as usize == operator {
                    let destination = edge[1] as usize;
                    pending[destination] -= 1;
                    if pending[destination] == 0 {
                        ready.push(destination);
                    }
                }
            }
        }
        if order.len() != 4 {
            return Err(PyValueError::new_err(
                "Four-operator FM current edges must be acyclic",
            ));
        }
        let mut audio = Array2::zeros((count, 1));
        for i in 0..count {
            let mut output = [0.0; 4];
            for operator in &order {
                let mut offset = 0.0;
                for (edge_index, edge) in edges.rows().into_iter().enumerate() {
                    if edge[1] as usize == *operator {
                        offset += indices[[i, edge_index]]
                            * if edge[2] == 0.0 {
                                output[edge[0] as usize]
                            } else {
                                history[edge_index]
                            };
                    }
                }
                output[*operator] = envelopes[[i, *operator]]
                    * (TAU * phases[[*operator, 0]] / rate + offset).sin();
            }
            audio[[i, 0]] = levels[i] * output[carrier];
            for operator in 0..4 {
                let increment = frequencies[[i, operator]] - phases[[operator, 1]];
                let total = phases[[operator, 0]] + increment;
                phases[[operator, 1]] = (total - phases[[operator, 0]]) - increment;
                phases[[operator, 0]] = total.rem_euclid(rate);
            }
            history = edges
                .rows()
                .into_iter()
                .map(|edge| output[edge[0] as usize])
                .collect();
        }
        Ok((audio, phases, history))
    })?;
    Ok((
        audio.into_pyarray(py),
        phases.into_pyarray(py),
        history.into_pyarray(py),
    ))
}
