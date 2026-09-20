//! Whole-graph native gain, multiply, and resonant-filter processing.

use crate::filters::{FilterBank, Inputs};
use numpy::ndarray::{Array2, Array3};
use numpy::{
    IntoPyArray, PyArray2, PyArray3, PyReadonlyArray2, PyReadonlyArray3, PyUntypedArrayMethods,
};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

type RenderedEffects<'py> = (Bound<'py, PyArray2<f64>>, Vec<Bound<'py, PyArray3<f64>>>);

#[pyfunction]
#[allow(clippy::too_many_arguments)]
pub fn render_effects<'py>(
    py: Python<'py>,
    inputs: PyReadonlyArray3<'py, f64>,
    kinds: Vec<u8>,
    sources: PyReadonlyArray2<'py, i64>,
    parameters: PyReadonlyArray3<'py, f64>,
    output_source: usize,
    rate: f64,
    filter_inputs: Vec<Option<Inputs<'py>>>,
    state_floors: Vec<f64>,
) -> PyResult<RenderedEffects<'py>> {
    let input_count = inputs.shape()[0];
    let frames = inputs.shape()[1];
    let channels = inputs.shape()[2];
    let nodes = kinds.len();
    if !rate.is_finite()
        || rate <= 0.0
        || sources.shape() != [nodes, 2]
        || parameters.shape() != [frames, nodes, 2]
        || filter_inputs.len() != nodes
        || state_floors.len() != nodes
        || output_source >= input_count + nodes
        || kinds.iter().any(|k| *k > 2)
        || state_floors.iter().any(|v| !v.is_finite() || *v < 0.0)
        || inputs.as_array().iter().any(|v| !v.is_finite())
        || parameters.as_array().iter().any(|v| !v.is_finite())
    {
        return Err(PyValueError::new_err("Invalid native effect graph buffers"));
    }
    let sources = sources.as_array().to_owned();
    for node in 0..nodes {
        let ports = if kinds[node] == 1 { 2 } else { 1 };
        for port in 0..ports {
            let source = sources[[node, port]];
            if source < 0 || source as usize >= input_count + node {
                return Err(PyValueError::new_err(
                    "Effect source must precede its consumer",
                ));
            }
        }
    }
    let inputs = inputs.as_array().to_owned();
    let parameters = parameters.as_array().to_owned();
    let mut banks = filter_inputs
        .into_iter()
        .map(|v| FilterBank::new(v, rate, frames, channels))
        .collect::<PyResult<Vec<_>>>()?;
    let (output, states) = py.detach(move || -> PyResult<_> {
        let mut output = Array2::zeros((frames, channels));
        let mut values = vec![vec![0.0; channels]; input_count + nodes];
        for frame in 0..frames {
            for input in 0..input_count {
                for channel in 0..channels {
                    values[input][channel] = inputs[[input, frame, channel]];
                }
            }
            for node in 0..nodes {
                let first = sources[[node, 0]] as usize;
                let mix = parameters[[frame, node, 1]];
                if !(0.0..=1.0).contains(&mix) {
                    return Err(PyValueError::new_err("Invalid native effect mix"));
                }
                let mut wet = values[first].clone();
                match kinds[node] {
                    0 => {
                        let gain = 10_f64.powf(parameters[[frame, node, 0]] / 20.0);
                        for value in &mut wet {
                            *value *= gain;
                        }
                    }
                    1 => {
                        let second = sources[[node, 1]] as usize;
                        for (channel, value) in wet.iter_mut().enumerate() {
                            *value *= values[second][channel];
                        }
                    }
                    _ => {
                        banks[node].process(frame, &mut wet)?;
                        let floor = state_floors[node];
                        if floor > 0.0 {
                            for value in &mut banks[node].states {
                                if value.abs() < floor {
                                    *value = 0.0;
                                }
                            }
                        }
                    }
                }
                let destination = input_count + node;
                for (channel, wet_value) in wet.iter().enumerate() {
                    values[destination][channel] =
                        (1.0 - mix) * values[first][channel] + mix * wet_value;
                }
            }
            for channel in 0..channels {
                let value = values[output_source][channel];
                if !value.is_finite() {
                    return Err(PyValueError::new_err(format!(
                        "Non-finite effect output at frame {frame}"
                    )));
                }
                output[[frame, channel]] = value;
            }
        }
        let states: Vec<Array3<f64>> = banks.into_iter().map(|v| v.states).collect();
        Ok((output, states))
    })?;
    Ok((
        output.into_pyarray(py),
        states.into_iter().map(|v| v.into_pyarray(py)).collect(),
    ))
}
