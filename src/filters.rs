//! Per-sample trapezoidal filters, shared by oscillator and sample voices.
use numpy::ndarray::{Array2, Array3};
use numpy::{
    IntoPyArray, PyArray2, PyArray3, PyReadonlyArray2, PyReadonlyArray3, PyUntypedArrayMethods,
};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use std::f64::consts::PI;

pub type Inputs<'py> = (
    Vec<u8>,
    PyReadonlyArray3<'py, f64>,
    PyReadonlyArray3<'py, f64>,
);
type RenderedFilters<'py> = (Bound<'py, PyArray2<f64>>, Bound<'py, PyArray3<f64>>);

pub struct FilterBank {
    responses: Vec<u8>,
    pub states: Array3<f64>,
    values: Array3<f64>,
    rate: f64,
}

impl FilterBank {
    pub fn new(
        inputs: Option<Inputs<'_>>,
        rate: f64,
        frames: usize,
        channels: usize,
    ) -> PyResult<Self> {
        if !rate.is_finite() || rate <= 0.0 {
            return Err(PyValueError::new_err("Invalid native filter rate"));
        }
        let Some((responses, states, values)) = inputs else {
            return Ok(Self {
                responses: vec![],
                states: Array3::zeros((0, channels, 2)),
                values: Array3::zeros((frames, 0, 2)),
                rate,
            });
        };
        let stages = responses.len();
        if responses.iter().any(|r| *r > 3)
            || states.shape()[0] != stages
            || states.shape()[2] != 2
            || (stages != 0 && states.shape()[1] != channels)
            || values.shape() != [frames, stages, 2]
            || states.as_array().iter().any(|v| !v.is_finite())
        {
            return Err(PyValueError::new_err("Invalid native filter buffers"));
        }
        let values = values.as_array().to_owned();
        for i in 0..frames {
            for j in 0..stages {
                let (cutoff, q) = (values[[i, j, 0]], values[[i, j, 1]]);
                if !cutoff.is_finite()
                    || cutoff <= 0.0
                    || cutoff >= rate / 2.0
                    || !q.is_finite()
                    || q <= 0.0
                {
                    return Err(PyValueError::new_err("Invalid native filter cutoff/Q"));
                }
            }
        }
        Ok(Self {
            responses,
            states: states.as_array().to_owned(),
            values,
            rate,
        })
    }

    pub fn process(&mut self, frame: usize, sources: &mut [f64]) -> PyResult<()> {
        for (j, response) in self.responses.iter().enumerate() {
            let cutoff = self.values[[frame, j, 0]];
            let q = self.values[[frame, j, 1]];
            let g = (PI * cutoff / self.rate).tan();
            let d = q * (1.0 + g * g) + g;
            let a1 = if q < 1.0 {
                q / d
            } else {
                1.0 / (1.0 + g * (g + 1.0 / q))
            };
            let a2 = g * a1;
            let a3 = g * a2;
            for (c, input) in sources.iter_mut().enumerate() {
                let (s1, s2) = (self.states[[j, c, 0]], self.states[[j, c, 1]]);
                let v3 = *input - s2;
                let v1 = a1 * s1 + a2 * v3;
                let v2 = s2 + a2 * s1 + a3 * v3;
                let band = if q < 1.0 {
                    s1 / d + (g / d) * v3
                } else {
                    v1 / q
                };
                *input = match response {
                    0 => v2,
                    1 => *input - band - v2,
                    2 => band,
                    _ => *input - band,
                };
                self.states[[j, c, 0]] = 2.0 * v1 - s1;
                self.states[[j, c, 1]] = 2.0 * v2 - s2;
                if !input.is_finite()
                    || !self.states[[j, c, 0]].is_finite()
                    || !self.states[[j, c, 1]].is_finite()
                {
                    return Err(PyValueError::new_err(format!(
                        "Non-finite filter output or state at frame {frame}"
                    )));
                }
            }
        }
        Ok(())
    }
}

#[pyfunction]
pub fn render_filters<'py>(
    py: Python<'py>,
    samples: PyReadonlyArray2<'py, f64>,
    rate: f64,
    filters: Inputs<'py>,
) -> PyResult<RenderedFilters<'py>> {
    let mut bank = FilterBank::new(Some(filters), rate, samples.shape()[0], samples.shape()[1])?;
    let mut audio: Array2<f64> = samples.as_array().to_owned();
    let (audio, states) = py.detach(move || -> PyResult<_> {
        let mut sources = vec![0.0; audio.ncols()];
        for i in 0..audio.nrows() {
            for (c, value) in sources.iter_mut().enumerate() {
                *value = audio[[i, c]];
            }
            bank.process(i, &mut sources)?;
            for (c, value) in sources.iter().enumerate() {
                audio[[i, c]] = *value;
            }
        }
        Ok((audio, bank.states))
    })?;
    Ok((audio.into_pyarray(py), states.into_pyarray(py)))
}
