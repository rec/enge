//! Per-sample trapezoidal filters, shared by oscillator and sample voices.
use crate::error::{NativeError, NativeResult};
use numpy::ndarray::{Array2, Array3};
use numpy::{
    IntoPyArray, PyArray2, PyArray3, PyReadonlyArray2, PyReadonlyArray3, PyUntypedArrayMethods,
};
use pyo3::prelude::*;
use std::f64::consts::PI;

pub type Inputs<'py> = (
    Vec<u8>,
    PyReadonlyArray3<'py, f64>,
    PyReadonlyArray3<'py, f64>,
);
type RenderedFilters<'py> = (Bound<'py, PyArray2<f64>>, Bound<'py, PyArray3<f64>>);

#[derive(Clone)]
pub struct FilterBank {
    coefficients: Vec<Coefficients>,
    responses: Vec<u8>,
    resonant_stages: usize,
    pub states: Array3<f64>,
    values: Array3<f64>,
    rate: f64,
}

#[derive(Clone, Default)]
pub(crate) struct Coefficients {
    inputs: Option<[f64; 3]>,
    values: [f64; 5],
}

impl Coefficients {
    pub(crate) fn get(&mut self, cutoff: f64, q: f64, rate: f64) -> [f64; 5] {
        let inputs = [cutoff, q, rate];
        if self.inputs != Some(inputs) {
            let g = (PI * cutoff / rate).tan();
            let d = q * (1.0 + g * g) + g;
            let a1 = if q < 1.0 {
                q / d
            } else {
                1.0 / (1.0 + g * (g + 1.0 / q))
            };
            self.values = [g, d, a1, g * a1, g * (g * a1)];
            self.inputs = Some(inputs);
        }
        self.values
    }
}

impl FilterBank {
    pub(crate) fn resonant_stages(&self) -> usize {
        self.resonant_stages
    }

    pub fn same_definition(&self, other: &Self) -> bool {
        self.responses == other.responses && self.values == other.values && self.rate == other.rate
    }

    pub fn new(
        inputs: Option<Inputs<'_>>,
        rate: f64,
        frames: usize,
        channels: usize,
    ) -> NativeResult<Self> {
        if !rate.is_finite() || rate <= 0.0 {
            return Err(NativeError::Message("Invalid native filter rate"));
        }
        let Some((responses, states, values)) = inputs else {
            return Ok(Self {
                coefficients: vec![],
                responses: vec![],
                resonant_stages: 0,
                states: Array3::zeros((0, channels, 2)),
                values: Array3::zeros((frames, 0, 2)),
                rate,
            });
        };
        let stages = responses.len();
        let resonant_stages = responses.iter().take_while(|r| **r < 4).count();
        let width = if resonant_stages == stages { 2 } else { 5 };
        if responses[resonant_stages..].iter().any(|r| *r != 4)
            || states.shape()[0] != stages
            || states.shape()[2] != 2
            || (stages != 0 && states.shape()[1] != channels)
            || values.shape() != [frames, stages, width]
            || states.as_array().iter().any(|v| !v.is_finite())
        {
            return Err(NativeError::Message("Invalid native filter buffers"));
        }
        let values = values.as_array().to_owned();
        for i in 0..frames {
            for j in 0..resonant_stages {
                let (cutoff, q) = (values[[i, j, 0]], values[[i, j, 1]]);
                if !cutoff.is_finite()
                    || cutoff <= 0.0
                    || cutoff >= rate / 2.0
                    || !q.is_finite()
                    || q <= 0.0
                {
                    return Err(NativeError::Message("Invalid native filter cutoff/Q"));
                }
            }
            for j in resonant_stages..stages {
                if (0..5).any(|k| !values[[i, j, k]].is_finite()) {
                    return Err(NativeError::Message("Invalid native EQ coefficients"));
                }
            }
        }
        Ok(Self {
            coefficients: vec![Coefficients::default(); resonant_stages],
            responses,
            resonant_stages,
            states: states.as_array().to_owned(),
            values,
            rate,
        })
    }

    pub fn process(&mut self, frame: usize, sources: &mut [f64]) -> NativeResult<()> {
        for j in 0..self.resonant_stages {
            self.process_stage(
                frame,
                j,
                self.values[[frame, j, 0]],
                self.values[[frame, j, 1]],
                sources,
            )?;
        }
        self.process_equalizer(frame, frame, sources)
    }

    pub(crate) fn process_equalizer(
        &mut self,
        values_frame: usize,
        output_frame: usize,
        sources: &mut [f64],
    ) -> NativeResult<()> {
        for j in self.resonant_stages..self.responses.len() {
            let [b0, b1, b2, a1, a2] = std::array::from_fn(|k| self.values[[values_frame, j, k]]);
            for (c, sample) in sources.iter_mut().enumerate() {
                let input = *sample;
                let output = b0 * input + self.states[[j, c, 0]];
                let first = b1 * input - a1 * output + self.states[[j, c, 1]];
                let second = b2 * input - a2 * output;
                if !output.is_finite() || !first.is_finite() || !second.is_finite() {
                    return Err(NativeError::FilterOutput(output_frame));
                }
                self.states[[j, c, 0]] = first;
                self.states[[j, c, 1]] = second;
                *sample = output;
            }
        }
        Ok(())
    }

    pub(crate) fn process_stage(
        &mut self,
        frame: usize,
        j: usize,
        cutoff: f64,
        q: f64,
        sources: &mut [f64],
    ) -> NativeResult<()> {
        let response = self.responses[j];
        let [g, d, a1, a2, a3] = self.coefficients[j].get(cutoff, q, self.rate);
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
                return Err(NativeError::FilterOutput(frame));
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
