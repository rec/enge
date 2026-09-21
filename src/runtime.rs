//! Persistent oscillator-to-effect runtime used to measure one native boundary.

use numpy::ndarray::Array2;
use numpy::{IntoPyArray, PyArray2, PyReadonlyArray2, PyUntypedArrayMethods};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use std::f64::consts::{PI, TAU};

#[pyclass]
pub struct OscillatorFilterRuntime {
    rate: f64,
    frequencies: Vec<f64>,
    phases: Vec<f64>,
    errors: Vec<f64>,
    gains: Vec<f64>,
    routes: Array2<f64>,
    states: Array2<f64>,
}

#[pymethods]
impl OscillatorFilterRuntime {
    #[new]
    fn new(
        rate: f64,
        frequencies: Vec<f64>,
        gains: Vec<f64>,
        routes: PyReadonlyArray2<'_, f64>,
    ) -> PyResult<Self> {
        if !rate.is_finite()
            || rate <= 0.0
            || frequencies.is_empty()
            || frequencies.len() != gains.len()
            || routes.shape()[0] != frequencies.len()
            || routes.shape()[1] == 0
            || frequencies.iter().any(|v| !v.is_finite() || *v <= 0.0)
            || gains.iter().any(|v| !v.is_finite())
            || routes.as_array().iter().any(|v| !v.is_finite())
        {
            return Err(PyValueError::new_err(
                "Invalid persistent runtime definition",
            ));
        }
        let channels = routes.shape()[1];
        Ok(Self {
            rate,
            phases: vec![0.0; frequencies.len()],
            errors: vec![0.0; frequencies.len()],
            frequencies,
            gains,
            routes: routes.as_array().to_owned(),
            states: Array2::zeros((channels, 2)),
        })
    }

    fn process<'py>(
        &mut self,
        py: Python<'py>,
        frames: usize,
        cutoff_hz: f64,
        q: f64,
        gain_db: f64,
    ) -> PyResult<Bound<'py, PyArray2<f64>>> {
        if frames == 0
            || !cutoff_hz.is_finite()
            || cutoff_hz <= 0.0
            || cutoff_hz >= self.rate / 2.0
            || !q.is_finite()
            || q <= 0.0
            || !gain_db.is_finite()
        {
            return Err(PyValueError::new_err("Invalid persistent runtime block"));
        }
        let channels = self.routes.ncols();
        let gain = 10_f64.powf(gain_db / 20.0);
        let g = (PI * cutoff_hz / self.rate).tan();
        let a1 = if q < 1.0 {
            q / (q * (1.0 + g * g) + g)
        } else {
            1.0 / (1.0 + g * (g + 1.0 / q))
        };
        let a2 = g * a1;
        let a3 = g * a2;
        let mut output = Array2::zeros((frames, channels));
        for frame in 0..frames {
            for voice in 0..self.frequencies.len() {
                let sample = (TAU * self.phases[voice] / self.rate).sin() * self.gains[voice];
                let increment = self.frequencies[voice] - self.errors[voice];
                let total = self.phases[voice] + increment;
                self.errors[voice] = (total - self.phases[voice]) - increment;
                self.phases[voice] = total.rem_euclid(self.rate);
                for channel in 0..channels {
                    output[[frame, channel]] += sample * self.routes[[voice, channel]];
                }
            }
            for channel in 0..channels {
                let input = output[[frame, channel]];
                let s1 = self.states[[channel, 0]];
                let s2 = self.states[[channel, 1]];
                let v3 = input - s2;
                let v1 = a1 * s1 + a2 * v3;
                let v2 = s2 + a2 * s1 + a3 * v3;
                self.states[[channel, 0]] = 2.0 * v1 - s1;
                self.states[[channel, 1]] = 2.0 * v2 - s2;
                output[[frame, channel]] = v2 * gain;
            }
        }
        Ok(output.into_pyarray(py))
    }
}
