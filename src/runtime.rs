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
    active: Vec<bool>,
    levels: Vec<f64>,
    level_steps: Vec<f64>,
    level_remaining: Vec<usize>,
    retire_after_ramp: Vec<bool>,
    frequency_steps: Vec<f64>,
    frequency_remaining: Vec<usize>,
    gain_steps: Vec<f64>,
    gain_remaining: Vec<usize>,
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
            active: gains.iter().map(|v| *v != 0.0).collect(),
            levels: gains
                .iter()
                .map(|v| if *v == 0.0 { 0.0 } else { 1.0 })
                .collect(),
            level_steps: vec![0.0; gains.len()],
            level_remaining: vec![0; gains.len()],
            retire_after_ramp: vec![false; gains.len()],
            frequency_steps: vec![0.0; gains.len()],
            frequency_remaining: vec![0; gains.len()],
            gain_steps: vec![0.0; gains.len()],
            gain_remaining: vec![0; gains.len()],
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
        self.render(py, frames, cutoff_hz, q, gain_db, &[])
    }

    fn process_actions<'py>(
        &mut self,
        py: Python<'py>,
        frames: usize,
        cutoff_hz: f64,
        q: f64,
        gain_db: f64,
        actions: PyReadonlyArray2<'_, f64>,
    ) -> PyResult<Bound<'py, PyArray2<f64>>> {
        if actions.shape()[1] != 6 || actions.as_array().iter().any(|v| !v.is_finite()) {
            return Err(PyValueError::new_err("Invalid persistent runtime actions"));
        }
        self.render(py, frames, cutoff_hz, q, gain_db, actions.as_slice()?)
    }
}

impl OscillatorFilterRuntime {
    fn render<'py>(
        &mut self,
        py: Python<'py>,
        frames: usize,
        cutoff_hz: f64,
        q: f64,
        gain_db: f64,
        actions: &[f64],
    ) -> PyResult<Bound<'py, PyArray2<f64>>> {
        if frames == 0
            || !cutoff_hz.is_finite()
            || cutoff_hz <= 0.0
            || cutoff_hz >= self.rate / 2.0
            || !q.is_finite()
            || q <= 0.0
            || !gain_db.is_finite()
            || actions
                .chunks_exact(6)
                .zip(actions.chunks_exact(6).skip(1))
                .any(|(a, b)| b[0] < a[0])
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
        let mut action = 0;
        for frame in 0..frames {
            while action < actions.len() && actions[action] as usize == frame {
                self.apply_action(&actions[action..action + 6], frames)?;
                action += 6;
            }
            for voice in 0..self.frequencies.len() {
                if !self.active[voice] {
                    continue;
                }
                let sample = (TAU * self.phases[voice] / self.rate).sin()
                    * self.gains[voice]
                    * self.levels[voice];
                let increment = self.frequencies[voice] - self.errors[voice];
                let total = self.phases[voice] + increment;
                self.errors[voice] = (total - self.phases[voice]) - increment;
                self.phases[voice] = total.rem_euclid(self.rate);
                for channel in 0..channels {
                    output[[frame, channel]] += sample * self.routes[[voice, channel]];
                }
                advance_ramp(
                    &mut self.frequencies[voice],
                    self.frequency_steps[voice],
                    &mut self.frequency_remaining[voice],
                );
                advance_ramp(
                    &mut self.gains[voice],
                    self.gain_steps[voice],
                    &mut self.gain_remaining[voice],
                );
                advance_ramp(
                    &mut self.levels[voice],
                    self.level_steps[voice],
                    &mut self.level_remaining[voice],
                );
                if self.level_remaining[voice] == 0 && self.retire_after_ramp[voice] {
                    self.active[voice] = false;
                    self.retire_after_ramp[voice] = false;
                    self.levels[voice] = 0.0;
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

    fn apply_action(&mut self, action: &[f64], frames: usize) -> PyResult<()> {
        let offset = action[0] as usize;
        let kind = action[1] as usize;
        let voice = action[2] as usize;
        let duration = action[5] as usize;
        if action[0] != offset as f64
            || offset >= frames
            || action[1] != kind as f64
            || action[2] != voice as f64
            || voice >= self.frequencies.len()
            || action[5] != duration as f64
        {
            return Err(PyValueError::new_err("Invalid persistent runtime action"));
        }
        match kind {
            0 => {
                if action[3] <= 0.0 || action[4] < 0.0 || self.active[voice] {
                    return Err(PyValueError::new_err("Invalid voice start"));
                }
                self.active[voice] = true;
                self.phases[voice] = 0.0;
                self.errors[voice] = 0.0;
                self.frequencies[voice] = action[3];
                self.gains[voice] = action[4];
                self.levels[voice] = if duration == 0 { 1.0 } else { 0.0 };
                self.level_steps[voice] = if duration == 0 {
                    0.0
                } else {
                    1.0 / duration as f64
                };
                self.level_remaining[voice] = duration;
                self.retire_after_ramp[voice] = false;
            }
            1 => {
                if duration == 0 {
                    self.active[voice] = false;
                } else {
                    self.level_steps[voice] = -self.levels[voice] / duration as f64;
                    self.level_remaining[voice] = duration;
                    self.retire_after_ramp[voice] = true;
                }
            }
            2 => self.active[voice] = false,
            3 => {
                if action[3] <= 0.0 || action[4] < 0.0 {
                    return Err(PyValueError::new_err("Invalid voice control"));
                }
                self.frequency_steps[voice] =
                    (action[3] - self.frequencies[voice]) / duration.max(1) as f64;
                self.frequency_remaining[voice] = duration;
                self.gain_steps[voice] = (action[4] - self.gains[voice]) / duration.max(1) as f64;
                self.gain_remaining[voice] = duration;
                if duration == 0 {
                    self.frequencies[voice] = action[3];
                    self.gains[voice] = action[4];
                }
            }
            _ => return Err(PyValueError::new_err("Unknown persistent runtime action")),
        }
        Ok(())
    }
}

fn advance_ramp(value: &mut f64, step: f64, remaining: &mut usize) {
    if *remaining > 0 {
        *value += step;
        *remaining -= 1;
    }
}
