//! Six all-pass stages with one frame of feedback and explicit finite state.

use crate::error::{NativeError, NativeResult};
use numpy::ndarray::Array2;
use std::f64::consts::PI;

#[derive(Clone)]
pub(super) struct Phaser {
    pub(super) stages: Array2<f64>,
    pub(super) feedback: Vec<f64>,
    state_floor: f64,
}

impl Phaser {
    pub(super) fn new(channels: usize, state_floor: f64) -> Self {
        Self {
            stages: Array2::zeros((6, channels)),
            feedback: vec![0.0; channels],
            state_floor,
        }
    }

    pub(super) fn same_capacity(&self, other: &Self) -> bool {
        self.stages.raw_dim() == other.stages.raw_dim() && self.state_floor == other.state_floor
    }

    pub(super) fn process(
        &mut self,
        wet: &mut [f64],
        parameters: [f64; 4],
        rate: f64,
    ) -> NativeResult<()> {
        let [center, depth, sweep, feedback] = parameters;
        let a = coefficient(center, depth, sweep, rate)?;
        let b = ((1.0 - a) * (1.0 + a)).sqrt();
        if !(-0.95..=0.95).contains(&feedback) {
            return Err(NativeError::Message("Invalid phaser feedback"));
        }
        for (channel, output) in wet.iter_mut().enumerate() {
            if !output.is_finite() {
                return Err(NativeError::Message("Non-finite phaser input"));
            }
            let mut value = *output + feedback * self.feedback[channel];
            for stage in 0..6 {
                let memory = self.stages[[stage, channel]];
                let y = a * value + b * memory;
                let memory = b * value - a * memory;
                if !y.is_finite() || !memory.is_finite() {
                    return Err(NativeError::Message("Non-finite phaser state"));
                }
                self.stages[[stage, channel]] = if memory.abs() < self.state_floor {
                    0.0
                } else {
                    memory
                };
                value = y;
            }
            self.feedback[channel] = if value.abs() < self.state_floor {
                0.0
            } else {
                value
            };
            *output = value;
        }
        Ok(())
    }
}

pub(super) fn coefficient(center: f64, depth: f64, sweep: f64, rate: f64) -> NativeResult<f64> {
    if !center.is_finite()
        || center <= 0.0
        || !depth.is_finite()
        || depth < 0.0
        || !(-1.0..=1.0).contains(&sweep)
    {
        return Err(NativeError::Message(
            "Invalid phaser center, depth or sweep",
        ));
    }
    let exponent = center.log2() + depth * sweep / 1200.0;
    if !exponent.is_finite() || exponent >= (rate / 2.0).log2() {
        return Err(NativeError::Message(
            "Phaser frequency must remain below Nyquist",
        ));
    }
    let g = (PI * exponent.exp2() / rate).tan();
    let value = (g - 1.0) / (g + 1.0);
    if !(-1.0..1.0).contains(&value) || value == -1.0 {
        return Err(NativeError::Message("Unrepresentable phaser coefficient"));
    }
    Ok(value)
}
