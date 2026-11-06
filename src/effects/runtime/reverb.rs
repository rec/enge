//! Eight-line Householder feedback network; predelay is owned by the graph.

use crate::error::{NativeError, NativeResult};
use std::f64::consts::{PI, SQRT_2};

#[derive(Clone)]
pub(super) struct Reverb {
    pub(super) lines: Vec<Vec<f64>>,
    pub(super) writes: [usize; 8],
    pub(super) damping: [f64; 8],
    pub(super) non_silent: usize,
}

impl Reverb {
    pub(super) fn new(lengths: [usize; 8]) -> Self {
        Self {
            lines: lengths
                .into_iter()
                .map(|length| vec![0.0; length])
                .collect(),
            writes: [0; 8],
            damping: [0.0; 8],
            non_silent: 0,
        }
    }

    pub(super) fn process(
        &mut self,
        input: [f64; 2],
        decay: f64,
        damping_hz: f64,
        rate: f64,
        state_floor: f64,
        threshold: f64,
    ) -> NativeResult<[f64; 2]> {
        let (a, gains) = self.coefficients(decay, damping_hz, rate)?;
        let mut output = [0.0; 2];
        let scale = 1.0 / (2.0 * SQRT_2);
        for i in 0..8 {
            let value = self.lines[i][self.writes[i]];
            output[0] += if i % 2 == 0 {
                scale * value
            } else {
                -scale * value
            };
            output[1] += if i.count_ones() % 2 == 0 {
                scale * value
            } else {
                -scale * value
            };
            let damped = (1.0 - a) * value + a * self.damping[i];
            self.damping[i] = if damped.abs() < state_floor {
                0.0
            } else {
                damped
            };
        }
        let sum = self.damping.iter().sum::<f64>();
        for (i, gain) in gains.into_iter().enumerate() {
            let left = if i / 4 % 2 == 0 { scale } else { -scale };
            let right = if i / 2 % 2 == 0 { scale } else { -scale };
            let value = left * input[0] + right * input[1] + gain * (self.damping[i] - sum / 4.0);
            let value = if value.abs() < state_floor {
                0.0
            } else {
                value
            };
            if !value.is_finite() || !self.damping[i].is_finite() {
                return Err(NativeError::Message("Non-finite reverb state"));
            }
            self.non_silent -= usize::from(self.lines[i][self.writes[i]].abs() >= threshold);
            self.non_silent += usize::from(value.abs() >= threshold);
            self.lines[i][self.writes[i]] = value;
            self.writes[i] = (self.writes[i] + 1) % self.lines[i].len();
        }
        Ok(output)
    }

    pub(super) fn coefficients(
        &self,
        decay: f64,
        damping_hz: f64,
        rate: f64,
    ) -> NativeResult<(f64, [f64; 8])> {
        if !decay.is_finite() || decay <= 0.0 || !damping_hz.is_finite() || damping_hz <= 0.0 {
            return Err(NativeError::Message("Invalid reverb decay or damping"));
        }
        let a = (-2.0 * PI * damping_hz / rate).exp();
        if !(0.0..1.0).contains(&a) {
            return Err(NativeError::Message("Unrepresentable reverb damping"));
        }
        let mut gains = [0.0; 8];
        for (gain, line) in gains.iter_mut().zip(&self.lines) {
            *gain = 10.0_f64.powf(-3.0 * line.len() as f64 / (rate * decay));
            if !(0.0..1.0).contains(gain) {
                return Err(NativeError::Message("Unrepresentable reverb decay"));
            }
        }
        Ok((a, gains))
    }
}
