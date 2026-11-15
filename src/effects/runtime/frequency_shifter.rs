//! Fixed antisymmetric Hilbert FIR with a shared continuous oscillator phase.

use crate::error::{NativeError, NativeResult};
use numpy::ndarray::Array2;
use std::f64::consts::TAU;

#[derive(Clone)]
pub(super) struct FrequencyShifter {
    pub(super) history: Array2<f64>,
    pub(super) write: usize,
    pub(super) phase: f64,
    pub(super) remaining: usize,
    coefficients: [f64; 256],
}

impl FrequencyShifter {
    pub(super) fn new(channels: usize) -> Self {
        Self {
            history: Array2::zeros((1025, channels)),
            write: 0,
            phase: 0.0,
            remaining: 1024,
            coefficients: std::array::from_fn(|i| {
                let n = (2 * i + 1) as f64;
                let angle = TAU * n / 1024.0;
                2.0 / (std::f64::consts::PI * (n - 512.0))
                    * (0.42 - 0.5 * angle.cos() + 0.08 * (2.0 * angle).cos())
            }),
        }
    }

    pub(super) fn same_capacity(&self, other: &Self) -> bool {
        self.history.raw_dim() == other.history.raw_dim()
    }

    pub(super) fn process(
        &mut self,
        dry: &mut [f64],
        wet: &mut [f64],
        shift: f64,
        rate: f64,
        ended: bool,
    ) -> NativeResult<bool> {
        if !(-2000.0..=2000.0).contains(&shift) || wet.iter().any(|v| !v.is_finite()) {
            return Err(NativeError::Message(
                "Invalid frequency shifter input or shift",
            ));
        }
        let (sin, cos) = (TAU * self.phase).sin_cos();
        for (channel, output) in wet.iter_mut().enumerate() {
            self.history[[self.write, channel]] = *output;
            dry[channel] = self.history[[(self.write + 1025 - 512) % 1025, channel]];
            let mut quadrature = 0.0;
            for (i, coefficient) in self.coefficients.iter().enumerate() {
                let tap = 2 * i + 1;
                let a = self.history[[(self.write + 1025 - tap) % 1025, channel]];
                let b = self.history[[(self.write + 1 + tap) % 1025, channel]];
                quadrature += coefficient * (a - b);
            }
            *output = dry[channel] * cos - quadrature * sin;
            if !output.is_finite() {
                return Err(NativeError::Message("Non-finite frequency shifter output"));
            }
        }
        self.write = (self.write + 1) % 1025;
        self.phase = (self.phase + shift / rate).rem_euclid(1.0);
        // rem_euclid can round a tiny negative remainder to its positive divisor.
        if self.phase == 1.0 {
            self.phase = 0.0;
        }
        let drained = ended && self.remaining == 0;
        self.remaining = if ended {
            self.remaining.saturating_sub(1)
        } else {
            1024
        };
        Ok(drained)
    }
}
