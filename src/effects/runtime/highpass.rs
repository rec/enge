//! One-pole highpass using one normalized all-pass memory per channel.

use crate::error::{NativeError, NativeResult};
use std::f64::consts::PI;

#[derive(Clone)]
pub(super) struct Highpass {
    pub(super) memory: Vec<f64>,
    state_floor: f64,
}

impl Highpass {
    pub(super) fn new(channels: usize, state_floor: f64) -> Self {
        Self {
            memory: vec![0.0; channels],
            state_floor,
        }
    }

    pub(super) fn same_capacity(&self, other: &Self) -> bool {
        self.memory.len() == other.memory.len() && self.state_floor == other.state_floor
    }

    pub(super) fn process(
        &mut self,
        source: &mut [f64],
        cutoff: f64,
        rate: f64,
    ) -> NativeResult<()> {
        let a = coefficient(cutoff, rate)?;
        let b = ((1.0 - a) * (1.0 + a)).sqrt();
        for (value, memory) in source.iter_mut().zip(&mut self.memory) {
            let wet = (*value - (a * *value + b * *memory)) / 2.0;
            let next = b * *value - a * *memory;
            if !wet.is_finite() || !next.is_finite() {
                return Err(NativeError::Message("Non-finite highpass state"));
            }
            *memory = if next.abs() < self.state_floor {
                0.0
            } else {
                next
            };
            *value = wet;
        }
        Ok(())
    }
}

pub(super) fn coefficient(cutoff: f64, rate: f64) -> NativeResult<f64> {
    if !cutoff.is_finite() || cutoff <= 0.0 || cutoff >= rate / 2.0 {
        return Err(NativeError::Message(
            "Highpass cutoff must be positive and below Nyquist",
        ));
    }
    let g = (PI * (cutoff / rate)).tan();
    let a = (g - 1.0) / (g + 1.0);
    if !(-1.0..1.0).contains(&a) || a == -1.0 {
        return Err(NativeError::Message("Unrepresentable highpass coefficient"));
    }
    Ok(a)
}
