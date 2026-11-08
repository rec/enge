//! Fixed-tuning input-excited resonance with passive scalar feedback.

use super::highpass::{self, Highpass};
use crate::error::{NativeError, NativeResult};
use numpy::ndarray::Array2;
use std::f64::consts::PI;

#[derive(Clone)]
pub(super) struct ResonantDelay {
    pub(super) history: Array2<f64>,
    pub(super) write: usize,
    pub(super) tuning: Highpass,
    pub(super) damping: Highpass,
    pub(super) non_silent: usize,
    frequency: f64,
    cutoff: f64,
    coefficient: f64,
    state_floor: f64,
}

impl ResonantDelay {
    pub(super) fn new(
        frequency: f64,
        cutoff: f64,
        rate: f64,
        channels: usize,
        state_floor: f64,
    ) -> NativeResult<Self> {
        if !frequency.is_finite() || frequency <= 0.0 || frequency > rate / 4.0 {
            return Err(NativeError::Message(
                "Resonant frequency must be positive and at most quarter-rate",
            ));
        }
        highpass::coefficient(cutoff, rate)?;
        let omega = 2.0 * PI * frequency / rate;
        let phase = (omega / 2.0).tan().atan2((PI * cutoff / rate).tan());
        let period = (2.0 * PI - phase) / omega;
        if !period.is_finite() || period >= isize::MAX as f64 {
            return Err(NativeError::Message(
                "Unrepresentable resonant delay capacity",
            ));
        }
        let frames = (period - 0.5).floor() as usize;
        let residual = (2.0 * PI - phase) - frames as f64 * omega;
        let q = (residual / 2.0).tan() / (omega / 2.0).tan();
        let coefficient = (1.0 - q) / (1.0 + q);
        if frames == 0
            || !(0.0..PI).contains(&residual)
            || residual == 0.0
            || !(-1.0..1.0).contains(&coefficient)
            || coefficient == -1.0
        {
            return Err(NativeError::Message(
                "Unrepresentable resonant tuning coefficient",
            ));
        }
        Ok(Self {
            history: Array2::zeros((frames, channels)),
            write: 0,
            tuning: Highpass::new(channels, state_floor),
            damping: Highpass::new(channels, state_floor),
            non_silent: 0,
            frequency,
            cutoff,
            coefficient,
            state_floor,
        })
    }

    pub(super) fn same_capacity(&self, other: &Self) -> bool {
        self.history.raw_dim() == other.history.raw_dim()
            && self.frequency == other.frequency
            && self.cutoff == other.cutoff
            && self.coefficient == other.coefficient
            && self.state_floor == other.state_floor
    }

    pub(super) fn drained(&self, threshold: f64) -> bool {
        self.non_silent == 0
            && self
                .tuning
                .memory
                .iter()
                .chain(&self.damping.memory)
                .all(|v| v.abs() < threshold)
    }

    pub(super) fn process(
        &mut self,
        dry: &[f64],
        wet: &mut [f64],
        filtered: &mut [f64],
        decay: f64,
        rate: f64,
        threshold: f64,
    ) -> NativeResult<()> {
        let gain = loss(self.frequency, decay)?;
        for (channel, value) in wet.iter_mut().enumerate() {
            *value = self.history[[self.write, channel]];
        }
        self.tuning.process_allpass(wet, self.coefficient)?;
        filtered.copy_from_slice(wet);
        self.damping.process(filtered, self.cutoff, rate)?;
        for channel in 0..wet.len() {
            let stored = dry[channel] + gain * (wet[channel] - filtered[channel]);
            if !stored.is_finite() {
                return Err(NativeError::Message("Non-finite resonant feedback"));
            }
            let stored = if stored.abs() < self.state_floor {
                0.0
            } else {
                stored
            };
            self.non_silent -= usize::from(self.history[[self.write, channel]].abs() >= threshold);
            self.non_silent += usize::from(stored.abs() >= threshold);
            self.history[[self.write, channel]] = stored;
        }
        self.write = (self.write + 1) % self.history.nrows();
        Ok(())
    }
}

pub(super) fn loss(frequency: f64, decay: f64) -> NativeResult<f64> {
    if !decay.is_finite() || decay <= 0.0 {
        return Err(NativeError::Message(
            "Resonant decay must be finite and positive",
        ));
    }
    let gain = ((-3.0 * 10.0_f64.ln() / frequency) / decay).exp();
    if !(0.0..1.0).contains(&gain) {
        return Err(NativeError::Message("Unrepresentable resonant decay"));
    }
    Ok(gain)
}
