//! Per-channel Schmitt divider, with the shared peak-envelope law.

use crate::error::{NativeError, NativeResult};
use numpy::ndarray::ArrayView1;

#[derive(Clone)]
pub(super) struct Octave {
    pub(super) levels: Vec<f64>,
    pub(super) armed: Vec<bool>,
    pub(super) positive: Vec<bool>,
    pub(super) active: Vec<bool>,
}

impl Octave {
    pub(super) fn new(channels: usize) -> Self {
        Self {
            levels: vec![0.0; channels],
            armed: vec![false; channels],
            positive: vec![false; channels],
            active: vec![false; channels],
        }
    }

    pub(super) fn process(
        &mut self,
        audio: &mut [f64],
        parameters: [f64; 3],
        rate: f64,
        floor: f64,
        tail: f64,
        ended: bool,
    ) -> NativeResult<bool> {
        let [threshold, attack, release] = parameters;
        if !threshold.is_finite()
            || threshold <= 0.0
            || !attack.is_finite()
            || !release.is_finite()
            || attack < 0.0
            || release < 0.0
        {
            return Err(NativeError::Message("Invalid octave divider controls"));
        }
        if audio.iter().any(|v| !v.is_finite()) {
            return Err(NativeError::Message("Non-finite octave divider input"));
        }
        for (i, value) in audio.iter_mut().enumerate() {
            super::follower::process(
                &mut self.levels[i],
                ArrayView1::from(&[*value]),
                attack,
                release,
                rate,
                floor,
            )?;
            if self.levels[i] == 0.0 {
                self.armed[i] = false;
                self.positive[i] = false;
                self.active[i] = false;
            } else if *value <= -threshold {
                self.armed[i] = true;
            } else if *value >= threshold && self.armed[i] {
                self.positive[i] = !self.positive[i];
                self.active[i] = true;
                self.armed[i] = false;
            }
            *value = if self.active[i] {
                self.levels[i] * if self.positive[i] { 1.0 } else { -1.0 }
            } else {
                0.0
            };
        }
        let drained = ended && self.levels.iter().all(|v| *v < tail);
        if drained {
            self.levels.fill(0.0);
            self.armed.fill(false);
            self.positive.fill(false);
            self.active.fill(false);
            audio.fill(0.0);
        }
        Ok(drained)
    }
}
