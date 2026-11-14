//! Linked two-follower contrast, with fixed detector times and bounded gain.

use crate::error::{NativeError, NativeResult};
use numpy::ndarray::ArrayView1;

#[derive(Clone, Default)]
pub(super) struct Transient {
    pub(super) fast: f64,
    pub(super) slow: f64,
}

impl Transient {
    pub(super) fn process(
        &mut self,
        audio: &mut [f64],
        attack_db: f64,
        sustain_db: f64,
        rate: f64,
        ended: bool,
    ) -> NativeResult<()> {
        if [attack_db, sustain_db]
            .iter()
            .any(|v| !v.is_finite() || !(-24.0..=24.0).contains(v))
        {
            return Err(NativeError::Message(
                "Transient shaper gains must be in [-24, 24] dB",
            ));
        }
        if audio.iter().any(|v| !v.is_finite()) {
            return Err(NativeError::Message("Non-finite transient shaper input"));
        }
        if ended {
            self.fast = 0.0;
            self.slow = 0.0;
            audio.fill(0.0);
            return Ok(());
        }
        super::follower::process(
            &mut self.fast,
            ArrayView1::from(&*audio),
            0.001,
            0.01,
            rate,
            1e-30,
        )?;
        super::follower::process(
            &mut self.slow,
            ArrayView1::from(&*audio),
            0.01,
            0.1,
            rate,
            1e-30,
        )?;
        let contrast = (self.fast - self.slow) / self.fast.max(self.slow).max(1e-12);
        let gain_db = attack_db * contrast.max(0.0) + sustain_db * (-contrast).max(0.0);
        let gain = 10_f64.powf(gain_db / 20.0);
        for value in audio.iter_mut() {
            *value *= gain;
        }
        if audio.iter().any(|v| !v.is_finite()) {
            return Err(NativeError::Message("Non-finite transient shaper output"));
        }
        Ok(())
    }
}
