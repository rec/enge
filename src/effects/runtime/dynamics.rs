//! Zero-look-ahead gating and downward expansion, linked across channels.

use crate::error::{NativeError, NativeResult};

#[derive(Clone, Default)]
pub(super) struct Gate {
    pub(super) gain: f64,
    pub(super) opened: bool,
    pub(super) elapsed: usize,
}

pub(super) fn hold_frames(seconds: f64, rate: f64) -> NativeResult<usize> {
    let frames = seconds * rate;
    if !frames.is_finite() || frames < 0.0 || frames >= (isize::MAX as f64) {
        return Err(NativeError::Message(
            "Gate hold cannot use the native frame format",
        ));
    }
    Ok(frames.ceil() as usize)
}

impl Gate {
    pub(super) fn process(
        &mut self,
        peak: f64,
        parameters: [f64; 5],
        rate: f64,
    ) -> NativeResult<f64> {
        let [threshold, hysteresis, hold, attack, release] = parameters;
        let closing = threshold - hysteresis;
        if hysteresis < 0.0 || !closing.is_finite() {
            return Err(NativeError::Message(
                "Invalid gate hysteresis or closing threshold",
            ));
        }
        let frames = hold_frames(hold, rate)?;
        let level = if peak > 0.0 {
            20.0 * peak.log10()
        } else {
            f64::NEG_INFINITY
        };
        if level >= threshold {
            self.opened = true;
            self.elapsed = 0;
        } else if self.opened {
            if level >= closing {
                self.elapsed = 0;
            } else if self.elapsed >= frames {
                self.opened = false;
                self.elapsed = 0;
            } else {
                self.elapsed += 1;
            }
        } else {
            self.elapsed = 0;
        }
        self.gain = smooth(self.gain, f64::from(self.opened), attack, release, rate)?;
        Ok(self.gain)
    }
}

pub(super) fn expand(gain: f64, peak: f64, parameters: [f64; 4], rate: f64) -> NativeResult<f64> {
    let [threshold, ratio, attack, release] = parameters;
    if ratio < 1.0 {
        return Err(NativeError::Message("Expander ratio must be at least one"));
    }
    let level = if peak > 0.0 {
        20.0 * peak.log10()
    } else {
        f64::NEG_INFINITY
    };
    let target = if ratio == 1.0 || level >= threshold {
        1.0
    } else {
        10_f64.powf(((ratio - 1.0) * (level - threshold)) / 20.0)
    };
    smooth(gain, target, attack, release, rate)
}

fn smooth(gain: f64, target: f64, attack: f64, release: f64, rate: f64) -> NativeResult<f64> {
    if attack < 0.0 || release < 0.0 || !attack.is_finite() || !release.is_finite() {
        return Err(NativeError::Message(
            "Dynamics attack/release must be nonnegative",
        ));
    }
    let seconds = if target > gain { attack } else { release };
    Ok(if seconds == 0.0 {
        target
    } else {
        gain + (target - gain) * -(-1.0 / (rate * seconds)).exp_m1()
    })
}
