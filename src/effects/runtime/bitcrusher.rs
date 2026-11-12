//! Deterministic signed quantization and capture-latched integer holds.

use crate::error::{NativeError, NativeResult};

#[derive(Clone)]
pub(super) struct Bitcrusher {
    pub(super) held: Vec<f64>,
    pub(super) remaining: usize,
}

impl Bitcrusher {
    pub(super) fn new(channels: usize) -> Self {
        Self {
            held: vec![0.0; channels],
            remaining: 0,
        }
    }

    pub(super) fn process(
        &mut self,
        audio: &mut [f64],
        bits: f64,
        hold: f64,
        ended: bool,
    ) -> NativeResult<bool> {
        if !bits.is_finite()
            || !(1.0..=24.0).contains(&bits)
            || bits.fract() != 0.0
            || !hold.is_finite()
            || !(1.0..=65536.0).contains(&hold)
            || hold.fract() != 0.0
        {
            return Err(NativeError::Message(
                "Bitcrusher bits and hold_frames must be bounded integers",
            ));
        }
        if audio.iter().any(|v| !v.is_finite()) {
            return Err(NativeError::Message("Non-finite bitcrusher input"));
        }
        if self.remaining == 0 {
            if ended {
                audio.fill(0.0);
                return Ok(true);
            }
            let step = 2.0_f64.powi(1 - bits as i32);
            for (held, value) in self.held.iter_mut().zip(audio.iter()) {
                *held = ((value.clamp(-1.0, 1.0) / step).round() * step).clamp(-1.0, 1.0 - step);
            }
            self.remaining = hold as usize;
        }
        audio.copy_from_slice(&self.held);
        self.remaining -= 1;
        Ok(false)
    }
}
