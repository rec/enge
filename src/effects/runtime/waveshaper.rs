//! Fixed 4x half-band oversampling and a latency-aligned dry path.

use crate::error::{NativeError, NativeResult};
use numpy::ndarray::Array2;

#[derive(Clone)]
pub(super) struct Waveshaper {
    pub(super) histories: [Array2<f64>; 4],
    pub(super) writes: [usize; 4],
    pub(super) dry_history: Array2<f64>,
    pub(super) dry_write: usize,
    pub(super) remaining: usize,
    pub(super) hard: bool,
    coefficients: [f64; 65],
}

impl Waveshaper {
    pub(super) fn new(channels: usize, hard: bool, coefficients: [f64; 65]) -> Self {
        Self {
            histories: std::array::from_fn(|_| Array2::zeros((65, channels))),
            writes: [0; 4],
            dry_history: Array2::zeros((48, channels)),
            dry_write: 0,
            remaining: 96,
            hard,
            coefficients,
        }
    }

    pub(super) fn same_capacity(&self, other: &Self) -> bool {
        self.histories[0].raw_dim() == other.histories[0].raw_dim()
            && self.hard == other.hard
            && self.coefficients == other.coefficients
    }

    pub(super) fn process(
        &mut self,
        dry: &mut [f64],
        wet: &mut [f64],
        drive: f64,
        bias: f64,
        output_db: f64,
        ended: bool,
    ) -> NativeResult<bool> {
        let gain = 10.0_f64.powf(output_db / 20.0);
        if !drive.is_finite() || drive <= 0.0 || !bias.is_finite() || !gain.is_finite() {
            return Err(NativeError::Message(
                "Invalid waveshaper drive, bias or output gain",
            ));
        }
        if wet.iter().any(|value| !value.is_finite()) {
            return Err(NativeError::Message("Non-finite waveshaper input"));
        }
        let offset = if self.hard {
            bias.clamp(-1.0, 1.0)
        } else {
            bias.tanh()
        };
        for channel in 0..dry.len() {
            let source = wet[channel];
            dry[channel] = self.dry_history[[self.dry_write, channel]];
            self.dry_history[[self.dry_write, channel]] = source;
            for i in 0..2 {
                let first = 2.0 * self.filter(0, channel, if i == 0 { source } else { 0.0 }, i)?;
                for j in 0..2 {
                    let second = 2.0
                        * self.filter(1, channel, if j == 0 { first } else { 0.0 }, 2 * i + j)?;
                    let value = drive * second + bias;
                    let shaped = (if self.hard {
                        value.clamp(-1.0, 1.0)
                    } else {
                        value.tanh()
                    }) - offset;
                    let third = self.filter(2, channel, shaped, 2 * i + j)?;
                    if j == 0 {
                        let fourth = self.filter(3, channel, third, i)?;
                        if i == 0 {
                            wet[channel] = fourth * gain;
                            if !wet[channel].is_finite() {
                                return Err(NativeError::Message("Non-finite waveshaper output"));
                            }
                        }
                    }
                }
            }
        }
        for (write, amount) in self.writes.iter_mut().zip([2, 4, 4, 2]) {
            *write = (*write + amount) % 65;
        }
        self.dry_write = (self.dry_write + 1) % 48;
        let drained = ended && self.remaining == 0;
        self.remaining = if ended {
            self.remaining.saturating_sub(1)
        } else {
            96
        };
        Ok(drained)
    }

    fn filter(
        &mut self,
        stage: usize,
        channel: usize,
        input: f64,
        offset: usize,
    ) -> NativeResult<f64> {
        if !input.is_finite() {
            return Err(NativeError::Message("Non-finite waveshaper FIR input"));
        }
        let write = (self.writes[stage] + offset) % 65;
        self.histories[stage][[write, channel]] = input;
        let output: f64 = self
            .coefficients
            .iter()
            .enumerate()
            .filter(|(_, coefficient)| **coefficient != 0.0)
            .map(|(i, coefficient)| {
                coefficient * self.histories[stage][[(write + 65 - i) % 65, channel]]
            })
            .sum();
        if !output.is_finite() {
            return Err(NativeError::Message("Non-finite waveshaper FIR output"));
        }
        Ok(output)
    }
}
