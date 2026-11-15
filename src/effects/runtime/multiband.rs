//! Fixed LR4 split, phase-compensated low branch and three linked compressors.

use crate::compressor;
use crate::error::{NativeError, NativeResult};
use crate::filters::Coefficients;
use numpy::ndarray::{Array3, ArrayView1};

#[derive(Clone)]
pub(super) struct Multiband {
    coefficients: [[f64; 5]; 2],
    floor: f64,
    bands: [Vec<f64>; 3],
    smoothing: [compressor::Coefficients; 3],
    pub(super) integrators: Array3<f64>,
    pub(super) reductions: [f64; 3],
}

impl Multiband {
    pub(super) fn new(
        channels: usize,
        lower: f64,
        upper: f64,
        floor: f64,
        threshold: f64,
        rate: f64,
    ) -> NativeResult<Self> {
        if !(lower.is_finite()
            && upper.is_finite()
            && 0.0 < lower
            && lower < upper
            && upper < rate / 2.0
            && floor.is_finite()
            && 0.0 < floor
            && floor < threshold)
        {
            return Err(NativeError::Message("Invalid multiband crossover profile"));
        }
        let q = 1.0 / 2_f64.sqrt();
        let coefficients = [
            Coefficients::default().get(lower, q, rate),
            Coefficients::default().get(upper, q, rate),
        ];
        if coefficients
            .iter()
            .any(|v| !(0.0 < v[2] && v[2] < 1.0 && v[3] > 0.0 && 0.0 < v[4] && v[4] < 1.0))
        {
            return Err(NativeError::Message(
                "Unrepresentable multiband crossover coefficients",
            ));
        }
        Ok(Self {
            coefficients,
            floor,
            bands: std::array::from_fn(|_| vec![0.0; channels]),
            smoothing: std::array::from_fn(|_| compressor::Coefficients::default()),
            integrators: Array3::zeros((9, channels, 2)),
            reductions: [0.0; 3],
        })
    }

    pub(super) fn process(
        &mut self,
        audio: &mut [f64],
        parameters: ArrayView1<'_, f64>,
        rate: f64,
    ) -> NativeResult<()> {
        for (channel, value) in audio.iter().enumerate() {
            if !value.is_finite() {
                return Err(NativeError::Message("Non-finite multiband input"));
            }
            let low = self.stage(0, channel, *value, 0, 0)?;
            let low = self.stage(1, channel, low, 0, 0)?;
            let upper = self.stage(2, channel, *value, 0, 1)?;
            let upper = self.stage(3, channel, upper, 0, 1)?;
            let mid = self.stage(4, channel, upper, 1, 0)?;
            self.bands[1][channel] = self.stage(5, channel, mid, 1, 0)?;
            let high = self.stage(6, channel, upper, 1, 1)?;
            self.bands[2][channel] = self.stage(7, channel, high, 1, 1)?;
            self.bands[0][channel] = self.stage(8, channel, low, 1, 2)?;
        }
        audio.fill(0.0);
        for band in 0..3 {
            let start = 7 + 6 * band;
            let values = std::array::from_fn(|p| parameters[start + p]);
            let peak = self.bands[band].iter().map(|v| v.abs()).fold(0.0, f64::max);
            let (gain, reduction) = compressor::gain(
                peak,
                values,
                self.reductions[band],
                rate,
                &mut self.smoothing[band],
            )?;
            self.reductions[band] = reduction;
            for (value, input) in audio.iter_mut().zip(&self.bands[band]) {
                *value += gain * input;
            }
        }
        if audio.iter().any(|v| !v.is_finite()) {
            return Err(NativeError::Message("Non-finite multiband output"));
        }
        Ok(())
    }

    pub(super) fn drained(&self, threshold: f64) -> bool {
        self.integrators.iter().all(|v| v.abs() < threshold)
    }

    fn stage(
        &mut self,
        index: usize,
        channel: usize,
        input: f64,
        crossover: usize,
        response: u8,
    ) -> NativeResult<f64> {
        let [g, d, a1, a2, a3] = self.coefficients[crossover];
        let s1 = self.integrators[[index, channel, 0]];
        let s2 = self.integrators[[index, channel, 1]];
        let v3 = input - s2;
        let v1 = a1 * s1 + a2 * v3;
        let v2 = s2 + a2 * s1 + a3 * v3;
        let band = s1 / d + (g / d) * v3;
        for (state, value) in [2.0 * v1 - s1, 2.0 * v2 - s2].into_iter().enumerate() {
            if !value.is_finite() {
                return Err(NativeError::Message("Non-finite multiband filter state"));
            }
            self.integrators[[index, channel, state]] =
                if value.abs() < self.floor { 0.0 } else { value };
        }
        Ok(match response {
            0 => v2,
            1 => input - band - v2,
            _ => input - 2.0 * band,
        })
    }
}
