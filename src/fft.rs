//! Prepared real transforms with retained scratch and NumPy-compatible scaling.

use crate::error::{NativeError, NativeResult};
use realfft::num_complex::Complex;
use realfft::{ComplexToReal, RealFftPlanner, RealToComplex};
use std::sync::Arc;

#[derive(Clone)]
pub(crate) struct RealFft {
    pub(crate) real: Vec<f64>,
    pub(crate) spectrum: Vec<Complex<f64>>,
    forward: Arc<dyn RealToComplex<f64>>,
    inverse: Arc<dyn ComplexToReal<f64>>,
    scratch: Vec<Complex<f64>>,
}

impl RealFft {
    pub(crate) fn new(planner: &mut RealFftPlanner<f64>, length: usize) -> NativeResult<Self> {
        if length < 2 {
            return Err(NativeError::Message("Real FFT length must be at least two"));
        }
        let forward = planner.plan_fft_forward(length);
        let inverse = planner.plan_fft_inverse(length);
        let scratch =
            vec![Complex::new(0.0, 0.0); forward.get_scratch_len().max(inverse.get_scratch_len())];
        Ok(Self {
            real: forward.make_input_vec(),
            spectrum: forward.make_output_vec(),
            forward,
            inverse,
            scratch,
        })
    }

    pub(crate) fn forward(&mut self) -> NativeResult<()> {
        self.forward
            .process_with_scratch(&mut self.real, &mut self.spectrum, &mut self.scratch)
            .map_err(|_| NativeError::Message("Invalid forward real FFT"))?;
        if self
            .spectrum
            .iter()
            .any(|v| !v.re.is_finite() || !v.im.is_finite())
        {
            return Err(NativeError::Message("Non-finite FFT spectrum"));
        }
        Ok(())
    }

    pub(crate) fn inverse(&mut self) -> NativeResult<()> {
        if self
            .spectrum
            .iter()
            .any(|v| !v.re.is_finite() || !v.im.is_finite())
        {
            return Err(NativeError::Message("Non-finite convolution spectrum"));
        }
        self.inverse
            .process_with_scratch(&mut self.spectrum, &mut self.real, &mut self.scratch)
            .map_err(|_| NativeError::Message("Invalid inverse real FFT"))?;
        let scale = 1.0 / self.real.len() as f64;
        for value in &mut self.real {
            *value *= scale;
            if !value.is_finite() {
                return Err(NativeError::Message("Non-finite inverse FFT"));
            }
        }
        Ok(())
    }
}
