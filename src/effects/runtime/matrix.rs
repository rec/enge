//! Stateless routing with one compensated partial per authored input channel.

use crate::error::{NativeError, NativeResult};
use crate::runtime::{add_partial, sum_partials};
use numpy::ndarray::{Array2, ArrayView1};

pub(super) struct RoutingMatrix {
    coefficients: Array2<f64>,
    partials: Vec<f64>,
}

impl RoutingMatrix {
    pub(super) fn new(coefficients: Array2<f64>) -> Self {
        Self {
            partials: vec![0.0; coefficients.ncols()],
            coefficients,
        }
    }

    pub(super) fn matches_parameters(&self, parameters: ArrayView1<'_, f64>) -> bool {
        parameters.len() >= self.coefficients.len() + 3
            && parameters[1] == 1.0
            && parameters[2] == 1.0
            && self
                .coefficients
                .iter()
                .zip(parameters.iter().skip(3))
                .all(|(a, b)| a == b)
    }

    pub(super) fn process(
        &mut self,
        input: ArrayView1<'_, f64>,
        output: &mut [f64],
    ) -> NativeResult<()> {
        for (row, target) in self.coefficients.rows().into_iter().zip(output) {
            let mut count = 0;
            for (coefficient, value) in row.iter().zip(input.iter()) {
                let product = coefficient * value;
                if !product.is_finite() {
                    return Err(NativeError::Message("Non-finite routing matrix product"));
                }
                add_partial(&mut self.partials, &mut count, product);
            }
            *target = sum_partials(&self.partials[..count]);
            if !target.is_finite() {
                return Err(NativeError::Message("Non-finite routing matrix sum"));
            }
        }
        Ok(())
    }
}
