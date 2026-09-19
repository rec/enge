//! noise-v1: counter-addressed SplitMix64 with shared filters and envelopes.
//! Algorithm: https://prng.di.unimi.it/splitmix64.c (public domain).
use crate::{envelope_amplitudes, envelope_spans, filters};
use numpy::ndarray::Array2;
use numpy::{
    IntoPyArray, PyArray2, PyArray3, PyReadonlyArray1, PyReadonlyArray2, PyUntypedArrayMethods,
};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

type RenderedNoise<'py> = (Bound<'py, PyArray2<f64>>, Bound<'py, PyArray3<f64>>);

#[pyfunction]
#[allow(clippy::too_many_arguments)]
pub fn render_noise<'py>(
    py: Python<'py>,
    key: u64,
    start: u128,
    rate: f64,
    gains: PyReadonlyArray1<'py, f64>,
    envelope: PyReadonlyArray2<'py, f64>,
    routes: PyReadonlyArray1<'py, f64>,
    filter_inputs: filters::Inputs<'py>,
) -> PyResult<RenderedNoise<'py>> {
    let count = gains.len();
    let limit = 1_u128 << 64;
    if start > limit
        || count as u128 > limit - start
        || routes.is_empty()
        || gains.as_array().iter().any(|v| !v.is_finite() || *v < 0.0)
        || routes.as_array().iter().any(|v| !v.is_finite())
    {
        return Err(PyValueError::new_err(
            "Invalid noise buffers or exhausted counter",
        ));
    }
    let gains: Vec<f64> = gains.as_array().iter().copied().collect();
    let routes: Vec<f64> = routes.as_array().iter().copied().collect();
    let spans = envelope_spans(envelope, count)?;
    let mut bank = filters::FilterBank::new(Some(filter_inputs), rate, count, 1)?;
    let (audio, memory) = py.detach(move || -> PyResult<_> {
        let amplitudes = envelope_amplitudes(&spans, count);
        let mut audio = Array2::zeros((count, routes.len()));
        for i in 0..count {
            let index = (start + i as u128) as u64;
            let mut word = key.wrapping_add(index.wrapping_add(1).wrapping_mul(0x9e3779b97f4a7c15));
            word = (word ^ (word >> 30)).wrapping_mul(0xbf58476d1ce4e5b9);
            word = (word ^ (word >> 27)).wrapping_mul(0x94d049bb133111eb);
            word ^= word >> 31;
            let mut source = [2.0 * ((word >> 11) as f64 / 9007199254740992.0) - 1.0];
            bank.process(i, &mut source)?;
            for (j, route) in routes.iter().enumerate() {
                audio[[i, j]] = (source[0] * (amplitudes[i] * gains[i])) * route;
            }
        }
        if audio.iter().any(|v| !v.is_finite()) {
            return Err(PyValueError::new_err("Non-finite noise output"));
        }
        Ok((audio, bank.states))
    })?;
    Ok((audio.into_pyarray(py), memory.into_pyarray(py)))
}
