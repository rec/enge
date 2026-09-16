//! LFO waveform and activation evaluation after exact boundary preparation.
use numpy::ndarray::Array2;
use numpy::{IntoPyArray, PyArray2, PyReadonlyArray2, PyUntypedArrayMethods};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use std::f64::consts::TAU;

#[pyfunction]
pub fn render_lfo<'py>(
    py: Python<'py>,
    waveform: u8,
    phases: PyReadonlyArray2<'py, f64>,
    envelope: PyReadonlyArray2<'py, f64>,
    frames: usize,
) -> PyResult<Bound<'py, PyArray2<f64>>> {
    if waveform > 2 || phases.shape()[1] != 5 || !phases.is_c_contiguous() {
        return Err(PyValueError::new_err("Invalid native LFO layout or shape"));
    }
    let phases = phases.as_slice()?.to_vec();
    let mut next = 0.0;
    for span in phases.chunks_exact(5) {
        if span.iter().any(|v| !v.is_finite())
            || span[0] != next
            || span[1] <= span[0]
            || span[1] > frames as f64
            || span[1] != span[1].floor()
            || !(0.0..=1.0).contains(&span[2])
            || !(0.0..=1.0).contains(&span[3])
            || ![0.0, 1.0].contains(&span[4])
        {
            return Err(PyValueError::new_err("Invalid native LFO phase span"));
        }
        next = span[1];
    }
    if next != frames as f64 {
        return Err(PyValueError::new_err("Incomplete native LFO spans"));
    }
    let envelope = super::envelope_spans(envelope, frames)?;
    let output = py.detach(move || {
        let weights = super::envelope_amplitudes(&envelope, frames);
        let mut output = Array2::zeros((frames, 2));
        for span in phases.chunks_exact(5) {
            for i in span[0] as usize..span[1] as usize {
                let phase = span[2] + (i as f64 - span[0]) * span[3];
                output[[i, 0]] = match waveform {
                    0 => (TAU * phase).sin(),
                    1 => {
                        if span[4] == 1.0 {
                            1.0
                        } else {
                            -1.0
                        }
                    }
                    _ if span[4] == 1.0 => 2.0 * phase - 1.0,
                    _ => 1.0 - 2.0 * phase,
                };
                // Preserve the declared source domains at floating-point edges.
                output[[i, 0]] = output[[i, 0]].clamp(-1.0, 1.0);
                output[[i, 1]] = weights[i].clamp(0.0, 1.0);
            }
        }
        output
    });
    Ok(output.into_pyarray(py))
}
