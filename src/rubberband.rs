use numpy::{IntoPyArray, PyArray2, PyReadonlyArray2, PyUntypedArrayMethods};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

struct Stretcher(*mut rubberband_sys::RubberBandState_);

impl Drop for Stretcher {
    fn drop(&mut self) {
        unsafe { rubberband_sys::rubberband_delete(self.0) };
    }
}

#[pyfunction]
pub fn rubberband_stretch<'py>(
    py: Python<'py>,
    audio: PyReadonlyArray2<'_, f64>,
    ratio: f64,
    pitch: f64,
) -> PyResult<Bound<'py, PyArray2<f64>>> {
    if !audio.is_c_contiguous() || !ratio.is_finite() || ratio <= 0.0 || !pitch.is_finite() || pitch <= 0.0 {
        return Err(PyValueError::new_err("Rubber Band requires contiguous audio and positive finite ratios"));
    }
    let shape = audio.shape();
    let (frames, channels) = (shape[0], shape[1]);
    if channels == 0 {
        return Err(PyValueError::new_err("Rubber Band requires at least one channel"));
    }
    let input = audio.as_slice()?;
    let planar: Vec<Vec<f32>> = (0..channels)
        .map(|channel| (0..frames).map(|frame| input[frame * channels + channel] as f32).collect())
        .collect();
    let pointers: Vec<*const f32> = planar.iter().map(Vec::as_ptr).collect();
    let state = unsafe {
        rubberband_sys::rubberband_new(
            48_000,
            channels as u32,
            (rubberband_sys::RubberBandOption_RubberBandOptionProcessOffline
                | rubberband_sys::RubberBandOption_RubberBandOptionEngineFiner)
                as i32,
            ratio,
            pitch,
        )
    };
    if state.is_null() {
        return Err(PyValueError::new_err("Rubber Band could not create a stretcher"));
    }
    let state = Stretcher(state);
    unsafe {
        rubberband_sys::rubberband_set_expected_input_duration(state.0, frames as u32);
        rubberband_sys::rubberband_study(state.0, pointers.as_ptr(), frames as u32, 1);
        rubberband_sys::rubberband_process(state.0, pointers.as_ptr(), frames as u32, 1);
    }
    let mut output = vec![Vec::<f32>::new(); channels];
    loop {
        let available = unsafe { rubberband_sys::rubberband_available(state.0) };
        if available <= 0 {
            break;
        }
        let count = available as usize;
        let mut block = vec![vec![0.0; count]; channels];
        let pointers: Vec<*mut f32> = block.iter_mut().map(Vec::as_mut_ptr).collect();
        let retrieved = unsafe { rubberband_sys::rubberband_retrieve(state.0, pointers.as_ptr(), count as u32) } as usize;
        for channel in 0..channels {
            output[channel].extend_from_slice(&block[channel][..retrieved]);
        }
    }
    let frames = output[0].len();
    let mut interleaved = Vec::with_capacity(frames * channels);
    for frame in 0..frames {
        for channel in 0..channels {
            interleaved.push(output[channel][frame] as f64);
        }
    }
    Ok(numpy::ndarray::Array2::from_shape_vec((frames, channels), interleaved)
        .map_err(|_| PyValueError::new_err("Rubber Band returned invalid audio"))?
        .into_pyarray(py))
}
