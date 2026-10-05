use numpy::{IntoPyArray, PyArray2, PyReadonlyArray2, PyUntypedArrayMethods};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

struct Stretcher(*mut rubberband_sys::RubberBandState_);

impl Drop for Stretcher {
    fn drop(&mut self) {
        unsafe { rubberband_sys::rubberband_delete(self.0) };
    }
}

#[pyclass(unsendable)]
pub struct RubberBandRealtimeStretcher {
    state: Stretcher,
    channels: usize,
    max_block_frames: usize,
    start_delay: usize,
    remaining_delay: usize,
    ended: bool,
}

#[pymethods]
impl RubberBandRealtimeStretcher {
    #[new]
    fn new(
        sample_rate: u32,
        channels: usize,
        ratio: f64,
        pitch: f64,
        max_block_frames: usize,
    ) -> PyResult<Self> {
        if sample_rate == 0
            || channels == 0
            || channels > u32::MAX as usize
            || !ratio.is_finite()
            || ratio <= 0.0
            || !pitch.is_finite()
            || pitch <= 0.0
            || max_block_frames == 0
            || max_block_frames > u32::MAX as usize
        {
            return Err(PyValueError::new_err(
                "Invalid real-time Rubber Band settings",
            ));
        }
        #[allow(
            clippy::unnecessary_cast,
            reason = "C enum signedness depends on the target"
        )]
        let options = rubberband_sys::RubberBandOption_RubberBandOptionProcessRealTime as i32;
        let state = unsafe {
            rubberband_sys::rubberband_new(sample_rate, channels as u32, options, ratio, pitch)
        };
        if state.is_null() {
            return Err(PyValueError::new_err(
                "Rubber Band could not create a stretcher",
            ));
        }
        let state = Stretcher(state);
        if max_block_frames
            > unsafe { rubberband_sys::rubberband_get_process_size_limit(state.0) } as usize
        {
            return Err(PyValueError::new_err(
                "Rubber Band block exceeds process size limit",
            ));
        }
        unsafe {
            rubberband_sys::rubberband_set_max_process_size(state.0, max_block_frames as u32);
        }
        let start_delay = unsafe { rubberband_sys::rubberband_get_start_delay(state.0) } as usize;
        let mut remaining_pad =
            unsafe { rubberband_sys::rubberband_get_preferred_start_pad(state.0) } as usize;
        let zero = vec![vec![0.0_f32; max_block_frames]; channels];
        let pointers: Vec<*const f32> = zero.iter().map(Vec::as_ptr).collect();
        while remaining_pad > 0 {
            let count = remaining_pad.min(max_block_frames);
            unsafe {
                rubberband_sys::rubberband_process(state.0, pointers.as_ptr(), count as u32, 0)
            };
            remaining_pad -= count;
        }
        Ok(Self {
            state,
            channels,
            max_block_frames,
            start_delay,
            remaining_delay: start_delay,
            ended: false,
        })
    }

    #[getter]
    fn start_delay(&self) -> usize {
        self.start_delay
    }

    fn process<'py>(
        &mut self,
        py: Python<'py>,
        audio: PyReadonlyArray2<'_, f64>,
        final_block: bool,
    ) -> PyResult<Bound<'py, PyArray2<f64>>> {
        if self.ended {
            return Err(PyValueError::new_err("Rubber Band stream is complete"));
        }
        if !audio.is_c_contiguous()
            || audio.shape()[1] != self.channels
            || audio.shape()[0] > self.max_block_frames
            || audio.shape()[0] == 0
        {
            return Err(PyValueError::new_err("Invalid real-time Rubber Band block"));
        }
        let frames = audio.shape()[0];
        let input = audio.as_slice()?;
        let planar: Vec<Vec<f32>> = (0..self.channels)
            .map(|channel| {
                (0..frames)
                    .map(|frame| input[frame * self.channels + channel] as f32)
                    .collect()
            })
            .collect();
        let pointers: Vec<*const f32> = planar.iter().map(Vec::as_ptr).collect();
        unsafe {
            rubberband_sys::rubberband_process(
                self.state.0,
                pointers.as_ptr(),
                frames as u32,
                i32::from(final_block),
            );
        }
        self.ended = final_block;
        let mut output = vec![Vec::<f32>::new(); self.channels];
        loop {
            let available = unsafe { rubberband_sys::rubberband_available(self.state.0) };
            if available <= 0 {
                break;
            }
            let mut block = vec![vec![0.0_f32; available as usize]; self.channels];
            let pointers: Vec<*mut f32> = block.iter_mut().map(Vec::as_mut_ptr).collect();
            let retrieved = unsafe {
                rubberband_sys::rubberband_retrieve(
                    self.state.0,
                    pointers.as_ptr(),
                    available as u32,
                )
            } as usize;
            if retrieved == 0 {
                break;
            }
            let skip = self.remaining_delay.min(retrieved);
            self.remaining_delay -= skip;
            for channel in 0..self.channels {
                output[channel].extend_from_slice(&block[channel][skip..retrieved]);
            }
        }
        let frames = output[0].len();
        let mut interleaved = Vec::with_capacity(frames * self.channels);
        for frame in 0..frames {
            for channel in &output {
                interleaved.push(channel[frame] as f64);
            }
        }
        Ok(
            numpy::ndarray::Array2::from_shape_vec((frames, self.channels), interleaved)
                .map_err(|_| PyValueError::new_err("Rubber Band returned invalid audio"))?
                .into_pyarray(py),
        )
    }
}

#[pyclass(unsendable)]
pub struct RubberBandLiveShifter {
    state: *mut rubberband_sys::RubberBandLiveState_,
    channels: usize,
    block_size: usize,
}

impl Drop for RubberBandLiveShifter {
    fn drop(&mut self) {
        unsafe { rubberband_sys::rubberband_live_delete(self.state) };
    }
}

#[pymethods]
impl RubberBandLiveShifter {
    #[new]
    fn new(sample_rate: u32, channels: usize, pitch: f64) -> PyResult<Self> {
        if sample_rate == 0 || channels == 0 || !pitch.is_finite() || pitch <= 0.0 {
            return Err(PyValueError::new_err(
                "Rubber Band requires positive sample rate, channels, and pitch",
            ));
        }
        let state = unsafe { rubberband_sys::rubberband_live_new(sample_rate, channels as u32, 0) };
        if state.is_null() {
            return Err(PyValueError::new_err(
                "Rubber Band could not create a live shifter",
            ));
        }
        unsafe { rubberband_sys::rubberband_live_set_pitch_scale(state, pitch) };
        let block_size = unsafe { rubberband_sys::rubberband_live_get_block_size(state) as usize };
        Ok(Self {
            state,
            channels,
            block_size,
        })
    }

    #[getter]
    fn block_size(&self) -> usize {
        self.block_size
    }

    #[getter]
    fn start_delay(&self) -> usize {
        unsafe { rubberband_sys::rubberband_live_get_start_delay(self.state) as usize }
    }

    fn shift<'py>(
        &mut self,
        py: Python<'py>,
        audio: PyReadonlyArray2<'_, f64>,
    ) -> PyResult<Bound<'py, PyArray2<f64>>> {
        if !audio.is_c_contiguous() || audio.shape() != [self.block_size, self.channels] {
            return Err(PyValueError::new_err(
                "Rubber Band live input must match its block size and channels",
            ));
        }
        let input = audio.as_slice()?;
        let planar: Vec<Vec<f32>> = (0..self.channels)
            .map(|channel| {
                (0..self.block_size)
                    .map(|frame| input[frame * self.channels + channel] as f32)
                    .collect()
            })
            .collect();
        let pointers: Vec<*const f32> = planar.iter().map(Vec::as_ptr).collect();
        let mut output = vec![vec![0.0; self.block_size]; self.channels];
        let mutable_pointers: Vec<*mut f32> = output.iter_mut().map(Vec::as_mut_ptr).collect();
        unsafe {
            rubberband_sys::rubberband_live_shift(
                self.state,
                pointers.as_ptr(),
                mutable_pointers.as_ptr(),
            )
        };
        let mut interleaved = Vec::with_capacity(self.block_size * self.channels);
        for frame in 0..self.block_size {
            for channel in &output {
                interleaved.push(channel[frame] as f64);
            }
        }
        Ok(
            numpy::ndarray::Array2::from_shape_vec((self.block_size, self.channels), interleaved)
                .map_err(|_| PyValueError::new_err("Rubber Band returned invalid audio"))?
                .into_pyarray(py),
        )
    }
}

#[pyfunction]
pub fn rubberband_stretch<'py>(
    py: Python<'py>,
    audio: PyReadonlyArray2<'_, f64>,
    ratio: f64,
    pitch: f64,
) -> PyResult<Bound<'py, PyArray2<f64>>> {
    if !audio.is_c_contiguous()
        || !ratio.is_finite()
        || ratio <= 0.0
        || !pitch.is_finite()
        || pitch <= 0.0
    {
        return Err(PyValueError::new_err(
            "Rubber Band requires contiguous audio and positive finite ratios",
        ));
    }
    let shape = audio.shape();
    let (frames, channels) = (shape[0], shape[1]);
    if channels == 0 {
        return Err(PyValueError::new_err(
            "Rubber Band requires at least one channel",
        ));
    }
    let input = audio.as_slice()?;
    let planar: Vec<Vec<f32>> = (0..channels)
        .map(|channel| {
            (0..frames)
                .map(|frame| input[frame * channels + channel] as f32)
                .collect()
        })
        .collect();
    let pointers: Vec<*const f32> = planar.iter().map(Vec::as_ptr).collect();
    #[allow(
        clippy::unnecessary_cast,
        reason = "C enum signedness depends on the target"
    )]
    let options = (rubberband_sys::RubberBandOption_RubberBandOptionProcessOffline
        | rubberband_sys::RubberBandOption_RubberBandOptionEngineFiner) as i32;
    let state =
        unsafe { rubberband_sys::rubberband_new(48_000, channels as u32, options, ratio, pitch) };
    if state.is_null() {
        return Err(PyValueError::new_err(
            "Rubber Band could not create a stretcher",
        ));
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
        let retrieved = unsafe {
            rubberband_sys::rubberband_retrieve(state.0, pointers.as_ptr(), count as u32)
        } as usize;
        for channel in 0..channels {
            output[channel].extend_from_slice(&block[channel][..retrieved]);
        }
    }
    let frames = output[0].len();
    let mut interleaved = Vec::with_capacity(frames * channels);
    for frame in 0..frames {
        for channel in &output {
            interleaved.push(channel[frame] as f64);
        }
    }
    Ok(
        numpy::ndarray::Array2::from_shape_vec((frames, channels), interleaved)
            .map_err(|_| PyValueError::new_err("Rubber Band returned invalid audio"))?
            .into_pyarray(py),
    )
}
