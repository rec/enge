use numpy::{IntoPyArray, PyArray2, PyReadonlyArray2, PyUntypedArrayMethods};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

pub fn register(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_function(wrap_pyfunction!(rubberband_stretch, module)?)?;
    module.add_class::<RubberBandLiveShifter>()?;
    module.add_class::<RubberBandRealtimeStretcher>()?;
    Ok(())
}

struct Stretcher(*mut rubberband_sys::RubberBandState_);

impl Drop for Stretcher {
    fn drop(&mut self) {
        unsafe { rubberband_sys::rubberband_delete(self.0) };
    }
}

#[pyclass(unsendable)]
pub struct RubberBandRealtimeStretcher {
    state: Stretcher,
    buffers: Buffers,
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
            buffers: Buffers::new(channels, max_block_frames),
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
        self.buffers.load(input)?;
        unsafe {
            rubberband_sys::rubberband_process(
                self.state.0,
                self.buffers.input_pointers.as_ptr(),
                frames as u32,
                i32::from(final_block),
            );
        }
        self.ended = final_block;
        let mut output = Vec::new();
        loop {
            let available = unsafe { rubberband_sys::rubberband_available(self.state.0) };
            if available <= 0 {
                break;
            }
            let count = (available as usize).min(self.max_block_frames);
            let retrieved = unsafe {
                rubberband_sys::rubberband_retrieve(
                    self.state.0,
                    self.buffers.output_pointers.as_ptr(),
                    count as u32,
                )
            } as usize;
            if retrieved == 0 {
                break;
            }
            let skip = self.remaining_delay.min(retrieved);
            self.remaining_delay -= skip;
            self.buffers.append(&mut output, skip, retrieved);
        }
        let frames = output.len() / self.channels;
        Ok(
            numpy::ndarray::Array2::from_shape_vec((frames, self.channels), output)
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
    buffers: Buffers,
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
        if sample_rate == 0
            || channels == 0
            || channels > u32::MAX as usize
            || !pitch.is_finite()
            || pitch <= 0.0
        {
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
            buffers: Buffers::new(channels, block_size),
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
        self.buffers.load(input)?;
        unsafe {
            rubberband_sys::rubberband_live_shift(
                self.state,
                self.buffers.input_pointers.as_ptr(),
                self.buffers.output_pointers.as_ptr(),
            )
        };
        let mut interleaved = Vec::with_capacity(self.block_size * self.channels);
        self.buffers.append(&mut interleaved, 0, self.block_size);
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
    let output_frames = (frames as f64 * ratio).round_ties_even();
    if frames == 0 || frames > u32::MAX as usize || channels == 0 || channels > u32::MAX as usize {
        return Err(PyValueError::new_err(
            "Rubber Band requires nonempty audio within C API frame/channel limits",
        ));
    }
    if !output_frames.is_finite()
        || output_frames < 1.0
        || output_frames >= (isize::MAX as usize / channels / size_of::<f64>()) as f64
    {
        return Err(PyValueError::new_err("Invalid Rubber Band output duration"));
    }
    let output_frames = output_frames as usize;
    let input = audio.as_slice()?;
    validate_audio(input)?;
    #[allow(
        clippy::unnecessary_cast,
        reason = "C enum signedness depends on the target"
    )]
    let options = (rubberband_sys::RubberBandOption_RubberBandOptionProcessOffline
        | rubberband_sys::RubberBandOption_RubberBandOptionEngineFiner
        | rubberband_sys::RubberBandOption_RubberBandOptionChannelsTogether
        | rubberband_sys::RubberBandOption_RubberBandOptionThreadingNever) as i32;
    let state =
        unsafe { rubberband_sys::rubberband_new(48_000, channels as u32, options, ratio, pitch) };
    if state.is_null() {
        return Err(PyValueError::new_err(
            "Rubber Band could not create a stretcher",
        ));
    }
    let state = Stretcher(state);
    let block_frames =
        4096.min(unsafe { rubberband_sys::rubberband_get_process_size_limit(state.0) } as usize);
    if block_frames == 0 {
        return Err(PyValueError::new_err(
            "Invalid Rubber Band process size limit",
        ));
    }
    let mut buffers = Buffers::new(channels, block_frames);
    unsafe {
        rubberband_sys::rubberband_set_expected_input_duration(state.0, frames as u32);
        rubberband_sys::rubberband_set_max_process_size(state.0, block_frames as u32);
    }
    for start in (0..frames).step_by(block_frames) {
        let end = (start + block_frames).min(frames);
        buffers.load(&input[start * channels..end * channels])?;
        unsafe {
            rubberband_sys::rubberband_study(
                state.0,
                buffers.input_pointers.as_ptr(),
                (end - start) as u32,
                i32::from(end == frames),
            );
        }
    }
    let mut output = Vec::with_capacity(output_frames * channels);
    for start in (0..frames).step_by(block_frames) {
        let end = (start + block_frames).min(frames);
        buffers.load(&input[start * channels..end * channels])?;
        unsafe {
            rubberband_sys::rubberband_process(
                state.0,
                buffers.input_pointers.as_ptr(),
                (end - start) as u32,
                i32::from(end == frames),
            );
        }
        loop {
            let available = unsafe { rubberband_sys::rubberband_available(state.0) };
            if available <= 0 {
                break;
            }
            let count = (available as usize).min(block_frames);
            let retrieved = unsafe {
                rubberband_sys::rubberband_retrieve(
                    state.0,
                    buffers.output_pointers.as_ptr(),
                    count as u32,
                )
            } as usize;
            if retrieved == 0 {
                return Err(PyValueError::new_err("Rubber Band output did not advance"));
            }
            buffers.append(&mut output, 0, retrieved);
        }
    }
    if output.len() < output_frames * channels {
        return Err(PyValueError::new_err(
            "Rubber Band output was shorter than the prepared duration",
        ));
    }
    output.truncate(output_frames * channels);
    Ok(
        numpy::ndarray::Array2::from_shape_vec((output_frames, channels), output)
            .map_err(|_| PyValueError::new_err("Rubber Band returned invalid audio"))?
            .into_pyarray(py),
    )
}

/// Planar conversion and retrieval memory remains stable for the owner's life.
/// The pointer vectors address the separately allocated inner Vec buffers.
struct Buffers {
    input: Vec<Vec<f32>>,
    output: Vec<Vec<f32>>,
    input_pointers: Vec<*const f32>,
    output_pointers: Vec<*mut f32>,
}

impl Buffers {
    fn new(channels: usize, frames: usize) -> Self {
        let input = vec![vec![0.0; frames]; channels];
        let mut output = vec![vec![0.0; frames]; channels];
        let input_pointers = input.iter().map(Vec::as_ptr).collect();
        let output_pointers = output.iter_mut().map(Vec::as_mut_ptr).collect();
        Self {
            input,
            output,
            input_pointers,
            output_pointers,
        }
    }

    fn load(&mut self, source: &[f64]) -> PyResult<()> {
        validate_audio(source)?;
        for (frame, values) in source.chunks_exact(self.input.len()).enumerate() {
            for (channel, value) in values.iter().enumerate() {
                self.input[channel][frame] = *value as f32;
            }
        }
        Ok(())
    }

    fn append(&self, target: &mut Vec<f64>, first: usize, last: usize) {
        for frame in first..last {
            for channel in &self.output {
                target.push(channel[frame] as f64);
            }
        }
    }
}

fn validate_audio(input: &[f64]) -> PyResult<()> {
    if input.iter().any(|value| !(*value as f32).is_finite()) {
        return Err(PyValueError::new_err(
            "Rubber Band audio must be finite and representable as float32",
        ));
    }
    Ok(())
}
