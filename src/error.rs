//! Errors retained without heap storage until returning to the Python boundary.
use pyo3::PyErr;
use pyo3::exceptions::PyValueError;

#[derive(Clone, Copy, Debug)]
pub(crate) enum NativeError {
    Message(&'static str),
    Core(envelfo_core::Error),
    Amplitude(envelfo_core::Error),
    FilterOutput(usize),
    CallbackCapacity(&'static str),
}

pub(crate) type NativeResult<T> = Result<T, NativeError>;

impl From<NativeError> for PyErr {
    fn from(error: NativeError) -> Self {
        match error {
            NativeError::Message(message) => PyValueError::new_err(message),
            NativeError::Core(error) => PyValueError::new_err(format!("Native EnveLFO: {error}")),
            NativeError::Amplitude(error) => {
                PyValueError::new_err(format!("Amplitude Envelope: {error:?}"))
            }
            NativeError::FilterOutput(frame) => PyValueError::new_err(format!(
                "Non-finite filter output or state at frame {frame}"
            )),
            NativeError::CallbackCapacity(subject) => {
                PyValueError::new_err(format!("Live {subject} actions exceed callback capacity"))
            }
        }
    }
}
