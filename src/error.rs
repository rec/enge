//! Errors retained without heap storage until returning to the Python boundary.
use pyo3::PyErr;
use pyo3::exceptions::PyValueError;
use ufor::modulation::math::CalculationError;

#[derive(Clone, Copy, Debug)]
pub(crate) enum NativeError {
    Message(&'static str),
    Core(engelfo::Error),
    Amplitude(engelfo::Error),
    FilterOutput(usize),
    CallbackCapacity(&'static str),
    Calculation(CalculationError),
}

pub(crate) type NativeResult<T> = Result<T, NativeError>;

impl From<CalculationError> for NativeError {
    fn from(error: CalculationError) -> Self {
        Self::Calculation(error)
    }
}

impl From<NativeError> for PyErr {
    fn from(error: NativeError) -> Self {
        match error {
            NativeError::Calculation(error) => {
                PyValueError::new_err(format!("Modulation: {error}"))
            }
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
