//! Linked peak compression shared by offline and native callback rendering.

use crate::error::{NativeError, NativeResult};

pub(crate) fn gain(
    peak: f64,
    parameters: [f64; 6],
    reduction: f64,
    rate: f64,
) -> NativeResult<(f64, f64)> {
    let [threshold, ratio, knee, attack, release, makeup] = parameters;
    if !peak.is_finite()
        || peak < 0.0
        || !reduction.is_finite()
        || reduction > 0.0
        || !rate.is_finite()
        || rate <= 0.0
        || parameters.iter().any(|value| !value.is_finite())
        || ratio < 1.0
        || knee < 0.0
        || attack < 0.0
        || release < 0.0
    {
        return Err(NativeError::Message(
            "Invalid compressor parameters or state",
        ));
    }
    let target = if peak == 0.0 {
        0.0
    } else {
        let x = 20.0 * peak.log10() - threshold;
        if x <= -knee / 2.0 {
            0.0
        } else if x >= knee / 2.0 {
            (1.0 / ratio - 1.0) * x
        } else {
            (0.5 * (1.0 / ratio - 1.0) * knee) * (x / knee + 0.5).powi(2)
        }
    };
    let time = if target < reduction { attack } else { release };
    let reduction = if time == 0.0 {
        target
    } else {
        target + (-1.0 / (rate * time)).exp() * (reduction - target)
    };
    let gain = 10_f64.powf((reduction + makeup) / 20.0);
    if !reduction.is_finite() || !gain.is_finite() {
        return Err(NativeError::Message("Non-finite compressor gain"));
    }
    Ok((gain, reduction))
}
