//! Linked peak analysis, independent of callback partitioning.

use crate::error::{NativeError, NativeResult};
use numpy::ndarray::ArrayView1;

pub(super) fn process(
    level: &mut f64,
    input: ArrayView1<'_, f64>,
    attack: f64,
    release: f64,
    rate: f64,
    floor: f64,
) -> NativeResult<()> {
    if !attack.is_finite() || !release.is_finite() || attack < 0.0 || release < 0.0 {
        return Err(NativeError::Message(
            "Follower attack/release must be nonnegative",
        ));
    }
    let peak = input.iter().map(|v| v.abs()).fold(0.0, f64::max);
    let seconds = if peak > *level { attack } else { release };
    *level = if seconds == 0.0 {
        peak
    } else {
        *level + (peak - *level) * -(-1.0 / (rate * seconds)).exp_m1()
    };
    if *level < floor {
        *level = 0.0;
    }
    Ok(())
}
