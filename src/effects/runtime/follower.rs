//! Linked peak analysis, independent of callback partitioning.

use crate::error::{NativeError, NativeResult};
use numpy::ndarray::ArrayView1;

#[derive(Clone, Default)]
pub(super) struct Coefficients {
    smoothing: [Option<(f64, f64, f64)>; 2],
}

#[allow(clippy::too_many_arguments)]
pub(super) fn process(
    level: &mut f64,
    input: ArrayView1<'_, f64>,
    attack: f64,
    release: f64,
    rate: f64,
    floor: f64,
    coefficients: &mut Coefficients,
) -> NativeResult<()> {
    if !attack.is_finite() || !release.is_finite() || attack < 0.0 || release < 0.0 {
        return Err(NativeError::Message(
            "Follower attack/release must be nonnegative",
        ));
    }
    let peak = input.iter().map(|v| v.abs()).fold(0.0, f64::max);
    let increasing = peak > *level;
    let seconds = if increasing { attack } else { release };
    *level = if seconds == 0.0 {
        peak
    } else {
        let entry = &mut coefficients.smoothing[usize::from(!increasing)];
        if !entry.is_some_and(|(s, r, _)| s == seconds && r == rate) {
            *entry = Some((seconds, rate, -(-1.0 / (rate * seconds)).exp_m1()));
        }
        *level + (peak - *level) * entry.unwrap().2
    };
    if *level < floor {
        *level = 0.0;
    }
    Ok(())
}
