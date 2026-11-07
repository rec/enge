//! Prepared multi-input mixing and continuously retimed selector fades.

use crate::error::{NativeError, NativeResult};
use crate::runtime::{add_partial, sum_partials};
use numpy::ndarray::{ArrayView1, ArrayView2};

#[derive(Clone)]
pub(super) struct InputFade {
    pub(super) value: f64,
    pub(super) target: f64,
    pub(super) step: f64,
    pub(super) remaining: usize,
}

#[derive(Clone)]
pub(super) struct Mixer {
    pub(super) fades: Vec<InputFade>,
    pub(super) fade_frames: usize,
    gains: Vec<f64>,
    partials: Vec<f64>,
}

impl Mixer {
    pub(super) fn new(parameters: ArrayView1<'_, f64>, ports: usize) -> NativeResult<Self> {
        let fade_frames = parameters[0] as usize;
        if fade_frames == 0
            || parameters[0] >= usize::MAX as f64
            || fade_frames as f64 != parameters[0]
            || parameters.len() < 3 + 3 * ports
        {
            return Err(NativeError::Message("Invalid mixer profile"));
        }
        selectors(parameters, ports)?;
        let selected = (0..ports).any(|p| parameters[5 + 3 * p] == 1.0);
        Ok(Self {
            fades: (0..ports)
                .map(|p| {
                    let value = if parameters[4 + 3 * p] == 0.0
                        && (!selected || parameters[5 + 3 * p] == 1.0)
                    {
                        1.0
                    } else {
                        0.0
                    };
                    InputFade {
                        value,
                        target: value,
                        step: 0.0,
                        remaining: 0,
                    }
                })
                .collect(),
            fade_frames,
            gains: vec![0.0; ports],
            partials: vec![0.0; ports],
        })
    }

    pub(super) fn same_capacity(&self, other: &Self) -> bool {
        self.fades.len() == other.fades.len() && self.fade_frames == other.fade_frames
    }

    pub(super) fn process(
        &mut self,
        input: ArrayView2<'_, f64>,
        parameters: ArrayView1<'_, f64>,
        output: &mut [f64],
    ) -> NativeResult<()> {
        let ports = self.fades.len();
        selectors(parameters, ports)?;
        let selected = (0..ports).any(|p| parameters[5 + 3 * p] == 1.0);
        for (port, fade) in self.fades.iter_mut().enumerate() {
            let target = if parameters[4 + 3 * port] == 0.0
                && (!selected || parameters[5 + 3 * port] == 1.0)
            {
                1.0
            } else {
                0.0
            };
            if target != fade.target {
                fade.target = target;
                fade.remaining = self.fade_frames;
                fade.step = (target - fade.value) / self.fade_frames as f64;
            }
            self.gains[port] = 10_f64.powf(parameters[3 + 3 * port] / 20.0) * fade.value;
            if !self.gains[port].is_finite() {
                return Err(NativeError::Message("Non-finite mixer gain"));
            }
        }
        for (channel, target) in output.iter_mut().enumerate() {
            let mut count = 0;
            for port in 0..ports {
                let product = input[[port, channel]] * self.gains[port];
                if !product.is_finite() {
                    return Err(NativeError::Message("Non-finite mixer product"));
                }
                add_partial(&mut self.partials, &mut count, product);
            }
            *target = sum_partials(&self.partials[..count]);
            if !target.is_finite() {
                return Err(NativeError::Message("Non-finite mixer sum"));
            }
        }
        for fade in &mut self.fades {
            if fade.remaining > 0 {
                fade.remaining -= 1;
                fade.value = if fade.remaining == 0 {
                    fade.target
                } else {
                    fade.value + fade.step
                };
            }
        }
        Ok(())
    }
}

fn selectors(parameters: ArrayView1<'_, f64>, ports: usize) -> NativeResult<()> {
    if (0..ports).any(|p| {
        !matches!(parameters[4 + 3 * p], 0.0 | 1.0) || !matches!(parameters[5 + 3 * p], 0.0 | 1.0)
    }) {
        return Err(NativeError::Message("Mixer selectors must be zero or one"));
    }
    Ok(())
}
