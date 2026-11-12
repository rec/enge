//! Uniform, always-wet matrix FIR with one partition of latency.

use crate::error::{NativeError, NativeResult};
use crate::fft::RealFft;
use numpy::PyReadonlyArray1;
use numpy::ndarray::{Array2, Array3, ArrayView1};
use realfft::RealFftPlanner;
use realfft::num_complex::Complex;
use sha2::{Digest, Sha256};
use std::collections::{BTreeMap, BTreeSet};
use std::sync::Arc;

pub(crate) type Definitions<'py> = Vec<(usize, Vec<(usize, usize, PyReadonlyArray1<'py, f64>)>)>;

#[derive(Clone)]
pub(super) struct Convolution {
    pub(super) input: Array2<f64>,
    pub(super) output: Array2<f64>,
    pub(super) overlap: Array2<f64>,
    pub(super) spectra: Array3<Complex<f64>>,
    pub(super) fill: usize,
    pub(super) slot: usize,
    pub(super) input_ended: bool,
    pub(super) tail_remaining: usize,
    pub(super) tail_frames: usize,
    pub(super) partition: usize,
    kernels: Vec<Kernel>,
    fft: RealFft,
    compensation: Vec<Complex<f64>>,
}

#[derive(Clone)]
struct Kernel {
    input: usize,
    output: usize,
    impulse: Arc<Impulse>,
}

struct Impulse {
    frames: usize,
    digest: [u8; 32],
    spectra: Array2<Complex<f64>>,
}

impl Convolution {
    pub(super) fn new(
        planner: &mut RealFftPlanner<f64>,
        partition: usize,
        inputs: usize,
        outputs: usize,
        definitions: Vec<(usize, usize, PyReadonlyArray1<'_, f64>)>,
    ) -> NativeResult<Self> {
        let length = partition
            .checked_mul(2)
            .filter(|_| partition.is_power_of_two())
            .ok_or(NativeError::Message("Invalid convolution partition size"))?;
        if definitions.is_empty() {
            return Err(NativeError::Message("Convolution requires impulse kernels"));
        }
        let mut fft = RealFft::new(planner, length)?;
        let bins = partition + 1;
        let mut kernels = Vec::with_capacity(definitions.len());
        let mut pairs = BTreeSet::new();
        let mut cache: BTreeMap<[u8; 32], Arc<Impulse>> = BTreeMap::new();
        let mut maximum = 0;
        for (input, output, values) in definitions {
            let values = values.as_array();
            if input >= inputs
                || output >= outputs
                || values.is_empty()
                || values.iter().any(|v| !v.is_finite())
                || !pairs.insert((input, output))
            {
                return Err(NativeError::Message("Invalid convolution kernel"));
            }
            maximum = maximum.max(values.len());
            let mut hash = Sha256::new();
            for value in &values {
                hash.update(value.to_le_bytes());
            }
            let digest: [u8; 32] = hash.finalize().into();
            let impulse = if let Some(impulse) = cache.get(&digest) {
                impulse.clone()
            } else {
                let mut spectra = Array2::zeros((values.len().div_ceil(partition), bins));
                for part in 0..spectra.nrows() {
                    fft.real.fill(0.0);
                    for (index, value) in values
                        .iter()
                        .skip(part * partition)
                        .take(partition)
                        .enumerate()
                    {
                        fft.real[index] = *value;
                    }
                    fft.forward()?;
                    for bin in 0..bins {
                        spectra[[part, bin]] = fft.spectrum[bin];
                    }
                }
                let impulse = Arc::new(Impulse {
                    frames: values.len(),
                    digest,
                    spectra,
                });
                cache.insert(digest, impulse.clone());
                impulse
            };
            kernels.push(Kernel {
                input,
                output,
                impulse,
            });
        }
        let tail_frames = maximum
            .checked_add(partition - 1)
            .ok_or(NativeError::Message("Convolution tail capacity overflow"))?;
        Ok(Self {
            input: Array2::zeros((partition, inputs)),
            output: Array2::zeros((partition, outputs)),
            overlap: Array2::zeros((partition, outputs)),
            spectra: Array3::zeros((maximum.div_ceil(partition), inputs, bins)),
            fill: 0,
            slot: 0,
            input_ended: false,
            tail_remaining: 0,
            tail_frames,
            partition,
            kernels,
            fft,
            compensation: vec![Complex::new(0.0, 0.0); bins],
        })
    }

    pub(super) fn same_capacity(&self, other: &Self) -> bool {
        self.input.raw_dim() == other.input.raw_dim()
            && self.output.raw_dim() == other.output.raw_dim()
            && self.spectra.raw_dim() == other.spectra.raw_dim()
            && self.tail_frames == other.tail_frames
            && self.kernels.len() == other.kernels.len()
            && self.kernels.iter().zip(&other.kernels).all(|(a, b)| {
                a.input == b.input
                    && a.output == b.output
                    && a.impulse.frames == b.impulse.frames
                    && a.impulse.digest == b.impulse.digest
            })
    }

    pub(super) fn process(
        &mut self,
        input: ArrayView1<'_, f64>,
        output: &mut [f64],
        ended: bool,
    ) -> NativeResult<bool> {
        if self.input_ended && self.tail_remaining == 0 {
            output.fill(0.0);
            return Ok(true);
        }
        if ended && !self.input_ended {
            self.input_ended = true;
            self.tail_remaining = self.tail_frames;
        }
        for (channel, value) in input.iter().enumerate() {
            self.input[[self.fill, channel]] = if self.input_ended { 0.0 } else { *value };
        }
        for (channel, value) in output.iter_mut().enumerate() {
            *value = self.output[[self.fill, channel]];
        }
        self.fill += 1;
        if self.fill == self.partition {
            self.fill = 0;
            self.finish_block()?;
        }
        if self.input_ended {
            self.tail_remaining -= 1;
        }
        Ok(false)
    }

    fn finish_block(&mut self) -> NativeResult<()> {
        let bins = self.partition + 1;
        let parts = self.spectra.shape()[0];
        for channel in 0..self.input.ncols() {
            self.fft.real.fill(0.0);
            for frame in 0..self.partition {
                self.fft.real[frame] = self.input[[frame, channel]];
            }
            self.fft.forward()?;
            for bin in 0..bins {
                self.spectra[[self.slot, channel, bin]] = self.fft.spectrum[bin];
            }
        }
        for channel in 0..self.output.ncols() {
            self.fft.spectrum.fill(Complex::new(0.0, 0.0));
            self.compensation.fill(Complex::new(0.0, 0.0));
            for kernel in self.kernels.iter().filter(|k| k.output == channel) {
                for part in 0..kernel.impulse.spectra.nrows() {
                    let history = (self.slot + parts - part) % parts;
                    for bin in 0..bins {
                        let value = self.spectra[[history, kernel.input, bin]]
                            * kernel.impulse.spectra[[part, bin]];
                        add_compensated(
                            &mut self.fft.spectrum[bin],
                            &mut self.compensation[bin],
                            value,
                        );
                    }
                }
            }
            for bin in 0..bins {
                self.fft.spectrum[bin] += self.compensation[bin];
            }
            self.fft.inverse()?;
            for frame in 0..self.partition {
                let value = self.fft.real[frame] + self.overlap[[frame, channel]];
                if !value.is_finite() {
                    return Err(NativeError::Message("Non-finite convolution overlap"));
                }
                self.output[[frame, channel]] = value;
                self.overlap[[frame, channel]] = self.fft.real[frame + self.partition];
            }
        }
        self.slot = (self.slot + 1) % parts;
        Ok(())
    }
}

fn add_compensated(sum: &mut Complex<f64>, correction: &mut Complex<f64>, value: Complex<f64>) {
    let total = *sum + value;
    correction.re += if sum.re.abs() >= value.re.abs() {
        (sum.re - total.re) + value.re
    } else {
        (value.re - total.re) + sum.re
    };
    correction.im += if sum.im.abs() >= value.im.abs() {
        (sum.im - total.im) + value.im
    } else {
        (value.im - total.im) + sum.im
    };
    *sum = total;
}
