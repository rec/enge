//! Prepared capture regions, chronological history and ordered delayed commands.

use crate::error::{NativeError, NativeResult};
use numpy::ndarray::Array2;
use std::collections::VecDeque;

const FADE: usize = 64;

#[derive(Clone, Copy, PartialEq)]
pub(super) enum Mode {
    Reverse,
    Scrub,
    Stutter,
}

#[derive(Clone, Copy)]
pub(super) struct Command {
    pub(super) frame: usize,
    pub(super) release: bool,
}

#[derive(Clone)]
pub(super) struct Region {
    pub(super) samples: Array2<f64>,
    pub(super) cursor: usize,
    pub(super) active: bool,
    pub(super) gain: f64,
    pub(super) target: f64,
    pub(super) step: f64,
    pub(super) remaining: usize,
}

impl Region {
    fn fade(&mut self, target: f64) {
        self.target = target;
        self.step = (target - self.gain) / FADE as f64;
        self.remaining = FADE;
        if target == 0.0 && self.gain == 0.0 {
            self.active = false;
        }
    }
}

pub(super) struct CapturedBuffer {
    pub(super) history: Array2<f64>,
    pub(super) write: usize,
    pub(super) count: usize,
    pub(super) regions: [Region; 2],
    pub(super) pending: VecDeque<Command>,
    pub(super) mode: Mode,
    pub(super) delay: usize,
    pub(super) capacity: usize,
}

impl Clone for CapturedBuffer {
    fn clone(&self) -> Self {
        let mut pending = VecDeque::with_capacity(self.capacity);
        pending.extend(self.pending.iter().copied());
        Self {
            history: self.history.clone(),
            write: self.write,
            count: self.count,
            regions: self.regions.clone(),
            pending,
            mode: self.mode,
            delay: self.delay,
            capacity: self.capacity,
        }
    }

    fn clone_from(&mut self, source: &Self) {
        self.history.assign(&source.history);
        self.write = source.write;
        self.count = source.count;
        for (target, source) in self.regions.iter_mut().zip(&source.regions) {
            target.samples.assign(&source.samples);
            target.cursor = source.cursor;
            target.active = source.active;
            target.gain = source.gain;
            target.target = source.target;
            target.step = source.step;
            target.remaining = source.remaining;
        }
        self.pending.clear();
        self.pending.extend(source.pending.iter().copied());
        self.mode = source.mode;
        self.delay = source.delay;
        self.capacity = source.capacity;
    }
}

impl CapturedBuffer {
    pub(super) fn new(
        frames: usize,
        channels: usize,
        mode: Mode,
        delay: usize,
        capacity: usize,
    ) -> Self {
        Self {
            history: Array2::zeros((frames, channels)),
            write: 0,
            count: 0,
            regions: std::array::from_fn(|_| Region {
                samples: Array2::zeros((frames, channels)),
                cursor: 0,
                active: false,
                gain: 0.0,
                target: 0.0,
                step: 0.0,
                remaining: 0,
            }),
            pending: VecDeque::with_capacity(capacity),
            mode,
            delay,
            capacity,
        }
    }

    pub(super) fn same_capacity(&self, other: &Self) -> bool {
        self.history.raw_dim() == other.history.raw_dim()
            && self.mode == other.mode
            && self.delay == other.delay
            && self.capacity == other.capacity
    }

    pub(super) fn enqueue(&mut self, frame: usize, release: bool) -> NativeResult<()> {
        if self.pending.len() == self.capacity {
            return Err(NativeError::Message(
                "Captured buffer command capacity exhausted",
            ));
        }
        let frame = frame
            .checked_add(self.delay)
            .ok_or(NativeError::Message("Capture command frame overflow"))?;
        self.pending.push_back(Command { frame, release });
        Ok(())
    }

    pub(super) fn process(
        &mut self,
        wet: &mut [f64],
        position: f64,
        frame: usize,
        ended: bool,
    ) -> NativeResult<bool> {
        if !(0.0..=1.0).contains(&position) || wet.iter().any(|v| !v.is_finite()) {
            return Err(NativeError::Message(
                "Invalid captured buffer input or position",
            ));
        }
        let frames = self.history.nrows();
        while self.pending.front().is_some_and(|c| c.frame <= frame) {
            let command = self.pending.pop_front().expect("pending capture command");
            if command.release {
                for region in &mut self.regions {
                    region.fade(0.0);
                }
            } else {
                if ended || self.count < frames {
                    return Err(NativeError::Message(
                        "Capture requires live input and a complete history window",
                    ));
                }
                let slot =
                    self.regions
                        .iter()
                        .position(|r| !r.active)
                        .ok_or(NativeError::Message(
                            "Captured buffer regions occupied by a crossfade",
                        ))?;
                for region in &mut self.regions {
                    region.fade(0.0);
                }
                let region = &mut self.regions[slot];
                for i in 0..frames {
                    for channel in 0..wet.len() {
                        region.samples[[i, channel]] =
                            self.history[[(self.write + i) % frames, channel]];
                    }
                }
                region.cursor = 0;
                region.active = true;
                region.gain = 0.0;
                region.fade(1.0);
            }
        }
        let drained = ended && self.pending.is_empty() && self.regions.iter().all(|r| !r.active);
        for (channel, value) in wet.iter_mut().enumerate() {
            let source = *value;
            if !ended {
                self.history[[self.write, channel]] = source;
            }
            for region in &self.regions {
                if !region.active {
                    continue;
                }
                let mut weight = region.gain;
                let sample = match self.mode {
                    Mode::Reverse => {
                        weight *= ((frames - 1 - region.cursor) as f64 / FADE as f64).min(1.0);
                        region.samples[[frames - 1 - region.cursor, channel]]
                    }
                    Mode::Scrub => {
                        let index = position * (frames - 1) as f64;
                        let lower = index.floor() as usize;
                        let a = region.samples[[lower, channel]];
                        let b = region.samples[[(lower + 1).min(frames - 1), channel]];
                        a + (index - lower as f64) * (b - a)
                    }
                    Mode::Stutter => {
                        let offset = region.cursor;
                        let a = region.samples[[offset, channel]];
                        if offset < FADE {
                            let b = region.samples[[frames - FADE + offset, channel]];
                            let mix = offset as f64 / FADE as f64;
                            b * (1.0 - mix) + a * mix
                        } else {
                            a
                        }
                    }
                };
                *value += weight * (sample - source);
            }
            if !value.is_finite() {
                return Err(NativeError::Message("Non-finite captured buffer output"));
            }
        }
        if !ended {
            self.write = (self.write + 1) % frames;
            self.count = (self.count + 1).min(frames);
        }
        for region in &mut self.regions {
            if !region.active {
                continue;
            }
            if self.mode != Mode::Scrub {
                region.cursor += 1;
                if self.mode == Mode::Stutter {
                    region.cursor %= frames - FADE;
                } else if region.cursor == frames {
                    region.active = false;
                }
            }
            if region.remaining > 0 {
                region.remaining -= 1;
                region.gain = if region.remaining == 0 {
                    region.target
                } else {
                    region.gain + region.step
                };
            }
            if region.gain == 0.0 && region.target == 0.0 {
                region.active = false;
            }
        }
        Ok(drained)
    }
}
