//! Linked sample-peak limiting with a prepared monotonic maximum queue.

use crate::error::{NativeError, NativeResult};
use numpy::ndarray::Array2;

#[derive(Clone)]
pub(super) struct Limiter {
    pub(super) history: Array2<f64>,
    pub(super) write: usize,
    pub(super) remaining: usize,
    pub(super) gain: f64,
    peaks: Vec<f64>,
    queue: Vec<usize>,
    head: usize,
    count: usize,
}

impl Limiter {
    pub(super) fn new(frames: usize, channels: usize) -> Self {
        Self {
            history: Array2::zeros((frames + 1, channels)),
            write: 0,
            remaining: frames,
            gain: 1.0,
            peaks: vec![0.0; frames + 1],
            queue: vec![0; frames + 1],
            head: 0,
            count: 0,
        }
    }

    fn push(&mut self, position: usize, peak: f64) {
        let capacity = self.queue.len();
        if self.count > 0 && self.queue[self.head] == position {
            self.head = (self.head + 1) % capacity;
            self.count -= 1;
        }
        while self.count > 0 {
            let back = (self.head + self.count - 1) % capacity;
            if self.peaks[self.queue[back]] > peak {
                break;
            }
            self.count -= 1;
        }
        self.peaks[position] = peak;
        self.queue[(self.head + self.count) % capacity] = position;
        self.count += 1;
    }

    pub(super) fn rebuild_queue(&mut self) {
        self.head = 0;
        self.count = 0;
        for i in 0..self.history.nrows() {
            let position = (self.write + i) % self.history.nrows();
            let peak = self
                .history
                .row(position)
                .iter()
                .map(|v| v.abs())
                .fold(0.0, f64::max);
            self.push(position, peak);
        }
    }

    pub(super) fn process(
        &mut self,
        dry: &mut [f64],
        wet: &mut [f64],
        ceiling_db: f64,
        release: f64,
        rate: f64,
        ended: bool,
    ) -> NativeResult<bool> {
        if !ceiling_db.is_finite() || ceiling_db > 0.0 || !release.is_finite() || release < 0.0 {
            return Err(NativeError::Message("Invalid limiter controls"));
        }
        if wet.iter().any(|v| !v.is_finite()) {
            return Err(NativeError::Message("Non-finite limiter input"));
        }
        let peak = wet.iter().map(|v| v.abs()).fold(0.0, f64::max);
        for (channel, value) in wet.iter().enumerate() {
            self.history[[self.write, channel]] = *value;
        }
        self.push(self.write, peak);
        self.write = (self.write + 1) % self.history.nrows();
        let peak = self.peaks[self.queue[self.head]];
        let ceiling = 10_f64.powf(ceiling_db / 20.0);
        let target = if peak == 0.0 {
            1.0
        } else {
            (ceiling / peak).min(1.0)
        };
        let released = if release == 0.0 {
            1.0
        } else {
            self.gain + (1.0 - self.gain) * -(-1.0 / (rate * release)).exp_m1()
        };
        self.gain = target.min(released);
        for (channel, value) in wet.iter_mut().enumerate() {
            dry[channel] = self.history[[self.write, channel]];
            *value = (dry[channel] * self.gain).clamp(-ceiling, ceiling);
        }
        let drained = ended && self.remaining == 0;
        self.remaining = if ended {
            self.remaining.saturating_sub(1)
        } else {
            self.history.nrows() - 1
        };
        Ok(drained)
    }
}
