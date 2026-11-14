//! Voice-owned amplitude policy over the standalone Envelope core.

use crate::error::{NativeError, NativeResult};
use envelfo_core::Status;
use envelfo_core::envelope::{
    Envelope, EnvelopeCommand, EnvelopeState, Initial, Phase, Playback, Retrigger, Segment, Start,
};
use envelfo_core::lfo::{Command, Value};
use envelfo_core::time::Time;

pub(crate) type AmplitudeSegments = Vec<(Option<(i64, i64)>, f64, f64)>;

#[derive(Clone, PartialEq)]
pub(crate) struct Amplitude {
    body: Envelope,
}

#[derive(Clone)]
pub(crate) struct AmplitudeState {
    state: EnvelopeState,
    ordinal: u64,
}

impl Amplitude {
    pub(crate) fn new(
        initial: f64,
        attack: AmplitudeSegments,
        release: AmplitudeSegments,
    ) -> NativeResult<Self> {
        let convert = |parts: AmplitudeSegments, level: f64| -> NativeResult<Vec<Segment>> {
            if parts.is_empty() {
                return Ok(vec![Segment {
                    approach: None,
                    duration: Some(Time::ZERO),
                    to: level,
                    curve: 0.0,
                }]);
            }
            parts
                .into_iter()
                .map(|(duration, to, curve)| {
                    Ok(Segment {
                        approach: None,
                        duration: duration
                            .map(|(n, d)| Time::new(n, d))
                            .transpose()
                            .map_err(error)?,
                        to,
                        curve,
                    })
                })
                .collect()
        };
        let body = Envelope {
            initial: Initial::Value(initial),
            segments: convert(attack, initial)?,
            release: convert(release, 0.0)?,
            bipolar: true,
            hold: true,
            retrigger: Retrigger::Current,
            start: Start::Activation,
            playback: Playback::Once,
            markers: vec![],
            loop_bounds: None,
            repeat_count: None,
        };
        body.validate().map_err(error)?;
        Ok(Self { body })
    }

    pub(crate) fn start(&self) -> NativeResult<AmplitudeState> {
        let mut state = self.body.initial(Time::ZERO).map_err(error)?;
        self.body
            .command(
                &mut state,
                Time::ZERO,
                0,
                EnvelopeCommand::Playback(Command::NoteOn),
            )
            .map_err(error)?;
        Ok(AmplitudeState { state, ordinal: 0 })
    }

    pub(crate) fn settle(
        &self,
        state: &mut AmplitudeState,
        release: Option<Time>,
        age: usize,
    ) -> NativeResult<()> {
        let age = frame_position(age)?;
        if let Some(at) = release.filter(|at| age >= *at && state.state.phase() == Phase::On) {
            self.send(state, at, EnvelopeCommand::Playback(Command::NoteOff))?;
        }
        Ok(())
    }

    pub(crate) fn command(
        &self,
        state: &mut AmplitudeState,
        age: usize,
        kind: usize,
        position: f64,
    ) -> NativeResult<()> {
        let command = match kind {
            0 => EnvelopeCommand::Playback(Command::Pause),
            1 => EnvelopeCommand::Playback(Command::Resume),
            2 => EnvelopeCommand::Playback(Command::Reverse),
            3 => EnvelopeCommand::Playback(Command::Seek(frame_time(position)?)),
            4 => EnvelopeCommand::Playback(Command::Shift(frame_time(position)?)),
            5 => EnvelopeCommand::Advance,
            _ => return Err(NativeError::Message("Invalid amplitude command")),
        };
        self.send(
            state,
            Time::from_integer(
                i64::try_from(age).map_err(|_| NativeError::Message("Amplitude age overflow"))?,
            ),
            command,
        )
    }

    pub(crate) fn value(&self, state: &AmplitudeState, age: usize) -> NativeResult<Value> {
        self.body
            .value(
                &state.state,
                Time::from_integer(
                    i64::try_from(age)
                        .map_err(|_| NativeError::Message("Amplitude age overflow"))?,
                ),
            )
            .map_err(error)
    }

    pub(crate) fn complete(&self, state: &AmplitudeState, value: &Value) -> bool {
        state.state.phase() == Phase::Release && value.status == Status::Complete
    }

    fn send(
        &self,
        state: &mut AmplitudeState,
        at: Time,
        command: EnvelopeCommand,
    ) -> NativeResult<()> {
        let ordinal = state
            .ordinal
            .checked_add(1)
            .ok_or(NativeError::Message("Amplitude ordinal overflow"))?;
        self.body
            .command(&mut state.state, at, ordinal, command)
            .map_err(error)?;
        state.ordinal = ordinal;
        Ok(())
    }
}

fn error(value: envelfo_core::Error) -> NativeError {
    NativeError::Amplitude(value)
}

pub(crate) fn frame_position(frame: usize) -> NativeResult<Time> {
    let frame =
        i64::try_from(frame).map_err(|_| NativeError::Message("Native EnveLFO frame overflow"))?;
    Ok(Time::from_integer(frame))
}

pub(crate) fn frame_time(value: f64) -> NativeResult<Time> {
    if !value.is_finite() {
        return Err(NativeError::Message("Invalid amplitude time"));
    }
    if value == 0.0 {
        return Ok(Time::ZERO);
    }
    let bits = value.abs().to_bits();
    if (bits >> 52) & 0x7ff == 0 {
        return Err(NativeError::Message("Amplitude time overflow"));
    }
    let mut mantissa = (bits & ((1_u64 << 52) - 1)) | (1_u64 << 52);
    let mut exponent = ((bits >> 52) & 0x7ff) as i32 - 1023 - 52;
    let zeros = mantissa.trailing_zeros();
    mantissa >>= zeros;
    exponent += zeros as i32;
    let sign = if value < 0.0 { -1_i128 } else { 1_i128 };
    let (numerator, denominator) = if exponent >= 0 {
        (
            i128::from(mantissa).checked_shl(exponent as u32),
            Some(1_i128),
        )
    } else {
        (
            Some(i128::from(mantissa)),
            1_i128.checked_shl((-exponent) as u32),
        )
    };
    let numerator = numerator
        .and_then(|n| n.checked_mul(sign))
        .and_then(|n| i64::try_from(n).ok())
        .ok_or(NativeError::Message("Amplitude time overflow"))?;
    let denominator = denominator
        .and_then(|d| i64::try_from(d).ok())
        .ok_or(NativeError::Message("Amplitude time overflow"))?;
    Time::new(numerator, denominator).map_err(error)
}
