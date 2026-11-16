//! Voice-owned amplitude policy over the standalone Envelope core.

use crate::error::{NativeError, NativeResult};
use engelfo::Status;
use engelfo::approach::ExponentialApproach;
use engelfo::envelope::{
    Envelope, EnvelopeCommand, EnvelopeCommandStorage, EnvelopeState, Initial, Phase, Playback,
    Retrigger, Segment, Start,
};
use engelfo::lfo::{Command, Value};
use engelfo::time::Time;

pub(crate) type AmplitudeSegments = Vec<(Option<(i64, i64)>, f64, f64)>;
pub(crate) type ApproachParameters = (f64, f64, f64, f64, f64, f64, f64);

#[derive(Clone, PartialEq)]
pub(crate) struct Amplitude {
    body: Envelope,
    initial: EnvelopeState,
}

#[derive(Clone)]
pub(crate) struct AmplitudeState {
    state: EnvelopeState,
    storage: EnvelopeCommandStorage,
    ordinal: u64,
}

impl Amplitude {
    pub(crate) fn set_release_approach(
        &mut self,
        index: usize,
        parameters: ApproachParameters,
        rate: f64,
    ) -> NativeResult<()> {
        let (
            asymptote,
            time_constant,
            threshold,
            transition_to,
            transition_seconds,
            minimum_transition_distance,
            minimum_time_constant,
        ) = parameters;
        let part = self
            .body
            .release
            .get_mut(index)
            .ok_or(NativeError::Message("Unknown amplitude release stage"))?;
        part.approach = Some(ExponentialApproach {
            clock_units_per_second: frame_time(rate)?,
            asymptote,
            time_constant,
            threshold,
            transition_to,
            transition_seconds,
            minimum_transition_distance,
            minimum_time_constant,
        });
        self.body.validate().map_err(error)?;
        self.initial = initial_state(&self.body)?;
        Ok(())
    }

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
        let initial = initial_state(&body)?;
        Ok(Self { body, initial })
    }

    pub(crate) fn start(&self) -> NativeResult<AmplitudeState> {
        let mut state = self.initial.clone();
        let storage = self.body.prepare_commands(&mut state).map_err(error)?;
        Ok(AmplitudeState {
            state,
            storage,
            ordinal: 0,
        })
    }

    pub(crate) fn reset(&self, state: &mut AmplitudeState) {
        state.state.clone_from(&self.initial);
        state.ordinal = 0;
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
            .command_with_storage(&mut state.state, &mut state.storage, at, ordinal, command)
            .map_err(error)?;
        state.ordinal = ordinal;
        Ok(())
    }
}

fn initial_state(body: &Envelope) -> NativeResult<EnvelopeState> {
    let mut state = body.initial(Time::ZERO).map_err(error)?;
    body.command(
        &mut state,
        Time::ZERO,
        0,
        EnvelopeCommand::Playback(Command::NoteOn),
    )
    .map_err(error)?;
    Ok(state)
}

fn error(value: engelfo::Error) -> NativeError {
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
