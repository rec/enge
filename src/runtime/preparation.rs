//! Preparation-only conversion from validated uFor definitions to engine plans.
pub(crate) mod bindings;
pub(crate) mod modulation;
pub(crate) mod patch;

use super::{LfoDefinition, NamedEnvelopeDefinition, StagedEnvelfoDefinition, core_error};
use crate::amplitude::{Amplitude, frame_time};
use engelfo::envelope;
use engelfo::lfo;
use engelfo::stages;
use engelfo::time::Time;
use numpy::{PyReadonlyArray2, PyUntypedArrayMethods};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use ufor::codec::{self, Format};
use ufor::control::{Polarity, Scope};
use ufor::envelfo::{self, Body, Clock, Generator, PreparationLimits};

pub(super) fn definitions(
    lfos: &[String],
    envelopes: &[String],
    maximum: usize,
) -> PyResult<(Vec<Generator>, Vec<Generator>)> {
    let limits = PreparationLimits::new(maximum).map_err(invalid)?;
    let mut count = 1_usize
        .checked_add(lfos.len())
        .and_then(|n| n.checked_add(envelopes.len()))
        .ok_or_else(|| invalid("Generator preparation size overflow"))?;
    let mut characters = 0_usize;
    let mut decode = |texts: &[String]| -> PyResult<Vec<Generator>> {
        texts
            .iter()
            .map(|text| {
                characters = characters
                    .checked_add(text.chars().count())
                    .ok_or_else(|| invalid("Generator preparation text overflow"))?;
                if characters > codec::MAX_SCORE_CHARACTERS {
                    return Err(invalid(
                        "Generator preparation exceeds document character limit",
                    ));
                }
                let remaining = maximum
                    .checked_sub(count)
                    .ok_or_else(|| invalid("Generator preparation exceeds record limit"))?;
                let value = codec::parse_generator(
                    text,
                    Format::Json,
                    PreparationLimits::new(remaining).map_err(invalid)?,
                )
                .map_err(invalid)?;
                count += envelfo::preparation_records(
                    &serde_json::to_value(&value).map_err(invalid)?,
                    limits,
                )
                .map_err(invalid)?;
                Ok(value)
            })
            .collect()
    };
    let lfos = decode(lfos)?;
    let envelopes = decode(envelopes)?;
    if count > maximum {
        return Err(invalid("Generator preparation exceeds record limit"));
    }
    Ok((lfos, envelopes))
}

pub(crate) fn amplitudes(
    documents: &[String],
    sample_rate: f64,
    maximum: usize,
) -> PyResult<Vec<Amplitude>> {
    let (_, values) = definitions(&[], documents, maximum)?;
    values
        .iter()
        .map(|generator| {
            let Body::Envelope(value) = generator.body() else {
                return Err(invalid("Voice amplitude requires an Envelope"));
            };
            if generator.settings().scope() != Scope::Voice
                || generator.settings().clock() != Clock::Seconds
                || value.start() != envelfo::Start::Activation
                || !matches!(value.initial(), envelfo::Initial::Number(_))
                || !value.hold()
                || value.playback() != envelfo::PlaybackMode::Once
                || !value.markers().is_empty()
                || value.release().is_empty()
                || value
                    .segments()
                    .iter()
                    .any(|s| matches!(s, envelfo::EnvelopeSegment::Approach(_)))
            {
                return Err(invalid("Unsupported voice amplitude Envelope profile"));
            }
            Ok(Amplitude::new(envelope(value, false, sample_rate)?)?)
        })
        .collect()
}

pub(super) fn lfos(
    definitions: &[Generator],
    routes: PyReadonlyArray2<'_, f64>,
    sample_rate: f64,
    parameter_count: usize,
) -> PyResult<Vec<LfoDefinition>> {
    if routes.shape() != [definitions.len(), 4] || routes.as_array().iter().any(|v| !v.is_finite())
    {
        return Err(invalid("Invalid LFO route metadata"));
    }
    definitions
        .iter()
        .zip(routes.as_array().rows())
        .map(|(generator, row)| {
            let Body::Lfo(value) = generator.body() else {
                return Err(invalid("LFO slot requires a literal LFO"));
            };
            let parameter = row[0] as usize;
            let operation = row[1] as usize;
            if row[0] != parameter as f64
                || parameter >= parameter_count
                || row[1] != operation as f64
                || operation > 1
            {
                return Err(invalid("Invalid LFO route metadata"));
            }
            let scope = match generator.settings().scope() {
                Scope::Instrument => 0,
                Scope::Part => 1,
                Scope::Voice => 3,
            };
            let beat = generator.settings().clock() == Clock::Beats;
            let mut body = lfo(value, beat, sample_rate)?;
            body.clock = if beat {
                lfo::Clock::Beats
            } else {
                lfo::Clock::Seconds
            };
            // The audio host owns activation/reset policy for each source scope.
            body.reset = if scope == 3 {
                lfo::Reset::Trigger
            } else {
                lfo::Reset::Free
            };
            if generator.settings().position_driver() == envelfo::PositionDriver::Transport {
                body.driver = lfo::Driver::Transport;
                body.reset = lfo::Reset::Transport;
            }
            body.validate().map_err(core_error)?;
            Ok(LfoDefinition {
                scope,
                body,
                parameter,
                operation,
                intercept: row[2],
                slope: row[3],
            })
        })
        .collect()
}

pub(super) fn envelopes(
    definitions: &[Generator],
    routes: Vec<(usize, usize, bool)>,
    sample_rate: f64,
    parameter_count: usize,
) -> PyResult<Vec<NamedEnvelopeDefinition>> {
    if definitions.len() != routes.len() {
        return Err(invalid("Invalid named Envelope route metadata"));
    }
    definitions
        .iter()
        .zip(routes)
        .map(|(generator, (parameter, operation, release_with_voice))| {
            if parameter >= parameter_count
                || operation > 1
                || generator.settings().scope() != Scope::Voice
            {
                return Err(invalid("Invalid named Envelope route metadata or scope"));
            }
            let beat_clock = generator.settings().clock() == Clock::Beats;
            let (body, sample_hold) = match generator.body() {
                Body::Envelope(value) => {
                    if !matches!(value.initial(), envelfo::Initial::Number(_)) {
                        return Err(invalid("Named Envelope requires an explicit initial value"));
                    }
                    (envelope(value, beat_clock, sample_rate)?, None)
                }
                // SampleHold retains its established Envelope-bank slot and host seed setup.
                Body::SampleHold(value) => (
                    envelope::Envelope {
                        initial: envelope::Initial::Value(0.0),
                        segments: vec![envelope::Segment {
                            approach: None,
                            duration: Some(Time::ZERO),
                            to: 0.0,
                            curve: 0.0,
                        }],
                        release: vec![],
                        bipolar: true,
                        hold: true,
                        retrigger: envelope::Retrigger::Current,
                        start: envelope::Start::Event,
                        playback: envelope::Playback::Once,
                        markers: vec![],
                        loop_bounds: None,
                        repeat_count: None,
                    },
                    Some((0, value.minimum(), value.maximum())),
                ),
                _ => {
                    return Err(invalid(
                        "Named Envelope slot requires an Envelope or SampleHold",
                    ));
                }
            };
            body.validate().map_err(core_error)?;
            let event_started = body.start == envelope::Start::Event;
            Ok(NamedEnvelopeDefinition {
                sample_hold,
                body,
                parameter,
                operation,
                release_with_voice,
                beat_clock,
                event_started,
            })
        })
        .collect()
}

pub(super) fn staged(
    inputs: Vec<(String, Option<usize>, usize)>,
    sample_rate: f64,
    parameter_count: usize,
    has_beat_clock: bool,
    maximum: usize,
) -> PyResult<Vec<StagedEnvelfoDefinition>> {
    let (documents, routes): (Vec<_>, Vec<_>) = inputs
        .into_iter()
        .map(|(document, parameter, operation)| (document, (parameter, operation)))
        .unzip();
    let (_, generators) = definitions(&[], &documents, maximum)?;
    generators
        .iter()
        .zip(routes)
        .map(|(generator, (parameter, operation))| {
            let Body::Stages(value) = generator.body() else {
                return Err(invalid("Stages slot requires a Stages definition"));
            };
            let beat_clock = generator.settings().clock() == Clock::Beats;
            if generator.settings().scope() != Scope::Voice
                || parameter.is_some_and(|p| p >= parameter_count)
                || operation > 1
                || (beat_clock && !has_beat_clock)
            {
                return Err(invalid("Invalid Stages route metadata, scope or clock"));
            }
            // Prepared stage names are indices, preserving the existing snapshot contract.
            let index = |name: &ufor::identifier::Identifier| {
                value
                    .stages()
                    .iter()
                    .position(|s| s.name() == name)
                    .expect("validated Stages reference")
                    .to_string()
            };
            let bodies = value
                .stages()
                .iter()
                .enumerate()
                .map(|(i, stage)| {
                    let (body, markers) = match stage.envelfo().body() {
                        Body::Envelope(value) => {
                            let mut body = envelope(value, beat_clock, sample_rate)?;
                            body.markers = markers(value.markers())?;
                            body.bipolar = true;
                            body.hold = true;
                            body.retrigger = envelope::Retrigger::Current;
                            (stages::Body::Envelope(body), Vec::new())
                        }
                        Body::Lfo(value) => {
                            let mut body = lfo(value, beat_clock, sample_rate)?;
                            body.center = value.center();
                            body.depth = value.depth();
                            (stages::Body::LFO(body), markers(value.markers())?)
                        }
                        _ => unreachable!("validated Stages child"),
                    };
                    Ok(stages::Stage {
                        name: i.to_string(),
                        body,
                        markers,
                    })
                })
                .collect::<PyResult<Vec<_>>>()?;
            let mut transitions = Vec::new();
            for transition in value.transitions() {
                for source in transition.from_stages() {
                    transitions.push(stages::Transition {
                        from: vec![index(source)],
                        event: transition.event().to_owned(),
                        action: match transition.action() {
                            envelfo::StageAction::Finish => stages::Action::Finish,
                            envelfo::StageAction::Enter { stage } => {
                                stages::Action::Enter(index(stage))
                            }
                        },
                    });
                }
            }
            let body = stages::Stages {
                stages: bodies,
                initial_stage: index(value.initial_stage()),
                transitions,
            };
            body.validate().map_err(core_error)?;
            Ok(StagedEnvelfoDefinition {
                body,
                parameter,
                operation,
                beat_clock,
            })
        })
        .collect()
}

fn lfo(value: &envelfo::Lfo, beat: bool, sample_rate: f64) -> PyResult<lfo::LFO> {
    let envelfo::Rate::Literal(rate) = value.rate() else {
        return Err(invalid("LFO public parameters must be materialized"));
    };
    let scale = if beat {
        Time::ONE
    } else {
        frame_time(sample_rate)?
    };
    let mut body =
        lfo::LFO::new(exact(*rate)?.checked_div(scale).map_err(core_error)?).map_err(core_error)?;
    body.shape = match value.shape() {
        envelfo::Waveform::Sine => lfo::Shape::Sine,
        envelfo::Waveform::Square => lfo::Shape::Square,
        envelfo::Waveform::Triangle => lfo::Shape::Triangle,
    };
    body.duty = exact(value.duty_cycle())?;
    body.phase = exact(value.phase())?;
    body.delay = exact(value.delay())?
        .checked_mul(scale)
        .map_err(core_error)?;
    body.fade = exact(value.fade_in())?
        .checked_mul(scale)
        .map_err(core_error)?;
    Ok(body)
}

fn markers(values: &[envelfo::Marker]) -> PyResult<Vec<envelope::Marker>> {
    values
        .iter()
        .map(|m| {
            Ok(envelope::Marker {
                name: m.name().as_str().to_owned(),
                position: exact(m.position())?,
            })
        })
        .collect()
}

fn envelope(
    value: &envelfo::Envelope,
    beat: bool,
    sample_rate: f64,
) -> PyResult<envelope::Envelope> {
    let scale = if beat {
        Time::ONE
    } else {
        frame_time(sample_rate)?
    };
    let bound = |name: Option<&ufor::identifier::Identifier>, fallback: Time| -> PyResult<Time> {
        match name {
            None => Ok(fallback),
            Some(name) => exact(
                value
                    .markers()
                    .iter()
                    .find(|m| m.name() == name)
                    .expect("validated Envelope loop marker")
                    .position(),
            ),
        }
    };
    let start = bound(value.loop_start(), Time::ZERO)?;
    let end = bound(value.loop_end(), Time::ONE)?;
    let loop_bounds = if start == Time::ZERO && end == Time::ONE {
        None
    } else {
        Some((start, end))
    };
    Ok(envelope::Envelope {
        initial: match value.initial() {
            envelfo::Initial::Number(value) => envelope::Initial::Value(*value),
            envelfo::Initial::Current => envelope::Initial::Current,
        },
        segments: segments(value.segments(), scale)?,
        release: segments(value.release(), scale)?,
        bipolar: value.polarity() == Polarity::Bipolar,
        hold: value.hold(),
        retrigger: match value.retrigger() {
            envelfo::Retrigger::Current => envelope::Retrigger::Current,
            envelfo::Retrigger::Reset => envelope::Retrigger::Initial,
            envelfo::Retrigger::Ignore => envelope::Retrigger::Ignore,
        },
        start: if value.start() == envelfo::Start::Event {
            envelope::Start::Event
        } else {
            envelope::Start::Activation
        },
        playback: match value.playback() {
            envelfo::PlaybackMode::Once => envelope::Playback::Once,
            envelfo::PlaybackMode::Loop => envelope::Playback::Loop,
            envelfo::PlaybackMode::PingPong => envelope::Playback::PingPong,
        },
        markers: loop_bounds
            .map(|(start, end)| {
                vec![
                    envelope::Marker {
                        name: "loop_start".into(),
                        position: start,
                    },
                    envelope::Marker {
                        name: "loop_end".into(),
                        position: end,
                    },
                ]
            })
            .unwrap_or_default(),
        loop_bounds,
        repeat_count: value.repeat_count().map(|v| v as u64),
    })
}

fn segments(values: &[envelfo::EnvelopeSegment], scale: Time) -> PyResult<Vec<envelope::Segment>> {
    values
        .iter()
        .map(|value| {
            Ok(match value {
                envelfo::EnvelopeSegment::Definite(v) => envelope::Segment {
                    approach: None,
                    duration: Some(
                        exact(v.duration())?
                            .checked_mul(scale)
                            .map_err(core_error)?,
                    ),
                    to: v.to().value(),
                    curve: v.curve(),
                },
                envelfo::EnvelopeSegment::Indefinite => envelope::Segment {
                    approach: None,
                    duration: None,
                    to: 0.0,
                    curve: 0.0,
                },
                envelfo::EnvelopeSegment::Approach(v) => envelope::Segment {
                    duration: Some(Time::ZERO),
                    to: v.transition_to(),
                    curve: 0.0,
                    approach: Some(engelfo::approach::ExponentialApproach {
                        clock_units_per_second: scale,
                        asymptote: v.asymptote(),
                        time_constant: v.time_constant(),
                        threshold: v.threshold(),
                        transition_to: v.transition_to(),
                        transition_seconds: v.transition_seconds(),
                        minimum_transition_distance: v.minimum_transition_distance(),
                        minimum_time_constant: v.minimum_time_constant(),
                    }),
                },
            })
        })
        .collect()
}

fn exact(value: ufor::number::Rational) -> PyResult<Time> {
    Time::new(value.numerator(), value.denominator()).map_err(|e| core_error(e).into())
}

pub(super) fn invalid(error: impl std::fmt::Display) -> PyErr {
    PyValueError::new_err(format!("Generator preparation: {error}"))
}
