//! Shared prepared generator state and modulation for native source engines.
use super::controls::ControlBank;
use super::patch_host::{PatchHost, clock_at};
use super::{
    BeatClockInput, NamedEnvelopeInput, PatchEventInput, StageConnectionInput, StageStartInput,
    StagedEnvelfoInput, initial_lfo, lfo_definitions, ratio_time, valid_envelfo_owners,
};
use super::{
    BeatPoint, EnvelfoTransformInput, EnvelfoTransformRoute, LfoDefinition, LfoPlayback,
    NamedEnvelopeDefinition, PatchEventConnection, PatchStageStartConnection, RouteMap,
    StagedConnection, StagedEnvelfoDefinition, add_partial, control_time, core_error, lfo_command,
};
use crate::amplitude::AmplitudeSegments;
use crate::amplitude::{frame_position, frame_time};
use envelfo_core::envelope::{
    Envelope, Initial, Marker, Playback, Retrigger, Segment as CoreSegment, Start,
};
use envelfo_core::lfo::Command;
use envelfo_core::patch::{ChildCommand, PatchState, PatchStorage};
use envelfo_core::stages;
use envelfo_core::{lfo, time::Time};
use numpy::PyReadonlyArray2;
use pyo3::{exceptions::PyValueError, prelude::*};

#[pyclass(skip_from_py_object)]
#[derive(Clone)]
pub(crate) struct GeneratorBank {
    pub(super) rate: f64,
    pub(super) parameter_count: usize,
    pub(super) patch_host: Option<PatchHost>,
    pub(super) patch_states: Vec<Option<PatchState>>,
    pub(super) patch_storage: Vec<Option<PatchStorage>>,
    pub(super) pending_releases: Vec<Vec<Option<Time>>>,
    pub(super) envelfo_transforms: Vec<EnvelfoTransformInput>,
    pub(super) envelfo_transform_routes: Vec<EnvelfoTransformRoute>,
    pub(super) envelfo_transform_values: Vec<f64>,
    pub(super) lfo_definitions: Vec<LfoDefinition>,
    pub(super) lfo_owners: Vec<usize>,
    pub(super) lfo_event_states: Vec<LfoPlayback>,
    pub(super) named_envelopes: Vec<NamedEnvelopeDefinition>,
    pub(super) named_owners: Vec<usize>,
    pub(super) patch_events: Vec<PatchEventConnection>,
    pub(super) patch_stage_starts: Vec<PatchStageStartConnection>,
    pub(super) envelfo_keys: Vec<u64>,
    pub(super) beat_points: Vec<BeatPoint>,
    pub(super) staged_envelfos: Vec<StagedEnvelfoDefinition>,
    pub(super) staged_owners: Vec<usize>,
    pub(super) staged_connections: Vec<StagedConnection>,
}

impl GeneratorBank {
    pub(super) fn release_voice(
        &mut self,
        voice: usize,
        frame: usize,
        age: usize,
        minimum_hold_frames: Time,
    ) -> PyResult<Time> {
        for source in 0..self.staged_envelfos.len() {
            if self.staged_owners[source] == source {
                self.patch_command(
                    voice,
                    2,
                    source,
                    frame_position(frame)?,
                    ChildCommand::Playback(Command::NoteOff),
                )?;
            }
        }
        let age = frame_position(age)?;
        let frame = frame_position(frame)?;
        let release_frame = age.max(minimum_hold_frames);
        for (index, definition) in self.named_envelopes.iter().enumerate() {
            if definition.body.release.is_empty() {
                continue;
            }
            let named_release_at = if definition.release_with_voice {
                frame
                    .checked_add(release_frame.checked_sub(age).map_err(core_error)?)
                    .map_err(core_error)?
            } else {
                frame
            };
            if self.named_owners[index] == index {
                self.pending_releases[voice][index] = Some(named_release_at);
            }
        }

        Ok(release_frame)
    }

    pub(super) fn apply_action(
        &mut self,
        action: &[f64],
        active: &[bool],
        frame: usize,
        rate: f64,
    ) -> PyResult<()> {
        if action.len() != 6 {
            return Err(PyValueError::new_err("Invalid generator action"));
        }
        let kind = action[1] as usize;
        let voice = action[2] as usize;
        if action[1] != kind as f64
            || action[2] != voice as f64
            || (kind != 8 && voice >= active.len())
        {
            return Err(PyValueError::new_err("Invalid generator action"));
        }
        match kind {
            12 => {
                if action[3] != action[3] as u32 as f64
                    || action[4] != action[4] as u32 as f64
                    || action[5] != 0.0
                {
                    return Err(PyValueError::new_err("Invalid Envelfo stream key"));
                }
                self.envelfo_keys[voice] = action[3] as u64 | ((action[4] as u64) << 32);
            }
            8 => {
                let lfo = voice;
                let action_kind = action[3] as usize;
                if lfo >= self.lfo_definitions.len()
                    || action[3] != action_kind as f64
                    || action_kind > 6
                    || !action[4].is_finite()
                    || (action_kind != 1
                        && action_kind != 5
                        && action_kind != 6
                        && action[4] != 0.0)
                    || (action_kind == 1 && action[4] < 0.0)
                    || (action_kind == 5 && !(0.0..=1.0).contains(&action[4]))
                {
                    return Err(PyValueError::new_err("Invalid synth LFO action"));
                }
                let definition = &self.lfo_definitions[lfo];
                if definition.scope != 0 || definition.body.driver == lfo::Driver::Transport {
                    return Err(PyValueError::new_err("Invalid synth LFO scope"));
                }
                let at = self.lfo_time(definition, frame_position(frame)?)?;
                let command = match action_kind {
                    0 => Command::Reset,
                    1 => Command::Rate(if definition.body.clock == lfo::Clock::Beats {
                        control_time(action[4])?
                    } else {
                        control_time(action[4])?
                            .checked_div(frame_time(rate)?)
                            .map_err(core_error)?
                    }),
                    2 => Command::Pause,
                    3 => Command::Resume,
                    4 => Command::Reverse,
                    5 => Command::Seek(control_time(action[4])?),
                    6 => Command::Shift(control_time(action[4])?),
                    _ => unreachable!(),
                };
                lfo_command(definition, &mut self.lfo_event_states[lfo], at, command)?;
            }
            9 => {
                let source = action[3] as usize;
                let command = action[4] as usize;
                if !active[voice]
                    || action[3] != source as f64
                    || source >= self.named_envelopes.len()
                    || action[4] != command as f64
                    || command > 5
                    || (command == 3 && !(0.0..=1.0).contains(&action[5]))
                    || (command == 4 && !action[5].is_finite())
                    || (command != 3 && command != 4 && action[5] != 0.0)
                {
                    return Err(PyValueError::new_err("Invalid named Envelfo action"));
                }
                if self.named_envelopes[source].sample_hold.is_some() {
                    return Ok(());
                }
                let command = match command {
                    0 => ChildCommand::Playback(Command::Pause),
                    1 => ChildCommand::Playback(Command::Resume),
                    2 => ChildCommand::Playback(Command::Reverse),
                    3 => ChildCommand::Playback(Command::Seek(frame_time(action[5])?)),
                    4 => ChildCommand::Playback(Command::Shift(frame_time(action[5])?)),
                    5 => ChildCommand::Advance,
                    _ => unreachable!(),
                };
                self.patch_command(voice, 1, source, frame_position(frame)?, command)?;
            }

            10 => {
                let source = action[3] as usize;
                let command = action[4] as usize;
                if !active[voice]
                    || action[3] != source as f64
                    || source >= self.staged_envelfos.len()
                    || action[4] != command as f64
                    || command > 5
                    || (command == 3 && !(0.0..=1.0).contains(&action[5]))
                    || (command == 4 && !action[5].is_finite())
                    || (command != 3 && command != 4 && action[5] != 0.0)
                {
                    return Err(PyValueError::new_err("Invalid staged Envelfo action"));
                }
                let command = match command {
                    0 => ChildCommand::Playback(Command::Pause),
                    1 => ChildCommand::Playback(Command::Resume),
                    2 => ChildCommand::Playback(Command::Reverse),
                    3 => ChildCommand::Playback(Command::Seek(frame_time(action[5])?)),
                    4 => ChildCommand::Playback(Command::Shift(frame_time(action[5])?)),
                    5 => ChildCommand::Advance,
                    _ => unreachable!(),
                };
                self.patch_command(voice, 2, source, frame_position(frame)?, command)?;
            }
            11 => {
                let source = action[3] as usize;
                let command = action[4] as usize;
                if !active[voice]
                    || action[3] != source as f64
                    || source >= self.lfo_definitions.len()
                    || self.lfo_definitions[source].scope != 3
                    || self.lfo_definitions[source].body.driver == lfo::Driver::Transport
                    || action[4] != command as f64
                    || command > 4
                    || (command == 3 && !(0.0..=1.0).contains(&action[5]))
                    || (command == 4 && !action[5].is_finite())
                    || (command != 3 && command != 4 && action[5] != 0.0)
                {
                    return Err(PyValueError::new_err("Invalid voice LFO action"));
                }
                let command = match command {
                    0 => Command::Pause,
                    1 => Command::Resume,
                    2 => Command::Reverse,
                    3 => Command::Seek(control_time(action[5])?),
                    4 => Command::Shift(control_time(action[5])?),
                    _ => unreachable!(),
                };
                self.patch_command(
                    voice,
                    0,
                    source,
                    frame_position(frame)?,
                    ChildCommand::Playback(command),
                )?;
            }

            _ => return Err(PyValueError::new_err("Unknown generator action")),
        }
        Ok(())
    }

    pub(super) fn lfo_time(&self, definition: &LfoDefinition, frame: Time) -> PyResult<Time> {
        if definition.body.driver == lfo::Driver::Transport {
            clock_at(&self.beat_points, 2, frame).map_err(core_error)
        } else if definition.body.clock == lfo::Clock::Beats {
            clock_at(&self.beat_points, 1, frame).map_err(core_error)
        } else {
            Ok(frame)
        }
    }

    pub(super) fn lfo_signal(
        &self,
        voice: usize,
        source: usize,
        frame: usize,
        controls: &ControlBank,
    ) -> PyResult<(f64, f64)> {
        let definition = &self.lfo_definitions[source];
        if definition.scope == 1 {
            controls.context_kind(controls.voice_part_contexts[voice], 1)?;
        }
        let value = if definition.scope == 3 {
            self.patch_value(voice, 0, source, frame)?
        } else {
            let state = &self.lfo_event_states[self.lfo_owners[source]];
            definition
                .body
                .value(
                    &state.core,
                    self.lfo_time(definition, frame_position(frame)?)?,
                )
                .map_err(core_error)?
        };
        Ok((value.value, value.weight))
    }

    #[allow(clippy::too_many_arguments)]
    pub(super) fn contributions(
        &self,
        voice: usize,
        parameter: usize,
        mut product: f64,
        frame: usize,
        controls: &ControlBank,
        maps: &[Vec<RouteMap>],
        partials: &mut [f64],
        count: &mut usize,
    ) -> PyResult<f64> {
        for (source, definition) in self.lfo_definitions.iter().enumerate() {
            if definition.parameter != parameter {
                continue;
            }
            let (value, weight) = self.lfo_signal(voice, source, frame, controls)?;
            let amount = mapped(
                maps,
                1,
                source,
                definition.intercept + definition.slope * value,
            )?;
            if definition.operation == 0 {
                add_partial(partials, count, weight * amount);
            } else {
                product *= 1.0 + weight * (amount - 1.0);
            }
        }
        for (source, definition) in self.named_envelopes.iter().enumerate() {
            if definition.parameter != parameter {
                continue;
            }
            let value = self.patch_value(voice, 1, source, frame)?.value;
            let amount = mapped(maps, 2, source, value)?;
            if definition.operation == 0 {
                add_partial(partials, count, amount);
            } else {
                product *= amount;
            }
        }
        for (source, definition) in self.staged_envelfos.iter().enumerate() {
            if definition.parameter != Some(parameter) {
                continue;
            }
            let value = self.patch_value(voice, 2, source, frame)?.value;
            let amount = mapped(maps, 3, source, value)?;
            if definition.operation == 0 {
                add_partial(partials, count, amount);
            } else {
                product *= amount;
            }
        }
        for (index, (source, target, operation)) in self.envelfo_transform_routes.iter().enumerate()
        {
            if *target != parameter {
                continue;
            }
            let amount = mapped(
                maps,
                4,
                index,
                self.envelfo_transform_values[voice * self.envelfo_transforms.len() + source],
            )?;
            if *operation == 0 {
                add_partial(partials, count, amount);
            } else {
                product *= amount;
            }
        }
        Ok(product)
    }
}

fn mapped(maps: &[Vec<RouteMap>], kind: usize, source: usize, value: f64) -> PyResult<f64> {
    maps[kind]
        .get(source)
        .ok_or_else(|| PyValueError::new_err("Unprepared modulation mapping"))?
        .value(value)
}

#[pymethods]
impl GeneratorBank {
    #[new]
    fn new(
        rate: f64,
        slots: usize,
        parameter_count: usize,
        lfos: PyReadonlyArray2<'_, f64>,
        lfo_rationals: Vec<(u64, u64)>,
    ) -> PyResult<Self> {
        let lfo_definitions = lfo_definitions(lfos, &lfo_rationals, rate)?;
        if !rate.is_finite()
            || rate <= 0.0
            || slots == 0
            || parameter_count == 0
            || (!lfo_definitions.is_empty() && rate != rate as u64 as f64)
            || lfo_definitions
                .iter()
                .any(|d| d.parameter >= parameter_count)
        {
            return Err(PyValueError::new_err("Invalid generator bank definition"));
        }
        let lfo_states = lfo_definitions
            .iter()
            .map(|d| initial_lfo(d, Time::ZERO))
            .collect::<PyResult<Vec<_>>>()?;
        let lfo_count = lfo_definitions.len();
        Ok(Self {
            rate,
            parameter_count,
            patch_host: None,
            patch_states: vec![None; slots],
            patch_storage: vec![None; slots],
            pending_releases: vec![vec![]; slots],
            envelfo_transforms: Vec::new(),
            envelfo_transform_routes: Vec::new(),
            envelfo_transform_values: Vec::new(),
            lfo_definitions,
            lfo_owners: (0..lfo_count).collect(),
            lfo_event_states: lfo_states,
            named_envelopes: Vec::new(),
            named_owners: Vec::new(),
            patch_events: Vec::new(),
            patch_stage_starts: Vec::new(),
            envelfo_keys: vec![0; slots],
            beat_points: Vec::new(),
            staged_envelfos: Vec::new(),
            staged_owners: Vec::new(),
            staged_connections: Vec::new(),
        })
    }

    fn set_named_envelopes(
        &mut self,
        initials: Vec<f64>,
        attacks: Vec<AmplitudeSegments>,
        releases: Vec<AmplitudeSegments>,
        parameters: Vec<NamedEnvelopeInput>,
    ) -> PyResult<()> {
        let count = initials.len();
        if attacks.len() != count || releases.len() != count || parameters.len() != count {
            return Err(PyValueError::new_err("Invalid named envelope definitions"));
        }
        let definitions = initials
            .into_iter()
            .zip(attacks)
            .zip(releases)
            .zip(parameters)
            .map(
                |(
                    ((initial, attack), release),
                    (
                        parameter,
                        operation,
                        release_with_voice,
                        playback,
                        loop_start,
                        loop_end,
                        repeat_count,
                        event_started,
                        (hold, retrigger, bipolar),
                    ),
                )| {
                    if parameter >= self.parameter_count || operation > 1 {
                        return Err(PyValueError::new_err("Invalid named envelope definition"));
                    }
                    let convert = |parts: AmplitudeSegments| -> PyResult<Vec<CoreSegment>> {
                        parts
                            .into_iter()
                            .map(|(duration, to, curve)| {
                                Ok(CoreSegment {
                                    duration: duration
                                        .map(|(n, d)| Time::new(n, d))
                                        .transpose()
                                        .map_err(core_error)?,
                                    to,
                                    curve,
                                })
                            })
                            .collect()
                    };
                    let start = Time::new(loop_start.0, loop_start.1).map_err(core_error)?;
                    let end = Time::new(loop_end.0, loop_end.1).map_err(core_error)?;
                    let bounds = if start == Time::ZERO && end == Time::ONE {
                        None
                    } else {
                        Some((start, end))
                    };
                    let mut segments = convert(attack)?;
                    // SampleHold shares the host route slots but does not execute this body.
                    if segments.is_empty() {
                        segments.push(CoreSegment {
                            duration: Some(Time::ZERO),
                            to: initial,
                            curve: 0.0,
                        });
                    }
                    let body = Envelope {
                        initial: Initial::Value(initial),
                        segments,
                        release: convert(release)?,
                        bipolar,
                        hold,
                        retrigger: match retrigger {
                            0 => Retrigger::Current,
                            1 => Retrigger::Initial,
                            2 => Retrigger::Ignore,
                            _ => {
                                return Err(PyValueError::new_err(
                                    "Invalid named envelope retrigger",
                                ));
                            }
                        },
                        start: if event_started {
                            Start::Event
                        } else {
                            Start::Activation
                        },
                        playback: match playback {
                            0 => Playback::Once,
                            1 => Playback::Loop,
                            2 => Playback::PingPong,
                            _ => {
                                return Err(PyValueError::new_err(
                                    "Invalid named envelope playback",
                                ));
                            }
                        },
                        markers: bounds
                            .map(|(start, end)| {
                                vec![
                                    Marker {
                                        name: "loop_start".into(),
                                        position: start,
                                    },
                                    Marker {
                                        name: "loop_end".into(),
                                        position: end,
                                    },
                                ]
                            })
                            .unwrap_or_default(),
                        loop_bounds: bounds,
                        repeat_count,
                    };
                    body.validate().map_err(core_error)?;
                    Ok(NamedEnvelopeDefinition {
                        sample_hold: None,
                        body,
                        parameter,
                        operation,
                        release_with_voice,
                        beat_clock: false,
                        event_started,
                    })
                },
            )
            .collect::<PyResult<Vec<_>>>()?;
        self.named_envelopes = definitions;
        self.named_owners = (0..count).collect();
        self.pending_releases = vec![vec![None; count]; self.patch_states.len()];
        Ok(())
    }

    fn set_beat_clock(&mut self, points: Vec<BeatClockInput>) -> PyResult<()> {
        let points = points
            .into_iter()
            .map(|(frame, beat, song, rate)| {
                let clock_frame = Time::new(frame.0, frame.1).map_err(core_error)?;
                let clock_beat = Time::new(beat.0, beat.1).map_err(core_error)?;
                let clock_rate = Time::new(rate.0, rate.1).map_err(core_error)?;
                let song_beat = Time::new(song.0, song.1).map_err(core_error)?;
                Ok(BeatPoint {
                    song_beat,
                    clock_frame,
                    clock_beat,
                    clock_rate,
                })
            })
            .collect::<PyResult<Vec<_>>>()?;
        if points.is_empty()
            || points[0].clock_frame != Time::ZERO
            || points
                .iter()
                .any(|p| p.clock_beat < Time::ZERO || p.clock_rate < Time::ZERO)
            || points
                .windows(2)
                .any(|p| p[1].clock_frame <= p[0].clock_frame)
        {
            return Err(PyValueError::new_err("Invalid native beat clock"));
        }
        self.beat_points = points;
        Ok(())
    }

    fn set_beat_named_envelopes(&mut self, indices: Vec<usize>) -> PyResult<()> {
        if self.beat_points.is_empty()
            || indices
                .iter()
                .any(|index| *index >= self.named_envelopes.len())
        {
            return Err(PyValueError::new_err("Invalid beat-clock named Envelfo"));
        }
        for index in indices {
            self.named_envelopes[index].beat_clock = true;
        }
        Ok(())
    }

    fn set_staged_envelfos(
        &mut self,
        inputs: Vec<StagedEnvelfoInput>,
        connections: Vec<StageConnectionInput>,
    ) -> PyResult<()> {
        let mut definitions = Vec::with_capacity(inputs.len());
        for (inputs, transitions, initial_stage, parameter, operation, beat_clock) in inputs {
            if inputs.is_empty()
                || initial_stage >= inputs.len()
                || (beat_clock && self.beat_points.is_empty())
                || parameter.is_some_and(|p| p >= self.parameter_count)
                || operation > 1
                || transitions
                    .iter()
                    .any(|(source, _, target)| *source >= inputs.len() || *target > inputs.len())
            {
                return Err(PyValueError::new_err("Invalid staged Envelfo definition"));
            }
            let mut bodies = Vec::with_capacity(inputs.len());
            for (
                index,
                (kind, initial, current, parts, lfo, markers, playback, start, end, repeats),
            ) in inputs.into_iter().enumerate()
            {
                let markers = markers
                    .into_iter()
                    .map(|(position, name)| {
                        Ok(Marker {
                            name,
                            position: ratio_time(position)?,
                        })
                    })
                    .collect::<PyResult<Vec<_>>>()?;
                let body = match kind {
                    1 => {
                        if lfo.is_some() {
                            return Err(PyValueError::new_err("Invalid staged Envelope"));
                        }
                        let start = ratio_time(start)?;
                        let end = ratio_time(end)?;
                        stages::Body::Envelope(Envelope {
                            initial: if current {
                                Initial::Current
                            } else {
                                Initial::Value(initial)
                            },
                            segments: parts
                                .into_iter()
                                .map(|(duration, to, curve)| {
                                    Ok(CoreSegment {
                                        duration: duration.map(ratio_time).transpose()?,
                                        to,
                                        curve,
                                    })
                                })
                                .collect::<PyResult<_>>()?,
                            release: vec![],
                            bipolar: true,
                            hold: true,
                            retrigger: Retrigger::Current,
                            start: Start::Activation,
                            playback: match playback {
                                0 => Playback::Once,
                                1 => Playback::Loop,
                                2 => Playback::PingPong,
                                _ => {
                                    return Err(PyValueError::new_err(
                                        "Invalid staged Envelope playback",
                                    ));
                                }
                            },
                            loop_bounds: if start == Time::ZERO && end == Time::ONE {
                                None
                            } else {
                                Some((start, end))
                            },
                            repeat_count: repeats,
                            markers: markers.clone(),
                        })
                    }
                    2 => {
                        if current || !parts.is_empty() || playback != 0 || repeats.is_some() {
                            return Err(PyValueError::new_err("Invalid staged LFO"));
                        }
                        let Some((shape, times, center, depth)) = lfo else {
                            return Err(PyValueError::new_err("Invalid staged LFO"));
                        };
                        let times = times
                            .into_iter()
                            .map(ratio_time)
                            .collect::<PyResult<Vec<_>>>()?;
                        if times.len() != 5 {
                            return Err(PyValueError::new_err("Invalid staged LFO timing"));
                        }
                        let mut body = lfo::LFO::new(times[0]).map_err(core_error)?;
                        body.shape = match shape {
                            0 => lfo::Shape::Sine,
                            1 => lfo::Shape::Square,
                            2 => lfo::Shape::Triangle,
                            _ => return Err(PyValueError::new_err("Invalid staged LFO shape")),
                        };
                        body.phase = times[1];
                        body.duty = times[2];
                        body.delay = times[3];
                        body.fade = times[4];
                        body.center = center;
                        body.depth = depth;
                        stages::Body::LFO(body)
                    }
                    _ => return Err(PyValueError::new_err("Invalid staged Envelfo stage")),
                };
                bodies.push(stages::Stage {
                    name: index.to_string(),
                    markers: if kind == 2 { markers } else { vec![] },
                    body,
                });
            }
            let count = bodies.len();
            let body = stages::Stages {
                initial_stage: initial_stage.to_string(),
                stages: bodies,
                transitions: transitions
                    .into_iter()
                    .map(|(source, event, target)| stages::Transition {
                        from: vec![source.to_string()],
                        event,
                        action: if target == count {
                            stages::Action::Finish
                        } else {
                            stages::Action::Enter(target.to_string())
                        },
                    })
                    .collect(),
            };
            body.validate().map_err(core_error)?;
            definitions.push(StagedEnvelfoDefinition {
                body,
                parameter,
                operation,
                beat_clock,
            });
        }
        let connections = connections
            .into_iter()
            .map(
                |(source, port, destination, cue, every, offset, probability, key, delay)| {
                    let delay = ratio_time(delay)?;
                    if source >= definitions.len()
                        || destination >= definitions.len()
                        || port.is_empty()
                        || cue.is_empty()
                        || every == 0
                        || probability > (1 << 53)
                        || delay < Time::ZERO
                    {
                        return Err(PyValueError::new_err("Invalid staged Envelfo connection"));
                    }
                    Ok(StagedConnection {
                        source,
                        port,
                        destination,
                        cue,
                        every,
                        offset,
                        probability,
                        key,
                        delay,
                    })
                },
            )
            .collect::<PyResult<_>>()?;
        self.staged_envelfos = definitions;
        self.staged_owners = (0..self.staged_envelfos.len()).collect();
        self.staged_connections = connections;
        Ok(())
    }

    fn set_sample_hold_envelfos(
        &mut self,
        definitions: Vec<(usize, u64, f64, f64)>,
    ) -> PyResult<()> {
        if definitions.iter().any(|(source, _, minimum, maximum)| {
            *source >= self.named_envelopes.len()
                || !minimum.is_finite()
                || !maximum.is_finite()
                || !(-1.0..=1.0).contains(minimum)
                || !(-1.0..=1.0).contains(maximum)
                || minimum > maximum
        }) {
            return Err(PyValueError::new_err("Invalid sample-and-hold definition"));
        }
        for (source, key, minimum, maximum) in definitions {
            self.named_envelopes[source].sample_hold = Some((key, minimum, maximum));
        }
        Ok(())
    }

    fn set_envelfo_state_owners(
        &mut self,
        lfos: Vec<usize>,
        envelopes: Vec<usize>,
        stages: Vec<usize>,
    ) -> PyResult<()> {
        if lfos.len() != self.lfo_definitions.len()
            || envelopes.len() != self.named_envelopes.len()
            || stages.len() != self.staged_envelfos.len()
            || !valid_envelfo_owners(&lfos)
            || !valid_envelfo_owners(&envelopes)
            || !valid_envelfo_owners(&stages)
        {
            return Err(PyValueError::new_err("Invalid shared Envelfo state owners"));
        }
        self.lfo_owners = lfos;
        self.named_owners = envelopes;
        self.staged_owners = stages;
        Ok(())
    }

    fn set_envelfo_transforms(
        &mut self,
        nodes: Vec<EnvelfoTransformInput>,
        routes: Vec<EnvelfoTransformRoute>,
    ) -> PyResult<()> {
        if nodes
            .iter()
            .enumerate()
            .any(|(node, (kind, inputs, scale, offset, initial))| {
                *kind > 6
                    || !scale.is_finite()
                    || !offset.is_finite()
                    || inputs.len() < if *kind >= 2 { 1 } else { 2 }
                    || (*kind >= 2 && inputs.len() != 1)
                    || (*kind == 3 && scale >= offset)
                    || (*kind == 4 && (*scale < 0.0 || *offset < 0.0))
                    || (*kind == 5 && *scale <= 0.0)
                    || initial.is_some_and(|value| *kind != 4 || !value.is_finite())
                    || inputs.iter().any(|(kind, index, offset, scale)| {
                        !offset.is_finite()
                            || !scale.is_finite()
                            || match kind {
                                0 => *index >= self.lfo_definitions.len(),
                                1 => *index >= self.named_envelopes.len(),
                                2 => *index >= self.staged_envelfos.len(),
                                3 => *index >= node,
                                _ => true,
                            }
                    })
            })
            || routes.iter().any(|(node, parameter, operation)| {
                *node >= nodes.len() || *parameter >= self.parameter_count || *operation > 1
            })
        {
            return Err(PyValueError::new_err("Invalid Envelfo transform graph"));
        }
        self.envelfo_transform_values = vec![0.0; self.patch_states.len() * nodes.len()];
        self.envelfo_transforms = nodes;
        self.envelfo_transform_routes = routes;
        Ok(())
    }

    fn set_patch_events(&mut self, connections: Vec<PatchEventInput>) -> PyResult<()> {
        if connections.iter().any(
            |(source, marker, destination, cue, _, every, _, probability, _, delay, capture)| {
                *every == 0
                    || *probability > (1 << 53)
                    || ratio_time(*delay).is_err()
                    || ratio_time(*delay).is_ok_and(|d| d < Time::ZERO)
                    || *source >= self.lfo_definitions.len()
                    || self.lfo_definitions[*source].scope != 3
                    || self.lfo_definitions[*source].body.rate == Time::ZERO
                    || self.lfo_definitions[*source].body.clock == lfo::Clock::Beats
                    || self.lfo_definitions[*source].body.driver == lfo::Driver::Transport
                    || ratio_time(*marker).is_err()
                    || ratio_time(*marker).is_ok_and(|m| m < Time::ZERO || m >= Time::ONE)
                    || if *capture {
                        cue.is_some()
                            || *destination >= self.envelfo_transforms.len()
                            || self.envelfo_transforms[*destination].0 != 6
                    } else if cue.is_some() {
                        *destination >= self.staged_envelfos.len()
                    } else {
                        *destination >= self.named_envelopes.len()
                            || !self.named_envelopes[*destination].event_started
                    }
            },
        ) {
            return Err(PyValueError::new_err("Invalid Patch event connections"));
        }
        self.patch_events = connections
            .into_iter()
            .map(
                |(
                    source,
                    marker,
                    destination,
                    cue,
                    _order,
                    every,
                    offset,
                    probability,
                    key,
                    delay,
                    capture,
                )| {
                    Ok(PatchEventConnection {
                        threshold: false,
                        capture,
                        source: self.lfo_owners[source],
                        marker: ratio_time(marker)?,
                        destination: if capture {
                            destination
                        } else if cue.is_some() {
                            self.staged_owners[destination]
                        } else {
                            self.named_owners[destination]
                        },
                        cue,
                        every,
                        offset,
                        probability,
                        key,
                        delay: ratio_time(delay)?,
                    })
                },
            )
            .collect::<PyResult<_>>()?;
        Ok(())
    }

    fn set_patch_stage_starts(&mut self, connections: Vec<StageStartInput>) -> PyResult<()> {
        if connections.iter().any(
            |(source, port, destination, _, every, _, probability, _, delay, capture)| {
                *source >= self.staged_envelfos.len()
                    || *every == 0
                    || *probability > (1 << 53)
                    || ratio_time(*delay).is_err()
                    || ratio_time(*delay).is_ok_and(|d| d < Time::ZERO)
                    || port.is_empty()
                    || if *capture {
                        *destination >= self.envelfo_transforms.len()
                            || self.envelfo_transforms[*destination].0 != 6
                    } else {
                        *destination >= self.named_envelopes.len()
                            || !self.named_envelopes[*destination].event_started
                    }
            },
        ) {
            return Err(PyValueError::new_err(
                "Invalid Patch stage start connections",
            ));
        }
        self.patch_stage_starts = connections
            .into_iter()
            .map(
                |(
                    source,
                    port,
                    destination,
                    _order,
                    every,
                    offset,
                    probability,
                    key,
                    delay,
                    capture,
                )| {
                    Ok(PatchStageStartConnection {
                        capture,
                        source: self.staged_owners[source],
                        port,
                        destination: if capture {
                            destination
                        } else {
                            self.named_owners[destination]
                        },
                        every,
                        offset,
                        probability,
                        key,
                        delay: ratio_time(delay)?,
                    })
                },
            )
            .collect::<PyResult<_>>()?;
        Ok(())
    }

    fn set_threshold_events(&mut self, connections: Vec<PatchEventInput>) -> PyResult<()> {
        if connections.iter().any(
            |(source, marker, destination, cue, _, every, _, probability, _, delay, capture)| {
                *source >= self.envelfo_transforms.len()
                    || self.envelfo_transforms[*source].0 != 3
                    || (*marker != (0, 1) && *marker != (1, 1))
                    || *every == 0
                    || *probability > (1 << 53)
                    || ratio_time(*delay).is_err()
                    || ratio_time(*delay).is_ok_and(|d| d < Time::ONE)
                    || if *capture {
                        cue.is_some()
                            || *destination >= self.envelfo_transforms.len()
                            || self.envelfo_transforms[*destination].0 != 6
                    } else if cue.is_some() {
                        *destination >= self.staged_envelfos.len()
                    } else {
                        *destination >= self.named_envelopes.len()
                            || !self.named_envelopes[*destination].event_started
                    }
            },
        ) {
            return Err(PyValueError::new_err("Invalid threshold event connections"));
        }
        let definitions = connections
            .into_iter()
            .map(
                |(
                    source,
                    marker,
                    destination,
                    cue,
                    _order,
                    every,
                    offset,
                    probability,
                    key,
                    delay,
                    capture,
                )| {
                    Ok(PatchEventConnection {
                        threshold: true,
                        capture,
                        source,
                        marker: ratio_time(marker)?,
                        destination: if capture {
                            destination
                        } else if cue.is_some() {
                            self.staged_owners[destination]
                        } else {
                            self.named_owners[destination]
                        },
                        cue,
                        every,
                        offset,
                        probability,
                        key,
                        delay: ratio_time(delay)?,
                    })
                },
            )
            .collect::<PyResult<Vec<_>>>()?;
        self.patch_events.extend(definitions);
        Ok(())
    }
}
