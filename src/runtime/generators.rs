//! Shared prepared generator state and modulation for native source engines.
use super::controls::ControlBank;
use super::patch_host::{PatchHost, clock_at};
use super::preparation::patch::{Detectors, GeneratorSlots, PreparedPatch, TransformMaps};
use super::{
    BeatClockInput, PatchEventInput, StageConnectionInput, StageStartInput, initial_lfo,
    ratio_time, valid_envelfo_owners,
};
use super::{
    BeatPoint, EnvelfoTransformInput, EnvelfoTransformRoute, LfoDefinition, LfoPlayback,
    NamedEnvelopeDefinition, PatchEventConnection, PatchStageStartConnection, StagedConnection,
    StagedEnvelfoDefinition, control_time, core_error, lfo_command, route_map,
};
use crate::amplitude::{frame_position, frame_time};
use crate::error::{NativeError, NativeResult};
use engelfo::lfo::Command;
use engelfo::patch::{ChildCommand, PatchState, PatchStorage};
use engelfo::{lfo, time::Time};
use numpy::PyReadonlyArray2;
use pyo3::{exceptions::PyValueError, prelude::*};
use ufor::modulation::math::{Mapping, add_partial};

#[pyclass(skip_from_py_object)]
#[derive(Clone)]
pub(crate) struct GeneratorBank {
    pub(crate) rate: f64,
    pub(crate) parameter_count: usize,
    pub(super) patch_host: Option<PatchHost>,
    pub(crate) patch_states: Vec<Option<PatchState>>,
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
    pub(crate) fn prepare_route_maps(
        &self,
        kind: usize,
        maps: Vec<(usize, Vec<(f64, f64)>)>,
    ) -> PyResult<Vec<Mapping>> {
        let count = match kind {
            1 => self.lfo_definitions.len(),
            2 => self.named_envelopes.len(),
            3 => self.staged_envelfos.len(),
            4 => self.envelfo_transform_routes.len(),
            _ => return Err(PyValueError::new_err("Invalid generator mapping kind")),
        };
        if maps.len() != count {
            return Err(PyValueError::new_err("Invalid generator mapping count"));
        }
        maps.into_iter()
            .map(|(kind, points)| route_map(kind, points))
            .collect()
    }

    pub(crate) fn same_definition(&self, other: &Self) -> bool {
        self.rate == other.rate
            && self.parameter_count == other.parameter_count
            && self.patch_states.len() == other.patch_states.len()
            && self.lfo_definitions == other.lfo_definitions
            && self.lfo_owners == other.lfo_owners
            && self.named_envelopes == other.named_envelopes
            && self.named_owners == other.named_owners
            && self.envelfo_transforms == other.envelfo_transforms
            && self.envelfo_transform_routes == other.envelfo_transform_routes
            && self.patch_events == other.patch_events
            && self.patch_stage_starts == other.patch_stage_starts
            && self.beat_points == other.beat_points
            && self.staged_envelfos == other.staged_envelfos
            && self.staged_owners == other.staged_owners
            && self.staged_connections == other.staged_connections
    }

    pub(crate) fn release_voice(
        &mut self,
        voice: usize,
        frame: usize,
        age: usize,
        minimum_hold_frames: Time,
    ) -> NativeResult<Time> {
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

    pub(crate) fn apply_action(
        &mut self,
        action: &[f64],
        active: &[bool],
        frame: usize,
        rate: f64,
    ) -> NativeResult<()> {
        if action.len() != 6 {
            return Err(NativeError::Message("Invalid generator action"));
        }
        let kind = action[1] as usize;
        let voice = action[2] as usize;
        if action[1] != kind as f64
            || action[2] != voice as f64
            || (kind != 8 && voice >= active.len())
        {
            return Err(NativeError::Message("Invalid generator action"));
        }
        match kind {
            12 => {
                if action[3] != action[3] as u32 as f64
                    || action[4] != action[4] as u32 as f64
                    || action[5] != 0.0
                {
                    return Err(NativeError::Message("Invalid Envelfo stream key"));
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
                    return Err(NativeError::Message("Invalid synth LFO action"));
                }
                let definition = &self.lfo_definitions[lfo];
                if definition.scope != 0 || definition.body.driver == lfo::Driver::Transport {
                    return Err(NativeError::Message("Invalid synth LFO scope"));
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
                    return Err(NativeError::Message("Invalid named Envelfo action"));
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
                    return Err(NativeError::Message("Invalid staged Envelfo action"));
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
                    return Err(NativeError::Message("Invalid voice LFO action"));
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

            _ => return Err(NativeError::Message("Unknown generator action")),
        }
        Ok(())
    }

    pub(super) fn lfo_time(&self, definition: &LfoDefinition, frame: Time) -> NativeResult<Time> {
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
    ) -> NativeResult<(f64, f64)> {
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
    pub(crate) fn contributions(
        &self,
        voice: usize,
        parameter: usize,
        mut product: f64,
        frame: usize,
        controls: &ControlBank,
        maps: &[Vec<Mapping>],
        partials: &mut [f64],
        count: &mut usize,
    ) -> NativeResult<f64> {
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
                add_partial(partials, count, weight * amount)?;
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
                add_partial(partials, count, amount)?;
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
                add_partial(partials, count, amount)?;
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
                add_partial(partials, count, amount)?;
            } else {
                product *= amount;
            }
        }
        Ok(product)
    }
}

fn mapped(maps: &[Vec<Mapping>], kind: usize, source: usize, value: f64) -> NativeResult<f64> {
    maps[kind]
        .get(source)
        .ok_or(NativeError::Message("Unprepared modulation mapping"))?
        .value(value)
        .map_err(NativeError::Calculation)
}

#[pymethods]
impl GeneratorBank {
    fn append(
        &mut self,
        other: &Self,
        parameters: Vec<usize>,
    ) -> PyResult<(usize, usize, usize, usize)> {
        if self.rate != other.rate
            || self.patch_states.len() != other.patch_states.len()
            || parameters.len() != other.parameter_count
            || parameters.iter().any(|p| *p >= self.parameter_count)
            || (!self.beat_points.is_empty()
                && !other.beat_points.is_empty()
                && self.beat_points != other.beat_points)
        {
            return Err(PyValueError::new_err("Incompatible generator bank layout"));
        }
        let offsets = [
            self.lfo_definitions.len(),
            self.named_envelopes.len(),
            self.staged_envelfos.len(),
            self.envelfo_transforms.len(),
        ];
        let mut other = other.clone();
        for definition in &mut other.lfo_definitions {
            definition.parameter = parameters[definition.parameter];
        }
        for owner in &mut other.lfo_owners {
            *owner += offsets[0];
        }
        for definition in &mut other.named_envelopes {
            definition.parameter = parameters[definition.parameter];
        }
        for owner in &mut other.named_owners {
            *owner += offsets[1];
        }
        for definition in &mut other.staged_envelfos {
            definition.parameter = definition.parameter.map(|p| parameters[p]);
        }
        for owner in &mut other.staged_owners {
            *owner += offsets[2];
        }
        for connection in &mut other.staged_connections {
            connection.source += offsets[2];
            connection.destination += offsets[2];
        }
        for (_, inputs, _, _, _) in &mut other.envelfo_transforms {
            for (kind, source, _, _) in inputs {
                *source += offsets[*kind as usize];
            }
        }
        for (node, parameter, _) in &mut other.envelfo_transform_routes {
            *node += offsets[3];
            *parameter = parameters[*parameter];
        }
        for connection in &mut other.patch_events {
            connection.source += offsets[if connection.threshold { 3 } else { 0 }];
            connection.destination += offsets[if connection.capture {
                3
            } else if connection.cue.is_some() {
                2
            } else {
                1
            }];
        }
        for connection in &mut other.patch_stage_starts {
            connection.source += offsets[2];
            connection.destination += offsets[if connection.capture { 3 } else { 1 }];
        }
        self.lfo_definitions.extend(other.lfo_definitions);
        self.lfo_owners.extend(other.lfo_owners);
        self.lfo_event_states.extend(other.lfo_event_states);
        self.named_envelopes.extend(other.named_envelopes);
        self.named_owners.extend(other.named_owners);
        self.staged_envelfos.extend(other.staged_envelfos);
        self.staged_owners.extend(other.staged_owners);
        self.staged_connections.extend(other.staged_connections);
        self.envelfo_transforms.extend(other.envelfo_transforms);
        self.envelfo_transform_routes
            .extend(other.envelfo_transform_routes);
        self.patch_events.extend(other.patch_events);
        self.patch_stage_starts.extend(other.patch_stage_starts);
        if self.beat_points.is_empty() {
            self.beat_points = other.beat_points;
        }
        for releases in &mut self.pending_releases {
            releases.resize(self.named_envelopes.len(), None);
        }
        self.envelfo_transform_values
            .resize(self.patch_states.len() * self.envelfo_transforms.len(), 0.0);
        Ok((offsets[0], offsets[1], offsets[2], offsets[3]))
    }

    #[new]
    #[allow(clippy::too_many_arguments)]
    #[pyo3(signature = (rate, slots, parameter_count, lfo_routes, lfo_documents, envelope_documents, envelope_routes, *, max_records=10000))]
    fn new(
        rate: f64,
        slots: usize,
        parameter_count: usize,
        lfo_routes: PyReadonlyArray2<'_, f64>,
        lfo_documents: Vec<String>,
        envelope_documents: Vec<String>,
        envelope_routes: Vec<(usize, usize, bool)>,
        max_records: usize,
    ) -> PyResult<Self> {
        if !rate.is_finite()
            || rate <= 0.0
            || slots == 0
            || parameter_count == 0
            || (!lfo_documents.is_empty() && rate != rate as u64 as f64)
        {
            return Err(PyValueError::new_err("Invalid generator bank definition"));
        }
        let (lfos, envelopes) =
            super::preparation::definitions(&lfo_documents, &envelope_documents, max_records)?;
        let lfo_definitions = super::preparation::lfos(&lfos, lfo_routes, rate, parameter_count)?;
        let named_envelopes =
            super::preparation::envelopes(&envelopes, envelope_routes, rate, parameter_count)?;
        let named_count = named_envelopes.len();
        let lfo_states = lfo_definitions
            .iter()
            .map(|d| initial_lfo(d, Time::ZERO))
            .collect::<NativeResult<Vec<_>>>()?;
        let lfo_count = lfo_definitions.len();
        Ok(Self {
            rate,
            parameter_count,
            patch_host: None,
            patch_states: vec![None; slots],
            patch_storage: vec![None; slots],
            pending_releases: vec![vec![None; named_count]; slots],
            envelfo_transforms: Vec::new(),
            envelfo_transform_routes: Vec::new(),
            envelfo_transform_values: Vec::new(),
            lfo_definitions,
            lfo_owners: (0..lfo_count).collect(),
            lfo_event_states: lfo_states,
            named_envelopes,
            named_owners: (0..named_count).collect(),
            patch_events: Vec::new(),
            patch_stage_starts: Vec::new(),
            envelfo_keys: vec![0; slots],
            beat_points: Vec::new(),
            staged_envelfos: Vec::new(),
            staged_owners: Vec::new(),
            staged_connections: Vec::new(),
        })
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

    #[pyo3(signature = (inputs, connections, *, max_records=10000))]
    fn set_staged_envelfos(
        &mut self,
        inputs: Vec<(String, Option<usize>, usize)>,
        connections: Vec<StageConnectionInput>,
        max_records: usize,
    ) -> PyResult<()> {
        let maximum = max_records
            .checked_sub(connections.len())
            .ok_or_else(|| PyValueError::new_err("Stages preparation exceeds record limit"))?;
        let definitions = super::preparation::staged(
            inputs,
            self.rate,
            self.parameter_count,
            !self.beat_points.is_empty(),
            maximum,
        )?;
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

    fn set_sample_hold_keys(&mut self, definitions: Vec<(usize, u64)>) -> PyResult<()> {
        if definitions.iter().any(|(source, _)| {
            *source >= self.named_envelopes.len()
                || self.named_envelopes[*source].sample_hold.is_none()
        }) {
            return Err(PyValueError::new_err("Invalid sample-and-hold definition"));
        }
        for (source, key) in definitions {
            self.named_envelopes[source]
                .sample_hold
                .as_mut()
                .expect("validated SampleHold slot")
                .0 = key;
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

    #[pyo3(signature = (context, generators, slots, targets, orders, *, max_records=10000))]
    fn configure_patches(
        &mut self,
        context: String,
        generators: Vec<(String, String)>,
        slots: GeneratorSlots,
        targets: Vec<(String, String, usize)>,
        orders: Vec<(String, Vec<String>)>,
        max_records: usize,
    ) -> PyResult<(Detectors, TransformMaps)> {
        let PreparedPatch {
            nodes,
            routes,
            events,
            starts,
            thresholds,
            detectors,
            mappings,
        } = PreparedPatch::prepare(
            self,
            &context,
            &generators,
            slots,
            targets,
            orders,
            max_records,
        )?;
        let mut prepared = self.clone();
        prepared.configure_transforms(nodes, routes)?;
        prepared.configure_events(events)?;
        prepared.configure_stage_starts(starts)?;
        prepared.configure_thresholds(thresholds)?;
        *self = prepared;
        Ok((detectors, mappings))
    }
}

impl GeneratorBank {
    fn configure_transforms(
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

    fn configure_events(&mut self, connections: Vec<PatchEventInput>) -> PyResult<()> {
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

    fn configure_stage_starts(&mut self, connections: Vec<StageStartInput>) -> PyResult<()> {
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

    fn configure_thresholds(&mut self, connections: Vec<PatchEventInput>) -> PyResult<()> {
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
