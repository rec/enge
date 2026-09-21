//! Persistent oscillator runtime with sample-accurate voice actions.

use numpy::ndarray::{Array2, ArrayViewMut2};
use numpy::{IntoPyArray, PyArray2, PyReadonlyArray2, PyReadwriteArray2, PyUntypedArrayMethods};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use std::f64::consts::{PI, TAU};

#[derive(Clone, PartialEq)]
struct Segment {
    frames: f64,
    target: f64,
}

#[derive(Clone, PartialEq)]
struct ControlDefinition {
    default: f64,
    smoothing_numerator: u64,
    smoothing_denominator: u64,
    scope: usize,
    parameter: usize,
    operation: usize,
    minimum: f64,
    maximum: f64,
    intercept: f64,
    slope: f64,
}

#[derive(Clone)]
struct ControlState {
    start: f64,
    target: f64,
    elapsed: usize,
}

#[derive(Clone, PartialEq)]
struct LfoDefinition {
    scope: usize,
    waveform: usize,
    parameter: usize,
    operation: usize,
    intercept: f64,
    slope: f64,
    duty: (u64, u64),
    rate: (u64, u64),
    phase: (u64, u64),
    delay_frames: (u64, u64),
    fade_frames: (u64, u64),
}

#[pyclass(frozen, skip_from_py_object)]
#[derive(Clone)]
pub struct SynthRuntimeSnapshot {
    rate: f64,
    waveform: u8,
    duty: f64,
    initial: f64,
    attack: Vec<Segment>,
    release: Vec<Segment>,
    minimum_hold_frames: f64,
    routes: Array2<f64>,
    control_definitions: Vec<ControlDefinition>,
    control_states: Vec<ControlState>,
    lfo_definitions: Vec<LfoDefinition>,
    context_kinds: Vec<usize>,
    context_states: Vec<ControlState>,
    parameter_definitions: Vec<f64>,
    frequencies: Vec<f64>,
    phases: Vec<f64>,
    errors: Vec<f64>,
    gains: Vec<f64>,
    active: Vec<bool>,
    ages: Vec<usize>,
    release_frames: Vec<Option<f64>>,
    release_levels: Vec<f64>,
    frequency_steps: Vec<f64>,
    frequency_remaining: Vec<usize>,
    gain_steps: Vec<f64>,
    gain_remaining: Vec<usize>,
    voice_part_contexts: Vec<usize>,
    voice_trigger_contexts: Vec<usize>,
    filter_states: Array2<f64>,
    frame: usize,
}

#[pyclass]
pub struct SynthRuntime {
    rate: f64,
    waveform: u8,
    duty: f64,
    initial: f64,
    attack: Vec<Segment>,
    release: Vec<Segment>,
    minimum_hold_frames: f64,
    routes: Array2<f64>,
    control_definitions: Vec<ControlDefinition>,
    control_states: Vec<ControlState>,
    lfo_definitions: Vec<LfoDefinition>,
    context_kinds: Vec<usize>,
    context_states: Vec<ControlState>,
    parameter_definitions: Vec<f64>,
    frequencies: Vec<f64>,
    phases: Vec<f64>,
    errors: Vec<f64>,
    gains: Vec<f64>,
    active: Vec<bool>,
    ages: Vec<usize>,
    release_frames: Vec<Option<f64>>,
    release_levels: Vec<f64>,
    frequency_steps: Vec<f64>,
    frequency_remaining: Vec<usize>,
    gain_steps: Vec<f64>,
    gain_remaining: Vec<usize>,
    voice_part_contexts: Vec<usize>,
    voice_trigger_contexts: Vec<usize>,
    filter_states: Array2<f64>,
    frame: usize,
}

#[pymethods]
impl SynthRuntime {
    #[new]
    #[allow(clippy::too_many_arguments)]
    fn new(
        rate: f64,
        waveform: u8,
        duty: f64,
        routes: PyReadonlyArray2<'_, f64>,
        initial: f64,
        attack: PyReadonlyArray2<'_, f64>,
        release: PyReadonlyArray2<'_, f64>,
        minimum_hold_frames: f64,
        controls: PyReadonlyArray2<'_, f64>,
        control_smoothing: Vec<(u64, u64)>,
        lfos: PyReadonlyArray2<'_, f64>,
        lfo_rationals: Vec<(u64, u64)>,
        parameters: Vec<f64>,
        context_capacity: usize,
    ) -> PyResult<Self> {
        let attack = segments(attack)?;
        let release = segments(release)?;
        let (control_definitions, control_states) =
            control_definitions(controls, &control_smoothing)?;
        let lfo_definitions = lfo_definitions(lfos, &lfo_rationals)?;
        if !rate.is_finite()
            || rate <= 0.0
            || waveform > 2
            || !duty.is_finite()
            || !(0.0..=1.0).contains(&duty)
            || routes.shape()[0] == 0
            || routes.shape()[1] == 0
            || routes.as_array().iter().any(|v| !v.is_finite())
            || !initial.is_finite()
            || !minimum_hold_frames.is_finite()
            || minimum_hold_frames < 0.0
            || parameters.len() != 6
            || parameters.iter().any(|v| !v.is_finite())
            || parameters[1] > parameters[0]
            || parameters[0] > parameters[2]
            || parameters[4] > parameters[3]
            || parameters[3] > parameters[5]
            || context_capacity == 0
            || (!lfo_definitions.is_empty() && rate != rate as u64 as f64)
            || lfo_definitions
                .iter()
                .any(|definition| lfo_phase_denominator(definition, rate as u64).is_none())
        {
            return Err(PyValueError::new_err("Invalid synth runtime definition"));
        }
        let slots = routes.shape()[0];
        let channels = routes.shape()[1];
        let context_states = (0..context_capacity)
            .flat_map(|_| control_states.clone())
            .collect();
        Ok(Self {
            rate,
            waveform,
            duty,
            initial,
            attack,
            release,
            minimum_hold_frames,
            routes: routes.as_array().to_owned(),
            control_definitions,
            control_states,
            lfo_definitions,
            context_kinds: vec![0; context_capacity],
            context_states,
            parameter_definitions: parameters,
            frequencies: vec![1.0; slots],
            phases: vec![0.0; slots],
            errors: vec![0.0; slots],
            gains: vec![0.0; slots],
            active: vec![false; slots],
            ages: vec![0; slots],
            release_frames: vec![None; slots],
            release_levels: vec![0.0; slots],
            frequency_steps: vec![0.0; slots],
            frequency_remaining: vec![0; slots],
            gain_steps: vec![0.0; slots],
            gain_remaining: vec![0; slots],
            voice_part_contexts: vec![usize::MAX; slots],
            voice_trigger_contexts: vec![usize::MAX; slots],
            filter_states: Array2::zeros((channels, 2)),
            frame: 0,
        })
    }

    fn process<'py>(
        &mut self,
        py: Python<'py>,
        frames: usize,
        cutoff_hz: f64,
        q: f64,
        gain_db: f64,
    ) -> PyResult<Bound<'py, PyArray2<f64>>> {
        self.render_owned(py, frames, cutoff_hz, q, gain_db, &[])
    }

    fn process_actions<'py>(
        &mut self,
        py: Python<'py>,
        frames: usize,
        cutoff_hz: f64,
        q: f64,
        gain_db: f64,
        actions: PyReadonlyArray2<'_, f64>,
    ) -> PyResult<Bound<'py, PyArray2<f64>>> {
        if actions.shape()[1] != 6 || actions.as_array().iter().any(|v| !v.is_finite()) {
            return Err(PyValueError::new_err("Invalid synth runtime actions"));
        }
        self.render_owned(py, frames, cutoff_hz, q, gain_db, actions.as_slice()?)
    }

    fn process_into(
        &mut self,
        mut output: PyReadwriteArray2<'_, f64>,
        cutoff_hz: f64,
        q: f64,
        gain_db: f64,
    ) -> PyResult<()> {
        if !output.is_c_contiguous() {
            return Err(PyValueError::new_err(
                "Synth runtime output must be C-contiguous",
            ));
        }
        self.render_into(output.as_array_mut(), cutoff_hz, q, gain_db, &[])
    }

    fn process_actions_into(
        &mut self,
        mut output: PyReadwriteArray2<'_, f64>,
        cutoff_hz: f64,
        q: f64,
        gain_db: f64,
        actions: PyReadonlyArray2<'_, f64>,
        action_count: usize,
    ) -> PyResult<()> {
        if !output.is_c_contiguous() || actions.shape()[1] != 6 || action_count > actions.shape()[0]
        {
            return Err(PyValueError::new_err(
                "Invalid synth runtime output or actions",
            ));
        }
        let actions = &actions.as_slice()?[..action_count * 6];
        if actions.iter().any(|v| !v.is_finite()) {
            return Err(PyValueError::new_err("Invalid synth runtime actions"));
        }
        self.render_into(output.as_array_mut(), cutoff_hz, q, gain_db, actions)
    }

    fn active_slots(&self) -> Vec<bool> {
        self.active.clone()
    }

    fn active_contexts(&self) -> Vec<bool> {
        self.context_kinds.iter().map(|kind| *kind != 0).collect()
    }

    fn snapshot(&self) -> SynthRuntimeSnapshot {
        SynthRuntimeSnapshot {
            rate: self.rate,
            waveform: self.waveform,
            duty: self.duty,
            initial: self.initial,
            attack: self.attack.clone(),
            release: self.release.clone(),
            minimum_hold_frames: self.minimum_hold_frames,
            routes: self.routes.clone(),
            control_definitions: self.control_definitions.clone(),
            control_states: self.control_states.clone(),
            lfo_definitions: self.lfo_definitions.clone(),
            context_kinds: self.context_kinds.clone(),
            context_states: self.context_states.clone(),
            parameter_definitions: self.parameter_definitions.clone(),
            frequencies: self.frequencies.clone(),
            phases: self.phases.clone(),
            errors: self.errors.clone(),
            gains: self.gains.clone(),
            active: self.active.clone(),
            ages: self.ages.clone(),
            release_frames: self.release_frames.clone(),
            release_levels: self.release_levels.clone(),
            frequency_steps: self.frequency_steps.clone(),
            frequency_remaining: self.frequency_remaining.clone(),
            gain_steps: self.gain_steps.clone(),
            gain_remaining: self.gain_remaining.clone(),
            voice_part_contexts: self.voice_part_contexts.clone(),
            voice_trigger_contexts: self.voice_trigger_contexts.clone(),
            filter_states: self.filter_states.clone(),
            frame: self.frame,
        }
    }

    fn restore(&mut self, snapshot: &SynthRuntimeSnapshot) -> PyResult<()> {
        if self.rate != snapshot.rate
            || self.waveform != snapshot.waveform
            || self.duty != snapshot.duty
            || self.initial != snapshot.initial
            || self.attack != snapshot.attack
            || self.release != snapshot.release
            || self.minimum_hold_frames != snapshot.minimum_hold_frames
            || self.routes != snapshot.routes
            || self.control_definitions != snapshot.control_definitions
            || self.lfo_definitions != snapshot.lfo_definitions
            || self.parameter_definitions != snapshot.parameter_definitions
            || self.context_kinds.len() != snapshot.context_kinds.len()
        {
            return Err(PyValueError::new_err(
                "Snapshot belongs to a different synth runtime",
            ));
        }
        self.frequencies.clone_from(&snapshot.frequencies);
        self.phases.clone_from(&snapshot.phases);
        self.errors.clone_from(&snapshot.errors);
        self.gains.clone_from(&snapshot.gains);
        self.active.clone_from(&snapshot.active);
        self.ages.clone_from(&snapshot.ages);
        self.release_frames.clone_from(&snapshot.release_frames);
        self.release_levels.clone_from(&snapshot.release_levels);
        self.frequency_steps.clone_from(&snapshot.frequency_steps);
        self.frequency_remaining
            .clone_from(&snapshot.frequency_remaining);
        self.gain_steps.clone_from(&snapshot.gain_steps);
        self.gain_remaining.clone_from(&snapshot.gain_remaining);
        self.filter_states.clone_from(&snapshot.filter_states);
        self.frame = snapshot.frame;
        self.control_states.clone_from(&snapshot.control_states);
        self.context_kinds.clone_from(&snapshot.context_kinds);
        self.context_states.clone_from(&snapshot.context_states);
        self.voice_part_contexts
            .clone_from(&snapshot.voice_part_contexts);
        self.voice_trigger_contexts
            .clone_from(&snapshot.voice_trigger_contexts);
        Ok(())
    }
}

impl SynthRuntime {
    fn render_owned<'py>(
        &mut self,
        py: Python<'py>,
        frames: usize,
        cutoff_hz: f64,
        q: f64,
        gain_db: f64,
        actions: &[f64],
    ) -> PyResult<Bound<'py, PyArray2<f64>>> {
        let mut output = Array2::zeros((frames, self.routes.ncols()));
        self.render_into(output.view_mut(), cutoff_hz, q, gain_db, actions)?;
        Ok(output.into_pyarray(py))
    }

    fn render_into(
        &mut self,
        mut output: ArrayViewMut2<'_, f64>,
        cutoff_hz: f64,
        q: f64,
        gain_db: f64,
        actions: &[f64],
    ) -> PyResult<()> {
        let frames = output.nrows();
        if frames == 0
            || output.ncols() != self.routes.ncols()
            || !cutoff_hz.is_finite()
            || cutoff_hz < 0.0
            || cutoff_hz >= self.rate / 2.0
            || !q.is_finite()
            || q <= 0.0
            || !gain_db.is_finite()
            || actions
                .chunks_exact(6)
                .zip(actions.chunks_exact(6).skip(1))
                .any(|(a, b)| b[0] < a[0])
        {
            return Err(PyValueError::new_err("Invalid synth runtime block"));
        }
        let channels = self.routes.ncols();
        let gain = 10_f64.powf(gain_db / 20.0);
        let filter = cutoff_hz > 0.0;
        let g = (PI * cutoff_hz / self.rate).tan();
        let a1 = if q < 1.0 {
            q / (q * (1.0 + g * g) + g)
        } else {
            1.0 / (1.0 + g * (g + 1.0 / q))
        };
        let a2 = g * a1;
        let a3 = g * a2;
        output.fill(0.0);
        let mut action = 0;
        for frame in 0..frames {
            while action < actions.len() && actions[action] as usize == frame {
                self.apply_action(&actions[action..action + 6], frames)?;
                action += 6;
            }
            for voice in 0..self.frequencies.len() {
                if !self.active[voice] {
                    continue;
                }
                let (amplitude, tuning_cents) = self.parameters(voice)?;
                let tuning = 2_f64.powf(tuning_cents / 1200.0);
                let age = self.ages[voice] as f64;
                let level = if let Some(release_frame) = self.release_frames[voice] {
                    let end = release_frame + total_frames(&self.release);
                    if age >= end.ceil() {
                        self.active[voice] = false;
                        continue;
                    }
                    if age >= release_frame.ceil() {
                        envelope_value(
                            self.release_levels[voice],
                            &self.release,
                            age - release_frame,
                        )
                    } else {
                        envelope_value(self.initial, &self.attack, age)
                    }
                } else {
                    envelope_value(self.initial, &self.attack, age)
                };
                let angle = TAU * self.phases[voice] / self.rate;
                let phase = (angle / TAU).rem_euclid(1.0);
                let wave = match self.waveform {
                    0 => angle.sin(),
                    1 => {
                        if phase < self.duty {
                            1.0
                        } else {
                            -1.0
                        }
                    }
                    _ if self.duty == 0.0 => 1.0 - 2.0 * phase,
                    _ if self.duty == 1.0 => 2.0 * phase - 1.0,
                    _ if phase < self.duty => 2.0 * phase / self.duty - 1.0,
                    _ => (1.0 + self.duty - 2.0 * phase) / (1.0 - self.duty),
                };
                let sample = wave * self.gains[voice] * level * amplitude;
                let increment = self.frequencies[voice] * tuning - self.errors[voice];
                let total = self.phases[voice] + increment;
                self.errors[voice] = (total - self.phases[voice]) - increment;
                self.phases[voice] = total.rem_euclid(self.rate);
                for channel in 0..channels {
                    output[[frame, channel]] += sample * self.routes[[voice, channel]];
                }
                self.ages[voice] += 1;
                advance_ramp(
                    &mut self.frequencies[voice],
                    self.frequency_steps[voice],
                    &mut self.frequency_remaining[voice],
                );
                advance_ramp(
                    &mut self.gains[voice],
                    self.gain_steps[voice],
                    &mut self.gain_remaining[voice],
                );
            }
            for (source, state) in self.control_states.iter_mut().enumerate() {
                if self.control_definitions[source].scope == 0 {
                    state.elapsed = state.elapsed.saturating_add(1);
                }
            }
            for context in 0..self.context_kinds.len() {
                let kind = self.context_kinds[context];
                if kind == 0 {
                    continue;
                }
                for source in 0..self.control_definitions.len() {
                    if self.control_definitions[source].scope == kind {
                        let state = &mut self.context_states
                            [context * self.control_definitions.len() + source];
                        state.elapsed = state.elapsed.saturating_add(1);
                    }
                }
            }
            for channel in 0..channels {
                if filter {
                    let input = output[[frame, channel]];
                    let s1 = self.filter_states[[channel, 0]];
                    let s2 = self.filter_states[[channel, 1]];
                    let v3 = input - s2;
                    let v1 = a1 * s1 + a2 * v3;
                    let v2 = s2 + a2 * s1 + a3 * v3;
                    self.filter_states[[channel, 0]] = 2.0 * v1 - s1;
                    self.filter_states[[channel, 1]] = 2.0 * v2 - s2;
                    output[[frame, channel]] = v2 * gain;
                } else {
                    output[[frame, channel]] *= gain;
                }
            }
            self.frame = self
                .frame
                .checked_add(1)
                .ok_or_else(|| PyValueError::new_err("Synth runtime frame overflow"))?;
        }
        self.retire_trigger_contexts();
        Ok(())
    }

    fn apply_action(&mut self, action: &[f64], frames: usize) -> PyResult<()> {
        let offset = action[0] as usize;
        let kind = action[1] as usize;
        let voice = action[2] as usize;
        if action[0] != offset as f64
            || offset >= frames
            || action[1] != kind as f64
            || action[2] != voice as f64
        {
            return Err(PyValueError::new_err("Invalid synth runtime action"));
        }
        if (kind <= 3 || kind == 5) && voice >= self.frequencies.len() {
            return Err(PyValueError::new_err("Invalid synth runtime voice"));
        }
        match kind {
            0 => {
                if action[3] <= 0.0 || action[4] < 0.0 || self.active[voice] {
                    return Err(PyValueError::new_err("Invalid voice start"));
                }
                self.active[voice] = true;
                self.phases[voice] = action[5].rem_euclid(self.rate);
                self.errors[voice] = 0.0;
                self.frequencies[voice] = action[3];
                self.gains[voice] = action[4];
                self.ages[voice] = 0;
                self.release_frames[voice] = None;
                self.voice_part_contexts[voice] = usize::MAX;
                self.voice_trigger_contexts[voice] = usize::MAX;
                self.frequency_remaining[voice] = 0;
                self.gain_remaining[voice] = 0;
            }
            1 => {
                if self.active[voice] && self.release_frames[voice].is_none() {
                    let release_frame = (self.ages[voice] as f64).max(self.minimum_hold_frames);
                    self.release_levels[voice] =
                        envelope_value(self.initial, &self.attack, release_frame);
                    self.release_frames[voice] = Some(release_frame);
                }
            }
            2 => self.active[voice] = false,
            3 => {
                let duration = action[5] as usize;
                if action[3] <= 0.0 || action[4] < 0.0 || action[5] != duration as f64 {
                    return Err(PyValueError::new_err("Invalid voice control"));
                }
                self.frequency_steps[voice] =
                    (action[3] - self.frequencies[voice]) / duration.max(1) as f64;
                self.frequency_remaining[voice] = duration;
                self.gain_steps[voice] = (action[4] - self.gains[voice]) / duration.max(1) as f64;
                self.gain_remaining[voice] = duration;
                if duration == 0 {
                    self.frequencies[voice] = action[3];
                    self.gains[voice] = action[4];
                }
            }
            4 => {
                let Some(definition) = self.control_definitions.get(voice).cloned() else {
                    return Err(PyValueError::new_err("Invalid synth runtime control"));
                };
                if action[3] < definition.minimum || action[3] > definition.maximum {
                    return Err(PyValueError::new_err("Invalid synth control value"));
                }
                let context = decode_context(action[4], self.context_kinds.len())?;
                let state = self.control_state_mut(voice, context)?;
                state.start = control_value(&definition, state);
                state.target = action[3];
                state.elapsed = 0;
            }
            5 => {
                let part = decode_context(action[3], self.context_kinds.len())?;
                let trigger = decode_context(action[4], self.context_kinds.len())?;
                if part.is_some_and(|context| self.context_kinds[context] != 1)
                    || trigger.is_some_and(|context| self.context_kinds[context] != 2)
                {
                    return Err(PyValueError::new_err("Invalid voice control contexts"));
                }
                self.voice_part_contexts[voice] = part.unwrap_or(usize::MAX);
                self.voice_trigger_contexts[voice] = trigger.unwrap_or(usize::MAX);
            }
            6 => {
                if voice >= self.context_kinds.len() || self.context_kinds[voice] != 0 {
                    return Err(PyValueError::new_err("Invalid synth control context"));
                }
                let context_kind = action[3] as usize;
                if action[3] != context_kind as f64 || !(1..=2).contains(&context_kind) {
                    return Err(PyValueError::new_err("Invalid synth context kind"));
                }
                self.context_kinds[voice] = context_kind;
                let sources = self.control_definitions.len();
                for source in 0..sources {
                    let default = self.control_definitions[source].default;
                    self.context_states[voice * sources + source] = ControlState {
                        start: default,
                        target: default,
                        elapsed: 0,
                    };
                }
            }
            7 => {
                let source = action[3] as usize;
                if voice >= self.context_kinds.len()
                    || action[3] != source as f64
                    || source >= self.control_definitions.len()
                    || action[4] < self.control_definitions[source].minimum
                    || action[4] > self.control_definitions[source].maximum
                    || self.context_kinds[voice] != self.control_definitions[source].scope
                {
                    return Err(PyValueError::new_err("Invalid synth context value"));
                }
                let index = voice * self.control_definitions.len() + source;
                self.context_states[index].start = action[4];
                self.context_states[index].target = action[4];
                self.context_states[index].elapsed = 0;
            }
            _ => return Err(PyValueError::new_err("Unknown synth runtime action")),
        }
        Ok(())
    }

    fn parameters(&self, voice: usize) -> PyResult<(f64, f64)> {
        let mut additions = [0.0, 0.0];
        let mut products = [1.0, 1.0];
        for (source, definition) in self.control_definitions.iter().enumerate() {
            let state = match definition.scope {
                0 => &self.control_states[source],
                1 => self.context_state(self.voice_part_contexts[voice], source, 1)?,
                _ => self.context_state(self.voice_trigger_contexts[voice], source, 2)?,
            };
            let amount = definition.intercept + definition.slope * control_value(definition, state);
            if definition.operation == 0 {
                additions[definition.parameter] += amount;
            } else {
                products[definition.parameter] *= amount;
            }
        }
        for definition in &self.lfo_definitions {
            let elapsed = match definition.scope {
                0 => self.frame,
                1 => {
                    self.context_kind(self.voice_part_contexts[voice], 1)?;
                    self.frame
                }
                _ => self.ages[voice],
            };
            let (value, weight) = lfo_value(definition, elapsed, self.rate as u64)?;
            let amount = definition.intercept + definition.slope * value;
            if definition.operation == 0 {
                additions[definition.parameter] += weight * amount;
            } else {
                products[definition.parameter] *= 1.0 + weight * (amount - 1.0);
            }
        }
        let amplitude = (self.parameter_definitions[0] + additions[0]) * products[0];
        let tuning = (self.parameter_definitions[3] + additions[1]) * products[1];
        if amplitude < self.parameter_definitions[1]
            || amplitude > self.parameter_definitions[2]
            || tuning < self.parameter_definitions[4]
            || tuning > self.parameter_definitions[5]
        {
            return Err(PyValueError::new_err(
                "Modulated persistent synth parameter is outside its domain",
            ));
        }
        Ok((amplitude, tuning))
    }

    fn context_state(&self, context: usize, source: usize, kind: usize) -> PyResult<&ControlState> {
        self.context_kind(context, kind)?;
        Ok(&self.context_states[context * self.control_definitions.len() + source])
    }

    fn context_kind(&self, context: usize, kind: usize) -> PyResult<()> {
        if context >= self.context_kinds.len() || self.context_kinds[context] != kind {
            return Err(PyValueError::new_err(
                "Voice is missing its control context",
            ));
        }
        Ok(())
    }

    fn control_state_mut(
        &mut self,
        source: usize,
        context: Option<usize>,
    ) -> PyResult<&mut ControlState> {
        let scope = self.control_definitions[source].scope;
        if scope == 0 && context.is_none() {
            return Ok(&mut self.control_states[source]);
        }
        let Some(context) = context else {
            return Err(PyValueError::new_err(
                "Control action is missing its context",
            ));
        };
        if self.context_kinds[context] != scope {
            return Err(PyValueError::new_err(
                "Control action has the wrong context",
            ));
        }
        Ok(&mut self.context_states[context * self.control_definitions.len() + source])
    }

    fn retire_trigger_contexts(&mut self) {
        for context in 0..self.context_kinds.len() {
            if self.context_kinds[context] == 2
                && !self
                    .active
                    .iter()
                    .enumerate()
                    .any(|(voice, active)| *active && self.voice_trigger_contexts[voice] == context)
            {
                self.context_kinds[context] = 0;
            }
        }
    }
}

fn decode_context(value: f64, capacity: usize) -> PyResult<Option<usize>> {
    if value == -1.0 {
        return Ok(None);
    }
    let context = value as usize;
    if value != context as f64 || context >= capacity {
        return Err(PyValueError::new_err("Invalid synth control context"));
    }
    Ok(Some(context))
}

fn control_definitions(
    values: PyReadonlyArray2<'_, f64>,
    smoothing: &[(u64, u64)],
) -> PyResult<(Vec<ControlDefinition>, Vec<ControlState>)> {
    if values.shape()[1] != 8
        || values.shape()[0] != smoothing.len()
        || smoothing.iter().any(|(_, denominator)| *denominator == 0)
        || values.as_array().iter().any(|v| !v.is_finite())
    {
        return Err(PyValueError::new_err("Invalid synth runtime controls"));
    }
    let mut definitions = Vec::with_capacity(values.shape()[0]);
    let mut states = Vec::with_capacity(values.shape()[0]);
    for (row, (smoothing_numerator, smoothing_denominator)) in
        values.as_array().rows().into_iter().zip(smoothing)
    {
        let scope = row[1] as usize;
        let parameter = row[2] as usize;
        let operation = row[3] as usize;
        if row[1] != scope as f64
            || scope > 2
            || row[2] != parameter as f64
            || parameter > 1
            || row[3] != operation as f64
            || operation > 1
            || row[4] > row[5]
            || row[0] < row[4]
            || row[0] > row[5]
        {
            return Err(PyValueError::new_err("Invalid synth runtime control"));
        }
        definitions.push(ControlDefinition {
            default: row[0],
            smoothing_numerator: *smoothing_numerator,
            smoothing_denominator: *smoothing_denominator,
            scope,
            parameter,
            operation,
            minimum: row[4],
            maximum: row[5],
            intercept: row[6],
            slope: row[7],
        });
        states.push(ControlState {
            start: row[0],
            target: row[0],
            elapsed: 0,
        });
    }
    Ok((definitions, states))
}

fn control_value(definition: &ControlDefinition, state: &ControlState) -> f64 {
    if state.elapsed as u128 * definition.smoothing_denominator as u128
        >= definition.smoothing_numerator as u128
    {
        state.target
    } else {
        state.start
            + (state.target - state.start)
                * state.elapsed as f64
                * definition.smoothing_denominator as f64
                / definition.smoothing_numerator as f64
    }
}

fn lfo_definitions(
    values: PyReadonlyArray2<'_, f64>,
    rationals: &[(u64, u64)],
) -> PyResult<Vec<LfoDefinition>> {
    if values.shape()[1] != 6
        || rationals.len() != values.shape()[0] * 5
        || rationals.iter().any(|(_, denominator)| *denominator == 0)
        || values.as_array().iter().any(|v| !v.is_finite())
    {
        return Err(PyValueError::new_err("Invalid synth runtime LFOs"));
    }
    let mut definitions = Vec::with_capacity(values.shape()[0]);
    for (index, row) in values.as_array().rows().into_iter().enumerate() {
        let scope = row[0] as usize;
        let waveform = row[1] as usize;
        let parameter = row[2] as usize;
        let operation = row[3] as usize;
        let exact = &rationals[index * 5..index * 5 + 5];
        if row[0] != scope as f64
            || ![0, 1, 3].contains(&scope)
            || row[1] != waveform as f64
            || waveform > 2
            || row[2] != parameter as f64
            || parameter > 1
            || row[3] != operation as f64
            || operation > 1
            || exact[0].0 > exact[0].1
            || exact[2].0 >= exact[2].1
        {
            return Err(PyValueError::new_err("Invalid synth runtime LFO"));
        }
        definitions.push(LfoDefinition {
            scope,
            waveform,
            parameter,
            operation,
            intercept: row[4],
            slope: row[5],
            duty: exact[0],
            rate: exact[1],
            phase: exact[2],
            delay_frames: exact[3],
            fade_frames: exact[4],
        });
    }
    Ok(definitions)
}

fn lfo_value(definition: &LfoDefinition, elapsed: usize, sample_rate: u64) -> PyResult<(f64, f64)> {
    let Some(phase_denominator) = lfo_phase_denominator(definition, sample_rate) else {
        return Err(PyValueError::new_err("Invalid synth runtime LFO phase"));
    };
    let base = modular_multiply(
        modular_multiply(
            definition.phase.0 as u128,
            definition.rate.1 as u128,
            phase_denominator,
        ),
        sample_rate as u128,
        phase_denominator,
    );
    let step = modular_multiply(
        definition.rate.0 as u128,
        definition.phase.1 as u128,
        phase_denominator,
    );
    let phase_numerator = modular_add(
        base,
        modular_multiply(elapsed as u128, step, phase_denominator),
        phase_denominator,
    );
    let rising = phase_numerator * (definition.duty.1 as u128)
        < definition.duty.0 as u128 * phase_denominator;
    let phase = phase_numerator as f64 / phase_denominator as f64;
    let value = match definition.waveform {
        0 => (TAU * phase).sin(),
        1 => {
            if rising {
                1.0
            } else {
                -1.0
            }
        }
        _ if definition.duty.0 == 0 => 1.0 - 2.0 * phase,
        _ if definition.duty.0 == definition.duty.1 => 2.0 * phase - 1.0,
        _ if rising => 2.0 * phase / (definition.duty.0 as f64 / definition.duty.1 as f64) - 1.0,
        _ => {
            let duty = definition.duty.0 as f64 / definition.duty.1 as f64;
            (1.0 + duty - 2.0 * phase) / (1.0 - duty)
        }
    };
    let delayed = (elapsed as u128)
        < (definition.delay_frames.0 as u128).div_ceil(definition.delay_frames.1 as u128);
    let weight = if delayed {
        0.0
    } else if definition.fade_frames.0 == 0 {
        1.0
    } else {
        ((elapsed as f64 - definition.delay_frames.0 as f64 / definition.delay_frames.1 as f64)
            / (definition.fade_frames.0 as f64 / definition.fade_frames.1 as f64))
            .clamp(0.0, 1.0)
    };
    Ok((value.clamp(-1.0, 1.0), weight))
}

fn lfo_phase_denominator(definition: &LfoDefinition, sample_rate: u64) -> Option<u128> {
    let denominator = (definition.phase.1 as u128)
        .checked_mul(definition.rate.1 as u128)?
        .checked_mul(sample_rate as u128)?;
    denominator.checked_mul(definition.duty.0.max(definition.duty.1) as u128)?;
    Some(denominator)
}

fn modular_add(first: u128, second: u128, modulus: u128) -> u128 {
    if first >= modulus - second {
        first - (modulus - second)
    } else {
        first + second
    }
}

fn modular_multiply(mut first: u128, mut second: u128, modulus: u128) -> u128 {
    first %= modulus;
    let mut result = 0;
    while second != 0 {
        if second & 1 != 0 {
            result = modular_add(result, first, modulus);
        }
        first = modular_add(first, first, modulus);
        second >>= 1;
    }
    result
}

fn segments(values: PyReadonlyArray2<'_, f64>) -> PyResult<Vec<Segment>> {
    if values.shape()[1] != 2
        || values.as_array().iter().any(|v| !v.is_finite())
        || values.as_array().rows().into_iter().any(|r| r[0] < 0.0)
    {
        return Err(PyValueError::new_err("Invalid synth runtime envelope"));
    }
    Ok(values
        .as_array()
        .rows()
        .into_iter()
        .map(|r| Segment {
            frames: r[0],
            target: r[1],
        })
        .collect())
}

fn total_frames(segments: &[Segment]) -> f64 {
    segments.iter().map(|s| s.frames).sum()
}

fn envelope_value(initial: f64, segments: &[Segment], elapsed: f64) -> f64 {
    let mut boundary = 0.0;
    let mut entry = initial;
    for segment in segments {
        let end = boundary + segment.frames;
        if elapsed < end {
            return entry + (segment.target - entry) * (elapsed - boundary) / segment.frames;
        }
        boundary = end;
        entry = segment.target;
    }
    entry
}

fn advance_ramp(value: &mut f64, step: f64, remaining: &mut usize) {
    if *remaining > 0 {
        *value += step;
        *remaining -= 1;
    }
}
