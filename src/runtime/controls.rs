//! Shared bounded control ramps, addressed contexts and voice bindings.
use super::{RouteMap, add_partial};
use numpy::{PyReadonlyArray2, PyUntypedArrayMethods};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

#[derive(Clone, PartialEq)]
pub(crate) struct ControlDefinition {
    pub(crate) default: f64,
    pub(crate) smoothing_numerator: u64,
    pub(crate) smoothing_denominator: u64,
    pub(crate) scope: usize,
    pub(crate) parameter: usize,
    pub(crate) operation: usize,
    pub(crate) minimum: f64,
    pub(crate) maximum: f64,
}

#[derive(Clone)]
pub(crate) struct ControlState {
    pub(crate) start: f64,
    pub(crate) target: f64,
    pub(crate) elapsed: usize,
}

#[derive(Clone)]
pub(crate) struct ControlBank {
    pub(crate) control_definitions: Vec<ControlDefinition>,
    pub(crate) control_states: Vec<ControlState>,
    pub(crate) context_kinds: Vec<usize>,
    pub(crate) context_states: Vec<ControlState>,
    pub(crate) voice_part_contexts: Vec<usize>,
    pub(crate) voice_trigger_contexts: Vec<usize>,
}

impl ControlBank {
    pub(crate) fn contributions(
        &self,
        voice: usize,
        parameter: usize,
        maps: &[RouteMap],
        partials: &mut [f64],
        count: &mut usize,
    ) -> PyResult<f64> {
        let mut product = 1.0;
        for (source, definition) in self.control_definitions.iter().enumerate() {
            if definition.parameter != parameter {
                continue;
            }
            let mapping = maps
                .get(source)
                .ok_or_else(|| PyValueError::new_err("Missing modulation mapping"))?;
            let amount = mapping.value(self.value(voice, source)?)?;
            if definition.operation == 0 {
                add_partial(partials, count, amount);
            } else {
                product *= amount;
            }
        }
        Ok(product)
    }

    pub(crate) fn new(
        control_definitions: Vec<ControlDefinition>,
        control_states: Vec<ControlState>,
        voices: usize,
        capacity: usize,
    ) -> Self {
        let context_states = (0..capacity).flat_map(|_| control_states.clone()).collect();
        Self {
            control_definitions,
            control_states,
            context_states,
            context_kinds: vec![0; capacity],
            voice_part_contexts: vec![usize::MAX; voices],
            voice_trigger_contexts: vec![usize::MAX; voices],
        }
    }

    pub(crate) fn value(&self, voice: usize, source: usize) -> PyResult<f64> {
        let definition = &self.control_definitions[source];
        let state = match definition.scope {
            0 => &self.control_states[source],
            1 => self.context_state(self.voice_part_contexts[voice], source, 1)?,
            _ => self.context_state(self.voice_trigger_contexts[voice], source, 2)?,
        };
        Ok(control_value(definition, state))
    }

    pub(crate) fn advance(&mut self) {
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
                    let state =
                        &mut self.context_states[context * self.control_definitions.len() + source];
                    state.elapsed = state.elapsed.saturating_add(1);
                }
            }
        }
    }

    pub(crate) fn apply(&mut self, action: &[f64]) -> PyResult<()> {
        let kind = action[1] as usize;
        let voice = action[2] as usize;
        if action.iter().any(|v| !v.is_finite())
            || action[1] != kind as f64
            || action[2] != voice as f64
            || (kind == 5 && voice >= self.voice_part_contexts.len())
        {
            return Err(PyValueError::new_err("Invalid native control action"));
        }
        match kind {
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

            _ => return Err(PyValueError::new_err("Unknown native control action")),
        }
        Ok(())
    }
    fn context_state(&self, context: usize, source: usize, kind: usize) -> PyResult<&ControlState> {
        self.context_kind(context, kind)?;
        Ok(&self.context_states[context * self.control_definitions.len() + source])
    }

    pub(crate) fn context_kind(&self, context: usize, kind: usize) -> PyResult<()> {
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

    pub(crate) fn retire_trigger_contexts(&mut self, active: &[bool]) {
        for context in 0..self.context_kinds.len() {
            if self.context_kinds[context] == 2
                && !active
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

pub(crate) fn control_definitions(
    values: PyReadonlyArray2<'_, f64>,
    smoothing: &[(u64, u64)],
) -> PyResult<(Vec<ControlDefinition>, Vec<ControlState>)> {
    if values.shape()[1] != 6
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
