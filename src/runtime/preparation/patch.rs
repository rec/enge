//! Compile validated Patch declarations into the existing audio-host layout.

use super::bindings::Context;
use super::{exact, invalid};
use crate::amplitude::frame_time;
use crate::runtime::generators::GeneratorBank;
use crate::runtime::{
    EnvelfoTransformInput, EnvelfoTransformRoute, PatchEventInput, StageStartInput, core_error,
};
use engelfo::time::Time;
use pyo3::prelude::*;
use std::collections::{BTreeMap, BTreeSet};
use ufor::envelfo::{Body, EventAction, PatchNode, SignalOperation};
use ufor::identifier::Identifier;
use ufor::modulation::{Interpolation, Operation};
use ufor::samples::processing::{Binding, GeneratorBinding};
use ufor::voice_start::envelfo_stream_key;

pub(crate) type GeneratorSlots = (
    Vec<(String, usize)>,
    Vec<(String, usize)>,
    Vec<(String, usize)>,
);
pub(crate) type TransformMaps = Vec<(usize, Vec<(f64, f64)>)>;
pub(crate) type Detectors = Vec<((String, String), usize)>;

pub(in crate::runtime) struct PreparedPatch {
    pub(in crate::runtime) nodes: Vec<EnvelfoTransformInput>,
    pub(in crate::runtime) routes: Vec<EnvelfoTransformRoute>,
    pub(in crate::runtime) events: Vec<PatchEventInput>,
    pub(in crate::runtime) starts: Vec<StageStartInput>,
    pub(in crate::runtime) thresholds: Vec<PatchEventInput>,
    pub(in crate::runtime) detectors: Detectors,
    pub(in crate::runtime) mappings: TransformMaps,
}

struct Layout {
    lfos: BTreeMap<String, usize>,
    envelopes: BTreeMap<String, usize>,
    stages: BTreeMap<String, usize>,
    targets: BTreeMap<(String, String), usize>,
}

impl PreparedPatch {
    pub(in crate::runtime) fn prepare(
        bank: &GeneratorBank,
        text: &str,
        documents: &[(String, String)],
        slots: GeneratorSlots,
        targets: Vec<(String, String, usize)>,
        orders: Vec<(String, Vec<String>)>,
        maximum: usize,
    ) -> PyResult<Self> {
        let records = [
            slots.0.len(),
            slots.1.len(),
            slots.2.len(),
            targets.len(),
            orders.len(),
        ]
        .into_iter()
        .chain(orders.iter().map(|(_, v)| v.len()))
        .try_fold(0_usize, |n, v| n.checked_add(v))
        .ok_or_else(|| invalid("Patch metadata size overflow"))?;
        let maximum = maximum
            .checked_sub(records)
            .ok_or_else(|| invalid("Patch preparation exceeds record limit"))?;
        let context = Context::parse(text, documents, maximum)?;
        let layout = Layout::new(bank, &context, slots, targets)?;
        let mut layouts = BTreeMap::new();
        for (name, order) in orders {
            if layouts.insert(name, order).is_some() {
                return Err(invalid("Duplicate Patch layout"));
            }
        }
        let mut result = Self {
            nodes: Vec::new(),
            routes: Vec::new(),
            events: Vec::new(),
            starts: Vec::new(),
            thresholds: Vec::new(),
            detectors: Vec::new(),
            mappings: Vec::new(),
        };
        let mut outputs = BTreeMap::new();
        for name in &context.order {
            let Body::Patch(parent) = context.generators[name].body() else {
                continue;
            };
            let order = layouts
                .remove(name.as_str())
                .ok_or_else(|| invalid("Missing Patch layout"))?;
            let nodes: BTreeMap<_, _> = parent
                .envelfos()
                .iter()
                .map(|(n, b)| (n.as_str(), b))
                .collect();
            let mut seen = BTreeSet::new();
            let mut children = BTreeMap::new();
            for child in &order {
                let node = nodes
                    .get(child.as_str())
                    .ok_or_else(|| invalid("Unknown Patch layout child"))?;
                if !seen.insert(child.as_str()) {
                    return Err(invalid("Duplicate Patch layout child"));
                }
                let reference = format!("patch-{}-{child}", name.as_str());
                let input = match node {
                    PatchNode::Generator(generator) => match generator.body() {
                        Body::Lfo(v) => layout
                            .lfos
                            .get(&reference)
                            .map(|i| (0, *i, v.center(), v.depth())),
                        Body::Envelope(_) | Body::SampleHold(_) => {
                            layout.envelopes.get(&reference).map(|i| (1, *i, 0.0, 1.0))
                        }
                        Body::Stages(_) => layout.stages.get(&reference).map(|i| (2, *i, 0.0, 1.0)),
                        _ => unreachable!("validated Patch child"),
                    },
                    PatchNode::Signal(value) => {
                        if value
                            .inputs()
                            .iter()
                            .any(|n| !seen.contains(n.as_str()) || n.as_str() == child)
                        {
                            return Err(invalid("Patch layout must follow signal dependencies"));
                        }
                        if value
                            .inputs()
                            .iter()
                            .any(|n| !children.contains_key(n.as_str()))
                        {
                            continue;
                        }
                        let inputs = value
                            .inputs()
                            .iter()
                            .map(|n| children[n.as_str()])
                            .collect();
                        let (kind, scale, offset, initial) = match value.operation() {
                            SignalOperation::Sum => (0, 1.0, 0.0, None),
                            SignalOperation::Product => (1, 1.0, 0.0, None),
                            SignalOperation::Affine { scale, offset } => (2, *scale, *offset, None),
                            SignalOperation::Threshold { lower, upper } => {
                                (3, *lower, *upper, None)
                            }
                            SignalOperation::Slew {
                                rise,
                                fall,
                                initial,
                            } => (
                                4,
                                rise * (1.0 / bank.rate),
                                fall * (1.0 / bank.rate),
                                *initial,
                            ),
                            SignalOperation::Quantize { step, origin } => (5, *step, *origin, None),
                            SignalOperation::Latch => (6, 1.0, 0.0, None),
                        };
                        let index = result.nodes.len();
                        if kind == 3 || kind == 6 {
                            result
                                .detectors
                                .push(((name.as_str().to_owned(), child.clone()), index));
                        }
                        result.nodes.push((kind, inputs, scale, offset, initial));
                        Some((3, index, 0.0, 1.0))
                    }
                };
                if let Some(input) = input {
                    children.insert(child.as_str(), input);
                }
            }
            if seen.len() != nodes.len() {
                return Err(invalid("Incomplete Patch layout"));
            }
            let stateful = parent.envelfos().iter().any(|(_, n)| matches!(n,
                PatchNode::Signal(s) if matches!(s.operation(), SignalOperation::Latch | SignalOperation::Threshold { .. } | SignalOperation::Slew { .. })));
            for (output, child) in parent.outputs() {
                if let Some(input) = children.get(child.as_str()) {
                    let key = (name.as_str().to_owned(), output.as_str().to_owned());
                    if input.0 == 3 {
                        outputs.insert(key, input.1);
                    } else if stateful {
                        result.nodes.push((2, vec![*input], 1.0, 0.0, None));
                        outputs.insert(key, result.nodes.len() - 1);
                    }
                }
            }
        }
        if !layouts.is_empty() {
            return Err(invalid("Layout requires a Patch declaration"));
        }
        let bindings: BTreeMap<_, _> = context.bindings.iter().map(|b| (b.name(), b)).collect();
        let mut routes: Vec<_> = context.modulation.routes().iter().collect();
        routes.sort_by(|a, b| (a.source(), a.name()).cmp(&(b.source(), b.name())));
        for route in routes {
            let Binding::Generator(binding) = bindings[route.source()] else {
                continue;
            };
            let Some(output) = binding.output() else {
                continue;
            };
            let index = outputs
                .get(&(
                    binding.reference().as_str().to_owned(),
                    output.as_str().to_owned(),
                ))
                .ok_or_else(|| invalid("Patch output has no prepared source"))?;
            let parameter = layout.target(
                route.target().name().as_str(),
                route.target().parameter().as_str(),
            )?;
            result.routes.push((
                *index,
                parameter,
                usize::from(route.operation() == Operation::Multiply),
            ));
            result.mappings.push((
                match route.interpolation() {
                    Interpolation::Linear => 0,
                    Interpolation::Step => 1,
                    Interpolation::Exponential => 2,
                },
                route
                    .points()
                    .iter()
                    .map(|p| (p.input(), p.amount()))
                    .collect(),
            ));
        }
        result.connections(bank, &context, &layout)?;
        Ok(result)
    }
}

impl PreparedPatch {
    fn connections(
        &mut self,
        bank: &GeneratorBank,
        context: &Context,
        layout: &Layout,
    ) -> PyResult<()> {
        let bindings: BTreeMap<_, _> = context.bindings.iter().map(|b| (b.name(), b)).collect();
        let detectors: BTreeMap<_, _> = self.detectors.iter().cloned().collect();
        for name in &context.order {
            let Body::Patch(parent) = context.generators[name].body() else {
                continue;
            };
            let bound = context
                .bindings
                .iter()
                .any(|b| matches!(b, Binding::Generator(b) if b.reference() == name));
            if parent.events().first().is_some_and(|e| {
                !context.generators.keys().any(|n| {
                    n.as_str() == format!("patch-{}-{}", name.as_str(), e.target().as_str())
                })
            }) && !bound
            {
                continue;
            }
            for (order, connection) in parent.events().iter().enumerate() {
                if !matches!(
                    connection.action(),
                    EventAction::Start | EventAction::Sample | EventAction::Capture
                ) {
                    continue;
                }
                let (source, port) = connection
                    .source()
                    .split_once('.')
                    .expect("validated Patch event");
                let node = parent
                    .envelfos()
                    .iter()
                    .find(|(n, _)| n.as_str() == source)
                    .map(|(_, v)| v)
                    .expect("validated Patch event source");
                if matches!(node, PatchNode::Signal(_)) {
                    continue;
                }
                let capture = connection.action() == EventAction::Capture;
                let target = format!("patch-{}-{}", name.as_str(), connection.target().as_str());
                let destination = if capture {
                    *detectors
                        .get(&(
                            name.as_str().to_owned(),
                            connection.target().as_str().to_owned(),
                        ))
                        .ok_or_else(|| invalid("Missing prepared capture target"))?
                } else {
                    layout.envelope(&target)?
                };
                let identity = if capture {
                    format!(
                        "{}:capture-{}-{order}",
                        name.as_str(),
                        connection.target().as_str()
                    )
                } else {
                    format!("{target}:start-{order}")
                };
                let (every, offset, probability) = policy(
                    connection.every(),
                    connection.offset(),
                    connection.probability(),
                )?;
                let delay = delay(connection.delay(), bank.rate, false)?;
                let reference = format!("patch-{}-{source}", name.as_str());
                match node {
                    PatchNode::Generator(generator) => match generator.body() {
                        Body::Lfo(value) => {
                            let marker = value
                                .markers()
                                .iter()
                                .find(|m| m.name().as_str() == port)
                                .ok_or_else(|| invalid("Unknown LFO marker"))?
                                .position();
                            let source = *layout
                                .lfos
                                .get(&reference)
                                .ok_or_else(|| invalid("Missing prepared marker source"))?;
                            self.events.push((
                                source,
                                (marker.numerator(), marker.denominator()),
                                destination,
                                None,
                                order,
                                every,
                                offset,
                                probability,
                                envelfo_stream_key(0, &identity),
                                delay,
                                capture,
                            ));
                        }
                        Body::Stages(_) => self.starts.push((
                            layout.stage(&reference)?,
                            port.to_owned(),
                            destination,
                            order,
                            every,
                            offset,
                            probability,
                            envelfo_stream_key(0, &identity),
                            delay,
                            capture,
                        )),
                        _ => return Err(invalid("Unsupported Patch event source")),
                    },
                    _ => unreachable!("signal events handled separately"),
                }
            }
        }
        for (order, connection) in context.connections.iter().enumerate() {
            let source = generator_binding(bindings[connection.source()])?;
            let Body::Lfo(value) = context.generators[source.reference()].body() else {
                continue;
            };
            let target = generator_binding(bindings[connection.destination()])?;
            let marker = value
                .markers()
                .iter()
                .find(|m| m.name().as_str() == connection.port())
                .ok_or_else(|| invalid("Unknown LFO marker"))?
                .position();
            let source = *layout
                .lfos
                .get(source.reference().as_str())
                .ok_or_else(|| invalid("Missing prepared marker source"))?;
            let (every, offset, probability) = policy(
                connection.every(),
                connection.offset(),
                connection.probability(),
            )?;
            self.events.push((
                source,
                (marker.numerator(), marker.denominator()),
                layout.stage(target.reference().as_str())?,
                Some(connection.cue().as_str().to_owned()),
                self.events.len(),
                every,
                offset,
                probability,
                envelfo_stream_key(0, &format!("{}:cue-{order}", target.reference().as_str())),
                delay(connection.delay(), bank.rate, false)?,
                false,
            ));
        }
        // Detector creation order is the legacy native order, not map-key order.
        for ((parent_name, child), node) in &self.detectors {
            let parent = context
                .generators
                .get(&Identifier::new(parent_name.clone()).map_err(invalid)?)
                .expect("prepared detector parent");
            let Body::Patch(parent) = parent.body() else {
                unreachable!("prepared detector parent");
            };
            if !parent.envelfos().iter().any(|(n, v)| {
                n.as_str() == child
                    && matches!(v,
                PatchNode::Signal(s) if matches!(s.operation(), SignalOperation::Threshold { .. }))
            }) {
                continue;
            }
            for (order, connection) in parent.events().iter().enumerate() {
                let (source, port) = connection
                    .source()
                    .split_once('.')
                    .expect("validated Patch event");
                if source != child {
                    continue;
                }
                let capture = connection.action() == EventAction::Capture;
                let target = format!("patch-{parent_name}-{}", connection.target().as_str());
                let cue = connection.cue().map(|n| n.as_str().to_owned());
                let destination = if capture {
                    *detectors
                        .get(&(parent_name.clone(), connection.target().as_str().to_owned()))
                        .ok_or_else(|| invalid("Missing prepared capture target"))?
                } else if cue.is_some() {
                    layout.stage(&target)?
                } else {
                    layout.envelope(&target)?
                };
                let identity = if capture {
                    format!(
                        "{parent_name}:capture-{}-{order}",
                        connection.target().as_str()
                    )
                } else {
                    format!(
                        "{target}:{}-{order}",
                        if cue.is_some() { "cue" } else { "start" }
                    )
                };
                let (every, offset, probability) = policy(
                    connection.every(),
                    connection.offset(),
                    connection.probability(),
                )?;
                self.thresholds.push((
                    *node,
                    (i64::from(port == "rising"), 1),
                    destination,
                    cue,
                    order,
                    every,
                    offset,
                    probability,
                    envelfo_stream_key(0, &identity),
                    delay(connection.delay(), bank.rate, true)?,
                    capture,
                ));
            }
            for (order, connection) in context.connections.iter().enumerate() {
                let binding = generator_binding(bindings[connection.source()])?;
                if binding.reference().as_str() != parent_name {
                    continue;
                }
                let source = parent
                    .event_outputs()
                    .iter()
                    .find(|(n, _)| n.as_str() == connection.port())
                    .map(|(_, v)| v)
                    .ok_or_else(|| invalid("Unknown Patch event output"))?;
                let (origin, port) = source.split_once('.').expect("validated Patch output");
                if origin != child {
                    continue;
                }
                let target = generator_binding(bindings[connection.destination()])?;
                let (every, offset, probability) = policy(
                    connection.every(),
                    connection.offset(),
                    connection.probability(),
                )?;
                self.thresholds.push((
                    *node,
                    (i64::from(port == "rising"), 1),
                    layout.stage(target.reference().as_str())?,
                    Some(connection.cue().as_str().to_owned()),
                    order,
                    every,
                    offset,
                    probability,
                    envelfo_stream_key(0, &format!("{}:cue-{order}", target.reference().as_str())),
                    delay(connection.delay(), bank.rate, true)?,
                    false,
                ));
            }
        }
        Ok(())
    }
}

fn generator_binding(value: &Binding) -> PyResult<&GeneratorBinding> {
    match value {
        Binding::Generator(v) => Ok(v),
        _ => Err(invalid("Event connection requires a generator binding")),
    }
}

fn policy(every: i64, offset: i64, probability: f64) -> PyResult<(usize, usize, u64)> {
    Ok((
        usize::try_from(every).map_err(invalid)?,
        usize::try_from(offset).map_err(invalid)?,
        (probability * (1_u64 << 53) as f64) as u64,
    ))
}

fn delay(value: ufor::number::Rational, rate: f64, threshold: bool) -> PyResult<(i64, i64)> {
    let mut frames = exact(value)?
        .checked_mul(frame_time(rate)?)
        .map_err(core_error)?;
    if threshold {
        frames = frames.checked_add(Time::ONE).map_err(core_error)?;
    }
    Ok((frames.numerator(), frames.denominator()))
}

impl Layout {
    fn new(
        bank: &GeneratorBank,
        context: &Context,
        slots: GeneratorSlots,
        targets: Vec<(String, String, usize)>,
    ) -> PyResult<Self> {
        let mut maps = Vec::new();
        for (kind, (rows, count)) in [
            (slots.0, bank.lfo_definitions.len()),
            (slots.1, bank.named_envelopes.len()),
            (slots.2, bank.staged_envelfos.len()),
        ]
        .into_iter()
        .enumerate()
        {
            let mut result = BTreeMap::new();
            for (name, index) in rows {
                let body = context
                    .generators
                    .get(&Identifier::new(name.clone()).map_err(invalid)?)
                    .ok_or_else(|| invalid("Unknown generator slot name"))?
                    .body();
                if index >= count
                    || !matches!(
                        (kind, body),
                        (0, Body::Lfo(_))
                            | (1, Body::Envelope(_) | Body::SampleHold(_))
                            | (2, Body::Stages(_))
                    )
                    || result.insert(name, index).is_some()
                {
                    return Err(invalid("Invalid or duplicate generator slot metadata"));
                }
            }
            maps.push(result);
        }
        let mut parameters = BTreeMap::new();
        for (name, parameter, index) in targets {
            if index >= bank.parameter_count
                || parameters.insert((name, parameter), index).is_some()
            {
                return Err(invalid("Invalid or duplicate target index"));
            }
        }
        let result = Self {
            lfos: maps.remove(0),
            envelopes: maps.remove(0),
            stages: maps.remove(0),
            targets: parameters,
        };
        for parameter in context.modulation.parameters() {
            result.target(
                parameter.target().name().as_str(),
                parameter.target().parameter().as_str(),
            )?;
        }
        Ok(result)
    }

    fn target(&self, name: &str, parameter: &str) -> PyResult<usize> {
        self.targets
            .get(&(name.to_owned(), parameter.to_owned()))
            .copied()
            .ok_or_else(|| invalid("Modulation target has no host parameter index"))
    }

    fn envelope(&self, name: &str) -> PyResult<usize> {
        self.envelopes
            .get(name)
            .copied()
            .ok_or_else(|| invalid("Missing prepared Envelope or SampleHold"))
    }

    fn stage(&self, name: &str) -> PyResult<usize> {
        self.stages
            .get(name)
            .copied()
            .ok_or_else(|| invalid("Missing prepared Stages"))
    }
}
