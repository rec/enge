//! Compile effect control tables from authored modulation declarations.
use super::runtime::controls::Definitions;
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use std::collections::{BTreeMap, BTreeSet};
use ufor::codec::{self, Format};
use ufor::control::Scope;
use ufor::envelfo::{PreparationLimits, preparation_records};
use ufor::identifier::Identifier;
use ufor::modulation::{Interpolation, Modulation, Operation};

#[pyfunction]
#[pyo3(signature = (document, processors, slots, followers, *, max_records=10000))]
pub(crate) fn prepare_effect_controls(
    document: &str,
    processors: Vec<(usize, String)>,
    slots: Vec<(usize, usize)>,
    followers: Vec<(String, usize)>,
    max_records: usize,
) -> PyResult<Definitions> {
    let limits = PreparationLimits::new(max_records).map_err(invalid)?;
    let mut records = slots
        .len()
        .checked_add(followers.len())
        .and_then(|n| n.checked_add(processors.len()))
        .ok_or_else(|| invalid("effect preparation record overflow"))?;
    limits.check_records(records).map_err(invalid)?;
    let characters = std::iter::once(document)
        .chain(processors.iter().map(|(_, text)| text.as_str()))
        .flat_map(str::chars)
        .take(codec::MAX_SCORE_CHARACTERS + 1)
        .count();
    if characters > codec::MAX_SCORE_CHARACTERS {
        return Err(invalid(
            "effect preparation exceeds document character limit",
        ));
    }
    let data = codec::parse_value(document, Format::Json).map_err(invalid)?;
    let remaining = PreparationLimits::new(max_records - records).map_err(invalid)?;
    records += preparation_records(&data, remaining).map_err(invalid)?;
    let modulation = Modulation::decode(data, max_records).map_err(invalid)?;
    if slots.len() != modulation.parameters().len() {
        return Err(invalid("effect slots must match modulation parameters"));
    }
    let mut definitions = BTreeMap::new();
    let mut names = BTreeSet::new();
    for (node, text) in processors {
        let available = PreparationLimits::new(max_records - records).map_err(invalid)?;
        let definition =
            codec::parse_effect_processor(&text, Format::Json, available).map_err(invalid)?;
        let data = serde_json::to_value(&definition).map_err(invalid)?;
        records += preparation_records(&data, available).map_err(invalid)?;
        if !names.insert(definition.name().clone())
            || definitions.insert(node, definition).is_some()
        {
            return Err(invalid("duplicate effect processor slot or name"));
        }
    }
    let mut addresses = BTreeMap::new();
    let mut targets = Vec::with_capacity(slots.len());
    let mut parameters_by_node = BTreeMap::<_, Vec<_>>::new();
    for (parameter, (node, field)) in modulation.parameters().iter().zip(slots) {
        if parameter.scope() != Scope::Voice {
            return Err(invalid("effect parameters belong to one graph instance"));
        }
        if !definitions.contains_key(&node) {
            return Err(invalid("effect target node does not match processor"));
        }
        parameters_by_node.entry(node).or_default().push(parameter);
        addresses.insert(parameter.target(), targets.len());
        targets.push((node, field, parameter.minimum(), parameter.maximum()));
    }
    for (node, parameters) in parameters_by_node {
        definitions[&node]
            .validate_parameters(parameters)
            .map_err(invalid)?;
    }
    let sources: BTreeMap<_, _> = modulation
        .sources()
        .iter()
        .map(|source| (source.name(), source))
        .collect();
    let mut follower_nodes = BTreeMap::new();
    for (name, node) in followers {
        let name = Identifier::new(name).map_err(invalid)?;
        if !sources.contains_key(&name)
            || follower_nodes
                .insert(name, i64::try_from(node).map_err(invalid)?)
                .is_some()
        {
            return Err(invalid("unknown or duplicate effect follower source"));
        }
    }
    let mut routes: Vec<_> = modulation.routes().iter().collect();
    routes.sort_by(|a, b| a.source().cmp(b.source()).then(a.name().cmp(b.name())));
    let routes = routes
        .into_iter()
        .map(|route| {
            let source = sources[route.source()];
            (
                addresses[route.target()],
                follower_nodes.get(route.source()).copied().unwrap_or(-1),
                u8::from(route.operation() == Operation::Multiply),
                match route.interpolation() {
                    Interpolation::Linear => 0,
                    Interpolation::Step => 1,
                    Interpolation::Exponential => 2,
                },
                route
                    .points()
                    .iter()
                    .map(|point| (point.input(), point.amount()))
                    .collect(),
                source.minimum(),
                source.maximum(),
                route.name().as_str().to_owned(),
            )
        })
        .collect();
    Ok((targets, routes))
}

fn invalid(error: impl std::fmt::Display) -> PyErr {
    PyValueError::new_err(format!("Effect control preparation: {error}"))
}
