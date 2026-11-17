//! Validate real shared declarations without claiming processor-target validity.

use super::invalid;
use pyo3::prelude::*;
use serde_json::Value;
use std::collections::BTreeMap;
use ufor::codec::{self, Format};
use ufor::envelfo::{self, Body, Generator, PreparationLimits};
use ufor::identifier::Identifier;
use ufor::modulation::Modulation;
use ufor::samples::controls::ControlDeclaration;
use ufor::samples::processing::{Binding, EnvelfoEventConnection};

pub(super) struct Context {
    pub(super) modulation: Modulation,
    pub(super) bindings: Vec<Binding>,
    pub(super) connections: Vec<EnvelfoEventConnection>,
    pub(super) generators: BTreeMap<Identifier, Generator>,
    pub(super) order: Vec<Identifier>,
}

/// Validate the supplied generator facts; Python retains processor-target checks.
#[pyfunction]
#[pyo3(signature = (context, generators, *, max_records=10000))]
pub(crate) fn validate_generator_bindings(
    context: String,
    generators: Vec<(String, String)>,
    max_records: usize,
) -> PyResult<()> {
    Context::parse(&context, &generators, max_records)?;
    Ok(())
}

impl Context {
    pub(super) fn parse(
        text: &str,
        documents: &[(String, String)],
        maximum: usize,
    ) -> PyResult<Self> {
        let limits = PreparationLimits::new(maximum).map_err(invalid)?;
        let characters = documents
            .iter()
            .try_fold(text.chars().count(), |n, (_, t)| {
                n.checked_add(t.chars().count())
                    .ok_or_else(|| invalid("Preparation text overflow"))
            })?;
        if characters > codec::MAX_SCORE_CHARACTERS {
            return Err(invalid("Preparation exceeds document character limit"));
        }
        let document = codec::parse_value(text, Format::Json).map_err(invalid)?;
        let mut records = envelfo::preparation_records(&document, limits).map_err(invalid)?;
        let Value::Object(mut fields) = document else {
            return Err(invalid("Binding declarations require an object"));
        };
        let modulation = Modulation::decode(
            fields
                .remove("modulation")
                .ok_or_else(|| invalid("Missing modulation declarations"))?,
            maximum,
        )
        .map_err(invalid)?;
        let bindings: Vec<Binding> = serde_json::from_value(
            fields
                .remove("bindings")
                .ok_or_else(|| invalid("Missing binding declarations"))?,
        )
        .map_err(invalid)?;
        let controls: BTreeMap<Identifier, ControlDeclaration> = serde_json::from_value(
            fields
                .remove("controls")
                .ok_or_else(|| invalid("Missing control declarations"))?,
        )
        .map_err(invalid)?;
        let connections: Vec<EnvelfoEventConnection> = serde_json::from_value(
            fields
                .remove("event_connections")
                .ok_or_else(|| invalid("Missing event connections"))?,
        )
        .map_err(invalid)?;
        if !fields.is_empty() {
            return Err(invalid("Unknown binding declaration field"));
        }
        let mut generators = BTreeMap::new();
        let mut order = Vec::with_capacity(documents.len());
        for (name, text) in documents {
            let name = Identifier::new(name.clone()).map_err(invalid)?;
            let remaining = maximum
                .checked_sub(records)
                .ok_or_else(|| invalid("Preparation exceeds record limit"))?;
            let generator = codec::parse_generator(
                text,
                Format::Json,
                PreparationLimits::new(remaining).map_err(invalid)?,
            )
            .map_err(invalid)?;
            records += envelfo::preparation_records(
                &serde_json::to_value(&generator).map_err(invalid)?,
                limits,
            )
            .map_err(invalid)?;
            if matches!(generator.body(), Body::Reference(_)) {
                return Err(invalid(
                    "Generator references must be materialized before preparation",
                ));
            }
            order.push(name.clone());
            if generators.insert(name, generator).is_some() {
                return Err(invalid("Duplicate generator declaration"));
            }
        }
        let checks = ufor::samples::bindings::validate_generator_bindings(
            &modulation,
            &bindings,
            &controls,
            &generators,
            &connections,
            limits,
        )
        .map_err(invalid)?;
        if !checks.unresolved_bindings().is_empty() || !checks.unresolved_connections().is_empty() {
            return Err(invalid(
                "Unresolved generator, follower or event requirements",
            ));
        }
        // Actual processor capabilities and target domains remain checked by the
        // Python settings/host profile, not by this partial shared validator.
        Ok(Self {
            modulation,
            bindings,
            connections,
            generators,
            order,
        })
    }
}
