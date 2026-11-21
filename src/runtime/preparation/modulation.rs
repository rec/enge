//! Compile canonical declarations into the host's existing indexed route tables.
use super::bindings::Context;
use super::{exact, invalid};
use crate::amplitude::frame_time;
use crate::runtime::core_error;
use engelfo::random::stream_key;
use numpy::IntoPyArray;
use numpy::ndarray::Array2;
use pyo3::prelude::*;
use pyo3::types::{PyDict, PySet, PyTuple};
use std::collections::{BTreeMap, BTreeSet};
use ufor::codec::{self, Format};
use ufor::control::{Polarity, Scope};
use ufor::envelfo::{self, Body, Initial, PreparationLimits};
use ufor::identifier::Identifier;
use ufor::modulation::{Interpolation, Operation, SourceScope, Target};
use ufor::samples::processing::{
    Binding, EventKind, FilterBoundary, FilterResponse, GeneratorBinding, ReleaseTiming,
    SoundSettings, TargetMinimum, TargetResolution,
};

type TargetInput = (String, String, usize, Option<f64>, f64, f64);
type RouteMaps = Vec<Vec<(usize, Vec<(f64, f64)>)>>;
type StageConnection = (
    usize,
    String,
    usize,
    String,
    usize,
    usize,
    u64,
    u64,
    (i64, i64),
);

/// Resolve authored defaults before applying renderer profile bounds and indices.
#[pyfunction]
#[pyo3(signature = (sample_rate, context, generators, targets, *, sound_settings=None, max_records=10000))]
pub(crate) fn prepare_modulation(
    py: Python<'_>,
    sample_rate: f64,
    context: String,
    generators: Vec<(String, String)>,
    mut targets: Vec<TargetInput>,
    sound_settings: Option<&str>,
    max_records: usize,
) -> PyResult<Py<PyTuple>> {
    let rate = frame_time(sample_rate)?;
    if sample_rate <= 0.0 {
        return Err(invalid("Invalid modulation host metadata"));
    }
    let limits = PreparationLimits::new(max_records).map_err(invalid)?;
    let mut records = 0;
    let sound = if let Some(text) = sound_settings {
        let length = generators
            .iter()
            .map(|(_, document)| document.as_str())
            .chain([context.as_str(), text])
            .flat_map(str::chars)
            .take(codec::MAX_SCORE_CHARACTERS + 1)
            .count();
        if length > codec::MAX_SCORE_CHARACTERS {
            return Err(invalid("Preparation exceeds document character limit"));
        }
        let settings = codec::parse_sound_settings(text, Format::Json, limits).map_err(invalid)?;
        records = envelfo::preparation_records(
            &serde_json::to_value(&settings).map_err(invalid)?,
            limits,
        )
        .map_err(invalid)?;
        Some(settings)
    } else {
        None
    };
    let first_filter = targets.len();
    let filters = match &sound {
        Some(settings) => prepare_filters(settings, sample_rate, &mut targets)?,
        None => Array2::zeros((0, 7)),
    };
    let metadata = targets
        .len()
        .checked_add(filters.nrows())
        .and_then(|n| n.checked_add(records))
        .ok_or_else(|| invalid("Preparation size overflow"))?;
    let remaining = max_records
        .checked_sub(metadata)
        .ok_or_else(|| invalid("Preparation exceeds record limit"))?;
    let declarations = Context::parse(&context, &generators, remaining)?;
    let mut values = vec![
        0.0;
        targets
            .len()
            .checked_mul(3)
            .ok_or_else(|| invalid("Parameter size overflow"))?
    ];
    let mut shared = vec![None; targets.len()];
    let mut indices = BTreeMap::new();
    let mut addresses = BTreeSet::new();
    for (name, parameter, index, host_default, minimum, maximum) in &targets {
        let target = Target::new(
            Identifier::new(name.clone()).map_err(invalid)?,
            Identifier::new(parameter.clone()).map_err(invalid)?,
        );
        if *index >= targets.len()
            || !addresses.insert(*index)
            || !minimum.is_finite()
            || !maximum.is_finite()
            || minimum > maximum
        {
            return Err(invalid("Invalid modulation target metadata"));
        }
        let default = match host_default {
            Some(_) if sound.is_some() && (name == "processing" || name.starts_with("filter-")) => {
                return Err(invalid(
                    "Shared modulation target defaults come from SoundSettings",
                ));
            }
            Some(default) => *default,
            None => {
                let settings = sound
                    .as_ref()
                    .ok_or_else(|| invalid("Missing SoundSettings for shared modulation target"))?;
                let TargetResolution::Resolved(facts) =
                    settings.resolve_target(&target).map_err(invalid)?
                else {
                    return Err(invalid("Unresolved SoundSettings target"));
                };
                shared[*index] = Some(facts);
                if *index >= first_filter && parameter == "cutoff_hz" {
                    facts.default_value.clamp(*minimum, *maximum)
                } else {
                    facts.default_value
                }
            }
        };
        if !default.is_finite()
            || default < *minimum
            || default > *maximum
            || indices.insert(target, *index).is_some()
        {
            return Err(invalid("Invalid modulation target metadata"));
        }
        values[3 * index..3 * index + 3].copy_from_slice(&[default, *minimum, *maximum]);
    }
    for parameter in declarations.modulation.parameters() {
        let index = indices
            .get(parameter.target())
            .ok_or_else(|| invalid("Persistent runtime has an unsupported modulation target"))?;
        if let Some(facts) = shared[*index] {
            if parameter.unit() != facts.unit || parameter.default_value() != facts.default_value {
                return Err(invalid(
                    "Modulation unit and default must match SoundSettings",
                ));
            }
            if match facts.minimum {
                TargetMinimum::Unrestricted => false,
                TargetMinimum::Nonnegative => parameter.minimum() < 0.0,
                TargetMinimum::Positive => parameter.minimum() <= 0.0,
            } {
                return Err(invalid("Modulation domain conflicts with SoundSettings"));
            }
        }
        values[3 * index..3 * index + 3].copy_from_slice(&[
            parameter.default_value(),
            parameter.minimum(),
            parameter.maximum(),
        ]);
    }
    let sources: BTreeMap<_, _> = declarations
        .modulation
        .sources()
        .iter()
        .map(|s| (s.name(), s))
        .collect();
    let bindings: BTreeMap<_, _> = declarations
        .bindings
        .iter()
        .map(|b| (b.name(), b))
        .collect();
    let mut events: BTreeSet<Identifier> = declarations
        .connections
        .iter()
        .flat_map(|c| [c.source().clone(), c.destination().clone()])
        .collect();
    let mut children = BTreeSet::new();
    for (name, generator) in &declarations.generators {
        if let Body::Patch(parent) = generator.body() {
            for (child, _) in parent.envelfos() {
                children.insert(format!("patch-{}-{}", name.as_str(), child.as_str()));
            }
        }
    }
    for binding in &declarations.bindings {
        if let Binding::Generator(binding) = binding {
            if children.contains(binding.reference().as_str()) {
                events.insert(binding.name().clone());
            }
        }
    }
    let mut routes: Vec<_> = declarations.modulation.routes().iter().collect();
    routes.sort_by_key(|r| (r.source(), r.name()));
    let routed: BTreeSet<_> = routes.iter().map(|r| r.source()).collect();
    if sources
        .keys()
        .filter(|n| !routed.contains(**n))
        .copied()
        .collect::<BTreeSet<_>>()
        != events.iter().filter(|n| !routed.contains(n)).collect()
    {
        return Err(invalid(
            "Persistent synth requires routes for modulation sources",
        ));
    }
    let documents: BTreeMap<_, _> = generators
        .iter()
        .map(|(n, t)| (n.as_str(), t.as_str()))
        .collect();
    let mut prepared = Prepared::new(declarations.records + metadata, limits);
    for route in routes {
        let source = sources
            .get(route.source())
            .ok_or_else(|| invalid("Unresolved modulation source"))?;
        let binding = bindings
            .get(route.source())
            .ok_or_else(|| invalid("Unresolved modulation binding"))?;
        let parameter = *indices
            .get(route.target())
            .ok_or_else(|| invalid("Unresolved modulation target"))?;
        let operation = usize::from(route.operation() == Operation::Multiply);
        let mapping = (
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
        );
        let scope = scope_name(source.scope());
        prepared.scopes.insert(scope.to_owned());
        match binding {
            Binding::Event(binding) => {
                prepared.event_bindings.insert(
                    prepared.controls.len(),
                    match binding.kind() {
                        EventKind::Key => "key".to_owned(),
                        EventKind::Velocity => "velocity".to_owned(),
                    },
                );
                prepared.controls.push([
                    source.minimum(),
                    3.0,
                    parameter as f64,
                    operation as f64,
                    source.minimum(),
                    source.maximum(),
                ]);
                prepared.smoothing.push((0, 1));
                prepared.control_scopes.push("voice".to_owned());
                prepared.source_controls.push(String::new());
                prepared.maps[0].push(mapping);
            }
            Binding::Control(binding) => {
                let control = &declarations.controls[binding.control()];
                if source.minimum()
                    > if control.polarity() == Polarity::Bipolar {
                        -1.0
                    } else {
                        0.0
                    }
                    || source.maximum() < 1.0
                {
                    return Err(invalid(
                        "Persistent synth source must cover its control declaration",
                    ));
                }
                let scope_index = match source.scope() {
                    SourceScope::Instrument => 0,
                    SourceScope::Part => 1,
                    SourceScope::Trigger => 2,
                    SourceScope::Voice => return Err(invalid("Control cannot have voice scope")),
                };
                let duration = exact(binding.smoothing())?
                    .checked_mul(rate)
                    .map_err(core_error)?;
                prepared
                    .control_sources
                    .insert(binding.control(), prepared.controls.len());
                prepared.controls.push([
                    control.default_value(),
                    scope_index as f64,
                    parameter as f64,
                    operation as f64,
                    source.minimum(),
                    source.maximum(),
                ]);
                prepared
                    .smoothing
                    .push((duration.numerator() as u64, duration.denominator() as u64));
                prepared.control_scopes.push(scope.to_owned());
                prepared
                    .source_controls
                    .push(binding.control().as_str().to_owned());
                prepared.maps[0].push(mapping);
            }
            Binding::Generator(binding) => {
                if let Some(kind) = prepared.generator(
                    &declarations,
                    &documents,
                    binding,
                    Some(parameter),
                    operation,
                    true,
                )? {
                    prepared.maps[kind].push(mapping);
                }
            }
            Binding::Follower(_) => {
                return Err(invalid("Persistent runtime has an unsupported binding"));
            }
        }
    }
    for binding in &declarations.bindings {
        if !events.contains(binding.name()) || routed.contains(binding.name()) {
            continue;
        }
        let Binding::Generator(binding) = binding else {
            return Err(invalid("Event source requires a generator binding"));
        };
        prepared.generator(&declarations, &documents, binding, None, 1, false)?;
    }
    for (order, connection) in declarations.connections.iter().enumerate() {
        let Binding::Generator(source) = bindings[connection.source()] else {
            continue;
        };
        if !matches!(
            declarations.generators[source.reference()].body(),
            Body::Stages(_)
        ) {
            continue;
        }
        let Binding::Generator(destination) = bindings[connection.destination()] else {
            return Err(invalid("Stage event destination requires a generator"));
        };
        let source = prepared
            .stage_bindings
            .get(connection.source())
            .copied()
            .ok_or_else(|| invalid("Missing event source stage slot"))?;
        let target = prepared
            .stage_bindings
            .get(connection.destination())
            .copied()
            .ok_or_else(|| invalid("Missing event destination stage slot"))?;
        let delay = exact(connection.delay())?
            .checked_mul(rate)
            .map_err(core_error)?;
        prepared.connections.push((
            source,
            connection.port().to_owned(),
            target,
            connection.cue().as_str().to_owned(),
            usize::try_from(connection.every()).map_err(invalid)?,
            usize::try_from(connection.offset()).map_err(invalid)?,
            (connection.probability() * (1_u64 << 53) as f64) as u64,
            stream_key(
                0,
                &format!("{}:cue-{order}", destination.reference().as_str()),
            ),
            (delay.numerator(), delay.denominator()),
        ));
    }
    for (mappings, count) in prepared.maps.iter_mut().zip([
        prepared.controls.len(),
        prepared.lfos.len(),
        prepared.envelopes.len(),
        prepared.stages.len(),
    ]) {
        mappings.resize_with(count, || (0, vec![(0.0, 1.0)]));
    }
    prepared.into_python(
        py,
        values,
        filters,
        context,
        generators,
        targets
            .into_iter()
            .map(|(n, p, i, _, _, _)| (n, p, i))
            .collect(),
    )
}

// Filter placement, runtime codes and rate-dependent limits belong to enge.
fn prepare_filters(
    sound: &SoundSettings,
    sample_rate: f64,
    targets: &mut Vec<TargetInput>,
) -> PyResult<Array2<f64>> {
    let filters = sound.processing().filters();
    let mut rows = Vec::new();
    for filter in filters {
        let minimum = filter.minimum_hz();
        let maximum = sample_rate / 2.0 * filter.nyquist_ratio();
        if minimum >= maximum {
            return Err(invalid("Filter minimum_hz must be below its Nyquist bound"));
        }
        let cutoff = targets.len();
        let q = cutoff + 1;
        let name = format!("filter-{}", filter.name().as_str());
        targets.push((
            name.clone(),
            "cutoff_hz".into(),
            cutoff,
            None,
            minimum,
            maximum,
        ));
        targets.push((name, "q".into(), q, None, f64::MIN_POSITIVE, 1e300));
        let response = match filter.response() {
            FilterResponse::Lowpass => 0.0,
            FilterResponse::Highpass => 1.0,
            FilterResponse::Bandpass => 2.0,
            FilterResponse::Notch => 3.0,
        };
        let boundary = match filter.boundary() {
            FilterBoundary::Error => 0.0,
            FilterBoundary::Clamp => 1.0,
        };
        rows.extend([
            response,
            f64::from(filter.stages()),
            boundary,
            minimum,
            maximum,
            cutoff as f64,
            q as f64,
        ]);
    }
    Array2::from_shape_vec((filters.len(), 7), rows).map_err(invalid)
}

#[derive(Default)]
struct SourceMap {
    order: Vec<String>,
    indices: BTreeMap<String, Vec<usize>>,
}
impl SourceMap {
    fn insert(&mut self, name: &Identifier, index: usize) {
        let name = name.as_str();
        if !self.indices.contains_key(name) {
            self.order.push(name.to_owned());
        }
        self.indices.entry(name.to_owned()).or_default().push(index);
    }
    fn into_python(self, py: Python<'_>) -> PyResult<Bound<'_, PyDict>> {
        let result = PyDict::new(py);
        for name in self.order {
            result.set_item(&name, &self.indices[&name])?;
        }
        Ok(result)
    }
}

struct Prepared {
    controls: Vec<[f64; 6]>,
    smoothing: Vec<(u64, u64)>,
    control_sources: SourceMap,
    control_scopes: Vec<String>,
    source_controls: Vec<String>,
    lfos: Vec<[f64; 4]>,
    lfo_documents: Vec<String>,
    lfo_sources: SourceMap,
    voice_lfo_sources: SourceMap,
    envelopes: Vec<String>,
    envelope_parameters: Vec<(usize, usize, bool)>,
    named_sources: SourceMap,
    scopes: BTreeSet<String>,
    stages: Vec<(String, Option<usize>, usize)>,
    stage_sources: SourceMap,
    stage_bindings: BTreeMap<Identifier, usize>,
    connections: Vec<StageConnection>,
    maps: RouteMaps,
    event_bindings: BTreeMap<usize, String>,
    records: usize,
    characters: usize,
    limits: PreparationLimits,
}
impl Prepared {
    fn new(records: usize, limits: PreparationLimits) -> Self {
        Self {
            controls: Vec::new(),
            smoothing: Vec::new(),
            control_sources: SourceMap::default(),
            control_scopes: Vec::new(),
            source_controls: Vec::new(),
            lfos: Vec::new(),
            lfo_documents: Vec::new(),
            lfo_sources: SourceMap::default(),
            voice_lfo_sources: SourceMap::default(),
            envelopes: Vec::new(),
            envelope_parameters: Vec::new(),
            named_sources: SourceMap::default(),
            scopes: BTreeSet::new(),
            stages: Vec::new(),
            stage_sources: SourceMap::default(),
            stage_bindings: BTreeMap::new(),
            connections: Vec::new(),
            maps: vec![Vec::new(); 4],
            event_bindings: BTreeMap::new(),
            records,
            characters: 0,
            limits,
        }
    }

    fn generator(
        &mut self,
        context: &Context,
        documents: &BTreeMap<&str, &str>,
        binding: &GeneratorBinding,
        parameter: Option<usize>,
        operation: usize,
        routed: bool,
    ) -> PyResult<Option<usize>> {
        let generator = &context.generators[binding.reference()];
        if matches!(generator.body(), Body::Patch(_)) {
            return Ok(None);
        }
        let document = documents[binding.reference().as_str()];
        self.records = self
            .records
            .checked_add(
                1 + envelfo::preparation_records(
                    &serde_json::to_value(generator).map_err(invalid)?,
                    self.limits,
                )
                .map_err(invalid)?,
            )
            .ok_or_else(|| invalid("Preparation size overflow"))?;
        self.limits.check_records(self.records).map_err(invalid)?;
        self.characters = self
            .characters
            .checked_add(document.chars().count())
            .ok_or_else(|| invalid("Preparation text overflow"))?;
        if self.characters > codec::MAX_SCORE_CHARACTERS {
            return Err(invalid(
                "Prepared generators exceed document character limit",
            ));
        }
        let index = parameter.unwrap_or(0);
        let scope = generator.settings().scope();
        match generator.body() {
            Body::Lfo(body) => {
                if !routed {
                    self.scopes.insert(generator_scope(scope).to_owned());
                }
                if scope == Scope::Voice {
                    self.voice_lfo_sources
                        .insert(binding.reference(), self.lfos.len());
                } else if scope == Scope::Instrument || !routed {
                    self.lfo_sources
                        .insert(binding.reference(), self.lfos.len());
                }
                self.lfos.push([
                    index as f64,
                    operation as f64,
                    if routed { body.center() } else { 1.0 },
                    if routed { body.depth() } else { 0.0 },
                ]);
                self.lfo_documents.push(document.to_owned());
                Ok(Some(1))
            }
            Body::Envelope(body) => {
                if scope != Scope::Voice || !matches!(body.initial(), Initial::Number(_)) {
                    return Err(invalid(
                        "Persistent named envelopes require numeric initial and voice scope",
                    ));
                }
                self.named_sources
                    .insert(binding.reference(), self.envelopes.len());
                self.envelopes.push(document.to_owned());
                self.envelope_parameters.push((
                    index,
                    operation,
                    binding.release_timing() == ReleaseTiming::Voice,
                ));
                Ok(Some(2))
            }
            Body::SampleHold(_) => {
                self.named_sources
                    .insert(binding.reference(), self.envelopes.len());
                self.envelopes.push(document.to_owned());
                self.envelope_parameters.push((index, operation, false));
                Ok(Some(2))
            }
            Body::Stages(_) => {
                if !routed {
                    self.scopes.insert(generator_scope(scope).to_owned());
                }
                self.stage_bindings
                    .entry(binding.name().clone())
                    .or_insert(self.stages.len());
                self.stage_sources
                    .insert(binding.reference(), self.stages.len());
                self.stages.push((
                    document.to_owned(),
                    parameter,
                    if routed { operation } else { 0 },
                ));
                Ok(Some(3))
            }
            Body::Patch(_) | Body::Reference(_) => Err(invalid("Unresolved generator body")),
        }
    }

    fn into_python(
        self,
        py: Python<'_>,
        values: Vec<f64>,
        filters: Array2<f64>,
        context: String,
        generators: Vec<(String, String)>,
        targets: Vec<(String, String, usize)>,
    ) -> PyResult<Py<PyTuple>> {
        let mut fields: Vec<Py<PyAny>> = Vec::with_capacity(23);
        macro_rules! field {
            ($value:expr) => {
                fields.push($value.into_pyobject(py)?.into_any().unbind());
            };
        }
        field!(
            Array2::from_shape_vec(
                (self.controls.len(), 6),
                self.controls.into_iter().flatten().collect()
            )
            .map_err(invalid)?
            .into_pyarray(py)
        );
        field!(self.smoothing);
        field!(values);
        field!(self.control_sources.into_python(py)?);
        field!(self.control_scopes);
        field!(self.source_controls);
        field!(
            Array2::from_shape_vec(
                (self.lfos.len(), 4),
                self.lfos.into_iter().flatten().collect()
            )
            .map_err(invalid)?
            .into_pyarray(py)
        );
        field!(self.lfo_documents);
        field!(self.lfo_sources.into_python(py)?);
        field!(self.voice_lfo_sources.into_python(py)?);
        field!(self.envelopes);
        field!(self.envelope_parameters);
        field!(self.named_sources.into_python(py)?);
        field!(PySet::new(py, self.scopes)?);
        field!(filters.into_pyarray(py));
        field!(self.stages);
        field!(self.stage_sources.into_python(py)?);
        field!(self.connections);
        field!(self.maps);
        field!(self.event_bindings);
        field!(context);
        field!(generators);
        field!(targets);
        Ok(PyTuple::new(py, fields)?.unbind())
    }
}

fn scope_name(scope: SourceScope) -> &'static str {
    match scope {
        SourceScope::Voice => "voice",
        SourceScope::Instrument => "instrument",
        SourceScope::Part => "part",
        SourceScope::Trigger => "trigger",
    }
}
fn generator_scope(scope: Scope) -> &'static str {
    match scope {
        Scope::Voice => "voice",
        Scope::Instrument => "instrument",
        Scope::Part => "part",
    }
}
