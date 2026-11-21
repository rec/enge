//! Compile authored FM graphs for the persistent renderer.
use super::{amplitude_definition, invalid};
use crate::amplitude::{Amplitude, frame_time};
use crate::runtime::preparation::modulation::TargetInput;
use crate::runtime::{GraphEdge, GraphOperator};
use pyo3::prelude::*;
use std::collections::BTreeMap;
use ufor::codec::{self, Format};
use ufor::envelfo::{self, PreparationLimits, Waveform};
use ufor::fm::Fm;

#[pyclass(frozen, skip_from_py_object)]
pub(crate) struct PreparedFm {
    pub(in crate::runtime) graph: Fm,
    pub(in crate::runtime) rate: f64,
    pub(in crate::runtime) targets: Vec<TargetInput>,
    pub(in crate::runtime) records: usize,
    pub(in crate::runtime) characters: usize,
    pub(in crate::runtime) operators: Vec<GraphOperator>,
    pub(in crate::runtime) edges: Vec<GraphEdge>,
    pub(in crate::runtime) amplitudes: Vec<Amplitude>,
    pub(in crate::runtime) carrier: usize,
    pub(in crate::runtime) carrier_parameter: usize,
}

#[pymethods]
impl PreparedFm {
    #[new]
    #[pyo3(signature = (document, rate, *, max_records=10000))]
    fn new(document: &str, rate: f64, max_records: usize) -> PyResult<Self> {
        frame_time(rate)?;
        if rate <= 0.0 {
            return Err(invalid("FM requires a positive sample rate"));
        }
        let limits = PreparationLimits::new(max_records).map_err(invalid)?;
        let graph = codec::parse_fm(document, Format::Json, limits).map_err(invalid)?;
        let mut targets = Vec::new();
        let mut slot = |name, parameter: &str, minimum, maximum| {
            // Amplitude and tuning occupy the first two slots.
            let index = 2 + targets.len();
            targets.push((name, parameter.into(), index, None, minimum, maximum));
            index
        };
        let names: BTreeMap<_, _> = graph
            .operators()
            .iter()
            .enumerate()
            .map(|(index, operator)| (operator.name(), index))
            .collect();
        let carrier = names[graph.carrier()];
        let operators = graph
            .operators()
            .iter()
            .map(|operator| {
                let name = format!("operator-{}", operator.name().as_str());
                GraphOperator {
                    waveform: match operator.waveform() {
                        Waveform::Sine => 0,
                        Waveform::Square => 1,
                        Waveform::Triangle => 2,
                    },
                    phase_offset: operator.phase_cycles(),
                    ratio_parameter: slot(name.clone(), "ratio", f64::MIN_POSITIVE, 1e300),
                    tuning_parameter: slot(name, "tuning_cents", -120000.0, 120000.0),
                }
            })
            .collect();
        let edges = graph
            .edges()
            .iter()
            .map(|edge| GraphEdge {
                source: names[edge.source()],
                destination: names[edge.destination()],
                delayed: edge.delayed(),
                parameter: slot(format!("edge-{}", edge.target_name()), "index", 0.0, 1e300),
            })
            .collect();
        let carrier_parameter = slot("fm".into(), "carrier_level", 0.0, 1e300);
        // Include materialized default Envelopes in the preparation budget.
        let records =
            envelfo::preparation_records(&serde_json::to_value(&graph).map_err(invalid)?, limits)
                .map_err(invalid)?;
        let amplitudes = graph
            .operators()
            .iter()
            .map(|operator| amplitude_definition(operator.envelope(), rate))
            .collect::<PyResult<_>>()?;
        Ok(Self {
            graph,
            rate,
            targets,
            records,
            characters: document.chars().count(),
            operators,
            edges,
            amplitudes,
            carrier,
            carrier_parameter,
        })
    }
}
