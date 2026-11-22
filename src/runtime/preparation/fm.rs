//! Compile authored FM graphs for the persistent renderer.
use super::{amplitude_definition, invalid};
use crate::amplitude::{Amplitude, frame_time};
use crate::runtime::GraphOperator;
use crate::runtime::preparation::modulation::TargetInput;
use pyo3::prelude::*;
use ufor::codec::{self, Format};
use ufor::envelfo::{self, PreparationLimits};
use ufor::fm::Fm;

#[pyclass(frozen, skip_from_py_object)]
pub(crate) struct PreparedFm {
    pub(crate) graph: Fm,
    pub(crate) rate: f64,
    pub(in crate::runtime) targets: Vec<TargetInput>,
    pub(in crate::runtime) records: usize,
    pub(in crate::runtime) characters: usize,
    pub(in crate::runtime) operators: Vec<GraphOperator>,
    pub(in crate::runtime) edge_parameters: Vec<usize>,
    pub(in crate::runtime) amplitudes: Vec<Amplitude>,
    pub(in crate::runtime) carrier_parameter: usize,
}

#[pymethods]
impl PreparedFm {
    #[getter]
    fn carrier_index(&self) -> usize {
        self.graph.carrier_index()
    }

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
        let operators = graph
            .operators()
            .iter()
            .map(|operator| {
                let name = format!("operator-{}", operator.name().as_str());
                GraphOperator {
                    waveform: operator.waveform(),
                    phase_offset: operator.phase_cycles(),
                    ratio_parameter: slot(name.clone(), "ratio", f64::MIN_POSITIVE, 1e300),
                    tuning_parameter: slot(name, "tuning_cents", -120000.0, 120000.0),
                }
            })
            .collect();
        let edge_parameters = graph
            .edges()
            .iter()
            .map(|edge| slot(format!("edge-{}", edge.target_name()), "index", 0.0, 1e300))
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
            edge_parameters,
            amplitudes,
            carrier_parameter,
        })
    }
}
