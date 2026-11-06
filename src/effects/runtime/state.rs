//! Canonical DSP state at the portable offline snapshot boundary.

use super::{DelayHead, EffectRuntime};
use numpy::ndarray::{ArrayView2, ArrayViewMut2, Axis, Slice};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::PyDict;

impl EffectRuntime {
    pub(super) fn export_processors<'py>(
        &self,
        py: Python<'py>,
    ) -> PyResult<Vec<Bound<'py, PyDict>>> {
        let mut result = Vec::with_capacity(self.kinds.len());
        let inputs = self.input_routes.len();
        for (node, kind) in self.kinds.iter().enumerate() {
            let value = PyDict::new(py);
            let channels = self.widths[node + inputs];
            match kind {
                2 => {
                    let mut filters = Vec::new();
                    for (stage, filter) in self.filter_rows.iter().enumerate() {
                        if filter.node == node {
                            let state = PyDict::new(py);
                            state.set_item(
                                "integrators",
                                rows(
                                    self.filter_states
                                        .index_axis(Axis(0), stage)
                                        .slice_axis(Axis(0), Slice::from(..channels)),
                                ),
                            )?;
                            filters.push(state);
                        }
                    }
                    value.set_item("filter_states", filters)?;
                }
                3 => {
                    let granular = self.granulators[node]
                        .as_ref()
                        .expect("prepared granulator");
                    let history = (0..granular.history_len)
                        .map(|index| {
                            let row = (granular.history_write + granular.history.nrows()
                                - granular.history_len
                                + index)
                                % granular.history.nrows();
                            granular.history.row(row).to_vec()
                        })
                        .collect::<Vec<_>>();
                    let state = PyDict::new(py);
                    state.set_item("history", &history)?;
                    state.set_item("history_start", granular.history_start)?;
                    state.set_item("phase", granular.phase)?;
                    state.set_item("launch_counter", granular.counter)?;
                    state.set_item("frozen", granular.frozen)?;
                    state.set_item("frozen_start", granular.frozen_start)?;
                    state.set_item(
                        "frozen_history",
                        if granular.frozen { history } else { Vec::new() },
                    )?;
                    let mut grains = Vec::with_capacity(granular.active_count);
                    for index in &granular.order[..granular.active_count] {
                        let grain = &granular.grains[*index];
                        let value = PyDict::new(py);
                        value.set_item(
                            "samples",
                            rows(grain.samples.slice_axis(Axis(0), Slice::from(..grain.len))),
                        )?;
                        value.set_item("index", grain.index)?;
                        value.set_item("gain", grain.gain)?;
                        grains.push(value);
                    }
                    state.set_item("grains", grains)?;
                    value.set_item("granulator", state)?;
                }
                5 | 8 => {
                    let delay = self.tap_delays[node].as_ref().expect("prepared tap delay");
                    let state = PyDict::new(py);
                    state.set_item("history", rows(delay.history.view()))?;
                    state.set_item("write", delay.write)?;
                    state.set_item("current", delay.current)?;
                    let mut heads = Vec::with_capacity(delay.heads.len());
                    for head in &delay.heads {
                        let value = PyDict::new(py);
                        value.set_item("delay_frames", head.delay)?;
                        value.set_item("gain", head.gain)?;
                        value.set_item("step", head.step)?;
                        value.set_item("remaining", head.remaining)?;
                        value.set_item("active", head.active)?;
                        heads.push(value);
                    }
                    state.set_item("heads", heads)?;
                    if *kind == 8 {
                        let reverb = self.reverbs[node].as_ref().expect("prepared reverb");
                        let network = PyDict::new(py);
                        network.set_item("predelay", state)?;
                        network.set_item("lines", &reverb.lines)?;
                        network.set_item("writes", reverb.writes)?;
                        network.set_item("damping", reverb.damping)?;
                        network.set_item("non_silent", reverb.non_silent + delay.non_silent)?;
                        value.set_item("reverb", network)?;
                    } else {
                        value.set_item("tap_delay", state)?;
                    }
                }
                6 => {
                    let delay = self.modulated_delays[node]
                        .as_ref()
                        .expect("prepared modulated delay");
                    let state = PyDict::new(py);
                    state.set_item("history", rows(delay.history.view()))?;
                    state.set_item("write", delay.write)?;
                    state.set_item("phase", delay.phase)?;
                    value.set_item("modulated_delay", state)?;
                }
                7 => value.set_item("reduction_db", self.reductions[node])?,
                _ => {}
            }
            result.push(value);
        }
        Ok(result)
    }

    pub(super) fn import_processors(
        &mut self,
        values: Vec<Bound<'_, PyDict>>,
        frame: usize,
    ) -> PyResult<()> {
        if values.len() != self.kinds.len() {
            return Err(PyValueError::new_err("Invalid native effect snapshot"));
        }
        let mut snapshot = self.snapshot();
        let inputs = self.input_routes.len();
        for (node, (kind, value)) in self.kinds.iter().zip(&values).enumerate() {
            let channels = self.widths[node + inputs];
            match kind {
                2 => {
                    let values: Vec<Bound<'_, PyDict>> =
                        field(value, "filter_states")?.extract()?;
                    let stages = self
                        .filter_rows
                        .iter()
                        .enumerate()
                        .filter(|(_, filter)| filter.node == node)
                        .map(|(index, _)| index)
                        .collect::<Vec<_>>();
                    if values.len() != stages.len() {
                        return Err(PyValueError::new_err("Invalid native filter snapshot"));
                    }
                    for (value, stage) in values.iter().zip(stages) {
                        let mut integrators = snapshot.filter_states.index_axis_mut(Axis(0), stage);
                        restore_rows(
                            integrators.slice_axis_mut(Axis(0), Slice::from(..channels)),
                            field(value, "integrators")?.extract()?,
                        )?;
                    }
                }
                3 => {
                    let value = field(value, "granulator")?;
                    let value = value.cast::<PyDict>()?;
                    let granular = snapshot.granulators[node]
                        .as_mut()
                        .expect("prepared granulator");
                    let history: Vec<Vec<f64>> = field(value, "history")?.extract()?;
                    let frozen_history: Vec<Vec<f64>> =
                        field(value, "frozen_history")?.extract()?;
                    granular.history_len = history.len();
                    granular.history_start = field(value, "history_start")?.extract()?;
                    granular.frozen_start = field(value, "frozen_start")?.extract()?;
                    granular.frozen = field(value, "frozen")?.extract()?;
                    granular.phase = field(value, "phase")?.extract()?;
                    granular.counter = field(value, "launch_counter")?.extract()?;
                    if granular.history_len > granular.history.nrows()
                        || !granular.phase.is_finite()
                        || !(0.0..1.0).contains(&granular.phase)
                        || (granular.frozen
                            && (frozen_history != history
                                || granular.frozen_start != granular.history_start))
                    {
                        return Err(PyValueError::new_err("Invalid native granular history"));
                    }
                    restore_rows(
                        granular
                            .history
                            .slice_axis_mut(Axis(0), Slice::from(..granular.history_len)),
                        history,
                    )?;
                    granular.history_write = granular.history_len % granular.history.nrows();
                    granular.frame = frame;
                    let grains: Vec<Bound<'_, PyDict>> = field(value, "grains")?.extract()?;
                    if grains.len() > granular.grains.len() {
                        return Err(PyValueError::new_err(
                            "Native granular snapshot exceeds its grain capacity",
                        ));
                    }
                    for grain in &mut granular.grains {
                        grain.active = false;
                    }
                    granular.active_count = grains.len();
                    for (index, value) in grains.iter().enumerate() {
                        let grain = &mut granular.grains[index];
                        let samples: Vec<Vec<f64>> = field(value, "samples")?.extract()?;
                        grain.len = samples.len();
                        grain.index = field(value, "index")?.extract()?;
                        grain.gain = field(value, "gain")?.extract()?;
                        if grain.len < 2
                            || grain.len > grain.samples.nrows()
                            || grain.index >= grain.len
                            || !grain.gain.is_finite()
                            || grain.gain < 0.0
                        {
                            return Err(PyValueError::new_err("Invalid native granular grain"));
                        }
                        restore_rows(
                            grain
                                .samples
                                .slice_axis_mut(Axis(0), Slice::from(..grain.len)),
                            samples,
                        )?;
                        grain.active = true;
                        granular.order[index] = index;
                    }
                }
                5 | 8 => {
                    let network = if *kind == 8 {
                        Some(field(value, "reverb")?)
                    } else {
                        None
                    };
                    let value = if let Some(network) = &network {
                        field(network.cast::<PyDict>()?, "predelay")?
                    } else {
                        field(value, "tap_delay")?
                    };
                    let value = value.cast::<PyDict>()?;
                    let delay = snapshot.tap_delays[node]
                        .as_mut()
                        .expect("prepared tap delay");
                    restore_rows(
                        delay.history.view_mut(),
                        field(value, "history")?.extract()?,
                    )?;
                    delay.write = field(value, "write")?.extract()?;
                    delay.current = field(value, "current")?.extract()?;
                    let heads: Vec<Bound<'_, PyDict>> = field(value, "heads")?.extract()?;
                    if delay.write >= delay.history.nrows()
                        || delay.current >= delay.heads.len()
                        || heads.len() != delay.heads.len()
                    {
                        return Err(PyValueError::new_err("Invalid native tap snapshot"));
                    }
                    for (head, value) in delay.heads.iter_mut().zip(&heads) {
                        *head = DelayHead {
                            delay: field(value, "delay_frames")?.extract()?,
                            gain: field(value, "gain")?.extract()?,
                            step: field(value, "step")?.extract()?,
                            remaining: field(value, "remaining")?.extract()?,
                            active: field(value, "active")?.extract()?,
                        };
                        if !head.delay.is_finite()
                            || head.delay < if *kind == 8 { 0.0 } else { 1.0 }
                            || head.delay
                                > if *kind == 8 {
                                    self.parameters[[node, 6]] * self.rate
                                } else {
                                    delay.history.nrows() as f64
                                }
                            || !head.gain.is_finite()
                            || !(0.0..=1.0).contains(&head.gain)
                            || !head.step.is_finite()
                            || head.step < 0.0
                            || head.remaining > delay.fade_frames
                        {
                            return Err(PyValueError::new_err("Invalid native tap read head"));
                        }
                    }
                    if !delay.heads[delay.current].active {
                        return Err(PyValueError::new_err("Invalid native tap destination"));
                    }
                    delay.non_silent = delay
                        .history
                        .iter()
                        .filter(|value| value.abs() >= self.tail_thresholds[node])
                        .count();
                    if let Some(network) = network {
                        let network = network.cast::<PyDict>()?;
                        let reverb = snapshot.reverbs[node].as_mut().expect("prepared reverb");
                        let lines: Vec<Vec<f64>> = field(network, "lines")?.extract()?;
                        if lines.len() != 8
                            || lines.iter().zip(&reverb.lines).any(|(a, b)| {
                                a.len() != b.len() || a.iter().any(|value| !value.is_finite())
                            })
                        {
                            return Err(PyValueError::new_err(
                                "Invalid reverb delay-line snapshot",
                            ));
                        }
                        reverb.writes = field(network, "writes")?.extract()?;
                        reverb.damping = field(network, "damping")?.extract()?;
                        if reverb
                            .writes
                            .iter()
                            .zip(&lines)
                            .any(|(write, line)| *write >= line.len())
                            || reverb.damping.iter().any(|value| !value.is_finite())
                        {
                            return Err(PyValueError::new_err("Invalid reverb network snapshot"));
                        }
                        reverb.non_silent = lines
                            .iter()
                            .flatten()
                            .filter(|value| value.abs() >= self.tail_thresholds[node])
                            .count();
                        let count: usize = field(network, "non_silent")?.extract()?;
                        if count != reverb.non_silent + delay.non_silent {
                            return Err(PyValueError::new_err("Invalid reverb tail count"));
                        }
                        reverb.lines = lines;
                    }
                }
                6 => {
                    let value = field(value, "modulated_delay")?;
                    let value = value.cast::<PyDict>()?;
                    let delay = snapshot.modulated_delays[node]
                        .as_mut()
                        .expect("prepared modulated delay");
                    restore_rows(
                        delay.history.view_mut(),
                        field(value, "history")?.extract()?,
                    )?;
                    delay.write = field(value, "write")?.extract()?;
                    delay.phase = field(value, "phase")?.extract()?;
                    if delay.write >= delay.history.nrows()
                        || !delay.phase.is_finite()
                        || !(0.0..1.0).contains(&delay.phase)
                    {
                        return Err(PyValueError::new_err(
                            "Invalid native modulated delay snapshot",
                        ));
                    }
                    delay.non_silent = delay
                        .history
                        .iter()
                        .filter(|value| value.abs() >= self.tail_thresholds[node])
                        .count();
                }
                7 => {
                    let reduction: f64 = field(value, "reduction_db")?.extract()?;
                    if !reduction.is_finite() || reduction > 0.0 {
                        return Err(PyValueError::new_err("Invalid native compressor snapshot"));
                    }
                    snapshot.reductions[node] = reduction;
                }
                _ => {}
            }
        }
        snapshot.ended.fill(false);
        snapshot.stopped = false;
        self.restore(&snapshot)
    }
}

fn rows(values: ArrayView2<'_, f64>) -> Vec<Vec<f64>> {
    values.rows().into_iter().map(|row| row.to_vec()).collect()
}

fn restore_rows(mut target: ArrayViewMut2<'_, f64>, values: Vec<Vec<f64>>) -> PyResult<()> {
    if values.len() != target.nrows()
        || values
            .iter()
            .any(|row| row.len() != target.ncols() || row.iter().any(|value| !value.is_finite()))
    {
        return Err(PyValueError::new_err(
            "Invalid native effect snapshot matrix",
        ));
    }
    for ((row, column), value) in target.indexed_iter_mut() {
        *value = values[row][column];
    }
    Ok(())
}

fn field<'py>(value: &Bound<'py, PyDict>, name: &str) -> PyResult<Bound<'py, PyAny>> {
    value
        .get_item(name)?
        .ok_or_else(|| PyValueError::new_err("Missing native effect snapshot field"))
}
