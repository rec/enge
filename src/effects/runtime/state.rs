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
            let alignment = PyDict::new(py);
            for (port, delay) in self.alignment[node].iter().enumerate() {
                if let Some(delay) = delay {
                    let state = PyDict::new(py);
                    state.set_item("history", rows(delay.history.view()))?;
                    state.set_item("write", delay.write)?;
                    state.set_item("remaining", delay.remaining)?;
                    alignment.set_item(&self.port_names[node][port], state)?;
                }
            }
            value.set_item("alignment", alignment)?;
            let controls = PyDict::new(py);
            for (parameter, delay) in self.control_delays[node].iter().enumerate() {
                if let Some(delay) = delay {
                    let state = PyDict::new(py);
                    state.set_item("history", &delay.history)?;
                    state.set_item("write", delay.write)?;
                    controls.set_item(parameter.to_string(), state)?;
                }
            }
            value.set_item("control_delays", controls)?;
            let bases = PyDict::new(py);
            for target in &self.controls.targets {
                if target.node != node {
                    continue;
                }
                if let Some(delay) = &target.base {
                    bases.set_item(target.parameter.to_string(), control_delay(py, delay)?)?;
                }
            }
            value.set_item("base_delays", bases)?;
            let routes = PyDict::new(py);
            for route in &self.controls.routes {
                if self.controls.targets[route.target].node != node {
                    continue;
                }
                if let Some(delay) = &route.delay {
                    routes.set_item(&route.name, control_delay(py, delay)?)?;
                }
            }
            value.set_item("route_delays", routes)?;
            value.set_item("follower_level", self.follower_levels[node])?;
            match kind {
                18 => {
                    let resonant = self.resonants[node]
                        .as_ref()
                        .expect("prepared resonant delay");
                    let state = PyDict::new(py);
                    state.set_item("history", rows(resonant.history.view()))?;
                    state.set_item("write", resonant.write)?;
                    let tuning = PyDict::new(py);
                    tuning.set_item("memory", &resonant.tuning.memory)?;
                    state.set_item("tuning", tuning)?;
                    let damping = PyDict::new(py);
                    damping.set_item("memory", &resonant.damping.memory)?;
                    state.set_item("damping", damping)?;
                    value.set_item("resonant_delay", state)?;
                }
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
                4 => {
                    let shaper = self.waveshapers[node]
                        .as_ref()
                        .expect("prepared waveshaper");
                    let state = PyDict::new(py);
                    state.set_item(
                        "histories",
                        shaper
                            .histories
                            .iter()
                            .map(|v| rows(v.view()))
                            .collect::<Vec<_>>(),
                    )?;
                    state.set_item("writes", shaper.writes)?;
                    state.set_item("dry_history", rows(shaper.dry_history.view()))?;
                    state.set_item("dry_write", shaper.dry_write)?;
                    state.set_item("remaining", shaper.remaining)?;
                    value.set_item("waveshaper", state)?;
                }
                5 | 8 | 17 => {
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
                    if matches!(*kind, 5 | 17) {
                        if let Some(damping) = &delay.damping {
                            let value = PyDict::new(py);
                            value.set_item("memory", &damping.memory)?;
                            state.set_item("damping", value)?;
                        } else {
                            state.set_item("damping", py.None())?;
                        }
                    }
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
                9 => {
                    let phaser = self.phasers[node].as_ref().expect("prepared phaser");
                    let state = PyDict::new(py);
                    state.set_item("stages", rows(phaser.stages.view()))?;
                    state.set_item("feedback", &phaser.feedback)?;
                    value.set_item("phaser", state)?;
                }
                10 => {
                    let state = PyDict::new(py);
                    state.set_item(
                        "memory",
                        &self.highpasses[node]
                            .as_ref()
                            .expect("prepared highpass")
                            .memory,
                    )?;
                    value.set_item("highpass", state)?;
                }
                15 => {
                    let gate = self.gates[node].as_ref().expect("prepared gate");
                    let state = PyDict::new(py);
                    state.set_item("gain", gate.gain)?;
                    state.set_item("opened", gate.opened)?;
                    state.set_item("elapsed", gate.elapsed)?;
                    value.set_item("gate", state)?;
                }
                16 => value.set_item("expansion_gain", self.expansion_gains[node])?,
                12 => {
                    let mixer = self.mixers[node].as_ref().expect("prepared mixer");
                    let state = PyDict::new(py);
                    let fades = mixer
                        .fades
                        .iter()
                        .map(|fade| {
                            let value = PyDict::new(py);
                            value.set_item("value", fade.value)?;
                            value.set_item("target", fade.target)?;
                            value.set_item("step", fade.step)?;
                            value.set_item("remaining", fade.remaining)?;
                            Ok(value)
                        })
                        .collect::<PyResult<Vec<_>>>()?;
                    state.set_item("fades", fades)?;
                    value.set_item("mixer", state)?;
                }
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
            let controls = field(value, "control_delays")?;
            let controls = controls.cast::<PyDict>()?;
            if controls.len()
                != snapshot.control_delays[node]
                    .iter()
                    .filter(|v| v.is_some())
                    .count()
            {
                return Err(PyValueError::new_err(
                    "Invalid control compensation snapshot",
                ));
            }
            for (parameter, delay) in snapshot.control_delays[node].iter_mut().enumerate() {
                if let Some(delay) = delay {
                    let state = field(controls, &parameter.to_string())?;
                    let state = state.cast::<PyDict>()?;
                    let history: Vec<f64> = field(state, "history")?.extract()?;
                    let write: usize = field(state, "write")?.extract()?;
                    if history.len() != delay.history.len()
                        || history.iter().any(|value| !value.is_finite())
                        || write >= history.len()
                    {
                        return Err(PyValueError::new_err("Invalid control compensation clock"));
                    }
                    delay.history = history;
                    delay.write = write;
                }
            }
            let alignment = field(value, "alignment")?;
            let alignment = alignment.cast::<PyDict>()?;
            if alignment.len()
                != snapshot.alignment[node]
                    .iter()
                    .filter(|v| v.is_some())
                    .count()
            {
                return Err(PyValueError::new_err("Invalid effect alignment snapshot"));
            }
            for (port, delay) in snapshot.alignment[node].iter_mut().enumerate() {
                if let Some(delay) = delay {
                    let state = field(alignment, &self.port_names[node][port])?;
                    let state = state.cast::<PyDict>()?;
                    restore_rows(
                        delay.history.view_mut(),
                        field(state, "history")?.extract()?,
                    )?;
                    delay.write = field(state, "write")?.extract()?;
                    delay.remaining = field(state, "remaining")?.extract()?;
                    if delay.write >= delay.history.nrows()
                        || delay.remaining > delay.history.nrows()
                    {
                        return Err(PyValueError::new_err("Invalid effect alignment clock"));
                    }
                }
            }
            match kind {
                18 => {
                    let state = field(value, "resonant_delay")?;
                    let state = state.cast::<PyDict>()?;
                    let resonant = snapshot.resonants[node]
                        .as_mut()
                        .expect("prepared resonant delay");
                    restore_rows(
                        resonant.history.view_mut(),
                        field(state, "history")?.extract()?,
                    )?;
                    resonant.write = field(state, "write")?.extract()?;
                    if resonant.write >= resonant.history.nrows() {
                        return Err(PyValueError::new_err("Invalid resonant history clock"));
                    }
                    for (name, filter) in [
                        ("tuning", &mut resonant.tuning),
                        ("damping", &mut resonant.damping),
                    ] {
                        let values = field(state, name)?;
                        let memory: Vec<f64> =
                            field(values.cast::<PyDict>()?, "memory")?.extract()?;
                        if memory.len() != filter.memory.len()
                            || memory.iter().any(|v| !v.is_finite())
                        {
                            return Err(PyValueError::new_err("Invalid resonant filter memory"));
                        }
                        filter.memory = memory;
                    }
                    resonant.non_silent = resonant
                        .history
                        .iter()
                        .filter(|v| v.abs() >= self.tail_thresholds[node])
                        .count();
                }
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
                4 => {
                    let state = field(value, "waveshaper")?;
                    let state = state.cast::<PyDict>()?;
                    let shaper = snapshot.waveshapers[node]
                        .as_mut()
                        .expect("prepared waveshaper");
                    let histories: Vec<Vec<Vec<f64>>> = field(state, "histories")?.extract()?;
                    if histories.len() != 4 {
                        return Err(PyValueError::new_err("Invalid waveshaper histories"));
                    }
                    for (target, history) in shaper.histories.iter_mut().zip(histories) {
                        restore_rows(target.view_mut(), history)?;
                    }
                    restore_rows(
                        shaper.dry_history.view_mut(),
                        field(state, "dry_history")?.extract()?,
                    )?;
                    shaper.writes = field(state, "writes")?.extract()?;
                    shaper.dry_write = field(state, "dry_write")?.extract()?;
                    shaper.remaining = field(state, "remaining")?.extract()?;
                    if shaper.writes.iter().any(|write| *write >= 65)
                        || shaper.dry_write >= 48
                        || shaper.remaining > 96
                    {
                        return Err(PyValueError::new_err("Invalid waveshaper clock"));
                    }
                }
                5 | 8 | 17 => {
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
                    if matches!(*kind, 5 | 17) {
                        let state = field(value, "damping")?;
                        match (&mut delay.damping, state.is_none()) {
                            (Some(damping), false) => {
                                let memory: Vec<f64> =
                                    field(state.cast::<PyDict>()?, "memory")?.extract()?;
                                if memory.len() != damping.memory.len()
                                    || memory.iter().any(|v| !v.is_finite())
                                {
                                    return Err(PyValueError::new_err(
                                        "Invalid tap damping memory",
                                    ));
                                }
                                damping.memory = memory;
                            }
                            (None, true) => {}
                            _ => {
                                return Err(PyValueError::new_err(
                                    "Tap damping preparation differs from snapshot",
                                ));
                            }
                        }
                    }
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
                9 => {
                    let state = field(value, "phaser")?;
                    let state = state.cast::<PyDict>()?;
                    let phaser = snapshot.phasers[node].as_mut().expect("prepared phaser");
                    restore_rows(phaser.stages.view_mut(), field(state, "stages")?.extract()?)?;
                    let feedback: Vec<f64> = field(state, "feedback")?.extract()?;
                    if feedback.len() != channels || feedback.iter().any(|value| !value.is_finite())
                    {
                        return Err(PyValueError::new_err("Invalid phaser feedback snapshot"));
                    }
                    phaser.feedback = feedback;
                }
                10 => {
                    let state = field(value, "highpass")?;
                    let state = state.cast::<PyDict>()?;
                    let memory: Vec<f64> = field(state, "memory")?.extract()?;
                    if memory.len() != channels || memory.iter().any(|value| !value.is_finite()) {
                        return Err(PyValueError::new_err("Invalid highpass memory snapshot"));
                    }
                    snapshot.highpasses[node]
                        .as_mut()
                        .expect("prepared highpass")
                        .memory = memory;
                }
                15 => {
                    let value = field(value, "gate")?;
                    let value = value.cast::<PyDict>()?;
                    let gain: f64 = field(value, "gain")?.extract()?;
                    let opened: bool = field(value, "opened")?.extract()?;
                    let elapsed: usize = field(value, "elapsed")?.extract()?;
                    if !(0.0..=1.0).contains(&gain) || elapsed > isize::MAX as usize {
                        return Err(PyValueError::new_err("Invalid gate snapshot"));
                    }
                    snapshot.gates[node] = Some(super::dynamics::Gate {
                        gain,
                        opened,
                        elapsed,
                    });
                }
                16 => {
                    let gain: f64 = field(value, "expansion_gain")?.extract()?;
                    if !(0.0..=1.0).contains(&gain) {
                        return Err(PyValueError::new_err("Invalid expander snapshot"));
                    }
                    snapshot.expansion_gains[node] = gain;
                }
                12 => {
                    let state = field(value, "mixer")?;
                    let state = state.cast::<PyDict>()?;
                    let fades: Vec<Bound<'_, PyDict>> = field(state, "fades")?.extract()?;
                    let mixer = snapshot.mixers[node].as_mut().expect("prepared mixer");
                    if fades.len() != mixer.fades.len() {
                        return Err(PyValueError::new_err("Invalid mixer fade snapshot"));
                    }
                    for (fade, value) in mixer.fades.iter_mut().zip(&fades) {
                        let current: f64 = field(value, "value")?.extract()?;
                        let target: f64 = field(value, "target")?.extract()?;
                        let step: f64 = field(value, "step")?.extract()?;
                        let remaining: usize = field(value, "remaining")?.extract()?;
                        if !(0.0..=1.0).contains(&current)
                            || !matches!(target, 0.0 | 1.0)
                            || !step.is_finite()
                            || remaining > mixer.fade_frames
                        {
                            return Err(PyValueError::new_err("Invalid mixer fade state"));
                        }
                        *fade = super::mixer::InputFade {
                            value: current,
                            target,
                            step,
                            remaining,
                        };
                    }
                }
                _ => {}
            }
        }
        for (node, value) in values.iter().enumerate() {
            let bases = field(value, "base_delays")?;
            let bases = bases.cast::<PyDict>()?;
            let routes = field(value, "route_delays")?;
            let routes = routes.cast::<PyDict>()?;
            let mut base_count = 0;
            let mut route_count = 0;
            for target in &mut snapshot.controls.targets {
                if target.node != node {
                    continue;
                }
                if let Some(delay) = &mut target.base {
                    restore_control_delay(delay, &field(bases, &target.parameter.to_string())?)?;
                    base_count += 1;
                }
            }
            for route in &mut snapshot.controls.routes {
                if snapshot.controls.targets[route.target].node != node {
                    continue;
                }
                if let Some(delay) = &mut route.delay {
                    restore_control_delay(delay, &field(routes, &route.name)?)?;
                    route_count += 1;
                }
            }
            if bases.len() != base_count || routes.len() != route_count {
                return Err(PyValueError::new_err(
                    "Invalid effect modulation delay layout",
                ));
            }
            let level: f64 = field(value, "follower_level")?.extract()?;
            if !level.is_finite() || level < 0.0 {
                return Err(PyValueError::new_err("Invalid follower level"));
            }
            snapshot.follower_levels[node] = level;
        }
        snapshot.ended.fill(false);
        snapshot.stopped = false;
        self.restore(&snapshot)
    }
}

fn rows(values: ArrayView2<'_, f64>) -> Vec<Vec<f64>> {
    values.rows().into_iter().map(|row| row.to_vec()).collect()
}

fn control_delay<'py>(
    py: Python<'py>,
    delay: &super::controls::Delay,
) -> PyResult<Bound<'py, PyDict>> {
    let value = PyDict::new(py);
    value.set_item("history", &delay.history)?;
    value.set_item("write", delay.write)?;
    value.set_item("remaining", delay.remaining)?;
    Ok(value)
}

fn restore_control_delay(
    delay: &mut super::controls::Delay,
    value: &Bound<'_, PyAny>,
) -> PyResult<()> {
    let value = value.cast::<PyDict>()?;
    let history: Vec<f64> = field(value, "history")?.extract()?;
    let write: usize = field(value, "write")?.extract()?;
    let remaining: usize = field(value, "remaining")?.extract()?;
    if history.len() != delay.history.len()
        || history.iter().any(|v| !v.is_finite())
        || write >= history.len()
        || remaining > history.len()
    {
        return Err(PyValueError::new_err(
            "Invalid effect modulation delay state",
        ));
    }
    delay.history = history;
    delay.write = write;
    delay.remaining = remaining;
    Ok(())
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
