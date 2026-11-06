//! One native owner for persistent generated sources and callback scratch.

use crate::amplitude::{Amplitude, AmplitudeSegments, AmplitudeState, frame_time};
use crate::effects::runtime::{EffectGenerators, EffectRuntime, EffectSnapshot};
use crate::error::{NativeError, NativeResult};
use crate::filters::{FilterBank, Inputs};
use crate::queue::{ActionBatch, BATCH_ACTIONS};
use crate::runtime::controls::{ControlBank, control_definitions};
use crate::runtime::generators::GeneratorBank;
use crate::runtime::{RouteMap, SynthRuntime, SynthRuntimeSnapshot, sum_partials};
use crate::sampler::{Cursor, SampleBuffer, Selection, State};
use numpy::ndarray::{Array2, ArrayViewMut2};
use numpy::{PyReadonlyArray2, PyReadwriteArray2, PyUntypedArrayMethods};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use rtrb::{Consumer, Producer, RingBuffer};
use std::f64::consts::PI;

struct Source {
    runtime: SourceRuntime,
    producer: Producer<ActionBatch>,
    consumer: Consumer<ActionBatch>,
    action_scratch: Vec<f64>,
    action_count: usize,
}

#[derive(Clone)]
enum SourceRuntime {
    Generated(Box<SynthRuntime>),
    Sample(Box<SampleRuntime>),
}

#[derive(Clone)]
struct SampleRuntime {
    controls: ControlBank,
    route_maps: Vec<Vec<RouteMap>>,
    generators: Option<GeneratorBank>,
    generator_active: Vec<bool>,
    crossfades: Vec<(bool, bool)>,
    filter_routes: Vec<SampleFilter>,
    // Common and slot amplitude/tuning, filter cutoff/Q, then crossfade positions.
    parameters: Vec<f64>,
    parameter_partials: Vec<f64>,
    filters: Vec<FilterBank>,
    filter_after_amplitude: bool,
    buffer: SampleBuffer,
    initial_cursor: Cursor,
    routes: Array2<f64>,
    amplitude: Amplitude,
    amplitude_states: Vec<AmplitudeState>,
    minimum_hold_frames: f64,
    cursors: Vec<Cursor>,
    steps: Vec<f64>,
    gains: Vec<f64>,
    active: Vec<bool>,
    ages: Vec<usize>,
    delays: Vec<usize>,
    release_frames: Vec<Option<f64>>,
    fade_starts: Vec<Option<usize>>,
    fade_frames: Vec<f64>,
    fade_gains: Vec<f64>,
    source_values: Vec<f64>,
    frame: usize,
}

#[derive(Clone, PartialEq)]
struct SampleFilter {
    cutoff: usize,
    q: usize,
    stages: usize,
    clamp: bool,
    minimum: f64,
    maximum: f64,
}

#[derive(Clone)]
enum SourceSnapshot {
    Generated(Box<SynthRuntimeSnapshot>),
    Sample(Box<SampleRuntime>),
}

#[pyclass(frozen, skip_from_py_object)]
#[derive(Clone)]
pub struct LiveRuntimeSnapshot {
    sources: Vec<SourceSnapshot>,
    effects: Option<EffectSnapshot>,
    frame: usize,
    failed: bool,
}

#[pyclass(unsendable)]
pub struct LiveRuntime {
    sources: Vec<Source>,
    source_scratch: Array2<f64>,
    output_scratch: Array2<f64>,
    maximum_block_frames: usize,
    channels: usize,
    action_capacity: usize,
    rate: f64,
    effects: Option<EffectRuntime>,
    frame: usize,
    failed: bool,
}

#[pymethods]
impl LiveRuntime {
    #[new]
    fn new(
        py: Python<'_>,
        runtimes: Vec<Py<SynthRuntime>>,
        maximum_block_frames: usize,
        action_capacity: usize,
        batch_capacity: usize,
    ) -> PyResult<Self> {
        if runtimes.is_empty()
            || maximum_block_frames == 0
            || action_capacity == 0
            || batch_capacity == 0
            || action_capacity < BATCH_ACTIONS
        {
            return Err(PyValueError::new_err("Invalid live runtime capacity"));
        }
        let runtimes = runtimes
            .iter()
            .map(|v| {
                let mut runtime = v.borrow(py).clone();
                runtime.generators.prepare_patch(runtime.frame())?;
                Ok(runtime)
            })
            .collect::<PyResult<Vec<_>>>()?;
        let channels = runtimes[0].channels();
        let rate = runtimes[0].rate();
        let frame = runtimes[0].frame();
        if runtimes
            .iter()
            .any(|v| v.channels() != channels || v.rate() != rate || v.frame() != frame)
        {
            return Err(PyValueError::new_err(
                "Live runtime sources must share rate, channels, and frame",
            ));
        }
        let sources = runtimes
            .into_iter()
            .map(|runtime| {
                let (producer, consumer) = RingBuffer::new(batch_capacity);
                Source {
                    runtime: SourceRuntime::Generated(Box::new(runtime)),
                    producer,
                    consumer,
                    action_scratch: vec![0.0; action_capacity * 6],
                    action_count: 0,
                }
            })
            .collect();
        Ok(Self {
            sources,
            source_scratch: Array2::zeros((maximum_block_frames, channels)),
            output_scratch: Array2::zeros((maximum_block_frames, channels)),
            maximum_block_frames,
            channels,
            action_capacity,
            rate,
            effects: None,
            frame,
            failed: false,
        })
    }

    #[staticmethod]
    fn empty(
        rate: f64,
        channels: usize,
        maximum_block_frames: usize,
        action_capacity: usize,
    ) -> PyResult<Self> {
        if !rate.is_finite()
            || rate <= 0.0
            || channels == 0
            || maximum_block_frames == 0
            || action_capacity < BATCH_ACTIONS
        {
            return Err(PyValueError::new_err("Invalid empty live runtime"));
        }
        Ok(Self {
            sources: vec![],
            source_scratch: Array2::zeros((maximum_block_frames, channels)),
            output_scratch: Array2::zeros((maximum_block_frames, channels)),
            maximum_block_frames,
            channels,
            action_capacity,
            rate,
            effects: None,
            frame: 0,
            failed: false,
        })
    }

    #[allow(clippy::too_many_arguments)]
    #[pyo3(signature = (kinds, sources, parameters, output_source, filters, granulators, tap_delays, modulated_delays, widths, batch_capacity, input_routes=None))]
    fn set_effect_graph(
        &mut self,
        kinds: Vec<u8>,
        sources: PyReadonlyArray2<'_, i64>,
        parameters: PyReadonlyArray2<'_, f64>,
        output_source: usize,
        filters: PyReadonlyArray2<'_, f64>,
        granulators: PyReadonlyArray2<'_, i64>,
        tap_delays: PyReadonlyArray2<'_, i64>,
        modulated_delays: PyReadonlyArray2<'_, f64>,
        widths: Vec<usize>,
        batch_capacity: usize,
        input_routes: Option<Vec<Vec<i64>>>,
    ) -> PyResult<()> {
        if self.frame != 0 || self.effects.is_some() || batch_capacity == 0 {
            return Err(PyValueError::new_err("Invalid live effect graph setup"));
        }
        let inputs = widths
            .len()
            .checked_sub(kinds.len())
            .ok_or_else(|| PyValueError::new_err("Invalid live effect port widths"))?;
        let routes = input_routes.unwrap_or_else(|| vec![vec![-1]]);
        if routes.len() != inputs
            || routes.iter().enumerate().any(|(input, route)| {
                route.is_empty()
                    || route
                        .iter()
                        .any(|source| *source < -2 || *source >= self.sources.len() as i64)
                    || (route.iter().any(|source| *source < 0) && route.len() != 1)
                    || (route != &[-2] && widths[input] != self.channels)
            })
        {
            return Err(PyValueError::new_err("Invalid live effect input routing"));
        }
        let prepared = EffectRuntime::new(
            kinds,
            sources,
            parameters,
            output_source,
            filters,
            granulators,
            tap_delays,
            modulated_delays,
            self.rate,
            widths,
            routes,
            self.maximum_block_frames,
            self.action_capacity,
            batch_capacity,
        )?;
        self.output_scratch =
            Array2::zeros((self.maximum_block_frames, prepared.widths[output_source]));
        self.effects = Some(prepared);
        Ok(())
    }

    #[allow(clippy::type_complexity)]
    fn set_effect_generators(
        &mut self,
        generators: &GeneratorBank,
        maps: Vec<Vec<(usize, Vec<(f64, f64)>)>>,
        targets: Vec<(usize, usize, f64, f64)>,
    ) -> PyResult<()> {
        let effects = self
            .effects
            .as_mut()
            .ok_or_else(|| PyValueError::new_err("Effect graph is not prepared"))?;
        if self.frame != 0
            || effects.generators.is_some()
            || generators.rate != self.rate
            || generators.patch_states.len() != 1
            || generators.parameter_count != targets.len()
            || maps.len() != 4
            || targets.iter().any(|(node, parameter, minimum, maximum)| {
                *node >= effects.parameters.nrows()
                    || *parameter >= effects.parameters.ncols()
                    || !minimum.is_finite()
                    || !maximum.is_finite()
                    || minimum > maximum
            })
        {
            return Err(PyValueError::new_err("Invalid live effect generator setup"));
        }
        let mut maps = maps
            .into_iter()
            .enumerate()
            .map(|(kind, maps)| generators.prepare_route_maps(kind + 1, maps))
            .collect::<PyResult<Vec<_>>>()?;
        maps.insert(0, Vec::new());
        let mut bank = generators.clone();
        bank.prepare_patch(0)?;
        bank.start_patch(0, 0)?;
        effects.generators = Some(EffectGenerators {
            bank,
            controls: ControlBank::new(Vec::new(), Vec::new(), 1, 0),
            partials: vec![0.0; maps.iter().map(Vec::len).sum()],
            maps,
            targets,
        });
        Ok(())
    }

    fn submit_effects(&mut self, actions: PyReadonlyArray2<'_, f64>) -> PyResult<()> {
        let Some(effects) = &mut self.effects else {
            return Err(PyValueError::new_err("Live runtime has no effect graph"));
        };
        push_batch(&mut effects.producer, actions, "effect")
    }

    #[allow(clippy::too_many_arguments)]
    #[pyo3(signature = (buffer, selection, state, rate, routes, initial, attack, release, minimum_hold_frames, voices, batch_capacity, filters, filter_after_amplitude, end_fade_frames=0.0))]
    fn add_sample(
        &mut self,
        py: Python<'_>,
        buffer: Py<SampleBuffer>,
        selection: Selection,
        state: State,
        rate: f64,
        routes: PyReadonlyArray2<'_, f64>,
        initial: f64,
        attack: AmplitudeSegments,
        release: AmplitudeSegments,
        minimum_hold_frames: f64,
        voices: usize,
        batch_capacity: usize,
        filters: Inputs<'_>,
        filter_after_amplitude: bool,
        end_fade_frames: f64,
    ) -> PyResult<usize> {
        if self.frame != 0
            || !end_fade_frames.is_finite()
            || end_fade_frames < 0.0
            || voices == 0
            || batch_capacity == 0
            || rate != self.rate
            || routes.shape() != [buffer.borrow(py).audio.ncols(), self.channels]
            || routes.as_array().iter().any(|v| !v.is_finite())
            || !initial.is_finite()
            || !minimum_hold_frames.is_finite()
            || minimum_hold_frames < 0.0
        {
            return Err(PyValueError::new_err("Invalid live sample source"));
        }
        let buffer = buffer.borrow(py).clone();
        let mut cursor = Cursor::new(selection, rate, state, end_fade_frames);
        cursor.settle(0);
        if cursor.exhausted.is_some() {
            return Err(PyValueError::new_err("Live sample starts exhausted"));
        }
        let source_channels = buffer.audio.ncols();
        let filters = FilterBank::new(Some(filters), rate, 1, source_channels)?;
        let amplitude = Amplitude::new(initial, attack, release)?;
        let amplitude_states = (0..voices)
            .map(|_| amplitude.start())
            .collect::<NativeResult<_>>()?;
        let runtime = SampleRuntime {
            controls: ControlBank::new(Vec::new(), Vec::new(), voices, 0),
            route_maps: vec![Vec::new(); 5],
            generators: None,
            generator_active: vec![false; voices],
            crossfades: Vec::new(),
            filter_routes: Vec::new(),
            parameters: vec![
                1.0, 0.0, 1e300, 0.0, -120000.0, 120000.0, 1.0, 0.0, 1e300, 0.0, -120000.0,
                120000.0,
            ],
            parameter_partials: Vec::new(),
            filters: vec![filters; voices],
            filter_after_amplitude,
            buffer,
            initial_cursor: cursor,
            routes: routes.as_array().to_owned(),
            amplitude,
            amplitude_states,
            minimum_hold_frames,
            cursors: vec![cursor; voices],
            steps: vec![1.0; voices],
            gains: vec![0.0; voices],
            active: vec![false; voices],
            ages: vec![0; voices],
            delays: vec![0; voices],
            release_frames: vec![None; voices],
            fade_starts: vec![None; voices],
            fade_frames: vec![1.0; voices],
            fade_gains: vec![1.0; voices],
            source_values: vec![0.0; source_channels],
            frame: 0,
        };
        let (producer, consumer) = RingBuffer::new(batch_capacity);
        self.sources.push(Source {
            runtime: SourceRuntime::Sample(Box::new(runtime)),
            producer,
            consumer,
            action_scratch: vec![0.0; self.action_capacity * 6],
            action_count: 0,
        });
        Ok(self.sources.len() - 1)
    }

    fn submit(&mut self, source: usize, actions: PyReadonlyArray2<'_, f64>) -> PyResult<()> {
        let Some(source) = self.sources.get_mut(source) else {
            return Err(PyValueError::new_err("Unknown live runtime source"));
        };
        push_batch(&mut source.producer, actions, "source")
    }

    fn active_slots(&self, source: usize) -> PyResult<Vec<bool>> {
        self.sources
            .get(source)
            .map(|v| v.runtime.active_slots())
            .ok_or_else(|| PyValueError::new_err("Unknown live runtime source"))
    }

    #[allow(clippy::too_many_arguments)]
    fn set_sample_controls(
        &mut self,
        source: usize,
        controls: PyReadonlyArray2<'_, f64>,
        smoothing: Vec<(u64, u64)>,
        parameters: Vec<f64>,
        maps: Vec<(usize, Vec<(f64, f64)>)>,
        context_capacity: usize,
        crossfades: Vec<(bool, bool)>,
        filters: Vec<(usize, usize, usize, bool, f64, f64)>,
    ) -> PyResult<()> {
        let Some(Source {
            runtime: SourceRuntime::Sample(runtime),
            ..
        }) = self.sources.get_mut(source)
        else {
            return Err(PyValueError::new_err("Invalid live sample source"));
        };
        if self.frame != 0
            || parameters.len() != 3 * (4 + 2 * filters.len() + crossfades.len())
            || parameters.iter().any(|v| !v.is_finite())
            || parameters
                .chunks_exact(3)
                .any(|p| p[0] < p[1] || p[0] > p[2])
        {
            return Err(PyValueError::new_err(
                "Invalid live sample parameters or state",
            ));
        }
        let (definitions, states) = control_definitions(controls, &smoothing)?;
        if filters
            .iter()
            .any(|&(cutoff, q, stages, _, minimum, maximum)| {
                cutoff >= parameters.len() / 3
                    || q >= parameters.len() / 3
                    || stages == 0
                    || !minimum.is_finite()
                    || !maximum.is_finite()
                    || minimum <= 0.0
                    || minimum >= maximum
                    || maximum >= self.rate / 2.0
            })
            || (!filters.is_empty()
                && filters
                    .iter()
                    .try_fold(0usize, |total, f| total.checked_add(f.2))
                    != Some(runtime.filters[0].states.shape()[0]))
        {
            return Err(PyValueError::new_err("Invalid live sample filter routes"));
        }
        if maps.len() != definitions.len()
            || definitions
                .iter()
                .any(|d| d.parameter >= parameters.len() / 3)
        {
            return Err(PyValueError::new_err("Invalid live sample control routes"));
        }
        let maps = maps
            .into_iter()
            .map(|(kind, points)| RouteMap::new(kind, points))
            .collect::<PyResult<Vec<_>>>()?;
        if runtime.generators.is_some() {
            return Err(PyValueError::new_err(
                "Sample controls must be configured before generators",
            ));
        }
        runtime.parameter_partials = vec![0.0; definitions.len()];
        runtime.controls =
            ControlBank::new(definitions, states, runtime.active.len(), context_capacity);
        runtime.route_maps[0] = maps;
        runtime.parameters = parameters;
        runtime.crossfades = crossfades;
        runtime.filter_routes = filters
            .into_iter()
            .map(
                |(cutoff, q, stages, clamp, minimum, maximum)| SampleFilter {
                    cutoff,
                    q,
                    stages,
                    clamp,
                    minimum,
                    maximum,
                },
            )
            .collect();
        Ok(())
    }

    #[allow(clippy::type_complexity)]
    fn set_sample_generators(
        &mut self,
        source: usize,
        generators: &GeneratorBank,
        maps: Vec<Vec<(usize, Vec<(f64, f64)>)>>,
    ) -> PyResult<()> {
        let Some(Source {
            runtime: SourceRuntime::Sample(runtime),
            ..
        }) = self.sources.get_mut(source)
        else {
            return Err(PyValueError::new_err("Invalid live sample source"));
        };
        if self.frame != 0
            || generators.rate != self.rate
            || generators.patch_states.len() != runtime.active.len()
            || generators.parameter_count != runtime.parameters.len() / 3
            || maps.len() != 4
        {
            return Err(PyValueError::new_err(
                "Invalid live sample generator layout or state",
            ));
        }
        let maps = maps
            .into_iter()
            .enumerate()
            .map(|(kind, maps)| generators.prepare_route_maps(kind + 1, maps))
            .collect::<PyResult<Vec<_>>>()?;
        let mut generators = generators.clone();
        generators.prepare_patch(0)?;
        for (target, maps) in runtime.route_maps[1..].iter_mut().zip(maps) {
            *target = maps;
        }
        runtime
            .parameter_partials
            .resize(runtime.route_maps.iter().map(Vec::len).sum(), 0.0);
        runtime.generators = Some(generators);
        Ok(())
    }

    fn active_contexts(&self, source: usize) -> PyResult<Vec<bool>> {
        let Some(source) = self.sources.get(source) else {
            return Err(PyValueError::new_err("Unknown live runtime source"));
        };
        source.runtime.active_contexts()
    }

    #[pyo3(signature = (output, inputs=Vec::new()))]
    fn process_into(
        &mut self,
        py: Python<'_>,
        mut output: PyReadwriteArray2<'_, f64>,
        inputs: Vec<PyReadonlyArray2<'_, f64>>,
    ) -> PyResult<()> {
        if !output.is_c_contiguous()
            || output.shape()[0] == 0
            || output.shape()[0] > self.maximum_block_frames
            || output.shape()[1] != self.output_scratch.ncols()
        {
            return Err(PyValueError::new_err("Invalid live runtime output"));
        }
        let frames = output.shape()[0];
        let expected = self.effects.as_ref().map_or(0, |effects| {
            effects
                .input_routes
                .iter()
                .filter(|route| *route == &[-2])
                .count()
        });
        if inputs.len() != expected {
            return Err(PyValueError::new_err(
                "Live host inputs do not match the prepared graph",
            ));
        }
        if let Some(effects) = &mut self.effects {
            let mut host = 0;
            for (input, route) in effects.input_routes.iter().enumerate() {
                if route == &[-2] {
                    let values = inputs[host].as_array();
                    if values.shape() != [frames, effects.widths[input]]
                        || values.iter().any(|value| !value.is_finite())
                    {
                        return Err(PyValueError::new_err("Invalid live host input"));
                    }
                    host += 1;
                }
            }
            let mut host = 0;
            for (input, route) in effects.input_routes.iter().enumerate() {
                if route == &[-2] {
                    for ((frame, channel), value) in inputs[host].as_array().indexed_iter() {
                        effects.inputs[[input, frame, channel]] = *value;
                    }
                    host += 1;
                } else {
                    for frame in 0..frames {
                        for channel in 0..effects.widths[input] {
                            effects.inputs[[input, frame, channel]] = 0.0;
                        }
                    }
                }
            }
        }
        let result = py.detach(|| {
            #[cfg(feature = "allocation-audit")]
            let audit = crate::heap_audit::Scope::new();
            let result = self.process(frames);
            #[cfg(feature = "allocation-audit")]
            audit.finish(result.is_ok());
            result
        });
        if let Err(error) = result {
            output.as_array_mut().fill(0.0);
            self.failed = true;
            // Construct the Python exception only after native processing and its audit end.
            return Err(error.into());
        }
        for (target, value) in output
            .as_array_mut()
            .iter_mut()
            .zip(self.output_scratch.iter())
        {
            *target = *value;
        }
        Ok(())
    }

    fn snapshot(&self) -> PyResult<LiveRuntimeSnapshot> {
        if self
            .sources
            .iter()
            .any(|v| v.consumer.slots() != 0 || v.action_count != 0)
            || self
                .effects
                .as_ref()
                .is_some_and(|v| v.consumer.slots() != 0 || v.action_count != 0)
        {
            return Err(PyValueError::new_err(
                "Cannot snapshot a live runtime with queued actions",
            ));
        }
        Ok(LiveRuntimeSnapshot {
            sources: self.sources.iter().map(|v| v.runtime.snapshot()).collect(),
            effects: self.effects.as_ref().map(EffectRuntime::snapshot),
            frame: self.frame,
            failed: self.failed,
        })
    }

    fn restore(&mut self, snapshot: &LiveRuntimeSnapshot) -> PyResult<()> {
        if snapshot.failed {
            return Err(PyValueError::new_err(
                "Cannot restore a failed live snapshot",
            ));
        }
        if snapshot.sources.len() != self.sources.len() {
            return Err(PyValueError::new_err(
                "Snapshot belongs to a different live runtime",
            ));
        }
        if snapshot.effects.is_some() != self.effects.is_some() {
            return Err(PyValueError::new_err(
                "Snapshot belongs to a different live effect graph",
            ));
        }
        for (source, state) in self.sources.iter().zip(&snapshot.sources) {
            source.runtime.validate_snapshot(state)?;
        }
        if let (Some(effects), Some(state)) = (&self.effects, &snapshot.effects) {
            effects.validate_snapshot(state)?;
        }
        for (source, state) in self.sources.iter_mut().zip(&snapshot.sources) {
            source.runtime.restore(state)?;
            while source.consumer.pop().is_ok() {}
            source.action_count = 0;
        }
        if let (Some(effects), Some(state)) = (&mut self.effects, &snapshot.effects) {
            effects.restore(state)?;
            while effects.consumer.pop().is_ok() {}
            effects.action_count = 0;
        }
        self.frame = snapshot.frame;
        self.failed = false;
        Ok(())
    }

    fn failed(&self) -> bool {
        self.failed
    }
}

impl LiveRuntime {
    fn process(&mut self, frames: usize) -> NativeResult<()> {
        let output_channels = self.output_scratch.ncols();
        for value in self
            .output_scratch
            .iter_mut()
            .take(frames * output_channels)
        {
            *value = 0.0;
        }
        if self.failed {
            return Ok(());
        }
        for source in &mut self.sources {
            source.action_count = drain_batches(
                &mut source.consumer,
                &mut source.action_scratch,
                self.action_capacity,
                "source",
            )?;
        }
        if let Some(effects) = &mut self.effects {
            effects.action_count = drain_batches(
                &mut effects.consumer,
                &mut effects.action_scratch,
                self.action_capacity,
                "effect",
            )?;
        }
        for (index, source) in self.sources.iter_mut().enumerate() {
            let mut scratch = ArrayViewMut2::from_shape(
                (frames, self.channels),
                &mut self
                    .source_scratch
                    .as_slice_mut()
                    .expect("owned callback scratch")[..frames * self.channels],
            )
            .expect("validated callback scratch shape");
            source.runtime.render_into(
                scratch.view_mut(),
                &source.action_scratch[..source.action_count * 6],
            )?;
            source.action_count = 0;
            if let Some(effects) = &mut self.effects {
                for (input, route) in effects.input_routes.iter().enumerate() {
                    if route == &[-1] || route.contains(&(index as i64)) {
                        for ((frame, channel), value) in scratch.indexed_iter() {
                            effects.inputs[[input, frame, channel]] += *value;
                        }
                    }
                }
            } else {
                for (mixed, value) in self.output_scratch.iter_mut().zip(scratch.iter()) {
                    *mixed += *value;
                }
            }
        }
        if let Some(effects) = &mut self.effects {
            effects.process(&mut self.output_scratch, frames, self.rate, self.frame)?;
        }
        if self
            .output_scratch
            .iter()
            .take(frames * output_channels)
            .any(|v| !v.is_finite())
        {
            return Err(NativeError::Message("Non-finite live runtime output"));
        }
        self.frame = self
            .frame
            .checked_add(frames)
            .ok_or(NativeError::Message("Live runtime frame overflow"))?;
        Ok(())
    }
}

fn push_batch(
    producer: &mut Producer<ActionBatch>,
    actions: PyReadonlyArray2<'_, f64>,
    subject: &str,
) -> PyResult<()> {
    if actions.shape()[1] != 6
        || actions.shape()[0] == 0
        || actions.as_array().iter().any(|v| !v.is_finite())
    {
        return Err(PyValueError::new_err(format!(
            "Invalid live {subject} action batch"
        )));
    }
    let values = actions.as_array();
    for start in (0..actions.shape()[0]).step_by(BATCH_ACTIONS) {
        let end = (start + BATCH_ACTIONS).min(actions.shape()[0]);
        let mut batch = ActionBatch {
            len: end - start,
            actions: [[0.0; 6]; BATCH_ACTIONS],
        };
        for (target, index) in batch.actions.iter_mut().zip(start..end) {
            target.copy_from_slice(
                values
                    .row(index)
                    .as_slice()
                    .ok_or_else(|| PyValueError::new_err("Action rows must be contiguous"))?,
            );
        }
        producer
            .push(batch)
            .map_err(|_| PyValueError::new_err(format!("Live {subject} action queue is full")))?;
    }
    Ok(())
}

fn drain_batches(
    consumer: &mut Consumer<ActionBatch>,
    scratch: &mut [f64],
    action_capacity: usize,
    subject: &'static str,
) -> NativeResult<usize> {
    let batches = consumer.slots();
    let mut count = 0;
    for _ in 0..batches {
        let batch = consumer.pop().expect("captured queue boundary");
        if count + batch.len > action_capacity {
            return Err(NativeError::CallbackCapacity(subject));
        }
        for action in &batch.actions[..batch.len] {
            scratch[count * 6..count * 6 + 6].copy_from_slice(action);
            count += 1;
        }
    }
    Ok(count)
}

impl SourceRuntime {
    fn active_slots(&self) -> Vec<bool> {
        match self {
            Self::Generated(runtime) => runtime.active_slots(),
            Self::Sample(runtime) => runtime.active.clone(),
        }
    }

    fn active_contexts(&self) -> PyResult<Vec<bool>> {
        match self {
            Self::Generated(runtime) => Ok(runtime.active_contexts()),
            Self::Sample(runtime) => Ok(runtime
                .controls
                .context_kinds
                .iter()
                .map(|k| *k != 0)
                .collect()),
        }
    }

    fn render_into(&mut self, output: ArrayViewMut2<'_, f64>, actions: &[f64]) -> NativeResult<()> {
        match self {
            Self::Generated(runtime) => runtime.render_into(output, 0.0, 1.0, 0.0, actions),
            Self::Sample(runtime) => runtime.render_into(output, actions),
        }
    }

    fn snapshot(&self) -> SourceSnapshot {
        match self {
            Self::Generated(runtime) => SourceSnapshot::Generated(Box::new(runtime.snapshot())),
            Self::Sample(runtime) => SourceSnapshot::Sample(runtime.clone()),
        }
    }

    fn validate_snapshot(&self, snapshot: &SourceSnapshot) -> PyResult<()> {
        match (self, snapshot) {
            (Self::Generated(runtime), SourceSnapshot::Generated(state)) => {
                runtime.validate_snapshot(state)
            }
            (Self::Sample(runtime), SourceSnapshot::Sample(state)) => {
                if runtime.routes != state.routes
                    || runtime.initial_cursor.end_fade_frames
                        != state.initial_cursor.end_fade_frames
                    || runtime.initial_cursor.selection != state.initial_cursor.selection
                    || runtime.initial_cursor.repeats_remaining
                        != state.initial_cursor.repeats_remaining
                    || runtime.initial_cursor.plays_remaining
                        != state.initial_cursor.plays_remaining
                    || runtime.controls.control_definitions != state.controls.control_definitions
                    || runtime.controls.context_kinds.len() != state.controls.context_kinds.len()
                    || runtime.route_maps != state.route_maps
                    || match (&runtime.generators, &state.generators) {
                        (Some(a), Some(b)) => !a.same_definition(b),
                        (None, None) => false,
                        _ => true,
                    }
                    || runtime.crossfades != state.crossfades
                    || runtime.filter_routes != state.filter_routes
                    || runtime.parameters != state.parameters
                    || runtime.filter_after_amplitude != state.filter_after_amplitude
                    || !runtime
                        .filters
                        .iter()
                        .zip(&state.filters)
                        .all(|(a, b)| a.same_definition(b))
                    || runtime.amplitude != state.amplitude
                    || runtime.minimum_hold_frames != state.minimum_hold_frames
                    || runtime.cursors.len() != state.cursors.len()
                    || runtime.buffer.audio.as_ref() != state.buffer.audio.as_ref()
                {
                    return Err(PyValueError::new_err(
                        "Snapshot belongs to a different live sample source",
                    ));
                }
                Ok(())
            }
            _ => Err(PyValueError::new_err(
                "Snapshot belongs to a different live source kind",
            )),
        }
    }

    fn restore(&mut self, snapshot: &SourceSnapshot) -> PyResult<()> {
        self.validate_snapshot(snapshot)?;
        match (self, snapshot) {
            (Self::Generated(runtime), SourceSnapshot::Generated(state)) => runtime.restore(state),
            (Self::Sample(runtime), SourceSnapshot::Sample(state)) => {
                runtime.clone_from(state);
                Ok(())
            }
            _ => unreachable!("validated source kind"),
        }
    }
}

impl SampleRuntime {
    fn start_generators(&mut self, voice: usize, frame: usize) -> NativeResult<()> {
        if !self.generator_active[voice] {
            if let Some(generators) = &mut self.generators {
                generators.start_patch(voice, frame)?;
            }
            self.generator_active[voice] = true;
        }
        Ok(())
    }

    fn parameter(&mut self, voice: usize, parameter: usize, frame: usize) -> NativeResult<f64> {
        let mut count = 0;
        let product = self.controls.contributions(
            voice,
            parameter,
            &self.route_maps[0],
            &mut self.parameter_partials,
            &mut count,
        )?;
        let product = if let Some(generators) = &self.generators {
            generators.contributions(
                voice,
                parameter,
                product,
                frame,
                &self.controls,
                &self.route_maps,
                &mut self.parameter_partials,
                &mut count,
            )?
        } else {
            product
        };
        let domain = &self.parameters[parameter * 3..parameter * 3 + 3];
        let value = (domain[0] + sum_partials(&self.parameter_partials[..count])) * product;
        if !value.is_finite() || value < domain[1] || value > domain[2] {
            return Err(NativeError::Message(
                "Modulated sample parameter is outside its domain",
            ));
        }
        Ok(value)
    }

    fn render_into(
        &mut self,
        mut output: ArrayViewMut2<'_, f64>,
        actions: &[f64],
    ) -> NativeResult<()> {
        let frames = output.nrows();
        if output.ncols() != self.routes.ncols()
            || actions.len() % 6 != 0
            || actions.chunks_exact(6).any(|a| {
                let offset = a[0] as usize;
                a[0] != offset as f64 || offset >= frames
            })
            || actions
                .chunks_exact(6)
                .zip(actions.chunks_exact(6).skip(1))
                .any(|(a, b)| b[0] < a[0])
        {
            return Err(NativeError::Message("Invalid live sample block"));
        }
        output.fill(0.0);
        let mut action = 0;
        let channels = output.ncols();
        for offset in 0..frames {
            let frame = self
                .frame
                .checked_add(offset)
                .ok_or(NativeError::Message("Live sample frame overflow"))?;
            if action < actions.len() && actions[action] as usize == offset {
                if let Some(generators) = &mut self.generators {
                    generators.settle_patches(&self.generator_active, frame, false)?;
                }
            }
            while action < actions.len() && actions[action] as usize == offset {
                self.apply_action(&actions[action..action + 6], frame)?;
                action += 6;
            }
            for voice in 0..self.active.len() {
                if self.active[voice] && self.delays[voice] == 0 {
                    self.start_generators(voice, frame)?;
                }
            }
            if let Some(generators) = &mut self.generators {
                generators.settle_patches(&self.generator_active, frame, true)?;
            }
            for voice in 0..self.active.len() {
                if !self.active[voice] {
                    continue;
                }
                if self.delays[voice] > 0 {
                    self.delays[voice] -= 1;
                    continue;
                }
                if let Some(generators) = &mut self.generators {
                    generators.observe_patch(voice, frame, &self.controls)?;
                }
                let age = self.ages[voice] as f64;
                let fade = self.fade_starts[voice].map_or(1.0, |start| {
                    self.fade_gains[voice]
                        * (1.0 - (self.ages[voice] - start) as f64 / self.fade_frames[voice])
                            .max(0.0)
                });
                if fade == 0.0 {
                    self.active[voice] = false;
                    continue;
                }
                if let Some(release_frame) = self.release_frames[voice] {
                    if age >= release_frame.ceil() && !self.cursors[voice].released {
                        self.cursors[voice].release();
                    }
                }
                self.amplitude.settle(
                    &mut self.amplitude_states[voice],
                    self.release_frames[voice].map(frame_time).transpose()?,
                    self.ages[voice],
                )?;
                let value = self
                    .amplitude
                    .value(&self.amplitude_states[voice], self.ages[voice])?;
                if self
                    .amplitude
                    .complete(&self.amplitude_states[voice], &value)
                {
                    self.active[voice] = false;
                    continue;
                }
                let mut level = value.value
                    * self.parameter(voice, 0, frame)?
                    * self.parameter(voice, 2, frame)?;
                for index in 0..self.crossfades.len() {
                    let position = self
                        .parameter(voice, 4 + 2 * self.filter_routes.len() + index, frame)?
                        .clamp(0.0, 1.0);
                    let (inward, equal_power) = self.crossfades[index];
                    let weight = if equal_power && position > 0.0 && position < 1.0 {
                        if inward {
                            (PI * position / 2.0).sin()
                        } else {
                            (PI * position / 2.0).cos()
                        }
                    } else if inward {
                        position
                    } else {
                        1.0 - position
                    };
                    level *= weight;
                }
                let step = self.steps[voice]
                    * ((self.parameter(voice, 1, frame)? + self.parameter(voice, 3, frame)?)
                        / 1200.0)
                        .exp2();
                let source_frame = i64::try_from(frame)
                    .map_err(|_| NativeError::Message("Live sample frame overflow"))?;
                let cursor = &mut self.cursors[voice];
                cursor.settle(source_frame);
                if cursor.exhausted.is_some() {
                    self.active[voice] = false;
                    continue;
                }
                let end_gain = cursor.end_gain(source_frame, step);
                if cursor.exhausted.is_some() {
                    self.active[voice] = false;
                    continue;
                }
                let mut following = *cursor;
                if cursor.position != 0.0 {
                    following.index += following.direction;
                    following.position = 0.0;
                    following.settle(source_frame);
                }
                for (channel, value) in self.source_values.iter_mut().enumerate() {
                    let first = cursor.read(&self.buffer.audio, channel);
                    let last = if following.exhausted.is_some() {
                        first
                    } else {
                        following.read(&self.buffer.audio, channel)
                    };
                    *value = if cursor.position == 0.0 {
                        first
                    } else {
                        first + (cursor.position / cursor.rate) * (last - first)
                    };
                }
                let amplitude = level * self.gains[voice];
                if self.filter_after_amplitude {
                    for value in &mut self.source_values {
                        *value *= amplitude;
                    }
                }
                if self.filter_routes.is_empty() {
                    self.filters[voice].process(0, &mut self.source_values)?;
                } else {
                    let mut stage = 0;
                    for index in 0..self.filter_routes.len() {
                        let route = self.filter_routes[index].clone();
                        let mut cutoff = self.parameter(voice, route.cutoff, frame)?;
                        let q = self.parameter(voice, route.q, frame)?;
                        if q <= 0.0
                            || (!route.clamp && (cutoff < route.minimum || cutoff > route.maximum))
                        {
                            return Err(NativeError::Message(
                                "Invalid live sample filter cutoff/Q",
                            ));
                        }
                        cutoff = cutoff.clamp(route.minimum, route.maximum);
                        for _ in 0..route.stages {
                            self.filters[voice].process_stage(
                                self.frame + offset,
                                stage,
                                cutoff,
                                q,
                                &mut self.source_values,
                            )?;
                            stage += 1;
                        }
                    }
                }
                let output_gain = fade
                    * end_gain
                    * if self.filter_after_amplitude {
                        1.0
                    } else {
                        amplitude
                    };
                for channel in 0..channels {
                    let mixed = self
                        .source_values
                        .iter()
                        .enumerate()
                        .map(|(source, value)| value * self.routes[[source, channel]])
                        .sum::<f64>();
                    output[[offset, channel]] += mixed * output_gain;
                }
                self.cursors[voice].advance(step, source_frame + 1);
                self.ages[voice] += 1;
            }
            for (running, active) in self.generator_active.iter_mut().zip(&self.active) {
                *running &= *active;
            }
            self.controls.advance();
        }
        self.controls.retire_trigger_contexts(&self.active);
        self.frame = self
            .frame
            .checked_add(frames)
            .ok_or(NativeError::Message("Live sample frame overflow"))?;
        Ok(())
    }

    fn apply_action(&mut self, action: &[f64], frame: usize) -> NativeResult<()> {
        let kind = action[1] as usize;
        if (7..=11).contains(&kind) {
            let mut control_action = [0.0; 6];
            control_action.copy_from_slice(action);
            control_action[1] -= 3.0;
            return self.controls.apply(&control_action);
        }
        if (12..=16).contains(&kind) {
            let mut generator_action = [0.0; 6];
            generator_action.copy_from_slice(action);
            generator_action[1] -= 4.0;
            let voice = action[2] as usize;
            if (13..=15).contains(&kind)
                && action[2] == voice as f64
                && voice < self.active.len()
                && self.active[voice]
                && self.delays[voice] == 0
            {
                self.start_generators(voice, frame)?;
            }
            let Some(generators) = &mut self.generators else {
                return Err(NativeError::Message("Sample source has no generators"));
            };
            return generators.apply_action(
                &generator_action,
                &self.generator_active,
                frame,
                generators.rate,
            );
        }
        let voice = action[2] as usize;
        if action[1] != kind as f64 || action[2] != voice as f64 || voice >= self.active.len() {
            return Err(NativeError::Message("Invalid live sample action"));
        }
        match kind {
            0 => {
                if self.active[voice]
                    || !action[3].is_finite()
                    || action[3] <= 0.0
                    || !action[4].is_finite()
                    || action[4] < 0.0
                {
                    return Err(NativeError::Message("Invalid live sample start"));
                }
                self.active[voice] = true;
                self.generator_active[voice] = false;
                self.cursors[voice] = self.initial_cursor;
                let offset = action[5];
                if !offset.is_finite() || offset.fract() != 0.0 || offset.abs() > 9007199254740992.0
                {
                    return Err(NativeError::Message("Invalid live sample offset"));
                }
                let cursor = &mut self.cursors[voice];
                cursor.index = cursor
                    .index
                    .checked_add(offset as i64)
                    .ok_or(NativeError::Message("Live sample offset overflow"))?;
                if !(cursor.selection.0..cursor.selection.1).contains(&cursor.index) {
                    cursor.exhausted = Some(frame as i64);
                    self.active[voice] = false;
                }
                self.steps[voice] = action[3];
                self.gains[voice] = action[4];
                self.ages[voice] = 0;
                self.controls.reset_voice(voice);
                self.filters[voice].states.fill(0.0);
                self.delays[voice] = 0;
                self.amplitude_states[voice] = self.amplitude.start()?;
                self.release_frames[voice] = None;
                self.fade_starts[voice] = None;
                self.fade_gains[voice] = 1.0;
            }
            1 => {
                if self.active[voice] && self.release_frames[voice].is_none() {
                    if self.delays[voice] > 0 {
                        self.active[voice] = false;
                        self.generator_active[voice] = false;
                        return Ok(());
                    }
                    self.start_generators(voice, frame)?;
                    if let Some(generators) = &mut self.generators {
                        generators.release_voice(
                            voice,
                            frame,
                            self.ages[voice],
                            frame_time(self.minimum_hold_frames)?,
                        )?;
                    }
                    let release_frame = (self.ages[voice] as f64).max(self.minimum_hold_frames);
                    self.release_frames[voice] = Some(release_frame);
                }
            }
            2 => {
                self.active[voice] = false;
                self.generator_active[voice] = false;
            }
            3 => {
                if !self.active[voice] || action[3] <= 0.0 || action[4] < 0.0 || action[5] != 0.0 {
                    return Err(NativeError::Message("Invalid live sample control"));
                }
                self.steps[voice] = action[3];
                self.gains[voice] = action[4];
            }
            4 => {
                let command = action[3] as usize;
                if action[3] != command as f64
                    || command > 5
                    || !action[4].is_finite()
                    || action[5] != 0.0
                {
                    return Err(NativeError::Message("Invalid live amplitude command"));
                }
                if !self.active[voice] {
                    return Ok(());
                }
                self.amplitude.settle(
                    &mut self.amplitude_states[voice],
                    self.release_frames[voice].map(frame_time).transpose()?,
                    self.ages[voice],
                )?;
                self.amplitude.command(
                    &mut self.amplitude_states[voice],
                    self.ages[voice],
                    command,
                    action[4],
                )?;
            }
            5 => {
                if !action[3].is_finite()
                    || action[3] <= 0.0
                    || action[4] != 0.0
                    || action[5] != 0.0
                {
                    return Err(NativeError::Message("Invalid live sample fade"));
                }
                if self.active[voice] {
                    if self.delays[voice] > 0 {
                        self.active[voice] = false;
                        self.generator_active[voice] = false;
                        return Ok(());
                    }
                    if let Some(start) = self.fade_starts[voice] {
                        self.fade_gains[voice] *= (1.0
                            - (self.ages[voice] - start) as f64 / self.fade_frames[voice])
                            .max(0.0);
                    }
                    self.fade_starts[voice] = Some(self.ages[voice]);
                    self.fade_frames[voice] = action[3];
                }
            }
            6 => {
                if !action[3].is_finite()
                    || action[3] < 0.0
                    || action[3].fract() != 0.0
                    || action[3] > 9007199254740992.0
                    || action[4] != 0.0
                    || action[5] != 0.0
                {
                    return Err(NativeError::Message("Invalid live sample delay"));
                }
                self.delays[voice] = action[3] as usize;
            }
            _ => return Err(NativeError::Message("Unknown live sample action")),
        }
        Ok(())
    }
}
