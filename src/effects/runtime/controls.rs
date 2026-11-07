//! Aligned route contributions resolved in audio/control dependency order.

use crate::error::{NativeError, NativeResult};
use crate::runtime::{RouteMap, add_partial, sum_partials};
use numpy::ndarray::{Array2, ArrayView2};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

pub(crate) type Definitions = (
    Vec<(usize, usize, f64, f64)>,
    Vec<(usize, i64, u8, usize, Vec<(f64, f64)>, f64, f64, String)>,
);

#[derive(Clone)]
pub(super) struct Delay {
    pub(super) history: Vec<f64>,
    pub(super) write: usize,
    pub(super) remaining: usize,
}

impl Delay {
    fn new(frames: usize, value: f64) -> Option<Self> {
        (frames > 0).then(|| Self {
            history: vec![value; frames],
            write: 0,
            remaining: frames,
        })
    }

    fn advance(&mut self, value: f64) -> f64 {
        let old = self.history[self.write];
        self.history[self.write] = value;
        self.write = (self.write + 1) % self.history.len();
        old
    }
}

#[derive(Clone)]
pub(super) struct Target {
    pub(super) node: usize,
    pub(super) parameter: usize,
    minimum: f64,
    maximum: f64,
    pub(super) base: Option<Delay>,
}

#[derive(Clone)]
pub(super) struct Route {
    pub(super) target: usize,
    source: Option<usize>,
    multiply: bool,
    mapping: RouteMap,
    minimum: f64,
    maximum: f64,
    pub(super) name: String,
    pub(super) delay: Option<Delay>,
}

#[derive(Clone)]
pub(crate) struct Controls {
    pub(crate) definitions: Definitions,
    pub(super) targets: Vec<Target>,
    pub(super) routes: Vec<Route>,
    pub(super) signals: Vec<f64>,
    partials: Vec<f64>,
}

impl Controls {
    pub(super) fn new(
        definitions: Definitions,
        kinds: &[u8],
        latencies: &[usize],
        inputs: usize,
        parameters: ArrayView2<'_, f64>,
    ) -> PyResult<Self> {
        let mut targets = Vec::new();
        for (node, parameter, minimum, maximum) in &definitions.0 {
            if *node >= kinds.len()
                || *parameter >= parameters.ncols()
                || !minimum.is_finite()
                || !maximum.is_finite()
                || minimum > maximum
                || targets
                    .iter()
                    .any(|t: &Target| t.node == *node && t.parameter == *parameter)
            {
                return Err(PyValueError::new_err("Invalid effect control target"));
            }
            let arrival = latencies[inputs + node] - if kinds[*node] == 4 { 48 } else { 0 };
            targets.push(Target {
                node: *node,
                parameter: *parameter,
                minimum: *minimum,
                maximum: *maximum,
                base: Delay::new(arrival, parameters[[*node, *parameter]]),
            });
        }
        let mut routes = Vec::new();
        for (target, source, operation, interpolation, points, minimum, maximum, name) in
            &definitions.1
        {
            let Some(destination) = targets.get(*target) else {
                return Err(PyValueError::new_err("Unknown effect control target"));
            };
            if *source < -1
                || *operation > 1
                || !minimum.is_finite()
                || !maximum.is_finite()
                || minimum > maximum
                || name.is_empty()
                || routes.iter().any(|r: &Route| r.name == *name)
                || (*source >= 0
                    && (*source as usize >= destination.node || kinds[*source as usize] != 14))
            {
                return Err(PyValueError::new_err("Invalid effect control route"));
            }
            let origin = (*source >= 0).then_some(*source as usize);
            let arrival = latencies[inputs + destination.node]
                - if kinds[destination.node] == 4 { 48 } else { 0 };
            let frames = arrival - origin.map_or(0, |s| latencies[inputs + s]);
            routes.push(Route {
                target: *target,
                source: origin,
                multiply: *operation == 1,
                mapping: RouteMap::new(*interpolation, points.clone())?,
                minimum: *minimum,
                maximum: *maximum,
                name: name.clone(),
                delay: Delay::new(frames, f64::from(*operation)),
            });
        }
        let capacity = routes.len();
        Ok(Self {
            definitions,
            targets,
            routes,
            signals: vec![0.0; capacity],
            partials: vec![0.0; capacity],
        })
    }

    pub(super) fn resolve(
        &mut self,
        node: usize,
        resolved: &mut Array2<f64>,
        values: &Array2<f64>,
        ended: &[bool],
        inputs: usize,
    ) -> NativeResult<bool> {
        let mut all_ended = true;
        for (index, target) in self.targets.iter_mut().enumerate() {
            if target.node != node {
                continue;
            }
            let mut base = resolved[[node, target.parameter]];
            if let Some(delay) = &mut target.base {
                base = delay.advance(base);
            }
            let mut count = 0;
            let mut product = 1.0;
            for (route_index, route) in self.routes.iter_mut().enumerate() {
                if route.target != index {
                    continue;
                }
                let mut route_ended = true;
                let mut amount = if let Some(source) = route.source {
                    let value = values[[inputs + source, 0]];
                    if value < route.minimum || value > route.maximum || !value.is_finite() {
                        return Err(NativeError::Message(
                            "Follower source value is outside its domain",
                        ));
                    }
                    route_ended = ended[inputs + source];
                    route.mapping.value(value)?
                } else {
                    self.signals[route_index]
                };
                if let Some(delay) = &mut route.delay {
                    amount = delay.advance(amount);
                    if route.source.is_some() {
                        let aligned = route_ended && delay.remaining == 0;
                        delay.remaining = if route_ended {
                            delay.remaining.saturating_sub(1)
                        } else {
                            delay.history.len()
                        };
                        route_ended = aligned;
                    }
                }
                all_ended &= route_ended;
                if !amount.is_finite() {
                    return Err(NativeError::Message(
                        "Non-finite effect control contribution",
                    ));
                }
                if route.multiply {
                    product *= amount;
                } else {
                    add_partial(&mut self.partials, &mut count, amount);
                }
            }
            let value = (base + sum_partials(&self.partials[..count])) * product;
            if !value.is_finite() || value < target.minimum || value > target.maximum {
                return Err(NativeError::Message(
                    "Modulated effect parameter is outside its domain",
                ));
            }
            resolved[[node, target.parameter]] = value;
        }
        Ok(all_ended)
    }
}
