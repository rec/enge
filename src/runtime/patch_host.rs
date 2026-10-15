//! Audio-host preparation, clocks and voice lifetime around the shared scheduler.

use super::{BeatPoint, SynthRuntime, core_error};
use crate::amplitude::frame_position;
use envelfo_core::envelope::Marker;
use envelfo_core::lfo;
use envelfo_core::lfo::Command;
use envelfo_core::patch::{
    Child, ChildCommand, ChildEvent, Connection, ConnectionAction, Node, Output, Patch, Port,
    StreamKeys,
};
use envelfo_core::sample_hold::SampleHold;
use envelfo_core::signal::Operator;
use envelfo_core::time::Time;
use pyo3::PyResult;
use std::collections::BTreeMap;

#[derive(Clone, PartialEq)]
pub(super) struct PatchHost {
    body: Patch,
    keys: StreamKeys,
    clocks: BTreeMap<String, u8>,
    lfos: Vec<String>,
    named: Vec<String>,
    staged: Vec<String>,
    transforms: Vec<String>,
    shapes: Vec<(f64, f64)>,
}

pub(super) struct Clocks<'a> {
    points: &'a [BeatPoint],
    kinds: &'a BTreeMap<String, u8>,
}

impl envelfo_core::patch::ClockMapping for Clocks<'_> {
    fn local_time(&self, child: &str, at: Time) -> Result<Time, envelfo_core::Error> {
        clock_at(self.points, self.kinds.get(child).copied().unwrap_or(0), at)
    }

    fn host_time(&self, child: &str, at: Time) -> Result<Option<Time>, envelfo_core::Error> {
        if self.kinds.get(child).copied().unwrap_or(0) == 0 {
            return Ok(Some(at));
        }
        for (index, point) in self.points.iter().enumerate() {
            if point.clock_rate > Time::ZERO && at >= point.clock_beat {
                let frame = point.clock_frame.checked_add(
                    at.checked_sub(point.clock_beat)?
                        .checked_div(point.clock_rate)?,
                )?;
                if self
                    .points
                    .get(index + 1)
                    .is_none_or(|n| frame <= n.clock_frame)
                {
                    return Ok(Some(frame));
                }
            }
        }
        Ok(None)
    }
}

impl PatchHost {
    pub(super) fn prepare(runtime: &SynthRuntime) -> PyResult<Option<Self>> {
        let lfos: Vec<String> = runtime
            .lfo_owners
            .iter()
            .map(|i| format!("lfo-{i}"))
            .collect();
        let named: Vec<String> = runtime
            .named_owners
            .iter()
            .map(|i| format!("named-{i}"))
            .collect();
        let staged: Vec<String> = runtime
            .staged_owners
            .iter()
            .map(|i| format!("stage-{i}"))
            .collect();
        let transforms: Vec<String> = (0..runtime.envelfo_transforms.len())
            .map(|i| format!("signal-{i}"))
            .collect();
        let mut body = Patch {
            children: vec![],
            outputs: vec![],
            event_outputs: vec![],
            connections: vec![],
        };
        let mut keys = StreamKeys::default();
        let mut clocks = BTreeMap::new();
        // Graph inputs include the authored LFO range. Audio route coefficients
        // are prepared against the raw waveform, so patch_value undoes this
        // affine range while keeping the core's activation weight separate.
        let mut shapes = vec![(0.0, 1.0); lfos.len()];
        for (_, inputs, _, _, _) in &runtime.envelfo_transforms {
            for &(kind, source, center, depth) in inputs {
                if kind == 0 {
                    shapes[runtime.lfo_owners[source]] = (center, depth);
                }
            }
        }
        for (i, definition) in runtime.lfo_definitions.iter().enumerate() {
            if runtime.lfo_owners[i] != i {
                continue;
            }
            let mut lfo = definition.body.clone();
            (lfo.center, lfo.depth) = shapes[i];
            let kind = if lfo.driver == lfo::Driver::Transport {
                2
            } else {
                u8::from(lfo.clock == lfo::Clock::Beats)
            };
            clocks.insert(lfos[i].clone(), kind);
            body.children.push(Child {
                name: lfos[i].clone(),
                node: if definition.scope == 3 {
                    Node::LFO {
                        body: lfo,
                        markers: vec![],
                    }
                } else {
                    Node::Input
                },
            });
        }
        for (i, definition) in runtime.named_envelopes.iter().enumerate() {
            if runtime.named_owners[i] != i {
                continue;
            }
            clocks.insert(named[i].clone(), u8::from(definition.beat_clock));
            let node = if let Some((key, minimum, maximum)) = definition.sample_hold {
                keys.samples.insert(named[i].clone(), key);
                Node::SampleHold(SampleHold { minimum, maximum })
            } else {
                Node::Envelope(definition.body.clone())
            };
            body.children.push(Child {
                name: named[i].clone(),
                node,
            });
        }
        for (i, definition) in runtime.staged_envelfos.iter().enumerate() {
            if runtime.staged_owners[i] != i {
                continue;
            }
            clocks.insert(staged[i].clone(), u8::from(definition.beat_clock));
            body.children.push(Child {
                name: staged[i].clone(),
                node: Node::Stages(definition.body.clone()),
            });
        }
        for (i, (kind, inputs, scale, offset, initial)) in
            runtime.envelfo_transforms.iter().enumerate()
        {
            let mut names = vec![];
            for &(source_kind, source, _, _) in inputs {
                let name = match source_kind {
                    0 => lfos[source].clone(),
                    1 => named[source].clone(),
                    2 => staged[source].clone(),
                    _ => transforms[source].clone(),
                };
                names.push(name);
            }
            let input = names[0].clone();
            let operator = match kind {
                0 => Operator::Sum(names),
                1 => Operator::Product(names),
                2 => Operator::Affine {
                    input,
                    scale: *scale,
                    offset: *offset,
                },
                3 => Operator::Threshold {
                    input,
                    lower: *scale,
                    upper: *offset,
                },
                4 => Operator::Slew {
                    input,
                    rise: *scale,
                    fall: *offset,
                    initial: *initial,
                },
                5 => Operator::Quantize {
                    input,
                    step: *scale,
                    origin: *offset,
                },
                _ => Operator::Latch(input),
            };
            body.children.push(Child {
                name: transforms[i].clone(),
                node: Node::Signal(operator),
            });
            body.outputs.push(Output {
                name: transforms[i].clone(),
                source: transforms[i].clone(),
            });
        }
        for c in &runtime.staged_connections {
            body.connections.push(Connection {
                source: Port {
                    child: staged[c.source].clone(),
                    port: c.port.clone(),
                },
                target: staged[c.destination].clone(),
                action: ConnectionAction::Cue(c.cue.clone()),
                every: c.every as u64,
                offset: c.offset as u64,
                probability: c.probability as f64 / (1_u64 << 53) as f64,
                delay: c.delay,
            });
            keys.connections.push(c.key);
        }
        for (i, c) in runtime.patch_events.iter().enumerate() {
            let source = if c.threshold {
                Port {
                    child: transforms[c.source].clone(),
                    port: if c.marker == Time::ONE {
                        "rising"
                    } else {
                        "falling"
                    }
                    .into(),
                }
            } else {
                let port = format!("marker-{i}");
                let child = body
                    .children
                    .iter_mut()
                    .find(|n| n.name == lfos[c.source])
                    .expect("prepared source exists");
                if let Node::LFO { markers, .. } = &mut child.node {
                    markers.push(Marker {
                        name: port.clone(),
                        position: c.marker,
                    });
                }
                Port {
                    child: lfos[c.source].clone(),
                    port,
                }
            };
            let (target, action) = destination(
                runtime,
                &named,
                &staged,
                &transforms,
                c.destination,
                c.capture,
                &c.cue,
            );
            body.connections.push(Connection {
                source,
                target,
                action,
                every: c.every as u64,
                offset: c.offset as u64,
                probability: c.probability as f64 / (1_u64 << 53) as f64,
                delay: if c.threshold {
                    c.delay.checked_sub(Time::ONE).map_err(core_error)?
                } else {
                    c.delay
                },
            });
            keys.connections.push(c.key);
        }
        for c in &runtime.patch_stage_starts {
            let (target, action) = destination(
                runtime,
                &named,
                &staged,
                &transforms,
                c.destination,
                c.capture,
                &None,
            );
            body.connections.push(Connection {
                source: Port {
                    child: staged[c.source].clone(),
                    port: c.port.clone(),
                },
                target,
                action,
                every: c.every as u64,
                offset: c.offset as u64,
                probability: c.probability as f64 / (1_u64 << 53) as f64,
                delay: c.delay,
            });
            keys.connections.push(c.key);
        }
        if body.children.is_empty() {
            return Ok(None);
        }
        if body.outputs.is_empty() {
            body.outputs.push(Output {
                name: "value".into(),
                source: body.children[0].name.clone(),
            });
        }
        Ok(Some(Self {
            body,
            keys,
            clocks,
            lfos,
            named,
            staged,
            transforms,
            shapes,
        }))
    }
}

impl SynthRuntime {
    pub(super) fn prepare_patch(&mut self) -> PyResult<()> {
        if self.patch_host.is_none() {
            self.patch_host = PatchHost::prepare(self)?;
        }
        Ok(())
    }

    pub(super) fn start_patch(&mut self, voice: usize) -> PyResult<()> {
        let Some(host) = &self.patch_host else {
            return Ok(());
        };
        let clock = Clocks {
            points: &self.beat_points,
            kinds: &host.clocks,
        };
        let at = frame_position(self.frame)?;
        let mut state = host
            .body
            .initial(
                at,
                "native",
                self.envelfo_keys[voice],
                Some(&host.keys),
                &clock,
            )
            .map_err(core_error)?;
        for child in &host.body.children {
            if matches!(child.node, Node::Envelope(_) | Node::Stages(_)) {
                let ordinal = host
                    .body
                    .next_ordinal(&state, &child.name)
                    .map_err(core_error)?;
                host.body
                    .command(
                        &mut state,
                        &child.name,
                        &ChildEvent {
                            at,
                            ordinal,
                            command: ChildCommand::Playback(Command::NoteOn),
                        },
                        &clock,
                        false,
                    )
                    .map_err(core_error)?;
            }
        }
        self.patch_states[voice] = Some(state);
        self.pending_releases[voice].fill(None);
        Ok(())
    }

    pub(super) fn patch_command(
        &mut self,
        voice: usize,
        kind: u8,
        source: usize,
        at: Time,
        command: ChildCommand,
    ) -> PyResult<()> {
        let Some(host) = &self.patch_host else {
            return Ok(());
        };
        let name = match kind {
            0 => &host.lfos[source],
            1 => &host.named[source],
            _ => &host.staged[source],
        };
        let state = self.patch_states[voice]
            .as_mut()
            .expect("active voice owns its Patch state");
        let ordinal = host.body.next_ordinal(state, name).map_err(core_error)?;
        host.body
            .command(
                state,
                name,
                &ChildEvent {
                    at,
                    ordinal,
                    command,
                },
                &Clocks {
                    points: &self.beat_points,
                    kinds: &host.clocks,
                },
                false,
            )
            .map_err(core_error)?;
        Ok(())
    }

    pub(super) fn settle_patches(&mut self, inclusive: bool) -> PyResult<()> {
        let at = frame_position(self.frame)?;
        for voice in 0..self.frequencies.len() {
            if !self.active[voice] {
                continue;
            }
            let mut due: Vec<_> = self.pending_releases[voice]
                .iter()
                .enumerate()
                .filter_map(|(i, t)| {
                    t.filter(|t| if inclusive { *t <= at } else { *t < at })
                        .map(|t| (t, i))
                })
                .collect();
            due.sort();
            for (t, source) in due {
                self.patch_command(
                    voice,
                    1,
                    source,
                    t,
                    ChildCommand::Playback(Command::NoteOff),
                )?;
                self.pending_releases[voice][source] = None;
            }
            // The frame observation performs the inclusive event phase once.
            if inclusive {
                continue;
            }
            if let (Some(host), Some(state)) = (&self.patch_host, &mut self.patch_states[voice]) {
                host.body
                    .settle(
                        state,
                        at,
                        inclusive,
                        &Clocks {
                            points: &self.beat_points,
                            kinds: &host.clocks,
                        },
                    )
                    .map_err(core_error)?;
            }
        }
        Ok(())
    }

    pub(super) fn observe_patch(&mut self, voice: usize) -> PyResult<()> {
        let Some(host) = &self.patch_host else {
            return Ok(());
        };
        let at = frame_position(self.frame)?;
        let mut inputs = vec![];
        for (i, definition) in self.lfo_definitions.iter().enumerate() {
            if definition.scope != 3 && self.lfo_owners[i] == i {
                let (value, weight) = self.lfo_signal(voice, i)?;
                let (center, depth) = host.shapes[i];
                inputs.push((host.lfos[i].clone(), (center + depth * value) * weight));
            }
        }
        let state = self.patch_states[voice]
            .as_mut()
            .expect("active voice owns its Patch state");
        for (name, value) in inputs {
            host.body
                .set_input(state, &name, value)
                .map_err(core_error)?;
        }
        let values = host
            .body
            .advance(
                state,
                at,
                at.checked_add(Time::ONE).map_err(core_error)?,
                &Clocks {
                    points: &self.beat_points,
                    kinds: &host.clocks,
                },
            )
            .map_err(core_error)?
            .values;
        let base = voice * self.envelfo_transforms.len();
        for (i, name) in host.transforms.iter().enumerate() {
            self.envelfo_transform_values[base + i] = values[name];
        }
        Ok(())
    }

    pub(super) fn patch_value(
        &self,
        voice: usize,
        kind: u8,
        source: usize,
    ) -> PyResult<lfo::Value> {
        let host = self
            .patch_host
            .as_ref()
            .expect("prepared Patch owns executable children");
        let name = match kind {
            0 => &host.lfos[source],
            1 => &host.named[source],
            _ => &host.staged[source],
        };
        let state = self.patch_states[voice]
            .as_ref()
            .expect("active voice owns its Patch state");
        let mut value = host
            .body
            .child_value(
                state,
                name,
                frame_position(self.frame)?,
                &Clocks {
                    points: &self.beat_points,
                    kinds: &host.clocks,
                },
            )
            .map_err(core_error)?;
        if kind == 0 {
            let (center, depth) = host.shapes[self.lfo_owners[source]];
            value.value = if depth == 0.0 {
                0.0
            } else {
                (value.value - center) / depth
            };
        }
        Ok(value)
    }
}

pub(super) fn clock_at(
    points: &[BeatPoint],
    kind: u8,
    at: Time,
) -> Result<Time, envelfo_core::Error> {
    if kind == 0 {
        return Ok(at);
    }
    let index = points.partition_point(|p| p.clock_frame <= at);
    let point = points
        .get(index.wrapping_sub(1))
        .ok_or(envelfo_core::Error::InvalidDefinition(
            "beat clock requires a tempo map",
        ))?;
    let origin = if kind == 2 {
        point.song_beat
    } else {
        point.clock_beat
    };
    origin.checked_add(
        at.checked_sub(point.clock_frame)?
            .checked_mul(point.clock_rate)?,
    )
}

fn destination(
    runtime: &SynthRuntime,
    named: &[String],
    staged: &[String],
    transforms: &[String],
    index: usize,
    capture: bool,
    cue: &Option<String>,
) -> (String, ConnectionAction) {
    if capture {
        (transforms[index].clone(), ConnectionAction::Capture)
    } else if let Some(cue) = cue {
        (staged[index].clone(), ConnectionAction::Cue(cue.clone()))
    } else {
        (
            named[index].clone(),
            if runtime.named_envelopes[index].sample_hold.is_some() {
                ConnectionAction::Sample
            } else {
                ConnectionAction::Start
            },
        )
    }
}
