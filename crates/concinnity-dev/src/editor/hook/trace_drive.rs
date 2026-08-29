// src/editor/hook/trace_drive.rs
//
// EditorHook: the execution-trace exchange with the behavior system. While the
// Behavior or Variables panel is open, a `TraceRequest` is published each frame
// (naming the selected entity and the resolved breakpoints) and the
// `ExecutionTrace` the last simulated tick reported is ingested: executed
// nodes refresh the open behavior's pulses, world variables and the selected
// entity's locals become the panels' live values, and a breakpoint hit pauses
// the transport on the node that fired. Closing both panels withdraws the
// request, so the running world records nothing.

use super::*;
use crate::ecs::{ExecutionTrace, TraceEvent, TracePaths, TraceRequest};
use crate::editor::behavior::path::Path;
use crate::editor::behavior::pulse::{self, NodePulse};
use crate::editor::behavior::trace;

impl EditorHook {
    pub(super) fn drive_trace(&mut self, world: &mut World) {
        if !self.behavior_open && !self.variables_open {
            world.remove_resource::<TraceRequest>();
            self.clear_trace();
            return;
        }
        // A stopped transport shows authored data, not a run's leftovers; the
        // request stays up so Play resumes reporting without a gap.
        if self.sim.state == sim::SimState::Stopped {
            self.clear_trace();
        }
        self.publish_trace_request(world);
        self.ingest_trace(world);
    }

    fn clear_trace(&mut self) {
        self.behavior_pulses.clear();
        self.live_vars.clear();
        self.live_locals.clear();
    }

    // The name the Behavior panel is open on. Live data is scoped to it:
    // pulses and locals address places inside one body.
    fn open_behavior_name(&self) -> String {
        self.behavior_entry()
            .and_then(|i| entry_name(&self.entries[i]))
            .unwrap_or("")
            .to_string()
    }

    // The open behavior's node paths in the editor's path type, indexed by
    // node id, from the runtime-published table.
    fn open_behavior_paths(&self, world: &World) -> Vec<Path> {
        let Some(id) = trace::id_of(&self.open_behavior_name()) else {
            return Vec::new();
        };
        world
            .resource::<TracePaths>()
            .and_then(|t| t.0.iter().find(|(b, _)| *b == id))
            .map(|(_, paths)| paths.iter().map(|p| trace::to_path(p)).collect())
            .unwrap_or_default()
    }

    fn publish_trace_request(&mut self, world: &mut World) {
        let entity = self
            .selection
            .active()
            .and_then(trace::id_of)
            .and_then(|id| {
                world
                    .resource::<concinnity_core::ecs::EntityByName>()
                    .and_then(|n| n.get(id))
            })
            .map(|e| e.to_bits());
        let breakpoints = self.resolve_breakpoints(world);
        world.insert_resource(TraceRequest {
            entity,
            breakpoints,
        });
    }

    // Breakpoints are held by behavior name + node path (both survive preview
    // rebuilds); the behavior system wants (asset id, node id), resolved fresh
    // against the published path table. One not yet resolvable (the table
    // appears one frame after the first request) simply stands down until it
    // is.
    fn resolve_breakpoints(&self, world: &World) -> Vec<TraceEvent> {
        let Some(table) = world.resource::<TracePaths>() else {
            return Vec::new();
        };
        self.behavior_breakpoints
            .iter()
            .filter_map(|(name, path)| {
                let id = trace::id_of(name)?;
                let (_, paths) = table.0.iter().find(|(b, _)| *b == id)?;
                let node = paths.iter().position(|p| trace::matches(p, path))?;
                Some(TraceEvent {
                    behavior: id,
                    node: node as u32,
                })
            })
            .collect()
    }

    fn ingest_trace(&mut self, world: &mut World) {
        let Some(t) = world.resource::<ExecutionTrace>() else {
            self.prune_pulses();
            return;
        };
        if t.frame == self.trace_seen {
            // A paused world republishes nothing; the pulses just decay.
            self.prune_pulses();
            return;
        }
        let frame = t.frame;
        let events = t.events.clone();
        let vars = t.vars.clone();
        let locals = t.locals.clone();
        let hit = t.hit;
        self.trace_seen = frame;

        let open_id = trace::id_of(&self.open_behavior_name());
        let open_paths = self.open_behavior_paths(world);
        let now = std::time::Instant::now();
        for event in &events {
            if Some(event.behavior) != open_id {
                continue;
            }
            let Some(path) = open_paths.get(event.node as usize) else {
                continue;
            };
            match self
                .behavior_pulses
                .iter_mut()
                .find(|p| p.node == event.node)
            {
                Some(p) => p.at = now,
                None => self.behavior_pulses.push(NodePulse {
                    node: event.node,
                    path: path.clone(),
                    at: now,
                }),
            }
        }
        self.prune_pulses();

        self.live_vars = vars
            .into_iter()
            .map(|(name, val)| {
                let (ty, text) = trace::text(val);
                (name, ty.to_string(), text)
            })
            .collect();
        self.live_locals = locals
            .into_iter()
            .filter(|(behavior, _, _)| Some(*behavior) == open_id)
            .map(|(_, name, val)| {
                let (ty, text) = trace::text(val);
                (name, ty.to_string(), text)
            })
            .collect();

        if let Some(hit) = hit {
            self.land_on_hit(hit, open_id, &open_paths, world);
        }
    }

    // A breakpoint fired: freeze the run mid-state and put the panel on the
    // node, so the pause reads as "stopped here" rather than just "stopped".
    fn land_on_hit(
        &mut self,
        hit: TraceEvent,
        open_id: Option<crate::ecs::asset_id::AssetId>,
        open_paths: &[Path],
        world: &mut World,
    ) {
        self.sim.pause();
        if Some(hit.behavior) != open_id {
            return;
        }
        let Some(path) = open_paths.get(hit.node as usize) else {
            return;
        };
        if let Some(row) = self.behavior_rows().iter().position(|r| &r.path == path) {
            self.select_behavior_row(row, world);
            self.ensure_behavior_visible();
        }
    }

    fn prune_pulses(&mut self) {
        self.behavior_pulses
            .retain(|p| p.at.elapsed().as_secs_f32() < pulse::PULSE_SECS);
    }

    // The Ctrl+click toggle on a chart card. Held by name + path, so it
    // survives rebuilds and stays put when unrelated edits shift node ids.
    pub(super) fn toggle_behavior_breakpoint(&mut self, path: &Path) {
        let name = self.open_behavior_name();
        if name.is_empty() {
            return;
        }
        match self
            .behavior_breakpoints
            .iter()
            .position(|(n, p)| n == &name && p == path)
        {
            Some(i) => {
                self.behavior_breakpoints.remove(i);
            }
            None => self.behavior_breakpoints.push((name, path.clone())),
        }
    }
}
