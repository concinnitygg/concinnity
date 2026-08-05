// Execution tracing: while an observer publishes a `TraceRequest` (the
// `cn editor` Behavior panel), each simulated tick publishes which nodes ran,
// the world variables, and the requested entity's locals as an
// `ExecutionTrace` resource. Without a request the system does no recording
// work, so a shipped runtime pays nothing.

use std::collections::BTreeSet;

use super::*;
use crate::ecs::{ExecutionTrace, TraceEvent, TracePaths, TraceRequest};

impl BehaviorSystem {
    // Publish this tick's observation. `fired` holds each run's executed node
    // ids by program index, in run order.
    pub(super) fn publish_trace(
        &mut self,
        ctx: &mut PipelineContext,
        request: &TraceRequest,
        fired: &[(usize, Vec<u32>)],
    ) {
        // The node-path table is a compile product; expose it once per world
        // so the observer can resolve node ids to authored-tree paths.
        if !self.trace_paths_published {
            self.trace_paths_published = true;
            ctx.insert_resource(TracePaths(
                self.programs
                    .iter()
                    .map(|p| (p.def.asset_id, p.paths.clone()))
                    .collect(),
            ));
        }

        let mut hit = None;
        let mut seen = BTreeSet::new();
        let mut events = Vec::new();
        for (i, nodes) in fired {
            let behavior = self.programs[*i].def.asset_id;
            for node in nodes {
                let event = TraceEvent {
                    behavior,
                    node: *node,
                };
                if hit.is_none() && request.breakpoints.contains(&event) {
                    hit = Some(event);
                }
                if seen.insert((behavior, *node)) {
                    events.push(event);
                }
            }
        }

        let vars = self
            .var_table
            .names()
            .iter()
            .zip(&self.vars)
            .map(|(name, value)| (name.clone(), value.to_trace()))
            .collect();

        let locals = request
            .entity
            .and_then(Entity::from_bits)
            .map(|entity| self.locals_of(entity))
            .unwrap_or_default();

        self.trace_frame += 1;
        ctx.insert_resource(ExecutionTrace {
            frame: self.trace_frame,
            events,
            vars,
            locals,
            hit,
        });
    }

    // The named locals every scoped behavior holds for `entity`, tagged by
    // behavior so the observer can filter to the one it shows.
    fn locals_of(&self, entity: Entity) -> Vec<(AssetId, String, crate::ecs::TraceVal)> {
        let mut out = Vec::new();
        for (i, program) in self.programs.iter().enumerate() {
            let Some(instance) = self.instances[i]
                .iter()
                .find(|inst| inst.entity == Some(entity))
            else {
                continue;
            };
            for (decl, value) in program.def.locals.iter().zip(&instance.locals) {
                out.push((program.def.asset_id, decl.name.clone(), value.to_trace()));
            }
        }
        out
    }
}
