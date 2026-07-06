// src/gfx/animation/commands.rs
//
// Runtime debug-command drain: `anim-crossfade` (flat buckets), `anim-param`
// and `anim-state` (graph buckets). Commands arrive on the process-wide
// `crate::app::anim_runtime` queue from the debug WS server and each carries
// a reply channel answered synchronously here. The drain is driven from the
// editor's per-frame `DebugHook::tick` (not from `step`) so a WS client
// blocked on a reply is never starved while a menu pauses playback.

use crate::app::anim_runtime::{AnimCommand, GraphStateReport};
use crate::ecs::asset_id::AssetId;
use crate::gfx::anim_graph::normalized_time;

use super::flat::Transition;
use super::graph::GraphTarget;
use super::{AnimationSystem, TargetMode};

impl AnimationSystem {
    // Drain pending runtime commands against the system's own clock. Uses the
    // same `start` / elapsed bookkeeping `step` uses, so the binary-only
    // `DebugHook::tick` drive can apply commands from outside the per-system
    // step. The library never calls this (the drive is in the `cn debug`
    // binary), hence the `dead_code` allowance. `step` runs after the hook on
    // the same frame, so the `start` anchor set here is shared.
    #[allow(dead_code)]
    pub fn apply_runtime_commands(&mut self) {
        let now = std::time::Instant::now();
        let start = *self.start.get_or_insert(now);
        let t = (now - start).as_secs_f32();
        self.drain_runtime_commands(t);
    }

    // Drain pending runtime commands and apply them. Commands run in queue
    // order, so a later command for the same target supersedes an earlier
    // one; a command that does not fit its target's mode fails without
    // touching anything.
    fn drain_runtime_commands(&mut self, now_secs: f32) {
        for cmd in crate::app::anim_runtime::drain() {
            match cmd {
                AnimCommand::Crossfade { req, reply } => {
                    let _ = reply.send(self.apply_crossfade(
                        req.target,
                        req.weights,
                        req.duration_secs,
                        now_secs,
                    ));
                }
                AnimCommand::SetParam { req, reply } => {
                    let _ = reply.send(self.queue_param(req.target, &req.name, req.value));
                }
                AnimCommand::QueryState { target, reply } => {
                    let _ = reply.send(self.graph_report(target));
                }
            }
        }
    }

    // Set up a weight ramp on a flat bucket from its current weights to
    // `weights` over `duration_secs`. `pub(super)` so tests can drive it
    // without the process-wide command queue.
    pub(super) fn apply_crossfade(
        &mut self,
        target: AssetId,
        weights: Vec<f32>,
        duration_secs: f32,
        now_secs: f32,
    ) -> Result<(), String> {
        let Some(state) = self.targets.get_mut(&target) else {
            return Err(format!(
                "anim-crossfade: no Animation registered for target {target}"
            ));
        };
        let TargetMode::Flat(flat) = &mut state.mode else {
            return Err(format!(
                "anim-crossfade: target {target} is graph-driven; set a parameter with \
                 anim-param instead"
            ));
        };
        if weights.len() != state.clips.len() {
            return Err(format!(
                "anim-crossfade: weight count {} does not match clip count {} for target {}",
                weights.len(),
                state.clips.len(),
                target,
            ));
        }
        flat.transition = Some(Transition {
            source_weights: flat.current_weights.clone(),
            target_weights: weights,
            start_secs: now_secs,
            duration_secs: duration_secs.max(0.0),
        });
        Ok(())
    }

    // Queue a parameter write on a graph bucket; it lands in the target's
    // `AnimParams` component at the top of the next animation step.
    // `pub(super)` for queue-free tests, like `apply_crossfade`.
    pub(super) fn queue_param(
        &mut self,
        target: AssetId,
        name: &str,
        value: f32,
    ) -> Result<(), String> {
        let g = self.graph_target_mut(&target, "anim-param")?;
        let Some(index) = g.graph.param_index(name) else {
            return Err(format!(
                "anim-param: graph for target {target} declares no parameter '{name}'"
            ));
        };
        g.pending.push((index, value));
        Ok(())
    }

    // Snapshot a graph bucket's live state for the `anim-state` command.
    // Parameter values are as of the last completed step. Also serves tests.
    pub(super) fn graph_report(&mut self, target: AssetId) -> Result<GraphStateReport, String> {
        let g = self.graph_target_mut(&target, "anim-state")?;
        let state = &g.graph.states[g.cursor.state];
        let fade = g.cursor.fade.as_ref();
        Ok(GraphStateReport {
            state: state.name.clone(),
            clock_secs: normalized_time(state, g.cursor.clock_secs) * state.duration_secs,
            fading_from: fade.map(|f| g.graph.states[f.from_state].name.clone()),
            fade_progress: fade.map(|f| f.progress()),
            params: g
                .graph
                .params
                .iter()
                .zip(&g.params)
                .map(|(spec, &value)| (spec.name.clone(), value))
                .collect(),
        })
    }

    fn graph_target_mut(
        &mut self,
        target: &AssetId,
        cmd: &str,
    ) -> Result<&mut GraphTarget, String> {
        let Some(state) = self.targets.get_mut(target) else {
            return Err(format!(
                "{cmd}: no animation registered for target {target}"
            ));
        };
        match &mut state.mode {
            TargetMode::Graph(g) => Ok(g),
            TargetMode::Flat(_) => Err(format!(
                "{cmd}: target {target} has no AnimGraph (its clips blend by weight; \
                 use anim-crossfade)"
            )),
        }
    }
}
