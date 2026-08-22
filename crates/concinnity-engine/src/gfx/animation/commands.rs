// src/gfx/animation/commands.rs
//
// Runtime debug-command drain: `anim-crossfade` (flat buckets), `anim-param`
// and `anim-state` (graph buckets). Commands arrive on the process-wide
// `crate::app::anim_runtime` queue from the debug WS server and each carries
// a reply channel answered synchronously here. The drain is driven from the
// editor's per-frame `DebugHook::tick` (not from `step`) so a WS client
// blocked on a reply is never starved while a menu pauses playback.

use crate::app::anim_runtime::{AnimCommand, GraphStateReport};
use crate::ecs::SkinnedMeshHandle;
use crate::gfx::anim_graph::normalized_time;

use super::flat::Transition;
use super::graph::GraphTarget;
use super::{AnimationSystem, TargetMode};

impl AnimationSystem {
    /// Drain pending runtime commands against the system's own clock. Uses the
    /// same `start` / elapsed bookkeeping `step` uses, so the binary-only
    /// `DebugHook::tick` drive can apply commands from outside the per-system
    /// step. The library never calls this (the drive is in the `cn debug`
    /// binary), hence the `dead_code` allowance. `step` runs after the hook on
    /// the same frame, so the `start` anchor set here is shared.
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
        // Commands address a mesh by its interned NAME id (the WS server
        // resolves the typed name against the interner); the buckets are keyed
        // by handle, so translate through the name index captured at init.
        for cmd in crate::app::anim_runtime::drain() {
            match cmd {
                AnimCommand::Crossfade { req, reply } => {
                    let target = self.name_index.get(req.target);
                    let _ = reply.send(self.apply_crossfade(
                        target,
                        req.weights,
                        req.duration_secs,
                        now_secs,
                    ));
                }
                AnimCommand::SetParam { req, reply } => {
                    let target = self.name_index.get(req.target);
                    let _ = reply.send(self.queue_param(target, &req.name, req.value));
                }
                AnimCommand::QueryState { target, reply } => {
                    let target = self.name_index.get(target);
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
        target: SkinnedMeshHandle,
        weights: Vec<f32>,
        duration_secs: f32,
        now_secs: f32,
    ) -> Result<(), String> {
        let Some(state) = self.targets.get_mut(&target) else {
            return Err(format!(
                "anim-crossfade: no Animation registered for target {target:?}"
            ));
        };
        let TargetMode::Flat(flat) = &mut state.mode else {
            return Err(format!(
                "anim-crossfade: target {target:?} is graph-driven; set a parameter with \
                 anim-param instead"
            ));
        };
        if weights.len() != state.clips.len() {
            return Err(format!(
                "anim-crossfade: weight count {} does not match clip count {} for target {:?}",
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
    // `AnimationParams` component at the top of the next animation step.
    // `pub(super)` for queue-free tests, like `apply_crossfade`.
    pub(super) fn queue_param(
        &mut self,
        target: SkinnedMeshHandle,
        name: &str,
        value: f32,
    ) -> Result<(), String> {
        let g = self.graph_target_mut(&target, "anim-param")?;
        let Some(index) = g.graph.param_index(name) else {
            return Err(format!(
                "anim-param: graph for target {target:?} declares no parameter '{name}'"
            ));
        };
        g.pending.push((index, value));
        Ok(())
    }

    // Snapshot a graph bucket's live state for the `anim-state` command.
    // Parameter values are as of the last completed step. Also serves tests.
    pub(super) fn graph_report(
        &mut self,
        target: SkinnedMeshHandle,
    ) -> Result<GraphStateReport, String> {
        let g = self.graph_target_mut(&target, "anim-state")?;
        let state = &g.graph.states[g.cursor.state];
        let fade = g.cursor.fade.as_ref();
        let weights = state.play.weights(&g.params);
        let effective_duration = state.play.effective_duration(&weights);
        Ok(GraphStateReport {
            state: state.name.clone(),
            clock_secs: normalized_time(state, g.cursor.clock, &g.params) * effective_duration,
            fading_from: fade.map(|f| g.graph.states[f.from_state].name.clone()),
            fade_progress: fade.map(|f| f.progress()),
            // Only meaningful for blendspace states; a single clip is
            // always [1.0], reported as None to keep the JSON quiet.
            blend_weights: (weights.len() > 1).then_some(weights),
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
        target: &SkinnedMeshHandle,
        cmd: &str,
    ) -> Result<&mut GraphTarget, String> {
        let Some(state) = self.targets.get_mut(target) else {
            return Err(format!(
                "{cmd}: no animation registered for target {target:?}"
            ));
        };
        match &mut state.mode {
            TargetMode::Graph(g) => Ok(g),
            TargetMode::Flat(_) => Err(format!(
                "{cmd}: target {target:?} has no AnimationGraph (its clips blend by weight; \
                 use anim-crossfade)"
            )),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::super::TargetState;
    use super::super::flat::{ClipEntry, FlatState};
    use super::*;
    use crate::app::anim_runtime::{CrossfadeRequest, SetParamRequest};
    use crate::assets::AnimationGraph;
    use crate::ecs::asset_id::AssetId;
    use crate::gfx::anim_graph::GraphCursor;
    use crate::gfx::skinned_mesh_map::SkinnedMeshNameIndex;
    use crate::gfx::skinning::AnimationClip;

    const TARGET: SkinnedMeshHandle = SkinnedMeshHandle(1);
    const MISSING: SkinnedMeshHandle = SkinnedMeshHandle(9);
    // The interned mesh name a command addresses, deliberately different from
    // the handle so the index translation is observable.
    const NAME: AssetId = AssetId(77);

    // A bare clip; the command surface never samples one.
    fn clip_entry() -> ClipEntry {
        ClipEntry {
            clip: AnimationClip {
                morph_keys: Vec::new(),
                duration: 1.0,
                looping: true,
                tracks: Vec::new(),
                root: None,
            },
            declared_weight: 1.0,
            fade_in_secs: 0.0,
        }
    }

    // A system holding one flat bucket of `clips` clips, each at full weight.
    fn flat_system(clips: usize) -> AnimationSystem {
        let mut sys = AnimationSystem::new();
        sys.targets.insert(
            TARGET,
            TargetState {
                clips: (0..clips).map(|_| clip_entry()).collect(),
                mode: TargetMode::Flat(FlatState {
                    current_weights: vec![1.0; clips],
                    transition: None,
                }),
            },
        );
        sys
    }

    // An idle/run graph on TARGET crossfading over `fade_secs` when `speed`
    // passes 0.5. Every state resolves onto the bucket's single clip: the
    // command surface reports the machine, it never samples a pose.
    fn graph_system(fade_secs: f32) -> AnimationSystem {
        crate::ecs::asset_id::ensure_name_resolver();
        let g: AnimationGraph = serde_json::from_value(serde_json::json!({
            "parameters": [{"name": "speed", "default": 0.0}],
            "initial": "idle",
            "states": [
                {"name": "idle", "clip": "cmd_idle_clip"},
                {"name": "run", "clip": "cmd_run_clip"}
            ],
            "transitions": [
                {"from": "idle", "to": "run", "duration_secs": fade_secs,
                 "conditions": [{"parameter": "speed", "op": "gt", "value": 0.5}]}
            ]
        }))
        .unwrap();
        let graph = g.compile(|_| Some((0, 1.0, true))).unwrap();
        let params = graph.default_params();
        let mut sys = AnimationSystem::new();
        sys.targets.insert(
            TARGET,
            TargetState {
                clips: vec![clip_entry()],
                mode: TargetMode::Graph(GraphTarget {
                    cursor: GraphCursor::start(&graph),
                    graph,
                    params,
                    pending: Vec::new(),
                    chains: Vec::new(),
                }),
            },
        );
        sys
    }

    fn name_index() -> SkinnedMeshNameIndex {
        SkinnedMeshNameIndex(std::collections::HashMap::from([(NAME, TARGET)]))
    }

    // Reach into a flat bucket's in-flight ramp.
    fn transition(sys: &mut AnimationSystem) -> Option<&Transition> {
        match &sys.targets.get(&TARGET)?.mode {
            TargetMode::Flat(f) => f.transition.as_ref(),
            TargetMode::Graph(_) => None,
        }
    }

    // Drive a graph bucket's cursor directly, so fades are driven by an
    // explicit dt rather than the wall clock.
    fn advance(sys: &mut AnimationSystem, dt: f32) {
        let Some(TargetState {
            mode: TargetMode::Graph(g),
            ..
        }) = sys.targets.get_mut(&TARGET)
        else {
            panic!("graph bucket");
        };
        let params = g.params.clone();
        g.cursor.advance(&g.graph, &params, dt);
    }

    // The command queue is process-wide: `drain` takes everything on it, so the
    // tests that drive it serialise on a shared lock rather than stealing each
    // other's commands. Any leftovers from a panicking earlier test are not ours.
    fn queue_guard() -> std::sync::MutexGuard<'static, ()> {
        let g = crate::app::anim_runtime::TEST_LOCK
            .lock()
            .unwrap_or_else(|e| e.into_inner());
        let _ = crate::app::anim_runtime::drain();
        g
    }

    // A crossfade on a target with no clips registered names the command that
    // failed rather than silently doing nothing.
    #[test]
    fn apply_crossfade_rejects_an_unregistered_target() {
        let mut sys = AnimationSystem::new();
        let err = sys
            .apply_crossfade(MISSING, vec![1.0], 0.0, 0.0)
            .unwrap_err();
        assert!(err.contains("anim-crossfade"), "{err}");
        assert!(err.contains("no Animation registered"), "{err}");
    }

    // A weight vector that does not match the bucket's clip count is refused,
    // and nothing is mutated: a half-applied blend is impossible.
    #[test]
    fn apply_crossfade_rejects_a_weight_count_that_misses_the_clips() {
        let mut sys = flat_system(2);
        let err = sys
            .apply_crossfade(TARGET, vec![1.0], 0.0, 0.0)
            .unwrap_err();
        assert!(err.contains("weight count 1"), "{err}");
        assert!(err.contains("clip count 2"), "{err}");
        assert!(transition(&mut sys).is_none(), "no ramp was installed");
    }

    // An accepted crossfade ramps from the bucket's live weights to the
    // requested ones, anchored at the caller's clock.
    #[test]
    fn apply_crossfade_ramps_from_the_live_weights() {
        let mut sys = flat_system(2);
        sys.apply_crossfade(TARGET, vec![0.0, 1.0], 0.5, 3.0)
            .unwrap();
        let tr = transition(&mut sys).expect("ramp installed");
        assert_eq!(tr.source_weights, vec![1.0, 1.0]);
        assert_eq!(tr.target_weights, vec![0.0, 1.0]);
        assert_eq!(tr.start_secs, 3.0);
        assert_eq!(tr.duration_secs, 0.5);
    }

    // A negative duration clamps to zero (an immediate snap) rather than
    // producing a ramp that never finishes.
    #[test]
    fn apply_crossfade_clamps_a_negative_duration_to_a_snap() {
        let mut sys = flat_system(1);
        sys.apply_crossfade(TARGET, vec![0.5], -1.0, 0.0).unwrap();
        assert_eq!(transition(&mut sys).unwrap().duration_secs, 0.0);
    }

    // A later crossfade for the same target supersedes the one in flight.
    #[test]
    fn a_second_crossfade_supersedes_the_ramp_in_flight() {
        let mut sys = flat_system(1);
        sys.apply_crossfade(TARGET, vec![0.0], 1.0, 0.0).unwrap();
        sys.apply_crossfade(TARGET, vec![0.25], 2.0, 4.0).unwrap();
        let tr = transition(&mut sys).unwrap();
        assert_eq!(tr.target_weights, vec![0.25]);
        assert_eq!(tr.start_secs, 4.0);
    }

    // Both graph commands report an unregistered target by name of the command
    // that asked, so a typo'd mesh is distinguishable from a mode mismatch.
    #[test]
    fn graph_commands_reject_an_unregistered_target() {
        let mut sys = AnimationSystem::new();
        let err = sys.queue_param(MISSING, "speed", 1.0).unwrap_err();
        assert!(err.contains("anim-param"), "{err}");
        assert!(err.contains("no animation registered"), "{err}");
        let err = sys.graph_report(MISSING).unwrap_err();
        assert!(err.contains("anim-state"), "{err}");
        assert!(err.contains("no animation registered"), "{err}");
    }

    // A parameter the graph does not declare is refused and queues nothing.
    #[test]
    fn queue_param_rejects_a_parameter_the_graph_does_not_declare() {
        let mut sys = graph_system(0.0);
        let err = sys.queue_param(TARGET, "nope", 1.0).unwrap_err();
        assert!(err.contains("declares no parameter 'nope'"), "{err}");
        let report = sys.graph_report(TARGET).unwrap();
        assert_eq!(report.params, vec![("speed".to_string(), 0.0)]);
    }

    // A queued write is held against the declared parameter's index until the
    // next step flushes it into the component.
    #[test]
    fn queue_param_holds_the_write_against_the_parameter_index() {
        let mut sys = graph_system(0.0);
        sys.queue_param(TARGET, "speed", 2.5).unwrap();
        let Some(TargetState {
            mode: TargetMode::Graph(g),
            ..
        }) = sys.targets.get(&TARGET)
        else {
            panic!("graph bucket");
        };
        assert_eq!(g.pending, vec![(0, 2.5)]);
    }

    // A parked graph reports its state and clock with no fade in flight.
    #[test]
    fn graph_report_of_a_parked_graph_carries_no_fade() {
        let mut sys = graph_system(0.5);
        let report = sys.graph_report(TARGET).unwrap();
        assert_eq!(report.state, "idle");
        assert_eq!(report.clock_secs, 0.0);
        assert!(report.fading_from.is_none());
        assert!(report.fade_progress.is_none());
        assert!(
            report.blend_weights.is_none(),
            "a single-clip state reports no blend weights"
        );
    }

    // Mid-transition the report names the outgoing state and how far the
    // crossfade has run.
    #[test]
    fn graph_report_carries_the_fade_while_a_transition_is_in_flight() {
        let mut sys = graph_system(0.5);
        sys.queue_param(TARGET, "speed", 2.0).unwrap();
        // The pending write only lands on a step, so seed the snapshot the
        // cursor reads directly.
        if let Some(TargetState {
            mode: TargetMode::Graph(g),
            ..
        }) = sys.targets.get_mut(&TARGET)
        {
            g.params = vec![2.0];
        }
        // One advance takes the transition and installs the fade at zero; the
        // next runs it a fifth of the way through.
        advance(&mut sys, 0.1);
        advance(&mut sys, 0.1);

        let report = sys.graph_report(TARGET).unwrap();
        assert_eq!(report.state, "run");
        assert_eq!(report.fading_from.as_deref(), Some("idle"));
        let progress = report.fade_progress.unwrap();
        assert!((progress - 0.2).abs() < 1e-4, "{progress}");
        // The clock reports seconds into the incoming state, not the fade.
        assert!((report.clock_secs - 0.1).abs() < 1e-4, "{report:?}");
    }

    // A fade that has run its length is dropped, so the report goes quiet again.
    #[test]
    fn graph_report_drops_the_fade_once_it_completes() {
        let mut sys = graph_system(0.5);
        if let Some(TargetState {
            mode: TargetMode::Graph(g),
            ..
        }) = sys.targets.get_mut(&TARGET)
        {
            g.params = vec![2.0];
        }
        advance(&mut sys, 0.1);
        advance(&mut sys, 0.6);
        let report = sys.graph_report(TARGET).unwrap();
        assert_eq!(report.state, "run");
        assert!(report.fading_from.is_none());
        assert!(report.fade_progress.is_none());
    }

    // Commands address a mesh by its interned NAME id; the drain translates it
    // through the index captured at init and applies the crossfade against the
    // clock it was handed, answering the caller's reply channel.
    #[test]
    fn drain_applies_a_crossfade_addressed_by_name() {
        let _guard = queue_guard();
        let mut sys = flat_system(2);
        sys.name_index = name_index();
        let (tx, rx) = std::sync::mpsc::sync_channel(1);
        crate::app::anim_runtime::enqueue(AnimCommand::Crossfade {
            req: CrossfadeRequest {
                target: NAME,
                weights: vec![0.0, 1.0],
                duration_secs: 0.25,
            },
            reply: tx,
        });
        sys.drain_runtime_commands(2.0);

        assert_eq!(rx.try_recv().unwrap(), Ok(()));
        let tr = transition(&mut sys).expect("the named target's bucket ramped");
        assert_eq!(tr.target_weights, vec![0.0, 1.0]);
        assert_eq!(tr.start_secs, 2.0, "the drain's clock anchors the ramp");
    }

    // A parameter write and a state query take the same name translation, and
    // each reply is answered synchronously by the drain.
    #[test]
    fn drain_answers_param_writes_and_state_queries() {
        let _guard = queue_guard();
        let mut sys = graph_system(0.0);
        sys.name_index = name_index();
        let (param_tx, param_rx) = std::sync::mpsc::sync_channel(1);
        crate::app::anim_runtime::enqueue(AnimCommand::SetParam {
            req: SetParamRequest {
                target: NAME,
                name: "speed".to_string(),
                value: 4.0,
            },
            reply: param_tx,
        });
        let (query_tx, query_rx) = std::sync::mpsc::sync_channel(1);
        crate::app::anim_runtime::enqueue(AnimCommand::QueryState {
            target: NAME,
            reply: query_tx,
        });
        sys.drain_runtime_commands(0.0);

        assert_eq!(param_rx.try_recv().unwrap(), Ok(()));
        assert_eq!(query_rx.try_recv().unwrap().unwrap().state, "idle");
    }

    // A command for a mesh the index does not know still gets its reply: the
    // failure is reported, never dropped on the floor.
    #[test]
    fn drain_replies_to_a_command_it_cannot_apply() {
        let _guard = queue_guard();
        let mut sys = AnimationSystem::new();
        let (tx, rx) = std::sync::mpsc::sync_channel(1);
        crate::app::anim_runtime::enqueue(AnimCommand::QueryState {
            target: NAME,
            reply: tx,
        });
        sys.drain_runtime_commands(0.0);
        assert!(rx.try_recv().unwrap().is_err());
    }

    // The hook drive anchors the system's clock on its first call and answers
    // whatever is queued, so a WS client blocked on a reply is never starved by
    // a paused world.
    #[test]
    fn apply_runtime_commands_anchors_the_clock_and_answers() {
        let _guard = queue_guard();
        let mut sys = graph_system(0.0);
        sys.name_index = name_index();
        let (tx, rx) = std::sync::mpsc::sync_channel(1);
        crate::app::anim_runtime::enqueue(AnimCommand::QueryState {
            target: NAME,
            reply: tx,
        });
        sys.apply_runtime_commands();
        assert_eq!(rx.try_recv().unwrap().unwrap().state, "idle");
        assert!(sys.start.is_some(), "the drive shares `step`'s origin");
    }

    // An empty queue is a no-op the drive can call every frame.
    #[test]
    fn draining_an_empty_queue_changes_nothing() {
        let _guard = queue_guard();
        let mut sys = flat_system(1);
        sys.drain_runtime_commands(1.0);
        assert!(transition(&mut sys).is_none());
    }
}
