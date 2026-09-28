//! Name-addressed runtime animation control: crossfade a flat bucket, write a
//! graph parameter, and report a graph's live state. Each call addresses a
//! mesh by its interned name id and applies against the system's own clip
//! clock, so it can be made between steps from outside the pipeline.

use concinnity_core::animation::anim_graph::normalized_time;
use concinnity_core::ecs::SkinnedMeshHandle;
use concinnity_core::ecs::asset_id::AssetId;

use super::flat::Transition;
use super::graph::GraphTarget;
use super::{AnimationSystem, TargetMode};

/// Snapshot of a graph target's live state. Parameter values are as of the
/// last completed animation step, so a pending [`AnimationSystem::set_param`]
/// write shows up after the next step.
#[derive(Debug, Clone)]
pub struct GraphStateReport {
    /// Name of the state the target is in.
    pub state: String,
    /// The state's clock, in seconds.
    pub clock_secs: f32,
    /// Name of the state being faded out of, when a fade is in flight.
    pub fading_from: Option<String>,
    /// Fade progress in `[0, 1]`, when a fade is in flight.
    pub fade_progress: Option<f32>,
    /// One weight per blendspace member (point / grid order); None when the
    /// active state plays a single clip.
    pub blend_weights: Option<Vec<f32>>,
    /// Every graph parameter with its value, as of the last step.
    pub params: Vec<(String, f32)>,
}

impl AnimationSystem {
    /// Crossfade the flat clip weights of the `SkinnedMesh` named `target`
    /// to `weights` (one per registered clip) over `duration_secs`, starting
    /// at the current clip clock. A zero duration snaps on the next step.
    /// Fails when the target is unknown, graph-driven, or the weight count
    /// misses its clips.
    pub fn crossfade(
        &mut self,
        target: AssetId,
        weights: Vec<f32>,
        duration_secs: f32,
    ) -> Result<(), String> {
        let target = self.name_index.get(target);
        self.apply_crossfade(target, weights, duration_secs, self.clip_secs)
    }

    /// Write the graph parameter `name` on the `SkinnedMesh` named `target`.
    /// The value lands in the target's `AnimationParams` on the next step.
    pub fn set_param(&mut self, target: AssetId, name: &str, value: f32) -> Result<(), String> {
        let target = self.name_index.get(target);
        self.queue_param(target, name, value)
    }

    /// Report the live graph state of the `SkinnedMesh` named `target`.
    pub fn graph_state(&mut self, target: AssetId) -> Result<GraphStateReport, String> {
        let target = self.name_index.get(target);
        self.graph_report(target)
    }

    // Set up a weight ramp on a flat bucket from its current weights to
    // `weights` over `duration_secs`, anchored at `now_secs`.
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

    // Snapshot a graph bucket's live state.
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
    use crate::gfx::skinned_mesh_map::SkinnedMeshNameIndex;
    use concinnity_core::animation::anim_graph::GraphCursor;
    use concinnity_core::animation::skeleton::AnimationClip;
    use concinnity_core::components::AnimationGraph;
    use concinnity_host::thread::asset_id;

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
        asset_id::ensure_name_resolver();
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

    // A crossfade addresses a mesh by its interned name id, translated through
    // the index captured at init.
    #[test]
    fn crossfade_translates_the_name_to_its_bucket() {
        let mut sys = flat_system(2);
        sys.name_index = name_index();
        sys.crossfade(NAME, vec![0.0, 1.0], 0.25).unwrap();
        let tr = transition(&mut sys).expect("the named target's bucket ramped");
        assert_eq!(tr.target_weights, vec![0.0, 1.0]);
        assert_eq!(tr.duration_secs, 0.25);
    }

    // A crossfade starts at the clip clock `step` has reached, so it ramps
    // from the current moment rather than from zero.
    #[test]
    fn crossfade_anchors_at_the_clip_clock() {
        let mut sys = flat_system(2);
        sys.name_index = name_index();
        sys.clip_secs = 7.5;
        sys.crossfade(NAME, vec![0.0, 1.0], 1.0).unwrap();
        assert_eq!(transition(&mut sys).unwrap().start_secs, 7.5);
    }

    // A parameter write and a state query take the same name translation.
    #[test]
    fn set_param_and_graph_state_translate_the_name() {
        let mut sys = graph_system(0.0);
        sys.name_index = name_index();
        sys.set_param(NAME, "speed", 4.0).unwrap();
        assert_eq!(sys.graph_state(NAME).unwrap().state, "idle");
    }

    // A name the index does not know is reported, never silently ignored.
    #[test]
    fn an_unknown_name_is_an_error() {
        let mut sys = AnimationSystem::new();
        assert!(sys.crossfade(NAME, vec![1.0], 0.0).is_err());
        assert!(sys.set_param(NAME, "speed", 1.0).is_err());
        assert!(sys.graph_state(NAME).is_err());
    }
}
