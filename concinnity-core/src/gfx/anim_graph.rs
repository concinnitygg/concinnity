// src/gfx/anim_graph.rs
//
// Animation state-machine runtime: the compiled graph representation plus the
// cursor that walks it. Pure math/state code with no asset or ECS
// dependencies -- the `AnimGraph` asset compiles into a `CompiledGraph` (see
// `assets/anim_graph.rs`), and the client's AnimationSystem owns one
// `GraphCursor` per graph target, advancing it each frame and sampling the
// blended pose through `sample_graph_pose`.

use crate::gfx::skinning::{self, AnimationClip, Mat4, Skeleton};

/// Comparison operator for a transition condition, evaluated as
/// `parameter <op> value`. `eq` / `ne` compare exactly and are mainly useful
/// for flag-like parameters holding whole numbers such as 0 and 1.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum CmpOp {
    /// Less than.
    #[default]
    Lt,
    /// Less than or equal.
    Le,
    /// Greater than.
    Gt,
    /// Greater than or equal.
    Ge,
    /// Equal (exact).
    Eq,
    /// Not equal (exact).
    Ne,
}

impl CmpOp {
    // Evaluate `lhs <op> rhs`.
    pub fn eval(self, lhs: f32, rhs: f32) -> bool {
        match self {
            CmpOp::Lt => lhs < rhs,
            CmpOp::Le => lhs <= rhs,
            CmpOp::Gt => lhs > rhs,
            CmpOp::Ge => lhs >= rhs,
            CmpOp::Eq => lhs == rhs,
            CmpOp::Ne => lhs != rhs,
        }
    }
}

// One compiled transition condition: parameter names are resolved to indices
// into the graph's parameter vector at compile time, so evaluation is a
// direct slice read (the runtime blob interner is empty; nothing resolves
// names at runtime).
#[derive(Debug, Clone)]
pub struct CompiledCondition {
    pub param: usize,
    pub op: CmpOp,
    pub value: f32,
}

// One compiled outgoing transition. Conditions AND together; an empty list is
// always true (useful with `exit_time` alone).
#[derive(Debug, Clone)]
pub struct CompiledTransition {
    pub to: usize,
    // Crossfade length between the outgoing and incoming state poses. Zero
    // snaps.
    pub duration_secs: f32,
    // When set, the transition is gated until the state's normalized time
    // reaches this value (see `normalized_time`).
    pub exit_time: Option<f32>,
    pub conditions: Vec<CompiledCondition>,
}

// One compiled state: a single clip played at `rate`, with outgoing
// transitions in declaration order (first match wins).
#[derive(Debug, Clone)]
pub struct CompiledState {
    pub name: String,
    // Index of the clip in the target's clip list.
    pub clip: usize,
    // Clock scale; 1.0 plays the clip at its authored speed.
    pub rate: f32,
    // Whether this state's clock wraps at the clip duration. Defaults to the
    // clip's own flag; the state may override it.
    pub looping: bool,
    // The clip's duration, copied at compile time for wrap / exit-time math.
    pub duration_secs: f32,
    pub transitions: Vec<CompiledTransition>,
}

/// A graph parameter: a named float, seeded to `default`.
#[derive(Debug, Clone)]
pub struct ParamSpec {
    pub name: String,
    pub default: f32,
}

// A compiled animation state machine. Built once from the `AnimGraph` asset;
// read-only afterwards.
#[derive(Debug, Clone)]
pub struct CompiledGraph {
    pub params: Vec<ParamSpec>,
    pub states: Vec<CompiledState>,
    pub initial: usize,
}

impl CompiledGraph {
    // Index of a parameter by name, for surfaces (debug commands) that still
    // speak names.
    pub fn param_index(&self, name: &str) -> Option<usize> {
        self.params.iter().position(|p| p.name == name)
    }

    // The parameter vector seeded to each parameter's declared default.
    pub fn default_params(&self) -> Vec<f32> {
        self.params.iter().map(|p| p.default).collect()
    }
}

// An in-flight crossfade from the previous state. The outgoing state's clock
// keeps advancing during the fade so its pose stays live rather than frozen.
#[derive(Debug, Clone)]
pub struct StateFade {
    pub from_state: usize,
    pub from_clock_secs: f32,
    pub elapsed_secs: f32,
    pub duration_secs: f32,
}

impl StateFade {
    // Fade progress in [0, 1]: 0 = all outgoing pose, 1 = all incoming.
    pub fn progress(&self) -> f32 {
        (self.elapsed_secs / self.duration_secs.max(1e-6)).clamp(0.0, 1.0)
    }
}

// The live position in a graph: current state, its clock, and any in-flight
// crossfade. One per graph target, owned by the client's AnimationSystem.
#[derive(Debug, Clone)]
pub struct GraphCursor {
    pub state: usize,
    // Seconds into the current state, already scaled by the state's `rate`.
    pub clock_secs: f32,
    pub fade: Option<StateFade>,
}

impl GraphCursor {
    // A cursor parked at the graph's initial state.
    pub fn start(graph: &CompiledGraph) -> Self {
        Self {
            state: graph.initial,
            clock_secs: 0.0,
            fade: None,
        }
    }

    // Advance clocks by `dt_secs` and take at most one transition. Transition
    // checks run against the current state's outgoing list in declaration
    // order; the first whose exit-time gate and conditions all pass wins.
    // Taking a transition while a fade is in flight replaces the fade: the
    // new fade blends from the interrupted fade's *incoming* state only, so a
    // rapid double transition can pop the older outgoing pose.
    pub fn advance(&mut self, graph: &CompiledGraph, params: &[f32], dt_secs: f32) {
        let dt = dt_secs.max(0.0);
        let state = &graph.states[self.state];
        self.clock_secs += dt * state.rate;
        if let Some(fade) = self.fade.as_mut() {
            fade.elapsed_secs += dt;
            fade.from_clock_secs += dt * graph.states[fade.from_state].rate;
            if fade.elapsed_secs >= fade.duration_secs {
                self.fade = None;
            }
        }

        let normalized = normalized_time(state, self.clock_secs);
        for tr in &state.transitions {
            if let Some(gate) = tr.exit_time
                && normalized < gate
            {
                continue;
            }
            let hold = tr.conditions.iter().any(|c| {
                let lhs = params.get(c.param).copied().unwrap_or(0.0);
                !c.op.eval(lhs, c.value)
            });
            if hold {
                continue;
            }
            self.fade = (tr.duration_secs > 0.0).then_some(StateFade {
                from_state: self.state,
                from_clock_secs: self.clock_secs,
                elapsed_secs: 0.0,
                duration_secs: tr.duration_secs,
            });
            self.state = tr.to;
            self.clock_secs = 0.0;
            break;
        }
    }
}

// A state's normalized time in [0, 1]: the fraction of the clip covered by
// the clock. Looping states report the fraction within the current pass, so
// an `exit_time` gate re-opens every loop; non-looping states saturate at 1.
pub fn normalized_time(state: &CompiledState, clock_secs: f32) -> f32 {
    if state.duration_secs <= 1e-6 {
        return 1.0;
    }
    let phase = clock_secs / state.duration_secs;
    if state.looping {
        phase.fract()
    } else {
        phase.clamp(0.0, 1.0)
    }
}

// The clip-local sample time for a state clock, honoring the *state's* loop
// mode (which may override the clip's own flag).
pub fn state_local_time(state: &CompiledState, clock_secs: f32) -> f32 {
    if state.duration_secs <= 1e-6 {
        return 0.0;
    }
    if state.looping {
        clock_secs.rem_euclid(state.duration_secs)
    } else {
        clock_secs.clamp(0.0, state.duration_secs)
    }
}

// Sample the cursor's blended pose: the current state's clip, crossfaded with
// the outgoing state's clip while a fade is in flight. `clip_at` maps a
// compiled state's clip index to the actual clip (the caller owns clip
// storage).
pub fn sample_graph_pose<'a>(
    graph: &CompiledGraph,
    cursor: &GraphCursor,
    clip_at: impl Fn(usize) -> &'a AnimationClip,
    skeleton: &Skeleton,
) -> Vec<Mat4> {
    let cur = &graph.states[cursor.state];
    let cur_locals = clip_at(cur.clip).sample_looped(
        state_local_time(cur, cursor.clock_secs),
        cur.looping,
        skeleton,
    );
    let Some(fade) = &cursor.fade else {
        return cur_locals;
    };
    let from = &graph.states[fade.from_state];
    let from_locals = clip_at(from.clip).sample_looped(
        state_local_time(from, fade.from_clock_secs),
        from.looping,
        skeleton,
    );
    skinning::blend_locals(&from_locals, &cur_locals, fade.progress())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::gfx::skinning::{Joint, JointPose, JointTrack, Keyframe};

    fn state(name: &str, clip: usize, transitions: Vec<CompiledTransition>) -> CompiledState {
        CompiledState {
            name: name.to_string(),
            clip,
            rate: 1.0,
            looping: true,
            duration_secs: 1.0,
            transitions,
        }
    }

    fn transition(to: usize, conditions: Vec<CompiledCondition>) -> CompiledTransition {
        CompiledTransition {
            to,
            duration_secs: 0.0,
            exit_time: None,
            conditions,
        }
    }

    fn cond(param: usize, op: CmpOp, value: f32) -> CompiledCondition {
        CompiledCondition { param, op, value }
    }

    fn two_state_graph(conditions: Vec<CompiledCondition>) -> CompiledGraph {
        CompiledGraph {
            params: vec![ParamSpec {
                name: "speed".into(),
                default: 0.0,
            }],
            states: vec![
                state("idle", 0, vec![transition(1, conditions)]),
                state("run", 1, Vec::new()),
            ],
            initial: 0,
        }
    }

    #[test]
    fn cmp_op_eval_covers_all_operators() {
        assert!(CmpOp::Lt.eval(1.0, 2.0));
        assert!(!CmpOp::Lt.eval(2.0, 2.0));
        assert!(CmpOp::Le.eval(2.0, 2.0));
        assert!(CmpOp::Gt.eval(3.0, 2.0));
        assert!(!CmpOp::Gt.eval(2.0, 2.0));
        assert!(CmpOp::Ge.eval(2.0, 2.0));
        assert!(CmpOp::Eq.eval(1.0, 1.0));
        assert!(CmpOp::Ne.eval(1.0, 0.0));
    }

    #[test]
    fn cursor_holds_state_while_conditions_fail() {
        let graph = two_state_graph(vec![cond(0, CmpOp::Gt, 0.5)]);
        let mut cursor = GraphCursor::start(&graph);
        cursor.advance(&graph, &[0.0], 0.1);
        assert_eq!(cursor.state, 0);
        assert!((cursor.clock_secs - 0.1).abs() < 1e-6);
    }

    #[test]
    fn cursor_transitions_when_conditions_pass() {
        let graph = two_state_graph(vec![cond(0, CmpOp::Gt, 0.5)]);
        let mut cursor = GraphCursor::start(&graph);
        cursor.advance(&graph, &[1.0], 0.1);
        assert_eq!(cursor.state, 1);
        assert_eq!(cursor.clock_secs, 0.0);
        // Snap transition (duration 0): no fade.
        assert!(cursor.fade.is_none());
    }

    #[test]
    fn multiple_conditions_and_together() {
        let graph = two_state_graph(vec![cond(0, CmpOp::Gt, 0.5), cond(0, CmpOp::Lt, 0.7)]);
        let mut cursor = GraphCursor::start(&graph);
        cursor.advance(&graph, &[0.9], 0.1);
        assert_eq!(cursor.state, 0, "second condition fails, must hold");
        cursor.advance(&graph, &[0.6], 0.1);
        assert_eq!(cursor.state, 1, "both conditions pass");
    }

    #[test]
    fn first_declared_transition_wins() {
        let mut graph = two_state_graph(Vec::new());
        graph.states.push(state("walk", 2, Vec::new()));
        graph.states[0].transitions = vec![transition(2, Vec::new()), transition(1, Vec::new())];
        let mut cursor = GraphCursor::start(&graph);
        cursor.advance(&graph, &[0.0], 0.1);
        assert_eq!(cursor.state, 2);
    }

    #[test]
    fn exit_time_gates_until_reached_non_looping() {
        let mut graph = two_state_graph(Vec::new());
        graph.states[0].looping = false;
        graph.states[0].transitions[0].exit_time = Some(0.9);
        let mut cursor = GraphCursor::start(&graph);
        cursor.advance(&graph, &[0.0], 0.5);
        assert_eq!(cursor.state, 0, "0.5 of 1.0s clip is before the 0.9 gate");
        cursor.advance(&graph, &[0.0], 0.5);
        assert_eq!(cursor.state, 1, "clock passed the gate");
    }

    #[test]
    fn exit_time_reopens_each_loop_for_looping_state() {
        let mut graph = two_state_graph(vec![cond(0, CmpOp::Gt, 0.5)]);
        graph.states[0].transitions[0].exit_time = Some(0.5);
        let mut cursor = GraphCursor::start(&graph);
        // Past the gate within the first loop, but the condition holds it.
        cursor.advance(&graph, &[0.0], 0.75);
        assert_eq!(cursor.state, 0);
        // Wrapped to phase 0.25 of the second loop: gate is closed again even
        // though the condition now passes.
        cursor.advance(&graph, &[1.0], 0.5);
        assert_eq!(cursor.state, 0);
        // Phase 0.75 of the second loop: gate open + condition passes.
        cursor.advance(&graph, &[1.0], 0.5);
        assert_eq!(cursor.state, 1);
    }

    #[test]
    fn rate_scales_the_state_clock() {
        let mut graph = two_state_graph(Vec::new());
        graph.states[0].rate = 2.0;
        graph.states[0].transitions.clear();
        let mut cursor = GraphCursor::start(&graph);
        cursor.advance(&graph, &[0.0], 0.25);
        assert!((cursor.clock_secs - 0.5).abs() < 1e-6);
    }

    #[test]
    fn fade_runs_its_duration_then_clears() {
        let mut graph = two_state_graph(Vec::new());
        graph.states[0].transitions[0].duration_secs = 0.2;
        let mut cursor = GraphCursor::start(&graph);
        cursor.advance(&graph, &[0.0], 0.1);
        assert_eq!(cursor.state, 1);
        let fade = cursor.fade.as_ref().expect("fade in flight");
        assert_eq!(fade.from_state, 0);
        assert!((fade.progress()).abs() < 1e-6);
        cursor.advance(&graph, &[0.0], 0.1);
        let fade = cursor.fade.as_ref().expect("fade still in flight");
        assert!((fade.progress() - 0.5).abs() < 1e-6);
        cursor.advance(&graph, &[0.0], 0.1);
        assert!(cursor.fade.is_none(), "fade complete");
    }

    #[test]
    fn interrupting_transition_replaces_fade_with_new_source() {
        let mut graph = two_state_graph(Vec::new());
        graph.states[0].transitions[0].duration_secs = 1.0;
        graph.states.push(state("walk", 2, Vec::new()));
        graph.states[1].transitions = vec![CompiledTransition {
            to: 2,
            duration_secs: 1.0,
            exit_time: None,
            conditions: Vec::new(),
        }];
        let mut cursor = GraphCursor::start(&graph);
        cursor.advance(&graph, &[0.0], 0.1);
        assert_eq!(cursor.state, 1);
        cursor.advance(&graph, &[0.0], 0.1);
        assert_eq!(cursor.state, 2);
        let fade = cursor.fade.as_ref().expect("second fade in flight");
        assert_eq!(
            fade.from_state, 1,
            "fade restarts from the interrupted target"
        );
        assert!((fade.elapsed_secs).abs() < 1e-6);
    }

    #[test]
    fn normalized_time_saturates_non_looping_and_wraps_looping() {
        let mut s = state("s", 0, Vec::new());
        s.duration_secs = 2.0;
        s.looping = false;
        assert!((normalized_time(&s, 5.0) - 1.0).abs() < 1e-6);
        s.looping = true;
        assert!((normalized_time(&s, 5.0) - 0.5).abs() < 1e-6);
    }

    #[test]
    fn state_local_time_honors_loop_override() {
        let mut s = state("s", 0, Vec::new());
        s.duration_secs = 2.0;
        s.looping = false;
        assert!(
            (state_local_time(&s, 5.0) - 2.0).abs() < 1e-6,
            "clamps to the end"
        );
        s.looping = true;
        assert!((state_local_time(&s, 5.0) - 1.0).abs() < 1e-6, "wraps");
    }

    // One-joint skeleton plus two constant-pose clips at distinct
    // translations, so blended output is directly readable off the matrix
    // translation column.
    fn pose_fixture() -> (Skeleton, AnimationClip, AnimationClip) {
        let skeleton = Skeleton::new(vec![Joint {
            parent: None,
            bind: JointPose::default(),
        }]);
        let clip_at = |x: f32| AnimationClip {
            duration: 1.0,
            looping: true,
            tracks: vec![JointTrack {
                joint: 0,
                keys: vec![Keyframe {
                    time: 0.0,
                    pose: JointPose {
                        translation: [x, 0.0, 0.0],
                        ..Default::default()
                    },
                }],
            }],
        };
        (skeleton, clip_at(0.0), clip_at(2.0))
    }

    #[test]
    fn sample_blends_outgoing_and_incoming_during_fade() {
        let (skeleton, clip_a, clip_b) = pose_fixture();
        let clips = [clip_a, clip_b];
        let graph = CompiledGraph {
            params: Vec::new(),
            states: vec![
                state(
                    "a",
                    0,
                    vec![CompiledTransition {
                        to: 1,
                        duration_secs: 1.0,
                        exit_time: None,
                        conditions: Vec::new(),
                    }],
                ),
                state("b", 1, Vec::new()),
            ],
            initial: 0,
        };
        let mut cursor = GraphCursor::start(&graph);
        cursor.advance(&graph, &[], 0.0);
        // Half-way through the 1s fade from clip at x=0 to clip at x=2.
        cursor.advance(&graph, &[], 0.5);
        let locals = sample_graph_pose(&graph, &cursor, |i| &clips[i], &skeleton);
        assert!((locals[0][3][0] - 1.0).abs() < 1e-4, "midpoint of 0 and 2");
        // Fade over: pure incoming pose.
        cursor.advance(&graph, &[], 0.6);
        let locals = sample_graph_pose(&graph, &cursor, |i| &clips[i], &skeleton);
        assert!((locals[0][3][0] - 2.0).abs() < 1e-4);
    }

    #[test]
    fn sample_without_fade_is_pure_current_state() {
        let (skeleton, clip_a, clip_b) = pose_fixture();
        let clips = [clip_a, clip_b];
        let graph = CompiledGraph {
            params: Vec::new(),
            states: vec![state("a", 0, Vec::new()), state("b", 1, Vec::new())],
            initial: 1,
        };
        let cursor = GraphCursor::start(&graph);
        let locals = sample_graph_pose(&graph, &cursor, |i| &clips[i], &skeleton);
        assert!((locals[0][3][0] - 2.0).abs() < 1e-4);
    }
}
