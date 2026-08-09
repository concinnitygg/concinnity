// src/gfx/anim_graph/tests.rs

use super::*;
use crate::gfx::skinning::{
    AnimationClip, Joint, JointPose, JointTrack, Keyframe, Mat4, PoseScratch, Skeleton,
};

// Sample through a throwaway scratch and hand the pose back by value, so
// assertions read like the old Vec-returning API.
fn sample_pose(
    graph: &CompiledGraph,
    cursor: &GraphCursor,
    params: &[f32],
    clips: &[AnimationClip],
    skeleton: &Skeleton,
) -> Vec<Mat4> {
    let mut scratch = PoseScratch::default();
    sample_graph_pose_into(graph, cursor, params, |i| &clips[i], skeleton, &mut scratch);
    scratch.locals
}

fn clip_play(clip: usize, duration_secs: f32) -> ClipPlay {
    ClipPlay {
        clip,
        duration_secs,
    }
}

fn state(name: &str, clip: usize, transitions: Vec<CompiledTransition>) -> CompiledState {
    CompiledState {
        name: name.to_string(),
        rate: 1.0,
        looping: true,
        play: StatePlay::Clip(clip_play(clip, 1.0)),
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
    assert!((cursor.clock - 0.1).abs() < 1e-6);
}

#[test]
fn cursor_transitions_when_conditions_pass() {
    let graph = two_state_graph(vec![cond(0, CmpOp::Gt, 0.5)]);
    let mut cursor = GraphCursor::start(&graph);
    cursor.advance(&graph, &[1.0], 0.1);
    assert_eq!(cursor.state, 1);
    assert_eq!(cursor.clock, 0.0);
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
    assert!((cursor.clock - 0.5).abs() < 1e-6);
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
    s.play = StatePlay::Clip(clip_play(0, 2.0));
    s.looping = false;
    assert!((normalized_time(&s, 5.0, &[]) - 1.0).abs() < 1e-6);
    s.looping = true;
    assert!((normalized_time(&s, 5.0, &[]) - 0.5).abs() < 1e-6);
}

// One-joint skeleton plus constant-pose clips at distinct X translations, so
// blended output is directly readable off the matrix translation column.
fn pose_fixture(xs: &[f32]) -> (Skeleton, Vec<AnimationClip>) {
    let skeleton = Skeleton::new(vec![Joint {
        name: String::new(),
        parent: None,
        bind: JointPose::default(),
    }]);
    let clips = xs
        .iter()
        .map(|&x| AnimationClip {
            root: None,
            morph_keys: Vec::new(),
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
        })
        .collect();
    (skeleton, clips)
}

#[test]
fn sample_blends_outgoing_and_incoming_during_fade() {
    let (skeleton, clips) = pose_fixture(&[0.0, 2.0]);
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
    let locals = sample_pose(&graph, &cursor, &[], &clips, &skeleton);
    assert!((locals[0][3][0] - 1.0).abs() < 1e-4, "midpoint of 0 and 2");
    // Fade over: pure incoming pose.
    cursor.advance(&graph, &[], 0.6);
    let locals = sample_pose(&graph, &cursor, &[], &clips, &skeleton);
    assert!((locals[0][3][0] - 2.0).abs() < 1e-4);
}

#[test]
fn sample_without_fade_is_pure_current_state() {
    let (skeleton, clips) = pose_fixture(&[0.0, 2.0]);
    let graph = CompiledGraph {
        params: Vec::new(),
        states: vec![state("a", 0, Vec::new()), state("b", 1, Vec::new())],
        initial: 1,
    };
    let cursor = GraphCursor::start(&graph);
    let locals = sample_pose(&graph, &cursor, &[], &clips, &skeleton);
    assert!((locals[0][3][0] - 2.0).abs() < 1e-4);
}

#[test]
fn blend1d_weights_bracket_clamp_and_hit_points_exactly() {
    let thresholds = [0.0, 1.6, 5.0];
    assert_eq!(blend1d_weights(&thresholds, -1.0), vec![1.0, 0.0, 0.0]);
    assert_eq!(blend1d_weights(&thresholds, 0.0), vec![1.0, 0.0, 0.0]);
    assert_eq!(blend1d_weights(&thresholds, 9.0), vec![0.0, 0.0, 1.0]);
    let w = blend1d_weights(&thresholds, 0.8);
    assert!((w[0] - 0.5).abs() < 1e-6 && (w[1] - 0.5).abs() < 1e-6 && w[2] == 0.0);
    let w = blend1d_weights(&thresholds, 1.6);
    assert!((w[1] - 1.0).abs() < 1e-6, "exact point plays alone: {w:?}");
    assert!(blend1d_weights(&[], 1.0).is_empty());
}

#[test]
fn blend2d_weights_bilinear_corners_center_and_edge_clamp() {
    let xs = [0.0, 1.0];
    let ys = [0.0, 1.0];
    // Dead center: quarter weight each.
    let w = blend2d_weights(&xs, &ys, 0.5, 0.5);
    assert!(w.iter().all(|&v| (v - 0.25).abs() < 1e-6), "{w:?}");
    // A corner takes full weight.
    let w = blend2d_weights(&xs, &ys, 0.0, 1.0);
    assert_eq!(w, vec![0.0, 0.0, 1.0, 0.0]);
    // Outside clamps to the nearest edge.
    let w = blend2d_weights(&xs, &ys, 2.0, -1.0);
    assert_eq!(w, vec![0.0, 1.0, 0.0, 0.0]);
    // Mid-x on the bottom edge splits between the bottom pair.
    let w = blend2d_weights(&xs, &ys, 0.5, 0.0);
    assert!((w[0] - 0.5).abs() < 1e-6 && (w[1] - 0.5).abs() < 1e-6);
}

fn blend1d_state(
    sync: bool,
    durations: &[f32],
    thresholds: &[f32],
    looping: bool,
) -> CompiledState {
    CompiledState {
        name: "locomotion".into(),
        rate: 1.0,
        looping,
        play: StatePlay::Blend1D(Blend1D {
            param: 0,
            thresholds: thresholds.to_vec(),
            plays: durations
                .iter()
                .enumerate()
                .map(|(i, &d)| clip_play(i, d))
                .collect(),
            sync,
        }),
        transitions: Vec::new(),
    }
}

#[test]
fn effective_duration_is_weight_averaged() {
    let s = blend1d_state(true, &[1.0, 0.5], &[0.0, 1.0], true);
    let w = s.play.weights(&[0.0]);
    assert!((s.play.effective_duration(&w) - 1.0).abs() < 1e-6);
    let w = s.play.weights(&[1.0]);
    assert!((s.play.effective_duration(&w) - 0.5).abs() < 1e-6);
    let w = s.play.weights(&[0.5]);
    assert!((s.play.effective_duration(&w) - 0.75).abs() < 1e-6);
}

#[test]
fn sync_clock_advances_in_phase_units() {
    let graph = CompiledGraph {
        params: vec![ParamSpec {
            name: "speed".into(),
            default: 0.0,
        }],
        states: vec![blend1d_state(true, &[1.0, 0.5], &[0.0, 1.0], true)],
        initial: 0,
    };
    let mut cursor = GraphCursor::start(&graph);
    // Fully on the 0.5s member: one wall-clock 0.25s = half a pass.
    cursor.advance(&graph, &[1.0], 0.25);
    assert!((cursor.clock - 0.5).abs() < 1e-6);
    // Normalized time IS the phase for synced blends.
    assert!((normalized_time(&graph.states[0], cursor.clock, &[1.0]) - 0.5).abs() < 1e-6);
}

#[test]
fn synced_members_sample_at_shared_phase() {
    // Two clips of different lengths whose X tracks ramp 0 -> 1 over their
    // own duration: at any shared phase both members agree on X, so the
    // blended pose equals that X regardless of the blend weight.
    let skeleton = Skeleton::new(vec![Joint {
        name: String::new(),
        parent: None,
        bind: JointPose::default(),
    }]);
    let ramp = |duration: f32| AnimationClip {
        root: None,
        duration,
        looping: true,
        tracks: vec![JointTrack {
            joint: 0,
            keys: vec![
                Keyframe {
                    time: 0.0,
                    pose: JointPose::default(),
                },
                Keyframe {
                    time: duration,
                    pose: JointPose {
                        translation: [1.0, 0.0, 0.0],
                        ..Default::default()
                    },
                },
            ],
        }],
        morph_keys: Vec::new(),
    };
    let clips = [ramp(1.0), ramp(0.5)];
    let graph = CompiledGraph {
        params: vec![ParamSpec {
            name: "speed".into(),
            default: 0.0,
        }],
        states: vec![blend1d_state(true, &[1.0, 0.5], &[0.0, 1.0], true)],
        initial: 0,
    };
    let cursor = GraphCursor {
        state: 0,
        clock: 0.25,
        fade: None,
    };
    // Mid-blend (weights 0.5 / 0.5) at phase 0.25: both members read 0.25.
    let locals = sample_pose(&graph, &cursor, &[0.5], &clips, &skeleton);
    assert!(
        (locals[0][3][0] - 0.25).abs() < 1e-4,
        "phase-locked members must agree: {}",
        locals[0][3][0]
    );
}

#[test]
fn non_sync_blend_samples_members_at_absolute_clock() {
    let (skeleton, clips) = pose_fixture(&[0.0, 2.0]);
    let graph = CompiledGraph {
        params: vec![ParamSpec {
            name: "speed".into(),
            default: 0.0,
        }],
        states: vec![blend1d_state(false, &[1.0, 1.0], &[0.0, 1.0], true)],
        initial: 0,
    };
    let mut cursor = GraphCursor::start(&graph);
    cursor.advance(&graph, &[0.5], 0.3);
    assert!((cursor.clock - 0.3).abs() < 1e-6, "seconds, not phase");
    let locals = sample_pose(&graph, &cursor, &[0.5], &clips, &skeleton);
    assert!(
        (locals[0][3][0] - 1.0).abs() < 1e-4,
        "even blend of 0 and 2"
    );
}

#[test]
fn zero_weight_members_do_not_affect_the_pose() {
    let (skeleton, clips) = pose_fixture(&[0.0, 2.0, 7.0]);
    let graph = CompiledGraph {
        params: vec![ParamSpec {
            name: "speed".into(),
            default: 0.0,
        }],
        states: vec![blend1d_state(
            true,
            &[1.0, 1.0, 1.0],
            &[0.0, 1.0, 2.0],
            true,
        )],
        initial: 0,
    };
    let cursor = GraphCursor::start(&graph);
    // Parked exactly on the middle member.
    let locals = sample_pose(&graph, &cursor, &[1.0], &clips, &skeleton);
    assert!((locals[0][3][0] - 2.0).abs() < 1e-4);
}

#[test]
fn refresh_clip_duration_updates_every_member_playing_it() {
    let mut graph = CompiledGraph {
        params: Vec::new(),
        states: vec![
            state("a", 0, Vec::new()),
            blend1d_state(true, &[1.0, 1.0], &[0.0, 1.0], true),
        ],
        initial: 0,
    };
    graph.refresh_clip_duration(0, 3.0);
    let StatePlay::Clip(play) = &graph.states[0].play else {
        panic!("state a is a clip");
    };
    assert_eq!(play.duration_secs, 3.0);
    let StatePlay::Blend1D(b) = &graph.states[1].play else {
        panic!("state b is a blend");
    };
    assert_eq!(b.plays[0].duration_secs, 3.0);
    assert_eq!(b.plays[1].duration_secs, 1.0, "other clip untouched");
}
