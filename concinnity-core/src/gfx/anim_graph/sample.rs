// src/gfx/anim_graph/sample.rs
//
// Pose sampling for a graph cursor: the current state's members are sampled
// at their clock-derived times, blended by the state's blendspace weights,
// then crossfaded with the outgoing state while a transition fade is in
// flight.

use crate::gfx::skinning::{self, AnimationClip, Mat4, Skeleton};

use super::{CompiledGraph, CompiledState, GraphCursor};

// Sample the cursor's blended pose. `clip_at` maps a member's clip index to
// the actual clip (the caller owns clip storage).
pub fn sample_graph_pose<'a>(
    graph: &CompiledGraph,
    cursor: &GraphCursor,
    params: &[f32],
    clip_at: impl Fn(usize) -> &'a AnimationClip,
    skeleton: &Skeleton,
) -> Vec<Mat4> {
    let cur = sample_state(
        &graph.states[cursor.state],
        cursor.clock,
        params,
        &clip_at,
        skeleton,
    );
    let Some(fade) = &cursor.fade else {
        return cur;
    };
    let from = sample_state(
        &graph.states[fade.from_state],
        fade.from_clock,
        params,
        &clip_at,
        skeleton,
    );
    skinning::blend_locals(&from, &cur, fade.progress())
}

// One state's pose at its clock. Zero-weight members are not sampled at all,
// so a blendspace parked on one member costs the same as a single clip.
fn sample_state<'a>(
    state: &CompiledState,
    clock: f32,
    params: &[f32],
    clip_at: &impl Fn(usize) -> &'a AnimationClip,
    skeleton: &Skeleton,
) -> Vec<Mat4> {
    let weights = state.play.weights(params);
    let members = state.play.members();

    let mut sampled: Vec<Vec<Mat4>> = Vec::new();
    let mut live_weights: Vec<f32> = Vec::new();
    for (member, &w) in members.iter().zip(&weights) {
        if w <= 0.0 {
            continue;
        }
        let t = member_time(state, member.duration_secs, clock);
        sampled.push(clip_at(member.clip).sample_looped(t, state.looping, skeleton));
        live_weights.push(w);
    }
    match sampled.len() {
        // Unreachable with a compiled graph (weights always mark at least
        // one member), but a bind pose beats a panic if it ever is.
        0 => skeleton
            .joints()
            .iter()
            .map(|j| j.bind.to_matrix())
            .collect(),
        1 => sampled.pop().expect("one sampled member"),
        _ => skinning::blend_many(&sampled, &live_weights),
    }
}

// A member's clip-local sample time. Synced blends carry a normalized phase
// clock, so every member is at the same fraction of its own cycle;
// everything else runs on seconds and lets `sample_looped` wrap or clamp by
// the state's loop mode.
fn member_time(state: &CompiledState, member_duration: f32, clock: f32) -> f32 {
    if state.play.sync() {
        let phase = if state.looping {
            clock.fract()
        } else {
            clock.clamp(0.0, 1.0)
        };
        phase * member_duration
    } else {
        clock
    }
}
