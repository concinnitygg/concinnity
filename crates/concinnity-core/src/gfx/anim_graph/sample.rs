// src/gfx/anim_graph/sample.rs
//
// Pose sampling for a graph cursor: the current state's members are sampled
// at their clock-derived times, blended by the state's blendspace weights,
// then crossfaded with the outgoing state while a transition fade is in
// flight.

use crate::gfx::pose_blend::{PoseBlend, blend_locals_in_place};
use crate::gfx::pose_scratch::PoseScratch;
use crate::gfx::skeleton::{AnimationClip, Skeleton};
use crate::gfx::transform::Mat4;
use crate::math::fract;
use alloc::vec::Vec;

use super::{CompiledGraph, CompiledState, GraphCursor};

/// Sample the cursor's blended pose into `scratch.locals`. `clip_at` maps a
/// member's clip index to the actual clip (the caller owns clip storage). The
/// remaining scratch buffers hold the fade's outgoing pose and the per-member
/// samples, so a warm scratch samples without allocating.
pub fn sample_graph_pose_into<'a>(
    graph: &CompiledGraph,
    cursor: &GraphCursor,
    params: &[f32],
    clip_at: impl Fn(usize) -> &'a AnimationClip,
    skeleton: &Skeleton,
    scratch: &mut PoseScratch,
) {
    let Some(fade) = &cursor.fade else {
        sample_state_into(
            &graph.states[cursor.state],
            cursor.clock,
            params,
            &clip_at,
            skeleton,
            &mut scratch.locals,
            BlendBufs {
                clip: &mut scratch.clip,
                weights: &mut scratch.weights,
            },
        );
        return;
    };
    // The outgoing state seeds the accumulator so the crossfade folds the
    // current state in at the fade's progress: f=0 is all outgoing, f=1 all
    // current, matching the fade clock's direction.
    sample_state_into(
        &graph.states[fade.from_state],
        fade.from_clock,
        params,
        &clip_at,
        skeleton,
        &mut scratch.locals,
        BlendBufs {
            clip: &mut scratch.clip,
            weights: &mut scratch.weights,
        },
    );
    sample_state_into(
        &graph.states[cursor.state],
        cursor.clock,
        params,
        &clip_at,
        skeleton,
        &mut scratch.aux,
        BlendBufs {
            clip: &mut scratch.clip,
            weights: &mut scratch.weights,
        },
    );
    blend_locals_in_place(&mut scratch.locals, &scratch.aux, fade.progress());
}

// The shared per-member sample and weight buffers a state blend folds
// through, disjoint from whichever output buffer the caller borrowed.
struct BlendBufs<'a> {
    clip: &'a mut Vec<Mat4>,
    weights: &'a mut Vec<f32>,
}

// One state's pose at its clock, written into `out` through the shared
// blend buffers. Zero-weight members are not sampled at all, so a
// blendspace parked on one member costs the same as a single clip.
fn sample_state_into<'a>(
    state: &CompiledState,
    clock: f32,
    params: &[f32],
    clip_at: &impl Fn(usize) -> &'a AnimationClip,
    skeleton: &Skeleton,
    out: &mut Vec<Mat4>,
    bufs: BlendBufs<'_>,
) {
    state.play.weights_into(params, bufs.weights);
    let members = state.play.members();

    let mut fold = PoseBlend::new(out);
    for (member, &w) in members.iter().zip(bufs.weights.iter()) {
        if w <= 0.0 {
            continue;
        }
        let t = member_time(state, member.duration_secs, clock);
        clip_at(member.clip).sample_looped_into(t, state.looping, skeleton, bufs.clip);
        fold.add(bufs.clip, w);
    }
    if !fold.seeded() {
        // Unreachable with a compiled graph (weights always mark at least
        // one member), but a bind pose beats a panic if it ever is.
        out.clear();
        out.extend(skeleton.joints().iter().map(|j| j.bind.to_matrix()));
    }
}

// A member's clip-local sample time. Synced blends carry a normalized phase
// clock, so every member is at the same fraction of its own cycle;
// everything else runs on seconds and lets `sample_looped` wrap or clamp by
// the state's loop mode.
fn member_time(state: &CompiledState, member_duration: f32, clock: f32) -> f32 {
    if state.play.sync() {
        let phase = if state.looping {
            fract(clock)
        } else {
            clock.clamp(0.0, 1.0)
        };
        phase * member_duration
    } else {
        clock
    }
}
