// src/gfx/animation/flat.rs
//
// The weighted-blend drive for a clip bucket: every clip targeting the mesh
// plays simultaneously, mixed by a live weight vector. Startup fade-ins and
// runtime `anim-crossfade` commands are both ramps between weight vectors.

use crate::gfx::skinning::AnimationClip;

// A runtime clip plus the static metadata captured from its `Animation`
// asset. The live blend weight is stored separately on the owning
// `FlatState` so a runtime crossfade can re-weight clips without rewriting
// their data. Shared by graph buckets too (a graph indexes into the same
// clip list; `declared_weight` / `fade_in_secs` are then unused).
pub(super) struct ClipEntry {
    pub clip: AnimationClip,
    // The declared steady-state weight from the asset. The bucket settles
    // on this once any initial fade-in or runtime crossfade completes.
    pub declared_weight: f32,
    // Seconds the clip ramps from zero to `declared_weight` at world start.
    // Zero plays the clip at full strength from the first frame.
    pub fade_in_secs: f32,
}

// One ramp between two weight vectors. The bucket holds at most one
// transition at a time; a new transition supersedes any in flight.
#[derive(Debug)]
pub(super) struct Transition {
    pub source_weights: Vec<f32>,
    pub target_weights: Vec<f32>,
    // Wall-clock seconds since the system's first step.
    pub start_secs: f32,
    // Length of the ramp. Zero snaps to `target_weights` on the next
    // `step`.
    pub duration_secs: f32,
}

// The live blend vector consumed by `skinning::blend_many`, plus any
// in-flight ramp.
#[derive(Default)]
pub(super) struct FlatState {
    pub current_weights: Vec<f32>,
    pub transition: Option<Transition>,
}

// Advance one bucket's `current_weights` along its active transition (if
// any).
pub(super) fn advance_weights(state: &mut FlatState, now_secs: f32) {
    if let Some(tr) = &state.transition {
        let finished = if tr.duration_secs <= 0.0 {
            true
        } else {
            now_secs >= tr.start_secs + tr.duration_secs
        };
        if finished {
            state.current_weights.clone_from(&tr.target_weights);
            state.transition = None;
        } else {
            let progress = ((now_secs - tr.start_secs) / tr.duration_secs).clamp(0.0, 1.0);
            for (slot, (src, dst)) in state
                .current_weights
                .iter_mut()
                .zip(tr.source_weights.iter().zip(tr.target_weights.iter()))
            {
                *slot = src + (dst - src) * progress;
            }
        }
    }
}
