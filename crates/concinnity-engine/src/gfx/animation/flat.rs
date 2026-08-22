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
    pub(crate) declared_weight: f32,
    // Seconds the clip ramps from zero to `declared_weight` at world start.
    // Zero plays the clip at full strength from the first frame.
    pub fade_in_secs: f32,
}

// One ramp between two weight vectors. The bucket holds at most one
// transition at a time; a new transition supersedes any in flight.
#[derive(Debug)]
pub(super) struct Transition {
    pub(crate) source_weights: Vec<f32>,
    pub(crate) target_weights: Vec<f32>,
    // Wall-clock seconds since the system's first step.
    pub(crate) start_secs: f32,
    // Length of the ramp. Zero snaps to `target_weights` on the next
    // `step`.
    pub duration_secs: f32,
}

// The live blend vector consumed by the per-pose weighted blend, plus any
// in-flight ramp.
#[derive(Default)]
pub(super) struct FlatState {
    pub(crate) current_weights: Vec<f32>,
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

#[cfg(test)]
mod tests {
    use super::*;

    // A bucket mid-ramp between two weight vectors.
    fn ramp(source: Vec<f32>, target: Vec<f32>, start: f32, duration: f32) -> FlatState {
        FlatState {
            current_weights: source.clone(),
            transition: Some(Transition {
                source_weights: source,
                target_weights: target,
                start_secs: start,
                duration_secs: duration,
            }),
        }
    }

    #[test]
    fn advance_weights_lerps_mid_ramp() {
        // A quarter of the way through a 1s ramp: each slot sits a quarter of
        // the way from its source toward its target, and the ramp stays live.
        let mut s = ramp(vec![0.0, 1.0], vec![1.0, 0.0], 0.0, 1.0);
        advance_weights(&mut s, 0.25);
        assert!((s.current_weights[0] - 0.25).abs() < 1e-6);
        assert!((s.current_weights[1] - 0.75).abs() < 1e-6);
        assert!(s.transition.is_some(), "ramp still in flight");
    }

    #[test]
    fn advance_weights_snaps_and_clears_at_end() {
        // At start + duration the ramp finishes: weights snap to the target and
        // the transition is dropped.
        let mut s = ramp(vec![0.0], vec![1.0], 0.0, 1.0);
        advance_weights(&mut s, 1.0);
        assert_eq!(s.current_weights, vec![1.0]);
        assert!(s.transition.is_none(), "finished ramp is cleared");
    }

    #[test]
    fn advance_weights_zero_duration_snaps_immediately() {
        // A zero-length ramp finishes on the next step, snapping to the target.
        let mut s = ramp(vec![0.3], vec![0.9], 0.0, 0.0);
        advance_weights(&mut s, 0.0);
        assert_eq!(s.current_weights, vec![0.9]);
        assert!(s.transition.is_none());
    }

    #[test]
    fn advance_weights_without_a_transition_is_a_noop() {
        // No ramp in flight: the weights are left untouched.
        let mut s = FlatState {
            current_weights: vec![0.4, 0.6],
            transition: None,
        };
        advance_weights(&mut s, 5.0);
        assert_eq!(s.current_weights, vec![0.4, 0.6]);
    }
}
