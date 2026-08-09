// src/gfx/anim_graph/root.rs
//
// Root-motion deltas for a graph cursor: the displacement covered by the
// active state's members (and a fading outgoing state) between two cursor
// snapshots. Mirrors the pose sampler's weighting exactly, so the character
// moves at the speed the blended feet imply.

use crate::gfx::skinning::AnimationClip;
use concinnity_core::gfx::root_motion::{add3, scale3};

use super::{CompiledGraph, CompiledState, GraphCursor};

// The displacement covered between the cursor snapshots taken before and
// after one `advance`, in the mesh's local space. A frame that took a
// transition contributes nothing (the clocks reset mid-frame; one frame of
// root motion at a state change is imperceptible and never wrong-direction).
// While a fade is in flight the outgoing state's displacement blends in by
// the fade's progress, matching the pose crossfade.
pub fn cursor_root_delta<'a>(
    graph: &CompiledGraph,
    before: &GraphCursor,
    after: &GraphCursor,
    params: &[f32],
    clip_at: &impl Fn(usize) -> &'a AnimationClip,
) -> [f32; 3] {
    if before.state != after.state {
        return [0.0; 3];
    }
    let state = &graph.states[after.state];
    let delta = state_root_delta(state, before.clock, after.clock, params, clip_at);
    if let (Some(fade_before), Some(fade_after)) = (&before.fade, &after.fade)
        && fade_before.from_state == fade_after.from_state
    {
        let from = &graph.states[fade_after.from_state];
        let from_delta = state_root_delta(
            from,
            fade_before.from_clock,
            fade_after.from_clock,
            params,
            clip_at,
        );
        let progress = fade_after.progress();
        return add3(scale3(from_delta, 1.0 - progress), scale3(delta, progress));
    }
    delta
}

// One state's displacement between two of its clock values (same units the
// state's clock runs on), blending member root tracks by the state's
// current weights. Members without a root track contribute nothing -- a
// blend may mix a root-motion walk with an in-place idle and the character
// slows down accordingly.
pub fn state_root_delta<'a>(
    state: &CompiledState,
    clock0: f32,
    clock1: f32,
    params: &[f32],
    clip_at: &impl Fn(usize) -> &'a AnimationClip,
) -> [f32; 3] {
    let weights = state.play.weights(params);
    let sync = state.play.sync();
    let mut delta = [0.0f32; 3];
    for (member, &w) in state.play.members().iter().zip(&weights) {
        if w <= 0.0 {
            continue;
        }
        let Some(root) = clip_at(member.clip).root.as_ref() else {
            continue;
        };
        // A synced state's clock is normalized phase; each member covers its
        // own duration per pass.
        let (t0, t1) = if sync {
            (clock0 * member.duration_secs, clock1 * member.duration_secs)
        } else {
            (clock0, clock1)
        };
        let d = root.delta(t0, t1, member.duration_secs, state.looping);
        delta = add3(delta, scale3(d, w));
    }
    delta
}

#[cfg(test)]
mod tests {
    use super::super::{
        Blend1D, ClipPlay, CompiledGraph, CompiledState, CompiledTransition, GraphCursor, StatePlay,
    };
    use super::*;
    use concinnity_core::gfx::root_motion::{RootKey, RootTrack};

    // A looping clip of `duration` covering `travel` X per cycle.
    fn walker(duration: f32, travel: f32) -> AnimationClip {
        AnimationClip {
            duration,
            looping: true,
            tracks: Vec::new(),
            morph_keys: Vec::new(),
            root: Some(RootTrack {
                keys: vec![
                    RootKey {
                        time: 0.0,
                        translation: [0.0; 3],
                    },
                    RootKey {
                        time: duration,
                        translation: [travel, 0.0, 0.0],
                    },
                ],
            }),
        }
    }

    fn in_place(duration: f32) -> AnimationClip {
        AnimationClip {
            duration,
            looping: true,
            tracks: Vec::new(),
            morph_keys: Vec::new(),
            root: None,
        }
    }

    fn clip_state(name: &str, clip: usize, duration: f32) -> CompiledState {
        CompiledState {
            name: name.into(),
            rate: 1.0,
            looping: true,
            play: StatePlay::Clip(ClipPlay {
                clip,
                duration_secs: duration,
            }),
            transitions: Vec::new(),
        }
    }

    #[test]
    fn clip_state_delta_tracks_the_clock() {
        let clips = [walker(1.0, 2.0)];
        let state = clip_state("walk", 0, 1.0);
        let d = state_root_delta(&state, 0.25, 0.75, &[], &|i| &clips[i]);
        assert!((d[0] - 1.0).abs() < 1e-5, "{d:?}");
        // Across a wrap.
        let d = state_root_delta(&state, 0.75, 1.25, &[], &|i| &clips[i]);
        assert!((d[0] - 1.0).abs() < 1e-5, "{d:?}");
    }

    #[test]
    fn blend_scales_displacement_by_member_weight() {
        // Walk (2/cycle over 1s) mixed 50/50 with an in-place idle at the
        // blend midpoint: half the walk's speed.
        let clips = [in_place(1.0), walker(1.0, 2.0)];
        let state = CompiledState {
            name: "locomotion".into(),
            rate: 1.0,
            looping: true,
            play: StatePlay::Blend1D(Blend1D {
                param: 0,
                thresholds: vec![0.0, 1.0],
                plays: vec![
                    ClipPlay {
                        clip: 0,
                        duration_secs: 1.0,
                    },
                    ClipPlay {
                        clip: 1,
                        duration_secs: 1.0,
                    },
                ],
                sync: true,
            }),
            transitions: Vec::new(),
        };
        // Synced state: clock is phase. Half a pass at 50/50.
        let d = state_root_delta(&state, 0.0, 0.5, &[0.5], &|i| &clips[i]);
        assert!((d[0] - 0.5).abs() < 1e-5, "half of half the cycle: {d:?}");
        // Fully on the walk: the full half-cycle displacement.
        let d = state_root_delta(&state, 0.0, 0.5, &[1.0], &|i| &clips[i]);
        assert!((d[0] - 1.0).abs() < 1e-5, "{d:?}");
    }

    #[test]
    fn cursor_delta_skips_transition_frames_and_blends_fades() {
        let clips = [walker(1.0, 2.0), walker(1.0, 4.0)];
        let mut graph = CompiledGraph {
            params: Vec::new(),
            states: vec![clip_state("a", 0, 1.0), clip_state("b", 1, 1.0)],
            initial: 0,
        };
        graph.states[0].transitions = vec![CompiledTransition {
            to: 1,
            duration_secs: 0.4,
            exit_time: None,
            conditions: Vec::new(),
        }];

        // Frame 1: the transition fires -> no root motion this frame.
        let before = GraphCursor::start(&graph);
        let mut cursor = before.clone();
        cursor.advance(&graph, &[], 0.1);
        assert_eq!(cursor.state, 1);
        let d = cursor_root_delta(&graph, &before, &cursor, &[], &|i| &clips[i]);
        assert_eq!(d, [0.0; 3]);

        // Frame 2: mid-fade. Incoming b (4/s) at progress 0.5 blended with
        // outgoing a (2/s): 0.2s * (0.5*4 + 0.5*2) = 0.6.
        let before = cursor.clone();
        cursor.advance(&graph, &[], 0.2);
        let d = cursor_root_delta(&graph, &before, &cursor, &[], &|i| &clips[i]);
        assert!((d[0] - 0.6).abs() < 1e-5, "{d:?}");

        // After the fade: pure incoming speed.
        let mut cursor2 = cursor.clone();
        cursor2.fade = None;
        let before = cursor2.clone();
        cursor2.advance(&graph, &[], 0.25);
        let d = cursor_root_delta(&graph, &before, &cursor2, &[], &|i| &clips[i]);
        assert!((d[0] - 1.0).abs() < 1e-5, "{d:?}");
    }
}
