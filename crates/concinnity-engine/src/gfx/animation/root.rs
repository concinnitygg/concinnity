// src/gfx/animation/root.rs
//
// Root-motion publication for flat clip buckets: the displacement the
// bucket's weighted blend covered between two absolute clip times. Graph
// buckets get the equivalent from `gfx::anim_graph::cursor_root_delta`;
// both feed the per-frame `RootMotionEvent` events consumed by the rig drive in
// PhysicsSystem.

use crate::gfx::root_motion::{add3, scale3};

use super::flat::ClipEntry;

// The displacement a flat bucket covered over `[t0, t1]` (absolute clip
// seconds), weighted like the pose blend: a single clip plays at full
// strength regardless of weight, several clips normalize. Clips without a
// root track contribute nothing.
pub(super) fn flat_root_delta(clips: &[ClipEntry], weights: &[f32], t0: f32, t1: f32) -> [f32; 3] {
    match clips {
        [] => [0.0; 3],
        [single] => single
            .clip
            .root
            .as_ref()
            .map(|root| root.delta(t0, t1, single.clip.duration, single.clip.looping))
            .unwrap_or([0.0; 3]),
        many => {
            let total: f32 = weights.iter().map(|w| w.max(0.0)).sum();
            if total <= 1e-6 {
                return [0.0; 3];
            }
            let mut out = [0.0; 3];
            for (entry, &w) in many.iter().zip(weights) {
                if w <= 0.0 {
                    continue;
                }
                if let Some(root) = &entry.clip.root {
                    let d = root.delta(t0, t1, entry.clip.duration, entry.clip.looping);
                    out = add3(out, scale3(d, w / total));
                }
            }
            out
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::gfx::root_motion::{RootKey, RootTrack};
    use crate::gfx::skinning::AnimationClip;

    // A clip whose root travels `per` along +X over its whole duration, linear.
    fn moving_clip(duration: f32, looping: bool, per: f32) -> ClipEntry {
        ClipEntry {
            clip: AnimationClip {
                morph_keys: Vec::new(),
                duration,
                looping,
                tracks: Vec::new(),
                root: Some(RootTrack {
                    keys: vec![
                        RootKey {
                            time: 0.0,
                            translation: [0.0, 0.0, 0.0],
                        },
                        RootKey {
                            time: duration,
                            translation: [per, 0.0, 0.0],
                        },
                    ],
                }),
            },
            declared_weight: 1.0,
            fade_in_secs: 0.0,
        }
    }

    // A clip with no root track: contributes no displacement.
    fn rootless_clip(duration: f32) -> ClipEntry {
        ClipEntry {
            clip: AnimationClip {
                morph_keys: Vec::new(),
                duration,
                looping: false,
                tracks: Vec::new(),
                root: None,
            },
            declared_weight: 1.0,
            fade_in_secs: 0.0,
        }
    }

    #[test]
    fn empty_bucket_has_no_displacement() {
        assert_eq!(flat_root_delta(&[], &[], 0.0, 1.0), [0.0; 3]);
    }

    #[test]
    fn single_clip_reports_its_root_delta() {
        // Half of a 1s +4 clip covers +2 on X; a single clip ignores its weight.
        let clips = [moving_clip(1.0, false, 4.0)];
        let d = flat_root_delta(&clips, &[0.0], 0.0, 0.5);
        assert!((d[0] - 2.0).abs() < 1e-5, "{d:?}");
        assert_eq!([d[1], d[2]], [0.0, 0.0]);
    }

    #[test]
    fn single_clip_without_a_root_track_is_silent() {
        let clips = [rootless_clip(1.0)];
        assert_eq!(flat_root_delta(&clips, &[1.0], 0.0, 1.0), [0.0; 3]);
    }

    #[test]
    fn many_clips_blend_by_normalized_weight() {
        // Two +4 clips over their full second, weighted 1 and 3: the normalized
        // blend of two identical deltas is just that delta (4 on X).
        let clips = [moving_clip(1.0, false, 4.0), moving_clip(1.0, false, 4.0)];
        let d = flat_root_delta(&clips, &[1.0, 3.0], 0.0, 1.0);
        assert!((d[0] - 4.0).abs() < 1e-5, "{d:?}");
    }

    #[test]
    fn many_clips_skip_zero_weight_members() {
        // The first clip is weighted off; only the second (weight 2 of a total
        // 2) contributes, so its +6 delta passes through unscaled.
        let clips = [moving_clip(1.0, false, 99.0), moving_clip(1.0, false, 6.0)];
        let d = flat_root_delta(&clips, &[0.0, 2.0], 0.0, 1.0);
        assert!((d[0] - 6.0).abs() < 1e-5, "{d:?}");
    }

    #[test]
    fn many_clips_with_zero_total_weight_are_silent() {
        // Every member weighted off: the guard returns no displacement rather
        // than dividing by a zero total.
        let clips = [moving_clip(1.0, false, 4.0), moving_clip(1.0, false, 4.0)];
        assert_eq!(flat_root_delta(&clips, &[0.0, 0.0], 0.0, 1.0), [0.0; 3]);
    }
}
