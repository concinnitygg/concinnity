// src/gfx/animation/root.rs
//
// Root-motion publication for flat clip buckets: the displacement the
// bucket's weighted blend covered between two absolute clip times. Graph
// buckets get the equivalent from `gfx::anim_graph::cursor_root_delta`;
// both feed the per-frame `RootMotion` events consumed by the rig drive in
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
