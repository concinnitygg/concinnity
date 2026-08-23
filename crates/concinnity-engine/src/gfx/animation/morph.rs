// Per-frame morph weights for a flat bucket: the clips' morph tracks blended
// by the bucket's live weights, then added onto the pose's static base layer.

use crate::gfx::morph_weights::compose_morph_weights;
use crate::gfx::pose_scratch::PoseScratch;

use super::flat::{ClipEntry, FlatState};

// Sample every clip with a morph track at `t`, normalised by the live weights
// of the clips that contributed (clips without a track do not dilute the
// result), and write `base + blend` into `out`. Leaves `out` untouched when
// no clip carries a morph track, so a pose with only a base layer keeps it.
pub(super) fn update_weights(
    clips: &[ClipEntry],
    flat: &FlatState,
    t: f32,
    base: &[f32],
    scratch: &mut PoseScratch,
    out: &mut Vec<f32>,
) {
    let acc = &mut scratch.morph;
    acc.clear();
    let mut weight_sum = 0.0f32;
    for (i, entry) in clips.iter().enumerate() {
        if entry.clip.morph_keys.is_empty() {
            continue;
        }
        let w = if clips.len() == 1 {
            1.0
        } else {
            flat.current_weights.get(i).copied().unwrap_or(0.0)
        };
        if w <= 0.0 {
            continue;
        }
        entry
            .clip
            .sample_morph_weights_into(t, entry.clip.looping, &mut scratch.weights);
        if acc.len() < scratch.weights.len() {
            acc.resize(scratch.weights.len(), 0.0);
        }
        for (a, s) in acc.iter_mut().zip(scratch.weights.iter()) {
            *a += s * w;
        }
        weight_sum += w;
    }
    if weight_sum > 0.0 {
        for a in acc.iter_mut() {
            *a /= weight_sum;
        }
        compose_morph_weights(base, acc, out);
    }
}
