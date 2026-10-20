//! Per-frame morph weights for a flat bucket: the clips' morph tracks blended
//! by the bucket's live weights, then added onto the pose's static base layer.

use alloc::vec::Vec;

use super::{ClipEntry, FlatState};
use crate::animation::morph_weights::compose_morph_weights;
use crate::animation::pose_scratch::PoseScratch;

/// Sample every clip with a morph track at `t`, normalized by the live weights
/// of the clips that contributed (clips without a track do not dilute the
/// result), and write `base + blend` into `out`. Leaves `out` untouched when
/// no clip carries a morph track, so a pose with only a base layer keeps it.
pub fn morph_weights_into(
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::animation::skeleton::AnimationClip;
    use alloc::vec;

    fn clip_without_morphs() -> ClipEntry {
        ClipEntry {
            clip: AnimationClip {
                morph_keys: Vec::new(),
                duration: 1.0,
                looping: false,
                tracks: Vec::new(),
                root: None,
            },
            declared_weight: 1.0,
            fade_in_secs: 0.0,
        }
    }

    // Clips without a morph track leave the pose's base layer as it was.
    #[test]
    fn no_morph_tracks_leave_the_output_alone() {
        let clips = [clip_without_morphs(), clip_without_morphs()];
        let flat = FlatState {
            current_weights: vec![0.5, 0.5],
            transition: None,
        };
        let mut out = vec![0.25, 0.75];
        morph_weights_into(
            &clips,
            &flat,
            0.5,
            &[1.0, 1.0],
            &mut PoseScratch::default(),
            &mut out,
        );
        assert_eq!(out, vec![0.25, 0.75]);
    }
}
