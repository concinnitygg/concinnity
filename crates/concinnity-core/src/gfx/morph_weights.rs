//! Composition of a pose's static morph base layer with the weights a clip
//! samples each frame.

use alloc::vec::Vec;

/// Write `base + clip` into `out` (cleared first), clamped to `[0, 1]` per
/// target. The longer input sets the length; the shorter one reads as zero
/// past its end.
pub fn compose_morph_weights(base: &[f32], clip: &[f32], out: &mut Vec<f32>) {
    out.clear();
    let n = base.len().max(clip.len());
    out.extend((0..n).map(|i| {
        let b = base.get(i).copied().unwrap_or(0.0);
        let c = clip.get(i).copied().unwrap_or(0.0);
        (b + c).clamp(0.0, 1.0)
    }));
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn clip_weights_add_onto_the_base_and_clamp() {
        let mut out = Vec::new();
        compose_morph_weights(&[0.5, 0.8, 0.0], &[0.25, 0.5, -0.5], &mut out);
        assert_eq!(out, [0.75, 1.0, 0.0]);
    }

    #[test]
    fn the_longer_side_sets_the_length() {
        let mut out = Vec::new();
        compose_morph_weights(&[0.5], &[0.0, 0.3], &mut out);
        assert_eq!(out, [0.5, 0.3]);
        compose_morph_weights(&[0.5, 0.2], &[], &mut out);
        assert_eq!(out, [0.5, 0.2]);
    }
}
