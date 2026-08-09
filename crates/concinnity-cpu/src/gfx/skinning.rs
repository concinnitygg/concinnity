// src/gfx/skinning.rs
//
// Pose blending: combining several sampled poses into one set of local joint
// matrices. The skeleton, clip, and transform types these operate on live in
// concinnity-core and are re-exported here under their historical paths.

pub use concinnity_core::gfx::pose_scratch::PoseScratch;
pub use concinnity_core::gfx::skeleton::{
    AnimationClip, Joint, JointPose, JointTrack, Keyframe, Skeleton,
};
pub use concinnity_core::gfx::transform::{
    IDENTITY, Mat4, blend_matrices, decompose, euler_yxz_from_quat, mat4_affine_inverse, mat4_mul,
    trs_matrix,
};

/// Blend `other` into `acc` in place by weight `f`, clamped to `[0, 1]`:
/// `f = 0` leaves `acc` unchanged, `f = 1` replaces it with `other`. Each
/// joint pair is interpolated in TRS space (see `blend_matrices`), the same
/// interpolation a single clip uses between keyframes, so a blended pose is
/// continuous with a clip's own sampling. Arrays of unequal length blend the
/// common prefix and keep the longer array's tail unchanged (growing `acc`
/// from `other` when `other` is longer).
pub fn blend_locals_in_place(acc: &mut Vec<Mat4>, other: &[Mat4], f: f32) {
    let n = acc.len().min(other.len());
    for (a, b) in acc[..n].iter_mut().zip(&other[..n]) {
        *a = blend_matrices(*a, *b, f);
    }
    if other.len() > acc.len() {
        acc.extend_from_slice(&other[acc.len()..]);
    }
}

/// Incremental normalized weighted blend of N local-pose arrays into a
/// caller-owned accumulator, allocation-free once the accumulator is warm.
/// After adding pose `i` the accumulator equals the weighted blend of poses
/// `0..=i`: the first pose always seeds the result (so an all-zero-weight
/// set falls back to it), later poses fold in at `w / total`, and negative
/// weights clamp to 0. Adding a single pose returns it unchanged.
pub struct PoseBlend<'a> {
    acc: &'a mut Vec<Mat4>,
    acc_weight: f32,
    seeded: bool,
}

impl<'a> PoseBlend<'a> {
    /// Start a blend into `acc`. The buffer is cleared on the first `add`,
    /// so its capacity carries over from previous frames.
    pub fn new(acc: &'a mut Vec<Mat4>) -> Self {
        Self {
            acc,
            acc_weight: 0.0,
            seeded: false,
        }
    }

    /// Whether a pose has seeded the accumulator yet. Callers use this to
    /// skip sampling zero-weight poses that would fold in as no-ops.
    pub fn seeded(&self) -> bool {
        self.seeded
    }

    /// Fold `pose` into the accumulator at weight `w`.
    pub fn add(&mut self, pose: &[Mat4], w: f32) {
        let w = w.max(0.0);
        if !self.seeded {
            self.acc.clear();
            self.acc.extend_from_slice(pose);
            self.acc_weight = w;
            self.seeded = true;
            return;
        }
        if w <= 0.0 {
            return;
        }
        let total = self.acc_weight + w;
        let f = if total > 1e-6 { w / total } else { 0.0 };
        blend_locals_in_place(self.acc, pose, f);
        self.acc_weight = total;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn approx(a: f32, b: f32) -> bool {
        (a - b).abs() < 1e-4
    }

    // The Vec-returning blend the in-place fold replaced, kept as the test
    // oracle: the fold must reproduce it bit for bit.
    fn blend_locals_reference(a: &[Mat4], b: &[Mat4], f: f32) -> Vec<Mat4> {
        let n = a.len().max(b.len());
        let mut out = Vec::with_capacity(n);
        for i in 0..n {
            match (a.get(i), b.get(i)) {
                (Some(&ma), Some(&mb)) => out.push(blend_matrices(ma, mb, f)),
                (Some(&ma), None) => out.push(ma),
                (None, Some(&mb)) => out.push(mb),
                (None, None) => unreachable!("index below max of both lengths"),
            }
        }
        out
    }

    fn blend_many_reference(poses: &[Vec<Mat4>], weights: &[f32]) -> Vec<Mat4> {
        let Some(first) = poses.first() else {
            return Vec::new();
        };
        let mut acc = first.clone();
        let mut acc_w = weights.first().copied().unwrap_or(1.0).max(0.0);
        for (i, pose) in poses.iter().enumerate().skip(1) {
            let w = weights.get(i).copied().unwrap_or(1.0).max(0.0);
            if w <= 0.0 {
                continue;
            }
            let total = acc_w + w;
            let f = if total > 1e-6 { w / total } else { 0.0 };
            acc = blend_locals_reference(&acc, pose, f);
            acc_w = total;
        }
        acc
    }

    fn blend_locals(a: &[Mat4], b: &[Mat4], f: f32) -> Vec<Mat4> {
        let mut acc = a.to_vec();
        blend_locals_in_place(&mut acc, b, f);
        acc
    }

    fn blend_many(poses: &[Vec<Mat4>], weights: &[f32]) -> Vec<Mat4> {
        let mut acc = Vec::new();
        let mut fold = PoseBlend::new(&mut acc);
        for (i, pose) in poses.iter().enumerate() {
            let w = weights.get(i).copied().unwrap_or(1.0);
            if fold.seeded() && w <= 0.0 {
                continue;
            }
            fold.add(pose, w);
        }
        acc
    }

    fn pose(t: [f32; 3], r: [f32; 3], s: [f32; 3]) -> Mat4 {
        JointPose {
            translation: t,
            rotation_deg: r,
            scale: s,
        }
        .to_matrix()
    }

    #[test]
    fn blend_locals_endpoints_are_exact() {
        // f=0 must equal a, f=1 must equal b: a cross-fade is continuous with
        // each source clip at its extremes.
        let a = vec![pose([1.0, 0.0, 0.0], [10.0, 20.0, 30.0], [1.0, 2.0, 1.0])];
        let b = vec![pose([0.0, 4.0, -1.0], [-50.0, 70.0, 5.0], [2.0, 1.0, 0.5])];
        let at0 = blend_locals(&a, &b, 0.0);
        let at1 = blend_locals(&a, &b, 1.0);
        for c in 0..4 {
            for row in 0..4 {
                assert!(approx(at0[0][c][row], a[0][c][row]), "f=0 [{}][{}]", c, row);
                assert!(approx(at1[0][c][row], b[0][c][row]), "f=1 [{}][{}]", c, row);
            }
        }
    }

    #[test]
    fn blend_locals_midpoint_slerps_rotation() {
        // The f=0.5 blend of two pure yaws is the yaw midpoint: rotation is
        // slerped, not matrix-lerped (which would shrink the rotation).
        let a = vec![pose([0.0; 3], [0.0, 0.0, 0.0], [1.0; 3])];
        let b = vec![pose([0.0; 3], [0.0, 90.0, 0.0], [1.0; 3])];
        let mid = blend_locals(&a, &b, 0.5);
        // For a pure yaw the first column is (cos, 0, -sin).
        let yaw = (-mid[0][0][2]).atan2(mid[0][0][0]).to_degrees();
        assert!(approx(yaw, 45.0), "yaw was {}", yaw);
    }

    #[test]
    fn blend_locals_unequal_lengths_keep_the_longer_tail() {
        let out = blend_locals(&[IDENTITY], &[IDENTITY, IDENTITY, IDENTITY], 0.5);
        assert_eq!(out.len(), 3);
        // The in-place direction too: a longer accumulator keeps its tail.
        let mut acc = vec![IDENTITY, IDENTITY, IDENTITY];
        blend_locals_in_place(&mut acc, &[IDENTITY], 0.5);
        assert_eq!(acc.len(), 3);
    }

    #[test]
    fn blend_many_normalises_weights() {
        // Three pure-yaw poses at 0/30/90 deg blended 1:1:2 must land on the
        // weighted-average yaw 52.5 deg. Equal scaling of every weight (the
        // normalisation) must not move the result.
        let yaw = |deg: f32| vec![pose([0.0; 3], [0.0, deg, 0.0], [1.0; 3])];
        let poses = vec![yaw(0.0), yaw(30.0), yaw(90.0)];
        let recover = |out: &[Mat4]| (-out[0][0][2]).atan2(out[0][0][0]).to_degrees();
        let a = blend_many(&poses, &[1.0, 1.0, 2.0]);
        let b = blend_many(&poses, &[5.0, 5.0, 10.0]);
        assert!(approx(recover(&a), 52.5), "yaw was {}", recover(&a));
        assert!(
            approx(recover(&a), recover(&b)),
            "weight scaling moved blend"
        );
    }

    #[test]
    fn blend_many_skips_zero_weight_and_falls_back_to_first() {
        let poses = vec![vec![IDENTITY], vec![IDENTITY]];
        // A second clip at weight 0 leaves the first untouched.
        let zeroed = blend_many(&poses, &[1.0, 0.0]);
        assert_eq!(zeroed, poses[0]);
        // All-zero weights also fall back to the first array.
        let all_zero = blend_many(&poses, &[0.0, 0.0]);
        assert_eq!(all_zero, poses[0]);
    }

    #[test]
    fn blend_many_with_zero_first_weight_picks_up_later_clip() {
        // A 0-weight first clip must not poison the fold: the second clip
        // (weight 1) should win outright.
        let a = vec![pose([9.0, 9.0, 9.0], [0.0; 3], [1.0; 3])];
        let b = vec![pose([1.0, 2.0, 3.0], [0.0; 3], [1.0; 3])];
        let out = blend_many(&[a, b.clone()], &[0.0, 1.0]);
        assert_eq!(out, b);
    }

    #[test]
    fn fold_is_bit_identical_to_the_reference_blend() {
        // Deterministic pseudo-random pose sets, folded both ways: the
        // in-place accumulator must reproduce the Vec-returning reference
        // exactly (same operations in the same order), so the rewrite cannot
        // have drifted the animation output.
        let mut seed = 0x12345678u32;
        let mut next = move || {
            seed = seed.wrapping_mul(1664525).wrapping_add(1013904223);
            (seed >> 8) as f32 / (1 << 24) as f32
        };
        for case in 0..20 {
            let n_poses = 1 + case % 4;
            let n_joints = 1 + case % 5;
            let poses: Vec<Vec<Mat4>> = (0..n_poses)
                .map(|_| {
                    (0..n_joints)
                        .map(|_| {
                            pose(
                                [next() * 4.0, next() * 4.0, next() * 4.0],
                                [next() * 180.0 - 90.0, next() * 180.0 - 90.0, 0.0],
                                [next() + 0.5, next() + 0.5, next() + 0.5],
                            )
                        })
                        .collect()
                })
                .collect();
            let weights: Vec<f32> = (0..n_poses).map(|_| next() * 2.0 - 0.25).collect();
            assert_eq!(
                blend_many(&poses, &weights),
                blend_many_reference(&poses, &weights),
                "case {case} diverged"
            );
        }
    }

    #[test]
    fn fold_reuses_a_warm_accumulator_without_reallocating() {
        let poses = [vec![IDENTITY; 8], vec![IDENTITY; 8]];
        let mut acc = Vec::new();
        for pass in 0..3 {
            let ptr = acc.as_ptr();
            let mut fold = PoseBlend::new(&mut acc);
            fold.add(&poses[0], 1.0);
            fold.add(&poses[1], 0.5);
            if pass > 0 {
                assert_eq!(acc.as_ptr(), ptr, "warm accumulator was reallocated");
            }
        }
    }
}
