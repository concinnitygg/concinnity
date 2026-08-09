// src/gfx/skinning.rs
//
// Pose blending: combining several sampled poses into one set of local joint
// matrices. The skeleton, clip, and transform types these operate on live in
// concinnity-core and are re-exported here under their historical paths.

pub use concinnity_core::gfx::skeleton::{
    AnimationClip, Joint, JointPose, JointTrack, Keyframe, Skeleton,
};
pub use concinnity_core::gfx::transform::{
    IDENTITY, Mat4, blend_matrices, decompose, euler_yxz_from_quat, mat4_affine_inverse, mat4_mul,
    trs_matrix,
};

// Blend two arrays of local joint matrices by weight `f`, clamped to `[0, 1]`:
// `f = 0` returns `a`, `f = 1` returns `b`. Each joint pair is interpolated in
// TRS space (see `blend_matrices`), the same interpolation a single clip uses
// between keyframes, so a blended pose is continuous with a clip's own
// sampling. Arrays of unequal length blend the common prefix and copy the
// longer array's tail through unchanged.
pub fn blend_locals(a: &[Mat4], b: &[Mat4], f: f32) -> Vec<Mat4> {
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

// Blend N arrays of local joint matrices into a single normalised weighted
// average. Implemented as an incremental normalised fold of `blend_locals`:
// after folding in array `i` the accumulator equals the weighted blend of
// arrays `0..=i`. Negative weights clamp to 0 and a 0-weight array is skipped;
// when every weight is 0 (or only one array is given) the first array is
// returned unchanged, so a single-clip mesh is unaffected. An empty input
// yields an empty result.
pub fn blend_many(poses: &[Vec<Mat4>], weights: &[f32]) -> Vec<Mat4> {
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
        acc = blend_locals(&acc, pose, f);
        acc_w = total;
    }
    acc
}

#[cfg(test)]
mod tests {
    use super::*;

    fn approx(a: f32, b: f32) -> bool {
        (a - b).abs() < 1e-4
    }

    #[test]
    fn blend_locals_endpoints_are_exact() {
        // f=0 must equal a, f=1 must equal b: a cross-fade is continuous with
        // each source clip at its extremes.
        let a = vec![
            JointPose {
                translation: [1.0, 0.0, 0.0],
                rotation_deg: [10.0, 20.0, 30.0],
                scale: [1.0, 2.0, 1.0],
            }
            .to_matrix(),
        ];
        let b = vec![
            JointPose {
                translation: [0.0, 4.0, -1.0],
                rotation_deg: [-50.0, 70.0, 5.0],
                scale: [2.0, 1.0, 0.5],
            }
            .to_matrix(),
        ];
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
        let a = vec![
            JointPose {
                rotation_deg: [0.0, 0.0, 0.0],
                ..JointPose::default()
            }
            .to_matrix(),
        ];
        let b = vec![
            JointPose {
                rotation_deg: [0.0, 90.0, 0.0],
                ..JointPose::default()
            }
            .to_matrix(),
        ];
        let mid = blend_locals(&a, &b, 0.5);
        // For a pure yaw the first column is (cos, 0, -sin).
        let yaw = (-mid[0][0][2]).atan2(mid[0][0][0]).to_degrees();
        assert!(approx(yaw, 45.0), "yaw was {}", yaw);
    }

    #[test]
    fn blend_locals_unequal_lengths_keep_the_longer_tail() {
        let a = vec![IDENTITY];
        let b = vec![IDENTITY, IDENTITY, IDENTITY];
        let out = blend_locals(&a, &b, 0.5);
        assert_eq!(out.len(), 3);
    }

    #[test]
    fn blend_many_normalises_weights() {
        // Three pure-yaw poses at 0/30/90 deg blended 1:1:2 must land on the
        // weighted-average yaw 52.5 deg. Equal scaling of every weight (the
        // normalisation) must not move the result.
        let yaw = |deg: f32| {
            vec![
                JointPose {
                    rotation_deg: [0.0, deg, 0.0],
                    ..JointPose::default()
                }
                .to_matrix(),
            ]
        };
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
        let a = vec![
            JointPose {
                translation: [9.0, 9.0, 9.0],
                ..JointPose::default()
            }
            .to_matrix(),
        ];
        let b = vec![
            JointPose {
                translation: [1.0, 2.0, 3.0],
                ..JointPose::default()
            }
            .to_matrix(),
        ];
        let out = blend_many(&[a, b.clone()], &[0.0, 1.0]);
        assert_eq!(out, b);
    }
}
