//! The engine's transform convention, in one place: the column-major 4x4 layout
//! every renderer uniform is written in, the multiply and the two inverses over
//! it, the `T * R(YXZ) * S` composition joints and props both build their matrix
//! through, and the quaternion conversions that let a rotation be interpolated
//! along the shorter arc rather than component-wise through its Euler angles.

use crate::math::{acos, sin, sin_cos, sqrt};

/// Column-major 4x4 matrix, `m[col][row]`: the layout shared by every renderer
/// uniform in this codebase.
pub type Mat4 = [[f32; 4]; 4];

/// Column-major 4x4 identity.
pub const IDENTITY: Mat4 = [
    [1.0, 0.0, 0.0, 0.0],
    [0.0, 1.0, 0.0, 0.0],
    [0.0, 0.0, 1.0, 0.0],
    [0.0, 0.0, 0.0, 1.0],
];

/// Column-major 4x4 multiply: `a * b`.
pub fn mat4_mul(a: Mat4, b: Mat4) -> Mat4 {
    let mut out = [[0.0f32; 4]; 4];
    for col in 0..4 {
        for row in 0..4 {
            for k in 0..4 {
                out[col][row] += a[k][row] * b[col][k];
            }
        }
    }
    out
}

/// Inverse of an affine matrix whose bottom row is `[0, 0, 0, 1]`. The upper
/// 3x3 is inverted via the adjugate; the translation is mapped through it.
/// A near-singular upper 3x3 (degenerate scale) falls back to identity rather
/// than producing NaNs.
pub fn mat4_affine_inverse(m: Mat4) -> Mat4 {
    // Upper-left 3x3, addressed as a[col][row].
    let a = m;
    let det = a[0][0] * (a[1][1] * a[2][2] - a[2][1] * a[1][2])
        - a[1][0] * (a[0][1] * a[2][2] - a[2][1] * a[0][2])
        + a[2][0] * (a[0][1] * a[1][2] - a[1][1] * a[0][2]);
    if det.abs() < 1e-12 {
        return IDENTITY;
    }
    let inv_det = 1.0 / det;
    // Inverse 3x3 (cofactor transpose * 1/det), again as inv[col][row].
    let mut inv = [[0.0f32; 4]; 4];
    inv[0][0] = (a[1][1] * a[2][2] - a[2][1] * a[1][2]) * inv_det;
    inv[1][0] = -(a[1][0] * a[2][2] - a[2][0] * a[1][2]) * inv_det;
    inv[2][0] = (a[1][0] * a[2][1] - a[2][0] * a[1][1]) * inv_det;
    inv[0][1] = -(a[0][1] * a[2][2] - a[2][1] * a[0][2]) * inv_det;
    inv[1][1] = (a[0][0] * a[2][2] - a[2][0] * a[0][2]) * inv_det;
    inv[2][1] = -(a[0][0] * a[2][1] - a[2][0] * a[0][1]) * inv_det;
    inv[0][2] = (a[0][1] * a[1][2] - a[1][1] * a[0][2]) * inv_det;
    inv[1][2] = -(a[0][0] * a[1][2] - a[1][0] * a[0][2]) * inv_det;
    inv[2][2] = (a[0][0] * a[1][1] - a[1][0] * a[0][1]) * inv_det;
    // Inverse translation: -inv3x3 * t.
    let t = [m[3][0], m[3][1], m[3][2]];
    inv[3][0] = -(inv[0][0] * t[0] + inv[1][0] * t[1] + inv[2][0] * t[2]);
    inv[3][1] = -(inv[0][1] * t[0] + inv[1][1] * t[1] + inv[2][1] * t[2]);
    inv[3][2] = -(inv[0][2] * t[0] + inv[1][2] * t[1] + inv[2][2] * t[2]);
    inv[3][3] = 1.0;
    inv
}

/// Inverse of a general 4x4 matrix, by cofactor expansion. Returns
/// [`IDENTITY`] when the determinant is singular or non-finite, so a degenerate
/// input yields a usable matrix instead of spreading NaNs through the pass that
/// asked. Unlike [`mat4_affine_inverse`] this handles a projection's bottom row,
/// which is what the screen-space passes need to invert a view-projection and
/// rebuild world position from depth.
pub fn mat4_inverse(m: Mat4) -> Mat4 {
    // Named row-major (aRC = row R, column C) for the standard cofactor layout,
    // read out of the column-major input and re-emitted column-major below.
    let a00 = m[0][0];
    let a01 = m[1][0];
    let a02 = m[2][0];
    let a03 = m[3][0];
    let a10 = m[0][1];
    let a11 = m[1][1];
    let a12 = m[2][1];
    let a13 = m[3][1];
    let a20 = m[0][2];
    let a21 = m[1][2];
    let a22 = m[2][2];
    let a23 = m[3][2];
    let a30 = m[0][3];
    let a31 = m[1][3];
    let a32 = m[2][3];
    let a33 = m[3][3];

    let b00 = a00 * a11 - a01 * a10;
    let b01 = a00 * a12 - a02 * a10;
    let b02 = a00 * a13 - a03 * a10;
    let b03 = a01 * a12 - a02 * a11;
    let b04 = a01 * a13 - a03 * a11;
    let b05 = a02 * a13 - a03 * a12;
    let b06 = a20 * a31 - a21 * a30;
    let b07 = a20 * a32 - a22 * a30;
    let b08 = a20 * a33 - a23 * a30;
    let b09 = a21 * a32 - a22 * a31;
    let b10 = a21 * a33 - a23 * a31;
    let b11 = a22 * a33 - a23 * a32;

    let det = b00 * b11 - b01 * b10 + b02 * b09 + b03 * b08 - b04 * b07 + b05 * b06;
    if det.abs() < 1e-20 || !det.is_finite() {
        return IDENTITY;
    }
    let inv_det = 1.0 / det;

    let i00 = (a11 * b11 - a12 * b10 + a13 * b09) * inv_det;
    let i01 = (-a01 * b11 + a02 * b10 - a03 * b09) * inv_det;
    let i02 = (a31 * b05 - a32 * b04 + a33 * b03) * inv_det;
    let i03 = (-a21 * b05 + a22 * b04 - a23 * b03) * inv_det;
    let i10 = (-a10 * b11 + a12 * b08 - a13 * b07) * inv_det;
    let i11 = (a00 * b11 - a02 * b08 + a03 * b07) * inv_det;
    let i12 = (-a30 * b05 + a32 * b02 - a33 * b01) * inv_det;
    let i13 = (a20 * b05 - a22 * b02 + a23 * b01) * inv_det;
    let i20 = (a10 * b10 - a11 * b08 + a13 * b06) * inv_det;
    let i21 = (-a00 * b10 + a01 * b08 - a03 * b06) * inv_det;
    let i22 = (a30 * b04 - a31 * b02 + a33 * b00) * inv_det;
    let i23 = (-a20 * b04 + a21 * b02 - a23 * b00) * inv_det;
    let i30 = (-a10 * b09 + a11 * b07 - a12 * b06) * inv_det;
    let i31 = (a00 * b09 - a01 * b07 + a02 * b06) * inv_det;
    let i32 = (-a30 * b03 + a31 * b01 - a32 * b00) * inv_det;
    let i33 = (a20 * b03 - a21 * b01 + a22 * b00) * inv_det;

    [
        [i00, i10, i20, i30],
        [i01, i11, i21, i31],
        [i02, i12, i22, i32],
        [i03, i13, i23, i33],
    ]
}

/// Column-major 3x3 rotation matrix, `m[col][row]`.
pub type Mat3 = [[f32; 3]; 3];

pub(crate) use crate::math::Quat;

// Column-major 3x3 rotation matrix from YXZ Euler degrees. Identical trig to
// [`JointPose::to_matrix`](crate::gfx::skeleton::JointPose::to_matrix),
// without the scale or translation.
pub(crate) fn rotation_mat3(rotation_deg: [f32; 3]) -> Mat3 {
    let [pitch, yaw, roll] = rotation_deg;
    let (sp, cp) = sin_cos(pitch.to_radians());
    let (syw, cyw) = sin_cos(yaw.to_radians());
    let (sr, cr) = sin_cos(roll.to_radians());
    [
        [cyw * cr + syw * sp * sr, cp * sr, -syw * cr + cyw * sp * sr],
        [-cyw * sr + syw * sp * cr, cp * cr, syw * sr + cyw * sp * cr],
        [syw * cp, -sp, cyw * cp],
    ]
}

/// Compose a column-major `T * R * S` affine matrix from a rotation 3x3,
/// per-axis scale, and translation.
pub fn compose(r: Mat3, scale: [f32; 3], t: [f32; 3]) -> Mat4 {
    let [sx, sy, sz] = scale;
    [
        [r[0][0] * sx, r[0][1] * sx, r[0][2] * sx, 0.0],
        [r[1][0] * sy, r[1][1] * sy, r[1][2] * sy, 0.0],
        [r[2][0] * sz, r[2][1] * sz, r[2][2] * sz, 0.0],
        [t[0], t[1], t[2], 1.0],
    ]
}

/// Column-major `T * R(YXZ) * S` model matrix. The single home of the engine's
/// transform convention: joints, props, and runtime transforms all build their
/// matrix here so they compose consistently.
pub fn trs_matrix(position: [f32; 3], rotation_deg: [f32; 3], scale: [f32; 3]) -> Mat4 {
    compose(rotation_mat3(rotation_deg), scale, position)
}

// Quaternion of a column-major rotation 3x3 (Shepperd's method: picks the
// largest-magnitude component to keep the division well-conditioned).
pub(crate) fn quat_from_mat3(m: Mat3) -> Quat {
    let (m00, m11, m22) = (m[0][0], m[1][1], m[2][2]);
    let trace = m00 + m11 + m22;
    if trace > 0.0 {
        let s = sqrt(trace + 1.0) * 2.0;
        [
            (m[1][2] - m[2][1]) / s,
            (m[2][0] - m[0][2]) / s,
            (m[0][1] - m[1][0]) / s,
            0.25 * s,
        ]
    } else if m00 > m11 && m00 > m22 {
        let s = sqrt(1.0 + m00 - m11 - m22) * 2.0;
        [
            0.25 * s,
            (m[1][0] + m[0][1]) / s,
            (m[2][0] + m[0][2]) / s,
            (m[1][2] - m[2][1]) / s,
        ]
    } else if m11 > m22 {
        let s = sqrt(1.0 + m11 - m00 - m22) * 2.0;
        [
            (m[1][0] + m[0][1]) / s,
            0.25 * s,
            (m[2][1] + m[1][2]) / s,
            (m[2][0] - m[0][2]) / s,
        ]
    } else {
        let s = sqrt(1.0 + m22 - m00 - m11) * 2.0;
        [
            (m[2][0] + m[0][2]) / s,
            (m[2][1] + m[1][2]) / s,
            0.25 * s,
            (m[0][1] - m[1][0]) / s,
        ]
    }
}

// Column-major rotation 3x3 of a unit quaternion.
pub(crate) fn quat_to_mat3(q: Quat) -> Mat3 {
    let [x, y, z, w] = q;
    [
        [
            1.0 - 2.0 * (y * y + z * z),
            2.0 * (x * y + w * z),
            2.0 * (x * z - w * y),
        ],
        [
            2.0 * (x * y - w * z),
            1.0 - 2.0 * (x * x + z * z),
            2.0 * (y * z + w * x),
        ],
        [
            2.0 * (x * z + w * y),
            2.0 * (y * z - w * x),
            1.0 - 2.0 * (x * x + y * y),
        ],
    ]
}

// Scale a quaternion to unit length, falling back to identity when it is too
// short to normalise.

// Spherical linear interpolation between two unit quaternions. Negates `b`
// when the pair points to opposite hemispheres so the interpolation always
// takes the shorter arc, and falls back to a normalised lerp when the two
// rotations are nearly parallel (the slerp denominator approaches zero there
// and nlerp is visually identical at that angle). `f` is clamped to `[0, 1]`.
pub(crate) fn quat_slerp(a: Quat, mut b: Quat, f: f32) -> Quat {
    let f = f.clamp(0.0, 1.0);
    let mut dot = a[0] * b[0] + a[1] * b[1] + a[2] * b[2] + a[3] * b[3];
    if dot < 0.0 {
        b = [-b[0], -b[1], -b[2], -b[3]];
        dot = -dot;
    }
    if dot > 0.9995 {
        return crate::math::quat_normalize([
            a[0] + (b[0] - a[0]) * f,
            a[1] + (b[1] - a[1]) * f,
            a[2] + (b[2] - a[2]) * f,
            a[3] + (b[3] - a[3]) * f,
        ]);
    }
    let theta_0 = acos(dot.clamp(-1.0, 1.0));
    let sin_0 = sin(theta_0);
    let s_a = sin((1.0 - f) * theta_0) / sin_0;
    let s_b = sin(f * theta_0) / sin_0;
    [
        a[0] * s_a + b[0] * s_b,
        a[1] * s_a + b[1] * s_b,
        a[2] * s_a + b[2] * s_b,
        a[3] * s_a + b[3] * s_b,
    ]
}

/// Decompose a column-major affine matrix into translation, a unit rotation
/// quaternion, and per-axis scale: the inverse of [`compose`] for a
/// positive-scale `T * R * S` matrix. Scale is recovered as the length of each
/// rotation column; a zero-length column yields a zero scale axis and the
/// rotation falls back to identity for that axis.
pub fn decompose(m: Mat4) -> ([f32; 3], Quat, [f32; 3]) {
    let t = [m[3][0], m[3][1], m[3][2]];
    let col_len = |c: usize| sqrt(m[c][0] * m[c][0] + m[c][1] * m[c][1] + m[c][2] * m[c][2]);
    let scale = [col_len(0), col_len(1), col_len(2)];
    let norm = |c: usize| {
        let s = scale[c];
        if s < 1e-12 {
            [0.0, 0.0, 0.0]
        } else {
            [m[c][0] / s, m[c][1] / s, m[c][2] / s]
        }
    };
    let r: Mat3 = [norm(0), norm(1), norm(2)];
    (t, crate::math::quat_normalize(quat_from_mat3(r)), scale)
}

/// Interpolate two affine matrices in TRS space by weight `f`, clamped to
/// `[0, 1]`. Translation and scale blend linearly while rotation is
/// quaternion-slerped along the shorter arc, matching what a clip does between
/// keyframes, so a blended pose is continuous with a clip's own sampling.
pub fn blend_matrices(a: Mat4, b: Mat4, f: f32) -> Mat4 {
    let f = f.clamp(0.0, 1.0);
    let (ta, qa, sa) = decompose(a);
    let (tb, qb, sb) = decompose(b);
    let mix = |x: [f32; 3], y: [f32; 3]| {
        [
            x[0] + (y[0] - x[0]) * f,
            x[1] + (y[1] - x[1]) * f,
            x[2] + (y[2] - x[2]) * f,
        ]
    };
    compose(
        quat_to_mat3(quat_slerp(qa, qb, f)),
        mix(sa, sb),
        mix(ta, tb),
    )
}

/// YXZ Euler angles in degrees recovered from a unit rotation quaternion: the
/// inverse of `rotation_mat3` composed with `quat_to_mat3`. glTF stores node
/// rotations as quaternions; the glTF importer converts them to the Euler
/// [`JointPose`](crate::gfx::skeleton::JointPose) representation this engine's
/// joints use. The conversion is matrix-exact for non-degenerate rotations; at
/// gimbal lock (pitch ±90°) it folds the rotation onto the yaw axis with zero
/// roll.
pub fn euler_yxz_from_quat(q: Quat) -> [f32; 3] {
    crate::math::euler_yxz_deg_from_quat(q)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn approx(a: f32, b: f32) -> bool {
        (a - b).abs() < 1e-4
    }

    #[test]
    fn affine_inverse_round_trips() {
        let m = trs_matrix([3.0, -2.0, 5.0], [0.0, 30.0, 0.0], [2.0, 2.0, 2.0]);
        let id = mat4_mul(m, mat4_affine_inverse(m));
        for col in 0..4 {
            for row in 0..4 {
                assert!(approx(id[col][row], IDENTITY[col][row]));
            }
        }
    }

    #[test]
    fn affine_inverse_of_a_degenerate_matrix_falls_back_to_identity() {
        // A zero scale axis leaves the upper 3x3 singular; the fallback keeps
        // NaNs out of the joint matrices rather than propagating them.
        let m = trs_matrix([1.0, 2.0, 3.0], [10.0, 20.0, 30.0], [1.0, 0.0, 1.0]);
        assert_eq!(mat4_affine_inverse(m), IDENTITY);
    }

    #[test]
    fn general_inverse_round_trips_a_projection_and_a_view() {
        // The screen-space passes invert a view-projection, whose bottom row is
        // not [0, 0, 0, 1], so mat4_affine_inverse cannot serve them.
        let proj = crate::gfx::projection::perspective_rh(75.0f32.to_radians(), 1.6, 0.1, 500.0);
        let view: Mat4 = [
            [0.92388, 0.0, -0.38268, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.38268, 0.0, 0.92388, 0.0],
            [-1.5, -0.7, 4.0, 1.0],
        ];
        for m in [proj, view, mat4_mul(proj, view)] {
            for id in [mat4_mul(m, mat4_inverse(m)), mat4_mul(mat4_inverse(m), m)] {
                for col in 0..4 {
                    for row in 0..4 {
                        assert!(
                            (id[col][row] - IDENTITY[col][row]).abs() < 1e-3,
                            "[{col}][{row}]: {}",
                            id[col][row]
                        );
                    }
                }
            }
        }
    }

    // The one place the four copies this replaced disagreed: the planar-reflection
    // copy guarded on determinant magnitude alone, which a NaN slips straight
    // through. Everything else about them was character-identical.
    #[test]
    fn general_inverse_falls_back_on_a_singular_or_non_finite_matrix() {
        // A zero column is singular; the fallback keeps the inverted VP usable
        // rather than filling a screen-space pass's uniforms with NaNs.
        let mut singular = IDENTITY;
        singular[1] = [0.0; 4];
        assert_eq!(mat4_inverse(singular), IDENTITY);
        // A non-finite entry makes the determinant NaN, which no magnitude test
        // catches on its own: `NaN.abs() < 1e-20` is false.
        let mut nan = IDENTITY;
        nan[0][0] = f32::NAN;
        assert_eq!(mat4_inverse(nan), IDENTITY);
        let mut inf = IDENTITY;
        inf[2][2] = f32::INFINITY;
        assert_eq!(mat4_inverse(inf), IDENTITY);
    }

    #[test]
    fn quat_mat3_round_trips() {
        // A rotation 3x3 -> quaternion -> rotation 3x3 must reproduce itself,
        // across the diagonal-dominant and trace-positive branches of
        // Shepperd's method. This is what makes blend_matrices' endpoints exact.
        for e in [
            [0.0, 0.0, 0.0],
            [30.0, 50.0, 20.0],
            [-80.0, 140.0, -25.0],
            [90.0, 0.0, 0.0],
            [0.0, 180.0, 0.0],
        ] {
            let r = rotation_mat3(e);
            let r2 = quat_to_mat3(quat_from_mat3(r));
            for c in 0..3 {
                for row in 0..3 {
                    assert!(
                        approx(r[c][row], r2[c][row]),
                        "e={:?} [{}][{}]: {} vs {}",
                        e,
                        c,
                        row,
                        r[c][row],
                        r2[c][row]
                    );
                }
            }
        }
    }

    #[test]
    fn quat_normalize_falls_back_for_a_zero_quaternion() {
        assert_eq!(crate::math::quat_normalize([0.0; 4]), [0.0, 0.0, 0.0, 1.0]);
        let n = crate::math::quat_normalize([0.0, 0.0, 0.0, 4.0]);
        assert!(approx(n[3], 1.0));
    }

    #[test]
    fn slerp_midpoint_splits_the_arc_equally() {
        // The defining property of slerp: the f=0.5 quaternion is equidistant
        // (equal rotation angle) from both endpoints. A component-wise Euler
        // lerp does not satisfy this for a multi-axis rotation difference.
        let qa = quat_from_mat3(rotation_mat3([10.0, 20.0, 30.0]));
        let qb = quat_from_mat3(rotation_mat3([70.0, -40.0, 80.0]));
        let qm = quat_slerp(qa, qb, 0.5);
        let angle = |x: Quat, y: Quat| {
            let d = (x[0] * y[0] + x[1] * y[1] + x[2] * y[2] + x[3] * y[3])
                .abs()
                .min(1.0);
            2.0 * acos(d)
        };
        assert!(
            approx(angle(qa, qm), angle(qm, qb)),
            "arcs {} vs {}",
            angle(qa, qm),
            angle(qm, qb)
        );
    }

    #[test]
    fn decompose_round_trips_a_composed_matrix() {
        // decompose must invert compose for a positive-scale TRS matrix, so a
        // blend interpolates the same transform a clip sampled.
        let m = trs_matrix([3.0, -2.0, 5.0], [25.0, -60.0, 40.0], [1.5, 0.5, 2.0]);
        let (t, q, s) = decompose(m);
        let rebuilt = compose(quat_to_mat3(q), s, t);
        for c in 0..4 {
            for row in 0..4 {
                assert!(
                    approx(rebuilt[c][row], m[c][row]),
                    "[{}][{}]: {} vs {}",
                    c,
                    row,
                    rebuilt[c][row],
                    m[c][row]
                );
            }
        }
    }

    #[test]
    fn blend_matrices_endpoints_are_exact_and_the_middle_slerps() {
        let a = trs_matrix([1.0, 2.0, 3.0], [10.0, 20.0, 30.0], [1.0, 1.5, 2.0]);
        let b = trs_matrix([-4.0, 0.0, 5.0], [70.0, -40.0, 15.0], [2.0, 1.0, 0.5]);
        for (f, want) in [(0.0, a), (1.0, b)] {
            let got = blend_matrices(a, b, f);
            for c in 0..4 {
                for row in 0..4 {
                    assert!(approx(got[c][row], want[c][row]), "f={f} [{c}][{row}]");
                }
            }
        }
        // The midpoint's rotation is the slerped one, not a matrix lerp: its
        // basis stays orthonormal, which an element-wise average would not.
        let mid = blend_matrices(a, b, 0.5);
        let (_, _, scale) = decompose(mid);
        assert!(approx(scale[0], 1.5) && approx(scale[1], 1.25) && approx(scale[2], 1.25));
        assert!(approx(mid[3][0], -1.5) && approx(mid[3][1], 1.0) && approx(mid[3][2], 4.0));
        // f outside [0, 1] clamps rather than extrapolating.
        assert_eq!(blend_matrices(a, b, -1.0), blend_matrices(a, b, 0.0));
        assert_eq!(blend_matrices(a, b, 2.0), blend_matrices(a, b, 1.0));
    }

    #[test]
    fn euler_from_quat_round_trips_through_the_rotation_matrix() {
        // quat -> YXZ Euler must reproduce the original rotation matrix, so
        // the glTF importer's quaternion node rotations land losslessly in the
        // Euler JointPose representation. Checked across multi-axis rotations.
        for e in [
            [0.0, 0.0, 0.0],
            [25.0, -60.0, 40.0],
            [-80.0, 140.0, -25.0],
            [10.0, 200.0, -170.0],
        ] {
            let r = rotation_mat3(e);
            let q = quat_from_mat3(r);
            let e2 = euler_yxz_from_quat(q);
            let r2 = rotation_mat3(e2);
            for c in 0..3 {
                for row in 0..3 {
                    assert!(
                        approx(r[c][row], r2[c][row]),
                        "e={:?} [{}][{}]: {} vs {}",
                        e,
                        c,
                        row,
                        r[c][row],
                        r2[c][row]
                    );
                }
            }
        }
    }

    #[test]
    fn euler_from_quat_handles_gimbal_lock() {
        // At pitch ±90° the conversion must stay finite and reproduce the
        // rotation matrix (with roll folded onto yaw).
        for e in [[90.0, 35.0, 0.0], [-90.0, -110.0, 0.0]] {
            let r = rotation_mat3(e);
            let e2 = euler_yxz_from_quat(quat_from_mat3(r));
            assert!(e2.iter().all(|v| v.is_finite()), "non-finite for {:?}", e);
            let r2 = rotation_mat3(e2);
            for c in 0..3 {
                for row in 0..3 {
                    assert!(approx(r[c][row], r2[c][row]), "e={:?} [{}][{}]", e, c, row);
                }
            }
        }
    }
}
