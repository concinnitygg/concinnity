//! Double-precision rotations for positions on a planet, beside the vectors in
//! `math::dvec3`.

use crate::math::dvec3::DVec3;
use crate::math::dvec3::{cross, dot, normalize_or, scale, sub};
use crate::math::{Quat, narrow_f32};

/// A column-major double-precision 3x3 matrix, `m[col][row]`.
pub type DMat3 = [[f64; 3]; 3];

/// The identity rotation.
pub const DMAT3_IDENTITY: DMat3 = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]];

/// `m * v`.
pub fn mul_vec(m: &DMat3, v: DVec3) -> DVec3 {
    core::array::from_fn(|i| m[0][i] * v[0] + m[1][i] * v[1] + m[2][i] * v[2])
}

/// `transpose(m) * v`: the inverse rotation of an orthonormal `m`.
pub fn mul_transpose_vec(m: &DMat3, v: DVec3) -> DVec3 {
    [dot(m[0], v), dot(m[1], v), dot(m[2], v)]
}

/// `m` with every entry times `s`.
pub fn scale_mat(m: &DMat3, s: f64) -> DMat3 {
    m.map(|c| scale(c, s))
}

/// `a * b`.
pub fn mul(a: &DMat3, b: &DMat3) -> DMat3 {
    [mul_vec(a, b[0]), mul_vec(a, b[1]), mul_vec(a, b[2])]
}

/// The columns of `m` made orthonormal again (Gram-Schmidt, `y` first, so the
/// column a frame's up rides on drifts least), keeping right-handedness.
pub fn orthonormalize(m: &DMat3) -> DMat3 {
    let y = normalize_or(m[1], [0.0, 1.0, 0.0]);
    let z = normalize_or(sub(m[2], scale(y, dot(m[2], y))), [0.0, 0.0, 1.0]);
    let x = cross(y, z);
    [x, y, z]
}

/// The rotation taking `+Y` onto the unit vector `to` along the shortest arc:
/// the one that turns nothing about `to` itself. Straight down has no shortest
/// arc; it turns half a turn about `+X`.
pub fn arc_from_up(to: DVec3) -> DMat3 {
    let c = to[1];
    if c <= -1.0 + 1e-12 {
        return [[1.0, 0.0, 0.0], [0.0, -1.0, 0.0], [0.0, 0.0, -1.0]];
    }
    // Rodrigues with k = Y x to = (to.z, 0, -to.x): R = I + [k] + [k]^2 / (1 + c).
    let (kx, kz) = (to[2], -to[0]);
    let f = 1.0 / (1.0 + c);
    [
        [1.0 - kz * kz * f, kz, kx * kz * f],
        [-kz, c, kx],
        [kx * kz * f, -kx, 1.0 - kx * kx * f],
    ]
}

/// The rotation the unit quaternion `[x, y, z, w]` describes.
pub fn mat_from_quat(q: Quat) -> DMat3 {
    let [x, y, z, w] = q.map(f64::from);
    let n = libm::sqrt(x * x + y * y + z * z + w * w);
    let [x, y, z, w] = if n >= 1e-12 {
        [x / n, y / n, z / n, w / n]
    } else {
        [0.0, 0.0, 0.0, 1.0]
    };
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

/// The unit quaternion `[x, y, z, w]` of the rotation `m`.
pub fn quat_from_mat(m: &DMat3) -> Quat {
    let (m00, m11, m22) = (m[0][0], m[1][1], m[2][2]);
    let trace = m00 + m11 + m22;
    let q = if trace > 0.0 {
        let s = libm::sqrt(trace + 1.0) * 2.0;
        [
            (m[1][2] - m[2][1]) / s,
            (m[2][0] - m[0][2]) / s,
            (m[0][1] - m[1][0]) / s,
            0.25 * s,
        ]
    } else if m00 > m11 && m00 > m22 {
        let s = libm::sqrt(1.0 + m00 - m11 - m22) * 2.0;
        [
            0.25 * s,
            (m[1][0] + m[0][1]) / s,
            (m[2][0] + m[0][2]) / s,
            (m[1][2] - m[2][1]) / s,
        ]
    } else if m11 > m22 {
        let s = libm::sqrt(1.0 + m11 - m00 - m22) * 2.0;
        [
            (m[1][0] + m[0][1]) / s,
            0.25 * s,
            (m[2][1] + m[1][2]) / s,
            (m[2][0] - m[0][2]) / s,
        ]
    } else {
        let s = libm::sqrt(1.0 + m22 - m00 - m11) * 2.0;
        [
            (m[2][0] + m[0][2]) / s,
            (m[2][1] + m[1][2]) / s,
            0.25 * s,
            (m[0][1] - m[1][0]) / s,
        ]
    };
    q.map(narrow_f32)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::math::dvec3::{length, to_f32};
    use crate::math::quat_from_axis_angle;

    fn close(a: DVec3, b: DVec3, tol: f64) -> bool {
        (0..3).all(|i| (a[i] - b[i]).abs() < tol)
    }

    #[test]
    fn the_arc_from_up_lands_on_its_target_and_stays_a_rotation() {
        for to in [
            [0.0, 1.0, 0.0],
            [0.6, 0.8, 0.0],
            [0.0, 0.0, 1.0],
            [-0.3, -0.5, 0.812],
            [0.0, -1.0, 0.0],
        ] {
            let to = normalize_or(to, [0.0, 1.0, 0.0]);
            let r = arc_from_up(to);
            assert!(close(mul_vec(&r, [0.0, 1.0, 0.0]), to, 1e-12), "{to:?}");
            for (i, col) in r.iter().enumerate() {
                assert!((length(*col) - 1.0).abs() < 1e-12);
                for other in &r[i + 1..] {
                    assert!(dot(*col, *other).abs() < 1e-12);
                }
            }
            assert!(close(cross(r[0], r[1]), r[2], 1e-12), "right-handed");
        }
    }

    // The shortest arc turns nothing about the axis it leans along: a vector
    // perpendicular to both up and the target stays put.
    #[test]
    fn the_arc_from_up_does_not_spin_about_its_target() {
        let to = normalize_or([0.2, 0.9, 0.0], [0.0, 1.0, 0.0]);
        let r = arc_from_up(to);
        assert!(close(mul_vec(&r, [0.0, 0.0, 1.0]), [0.0, 0.0, 1.0], 1e-12));
    }

    #[test]
    fn orthonormalize_repairs_a_drifted_basis() {
        let drifted = [[1.0, 1e-6, 0.0], [0.0, 1.0, 2e-6], [3e-6, 0.0, 1.0]];
        let m = orthonormalize(&drifted);
        assert!(close(cross(m[0], m[1]), m[2], 1e-12));
        assert!((length(m[0]) - 1.0).abs() < 1e-12);
        assert!(dot(m[0], m[2]).abs() < 1e-12);
    }

    #[test]
    fn the_quaternion_rotates_like_its_matrix() {
        let r = arc_from_up(normalize_or([0.4, 0.7, -0.3], [0.0, 1.0, 0.0]));
        let q = quat_from_mat(&r);
        let v = [0.25, -1.0, 0.5];
        let by_q = crate::math::quat_rotate(q, to_f32(v));
        let by_m = to_f32(mul_vec(&r, v));
        assert!((0..3).all(|i| (by_q[i] - by_m[i]).abs() < 1e-6));
    }

    #[test]
    fn transpose_multiply_inverts_a_rotation() {
        let r = arc_from_up(normalize_or([0.1, 0.2, 0.9], [0.0, 1.0, 0.0]));
        let v = [3.0, -2.0, 7.5];
        assert!(close(mul_transpose_vec(&r, mul_vec(&r, v)), v, 1e-12));
        assert!(close(mul(&r, &DMAT3_IDENTITY)[2], r[2], 1e-15));
    }

    #[test]
    fn a_matrix_round_trips_through_its_quaternion() {
        let r = arc_from_up(normalize_or([-0.5, 0.3, 0.8], [0.0, 1.0, 0.0]));
        let back = mat_from_quat(quat_from_mat(&r));
        for (a, b) in r.iter().zip(&back) {
            assert!(close(*a, *b, 1e-6), "{a:?} vs {b:?}");
        }
    }

    #[test]
    fn a_quaternion_is_normalized_and_a_null_one_is_the_identity() {
        let doubled = quat_from_axis_angle([0.0, 1.0, 0.0], 0.7).map(|c| c * 2.0);
        let m = mat_from_quat(doubled);
        assert!((length(m[0]) - 1.0).abs() < 1e-6);
        assert_eq!(mat_from_quat([0.0; 4]), DMAT3_IDENTITY);
    }

    #[test]
    fn scale_mat_scales_every_entry() {
        let m = scale_mat(&DMAT3_IDENTITY, 2.5);
        assert_eq!(m, [[2.5, 0.0, 0.0], [0.0, 2.5, 0.0], [0.0, 0.0, 2.5]]);
    }
}
