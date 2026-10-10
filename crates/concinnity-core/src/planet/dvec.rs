// Double-precision vectors and rotations for positions on a planet, where an
// f32 runs out of digits long before the planet does.

/// A double-precision 3-vector.
pub type DVec3 = [f64; 3];

/// A column-major double-precision 3x3 matrix, `m[col][row]`.
pub type DMat3 = [[f64; 3]; 3];

pub(crate) const DMAT3_IDENTITY: DMat3 = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]];

pub(crate) fn add(a: DVec3, b: DVec3) -> DVec3 {
    [a[0] + b[0], a[1] + b[1], a[2] + b[2]]
}

pub(crate) fn sub(a: DVec3, b: DVec3) -> DVec3 {
    [a[0] - b[0], a[1] - b[1], a[2] - b[2]]
}

pub(crate) fn scale(v: DVec3, s: f64) -> DVec3 {
    [v[0] * s, v[1] * s, v[2] * s]
}

pub(crate) fn dot(a: DVec3, b: DVec3) -> f64 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}

pub(crate) fn cross(a: DVec3, b: DVec3) -> DVec3 {
    [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ]
}

pub(crate) fn length(v: DVec3) -> f64 {
    libm::sqrt(dot(v, v))
}

// `v` scaled to unit length, or `fallback` when it has no direction.
pub(crate) fn normalize_or(v: DVec3, fallback: DVec3) -> DVec3 {
    let len = length(v);
    if len > 1e-300 && len.is_finite() {
        scale(v, 1.0 / len)
    } else {
        fallback
    }
}

pub(crate) fn from_f32(v: [f32; 3]) -> DVec3 {
    [f64::from(v[0]), f64::from(v[1]), f64::from(v[2])]
}

pub(crate) fn to_f32(v: DVec3) -> [f32; 3] {
    [v[0] as f32, v[1] as f32, v[2] as f32]
}

// `m * v`.
pub(crate) fn mul_vec(m: &DMat3, v: DVec3) -> DVec3 {
    core::array::from_fn(|i| m[0][i] * v[0] + m[1][i] * v[1] + m[2][i] * v[2])
}

// `transpose(m) * v`: the inverse rotation of an orthonormal `m`.
pub(crate) fn mul_transpose_vec(m: &DMat3, v: DVec3) -> DVec3 {
    [dot(m[0], v), dot(m[1], v), dot(m[2], v)]
}

// `a * b`.
pub(crate) fn mul(a: &DMat3, b: &DMat3) -> DMat3 {
    [mul_vec(a, b[0]), mul_vec(a, b[1]), mul_vec(a, b[2])]
}

// The columns of `m` made orthonormal again (Gram-Schmidt, `y` first, so the
// column a frame's up rides on drifts least), keeping right-handedness.
pub(crate) fn orthonormalize(m: &DMat3) -> DMat3 {
    let y = normalize_or(m[1], [0.0, 1.0, 0.0]);
    let z = normalize_or(sub(m[2], scale(y, dot(m[2], y))), [0.0, 0.0, 1.0]);
    let x = cross(y, z);
    [x, y, z]
}

// The rotation taking `+Y` onto the unit vector `to` along the shortest arc:
// the one that turns nothing about `to` itself. Straight down has no shortest
// arc; it turns half a turn about `+X`.
pub(crate) fn arc_from_up(to: DVec3) -> DMat3 {
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

// The unit quaternion `[x, y, z, w]` of the rotation `m`.
pub(crate) fn quat_of(m: &DMat3) -> [f32; 4] {
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
    q.map(|c| c as f32)
}

#[cfg(test)]
mod tests {
    use super::*;

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
        let q = quat_of(&r);
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
}
