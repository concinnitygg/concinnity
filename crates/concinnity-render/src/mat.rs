//! The view and orthographic-projection builders the shadow passes share
//! (`csm.rs` for the directional cascades, `spot_shadow.rs` for the spot slices).
//! Right-handed with depth mapped to [0, 1], matching
//! [`concinnity_core::gfx::projection`], so the matrices built here are valid for
//! every backend's shadow sampling.

use concinnity_core::math::sqrt;
use concinnity_core::math::vec3::{cross, dot, sub};

/// The 4x4 identity, column-major.
pub const IDENTITY4: [[f32; 4]; 4] = [
    [1.0, 0.0, 0.0, 0.0],
    [0.0, 1.0, 0.0, 0.0],
    [0.0, 0.0, 1.0, 0.0],
    [0.0, 0.0, 0.0, 1.0],
];

pub(crate) fn look_at(eye: [f32; 3], centre: [f32; 3], up: [f32; 3]) -> [[f32; 4]; 4] {
    let f = normalize3(sub(centre, eye));
    let r = normalize3(cross(f, up));
    let u = cross(r, f);
    [
        [r[0], u[0], -f[0], 0.0],
        [r[1], u[1], -f[1], 0.0],
        [r[2], u[2], -f[2], 0.0],
        [-dot(r, eye), -dot(u, eye), dot(f, eye), 1.0],
    ]
}

// Right-handed orthographic projection with depth mapped to [0, 1].
pub(crate) fn ortho_rh(
    left: f32,
    right: f32,
    bottom: f32,
    top: f32,
    near: f32,
    far: f32,
) -> [[f32; 4]; 4] {
    let rml = right - left;
    let tmb = top - bottom;
    let fmn = far - near;
    [
        [2.0 / rml, 0.0, 0.0, 0.0],
        [0.0, 2.0 / tmb, 0.0, 0.0],
        [0.0, 0.0, -1.0 / fmn, 0.0],
        [
            -(right + left) / rml,
            -(top + bottom) / tmb,
            -near / fmn,
            1.0,
        ],
    ]
}

// Unit-length `v`, with the length floored so a degenerate input yields a huge
// but finite vector rather than NaNs in the shadow basis.
pub(crate) fn normalize3(v: [f32; 3]) -> [f32; 3] {
    let len = sqrt(dot(v, v)).max(1e-6);
    [v[0] / len, v[1] / len, v[2] / len]
}

// An axis not parallel to `dir`, for building a look-at basis. Cone axes are
// commonly straight up or down, where the usual +Y up vector is degenerate.
pub(crate) fn up_for(dir: [f32; 3]) -> [f32; 3] {
    if dir[1].abs() > 0.99 {
        [0.0, 0.0, 1.0]
    } else {
        [0.0, 1.0, 0.0]
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn transform(m: [[f32; 4]; 4], p: [f32; 3]) -> [f32; 4] {
        let mut out = [0.0_f32; 4];
        for row in 0..4 {
            out[row] = m[0][row] * p[0] + m[1][row] * p[1] + m[2][row] * p[2] + m[3][row];
        }
        out
    }

    #[test]
    fn look_at_puts_the_eye_at_the_origin_looking_down_negative_z() {
        let v = look_at([0.0, 5.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 1.0]);
        let eye = transform(v, [0.0, 5.0, 0.0]);
        assert!(eye[0].abs() < 1e-5 && eye[1].abs() < 1e-5 && eye[2].abs() < 1e-5);
        // The target sits 5 units ahead, i.e. at -5 on the view Z axis.
        let target = transform(v, [0.0, 0.0, 0.0]);
        assert!((target[2] + 5.0).abs() < 1e-5);
    }

    // A straight-up or straight-down axis must not pick a parallel up vector,
    // or the look-at basis collapses.
    #[test]
    fn up_for_avoids_a_degenerate_basis() {
        assert_eq!(up_for([0.0, -1.0, 0.0]), [0.0, 0.0, 1.0]);
        assert_eq!(up_for([0.0, 1.0, 0.0]), [0.0, 0.0, 1.0]);
        assert_eq!(up_for([1.0, 0.0, 0.0]), [0.0, 1.0, 0.0]);
        // The chosen up is never parallel to the axis.
        for dir in [[0.0, -1.0, 0.0], [0.3, -0.9, 0.2], [1.0, 0.0, 0.0]] {
            let d = normalize3(dir);
            assert!(dot(d, up_for(d)).abs() < 0.999);
        }
    }
}
