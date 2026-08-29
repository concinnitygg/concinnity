//! The engine's view and projection matrices, in the convention every backend
//! already agreed on: right-handed, looking down `-z`, with depth mapped to
//! `[0, 1]` after the perspective divide. Metal, Vulkan, and DirectX all sample
//! depth that way, so the same matrices are valid for all three; Vulkan and D3D12
//! compensate for their Y-down NDC with a negative-height viewport rather than
//! by flipping the projection.
//!
//! Both projections and both ways of building a view matrix sit together because
//! they have to agree: a shadow cascade's ortho and a probe face's perspective
//! are sampled by the same shaders as the main camera's.

use crate::gfx::transform::Mat4;
use crate::math::vec3::{cross, dot, sub};
use crate::math::{sqrt, tan};

// Floor applied to the half-FOV tangent. A zero or near-zero vertical FOV would
// otherwise divide by zero and fill the matrix with infinities.
const MIN_HALF_FOV_TAN: f32 = 1.0e-6;

/// Right-handed perspective projection with depth in `[0, 1]`. `fov_y_radians`
/// is the full vertical field of view; `aspect` is width over height.
pub fn perspective_rh(fov_y_radians: f32, aspect: f32, near: f32, far: f32) -> Mat4 {
    let ys = 1.0 / tan(fov_y_radians * 0.5).max(MIN_HALF_FOV_TAN);
    let xs = ys / aspect;
    let zs = far / (near - far);
    [
        [xs, 0.0, 0.0, 0.0],
        [0.0, ys, 0.0, 0.0],
        [0.0, 0.0, zs, -1.0],
        [0.0, 0.0, zs * near, 0.0],
    ]
}

/// Right-handed orthographic projection with depth in `[0, 1]`.
pub fn ortho_rh(left: f32, right: f32, bottom: f32, top: f32, near: f32, far: f32) -> Mat4 {
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

/// World-to-view for an orthonormal camera basis at `eye`, looking down
/// `-forward`. The basis is taken as given, for a caller that already has one
/// (a cube face, a light frustum) rather than a point to aim at.
pub fn view_from_basis(eye: [f32; 3], right: [f32; 3], up: [f32; 3], forward: [f32; 3]) -> Mat4 {
    [
        [right[0], up[0], -forward[0], 0.0],
        [right[1], up[1], -forward[1], 0.0],
        [right[2], up[2], -forward[2], 0.0],
        [-dot(right, eye), -dot(up, eye), dot(forward, eye), 1.0],
    ]
}

/// World-to-view for a camera at `eye` aimed at `centre`, with `up` resolving
/// the roll.
pub fn look_at(eye: [f32; 3], centre: [f32; 3], up: [f32; 3]) -> Mat4 {
    let f = normalize3(sub(centre, eye));
    let r = normalize3(cross(f, up));
    view_from_basis(eye, r, cross(r, f), f)
}

/// Unit-length `v`, with the length floored so a degenerate input yields a huge
/// but finite vector rather than NaNs in a view basis. Distinct from
/// [`crate::math::vec3::vec3_normalise`], which substitutes a fallback axis
/// instead: a basis wants the direction it was given, however short.
pub fn normalize3(v: [f32; 3]) -> [f32; 3] {
    let len = sqrt(dot(v, v)).max(1e-6);
    [v[0] / len, v[1] / len, v[2] / len]
}

/// An axis not parallel to `dir`, for building a [`look_at`] basis. Cone and
/// cascade axes are commonly straight up or down, where the usual `+Y` up
/// vector is degenerate.
pub fn up_for(dir: [f32; 3]) -> [f32; 3] {
    if dir[1].abs() > 0.99 {
        [0.0, 0.0, 1.0]
    } else {
        [0.0, 1.0, 0.0]
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn transform(m: Mat4, p: [f32; 3]) -> [f32; 4] {
        let mut out = [0.0f32; 4];
        for (row, o) in out.iter_mut().enumerate() {
            *o = m[0][row] * p[0] + m[1][row] * p[1] + m[2][row] * p[2] + m[3][row];
        }
        out
    }

    #[test]
    fn near_and_far_map_to_zero_and_one() {
        let p = perspective_rh(75.0f32.to_radians(), 1.6, 0.1, 500.0);
        let near = transform(p, [0.0, 0.0, -0.1]);
        let far = transform(p, [0.0, 0.0, -500.0]);
        assert!((near[2] / near[3]).abs() < 1e-4);
        assert!((far[2] / far[3] - 1.0).abs() < 1e-4);
    }

    #[test]
    fn the_frustum_edge_lands_on_the_ndc_boundary() {
        // At a 90 degree vertical FOV and square aspect the frustum edge sits
        // at x = -z, so an edge point projects exactly onto NDC x = 1.
        let p = perspective_rh(90.0f32.to_radians(), 1.0, 0.1, 50.0);
        let edge = transform(p, [10.0, 0.0, -10.0]);
        assert!((edge[0] / edge[3] - 1.0).abs() < 1e-4);
    }

    #[test]
    fn a_degenerate_fov_stays_finite() {
        let p = perspective_rh(0.0, 1.0, 0.1, 100.0);
        assert!(p.iter().flatten().all(|v| v.is_finite()));
    }

    // The backends and the shadow builders each reached this matrix their own
    // way before it moved here, through a different arrangement of the same
    // algebra. Reassociating a float product is not free, so pin the gap: fed
    // the same tangent, every element must land within one ulp of the other
    // ordering. Both sides take the shim's tangent rather than std's: std's is
    // the host's libm, which would make the bound a property of the machine
    // running the test rather than of the arrangement. `math::scalar` is where
    // the shim is held to std.
    #[test]
    fn the_reassociated_form_agrees_to_within_one_ulp() {
        fn other(fov_y_radians: f32, aspect: f32, near: f32, far: f32) -> Mat4 {
            let t = tan(fov_y_radians * 0.5).max(MIN_HALF_FOV_TAN);
            let fmn = far - near;
            [
                [1.0 / (aspect * t), 0.0, 0.0, 0.0],
                [0.0, 1.0 / t, 0.0, 0.0],
                [0.0, 0.0, -far / fmn, -1.0],
                [0.0, 0.0, -(far * near) / fmn, 0.0],
            ]
        }
        for fov_deg in [10.0f32, 45.0, 60.0, 75.0, 90.0, 140.0, 179.0] {
            for aspect in [0.5f32, 1.0, 1.6, 2.35, 3.0] {
                for near in [0.01f32, 0.1, 1.0] {
                    for far in [10.0f32, 500.0, 10_000.0] {
                        let fov = fov_deg.to_radians();
                        let a = perspective_rh(fov, aspect, near, far);
                        let b = other(fov, aspect, near, far);
                        for col in 0..4 {
                            for row in 0..4 {
                                let ulps = (i64::from(a[col][row].to_bits())
                                    - i64::from(b[col][row].to_bits()))
                                .abs();
                                assert!(
                                    ulps <= 1,
                                    "fov={fov_deg} aspect={aspect} near={near} far={far} \
                                     [{col}][{row}]: {} vs {} ({ulps} ulps)",
                                    a[col][row],
                                    b[col][row]
                                );
                            }
                        }
                    }
                }
            }
        }
    }

    #[test]
    fn ortho_maps_the_box_onto_the_ndc_cube() {
        let p = ortho_rh(-2.0, 2.0, -1.0, 1.0, 1.0, 11.0);
        let near = transform(p, [0.0, 0.0, -1.0]);
        let far = transform(p, [2.0, 1.0, -11.0]);
        assert!(near[2].abs() < 1e-5, "near depth {}", near[2]);
        assert!((far[2] - 1.0).abs() < 1e-5, "far depth {}", far[2]);
        assert!((far[0] - 1.0).abs() < 1e-5 && (far[1] - 1.0).abs() < 1e-5);
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

    // `look_at` is `view_from_basis` with the basis derived from a target, so
    // handing the derived basis straight over must give the same matrix.
    #[test]
    fn look_at_agrees_with_the_basis_it_derives() {
        let eye = [3.0, -1.5, 2.0];
        let centre = [0.4, 0.9, -2.0];
        let up = [0.0, 1.0, 0.0];
        let f = normalize3(sub(centre, eye));
        let r = normalize3(cross(f, up));
        assert_eq!(
            look_at(eye, centre, up),
            view_from_basis(eye, r, cross(r, f), f)
        );
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

    #[test]
    fn a_degenerate_direction_stays_finite() {
        assert!(normalize3([0.0; 3]).iter().all(|v| v.is_finite()));
    }
}
