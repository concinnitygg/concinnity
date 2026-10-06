//! The engine's view and projection matrices: right-handed, looking down `-z`.
//! Metal, Vulkan, and DirectX share them; Vulkan and D3D12 compensate for their
//! Y-down NDC with a negative-height viewport rather than by flipping the
//! projection.
//!
//! The projections here map near to device depth 0 and far to 1, or, for the
//! `reversed_infinite_` ones, near to 1 and infinity to 0. They are the builders behind
//! [`crate::render::depth`]'s entry points, which pick one per depth
//! convention, so a caller never names a depth mapping directly.
//!
//! The projections and both ways of building a view matrix sit together because
//! they have to agree: a shadow cascade's ortho and a probe face's perspective
//! are sampled by the same shaders as the main camera's.

use crate::math::tan;
use crate::math::vec3::{cross, dot, normalize_clamped, sub};
use crate::transform::Mat4;

// Floor applied to the half-FOV tangent. A zero or near-zero vertical FOV would
// otherwise divide by zero and fill the matrix with infinities.
const MIN_HALF_FOV_TAN: f32 = 1.0e-6;

/// Right-handed perspective projection with depth in `[0, 1]`. `fov_y_radians`
/// is the full vertical field of view; `aspect` is width over height.
pub(crate) fn perspective_rh(fov_y_radians: f32, aspect: f32, near: f32, far: f32) -> Mat4 {
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
pub(crate) fn ortho_rh(left: f32, right: f32, bottom: f32, top: f32, near: f32, far: f32) -> Mat4 {
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

/// Right-handed perspective projection with reversed depth and no far plane:
/// near maps to device depth 1 and depth falls toward 0 as distance goes to
/// infinity, reaching it only there. Its x, y and w rows are
/// [`perspective_rh`]'s.
pub(crate) fn reversed_infinite_perspective_rh(fov_y_radians: f32, aspect: f32, near: f32) -> Mat4 {
    let ys = 1.0 / tan(fov_y_radians * 0.5).max(MIN_HALF_FOV_TAN);
    let xs = ys / aspect;
    [
        [xs, 0.0, 0.0, 0.0],
        [0.0, ys, 0.0, 0.0],
        [0.0, 0.0, 0.0, -1.0],
        [0.0, 0.0, near, 0.0],
    ]
}

/// Oblique near-plane clipping (Lengyel) for a
/// [`reversed_infinite_perspective_rh`] matrix. Replaces the projection's z
/// (depth) row so the near clip plane coincides with `clip_plane` (given in the
/// projection's view space), clipping everything on the negative side of that
/// plane. Depth still reaches 0 only at infinity.
///
/// The near plane is where depth equals w, so the new z-row is
/// `w_row - alpha * C` for the clip plane C (any alpha keeps the near plane at
/// C). Depth must stay non-negative over the whole frustum, and the direction
/// that bounds it is the far frustum corner the plane faces, taken at infinity:
/// the point q = (sgn(Cx) / xs, sgn(Cy) / ys, -1, 0). Requiring depth 0 there
/// gives alpha = (w_row . q) / (C . q) = 1 / (C . q), the limit of the finite
/// derivation as the far plane recedes.
pub(crate) fn reversed_infinite_oblique_rh(proj: Mat4, clip_plane: [f32; 4]) -> Mat4 {
    let xs = proj[0][0];
    let ys = proj[1][1];
    if xs.abs() < 1e-12 || ys.abs() < 1e-12 {
        return proj;
    }

    let sgn = |v: f32| {
        if v > 0.0 {
            1.0
        } else if v < 0.0 {
            -1.0
        } else {
            0.0
        }
    };
    // The far frustum corner toward the clip plane, as a point at infinity.
    let q = [sgn(clip_plane[0]) / xs, sgn(clip_plane[1]) / ys, -1.0];
    let denom = clip_plane[0] * q[0] + clip_plane[1] * q[1] + clip_plane[2] * q[2];
    if denom.abs() < 1e-12 {
        return proj;
    }
    let alpha = 1.0 / denom;

    let mut out = proj;
    // Replace the z (depth) row: row index 2 across all four columns.
    out[0][2] = -alpha * clip_plane[0];
    out[1][2] = -alpha * clip_plane[1];
    out[2][2] = -1.0 - alpha * clip_plane[2];
    out[3][2] = -alpha * clip_plane[3];
    out
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

/// World-to-view for a camera at `eye` aimed at `center`, with `up` resolving
/// the roll.
pub fn look_at(eye: [f32; 3], center: [f32; 3], up: [f32; 3]) -> Mat4 {
    let f = normalize_clamped(sub(center, eye), 1e-6);
    let r = normalize_clamped(cross(f, up), 1e-6);
    view_from_basis(eye, r, cross(r, f), f)
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

    fn depth(m: Mat4, p: [f32; 3]) -> f32 {
        let c = transform(m, p);
        c[2] / c[3]
    }

    fn sgn(v: f32) -> f32 {
        if v == 0.0 { 0.0 } else { v.signum() }
    }

    // The standard-depth infinite projection: [`perspective_rh`] as the far
    // plane recedes, near at ndc 0 and infinity at 1.
    fn standard_infinite_rh(fov_y_radians: f32, aspect: f32, near: f32) -> Mat4 {
        let mut p = perspective_rh(fov_y_radians, aspect, near, 1.0e3);
        p[2][2] = -1.0;
        p[3][2] = -near;
        p
    }

    // Lengyel's infinite derivation for [`standard_infinite_rh`]: near at ndc 0,
    // the far corner at infinity held at ndc 1.
    fn standard_infinite_oblique_rh(proj: Mat4, c: [f32; 4]) -> Mat4 {
        let q = [sgn(c[0]) / proj[0][0], sgn(c[1]) / proj[1][1], -1.0];
        let alpha = 1.0 / (c[0] * q[0] + c[1] * q[1] + c[2] * q[2]);
        let mut out = proj;
        for (col, cv) in c.iter().enumerate() {
            out[col][2] = alpha * cv;
        }
        out
    }

    const CAMERAS: [(f32, f32, f32); 4] = [
        (1.2, 1.6, 0.1),
        (0.4, 0.5, 0.01),
        (1.5, 2.35, 0.5),
        (0.9, 1.0, 1.0),
    ];

    const DISTANCES: [f32; 7] = [1.0, 3.0, 40.0, 900.0, 2.0e4, 1.0e6, 1.0e9];

    // Reversed infinite depth is exactly 1 - standard infinite depth: the near
    // plane at 1, infinity at 0, and every distance between mirrored, under the
    // same x/y/w as the finite projection.
    #[test]
    fn the_reversed_infinite_perspective_mirrors_standard_depth() {
        for (fov, aspect, near) in CAMERAS {
            let finite = perspective_rh(fov, aspect, near, 100.0);
            let standard = standard_infinite_rh(fov, aspect, near);
            let reversed = reversed_infinite_perspective_rh(fov, aspect, near);
            for row in [0, 1, 3] {
                for col in 0..4 {
                    assert_eq!(reversed[col][row], finite[col][row]);
                }
            }
            assert_eq!(depth(reversed, [0.0, 0.0, -near]), 1.0);
            for d in DISTANCES {
                let z = -(near + d);
                let p = [0.3 * z, -0.2 * z, z];
                let (s, r) = (depth(standard, p), depth(reversed, p));
                assert!((r - (1.0 - s)).abs() < 1e-5, "{d}: {r} vs 1 - {s}");
            }
        }
    }

    // The reversed infinite oblique matrix clips at the same plane as Lengyel's
    // standard-Z infinite derivation and keeps the same side, with every depth
    // mirrored: points on the clip plane land on the near plane (1), and the far
    // corner the plane faces reaches the far plane (0) only at infinity. Every
    // other direction inside the frustum stays in front of it.
    #[test]
    fn the_reversed_infinite_oblique_mirrors_the_standard_derivation() {
        // Mirror-like planes ahead of the eye, which sits on the clipped side.
        let planes = [
            [0.0, 0.0, -1.0, -3.0],
            [0.2, 0.1, -1.0, -3.0],
            [-0.3, 0.25, -1.0, -4.0],
            [0.1, -0.3, -1.0, -2.0],
        ];
        for (fov, aspect, near) in CAMERAS {
            let standard = standard_infinite_rh(fov, aspect, near);
            let reversed = reversed_infinite_perspective_rh(fov, aspect, near);
            for c in planes {
                let s = standard_infinite_oblique_rh(standard, c);
                let r = reversed_infinite_oblique_rh(reversed, c);
                for row in [0, 1, 3] {
                    for col in 0..4 {
                        assert_eq!(r[col][row], s[col][row]);
                    }
                }
                let side = |p: [f32; 3]| c[0] * p[0] + c[1] * p[1] + c[2] * p[2] + c[3];
                for d in [near, 2.5, 7.0, 300.0, 1.0e5] {
                    for (fx, fy) in [(0.3, -0.2), (-0.4, 0.4), (0.0, 0.0)] {
                        let p = [-fx * d, -fy * d, -d];
                        let (ds, dr) = (depth(s, p), depth(r, p));
                        let tolerance = 1e-4 * ds.abs().max(1.0);
                        assert!((dr - (1.0 - ds)).abs() < tolerance, "{dr} vs 1 - {ds}");
                        // Same culled side: behind the plane falls past the
                        // near plane under both.
                        if side(p) < 0.0 {
                            assert!(ds < 0.0 && dr > 1.0, "{p:?}: {ds} / {dr}");
                        }
                    }
                }
                // Points on the clip plane sit on the reversed near plane.
                for (a, b) in [(0.1, 0.2), (-0.3, 0.05), (0.25, -0.4)] {
                    let s = -c[3] / (c[0] * a + c[1] * b - c[2]);
                    let on_plane = [s * a, s * b, -s];
                    assert!(side(on_plane).abs() < 1e-4);
                    assert!((depth(r, on_plane) - 1.0).abs() < 1e-4);
                }
                // The far corner the plane faces, as a point at infinity: clip
                // z is 0 there while w stays positive.
                let q = [sgn(c[0]) / standard[0][0], sgn(c[1]) / standard[1][1], -1.0];
                let z = r[0][2] * q[0] + r[1][2] * q[1] + r[2][2] * q[2];
                let w = r[0][3] * q[0] + r[1][3] * q[1] + r[2][3] * q[2];
                assert!(z.abs() < 1e-5 && w > 0.0, "{z} / {w}");
                // Toward that corner and the other three, depth only approaches
                // the far plane.
                for (sx, sy) in [(1.0, 1.0), (1.0, -1.0), (-1.0, 1.0), (-1.0, -1.0)] {
                    let mut prev = 1.0;
                    for d in [1.0e3, 1.0e4, 1.0e5, 1.0e6] {
                        let corner = [sx * d / r[0][0], sy * d / r[1][1], -d];
                        let dr = depth(r, corner);
                        assert!(dr > 0.0 && dr < prev, "{c:?} {sx},{sy} at {d}: {dr}");
                        prev = dr;
                    }
                }
            }
        }
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
        let center = [0.4, 0.9, -2.0];
        let up = [0.0, 1.0, 0.0];
        let f = normalize_clamped(sub(center, eye), 1e-6);
        let r = normalize_clamped(cross(f, up), 1e-6);
        assert_eq!(
            look_at(eye, center, up),
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
            let d = normalize_clamped(dir, 1e-6);
            assert!(dot(d, up_for(d)).abs() < 0.999);
        }
    }

    #[test]
    fn a_degenerate_direction_stays_finite() {
        assert!(
            normalize_clamped([0.0; 3], 1e-6)
                .iter()
                .all(|v| v.is_finite())
        );
    }
}
