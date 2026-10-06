//! Backend-agnostic frustum culling.
//!
//! Given a column-major view-projection matrix the six clip-space planes are
//! extracted using the Gribb-Hartmann method (left/right/bottom/top/near/far).
//! [`Frustum::from_camera`] takes a camera view-projection (reversed, infinite
//! depth: near at device depth 1, no far plane) and [`Frustum::from_shadow`] a
//! shadow one (near at 0, far at 1); only their near and far planes differ.
//! `Frustum::intersects_aabb` returns false only when an axis-aligned bounding
//! box is fully outside at least one plane.  False positives are acceptable for
//! culling (a few extra draws), false negatives are not, so the test treats
//! the box as visible whenever it overlaps any plane.

use crate::math::vec3::length;

/// One frustum plane in clip space.
#[derive(Copy, Clone, Debug)]
pub struct Plane {
    /// Plane equation in clip space: dot(normal, p) + d >= 0 == inside.
    pub normal: [f32; 3],
    /// Plane constant: the signed distance from the origin along `normal`.
    pub d: f32,
}

/// The six clip-space planes of a view frustum.
#[derive(Copy, Clone, Debug)]
pub struct Frustum {
    /// Left, right, bottom, top, near, far. A camera frustum with no view
    /// distance has no far plane, so its far slot repeats the near plane.
    pub planes: [Plane; 6],
}

impl Frustum {
    /// The frustum of a camera view-projection: the main camera, a reflection
    /// probe face or a planar reflection. `vp[col][row]`, column-major, the
    /// layout the renderer's ViewUniforms use. The projection has no far plane,
    /// so the frustum is open-ended unless `view_distance` closes it at that
    /// distance along the view axis.
    pub fn from_camera(vp: [[f32; 4]; 4], view_distance: Option<f32>) -> Self {
        // Reversed depth keeps z <= w: near is w - z >= 0. The infinite
        // projection's z row is constant, so it bounds nothing on the far side.
        Self::extract(vp, |z, w| {
            let near = combine(w, z, -1.0);
            let far = view_distance
                .and_then(|distance| view_distance_plane(w, distance))
                .unwrap_or(near);
            (near, far)
        })
    }

    /// The frustum of a shadow view-projection: a directional cascade or a spot
    /// slice. Same layout as [`Frustum::from_camera`].
    pub fn from_shadow(vp: [[f32; 4]; 4]) -> Self {
        // Far is w - z >= 0. Near is the -1..1 plane w + z >= 0, which sits
        // behind the 0..1 one and only keeps more.
        Self::extract(vp, |z, w| (combine(w, z, 1.0), combine(w, z, -1.0)))
    }

    // Gribb-Hartmann extraction. `depth_planes` turns the z and w rows into the
    // near and far planes the projection's depth range implies.
    fn extract(
        vp: [[f32; 4]; 4],
        depth_planes: impl Fn([f32; 4], [f32; 4]) -> ([f32; 4], [f32; 4]),
    ) -> Self {
        // Row r of vp = [vp[0][r], vp[1][r], vp[2][r], vp[3][r]].
        let row = |r: usize| -> [f32; 4] { [vp[0][r], vp[1][r], vp[2][r], vp[3][r]] };
        let r0 = row(0);
        let r1 = row(1);
        let r2 = row(2);
        let r3 = row(3);

        let (near, far) = depth_planes(r2, r3);

        Self {
            planes: [
                normalize_plane(combine(r3, r0, 1.0)),  // left:   row3 + row0
                normalize_plane(combine(r3, r0, -1.0)), // right:  row3 - row0
                normalize_plane(combine(r3, r1, 1.0)),  // bottom: row3 + row1
                normalize_plane(combine(r3, r1, -1.0)), // top:    row3 - row1
                normalize_plane(near),
                normalize_plane(far),
            ],
        }
    }

    /// True when the AABB is not entirely outside any plane.
    pub fn intersects_aabb(&self, bb_min: [f32; 3], bb_max: [f32; 3]) -> bool {
        for plane in &self.planes {
            // Pick the AABB corner furthest along the plane normal ("p-vertex"
            // in the SAT against a plane). If that corner is still behind the
            // plane the entire AABB is outside.
            let mut farthest = [0.0f32; 3];
            for (i, n) in plane.normal.iter().enumerate() {
                farthest[i] = if *n >= 0.0 { bb_max[i] } else { bb_min[i] };
            }
            let dist = plane.normal[0] * farthest[0]
                + plane.normal[1] * farthest[1]
                + plane.normal[2] * farthest[2]
                + plane.d;
            if dist < 0.0 {
                return false;
            }
        }
        true
    }
}

// The plane `distance - w >= 0`: clip w is the view-axis depth of a camera
// projection, so this keeps everything nearer than `distance` along the view
// axis. `None` when the w row carries no view axis to measure along.
fn view_distance_plane(w: [f32; 4], distance: f32) -> Option<[f32; 4]> {
    (length([w[0], w[1], w[2]]) > 1e-6).then_some([-w[0], -w[1], -w[2], distance - w[3]])
}

// `a + sign * b`, componentwise.
fn combine(a: [f32; 4], b: [f32; 4], sign: f32) -> [f32; 4] {
    [
        b[0] * sign + a[0],
        b[1] * sign + a[1],
        b[2] * sign + a[2],
        b[3] * sign + a[3],
    ]
}

fn normalize_plane(p: [f32; 4]) -> Plane {
    let len = length([p[0], p[1], p[2]]);
    let inv = if len > 1e-6 { 1.0 / len } else { 1.0 };
    Plane {
        normal: [p[0] * inv, p[1] * inv, p[2] * inv],
        d: p[3] * inv,
    }
}

/// Compute the world-space AABB enclosing a local-space AABB transformed by
/// a column-major model matrix.  All eight corners are transformed and
/// min/max'd component-wise.
pub fn transform_aabb(
    bb_min: [f32; 3],
    bb_max: [f32; 3],
    model: [[f32; 4]; 4],
) -> ([f32; 3], [f32; 3]) {
    let corners = [
        [bb_min[0], bb_min[1], bb_min[2]],
        [bb_max[0], bb_min[1], bb_min[2]],
        [bb_min[0], bb_max[1], bb_min[2]],
        [bb_max[0], bb_max[1], bb_min[2]],
        [bb_min[0], bb_min[1], bb_max[2]],
        [bb_max[0], bb_min[1], bb_max[2]],
        [bb_min[0], bb_max[1], bb_max[2]],
        [bb_max[0], bb_max[1], bb_max[2]],
    ];
    let mut out_min = [f32::INFINITY; 3];
    let mut out_max = [f32::NEG_INFINITY; 3];
    for c in &corners {
        // column-major mul: out = M * (c.x, c.y, c.z, 1)
        let x = model[0][0] * c[0] + model[1][0] * c[1] + model[2][0] * c[2] + model[3][0];
        let y = model[0][1] * c[0] + model[1][1] * c[1] + model[2][1] * c[2] + model[3][1];
        let z = model[0][2] * c[0] + model[1][2] * c[1] + model[2][2] * c[2] + model[3][2];
        out_min[0] = out_min[0].min(x);
        out_min[1] = out_min[1].min(y);
        out_min[2] = out_min[2].min(z);
        out_max[0] = out_max[0].max(x);
        out_max[1] = out_max[1].max(y);
        out_max[2] = out_max[2].max(z);
    }
    (out_min, out_max)
}

/// Squared distance from `cam` to the closest point on the AABB.
/// Returns 0 if `cam` is inside.
pub fn aabb_distance_sq(cam: [f32; 3], bb_min: [f32; 3], bb_max: [f32; 3]) -> f32 {
    let mut sq = 0.0f32;
    for i in 0..3 {
        let v = cam[i];
        if v < bb_min[i] {
            let d = bb_min[i] - v;
            sq += d * d;
        } else if v > bb_max[i] {
            let d = v - bb_max[i];
            sq += d * d;
        }
    }
    sq
}

#[cfg(test)]
mod tests {
    use super::*;

    fn identity4() -> [[f32; 4]; 4] {
        [
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ]
    }

    #[test]
    fn identity_vp_contains_origin_aabb() {
        // Identity VP defines the clip box (x, y in [-1, 1], depth up to 1) as
        // the visible region.
        let f = Frustum::from_camera(identity4(), None);
        assert!(f.intersects_aabb([-0.5, -0.5, -0.5], [0.5, 0.5, 0.5]));
        // Its w row has no view axis, so a view distance adds no plane.
        let bounded = Frustum::from_camera(identity4(), Some(0.5));
        assert!(bounded.intersects_aabb([-0.5, -0.5, -0.5], [0.5, 0.5, 0.5]));
    }

    #[test]
    fn identity_vp_rejects_far_aabb() {
        let f = Frustum::from_camera(identity4(), None);
        // entirely past the right clip plane
        assert!(!f.intersects_aabb([5.0, -0.5, -0.5], [6.0, 0.5, 0.5]));
    }

    fn point_inside(f: &Frustum, p: [f32; 3]) -> bool {
        f.intersects_aabb(p, p)
    }

    // The camera frustum's near plane is exactly the projection's, and with no
    // view distance nothing ahead of it is ever culled for being far.
    #[test]
    fn the_camera_frustum_starts_at_near_and_never_ends() {
        let vp = crate::render::depth::camera_projection(1.2, 1.6, 0.1);
        let f = Frustum::from_camera(vp, None);
        assert!(point_inside(&f, [0.0, 0.0, -0.1001]));
        assert!(!point_inside(&f, [0.0, 0.0, -0.0999]));
        for distance in [500.0, 1.0e4, 1.0e6, 1.0e9] {
            assert!(point_inside(&f, [0.0, 0.0, -distance]), "{distance}");
            assert!(point_inside(
                &f,
                [0.3 * distance, -0.2 * distance, -distance]
            ));
        }
        assert!(point_inside(&f, [10.0, -5.0, -100.0]));
        assert!(!point_inside(&f, [10.0, 0.0, 1.0]));
        assert!(
            !point_inside(&f, [1.0e7, 0.0, -1.0e6]),
            "still bounded at the sides"
        );
        for plane in &f.planes {
            assert!((length(plane.normal) - 1.0).abs() < 1e-5, "{plane:?}");
        }
    }

    // A view distance closes the frustum at that distance along the view axis,
    // wherever the camera sits and whichever way it looks.
    #[test]
    fn a_view_distance_closes_the_camera_frustum() {
        let proj = crate::render::depth::camera_projection(1.2, 1.6, 0.1);
        let f = Frustum::from_camera(proj, Some(500.0));
        assert!(point_inside(&f, [0.0, 0.0, -499.9]));
        assert!(!point_inside(&f, [0.0, 0.0, -500.1]));
        assert!(point_inside(&f, [0.0, 0.0, -0.1001]));

        let eye = [1.0e3, 20.0, -4.0e3];
        let view = crate::gfx::projection::look_at(
            eye,
            [1.0e3 + 3.0, 20.0, -4.0e3 - 4.0],
            [0.0, 1.0, 0.0],
        );
        let f = Frustum::from_camera(crate::transform::mat4_mul(proj, view), Some(100.0));
        let ahead = |d: f32| [eye[0] + 0.6 * d, eye[1], eye[2] - 0.8 * d];
        assert!(point_inside(&f, ahead(99.5)));
        assert!(!point_inside(&f, ahead(100.5)));
        let far_plane = f.planes[5];
        assert!((length(far_plane.normal) - 1.0).abs() < 1e-5);
    }

    #[test]
    fn the_shadow_frustum_keeps_its_near_to_far_box() {
        let spot = crate::render::depth::shadow_perspective(1.0, 1.0, 0.05, 40.0);
        let f = Frustum::from_shadow(spot);
        assert!(point_inside(&f, [0.0, 0.0, -0.06]));
        assert!(point_inside(&f, [0.0, 0.0, -39.9]));
        assert!(!point_inside(&f, [0.0, 0.0, -40.1]));

        let cascade = crate::render::depth::shadow_ortho(-4.0, 4.0, -4.0, 4.0, -2.0, 8.0);
        let f = Frustum::from_shadow(cascade);
        assert!(point_inside(&f, [3.9, -3.9, 1.9]));
        assert!(point_inside(&f, [0.0, 0.0, -7.9]));
        assert!(!point_inside(&f, [0.0, 0.0, -8.1]));
        assert!(!point_inside(&f, [4.1, 0.0, 0.0]));
    }

    #[test]
    fn transform_aabb_identity_passthrough() {
        let (mn, mx) = transform_aabb([0.0, 0.0, 0.0], [1.0, 2.0, 3.0], identity4());
        assert_eq!(mn, [0.0, 0.0, 0.0]);
        assert_eq!(mx, [1.0, 2.0, 3.0]);
    }

    #[test]
    fn transform_aabb_translates_corners() {
        let mut model = identity4();
        model[3][0] = 5.0;
        model[3][1] = -2.0;
        let (mn, mx) = transform_aabb([0.0, 0.0, 0.0], [1.0, 1.0, 1.0], model);
        assert_eq!(mn, [5.0, -2.0, 0.0]);
        assert_eq!(mx, [6.0, -1.0, 1.0]);
    }

    #[test]
    fn aabb_distance_inside_is_zero() {
        let d = aabb_distance_sq([0.5, 0.5, 0.5], [0.0, 0.0, 0.0], [1.0, 1.0, 1.0]);
        assert_eq!(d, 0.0);
    }

    #[test]
    fn aabb_distance_outside_is_squared() {
        // Camera 3 units to the right of a unit box at origin
        let d = aabb_distance_sq([4.0, 0.0, 0.0], [0.0, 0.0, 0.0], [1.0, 1.0, 1.0]);
        assert!((d - 9.0).abs() < 1e-5);
    }
}
