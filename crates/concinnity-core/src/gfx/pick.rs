// src/gfx/pick.rs
//
// Pure picking math: a world-space ray through a window pixel, and a
// ray-vs-AABB intersection test. Consumed by the editor's viewport picking;
// nothing here touches a backend or the ECS.

/// A world-space ray: `origin` plus a normalized direction.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct PickRay {
    pub origin: [f32; 3],
    pub dir: [f32; 3],
}

/// The world-space ray through window pixel `mouse` (top-left origin, logical
/// pixels, as sampled into `FrameInput`), for a camera described by its
/// world-to-view matrix (the [view_matrix](#method.view_matrix) convention:
/// column-major, right-handed, view-space forward is -Z), its world position,
/// and its vertical field of view. Uses the un-jittered projection: TAA jitter
/// is a backend concern and must never skew a pick.
///
/// Returns `None` for a degenerate viewport (either extent zero or negative,
/// e.g. frame 0 before the backend reports a size).
pub fn screen_ray(
    view: &[[f32; 4]; 4],
    cam_pos: [f32; 3],
    fov_y_radians: f32,
    viewport: [f32; 2],
    mouse: [f32; 2],
) -> Option<PickRay> {
    let (vw, vh) = (viewport[0], viewport[1]);
    if vw <= 0.0 || vh <= 0.0 || !vw.is_finite() || !vh.is_finite() {
        return None;
    }
    let ndc_x = 2.0 * mouse[0] / vw - 1.0;
    let ndc_y = 1.0 - 2.0 * mouse[1] / vh;
    let tan_half = (fov_y_radians * 0.5).tan();
    if tan_half <= 0.0 || !tan_half.is_finite() {
        return None;
    }
    let aspect = vw / vh;

    // View-space direction of the pixel at unit depth (forward is -Z).
    let d = [ndc_x * tan_half * aspect, ndc_y * tan_half, -1.0];
    // The view rotation is orthonormal, so its inverse is the transpose:
    // world = R^T * d. Column-major view[c][r] makes that a per-column dot.
    let world = [
        view[0][0] * d[0] + view[0][1] * d[1] + view[0][2] * d[2],
        view[1][0] * d[0] + view[1][1] * d[1] + view[1][2] * d[2],
        view[2][0] * d[0] + view[2][1] * d[1] + view[2][2] * d[2],
    ];
    let len = (world[0] * world[0] + world[1] * world[1] + world[2] * world[2]).sqrt();
    if len <= 0.0 || !len.is_finite() {
        return None;
    }
    Some(PickRay {
        origin: cam_pos,
        dir: [world[0] / len, world[1] / len, world[2] / len],
    })
}

/// The distance along `ray` to the world-space AABB `[bb_min, bb_max]`, via
/// the slab test. `Some(0.0)` when the ray starts inside the box; `None` when
/// the ray misses, the box is entirely behind the origin, or the box is not
/// finite (the renderer's non-cullable sentinel).
pub fn ray_aabb(ray: &PickRay, bb_min: [f32; 3], bb_max: [f32; 3]) -> Option<f32> {
    for i in 0..3 {
        if !bb_min[i].is_finite() || !bb_max[i].is_finite() {
            return None;
        }
    }
    let mut t_enter = f32::NEG_INFINITY;
    let mut t_exit = f32::INFINITY;
    for i in 0..3 {
        if ray.dir[i] == 0.0 {
            // Parallel to this slab: inside it or a clean miss, with no
            // division (0 * inf would poison the interval with NaN).
            if ray.origin[i] < bb_min[i] || ray.origin[i] > bb_max[i] {
                return None;
            }
            continue;
        }
        let inv = 1.0 / ray.dir[i];
        let (t1, t2) = (
            (bb_min[i] - ray.origin[i]) * inv,
            (bb_max[i] - ray.origin[i]) * inv,
        );
        let (near, far) = if t1 <= t2 { (t1, t2) } else { (t2, t1) };
        t_enter = t_enter.max(near);
        t_exit = t_exit.min(far);
        if t_enter > t_exit {
            return None;
        }
    }
    if t_exit < 0.0 {
        return None;
    }
    Some(t_enter.max(0.0))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::gfx::camera::view_matrix;

    const VP: [f32; 2] = [1280.0, 720.0];
    const FOV: f32 = core::f32::consts::FRAC_PI_2;

    fn assert_dir(ray: PickRay, expect: [f32; 3]) {
        for i in 0..3 {
            assert!(
                (ray.dir[i] - expect[i]).abs() < 1e-5,
                "dir {:?} != {expect:?}",
                ray.dir
            );
        }
    }

    #[test]
    fn center_pixel_rays_along_camera_forward() {
        // Yaw 0, pitch 0 faces -Z (the view_matrix convention).
        let view = view_matrix([1.0, 2.0, 3.0], 0.0, 0.0);
        let ray = screen_ray(&view, [1.0, 2.0, 3.0], FOV, VP, [640.0, 360.0]).unwrap();
        assert_eq!(ray.origin, [1.0, 2.0, 3.0]);
        assert_dir(ray, [0.0, 0.0, -1.0]);

        // Yaw pi/2 faces -X.
        let view = view_matrix([0.0; 3], core::f32::consts::FRAC_PI_2, 0.0);
        let ray = screen_ray(&view, [0.0; 3], FOV, VP, [640.0, 360.0]).unwrap();
        assert_dir(ray, [-1.0, 0.0, 0.0]);

        // Pitch pi/2 faces straight up.
        let view = view_matrix([0.0; 3], 0.0, core::f32::consts::FRAC_PI_2);
        let ray = screen_ray(&view, [0.0; 3], FOV, VP, [640.0, 360.0]).unwrap();
        assert_dir(ray, [0.0, 1.0, 0.0]);
    }

    #[test]
    fn edge_pixels_span_the_field_of_view() {
        let view = view_matrix([0.0; 3], 0.0, 0.0);
        // Top-center pixel: the vertical view angle is fov/2 above forward.
        let top = screen_ray(&view, [0.0; 3], FOV, VP, [640.0, 0.0]).unwrap();
        let vert_tan = top.dir[1] / -top.dir[2];
        assert!((vert_tan - (FOV * 0.5).tan()).abs() < 1e-4, "{vert_tan}");
        assert!(top.dir[0].abs() < 1e-6);

        // Right-center pixel: the horizontal half-angle is scaled by aspect.
        let right = screen_ray(&view, [0.0; 3], FOV, VP, [1280.0, 360.0]).unwrap();
        let horiz_tan = right.dir[0] / -right.dir[2];
        let expect = (FOV * 0.5).tan() * (VP[0] / VP[1]);
        assert!((horiz_tan - expect).abs() < 1e-4, "{horiz_tan} vs {expect}");
        assert!(right.dir[1].abs() < 1e-6);
    }

    #[test]
    fn degenerate_viewport_or_fov_yields_no_ray() {
        let view = view_matrix([0.0; 3], 0.0, 0.0);
        assert_eq!(
            screen_ray(&view, [0.0; 3], FOV, [0.0, 720.0], [0.0; 2]),
            None
        );
        assert_eq!(
            screen_ray(&view, [0.0; 3], FOV, [1280.0, 0.0], [0.0; 2]),
            None
        );
        assert_eq!(screen_ray(&view, [0.0; 3], 0.0, VP, [0.0; 2]), None);
    }

    fn ray(origin: [f32; 3], dir: [f32; 3]) -> PickRay {
        PickRay { origin, dir }
    }

    #[test]
    fn ray_hits_a_box_at_the_entry_distance() {
        let r = ray([0.0, 0.0, 5.0], [0.0, 0.0, -1.0]);
        let t = ray_aabb(&r, [-1.0, -1.0, -1.0], [1.0, 1.0, 1.0]).unwrap();
        assert!((t - 4.0).abs() < 1e-6, "{t}");
    }

    #[test]
    fn ray_misses_beside_behind_and_non_finite_boxes() {
        let (mn, mx) = ([-1.0, -1.0, -1.0], [1.0, 1.0, 1.0]);
        // Beside: aimed parallel past the box.
        assert_eq!(
            ray_aabb(&ray([0.0, 3.0, 5.0], [0.0, 0.0, -1.0]), mn, mx),
            None
        );
        // Behind: the box is entirely behind the ray origin.
        assert_eq!(
            ray_aabb(&ray([0.0, 0.0, 5.0], [0.0, 0.0, 1.0]), mn, mx),
            None
        );
        // The renderer's non-cullable NaN sentinel is unpickable, not infinite.
        assert_eq!(
            ray_aabb(
                &ray([0.0; 3], [0.0, 0.0, -1.0]),
                [f32::NAN; 3],
                [f32::NAN; 3]
            ),
            None
        );
    }

    #[test]
    fn ray_inside_a_box_hits_at_zero() {
        let t = ray_aabb(
            &ray([0.0; 3], [0.0, 1.0, 0.0]),
            [-1.0, -1.0, -1.0],
            [1.0, 1.0, 1.0],
        )
        .unwrap();
        assert_eq!(t, 0.0);
    }

    #[test]
    fn axis_parallel_rays_respect_the_perpendicular_slabs() {
        let (mn, mx) = ([-1.0, -1.0, -1.0], [1.0, 1.0, 1.0]);
        // Sliding along Y inside the X/Z slabs: a hit.
        assert!(ray_aabb(&ray([0.5, -5.0, 0.5], [0.0, 1.0, 0.0]), mn, mx).is_some());
        // Same direction but outside the X slab: a miss, not a NaN artifact.
        assert_eq!(
            ray_aabb(&ray([2.0, -5.0, 0.5], [0.0, 1.0, 0.0]), mn, mx),
            None
        );
        // Origin exactly on a slab face with zero direction on that axis.
        assert!(ray_aabb(&ray([1.0, -5.0, 0.0], [0.0, 1.0, 0.0]), mn, mx).is_some());
    }
}
