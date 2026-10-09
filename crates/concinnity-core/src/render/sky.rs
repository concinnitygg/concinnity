//! The world's environment drawn as the background.
//!
//! The sky is one triangle covering the viewport at the far plane, drawn last
//! in each pass that renders the opaque scene from one viewpoint: the main
//! camera, a reflection-probe face, a planar mirror. It is depth-tested
//! inclusively without writing, so it lands exactly where no surface did, and
//! each pixel samples the environment along its own view ray. A second draw at
//! the tail of the geometry pre-pass writes the sky's motion, which is the
//! camera's rotation alone, so temporal resolves and upscalers reproject it.
//!
//! The ray math lives in `shaders/sky_ray.hlsl`; the tests here run a mirror of
//! it against the camera's own projections.

use crate::gfx::view_modes::ViewMode;

/// Whether a view draws the environment behind its geometry: the world has an
/// environment map loaded and draws it as the background, and the view shows
/// shaded surfaces. The wireframe mode shows triangle edges over the clear
/// color, so it draws no sky.
pub fn draws_sky(environment_loaded: bool, background: bool, mode: ViewMode) -> bool {
    environment_loaded && background && mode != ViewMode::Wireframe
}

#[cfg(test)]
mod tests {
    use crate::gfx::projection::{look_at, view_from_basis};
    use crate::math::vec3::{cross, dot, scale, sub};
    use crate::math::{cos, sin, sqrt};
    use crate::render::depth::{camera_oblique_projection, camera_projection};
    use crate::transform::{Mat4, mat4_inverse, mat4_mul};

    // Row `r` of a column-major matrix, as the shader's `vp[r]` reads it.
    fn row(m: Mat4, r: usize) -> [f32; 4] {
        [m[0][r], m[1][r], m[2][r], m[3][r]]
    }

    fn xyz(v: [f32; 4]) -> [f32; 3] {
        [v[0], v[1], v[2]]
    }

    fn transform(m: Mat4, v: [f32; 4]) -> [f32; 4] {
        core::array::from_fn(|r| dot4(row(m, r), v))
    }

    fn dot4(a: [f32; 4], b: [f32; 4]) -> f32 {
        a[0] * b[0] + a[1] * b[1] + a[2] * b[2] + a[3] * b[3]
    }

    fn normalize(v: [f32; 3]) -> [f32; 3] {
        scale(v, 1.0 / sqrt(dot(v, v)))
    }

    // `sky_corner` in sky_ray.hlsl.
    fn sky_corner(vid: u32) -> [f32; 2] {
        [
            if vid == 1 { 3.0 } else { -1.0 },
            if vid == 2 { 3.0 } else { -1.0 },
        ]
    }

    // `sky_ray` in sky_ray.hlsl.
    fn sky_ray(vp: Mat4, ndc: [f32; 2]) -> [f32; 3] {
        let (rx, ry, rw) = (xyz(row(vp, 0)), xyz(row(vp, 1)), xyz(row(vp, 3)));
        let ray = [0, 1, 2]
            .map(|i| cross(ry, rw)[i] * ndc[0] + cross(rw, rx)[i] * ndc[1] + cross(rx, ry)[i]);
        scale(ray, 1.0 / dot(rx, cross(ry, rw)))
    }

    // `gbuffer_sky_vertex` + `gb_motion` in gbuffer_sky.hlsl / gbuffer_common.hlsl: the motion
    // the pre-pass stores for the sky under NDC point `ndc`.
    fn sky_motion(jittered_vp: Mat4, cur_vp: Mat4, prev_vp: Mat4, ndc: [f32; 2]) -> [f32; 2] {
        let ray = sky_ray(jittered_vp, ndc);
        let dir = [ray[0], ray[1], ray[2], 0.0];
        motion(transform(cur_vp, dir), transform(prev_vp, dir))
    }

    // `GB_MOTION_LIMIT` and `GB_MIN_PREV_W` in gbuffer_common.hlsl.
    const MOTION_LIMIT: f32 = 2.0;
    const MIN_PREV_W: f32 = 1e-6;

    #[test]
    fn the_motion_mirror_matches_the_shader() {
        use crate::render::shader_consts::float;
        let src = crate::render::shaders::GBUFFER_COMMON;
        assert_eq!(float(src, "GB_MOTION_LIMIT"), MOTION_LIMIT);
        assert_eq!(float(src, "GB_MIN_PREV_W"), MIN_PREV_W);
    }

    fn motion(cur_clip: [f32; 4], prev_clip: [f32; 4]) -> [f32; 2] {
        if prev_clip[3].is_nan() || prev_clip[3] <= MIN_PREV_W {
            return [MOTION_LIMIT; 2];
        }
        let uv = |c: [f32; 4]| [c[0] / c[3] * 0.5 + 0.5, 0.5 - c[1] / c[3] * 0.5];
        let (cur, prev) = (uv(cur_clip), uv(prev_clip));
        [prev[0] - cur[0], prev[1] - cur[1]].map(|m| m.clamp(-MOTION_LIMIT, MOTION_LIMIT))
    }

    // What TAA and SSGI test before reading history at `uv + motion`.
    fn has_history(uv: [f32; 2], m: [f32; 2]) -> bool {
        (0..2).all(|i| (0.0..=1.0).contains(&(uv[i] + m[i])))
    }

    fn ndc_uv(ndc: [f32; 2]) -> [f32; 2] {
        [ndc[0] * 0.5 + 0.5, 0.5 - ndc[1] * 0.5]
    }

    // A camera at `EYE` looking `yaw` radians left of -z.
    fn yawed(yaw: f32) -> Mat4 {
        let proj = camera_projection(1.2, 16.0 / 9.0, 0.1);
        let dir = [-sin(yaw), 0.0, -cos(yaw)];
        mat4_mul(
            proj,
            look_at(EYE, [EYE[0] + dir[0], EYE[1], EYE[2] + dir[2]], UP),
        )
    }

    // The camera-relative direction through `ndc`, found the long way: the
    // inverse view-projection, a point on the ray, minus the eye.
    fn unprojected_ray(vp: Mat4, eye: [f32; 3], ndc: [f32; 2]) -> [f32; 3] {
        let h = transform(mat4_inverse(vp), [ndc[0], ndc[1], 0.5, 1.0]);
        normalize(sub(scale(xyz(h), 1.0 / h[3]), eye))
    }

    fn assert_close(a: [f32; 3], b: [f32; 3], tol: f32) {
        let d = sub(a, b);
        assert!(dot(d, d) <= tol * tol, "{a:?} vs {b:?}");
    }

    fn assert_close2(a: [f32; 2], b: [f32; 2], tol: f32) {
        let (dx, dy) = (a[0] - b[0], a[1] - b[1]);
        assert!(dx * dx + dy * dy <= tol * tol, "{a:?} vs {b:?}");
    }

    const EYE: [f32; 3] = [12.0, 3.0, -40.0];
    const TARGET: [f32; 3] = [20.0, 4.0, -50.0];
    const UP: [f32; 3] = [0.0, 1.0, 0.0];
    const SAMPLES: [[f32; 2]; 6] = [
        [0.0, 0.0],
        [0.9, -0.6],
        [-1.0, 1.0],
        [1.0, -1.0],
        [-0.3, 0.75],
        [0.5, 0.5],
    ];

    fn camera(eye: [f32; 3], target: [f32; 3]) -> Mat4 {
        let proj = camera_projection(1.2, 16.0 / 9.0, 0.1);
        mat4_mul(proj, look_at(eye, target, UP))
    }

    // A sub-pixel jitter on the projection, the way the temporal resolves
    // offset it.
    fn jittered(proj: Mat4, jx: f32, jy: f32) -> Mat4 {
        let mut p = proj;
        p[2][0] += jx;
        p[2][1] += jy;
        p
    }

    #[test]
    fn the_triangle_covers_the_viewport() {
        let [a, b, c] = [0, 1, 2].map(sky_corner);
        // Every viewport point satisfies x >= -1, y >= -1 and x + y <= 2, the
        // three edges of the triangle through those corners.
        assert_eq!((a, b, c), ([-1.0, -1.0], [3.0, -1.0], [-1.0, 3.0]));
        for p in [[-1.0, -1.0], [1.0, 1.0], [1.0, -1.0], [-1.0, 1.0]] {
            assert!(p[0] >= -1.0 && p[1] >= -1.0 && p[0] + p[1] <= 2.0);
        }
    }

    #[test]
    fn the_ray_is_the_one_the_inverse_projection_gives() {
        let vp = camera(EYE, TARGET);
        for ndc in SAMPLES {
            assert_close(
                normalize(sky_ray(vp, ndc)),
                unprojected_ray(vp, EYE, ndc),
                1e-4,
            );
        }
    }

    // The direction does not depend on where the camera stands, so the sky
    // never moves as the camera travels.
    #[test]
    fn the_ray_ignores_the_camera_position() {
        let offset = [500.0, -20.0, 900.0];
        let here = camera(EYE, TARGET);
        let there = camera(
            [EYE[0] + offset[0], EYE[1] + offset[1], EYE[2] + offset[2]],
            [
                TARGET[0] + offset[0],
                TARGET[1] + offset[1],
                TARGET[2] + offset[2],
            ],
        );
        for ndc in SAMPLES {
            assert_close(
                normalize(sky_ray(here, ndc)),
                normalize(sky_ray(there, ndc)),
                1e-4,
            );
        }
    }

    // Only the x, y and w rows take part, so neither the depth mapping nor an
    // oblique near plane moves the ray: a planar mirror's clipped projection
    // sees the same sky its unclipped camera would.
    #[test]
    fn the_depth_row_never_moves_the_ray() {
        let proj = camera_projection(1.2, 16.0 / 9.0, 0.1);
        let view = look_at(EYE, TARGET, UP);
        let oblique = camera_oblique_projection(proj, [0.2, 0.9, -0.1, -3.0]);
        let mut reversed = proj;
        for c in 0..4 {
            reversed[c][2] = reversed[c][3] - proj[c][2];
        }
        for ndc in SAMPLES {
            let reference = sky_ray(mat4_mul(proj, view), ndc);
            for other in [oblique, reversed] {
                assert_close(sky_ray(mat4_mul(other, view), ndc), reference, 1e-5);
            }
        }
    }

    // Scaled to clip w = 1, the ray always points out of the camera, even
    // through a mirror's reflected (handedness-flipping) view.
    #[test]
    fn the_ray_points_forward_through_a_mirror() {
        let proj = camera_projection(1.2, 16.0 / 9.0, 0.1);
        let mirrored = view_from_basis(EYE, [-1.0, 0.0, 0.0], UP, [0.0, 0.0, -1.0]);
        let vp = mat4_mul(proj, mirrored);
        for ndc in SAMPLES {
            let ray = sky_ray(vp, ndc);
            assert!(dot4(row(vp, 3), [ray[0], ray[1], ray[2], 0.0]) > 0.999);
        }
    }

    // The ray is linear in NDC, so interpolating the three corners' rays over
    // the triangle gives each pixel exactly its own ray.
    #[test]
    fn interpolating_the_corners_gives_each_pixel_its_ray() {
        let vp = camera(EYE, TARGET);
        let [a, b, c] = [0, 1, 2].map(|v| sky_ray(vp, sky_corner(v)));
        for ndc in SAMPLES {
            // Barycentrics of `ndc` in the corner triangle.
            let wb = (ndc[0] + 1.0) / 4.0;
            let wc = (ndc[1] + 1.0) / 4.0;
            let wa = 1.0 - wb - wc;
            let lerped = [0, 1, 2].map(|i| a[i] * wa + b[i] * wb + c[i] * wc);
            assert_close(lerped, sky_ray(vp, ndc), 1e-4);
        }
    }

    #[test]
    fn a_camera_that_only_moves_gives_the_sky_no_motion() {
        let cur = camera(EYE, TARGET);
        let prev = camera(
            [EYE[0] - 4.0, EYE[1], EYE[2] + 3.0],
            [TARGET[0] - 4.0, TARGET[1], TARGET[2] + 3.0],
        );
        for ndc in SAMPLES {
            assert_close2(sky_motion(cur, cur, prev, ndc), [0.0, 0.0], 1e-5);
        }
    }

    #[test]
    fn jitter_never_leaks_into_the_motion() {
        let proj = camera_projection(1.2, 16.0 / 9.0, 0.1);
        let view = look_at(EYE, TARGET, UP);
        let cur = mat4_mul(proj, view);
        let shaky = mat4_mul(jittered(proj, 0.0013, -0.0007), view);
        for ndc in SAMPLES {
            assert_close2(sky_motion(shaky, cur, cur, ndc), [0.0, 0.0], 1e-6);
        }
    }

    // Under a pure yaw the sky pixel at the screen center was, last frame,
    // where the previous camera saw that same direction.
    #[test]
    fn a_turning_camera_moves_the_sky_by_its_rotation() {
        let proj = camera_projection(1.2, 16.0 / 9.0, 0.1);
        let yaw = 0.05f32;
        let turned = [-sin(yaw), 0.0, -cos(yaw)];
        let prev = mat4_mul(proj, look_at(EYE, [EYE[0], EYE[1], EYE[2] - 1.0], UP));
        let cur = mat4_mul(
            proj,
            look_at(EYE, [EYE[0] + turned[0], EYE[1], EYE[2] + turned[2]], UP),
        );
        let m = sky_motion(cur, cur, prev, [0.0, 0.0]);
        // The current center direction, seen by the previous camera.
        let seen = transform(prev, [turned[0], turned[1], turned[2], 0.0]);
        let prev_uv = [seen[0] / seen[3] * 0.5 + 0.5, 0.5 - seen[1] / seen[3] * 0.5];
        assert_close2(m, [prev_uv[0] - 0.5, prev_uv[1] - 0.5], 1e-5);
        assert!(
            m[0] < -0.01,
            "turning left moves the sky right on screen: {m:?}"
        );
    }

    // The motion a surface at any finite distance along the ray would get if it
    // traveled with the camera: what a sky drawn as geometry carried along with
    // the camera wrote, which this reproduces.
    #[test]
    fn the_motion_matches_geometry_carried_with_the_camera() {
        let prev_eye = [EYE[0] - 2.0, EYE[1] + 0.5, EYE[2] + 6.0];
        let prev = camera(prev_eye, [TARGET[0], TARGET[1] - 2.0, TARGET[2]]);
        let cur = camera(EYE, TARGET);
        for ndc in SAMPLES {
            let ray = normalize(sky_ray(cur, ndc));
            let far = 150.0;
            let cur_point = [
                EYE[0] + ray[0] * far,
                EYE[1] + ray[1] * far,
                EYE[2] + ray[2] * far,
                1.0,
            ];
            let prev_point = [
                prev_eye[0] + ray[0] * far,
                prev_eye[1] + ray[1] * far,
                prev_eye[2] + ray[2] * far,
                1.0,
            ];
            let carried = motion(transform(cur, cur_point), transform(prev, prev_point));
            assert_close2(sky_motion(cur, cur, prev, ndc), carried, 1e-4);
        }
    }

    // The previous view seeded from the current one on the first motion frame
    // reads no motion; the identity placeholder it replaces gives the sky's
    // directions a previous w of zero.
    #[test]
    fn the_first_motion_frame_gives_the_sky_no_motion() {
        use crate::render::view_history::{ViewFrame, ViewHistory};
        use crate::transform::IDENTITY;
        let cur = camera(EYE, TARGET);
        let prev = ViewHistory::default()
            .prev_or(ViewFrame {
                vp: cur,
                elapsed: 0.0,
                cam_pos: EYE,
            })
            .vp;
        for ndc in SAMPLES {
            assert_close2(sky_motion(cur, cur, prev, ndc), [0.0, 0.0], 1e-6);
            let placeholder = sky_motion(cur, cur, IDENTITY, ndc);
            assert!(placeholder.iter().all(|m| m.is_finite()), "{placeholder:?}");
            assert!(!has_history(ndc_uv(ndc), placeholder));
        }
    }

    // A cut that turns the camera around leaves the sky ahead of it behind the
    // previous camera: no history, and never a non-finite motion.
    #[test]
    fn a_turn_past_the_previous_view_has_no_history() {
        let cur = yawed(0.0);
        for yaw in [1.4, 1.6, 2.0, 3.1] {
            let prev = yawed(yaw);
            for ndc in SAMPLES {
                let m = sky_motion(cur, cur, prev, ndc);
                assert!(
                    m.iter().all(|c| c.is_finite() && c.abs() <= MOTION_LIMIT),
                    "{yaw} {ndc:?}: {m:?}"
                );
                // Past about 115 degrees no current direction is in the previous view.
                if yaw >= 2.0 {
                    assert!(!has_history(ndc_uv(ndc), m), "{yaw} {ndc:?}: {m:?}");
                }
            }
        }
        // Near a half turn every direction ahead lies behind the previous camera.
        let m = sky_motion(cur, cur, yawed(3.1), [0.0, 0.0]);
        assert_eq!(m, [MOTION_LIMIT; 2]);
    }
}
