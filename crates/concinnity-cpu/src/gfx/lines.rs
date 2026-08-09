// src/gfx/lines.rs
//
// The CPU expansion that turns a world-space `Line` into the camera-facing
// ribbon the line pass rasterises.
//
// A hardware line primitive cannot carry a pixel width portably, so each segment
// expands into a quad whose corners are offset perpendicular to the line and
// perpendicular to the eye vector, scaled by the world-per-pixel size at that
// corner's depth. Both edges of the ribbon are straight world-space lines, so
// they project to straight screen-space lines separated by exactly the
// requested pixel width along the whole run, however far the far end reaches.

use crate::geometry::vec3::{cross, dot, lerp as lerp3, sub};

use concinnity_core::gfx::render_types::LineVertex;

pub use concinnity_core::gfx::lines::Line;

// Vertices emitted per expanded segment: two triangles, unindexed.
const VERTS_PER_SEGMENT: usize = 6;

// The camera the expansion projects against. `view` is the world-to-view matrix
// the frame renders with (column-major, `view[col][row]`) and `cam_pos` its
// world-space position: in a camera-relative world both are the rebased pair,
// so callers must express their lines in that same space.
#[derive(Copy, Clone, Debug)]
pub struct LineCamera {
    pub view: [[f32; 4]; 4],
    pub cam_pos: [f32; 3],
    pub fov_y_radians: f32,
    pub viewport: [f32; 2],
    pub near: f32,
}

impl LineCamera {
    // The camera's world-space forward (the -Z view axis).
    fn forward(&self) -> [f32; 3] {
        [-self.view[0][2], -self.view[1][2], -self.view[2][2]]
    }

    // The camera's world-space right (the +X view axis), used as the ribbon
    // normal when a line points straight at the eye.
    fn right(&self) -> [f32; 3] {
        [self.view[0][0], self.view[1][0], self.view[2][0]]
    }

    // Tangent of the half vertical FOV. Fixed for a camera, so a caller that
    // needs it per vertex computes it once and passes it down.
    fn tan_half_fov(&self) -> f32 {
        (self.fov_y_radians * 0.5).tan()
    }

    // World units per vertical pixel at view depth `depth`.
    fn world_per_pixel(&self, depth: f32, tan_half: f32) -> f32 {
        2.0 * depth * tan_half / self.viewport[1]
    }

    fn usable(&self, tan_half: f32) -> bool {
        self.viewport[0] > 0.0 && self.viewport[1] > 0.0 && tan_half > 0.0 && tan_half.is_finite()
    }
}

fn normalize(v: [f32; 3]) -> Option<[f32; 3]> {
    let len = dot(v, v).sqrt();
    (len > 1e-6).then(|| [v[0] / len, v[1] / len, v[2] / len])
}

fn lerp4(a: [f32; 4], b: [f32; 4], t: f32) -> [f32; 4] {
    [
        a[0] + (b[0] - a[0]) * t,
        a[1] + (b[1] - a[1]) * t,
        a[2] + (b[2] - a[2]) * t,
        a[3] + (b[3] - a[3]) * t,
    ]
}

// A segment trimmed to the part in front of the near plane, with its colours
// interpolated to wherever the trim landed.
struct Clipped {
    start: [f32; 3],
    end: [f32; 3],
    start_color: [f32; 4],
    end_color: [f32; 4],
}

// Trim a segment to the part in front of the near plane. `None` when the whole
// segment is at or behind the plane (nothing to draw).
fn clip_to_near(line: &Line, cam: &LineCamera) -> Option<Clipped> {
    let fwd = cam.forward();
    let d0 = dot(sub(line.start, cam.cam_pos), fwd);
    let d1 = dot(sub(line.end, cam.cam_pos), fwd);
    let near = cam.near.max(1e-4);
    match (d0 >= near, d1 >= near) {
        (true, true) => Some(Clipped {
            start: line.start,
            end: line.end,
            start_color: line.start_color,
            end_color: line.end_color,
        }),
        (false, false) => None,
        (true, false) => {
            let t = (d0 - near) / (d0 - d1);
            Some(Clipped {
                start: line.start,
                end: lerp3(line.start, line.end, t),
                start_color: line.start_color,
                end_color: lerp4(line.start_color, line.end_color, t),
            })
        }
        (false, true) => {
            let t = (near - d0) / (d1 - d0);
            Some(Clipped {
                start: lerp3(line.start, line.end, t),
                end: line.end,
                start_color: lerp4(line.start_color, line.end_color, t),
                end_color: line.end_color,
            })
        }
    }
}

// Expand `lines` into the ribbon triangles the line pass draws. Segments
// wholly behind the near plane, degenerate segments, and fully transparent
// segments contribute nothing, so an empty result means the pass can be
// skipped entirely.
pub fn build_vertices(lines: &[Line], cam: &LineCamera) -> Vec<LineVertex> {
    let tan_half = cam.tan_half_fov();
    if !cam.usable(tan_half) {
        return Vec::new();
    }
    let mut out = Vec::with_capacity(lines.len() * VERTS_PER_SEGMENT);
    let fwd = cam.forward();
    for line in lines {
        if line.width_px <= 0.0 || (line.start_color[3] <= 0.0 && line.end_color[3] <= 0.0) {
            continue;
        }
        let Some(seg) = clip_to_near(line, cam) else {
            continue;
        };
        let Some(dir) = normalize(sub(seg.end, seg.start)) else {
            continue;
        };
        let half_px = line.width_px * 0.5;
        // Ribbon normal per endpoint: perpendicular to both the line and the
        // eye vector, so the quad always faces the camera. A line aimed at the
        // eye has no such perpendicular; the camera right is the stable
        // fallback (the ribbon is a dot on screen there anyway).
        let corner = |p: [f32; 3], color: [f32; 4], side: f32| {
            let to_eye = sub(p, cam.cam_pos);
            let normal = normalize(cross(dir, to_eye)).unwrap_or_else(|| cam.right());
            let half =
                half_px * cam.world_per_pixel(dot(to_eye, fwd).max(cam.near.max(1e-4)), tan_half);
            LineVertex {
                pos: [
                    p[0] + normal[0] * half * side,
                    p[1] + normal[1] * half * side,
                    p[2] + normal[2] * half * side,
                ],
                edge: side,
                color,
            }
        };
        let a = corner(seg.start, seg.start_color, -1.0);
        let b = corner(seg.start, seg.start_color, 1.0);
        let c = corner(seg.end, seg.end_color, -1.0);
        let d = corner(seg.end, seg.end_color, 1.0);
        out.extend_from_slice(&[a, b, c, c, b, d]);
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use concinnity_core::gfx::camera::view_matrix;

    const VP: [f32; 2] = [1280.0, 720.0];

    fn camera() -> LineCamera {
        LineCamera {
            // Facing -Z from the origin.
            view: view_matrix([0.0; 3], 0.0, 0.0),
            cam_pos: [0.0; 3],
            fov_y_radians: core::f32::consts::FRAC_PI_2,
            viewport: VP,
            near: 0.1,
        }
    }

    fn line(start: [f32; 3], end: [f32; 3]) -> Line {
        Line {
            start,
            end,
            start_color: [1.0, 0.0, 0.0, 1.0],
            end_color: [1.0, 0.0, 0.0, 0.0],
            width_px: 2.0,
        }
    }

    // Project a world point to pixels with the same convention the renderer
    // uses, so a test can measure the ribbon's on-screen width.
    fn project(cam: &LineCamera, p: [f32; 3]) -> [f32; 2] {
        let v = &cam.view;
        let x = v[0][0] * p[0] + v[1][0] * p[1] + v[2][0] * p[2] + v[3][0];
        let y = v[0][1] * p[0] + v[1][1] * p[1] + v[2][1] * p[2] + v[3][1];
        let z = v[0][2] * p[0] + v[1][2] * p[1] + v[2][2] * p[2] + v[3][2];
        let depth = -z;
        let tan_half = (cam.fov_y_radians * 0.5).tan();
        let aspect = cam.viewport[0] / cam.viewport[1];
        [
            (x / (depth * tan_half * aspect) + 1.0) * 0.5 * cam.viewport[0],
            (1.0 - y / (depth * tan_half)) * 0.5 * cam.viewport[1],
        ]
    }

    #[test]
    fn each_segment_expands_to_two_triangles() {
        let cam = camera();
        let verts = build_vertices(&[line([0.0, 0.0, -5.0], [0.0, 0.0, -50.0])], &cam);
        assert_eq!(verts.len(), VERTS_PER_SEGMENT);
        // Opposite ribbon edges, so the fragment fade has a full -1..1 span.
        assert!(verts.iter().any(|v| v.edge == -1.0));
        assert!(verts.iter().any(|v| v.edge == 1.0));
    }

    #[test]
    fn ribbon_holds_its_pixel_width_at_any_depth() {
        let cam = camera();
        // A line running left to right across the view, one end 4x further out
        // than the other: both ends must still measure `width_px` on screen.
        let l = line([-5.0, 0.0, -5.0], [40.0, 0.0, -20.0]);
        let verts = build_vertices(&[l], &cam);
        let near_w = {
            let a = project(&cam, verts[0].pos);
            let b = project(&cam, verts[1].pos);
            ((a[0] - b[0]).powi(2) + (a[1] - b[1]).powi(2)).sqrt()
        };
        let far_w = {
            let a = project(&cam, verts[2].pos);
            let b = project(&cam, verts[5].pos);
            ((a[0] - b[0]).powi(2) + (a[1] - b[1]).powi(2)).sqrt()
        };
        assert!((near_w - l.width_px).abs() < 0.05, "near end {near_w}");
        assert!((far_w - l.width_px).abs() < 0.05, "far end {far_w}");
    }

    #[test]
    fn colors_carry_the_fade_to_the_far_end() {
        let cam = camera();
        let verts = build_vertices(&[line([0.0, 0.0, -5.0], [0.0, 0.0, -500.0])], &cam);
        assert_eq!(verts[0].color[3], 1.0, "solid at the near end");
        assert_eq!(verts[5].color[3], 0.0, "faded out at the far end");
    }

    #[test]
    fn segments_behind_the_camera_are_clipped_away() {
        let cam = camera();
        // Wholly behind: nothing to draw.
        assert!(build_vertices(&[line([0.0, 0.0, 5.0], [0.0, 0.0, 50.0])], &cam).is_empty());
        // Straddling the near plane: trimmed to the visible part, and every
        // emitted corner sits in front of the camera.
        let verts = build_vertices(&[line([0.0, 0.0, 10.0], [0.0, 0.0, -10.0])], &cam);
        assert_eq!(verts.len(), VERTS_PER_SEGMENT);
        for v in &verts {
            assert!(
                v.pos[2] <= -cam.near,
                "corner {:?} is behind the near",
                v.pos
            );
        }
    }

    #[test]
    fn clipping_interpolates_the_endpoint_colour() {
        let cam = camera();
        // Half the run is behind the camera, so the near-plane end takes
        // (nearly) the midpoint colour rather than the authored start colour.
        let l = Line {
            start: [0.0, 0.0, 10.0],
            end: [0.0, 0.0, -10.0],
            start_color: [1.0, 0.0, 0.0, 1.0],
            end_color: [1.0, 0.0, 0.0, 0.0],
            width_px: 2.0,
        };
        let verts = build_vertices(&[l], &cam);
        assert!(
            (verts[0].color[3] - 0.5).abs() < 0.02,
            "{:?}",
            verts[0].color
        );
    }

    #[test]
    fn nothing_to_draw_yields_no_vertices() {
        let cam = camera();
        assert!(build_vertices(&[], &cam).is_empty());
        // Degenerate (zero-length), zero-width, and fully transparent lines all
        // drop out before expansion.
        let zero_len = line([1.0, 2.0, -3.0], [1.0, 2.0, -3.0]);
        assert!(build_vertices(&[zero_len], &cam).is_empty());
        let mut no_width = line([0.0, 0.0, -5.0], [0.0, 0.0, -50.0]);
        no_width.width_px = 0.0;
        assert!(build_vertices(&[no_width], &cam).is_empty());
        let mut invisible = line([0.0, 0.0, -5.0], [0.0, 0.0, -50.0]);
        invisible.start_color[3] = 0.0;
        invisible.end_color[3] = 0.0;
        assert!(build_vertices(&[invisible], &cam).is_empty());
    }

    #[test]
    fn a_degenerate_viewport_draws_nothing() {
        let mut cam = camera();
        cam.viewport = [0.0, 720.0];
        assert!(build_vertices(&[line([0.0, 0.0, -5.0], [0.0, 0.0, -50.0])], &cam).is_empty());
    }

    #[test]
    fn a_line_aimed_at_the_eye_falls_back_to_the_camera_right() {
        let cam = camera();
        // Running straight away from the camera: the eye vector and the line
        // are parallel, so the cross product is degenerate.
        let verts = build_vertices(&[line([0.0, 0.0, -5.0], [0.0, 0.0, -50.0])], &cam);
        assert_eq!(verts.len(), VERTS_PER_SEGMENT);
        // The fallback normal is the camera right, so the corners separate
        // along world X and stay finite.
        assert!(verts[0].pos[0] < 0.0 && verts[1].pos[0] > 0.0);
        for v in &verts {
            assert!(v.pos.iter().all(|c| c.is_finite()));
        }
    }
}
