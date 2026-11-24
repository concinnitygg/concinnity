//! The translate gizmo: three world-axis handles drawn over the selected
//! asset's origin as dotted screen-space lines (a canvas draws only upright
//! rectangles, so each line is a run of small square segments) ending in a
//! draggable tip handle.
//! This module is the pure half: the screen layout, the handle hit test, and
//! the axis-drag math. The hook (`hook/drag/gizmo.rs`) owns the drag state and
//! the write-back.

use concinnity_engine::components::TextAlign;
use concinnity_engine::gfx::pick::PickRay;

use crate::editor::theme;
use crate::editor::ui::shapes::{Shape, Shapes};

pub(crate) const SEGMENTS: usize = 6;

const SEGMENT_PX: f32 = 3.0;
const TIP_PX: f32 = 10.0;
// Extra slop around the tip handle for the press hit test.
const TIP_GRAB_PX: f32 = 3.0;
// Screen-space axis length: constant regardless of camera distance, like any
// editor gizmo.
const AXIS_PX: f32 = 70.0;

// X red, Y green, Z blue.
const AXIS_TINTS: [[f32; 4]; 3] = [
    [0.86, 0.28, 0.28, 1.0],
    [0.30, 0.78, 0.34, 1.0],
    [0.32, 0.52, 0.94, 1.0],
];

pub(crate) const AXES: [[f32; 3]; 3] = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]];

// What the gizmo edits. One shared handle skeleton; the mode decides the drag
// math, the arg written back, and the tip shape.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub(crate) enum GizmoMode {
    #[default]
    Translate,
    Rotate,
    Scale,
}

impl GizmoMode {
    // The authored arg (and `Transform` field) this mode edits.
    pub(crate) fn arg_key(self) -> &'static str {
        match self {
            Self::Translate => "position",
            Self::Rotate => "rotation_deg",
            Self::Scale => "scale",
        }
    }

    pub(crate) fn caption(self) -> &'static str {
        match self {
            Self::Translate => "move",
            Self::Rotate => "rotate",
            Self::Scale => "scale",
        }
    }

    // Tip shape per mode: soft square, circle, hard square.
    fn tip_radius(self) -> f32 {
        match self {
            Self::Translate => 2.0,
            Self::Rotate => TIP_PX * 0.5,
            Self::Scale => 0.0,
        }
    }
}

// Wrap a degree delta into (-180, 180], keeping per-frame rotation steps
// continuous across the atan2 seam.
pub(crate) fn wrap_deg(d: f32) -> f32 {
    -((-d + 180.0).rem_euclid(360.0) - 180.0)
}

// The gizmo's screen geometry for one frame: the projected origin and the
// three projected axis-tip positions.
#[derive(Debug, Clone, Copy, PartialEq)]
pub(crate) struct Layout {
    pub origin: [f32; 2],
    pub tips: [[f32; 2]; 3],
    // World-space length each on-screen axis run represents (the tips sit at
    // `origin_world + AXES[i] * world_len`).
    pub world_len: f32,
}

// Compute the gizmo layout for a world-space origin: project the origin, size
// the axes to a constant screen length from the world-per-pixel scale at the
// origin's depth, and project each tip. `None` when the origin (or any tip)
// falls at or behind the camera plane, or the viewport is degenerate.
pub(crate) fn layout(
    view: &[[f32; 4]; 4],
    fov_y_radians: f32,
    viewport: [f32; 2],
    origin_world: [f32; 3],
) -> Option<Layout> {
    let (vw, vh) = (viewport[0], viewport[1]);
    if vw <= 0.0 || vh <= 0.0 {
        return None;
    }
    let tan_half = (fov_y_radians * 0.5).tan();
    if tan_half <= 0.0 || !tan_half.is_finite() {
        return None;
    }
    let project = |p: [f32; 3]| -> Option<[f32; 2]> {
        let v = [
            view[0][0] * p[0] + view[1][0] * p[1] + view[2][0] * p[2] + view[3][0],
            view[0][1] * p[0] + view[1][1] * p[1] + view[2][1] * p[2] + view[3][1],
            view[0][2] * p[0] + view[1][2] * p[1] + view[2][2] * p[2] + view[3][2],
        ];
        let depth = -v[2];
        if depth <= 1e-4 {
            return None;
        }
        let aspect = vw / vh;
        Some([
            (v[0] / (depth * tan_half * aspect) + 1.0) * 0.5 * vw,
            (1.0 - v[1] / (depth * tan_half)) * 0.5 * vh,
        ])
    };

    let origin = project(origin_world)?;
    // World units per pixel at the origin's depth (vertical), sizing the axis
    // runs to a constant screen length.
    let depth = {
        let v = [view[0][2] * origin_world[0]
            + view[1][2] * origin_world[1]
            + view[2][2] * origin_world[2]
            + view[3][2]];
        -v[0]
    };
    let world_len = AXIS_PX * (depth * tan_half * 2.0) / vh;
    let mut tips = [[0.0f32; 2]; 3];
    for (i, axis) in AXES.iter().enumerate() {
        tips[i] = project([
            origin_world[0] + axis[0] * world_len,
            origin_world[1] + axis[1] * world_len,
            origin_world[2] + axis[2] * world_len,
        ])?;
    }
    Some(Layout {
        origin,
        tips,
        world_len,
    })
}

// The axis whose tip handle contains `mouse`, if any. Later axes win a tie,
// matching the draw order (they are drawn on top).
pub(crate) fn hit_axis(layout: &Layout, mouse: [f32; 2]) -> Option<usize> {
    let half = TIP_PX * 0.5 + TIP_GRAB_PX;
    (0..3).rev().find(|&i| {
        (layout.tips[i][0] - mouse[0]).abs() <= half && (layout.tips[i][1] - mouse[1]).abs() <= half
    })
}

// The parameter `t` of the point on the world-space axis line
// `origin + t * axis` closest to `ray`. `None` when the axis is (nearly)
// parallel to the ray, where the drag has no stable solution.
pub(crate) fn axis_drag_t(origin: [f32; 3], axis: [f32; 3], ray: &PickRay) -> Option<f32> {
    let dot = |a: [f32; 3], b: [f32; 3]| a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
    let w0 = [
        origin[0] - ray.origin[0],
        origin[1] - ray.origin[1],
        origin[2] - ray.origin[2],
    ];
    // Both directions are unit-length, so the closest-point system reduces to
    // denom = 1 - (axis . dir)^2.
    let b = dot(axis, ray.dir);
    let denom = 1.0 - b * b;
    if denom < 1e-5 {
        return None;
    }
    let d0 = dot(axis, w0);
    let e = dot(ray.dir, w0);
    Some((b * e - d0) / denom)
}

// The caption's offset from the origin, and its box.
const CAPTION_AT: [f32; 2] = [12.0, -26.0];
const CAPTION_SIZE: [f32; 2] = [80.0, 20.0];

// Lay the dotted runs from just outside the origin to each tip, center the
// tip handles (shaped by the mode) on their projected points, and caption the
// mode beside the origin.
pub(crate) fn push(shapes: &mut Shapes<'_>, layout: &Layout, mode: GizmoMode) {
    for (axis, tint) in AXIS_TINTS.iter().enumerate() {
        for seg in 0..SEGMENTS {
            // Segments start away from the origin so the runs do not overdraw
            // the object's center, and stop short of the tip handle.
            let f = (seg as f32 + 1.0) / (SEGMENTS as f32 + 1.0);
            let x = layout.origin[0] + (layout.tips[axis][0] - layout.origin[0]) * f;
            let y = layout.origin[1] + (layout.tips[axis][1] - layout.origin[1]) * f;
            push_square(shapes, [x, y], SEGMENT_PX, 0.0, *tint);
        }
        push_square(shapes, layout.tips[axis], TIP_PX, mode.tip_radius(), *tint);
    }
    shapes.push(Shape {
        rect: [
            layout.origin[0] + CAPTION_AT[0],
            layout.origin[1] + CAPTION_AT[1],
            CAPTION_SIZE[0],
            CAPTION_SIZE[1],
        ],
        text: mode.caption(),
        text_color: theme::LABEL_DIM,
        align: TextAlign::Left,
        ..Shape::default()
    });
}

fn push_square(shapes: &mut Shapes<'_>, center: [f32; 2], size: f32, radius: f32, tint: [f32; 4]) {
    shapes.push(Shape {
        rect: [center[0] - size * 0.5, center[1] - size * 0.5, size, size],
        fill: tint,
        radius,
        ..Shape::default()
    });
}

#[cfg(test)]
mod tests {
    use super::*;
    use concinnity_engine::gfx::camera::view_matrix;

    const VP: [f32; 2] = [1280.0, 720.0];
    const FOV: f32 = core::f32::consts::FRAC_PI_2;

    // Each mode names the authored arg it edits and the caption the mode label
    // shows, and no two modes share either -- the arg key is what the drag
    // writes back, so a collision would edit the wrong field.
    #[test]
    fn every_mode_has_its_own_arg_key_and_caption() {
        let modes = [GizmoMode::Translate, GizmoMode::Rotate, GizmoMode::Scale];
        let keys: Vec<&str> = modes.iter().map(|m| m.arg_key()).collect();
        let captions: Vec<&str> = modes.iter().map(|m| m.caption()).collect();

        assert_eq!(keys, ["position", "rotation_deg", "scale"]);
        assert_eq!(captions, ["move", "rotate", "scale"]);
        for list in [&keys, &captions] {
            let mut seen = list.clone();
            seen.sort_unstable();
            seen.dedup();
            assert_eq!(seen.len(), modes.len(), "{list:?} repeats");
            assert!(list.iter().all(|s| !s.is_empty()));
        }
        assert_eq!(GizmoMode::default(), GizmoMode::Translate);
    }

    #[test]
    fn layout_spans_a_constant_screen_length() {
        let view = view_matrix([0.0; 3], 0.0, 0.0);
        // For a camera facing -Z: +X projects right, +Y up, at ~AXIS_PX.
        for depth in [5.0f32, 50.0] {
            let l = layout(&view, FOV, VP, [0.0, 0.0, -depth]).unwrap();
            assert!((l.origin[0] - 640.0).abs() < 0.5);
            assert!((l.origin[1] - 360.0).abs() < 0.5);
            let dx = l.tips[0][0] - l.origin[0];
            assert!((dx - AXIS_PX).abs() < 1.0, "X tip {AXIS_PX} px right: {dx}");
            let dy = l.tips[1][1] - l.origin[1];
            assert!((dy + AXIS_PX).abs() < 1.0, "Y tip {AXIS_PX} px up: {dy}");
            // Z points at the camera: its tip stays at the origin's pixel.
            assert!((l.tips[2][0] - l.origin[0]).abs() < 1.0);
        }
    }

    #[test]
    fn layout_hides_behind_the_camera() {
        let view = view_matrix([0.0; 3], 0.0, 0.0);
        assert_eq!(layout(&view, FOV, VP, [0.0, 0.0, 5.0]), None);
        assert_eq!(layout(&view, FOV, [0.0, 720.0], [0.0, 0.0, -5.0]), None);
    }

    #[test]
    fn hit_axis_takes_the_tip_squares() {
        let view = view_matrix([0.0; 3], 0.0, 0.0);
        let l = layout(&view, FOV, VP, [0.0, 0.0, -5.0]).unwrap();
        assert_eq!(hit_axis(&l, l.tips[0]), Some(0));
        assert_eq!(hit_axis(&l, l.tips[1]), Some(1));
        assert_eq!(
            hit_axis(&l, [l.tips[0][0] + TIP_PX, l.tips[0][1]]),
            None,
            "outside the grab slop"
        );
    }

    #[test]
    fn axis_drag_follows_the_mouse_ray() {
        // Camera at origin facing -Z, object 10 ahead; drag along world X.
        let origin = [0.0, 0.0, -10.0];
        let axis = [1.0, 0.0, 0.0];
        // A ray straight at the object: closest point is t = 0.
        let straight = PickRay {
            origin: [0.0; 3],
            dir: [0.0, 0.0, -1.0],
        };
        let t0 = axis_drag_t(origin, axis, &straight).unwrap();
        assert!(t0.abs() < 1e-5);
        // Aim 45 degrees right: the closest point on the axis is x = 10.
        let inv = 0.5f32.sqrt();
        let right = PickRay {
            origin: [0.0; 3],
            dir: [inv, 0.0, -inv],
        };
        let t1 = axis_drag_t(origin, axis, &right).unwrap();
        assert!((t1 - 10.0).abs() < 1e-3, "{t1}");
    }

    #[test]
    fn axis_parallel_to_the_ray_has_no_drag_solution() {
        let ray = PickRay {
            origin: [0.0; 3],
            dir: [0.0, 0.0, -1.0],
        };
        assert_eq!(axis_drag_t([0.0, 0.0, -10.0], [0.0, 0.0, -1.0], &ray), None);
    }

    #[test]
    fn wrap_deg_stays_in_half_open_range() {
        assert_eq!(wrap_deg(0.0), 0.0);
        assert_eq!(wrap_deg(180.0), 180.0);
        assert!((wrap_deg(190.0) + 170.0).abs() < 1e-4);
        assert!((wrap_deg(-190.0) - 170.0).abs() < 1e-4);
        assert!((wrap_deg(720.0)).abs() < 1e-3);
    }

    // Each axis draws its segments, then its tip centered on the projected
    // point, and the caption names the mode.
    #[test]
    fn push_draws_each_axis_run_then_the_caption() {
        let view = view_matrix([0.0; 3], 0.0, 0.0);
        let l = layout(&view, FOV, VP, [0.0, 0.0, -5.0]).unwrap();
        let mut list = Vec::new();
        let mut shapes = Shapes::new(&mut list);
        push(&mut shapes, &l, GizmoMode::Scale);
        shapes.finish();

        assert_eq!(list.len(), 3 * (SEGMENTS + 1) + 1);
        let tip = &list[SEGMENTS];
        assert_eq!(tip.fill, AXIS_TINTS[0]);
        assert!((tip.rect[0] + TIP_PX * 0.5 - l.tips[0][0]).abs() < 1e-3);
        assert!((tip.rect[1] + TIP_PX * 0.5 - l.tips[0][1]).abs() < 1e-3);
        assert_eq!(list.last().unwrap().text, "scale");
    }
}
