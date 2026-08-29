// src/editor/framing.rs
//
// Pure camera-framing math: selection bounds union, the distance that fits a
// bounding sphere in the view frustum, and the eased pose interpolation the
// glide drive steps through. No world or hook access, so all of it is
// unit-tested directly.

// A free camera pose, the unit the glide and the bookmarks move around.
#[derive(Debug, Clone, Copy, PartialEq, serde::Serialize, serde::Deserialize)]
pub(crate) struct CameraPose {
    pub position: [f32; 3],
    pub yaw: f32,
    pub pitch: f32,
}

// Half-extent used when a selection member has no AABB (a billboard-backed
// light or volume): its position is framed as a box this big.
pub(crate) const POINT_PAD: f32 = 0.5;

// Never frame closer than this, so a zero-extent selection still lands at a
// workable distance.
const MIN_RADIUS: f32 = 0.1;
// Breathing room around the fitted sphere.
const FIT_MARGIN: f32 = 1.05;

// The union of a set of world-space AABBs; `None` for an empty set.
pub(crate) fn union_bounds(
    boxes: impl IntoIterator<Item = ([f32; 3], [f32; 3])>,
) -> Option<([f32; 3], [f32; 3])> {
    let mut out: Option<([f32; 3], [f32; 3])> = None;
    for (mn, mx) in boxes {
        let (omn, omx) = out.get_or_insert((mn, mx));
        for a in 0..3 {
            omn[a] = omn[a].min(mn[a]);
            omx[a] = omx[a].max(mx[a]);
        }
    }
    out
}

// A point member's padded AABB.
pub(crate) fn pad_point(p: [f32; 3]) -> ([f32; 3], [f32; 3]) {
    (
        [p[0] - POINT_PAD, p[1] - POINT_PAD, p[2] - POINT_PAD],
        [p[0] + POINT_PAD, p[1] + POINT_PAD, p[2] + POINT_PAD],
    )
}

// Center and bounding-sphere radius of an AABB.
pub(crate) fn bounding_sphere(mn: [f32; 3], mx: [f32; 3]) -> ([f32; 3], f32) {
    let center = [
        (mn[0] + mx[0]) * 0.5,
        (mn[1] + mx[1]) * 0.5,
        (mn[2] + mx[2]) * 0.5,
    ];
    let half = [
        (mx[0] - mn[0]) * 0.5,
        (mx[1] - mn[1]) * 0.5,
        (mx[2] - mn[2]) * 0.5,
    ];
    let radius = (half[0] * half[0] + half[1] * half[1] + half[2] * half[2]).sqrt();
    (center, radius)
}

// The camera distance at which a sphere of `radius` fits inside both frustum
// planes: the tighter of the vertical and horizontal half-angles governs.
pub(crate) fn fit_distance(radius: f32, fov_y_radians: f32, aspect: f32) -> f32 {
    let radius = radius.max(MIN_RADIUS);
    let half_v = (fov_y_radians * 0.5).clamp(0.01, std::f32::consts::FRAC_PI_2 - 0.01);
    let half_h =
        ((half_v.tan() * aspect.max(0.1)).atan()).clamp(0.01, std::f32::consts::FRAC_PI_2 - 0.01);
    let half = half_v.min(half_h);
    radius * FIT_MARGIN / half.sin()
}

// The forward unit vector for a yaw/pitch pair, matching the fly camera's
// free-fly basis (yaw 0 faces -Z, positive pitch looks up).
pub(crate) fn forward(yaw: f32, pitch: f32) -> [f32; 3] {
    let cp = pitch.cos();
    [-yaw.sin() * cp, pitch.sin(), -yaw.cos() * cp]
}

// The pose that frames a bounding sphere: the camera keeps its orientation and
// backs away from the center along its own view direction.
pub(crate) fn frame_pose(
    current: &CameraPose,
    center: [f32; 3],
    radius: f32,
    fov_y_radians: f32,
    aspect: f32,
) -> CameraPose {
    let d = fit_distance(radius, fov_y_radians, aspect);
    let f = forward(current.yaw, current.pitch);
    CameraPose {
        position: [
            center[0] - f[0] * d,
            center[1] - f[1] * d,
            center[2] - f[2] * d,
        ],
        yaw: current.yaw,
        pitch: current.pitch,
    }
}

// Smoothstep ease for the glide's normalized clock.
pub(crate) fn ease(t: f32) -> f32 {
    let t = t.clamp(0.0, 1.0);
    t * t * (3.0 - 2.0 * t)
}

// Wrap an angle difference to the shortest signed arc.
fn shortest_arc(delta: f32) -> f32 {
    use std::f32::consts::{PI, TAU};
    (delta + PI).rem_euclid(TAU) - PI
}

// Interpolate between two poses at eased parameter `s` in [0, 1]. Yaw takes
// the shortest arc so a glide never spins the long way around.
pub(crate) fn lerp_pose(from: &CameraPose, to: &CameraPose, s: f32) -> CameraPose {
    let lerp = |a: f32, b: f32| a + (b - a) * s;
    CameraPose {
        position: [
            lerp(from.position[0], to.position[0]),
            lerp(from.position[1], to.position[1]),
            lerp(from.position[2], to.position[2]),
        ],
        yaw: from.yaw + shortest_arc(to.yaw - from.yaw) * s,
        pitch: lerp(from.pitch, to.pitch),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn union_bounds_covers_every_box() {
        assert_eq!(union_bounds([]), None);
        let u = union_bounds([
            ([-1.0, 0.0, 0.0], [1.0, 2.0, 1.0]),
            ([0.0, -3.0, 0.5], [0.5, 0.0, 4.0]),
        ])
        .unwrap();
        assert_eq!(u.0, [-1.0, -3.0, 0.0]);
        assert_eq!(u.1, [1.0, 2.0, 4.0]);
    }

    #[test]
    fn pad_point_is_symmetric() {
        let (mn, mx) = pad_point([1.0, 2.0, 3.0]);
        assert_eq!(mn, [1.0 - POINT_PAD, 2.0 - POINT_PAD, 3.0 - POINT_PAD]);
        assert_eq!(mx, [1.0 + POINT_PAD, 2.0 + POINT_PAD, 3.0 + POINT_PAD]);
    }

    #[test]
    fn fit_distance_scales_with_radius_and_never_degenerates() {
        let fov = 60.0_f32.to_radians();
        let d1 = fit_distance(1.0, fov, 16.0 / 9.0);
        let d2 = fit_distance(2.0, fov, 16.0 / 9.0);
        assert!((d2 - 2.0 * d1).abs() < 1e-4, "linear in radius");
        // A zero-extent selection clamps to the minimum radius.
        let d0 = fit_distance(0.0, fov, 16.0 / 9.0);
        assert!(d0.is_finite() && d0 > 0.0);
        assert!((d0 - fit_distance(MIN_RADIUS, fov, 16.0 / 9.0)).abs() < 1e-6);
    }

    #[test]
    fn fit_distance_respects_the_narrow_axis() {
        let fov = 60.0_f32.to_radians();
        // A portrait viewport is horizontally tighter, so the camera must back
        // off further than in a wide one.
        assert!(fit_distance(1.0, fov, 0.5) > fit_distance(1.0, fov, 2.0));
    }

    #[test]
    fn frame_pose_looks_at_the_center() {
        let cur = CameraPose {
            position: [10.0, 5.0, 3.0],
            yaw: 0.7,
            pitch: -0.3,
        };
        let center = [1.0, 2.0, 3.0];
        let to = frame_pose(&cur, center, 2.0, 60.0_f32.to_radians(), 16.0 / 9.0);
        // The center sits along the framed camera's forward at fit distance.
        let f = forward(to.yaw, to.pitch);
        let d = fit_distance(2.0, 60.0_f32.to_radians(), 16.0 / 9.0);
        for a in 0..3 {
            assert!((to.position[a] + f[a] * d - center[a]).abs() < 1e-4);
        }
        assert_eq!((to.yaw, to.pitch), (cur.yaw, cur.pitch));
    }

    #[test]
    fn ease_is_smooth_and_clamped() {
        assert_eq!(ease(-1.0), 0.0);
        assert_eq!(ease(0.0), 0.0);
        assert_eq!(ease(1.0), 1.0);
        assert_eq!(ease(2.0), 1.0);
        assert!((ease(0.5) - 0.5).abs() < 1e-6);
        assert!(ease(0.25) < 0.25, "eases in");
        assert!(ease(0.75) > 0.75, "eases out");
    }

    #[test]
    fn lerp_pose_takes_the_short_yaw_arc() {
        use std::f32::consts::PI;
        let from = CameraPose {
            position: [0.0; 3],
            yaw: PI - 0.1,
            pitch: 0.0,
        };
        let to = CameraPose {
            position: [0.0; 3],
            yaw: -PI + 0.1,
            pitch: 0.0,
        };
        // Short way crosses PI (0.2 rad total), not back through zero.
        let mid = lerp_pose(&from, &to, 0.5);
        assert!(
            (mid.yaw.abs() - PI).abs() < 1e-4,
            "midpoint sits at the wrap seam: {}",
            mid.yaw
        );
        let end = lerp_pose(&from, &to, 1.0);
        assert!((shortest_arc(end.yaw - to.yaw)).abs() < 1e-4);
    }
}
