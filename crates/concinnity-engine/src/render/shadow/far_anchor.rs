//! Where the far shadow centers over a planet: under the camera, moved only as
//! far toward the ground the camera sees as it takes to cover it, so a camera
//! flying high still finds the far shadow on the ground in view, and one near
//! the ground keeps it the same whichever way it looks.

use crate::render::shadow::csm::CameraBasis;
use crate::render::shadow::far::{FAR_FADE_FRACTION, FAR_RECENTER_FRACTION};
use crate::render::shadow::schedule::CascadeCamera;
use concinnity_core::math::vec3::{add, dot, length, scale, sub};
use concinnity_core::math::{asin, cos, sin, sqrt};

/// The ground a planet world's camera flies over, as the far shadow sees it:
/// a sphere through the ground right under the camera, about the planet's
/// center.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct GroundSphere {
    /// The planet's center, in the space the frame renders in.
    pub center: [f32; 3],
    /// How far the ground under the camera lies from `center`.
    pub radius: f32,
}

/// The point on `ground` the far view of `reach` centers on for `camera`: the
/// ground under the camera while the nearest ground it sees lies within half
/// the view's whole coverage of it, and past that moved along the way the
/// camera looks by twice the nearest ground's distance past that half, so
/// the view covers the ground in sight from its near edge on, however far the
/// camera drifts before the view re-centers. It stays near enough to the
/// ground under the camera that the ground the cascades of `cascade_reach`
/// shade stays inside the fade too. A camera in the ground, or seeing none,
/// keeps the far view under it. Looking level, a camera higher than about 1.1
/// times the reach (4.5 km for 4 km) first sees ground farther out than the
/// view can cover.
pub(crate) fn far_anchor(
    camera: &CascadeCamera,
    ground: &GroundSphere,
    reach: f32,
    cascade_reach: f32,
) -> [f32; 3] {
    let rel = sub(camera.position, ground.center);
    let distance = length(rel);
    let r = ground.radius;
    if distance <= r || r <= 0.0 {
        return camera.position;
    }
    let up = scale(rel, 1.0 / distance);
    let under = add(ground.center, scale(up, r));
    let forward = CameraBasis::of_view(&camera.view, camera.position).forward();
    let level = sub(forward, scale(up, dot(forward, up)));
    let level_len = length(level);
    if level_len < 1e-4 {
        return under;
    }
    let heading = scale(level, 1.0 / level_len);
    // The view's lowest ray, below level by its pitch and half its height.
    let lowest = asin(-dot(forward, up)) + 0.5 * camera.fov_y_rad;
    let nearest = if lowest >= core::f32::consts::FRAC_PI_2 {
        0.0
    } else {
        let ray = sub(scale(heading, cos(lowest)), scale(up, sin(lowest)));
        match hit(rel, ray, r) {
            Some(p) => length(sub(add(ground.center, p), under)),
            None => return under,
        }
    };
    // Whole coverage, less what the camera may drift before a redraw.
    let inside = reach * (1.0 - FAR_FADE_FRACTION - FAR_RECENTER_FRACTION);
    // The ground the cascades shade lies within this of the ground under the
    // camera.
    let height = distance - r;
    let shaded = sqrt((cascade_reach * cascade_reach - height * height).max(0.0));
    let most = (inside - shaded).max(0.0);
    let chord = (2.0 * nearest - inside).clamp(0.0, most);
    let angle = 2.0 * asin((0.5 * chord / r).min(1.0));
    add(
        ground.center,
        scale(add(scale(up, cos(angle)), scale(heading, sin(angle))), r),
    )
}

// Where the ray from `from` (relative to the sphere's center) along the unit
// `dir` first meets the sphere of `radius` about the center, relative to it.
fn hit(from: [f32; 3], dir: [f32; 3], radius: f32) -> Option<[f32; 3]> {
    let b = dot(from, dir);
    let c = dot(from, from) - radius * radius;
    let disc = b * b - c;
    if disc < 0.0 {
        return None;
    }
    let t = -b - sqrt(disc);
    (t > 0.0).then(|| add(from, scale(dir, t)))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::gfx::projection::look_at;

    const RADIUS: f32 = 50_000.0;
    const REACH: f32 = 4_000.0;
    const CASCADES: f32 = 250.0;
    const FADE: f32 = FAR_FADE_FRACTION;

    const GROUND: GroundSphere = GroundSphere {
        center: [0.0, -RADIUS, 0.0],
        radius: RADIUS,
    };

    // A camera `height` over the ground at the frame's origin, looking along
    // `look` from it.
    fn camera(height: f32, look: [f32; 3]) -> CascadeCamera {
        let position = [0.0, height, 0.0];
        CascadeCamera {
            view: look_at(position, add(position, look), [0.0, 1.0, 0.0]),
            position,
            fov_y_rad: 70.0_f32.to_radians(),
            aspect: 16.0 / 9.0,
            near: 0.1,
            view_distance: None,
        }
    }

    fn anchor(camera: &CascadeCamera) -> [f32; 3] {
        far_anchor(camera, &GROUND, REACH, CASCADES)
    }

    fn distance(a: [f32; 3], b: [f32; 3]) -> f32 {
        length(sub(a, b))
    }

    // Where the ray from `camera` `below` radians under level along -Z meets
    // the ground.
    fn seen(camera: &CascadeCamera, below: f32) -> [f32; 3] {
        let rel = sub(camera.position, GROUND.center);
        let dir = [0.0, -sin(below), -cos(below)];
        add(
            GROUND.center,
            hit(rel, dir, RADIUS).expect("the ray meets the ground"),
        )
    }

    // A level camera 3 km up first sees the ground kilometers out; the far
    // view moves out over it, so that ground lies inside the fade with room
    // for the camera to drift, and so does the ground 2 km past it.
    #[test]
    fn a_high_level_camera_finds_the_far_view_on_the_ground_it_sees() {
        let high = camera(3_000.0, [0.0, 0.0, -1.0]);
        let center = anchor(&high);
        let nearest = seen(&high, 35.0_f32.to_radians());
        assert!(distance(nearest, [0.0; 3]) > 3_000.0);
        let inside = REACH * (1.0 - FADE - FAR_RECENTER_FRACTION);
        assert!(distance(center, nearest) <= inside, "{center:?}");
        let from = distance(nearest, [0.0; 3]);
        for k in 0..=20 {
            let chord = from + k as f32 * 100.0;
            let angle = 2.0 * asin(0.5 * chord / RADIUS);
            let ground = add(
                GROUND.center,
                [0.0, RADIUS * cos(angle), -RADIUS * sin(angle)],
            );
            let gap = distance(center, ground);
            assert!(gap <= REACH * (1.0 - FADE), "{chord}: {gap}");
        }
        assert!(distance(center, [0.0; 3]) <= REACH);
        assert!((distance(center, GROUND.center) - RADIUS).abs() < 1.0);
    }

    // Near the ground, the ground the camera sees is already covered: the far
    // view stays under the camera whichever way it looks, so turning never
    // re-centers it.
    #[test]
    fn near_the_ground_the_far_view_stays_under_the_camera() {
        for height in [2.0, 200.0, 800.0] {
            for look in [[0.0, 0.0, -1.0], [1.0, -0.3, 0.0], [-0.7, 0.05, 0.7]] {
                let center = anchor(&camera(height, look));
                assert!(
                    distance(center, [0.0; 3]) < 1.0,
                    "{height} {look:?}: {center:?}"
                );
            }
        }
    }

    // A camera just under the cascades' reach, looking out at ground near its
    // horizon, moves the far view ahead, and the ground the cascades shade
    // still stays inside the fade after the camera drifts the re-centering
    // distance.
    #[test]
    fn the_cascades_ground_stays_inside_the_fade_between_redraws() {
        let height = 240.0;
        let narrow = CascadeCamera {
            fov_y_rad: 10.0_f32.to_radians(),
            ..camera(height, [0.0, -0.014, -1.0])
        };
        let center = anchor(&narrow);
        assert!(distance(center, [0.0; 3]) > 100.0, "{center:?}");
        let drift = FAR_RECENTER_FRACTION * REACH;
        let shaded = sqrt(CASCADES * CASCADES - height * height);
        for i in 0..16 {
            let a = i as f32 * core::f32::consts::TAU / 16.0;
            let ring = [shaded * cos(a), 0.0, shaded * sin(a)];
            let gap = distance(center, ring) + drift;
            assert!(gap <= REACH * (1.0 - FADE) + 1.0, "{gap}");
        }
    }

    // Looking straight down, or at the sky only, the far view stays under the
    // camera; a camera in the ground keeps it on itself.
    #[test]
    fn with_no_ground_ahead_the_far_view_stays_under_the_camera() {
        for look in [[0.0, -1.0, -0.001], [0.0, 1.0, -0.2]] {
            let center = anchor(&camera(500.0, look));
            assert!(distance(center, [0.0; 3]) < 1.0, "{look:?}: {center:?}");
        }
        let buried = camera(-5.0, [0.0, 0.0, -1.0]);
        assert_eq!(anchor(&buried), buried.position);
    }
}
