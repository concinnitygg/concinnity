// The flight laid over the planet: the course it follows, the points its legs
// run between, and where the camera looks along the way.

use concinnity::components::{CameraTravel, CameraTurn, Ease, Planet};
use concinnity_core::math::narrow_f32;

use super::flight::{self, Fix, PHASES, smoothstep};
use concinnity::math::dvec3::{DVec3, add, cross, dot, length, scale, sub};

/// Degrees clockwise from north the course sets out on from where the world
/// starts.
const HEADING_DEG: f64 = 24.0;
/// How often the path is pinned to the flight: often enough that a leg over
/// the meadow strays from the ground under it by centimeters.
const STEP_SECONDS: f64 = 0.2;
const GROUND_PITCH_DEG: f64 = -6.0;
// How far over the horizon the camera looks while it watches the sun: the
// ground stays in the lower half, and from orbit the whole disc stays in view.
const LIFT_DEG: f64 = 4.0;
// How long the camera takes to turn from the course to the sun.
const LOOK_SECONDS: f64 = 8.0;

// The authored axes: the world starts at latitude 0, longitude 0, along +Y
// from the center, with north along -Z and east along +X.
const START_UP: DVec3 = [0.0, 1.0, 0.0];
const NORTH: DVec3 = [0.0, 0.0, -1.0];
const EAST: DVec3 = [1.0, 0.0, 0.0];

/// The great circle the camera follows round the planet from where the world
/// starts. Along a great circle the camera's level frame turns with the
/// course, so a heading taken against the course holds for the whole flight.
struct Course {
    forward: DVec3,
}

impl Course {
    fn new(heading_deg: f64) -> Self {
        let h = heading_deg.to_radians();
        Self {
            forward: add(scale(NORTH, h.cos()), scale(EAST, h.sin())),
        }
    }

    /// The unit direction from the center `arc` radians along the course.
    fn up(&self, arc: f64) -> DVec3 {
        add(scale(START_UP, arc.cos()), scale(self.forward, arc.sin()))
    }

    /// The unit direction of travel `arc` radians along the course.
    fn ahead(&self, arc: f64) -> DVec3 {
        add(scale(START_UP, -arc.sin()), scale(self.forward, arc.cos()))
    }
}

/// The path from the start: where the camera begins and how it is posed, and
/// the legs it travels and turns through.
pub(super) struct Path {
    pub(super) start: [f32; 3],
    pub(super) yaw_deg: f32,
    pub(super) pitch_deg: f32,
    pub(super) travel: Vec<CameraTravel>,
    pub(super) turn: Vec<CameraTurn>,
}

impl Path {
    pub(super) fn new(planet: &Planet, sun: impl Fn(f64) -> DVec3) -> Self {
        let shape = planet.shape();
        let course = Course::new(HEADING_DEG);
        let fixes = flight::fixes(shape.radius, STEP_SECONDS);
        let points: Vec<DVec3> = fixes
            .iter()
            .map(|fix| {
                let up = course.up(fix.arc);
                add(
                    shape.center,
                    scale(up, shape.radius + shape.height(up) + fix.height),
                )
            })
            .collect();
        let views = views(&fixes, &course, sun, |up| shape.radius + shape.height(up));

        let mut travel = Vec::with_capacity(fixes.len() - 1);
        let mut turn = Vec::with_capacity(fixes.len() - 1);
        for i in 1..fixes.len() {
            let (from, to) = (fixes[i - 1], fixes[i]);
            let seconds = narrow_f32(to.seconds - from.seconds);
            let step = sub(points[i], points[i - 1]);
            travel.push(CameraTravel {
                direction: step.map(narrow_f32),
                distance: narrow_f32(length(step)),
                seconds,
                ease: Ease::Linear,
                ..Default::default()
            });
            let [yaw, pitch] = views[i];
            turn.push(CameraTurn {
                yaw_deg: Some(narrow_f32(yaw)),
                pitch_deg: Some(narrow_f32(pitch)),
                seconds,
                ease: Ease::Linear,
                ..Default::default()
            });
        }
        let [yaw, pitch] = views[0];
        Self {
            start: points[0].map(narrow_f32),
            yaw_deg: narrow_f32(yaw),
            pitch_deg: narrow_f32(pitch),
            travel,
            turn,
        }
    }
}

// The camera's yaw and pitch in degrees at every fix. It looks down the course
// over the meadow, then turns to watch the sun, aimed a little over the
// horizon below it. Yaw is counterclockwise from the course's start heading,
// which along the great circle stays the same angle from the course.
fn views(
    fixes: &[Fix],
    course: &Course,
    sun: impl Fn(f64) -> DVec3,
    ground_radius: impl Fn(DVec3) -> f64,
) -> Vec<[f64; 2]> {
    let look_from = look_from();
    let mut toward_sun: Option<f64> = None;
    fixes
        .iter()
        .map(|fix| {
            let watch = if PHASES[fix.phase].watches_sun {
                smoothstep((fix.seconds - look_from) / LOOK_SECONDS)
            } else {
                0.0
            };
            if watch == 0.0 {
                toward_sun = None;
                return [-HEADING_DEG, GROUND_PITCH_DEG];
            }
            let up = course.up(fix.arc);
            let ahead = course.ahead(fix.arc);
            let toward = sun(fix.seconds);
            let left = cross(up, ahead);
            let raw = dot(toward, left).atan2(dot(toward, ahead)).to_degrees();
            // Unwrapped from the last fix, so the camera never turns the long
            // way round when the sun crosses behind it.
            let bearing = toward_sun.map_or(raw, |last| last + wrap_deg(raw - last));
            toward_sun = Some(bearing);
            let ground = ground_radius(up);
            let dip = (ground / (ground + fix.height)).acos().to_degrees();
            let pitch = GROUND_PITCH_DEG + (LIFT_DEG - dip - GROUND_PITCH_DEG) * watch;
            [-HEADING_DEG + bearing * watch, pitch]
        })
        .collect()
}

// When the camera starts to turn toward the sun: as the first phase that
// watches it begins.
fn look_from() -> f64 {
    PHASES
        .iter()
        .take_while(|p| !p.watches_sun)
        .map(|p| p.seconds)
        .sum()
}

// `deg` brought into [-180, 180).
fn wrap_deg(deg: f64) -> f64 {
    (deg + 180.0).rem_euclid(360.0) - 180.0
}
