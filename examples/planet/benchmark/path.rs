// The flight laid over the planet: the course the marble follows, the points
// its legs run between, and how the camera films it along the way.

use concinnity::components::{CameraBoom, CameraTravel, CameraTurn, Ease, Planet};
use concinnity_core::math::narrow_f32;

use super::flight::{self, Fix};
use super::framing::{self, Aim, Shot};
use concinnity::math::dvec3::{DVec3, add, cross, dot, length, scale, sub};

/// Degrees clockwise from north the course sets out on from where the world
/// starts: across the shore a couple of hundred meters east, out to sea.
const HEADING_DEG: f64 = 96.0;
/// How far back along the course from where the world starts the marble
/// sets off, in meters: far enough inland for a long run through the meadow.
const START_BACK: f64 = 150.0;
/// How often the path is pinned to the flight: often enough that a leg over
/// the meadow strays from the ground under it by centimeters.
const STEP_SECONDS: f64 = 0.2;

// The authored axes: the world starts at latitude 0, longitude 0, along +Y
// from the center, with north along -Z and east along +X.
const START_UP: DVec3 = [0.0, 1.0, 0.0];
const NORTH: DVec3 = [0.0, 0.0, -1.0];
const EAST: DVec3 = [1.0, 0.0, 0.0];

/// The great circle the marble follows round the planet through where the
/// world starts. Along a great circle the camera's level frame turns with the
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

/// The path from the start: where the marble begins and how the camera is
/// aimed at it, and the legs they travel, turn and frame through.
pub(super) struct Path {
    pub(super) start: [f32; 3],
    pub(super) yaw_deg: f32,
    pub(super) pitch_deg: f32,
    pub(super) travel: Vec<CameraTravel>,
    pub(super) turn: Vec<CameraTurn>,
    pub(super) boom: Vec<CameraBoom>,
}

impl Path {
    /// The path over `planet`, flooded to `sea_level` meters above its
    /// radius when it has a sea, with the sun toward `sun` seconds into the
    /// run.
    pub(super) fn new(planet: &Planet, sea_level: Option<f64>, sun: impl Fn(f64) -> DVec3) -> Self {
        let shape = planet.shape();
        let course = Course::new(HEADING_DEG);
        let fixes = flight::fixes(shape.radius, STEP_SECONDS, -START_BACK / shape.radius);
        let floor = sea_level.unwrap_or(f64::NEG_INFINITY);
        let surface = |up: DVec3| shape.radius + shape.height(up).max(floor);
        let points: Vec<DVec3> = fixes
            .iter()
            .map(|fix| {
                let up = course.up(fix.arc);
                add(shape.center, scale(up, surface(up) + fix.height))
            })
            .collect();
        let views: Vec<View> = fixes
            .iter()
            .map(|fix| {
                let up = course.up(fix.arc);
                let sky = Sky {
                    up,
                    ahead: course.ahead(fix.arc),
                    sun: sun(fix.seconds),
                    dip: (surface(up) / (surface(up) + fix.height))
                        .acos()
                        .to_degrees(),
                };
                view(fix, &sky)
            })
            .collect();

        let legs = fixes.len() - 1;
        let mut travel = Vec::with_capacity(legs);
        let mut turn = Vec::with_capacity(legs);
        let mut boom = Vec::with_capacity(legs + 1);
        boom.push(views[0].boom(0.0));
        for i in 1..fixes.len() {
            let seconds = narrow_f32(fixes[i].seconds - fixes[i - 1].seconds);
            let step = sub(points[i], points[i - 1]);
            travel.push(CameraTravel {
                direction: step.map(narrow_f32),
                distance: narrow_f32(length(step)),
                seconds,
                ease: Ease::Linear,
                ..Default::default()
            });
            turn.push(CameraTurn {
                yaw_deg: Some(narrow_f32(views[i].yaw)),
                pitch_deg: Some(narrow_f32(views[i].pitch)),
                seconds,
                ease: Ease::Linear,
                ..Default::default()
            });
            boom.push(views[i].boom(seconds));
        }
        Self {
            start: points[0].map(narrow_f32),
            yaw_deg: narrow_f32(views[0].yaw),
            pitch_deg: narrow_f32(views[0].pitch),
            travel,
            turn,
            boom,
        }
    }
}

// How the camera films the marble at one fix: its yaw and pitch in degrees,
// and the boom's distance and offset.
struct View {
    yaw: f64,
    pitch: f64,
    distance: f64,
    offset: [f64; 2],
}

impl View {
    // The boom leg that reaches this framing over `seconds`.
    fn boom(&self, seconds: f32) -> CameraBoom {
        CameraBoom {
            distance: Some(narrow_f32(self.distance)),
            offset_deg: Some(self.offset.map(narrow_f32)),
            seconds,
            ease: Ease::Linear,
            ..Default::default()
        }
    }
}

// What a fix's shots are aimed by: the marble's up and the course ahead of
// it, the direction toward the sun, and how far below level the horizon
// lies, in degrees.
struct Sky {
    up: DVec3,
    ahead: DVec3,
    sun: DVec3,
    dip: f64,
}

impl Sky {
    // `shot`'s yaw, counterclockwise from the course, and pitch, both in
    // degrees from level.
    fn angles(&self, shot: &Shot) -> [f64; 2] {
        match shot.aim {
            Aim::Course => [shot.yaw, shot.pitch - self.dip],
            Aim::Sun => {
                let left = cross(self.up, self.ahead);
                let bearing = dot(self.sun, left).atan2(dot(self.sun, self.ahead));
                let elevation = dot(self.sun, self.up).clamp(-1.0, 1.0).asin();
                [
                    bearing.to_degrees() + shot.yaw,
                    elevation.to_degrees() + shot.pitch,
                ]
            }
        }
    }
}

// The framing at `fix` under `sky`, eased from one shot to the next. Yaw is
// counterclockwise from the course's start heading, which along the great
// circle stays the same angle from the course.
fn view(fix: &Fix, sky: &Sky) -> View {
    let (from, to, k) = framing::at(fix.seconds);
    let mix = |a: f64, b: f64| a + (b - a) * k;
    let [from_yaw, from_pitch] = sky.angles(from);
    let [to_yaw, to_pitch] = sky.angles(to);
    View {
        yaw: -HEADING_DEG + mix(from_yaw, to_yaw),
        pitch: mix(from_pitch, to_pitch),
        distance: mix(from.distance, to.distance),
        offset: [
            mix(from.offset[0], to.offset[0]),
            mix(from.offset[1], to.offset[1]),
        ],
    }
}
