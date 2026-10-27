// The flight laid over the planet: the course the marble follows, the points
// its legs run between, and how the camera films it along the way.

use alloc::vec::Vec;

use crate::components::{CameraBoom, CameraTravel, CameraTurn, Ease};
use crate::planet::PlanetShape;
use concinnity_core::math::narrow_f32;

use super::framing::Framing;
use super::timetable::Phase;
use super::timetable::{self, Fix};
use concinnity_core::math::dvec3::{DVec3, add, length, scale, sub};

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
            forward: add(scale(NORTH, libm::cos(h)), scale(EAST, libm::sin(h))),
        }
    }

    /// The unit direction from the center `arc` radians along the course.
    fn up(&self, arc: f64) -> DVec3 {
        add(
            scale(START_UP, libm::cos(arc)),
            scale(self.forward, libm::sin(arc)),
        )
    }
}

/// Where the tour goes and how it films: the course's heading clockwise from
/// north, how far behind where the world starts it sets out, its phases and
/// its framing.
pub(super) struct Route<'a> {
    pub(super) heading_deg: f64,
    pub(super) start_back: f64,
    /// The least height the path keeps over dry ground.
    pub(super) clearance: f64,
    pub(super) phases: &'a [&'static Phase],
    pub(super) framing: &'a Framing,
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
    /// The path over `shape`, flooded to `sea_level` meters above its radius
    /// when it has a sea, setting out `heading_deg` clockwise from north
    /// `start_back` meters behind where the world starts.
    pub(super) fn new(shape: &PlanetShape, sea_level: Option<f64>, route: &Route<'_>) -> Self {
        let heading_deg = route.heading_deg;
        let course = Course::new(heading_deg);
        let arc = -route.start_back / shape.radius;
        let fixes = timetable::fixes(route.phases, shape.radius, STEP_SECONDS, arc);
        let floor = sea_level.unwrap_or(f64::NEG_INFINITY);
        let surface = |up: DVec3| shape.radius + shape.height(up).max(floor);
        let points: Vec<DVec3> = fixes
            .iter()
            .map(|fix| {
                let up = course.up(fix.arc);
                let ground = shape.height(up);
                let height = if ground >= floor {
                    fix.height.max(route.clearance)
                } else {
                    fix.height
                };
                add(shape.center, scale(up, surface(up) + height))
            })
            .collect();
        let views: Vec<View> = fixes
            .iter()
            .map(|fix| {
                let ground = surface(course.up(fix.arc));
                let dip = libm::acos(ground / (ground + fix.height)).to_degrees();
                view(fix, dip, route.framing, heading_deg)
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

// The framing at `fix`, with the horizon `dip` degrees below level, eased
// from one shot to the next. Yaw is counterclockwise from the course's start
// heading, which along the great circle stays the same angle from the course.
fn view(fix: &Fix, dip: f64, framing: &Framing, heading_deg: f64) -> View {
    let (from, to, k) = framing.at(fix.seconds);
    let mix = |a: f64, b: f64| a + (b - a) * k;
    View {
        yaw: -heading_deg + mix(from.yaw, to.yaw),
        pitch: mix(from.pitch, to.pitch) - dip,
        distance: mix(from.distance, to.distance),
        offset: [
            mix(from.offset[0], to.offset[0]),
            mix(from.offset[1], to.offset[1]),
        ],
    }
}
