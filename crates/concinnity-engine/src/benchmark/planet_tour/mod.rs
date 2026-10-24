// The benchmark route a planet gets by default: a tour along one great circle
// round the planet through where the world starts, timed against its sun.
//
// The tour pays the planet's costs along a path rather than at a station: on
// the ground, where the grass and the finest tiles are, over the water,
// through a climb and a descent that cross hundreds of frame moves, and from
// orbit, where the whole planet is in view. Each point of the path is a fixed
// height over the ground the planet's own shape puts there, or over the sea
// where the ground lies under it, so the subject rides the hills and the waves
// rather than cutting through them.

mod course;
mod framing;
mod timetable;

use alloc::string::ToString;
use alloc::vec::Vec;

use crate::components::{
    BenchmarkPhase, CameraBoom, CameraPose, CameraTravel, CameraTurn, DirectionalLight, DriveKey,
    Planet, SkyRotation,
};
use crate::ecs::World;
use concinnity_core::math::dvec3::{self, DVec3};
use concinnity_core::math::narrow_f32;

use course::{Path, Route};
use framing::Framing;

/// How a [`BenchmarkConfig`](#benchmarkconfig) tours a [`Planet`](#planet).
///
/// The tour follows one great circle round the planet through where the world
/// starts. It sets out over the ground, crosses the shore and the sea, climbs
/// toward the sun until it leaves the air, shows the whole planet from orbit,
/// drops into the night and skims it, and comes down on the far side as the
/// sun rises. The camera films the benchmark's `subject` from behind, framed
/// by where it is on the tour and where the sun is.
///
/// A subject with a [`RollingDrive`](#rollingdrive) rolls itself over the
/// ground first: through the grass, down the beach and into the sea, where it
/// floats and paddles, then is towed up to speed through the water, until the
/// tour picks it up and flies it; on a planet with no sea it is picked up as
/// the roll ends. A subject
/// without one, or none at all, skims the ground instead.
///
/// ```json
/// ["BenchmarkConfig", { "subject": "ball", "tour": { "heading_deg": 96.0 } }]
/// ```
#[derive(
    Debug,
    Clone,
    Copy,
    PartialEq,
    serde::Serialize,
    serde::Deserialize,
    crate::ecs::AssetFields,
    crate::ecs::AssetDefault,
)]
#[serde(default)]
pub struct PlanetTour {
    /// The direction the tour sets out in from where the world starts, in
    /// degrees clockwise from north.
    #[asset(default = 96.0)]
    pub heading_deg: f32,
    /// How far behind where the world starts the tour begins, in meters.
    #[asset(default = 150.0)]
    pub start_back: f32,
}

/// The legs and phases a tour resolves to over one world.
#[derive(Debug, Clone, PartialEq)]
pub(crate) struct TourPlan {
    pub(crate) start: CameraPose,
    pub(crate) travel: Vec<CameraTravel>,
    pub(crate) turn: Vec<CameraTurn>,
    pub(crate) boom: Vec<CameraBoom>,
    pub(crate) phases: Vec<BenchmarkPhase>,
    /// How a rolling subject's drive is steered, and when the tour picks it
    /// up, both in seconds after the warm-up.
    pub(crate) drive: Vec<DriveKey>,
    pub(crate) carry_from_seconds: f32,
}

/// The tour over `world`'s planet, its sea and its sun, starting after
/// `warmup_seconds`, rolled over the ground first when `rolls`, or `None`
/// for a world with no planet.
pub(crate) fn plan(
    world: &World,
    tour: PlanetTour,
    warmup_seconds: f32,
    rolls: bool,
) -> Option<TourPlan> {
    let planet = world.query::<Planet>().next()?;
    let shape = planet.shape();
    let sea_level = planet.sea.map(|s| f64::from(s.level));
    let sky = world.query::<SkyRotation>().next().cloned();
    let light = world
        .query::<DirectionalLight>()
        .next()
        .map_or([0.0, 1.0, 0.0], |l| l.direction);
    // The path starts when the warm-up ends, by which time the sky has turned.
    let warmup = f64::from(warmup_seconds);
    let sun = |seconds: f64| -> DVec3 {
        let toward = sky
            .as_ref()
            .map_or(light, |sky| sky.turned(light, warmup + seconds));
        dvec3::normalize_or(dvec3::from_f32(toward), [0.0, 1.0, 0.0])
    };
    let phases = timetable::phases(rolls, sea_level.is_some());
    let framing = Framing::new(&phases, rolls);
    let route = Route {
        heading_deg: f64::from(tour.heading_deg),
        start_back: f64::from(tour.start_back),
        clearance: timetable::ground_clearance(rolls),
        phases: &phases,
        framing: &framing,
    };
    let path = Path::new(&shape, sea_level, &route, sun);
    let (drive, carry_from_seconds) = if rolls {
        roll_script(&phases, tour.heading_deg)
    } else {
        (Vec::new(), 0.0)
    };
    Some(TourPlan {
        start: CameraPose {
            position: path.start,
            yaw_deg: path.yaw_deg,
            pitch_deg: path.pitch_deg,
        },
        travel: path.travel,
        turn: path.turn,
        boom: path.boom,
        phases: phases
            .iter()
            .map(|phase| BenchmarkPhase {
                name: phase.name.to_string(),
                seconds: narrow_f32(phase.seconds),
            })
            .collect(),
        drive,
        carry_from_seconds,
    })
}

// The rolling subject's drive: held through the settle, then sprinting along
// the course's heading through the roll and paddling on through any float,
// towed up to the wake's pace through any wake, and picked up as the sea phase
// opens.
fn roll_script(phases: &[&timetable::Phase], heading_deg: f32) -> (Vec<DriveKey>, f32) {
    let start = |name| narrow_f32(timetable::phase_start(phases, name).unwrap_or(0.0));
    let mut keys = alloc::vec![
        DriveKey {
            seconds: 0.0,
            heading_deg,
            throttle: 0.0,
            ..DriveKey::default()
        },
        DriveKey {
            seconds: start("roll"),
            heading_deg,
            throttle: 1.0,
            sprint: true,
            ..DriveKey::default()
        },
    ];
    if let Some(wake) = timetable::phase_start(phases, "wake") {
        keys.push(DriveKey {
            seconds: narrow_f32(wake),
            heading_deg,
            pace: narrow_f32(timetable::WAKE_SPEED),
            ..DriveKey::default()
        });
    }
    (keys, start("sea"))
}

#[cfg(test)]
mod tests;
