//! The benchmark run: the meadow crossed at a run just over the grass, then
//! climbed out of to look along the curve to the horizon, timed frame by frame
//! and reported when the camera reaches the end of its path.
//!
//! A planet is simulated in a frame that moves with the camera, every
//! kilometer, and draws its ground as tiles that refine and coarsen as the
//! camera passes, so its cost is paid along a path rather than at a station.
//! The run crosses several of those moves on the ground, where the grass and
//! the finest tiles are, and more of them aloft, where the tiles reach out to
//! the horizon and the grass gives way to the color it shows from afar.
//!
//! The path is laid out by latitude along the prime meridian, northward from
//! where the world starts, each leg ending a fixed height over the ground the
//! planet's own shape puts there: the camera rides the hills rather than
//! cutting through them. The track runs on the fixed simulation clock, so a
//! slower host visits the same poses at the same track times and simply
//! samples fewer of them.

use concinnity::components::{CameraTravel, CameraTurn, Ease, FrameReport, Planet};
use concinnity::cook::{Camera3D, CameraTrack, WorldBuilder};

// The ground run: this far along the ground, this high over the grass, this
// fast. Long enough to cross three frame moves.
const GROUND_METERS: f64 = 3_200.0;
const GROUND_HEIGHT: f64 = 2.5;
const GROUND_SPEED: f32 = 30.0;
// The climb: up to the cruising height over this much ground.
const CLIMB_METERS: f64 = 1_000.0;
const CLIMB_SPEED: f32 = 40.0;
// The cruise: this high over the ground, this far, this fast.
const ALOFT_METERS: f64 = 3_000.0;
const ALOFT_HEIGHT: f64 = 400.0;
const ALOFT_SPEED: f32 = 75.0;
// How far apart the points the path is laid through are: close enough that a
// leg never strays from the ground under it by more than a few centimeters.
const GROUND_STEP: f64 = 25.0;
const ALOFT_STEP: f64 = 100.0;

const SETTLE_SECONDS: f32 = 4.0;
const WARMUP_SECONDS: f32 = 2.5;
const GROUND_PITCH_DEG: f32 = -6.0;
const ALOFT_PITCH_DEG: f32 = -14.0;
const CAMERA_FOV_Y_DEGREES: f32 = 70.0;

// What the world file declares that the run replaces: the walker, and the
// camera it steers, which the track drives instead.
const WALKER: &str = "RigidBody#0";
const CAMERA: &str = "cam";

/// Swap the walker for a camera on the path over `planet`, and time the run.
pub(crate) fn inject(world: &mut WorldBuilder, planet: &Planet) {
    let path = Path::new(planet);
    world.remove(WALKER).remove(CAMERA);
    world.add(
        CAMERA,
        Camera3D {
            fov_y_degrees: CAMERA_FOV_Y_DEGREES,
            near: 0.1,
            view_distance: None,
            position: path.start,
            yaw: 0.0,
            pitch: GROUND_PITCH_DEG.to_radians(),
            controller: None,
        },
    );
    world.add(
        "track",
        CameraTrack {
            travel: path.travel,
            turn: path.turn,
        },
    );
    world.add(
        "report",
        FrameReport {
            warmup_seconds: WARMUP_SECONDS,
            ..Default::default()
        },
    );
}

// The path north from the start: where the camera begins, and the legs it
// travels and turns through.
struct Path {
    start: [f32; 3],
    travel: Vec<CameraTravel>,
    turn: Vec<CameraTurn>,
}

impl Path {
    fn new(planet: &Planet) -> Self {
        let shape = planet.shape();
        // Degrees of latitude per meter along the ground.
        let per_meter = (1.0 / f64::from(planet.radius)).to_degrees();
        let at = |meters: f64, height: f64| shape.geo_point(meters * per_meter, 0.0, height);

        let origin = at(0.0, GROUND_HEIGHT);
        let mut travel = vec![CameraTravel {
            seconds: SETTLE_SECONDS,
            segment: "settle".to_string(),
            ..Default::default()
        }];
        let mut turn = vec![CameraTurn {
            seconds: SETTLE_SECONDS,
            ..Default::default()
        }];
        let mut from = origin;
        for (meters, height, speed, segment) in stops() {
            let to = at(meters, height);
            let step: [f64; 3] = std::array::from_fn(|i| to[i] - from[i]);
            let distance = step.iter().map(|c| c * c).sum::<f64>().sqrt();
            from = to;
            travel.push(CameraTravel {
                direction: step.map(|c| c as f32),
                distance: distance as f32,
                speed,
                ease: Ease::Linear,
                segment: segment.to_string(),
                ..Default::default()
            });
            let lift = ((height - GROUND_HEIGHT) / (ALOFT_HEIGHT - GROUND_HEIGHT)) as f32;
            turn.push(CameraTurn {
                yaw_deg: Some(0.0),
                pitch_deg: Some(GROUND_PITCH_DEG + (ALOFT_PITCH_DEG - GROUND_PITCH_DEG) * lift),
                seconds: distance as f32 / speed,
                ease: Ease::Linear,
                ..Default::default()
            });
        }
        Self {
            start: origin.map(|c| c as f32),
            travel,
            turn,
        }
    }
}

// The points the path is laid through, as meters along the ground, height
// over it, speed to reach them, and the segment a leg opens (empty where it
// continues the one before).
fn stops() -> Vec<(f64, f64, f32, &'static str)> {
    let mut stops = Vec::new();
    let mut along = 0.0;
    let mut first = true;
    while along < GROUND_METERS {
        along = (along + GROUND_STEP).min(GROUND_METERS);
        stops.push((
            along,
            GROUND_HEIGHT,
            GROUND_SPEED,
            if first { "ground" } else { "" },
        ));
        first = false;
    }
    let climb_from = along;
    first = true;
    while along < climb_from + CLIMB_METERS {
        along = (along + ALOFT_STEP).min(climb_from + CLIMB_METERS);
        let t = (along - climb_from) / CLIMB_METERS;
        let eased = t * t * (3.0 - 2.0 * t);
        let height = GROUND_HEIGHT + (ALOFT_HEIGHT - GROUND_HEIGHT) * eased;
        stops.push((along, height, CLIMB_SPEED, if first { "climb" } else { "" }));
        first = false;
    }
    let cruise_from = along;
    first = true;
    while along < cruise_from + ALOFT_METERS {
        along = (along + ALOFT_STEP).min(cruise_from + ALOFT_METERS);
        stops.push((
            along,
            ALOFT_HEIGHT,
            ALOFT_SPEED,
            if first { "aloft" } else { "" },
        ));
        first = false;
    }
    stops
}
