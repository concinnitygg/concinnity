//! The planet run: a meadow on a 50 km planet, crossed at a run just over the
//! grass, then climbed out of to look along the curve to the horizon.
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
//! cutting through them. Posts stand beside the path every few hundred meters,
//! the static content a move carries along with the ground.

use concinnity::components::{
    CameraTravel, CameraTurn, DirectionalLight, Ease, FrameReport, Grass, Planet, ProceduralMesh,
    Prop, TerrainLayer, Wind,
};
use concinnity::cook::{Camera3D, CameraTrack, EnvironmentMap, Material, WorldBuilder};

const RADIUS: f32 = 50_000.0;
const AMPLITUDE: f32 = 40.0;

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

// Posts beside the ground run: how often, how far off the path, how tall.
const POST_EVERY: f64 = 400.0;
const POST_OFFSET: f64 = 5.0;
const POST_HALF_HEIGHT: f32 = 16.0;

/// The world: the planet, its meadow, the sun and sky, the posts, the camera,
/// its path, and the report.
pub(crate) fn declare(world: &mut WorldBuilder) {
    let planet = Planet {
        center: [0.0, -RADIUS, 0.0],
        radius: RADIUS,
        amplitude: AMPLITUDE,
        feature_size: 2_000.0,
        octaves: 8,
        seed: 4,
        layers: vec![TerrainLayer::default()],
        ..Default::default()
    };
    let path = Path::new(&planet);

    world.add(
        "sun",
        DirectionalLight {
            direction: [-0.55, 0.5, 0.4],
            color: [1.0, 0.95, 0.84],
            intensity: 3.2,
        },
    );
    world.add(
        "sky",
        EnvironmentMap {
            generator: "sky".to_string(),
            prefilter_face_size: 256,
            irradiance_face_size: 32,
            prefilter_samples: 64,
            ..Default::default()
        },
    );
    world.add(
        "turf",
        Material {
            tint: [0.11, 0.15, 0.06],
            roughness: 0.92,
            ..Default::default()
        },
    );
    world.add(
        "wind",
        Wind {
            direction: [1.0, 0.35],
            strength: 4.0,
            gustiness: 0.6,
            gust_scale: 16.0,
        },
    );
    world.add(
        "meadow",
        Grass {
            height: 0.55,
            height_variance: 0.45,
            width: 0.03,
            density: 100.0,
            clump_size: 0.7,
            ..Default::default()
        },
    );
    world
        .add("world", planet.clone())
        .reference("material", "turf")
        .reference("layers.0.grass", "meadow");

    declare_posts(world, &planet);

    world.add(
        "camera",
        Camera3D {
            fov_y_degrees: 70.0,
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
        let per_meter = (1.0 / f64::from(RADIUS)).to_degrees();
        let at = |meters: f64, height: f64| shape.geo_point(meters * per_meter, 0.0, height);

        let mut stops: Vec<(f64, f64, f32, &str)> = Vec::new();
        let mut along = 0.0;
        let mut first = true;
        while along < GROUND_METERS {
            along = (along + GROUND_STEP).min(GROUND_METERS);
            let name = if first { "ground" } else { "" };
            stops.push((along, GROUND_HEIGHT, GROUND_SPEED, name));
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
        for (meters, height, speed, segment) in stops {
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

// Posts beside the ground run, each standing on the ground along the planet's
// up there.
fn declare_posts(world: &mut WorldBuilder, planet: &Planet) {
    let shape = planet.shape();
    let per_meter = (1.0 / f64::from(RADIUS)).to_degrees();
    world.add(
        "post_mesh",
        ProceduralMesh {
            generator: "box".to_string(),
            half_extents: Some([0.5, POST_HALF_HEIGHT, 0.5]),
            ..Default::default()
        },
    );
    world.add(
        "post_paint",
        Material {
            tint: [0.85, 0.85, 0.8],
            roughness: 0.6,
            ..Default::default()
        },
    );
    let mut along = POST_EVERY * 0.5;
    let mut n = 0;
    while along < GROUND_METERS {
        let latitude = along * per_meter;
        // Sunk a little, so it stands in the ground on any slope.
        let center = shape.geo_point(
            latitude,
            POST_OFFSET * per_meter,
            f64::from(POST_HALF_HEIGHT) - 2.0,
        );
        world
            .add(
                format!("post_{n}"),
                Prop {
                    position: center.map(|c| c as f32),
                    // North along the meridian the up tips from +Y toward -Z.
                    rotation_deg: [-latitude as f32, 0.0, 0.0],
                    ..Default::default()
                },
            )
            .reference("mesh", "post_mesh")
            .reference("material", "post_paint");
        along += POST_EVERY;
        n += 1;
    }
}
