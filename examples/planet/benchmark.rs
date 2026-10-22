//! The benchmark run: the chrome marble flown through the world and filmed
//! from behind it, timed frame by frame and reported when the flight ends.
//! It skims the meadow and crosses the beach and the surf, over the two
//! floating crates, out over the sea; climbs toward the morning sun until it
//! leaves the air; turns to show the whole planet from orbit and the
//! terminator ahead; drops into the night and skims it low and fast; and
//! lands on the next continent's meadow as the sun rises.
//!
//! A planet is simulated in a frame that moves with the camera, every
//! kilometer, and draws its ground as tiles that refine and coarsen as the
//! camera passes, so its cost is paid along a path rather than at a station.
//! The run pays it on the ground, where the grass and the finest tiles are,
//! over the water, through a climb and a descent that cross hundreds of frame
//! moves, and from orbit, where the whole planet is in view.
//!
//! The path follows one great circle round the planet through where the
//! world starts, setting out east toward the nearest shore, each point a
//! fixed height over the ground the planet's own shape puts there, or over
//! the sea where the ground lies under it: the marble rides the hills and the
//! waves rather than cutting through them. Height and speed change by a
//! constant ratio at the fastest point of the climb and the descent, so a
//! kilometer is crossed as gently at the top as a meter at the bottom. The
//! track runs on the fixed simulation clock, so a slower host visits the same
//! poses at the same track times and simply samples fewer of them.
//!
//! The run is declared as the world's `BenchmarkConfig`: the marble it films
//! and where it starts, its legs, the camera's turns and framing, the phases
//! the report is cut into and the frames it saves under `benchmark/` in the
//! run's screenshot directory (`screenshots/` unless `--screenshots-dir`
//! names another). It changes nothing until the app is benchmarked. The
//! world's own sky turns as it always does, through the warm-up too, so the
//! flight's times are laid out against where that sky carries the sun.

mod flight;
mod framing;
mod path;

use concinnity::components::{
    BenchmarkConfig, BenchmarkPhase, CameraPose, DirectionalLight, FrameCapture, Planet,
    SkyRotation,
};
use concinnity::cook::WorldBuilder;
use concinnity::math::dvec3::{self, DVec3};
use concinnity_core::math::narrow_f32;

use flight::PHASES;
use path::Path;

// Long enough for the meadow's tiles and grass to settle at the start.
const WARMUP_SECONDS: f32 = 2.5;

// The frames the run saves: the file name and the moment, in seconds of the
// run as the HUD's `TRACK` chip counts them (the warm-up included), each named
// for what it shows.
const CAPTURES: [(&str, f32); 13] = [
    ("meadow-side", 11.5),
    ("meadow-chase", 19.0),
    ("beach", 25.0),
    ("surf-crates", 29.4),
    ("sea-skim", 40.0),
    ("climb-sun", 53.0),
    ("orbit-dayside", 66.0),
    ("orbit-terminator", 76.0),
    ("night-disc", 93.0),
    ("night-skim", 105.0),
    ("night-moonpath", 109.5),
    ("sunrise-coast", 123.0),
    ("sunrise-meadow", 133.0),
];

// The prop the run films.
const SUBJECT: &str = "marble";

// What the world declares that the run reads: the planet and sea its path
// rides, and the sky and light it aims by.
struct Declared {
    planet: Planet,
    sea_level: Option<f64>,
    sky: SkyRotation,
    sun: DirectionalLight,
}

/// Declare the benchmark over `world`, already declared in `builder`: the
/// marble starts where its path does and flies it, filmed from behind, one
/// phase per stretch of the flight.
pub(crate) fn declare(builder: &mut WorldBuilder, world: &str) -> Result<(), String> {
    let declared = Declared {
        // The planet's references are names, which its shape does not need.
        planet: serde_json::from_value(read(world, "Planet", &["material", "covers", "layers"])?)
            .map_err(|e| e.to_string())?,
        sea_level: sea_level(world),
        sky: serde_json::from_value(read(world, "SkyRotation", &[])?).map_err(|e| e.to_string())?,
        sun: serde_json::from_value(read(world, "DirectionalLight", &[])?)
            .map_err(|e| e.to_string())?,
    };
    // The path starts when the warm-up ends, by which time the sky has turned.
    let warmup = f64::from(WARMUP_SECONDS);
    let path = Path::new(&declared.planet, declared.sea_level, |seconds| {
        sun(&declared, warmup + seconds)
    });
    builder
        .add(
            "benchmark",
            &BenchmarkConfig {
                warmup_seconds: WARMUP_SECONDS,
                start: Some(CameraPose {
                    position: path.start,
                    yaw_deg: path.yaw_deg,
                    pitch_deg: path.pitch_deg,
                }),
                travel: path.travel,
                turn: path.turn,
                boom: path.boom,
                phases: PHASES
                    .iter()
                    .map(|phase| BenchmarkPhase {
                        name: phase.name.to_string(),
                        seconds: narrow_f32(phase.seconds),
                    })
                    .collect(),
                captures: CAPTURES
                    .iter()
                    .map(|&(name, track_seconds)| FrameCapture {
                        seconds: track_seconds - WARMUP_SECONDS,
                        path: format!("benchmark/{name}.png"),
                    })
                    .collect(),
                ..Default::default()
            },
        )
        .reference("subject", SUBJECT);
    Ok(())
}

// The args of the first `ty` entry `world` declares, without its `unnamed`
// fields.
fn read(world: &str, ty: &str, unnamed: &[&str]) -> Result<serde_json::Value, String> {
    for line in world.lines() {
        let Ok(serde_json::Value::Array(columns)) = serde_json::from_str(line) else {
            continue;
        };
        if let [
            serde_json::Value::String(declared),
            serde_json::Value::Object(args),
        ] = &columns[..]
            && declared == ty
        {
            let mut args = args.clone();
            for field in unnamed {
                args.remove(*field);
            }
            return Ok(args.into());
        }
    }
    Err(format!("the world declares no {ty}"))
}

// The level of the first planet sea `world` declares, in meters above the
// planet's radius.
fn sea_level(world: &str) -> Option<f64> {
    world.lines().find_map(|line| {
        let serde_json::Value::Array(columns) = serde_json::from_str(line).ok()? else {
            return None;
        };
        let [serde_json::Value::String(ty), args] = &columns[..] else {
            return None;
        };
        (ty == "WaterSurface" && args.get("shape")?.as_str()? == "planet_sea").then(|| {
            args.get("sea_level")
                .and_then(serde_json::Value::as_f64)
                .unwrap_or(0.0)
        })
    })
}

// The unit direction toward the sun `seconds` into the run.
fn sun(declared: &Declared, seconds: f64) -> DVec3 {
    let toward = declared.sky.turned(declared.sun.direction, seconds);
    dvec3::normalize_or(dvec3::from_f32(toward), [0.0, 1.0, 0.0])
}
