//! The benchmark run: a short run over the meadow and across the coast out
//! over the sea, then a climb that speeds up until the camera leaves the air,
//! a pass over the planet from orbit, and a descent to skim the ground far
//! round the planet, timed frame by frame and reported when the camera reaches
//! the end of its path.
//!
//! A planet is simulated in a frame that moves with the camera, every
//! kilometer, and draws its ground as tiles that refine and coarsen as the
//! camera passes, so its cost is paid along a path rather than at a station.
//! The run pays it on the ground, where the grass and the finest tiles are,
//! through a climb and a descent that cross hundreds of frame moves, and from
//! orbit, where the whole planet is in view.
//!
//! The path follows one great circle round the planet from where the world
//! starts, setting out east toward the nearest shore, each point a fixed
//! height over the ground the planet's own shape puts there, or over the sea
//! where the ground lies under it: the camera rides the hills and the waves
//! rather than cutting through them. Height and speed change by a constant
//! ratio at the fastest point of the climb and the descent, so a kilometer is
//! crossed as gently at the top as a meter at the bottom. The camera looks down
//! the course over the meadow, then turns to watch the sun. The track runs on
//! the fixed simulation clock, so a slower host visits the same poses at the
//! same track times and simply samples fewer of them.
//!
//! The run is declared as the world's `BenchmarkConfig`: the camera's start,
//! its legs, the phases the report is cut into and the frames it saves under
//! `benchmark/` in the run's screenshot directory (`screenshots/` unless
//! `--screenshots-dir` names another). It changes nothing until the app is benchmarked. The world's own sky turns as it always does, through the
//! warm-up too; the camera's start and its track come from the planet's shape
//! and where that sky carries the sun.

mod flight;
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
// run as the HUD's `TRACK` chip counts them (the warm-up included). Day over
// the meadow and out over the sea, climbing out of it, the planet from high
// up and from orbit, then the night skimmed low and back on the ground.
const CAPTURES: [(&str, f32); 8] = [
    ("ground-day", 13.3),
    ("sea-day", 22.0),
    ("climb-6m", 31.93),
    ("climb-30m", 34.65),
    ("climb-600m", 38.5),
    ("orbit", 55.0),
    ("low-night", 95.0),
    ("ground-night", 132.0),
];

// What the world declares that the run reads: the planet and sea its path
// rides, and the sky and light it aims by.
struct Declared {
    planet: Planet,
    sea_level: Option<f64>,
    sky: SkyRotation,
    sun: DirectionalLight,
}

/// Declare the benchmark over `world`, already declared in `builder`: the
/// camera starts where its path does and flies it, one phase per stretch of
/// the flight.
pub(crate) fn declare(builder: &mut WorldBuilder, world: &str) -> Result<(), String> {
    let declared = Declared {
        // The planet's references are names, which its shape does not need.
        planet: serde_json::from_value(read(world, "Planet", &["material", "layers"])?)
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
    builder.add(
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
    );
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
