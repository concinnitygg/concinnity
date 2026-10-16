//! The benchmark run: a short run over the meadow, then a climb that speeds
//! up until the camera leaves the air, a pass over the planet from orbit, and
//! a descent to skim the ground far to the north, timed frame by frame and
//! reported when the camera reaches the end of its path.
//!
//! A planet is simulated in a frame that moves with the camera, every
//! kilometer, and draws its ground as tiles that refine and coarsen as the
//! camera passes, so its cost is paid along a path rather than at a station.
//! The run pays it on the ground, where the grass and the finest tiles are,
//! through a climb and a descent that cross hundreds of frame moves, and from
//! orbit, where the whole planet is in view.
//!
//! The path follows one great circle round the planet from where the world
//! starts, setting out a little east of north, each point a fixed height over
//! the ground the planet's own shape puts there: the camera rides the hills
//! rather than cutting through them. Height and speed change by a constant
//! ratio at the fastest point of the climb and the descent, so a kilometer is
//! crossed as gently at the top as a meter at the bottom. The camera looks down
//! the course over the meadow, then turns to watch the sun. The track runs on
//! the fixed simulation clock, so a slower host visits the same poses at the
//! same track times and simply samples fewer of them.
//!
//! What the run changes in the world is `bench.jsonl`, laid over it: the frame
//! report, the camera without the player's controls, and a frame rate left
//! uncapped (a capped GPU idles between frames and lowers its clocks). The
//! world's own sky turns as it always does; the camera's start and its track
//! come from the planet's shape and where that sky carries the sun.

mod flight;
mod path;

use concinnity::components::{DirectionalLight, Planet, SkyRotation};
use concinnity::cook::{CameraTrack, WorldBuilder};
use concinnity::math::dvec3::{self, DVec3};

use path::Path;

const BENCH: &str = include_str!("bench.jsonl");

// The camera the track drives.
const CAMERA: &str = "cam";

// What the world declares that the run reads: the planet its path rides, and
// the sky and light it aims by.
struct Declared {
    planet: Planet,
    sky: SkyRotation,
    sun: DirectionalLight,
}

/// Lay `bench.jsonl` over `world`, already declared in `builder`, start the
/// camera where its path does, and add the track that flies it. The overlay
/// leaves the planet, the sky and the light as `world` declares them.
pub(crate) fn declare(builder: &mut WorldBuilder, world: &str) -> Result<(), String> {
    builder.overlay_jsonl(BENCH);
    let declared = Declared {
        // The planet's references are names, which its shape does not need.
        planet: serde_json::from_value(read(world, "Planet", &["material", "layers"])?)
            .map_err(|e| e.to_string())?,
        sky: serde_json::from_value(read(world, "SkyRotation", &[])?).map_err(|e| e.to_string())?,
        sun: serde_json::from_value(read(world, "DirectionalLight", &[])?)
            .map_err(|e| e.to_string())?,
    };
    let path = Path::new(&declared.planet, |seconds| sun(&declared, seconds));
    let start = serde_json::json!([
        "Camera3D",
        {
            "$id": CAMERA,
            "position": path.start,
            "yaw": path.yaw_deg.to_radians(),
            "pitch": path.pitch_deg.to_radians(),
        }
    ]);
    builder.overlay_jsonl(&start.to_string()).add(
        "track",
        CameraTrack {
            travel: path.travel,
            turn: path.turn,
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

// The unit direction toward the sun `seconds` into the run.
fn sun(declared: &Declared, seconds: f64) -> DVec3 {
    let toward = declared.sky.turned(declared.sun.direction, seconds);
    dvec3::normalize_or(dvec3::from_f32(toward), [0.0, 1.0, 0.0])
}
