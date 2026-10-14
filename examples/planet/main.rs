//! A planet to walk on: a 50 km sphere of rolling ground grown over with a
//! windblown meadow, under an atmosphere whose sky turns through a
//! four-minute day. The afternoon gives way to a sunset, then to a moonlit
//! night of stars, the Milky Way and distant galaxies, and the sun comes back
//! up in the east.
//!
//! The whole world is `planet.jsonl`, beside this file. This program only
//! compiles it into a blob next to the executable and plays that blob.
//!
//! Run it with `cargo run --release --features cook --example planet`.
//!
//! `--benchmark` flies the camera over the planet instead of handing it to
//! the walker, times every frame, and closes when the path ends, printing what
//! the frames cost, cut by stretch of the path (see `benchmark.rs`).

mod benchmark;

use std::path::PathBuf;

use concinnity::components::{GraphicsConfig, Planet};
use concinnity::{App, cook};

const WORLD: &str = include_str!("planet.jsonl");

fn main() {
    if let Err(error) = run() {
        eprintln!("{error}");
        std::process::exit(1);
    }
}

fn run() -> Result<(), String> {
    let benchmark = parse_args(std::env::args().skip(1))?;
    let mut world = cook::world();
    world.add_jsonl(WORLD);
    if benchmark {
        benchmark::inject(
            &mut world,
            &declared_planet(WORLD)?,
            declared_graphics(WORLD)?,
        );
    }
    let blob = blob_path();
    world.write_blob(&blob).map_err(|e| e.to_string())?;
    App::from_blob(&blob)
        .and_then(App::run)
        .map_err(|e| e.to_string())
}

// Whether `--benchmark` was passed, the one argument this takes.
fn parse_args(args: impl Iterator<Item = String>) -> Result<bool, String> {
    let mut benchmark = false;
    for arg in args {
        match arg.as_str() {
            "--benchmark" => benchmark = true,
            _ => {
                return Err(format!(
                    "unknown argument {arg:?}; the one option is --benchmark"
                ));
            }
        }
    }
    Ok(benchmark)
}

// The planet the world file declares, for the shape the benchmark path rides.
// Its references are names in the file, so they are left out.
fn declared_planet(text: &str) -> Result<Planet, String> {
    let args = declared(text, "Planet", &["material", "layers"])?;
    serde_json::from_value(args).map_err(|e| e.to_string())
}

// The graphics settings the world file declares, if any, which the benchmark
// keeps but for the frame-rate cap.
fn declared_graphics(text: &str) -> Result<Option<GraphicsConfig>, String> {
    let Ok(args) = declared(text, "GraphicsConfig", &[]) else {
        return Ok(None);
    };
    serde_json::from_value(args)
        .map(Some)
        .map_err(|e| e.to_string())
}

// The args of the first `ty` entry the world file declares, without the
// `unnamed` fields.
fn declared(text: &str, ty: &str, unnamed: &[&str]) -> Result<serde_json::Value, String> {
    for line in text.lines() {
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

// The blob lands beside the executable, inside the build's target dir.
fn blob_path() -> PathBuf {
    let dir = std::env::current_exe()
        .ok()
        .and_then(|exe| exe.parent().map(|dir| dir.to_path_buf()))
        .unwrap_or_default();
    dir.join("planet-world").join("0")
}
