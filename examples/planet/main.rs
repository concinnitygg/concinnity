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
//! the player's controls, times every frame, and closes when the path ends, printing what
//! the frames cost, cut by stretch of the path (see `benchmark.rs`).
//! `--capture DIR` with it also saves a few frames of the run into `DIR` as
//! PNG images, the same moments of the path on every run.

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
    let args = parse_args(std::env::args().skip(1))?;
    let mut world = cook::world();
    world.add_jsonl(WORLD);
    if args.benchmark {
        if let Some(dir) = &args.capture {
            std::fs::create_dir_all(dir)
                .map_err(|e| format!("cannot create {}: {e}", dir.display()))?;
        }
        benchmark::inject(
            &mut world,
            &declared_planet(WORLD)?,
            declared_graphics(WORLD)?,
            args.capture.as_deref(),
        );
    }
    let blob = blob_path();
    world.write_blob(&blob).map_err(|e| e.to_string())?;
    App::from_blob(&blob)
        .and_then(App::run)
        .map_err(|e| e.to_string())
}

// What the command line asked for.
#[derive(Debug, Default, PartialEq)]
struct Args {
    benchmark: bool,
    // Where the benchmark saves its frames, if it does.
    capture: Option<PathBuf>,
}

const USAGE: &str = "the options are --benchmark and --capture DIR";

fn parse_args(mut args: impl Iterator<Item = String>) -> Result<Args, String> {
    let mut parsed = Args::default();
    while let Some(arg) = args.next() {
        match arg.as_str() {
            "--benchmark" => parsed.benchmark = true,
            "--capture" => match args.next() {
                Some(dir) if !dir.starts_with("--") => parsed.capture = Some(dir.into()),
                _ => return Err(format!("--capture needs a directory; {USAGE}")),
            },
            _ => return Err(format!("unknown argument {arg:?}; {USAGE}")),
        }
    }
    if parsed.capture.is_some() && !parsed.benchmark {
        return Err("--capture saves frames of the benchmark run, so it needs --benchmark".into());
    }
    Ok(parsed)
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

#[cfg(test)]
mod tests {
    use super::*;

    fn parse(args: &[&str]) -> Result<Args, String> {
        parse_args(args.iter().map(|a| a.to_string()))
    }

    #[test]
    fn no_arguments_walk_the_planet() {
        assert_eq!(parse(&[]), Ok(Args::default()));
    }

    #[test]
    fn the_benchmark_runs_with_or_without_captures() {
        assert_eq!(
            parse(&["--benchmark"]),
            Ok(Args {
                benchmark: true,
                capture: None
            })
        );
        let with_dir = Args {
            benchmark: true,
            capture: Some(PathBuf::from("shots")),
        };
        assert_eq!(parse(&["--benchmark", "--capture", "shots"]), Ok(with_dir));
        let either_order = parse(&["--capture", "shots", "--benchmark"]);
        assert_eq!(either_order.map(|a| a.capture), Ok(Some("shots".into())));
    }

    #[test]
    fn captures_need_the_benchmark() {
        assert!(parse(&["--capture", "shots"]).is_err());
    }

    #[test]
    fn captures_need_a_directory() {
        assert!(parse(&["--benchmark", "--capture"]).is_err());
        assert!(parse(&["--capture", "--benchmark"]).is_err());
    }

    #[test]
    fn an_unknown_argument_is_refused() {
        assert!(parse(&["--benchmark", "--fast"]).is_err());
    }
}
