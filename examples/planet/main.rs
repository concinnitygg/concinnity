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
//! `--benchmark` lays `bench.jsonl` over the world and flies the camera
//! instead of handing it to the player's controls: over the meadow, up out of
//! the air to see the planet from orbit, and back down to skim the ground. It
//! times every frame and closes when the path ends, printing what the frames
//! cost, cut by stretch of the path (see `benchmark.rs`).

mod benchmark;

use std::path::PathBuf;

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
        benchmark::declare(&mut world, WORLD)?;
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
}

fn parse_args(args: impl Iterator<Item = String>) -> Result<Args, String> {
    let mut parsed = Args::default();
    for arg in args {
        match arg.as_str() {
            "--benchmark" => parsed.benchmark = true,
            _ => {
                return Err(format!(
                    "unknown argument {arg:?}; the option is --benchmark"
                ));
            }
        }
    }
    Ok(parsed)
}

// The blob lands beside the executable, inside the build's target dir.
fn blob_path() -> PathBuf {
    let dir = std::env::current_exe()
        .ok()
        .and_then(|exe| exe.parent().map(|dir| dir.to_path_buf()))
        .unwrap_or_default();
    dir.join("planet-world").join("0")
}
