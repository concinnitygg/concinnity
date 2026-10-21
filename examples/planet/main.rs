//! A planet to walk on: a 50 km sphere of continents and oceans, its rolling
//! ground grown over with a windblown meadow that runs down gentle shores to
//! the sea, under an atmosphere whose sky turns through a four-minute day.
//! The afternoon gives way to a sunset, then to a moonlit night of stars, the
//! Milky Way and distant galaxies, and the sun comes back up in the east.
//!
//! The whole world is `planet.jsonl`, beside this file. This program only
//! compiles it into a blob next to the executable and plays that blob.
//!
//! Run it with `cargo run --release --example planet`.
//!
//! The world also declares a benchmark (see `benchmark.rs`), which
//! `--benchmark` runs: the camera flies instead of answering the player's
//! controls, over the meadow and out over the sea, up out of the air to see the planet from orbit,
//! and back down to skim the ground. The run times every frame and closes when
//! the flight ends, printing what the frames cost, phase by phase.
//!
//! It takes the engine's launch flags (`--help` lists them), `--vsync` among
//! them.

mod benchmark;

use std::path::PathBuf;

use concinnity::launch::{LaunchArgs, LaunchArgsError};
use concinnity::{App, cook};

const WORLD: &str = include_str!("planet.jsonl");

fn main() {
    let args = match LaunchArgs::parse(std::env::args().skip(1)) {
        Ok(args) => args,
        Err(LaunchArgsError::Help) => {
            print!("{}", LaunchArgsError::Help);
            return;
        }
        Err(error) => {
            eprintln!("{error}");
            std::process::exit(2);
        }
    };
    if let Err(error) = run(args) {
        eprintln!("{error}");
        std::process::exit(1);
    }
}

fn run(args: LaunchArgs) -> Result<(), String> {
    let mut world = cook::world();
    world.add_jsonl(WORLD);
    benchmark::declare(&mut world, WORLD)?;
    let blob = blob_path();
    world.write_blob(&blob).map_err(|e| e.to_string())?;
    App::from_blob(&blob)
        .map(|app| app.with_args(args))
        .and_then(App::run)
        .map_err(|e| e.to_string())
}

// The blob lands beside the executable, inside the build's target dir.
fn blob_path() -> PathBuf {
    let dir = std::env::current_exe()
        .ok()
        .and_then(|exe| exe.parent().map(std::path::Path::to_path_buf))
        .unwrap_or_default();
    dir.join("planet-world").join("0")
}
