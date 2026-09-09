//! A world that measures itself.
//!
//! The camera flies down a corridor of eight stations, each loading the engine
//! somewhere different -- a lattice of instanced spheres, a grove under casting
//! lights, a colonnade under a field of local lights, a row of particle plumes,
//! a hall of glass over a marked floor, a pen of falling bodies, a drift of
//! tokens that wakes when the camera arrives, and a colonnade standing in water.
//! It circles each one in turn, and when it reaches the end it prints what every
//! frame cost, cut by the station it was at.
//!
//! The measurement is two of the world's own components. A `CameraTrack` drives
//! the camera along the path and names each stretch of it; a `FrameReport` times
//! every frame and reports the distribution under those names when the run ends.
//!
//! The two together are what make the numbers comparable between machines. The
//! track runs on the fixed simulation clock rather than on frame time, so a host
//! that renders half as fast visits the same poses at the same track times and
//! simply samples fewer of them, and a segment here covers the same content as
//! that segment there. The report's opening discard covers the seconds over
//! which shader compilation, streaming residency, temporal-antialiasing history
//! and auto-exposure converge, which a run that kept them would disagree with
//! its own repeat over.
//!
//! # Why it circles
//!
//! A camera that flew past a station would measure the approach rather than the
//! station: the subject would swing from far ahead to hard off to one side while
//! its distance ran from the far plane down to nothing, so no two frames of a
//! segment would be looking at the same thing. Holding the camera still would
//! fix the framing and break something else, since a still frame lets temporal
//! antialiasing and reprojection converge on an answer no moving frame ever
//! gets.
//!
//! So each station is circled instead. The camera reaches it along a half circle
//! of that station's own radius, which holds the subject at one distance and in
//! the middle of the frame while the camera keeps moving. The half circles meet
//! where their tangents agree, so the path runs on without a straight stretch
//! between them, and every measured frame belongs to a station.
//!
//! # Reading the report
//!
//! A station earns its place by owning a column. The drift is read from the
//! per-system CPU breakdown, since it searches the world once per token per rule
//! per tick. The rest are read from the pass list. What a station cannot do is
//! separate a cost the engine pays wherever the camera is: the pool and the pane
//! each re-render the scene for their reflection, and the solver steps every
//! body in the corridor. Those show up in every segment, and they are in the
//! world because a frame without them is not the frame an application ships.
//!
//! # How it is built
//!
//! The world is declared here asset by asset and compiled in memory by the
//! `cook` module. Nothing is read from disk: every texture, mesh and glyph comes
//! from a built-in generator, so the corridor stands up in any checkout. The
//! cook is what makes textured surfaces, decals and text reachable at all, since
//! a texture handle is only ever resolved from a declared name.
//!
//! Run it with `cargo run --release --features cook --example benchmark`.

mod overlay;
mod palette;
mod stations;
mod track;

use concinnity::components::{FrameReport, GraphicsConfig, PostProcessConfig, Window};
use concinnity::cook::{self, Camera3D, CameraTrack};
use concinnity::{App, World};

// How far the camera sees. Long enough to reach across the widest station from
// the far side of its own circle, short enough that the stations past the next
// one are behind the far plane rather than rendered into every segment before
// them.
const CAMERA_FAR: f32 = 80.0;
const CAMERA_FOV_Y_DEGREES: f32 = 65.0;

// Frames earlier than this are thrown away. Shorter than the opening hold, so
// the discard never reaches into the first station's segment.
const WARMUP_SECONDS: f32 = 2.5;

fn main() {
    let world = benchmark_world().expect("the benchmark world compiles");
    App::from_world(world).run().expect("the app runs");
}

fn benchmark_world() -> Result<World, String> {
    let mut world = cook::world();

    world.add(
        "window",
        Window {
            title: "Benchmark".to_string(),
            width: 1280,
            height: 720,
            resizable: true,
            ..Default::default()
        },
    );
    world.add(
        "graphics",
        GraphicsConfig {
            clear_color: [0.05, 0.07, 0.10, 1.0],
            vsync: false,
            ..Default::default()
        },
    );
    world.add(
        "post",
        PostProcessConfig {
            ambient_intensity: 0.35,
            bloom_intensity: 0.25,
            bloom_threshold: 1.4,
            ..Default::default()
        },
    );
    // The camera has no controller: the track drives it, and a controller would
    // be built for nothing.
    world.add(
        "camera",
        Camera3D {
            fov_y_degrees: CAMERA_FOV_Y_DEGREES,
            near: 0.05,
            far: CAMERA_FAR,
            position: [0.0, track::HEIGHT, stations::START_Z],
            yaw: 0.0,
            pitch: (-4.0_f32).to_radians(),
            controller: None,
        },
    );
    world.add(
        "track",
        CameraTrack {
            travel: track::travel_legs(),
            turn: track::turn_legs(),
        },
    );
    world.add("report", frame_report());

    palette::declare(&mut world);
    overlay::declare(&mut world);
    stations::declare_all(&mut world);

    world.compile().map_err(|e| e.to_string())
}

// What the world asks to be measured against. The discard is the reason two
// runs of this world agree with each other.
fn frame_report() -> FrameReport {
    FrameReport {
        warmup_seconds: WARMUP_SECONDS,
        ..Default::default()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // The compile is checked here, not by the compiler: an unknown generator,
    // an unresolved reference, or a declaration that fails validation surfaces
    // as the error string it returns.
    #[test]
    fn the_benchmark_world_compiles() {
        benchmark_world().expect("the benchmark world compiles");
    }

    // Every measured frame belongs to a station. The opening hold has to
    // outlast the discard, or the first station's frames would be thrown away
    // with the settling and its segment would be short a second of path.
    #[test]
    fn the_opening_hold_covers_the_discard() {
        assert!(track::SETTLE_SECONDS > WARMUP_SECONDS);
    }

    // The camera has to see across a station from the far side of its own
    // circle, or the station is clipped in the middle of its own segment.
    #[test]
    fn the_camera_sees_across_the_station_it_is_circling() {
        let widest = stations::STATIONS
            .iter()
            .map(|s| s.radius)
            .fold(0.0_f32, f32::max);
        assert!(CAMERA_FAR > widest * 2.0, "the far side is clipped");
    }

    // The report and the track are what make this world a benchmark rather than
    // a demo, and between them they are the whole of the instrument.
    #[test]
    fn the_two_components_that_measure_the_world_agree_with_its_path() {
        let report = frame_report();
        assert_eq!(report.warmup_seconds, WARMUP_SECONDS);
        // Reaching the end of the track is what ends the run.
        assert!(report.stop_when_complete);
    }
}
