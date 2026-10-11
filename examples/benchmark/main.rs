//! A world that measures itself.
//!
//! The camera flies down a corridor of eight stations, each loading the engine
//! somewhere different -- a lattice of instanced spheres, a grove under casting
//! lights, a colonnade under a field of local lights, a row of particle plumes,
//! a hall of glass over a marked floor, a pen of bodies that drop when the
//! camera arrives, a drift of tokens that wakes when it does too, and a
//! colonnade standing in water. It circles each one in turn, and when it
//! reaches the end it prints what every frame cost, cut by the station it was
//! at.
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
//! awake body in the corridor. Those show up in every segment, and they are in
//! the world because a frame without them is not the frame an application
//! ships.
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
//! `--frame-priority normal|main_threads|all_threads` runs it under that
//! `AppConfig::frame_priority` instead of the default, which is how the
//! setting's cost and benefit are compared on a busy machine.
//!
//! `--world planet` runs a second world instead: a planet crossed over its
//! grass and then from high above it (see `planet.rs`), reported the same way.

mod overlay;
mod palette;
mod planet;
mod stations;
mod track;

use concinnity::components::{
    FramePriority, FrameReport, GraphicsConfig, PostProcessConfig, Window,
};
use concinnity::cook::{self, Camera3D, CameraTrack};
use concinnity::{App, World};

// How far the camera sees. Long enough to reach across the widest station from
// the far side of its own circle, short enough that the stations past the next
// one lie beyond it, culled rather than rendered into every segment before
// them.
const VIEW_DISTANCE: f32 = 80.0;
const CAMERA_FOV_Y_DEGREES: f32 = 65.0;

// Frames earlier than this are thrown away. Shorter than the opening hold, so
// the discard never reaches into the first station's segment.
const WARMUP_SECONDS: f32 = 2.5;

fn main() {
    let options = match Options::parse(std::env::args().skip(1)) {
        Ok(options) => options,
        Err(error) => {
            eprintln!("{error}");
            std::process::exit(2);
        }
    };
    let world = benchmark_world(&options).expect("the benchmark world compiles");
    App::from_world(world).run().expect("the app runs");
}

// Which world runs, under which frame priority.
struct Options {
    priority: FramePriority,
    planet: bool,
}

impl Options {
    // `--frame-priority NAME` and `--world corridor|planet`, each optional.
    fn parse(mut args: impl Iterator<Item = String>) -> Result<Self, String> {
        let mut options = Self {
            priority: FramePriority::default(),
            planet: false,
        };
        while let Some(arg) = args.next() {
            match arg.as_str() {
                "--frame-priority" => {
                    let name = args.next().ok_or("--frame-priority needs a value")?;
                    options.priority = FramePriority::ALL
                        .iter()
                        .copied()
                        .find(|p| p.as_str() == name)
                        .ok_or_else(|| {
                            format!(
                                "--frame-priority takes one of {}",
                                FramePriority::NAMES.join(", ")
                            )
                        })?;
                }
                "--world" => {
                    options.planet = match args.next().as_deref() {
                        Some("corridor") => false,
                        Some("planet") => true,
                        _ => return Err("--world takes corridor or planet".to_string()),
                    };
                }
                _ => return Err(format!("unknown argument {arg:?}")),
            }
        }
        Ok(options)
    }
}

fn benchmark_world(options: &Options) -> Result<World, String> {
    let mut world = cook::world();
    let frame_priority = options.priority;

    world.add(
        "app",
        cook::AppConfig {
            frame_priority,
            ..Default::default()
        },
    );

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
    if options.planet {
        world.add(
            "post",
            PostProcessConfig {
                ssao: true,
                ssao_radius: 0.5,
                bloom_intensity: 0.3,
                bloom_threshold: 1.2,
                exposure_ev: -0.6,
                ..Default::default()
            },
        );
        planet::declare(&mut world);
        return world.compile().map_err(|e| e.to_string());
    }
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
            view_distance: Some(VIEW_DISTANCE),
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
