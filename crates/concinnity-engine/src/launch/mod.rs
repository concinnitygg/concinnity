//! What a launch asks of the run, from the command line or from code: the
//! [`LaunchArgs`] parsed from a [`LaunchCommand`] and the flags in
//! [`LAUNCH_FLAGS`].
//!
//! The commands and flags are described once, in the table, so every front
//! that offers them (an app's own command line, `cn run`, a debug client)
//! shows the same names and descriptions.

mod error;
mod flags;
mod parse;
mod pipeline;
mod request;

pub use error::LaunchArgsError;
pub use flags::{
    DEFAULT_DEBUG_PORT, DLSS_PRESET_VALUES, GRAPHICS_API_VALUES, HIDE_PASS_VALUES, LAUNCH_FLAGS,
    LaunchCommand, LaunchFlag, QUALITY_PRESET_VALUES, RT_DYNAMIC_VALUES, WINDOW_MODE_VALUES, flag,
    short_flag, usage,
};
pub use pipeline::PipelineMode;
pub use request::{DEFAULT_SCREENSHOTS_DIR, LaunchRequest, screenshot_path};

use alloc::string::{String, ToString};
use alloc::vec::Vec;
use core::num::NonZeroU32;

use crate::gfx::view_modes::ShowFlags;
use crate::platform::Platform;
use crate::render::output::dlss::DlssPreset;
use crate::render::rt_accel::geom::RtDynamicMode;
use crate::settings::QualityPreset;
use crate::window::WindowMode;
use concinnity_core::math::{SaturatingFrom, round};

/// What a launch asks of the run. Each field left at its default leaves the
/// run as the world and the saved settings make it.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LaunchArgs {
    /// What the launch does with its world.
    pub command: LaunchCommand,
    /// Install no log output.
    pub quiet: bool,
    /// Force the graphics-quality preset for this run, unsaved.
    pub graphics_quality: Option<QualityPreset>,
    /// Render with this graphics API, over the one chosen in the saved
    /// settings, without saving it.
    pub graphics_api: Option<Platform>,
    /// Open the window in this mode, over the world's and the saved settings'
    /// choice, without saving it.
    pub window_mode: Option<WindowMode>,
    /// Force the graphics API's validation layer on or off. `None` follows the
    /// build profile.
    pub graphics_validation: Option<bool>,
    /// Turn temporal upscaling on or off, over the world's and the saved
    /// settings' choice, without saving it.
    pub upscaling: Option<bool>,
    /// Force how the ray-tracing acceleration structure tracks moving props.
    /// `None` is [`RtDynamicMode::Auto`].
    pub rt_dynamic: Option<RtDynamicMode>,
    /// Turn ray-traced reflections on or off, over the world's, the saved
    /// settings' and the quality preset's choice, without saving it. A device
    /// without ray tracing still runs without them.
    pub rt_reflections: Option<bool>,
    /// Whether skinned meshes join the ray-tracing acceleration structure.
    /// `None` leaves them in.
    pub rt_skinned_geometry: Option<bool>,
    /// Turn vertical sync on or off, over the world's settings and the saved ones.
    pub vsync: Option<bool>,
    /// Force the DLSS render preset over the saved choice, unsaved.
    pub dlss_preset: Option<DlssPreset>,
    /// Step the frame clock by exactly `1 / rate` seconds per frame instead of
    /// the wall time between frames, so a given frame sees the same clock on
    /// every launch. `None` follows the wall clock.
    pub fixed_frame_rate: Option<NonZeroU32>,
    /// The directory screenshots are written under, instead of
    /// [`DEFAULT_SCREENSHOTS_DIR`].
    pub screenshots_dir: Option<String>,
    /// The feature passes the run starts with. Every flag set is the shipping
    /// frame.
    pub show: ShowFlags,
    /// Stop after this many frames.
    pub limit_frames: Option<u64>,
    /// Cap the frame rate at this many frames a second, `0` for uncapped,
    /// over the world's settings, the saved ones and a benchmark's own cap.
    pub limit_fps: Option<u32>,
    /// Serve the debug protocol on this localhost port, where the host carries
    /// a debug server.
    pub debug_port: Option<u16>,
    /// Milliseconds into the run the simulation runs ahead to before it keeps
    /// the normal pace ([`Self::start_at`] in seconds).
    pub start_at_millis: Option<u32>,
    /// Milliseconds into the run a benchmark stops at, before its last phase
    /// ends ([`Self::stop_at`] in seconds).
    pub stop_at_millis: Option<u32>,
    /// Screenshots to take at moments of the run, alongside the world's own. A
    /// file another shot writes is taken only once.
    pub screenshot_at: Vec<ScreenshotAt>,
}

impl Default for LaunchArgs {
    fn default() -> Self {
        Self::NONE
    }
}

/// A screenshot a launch asks for at a moment of the run, as
/// `--screenshot-at SECONDS[:FILE]` names it.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ScreenshotAt {
    // Milliseconds of the run, so the launch stays comparable for equality.
    millis: u32,
    /// The image's path under the screenshot directory.
    pub file: String,
}

// Milliseconds of `seconds` into the run, none before it starts.
fn run_millis(seconds: f32) -> u32 {
    u32::saturating_from(round(seconds.max(0.0) * 1000.0))
}

// A moment of the run given as seconds: `None` for one that is not a finite,
// non-negative number.
fn parse_seconds(value: &str) -> Option<f32> {
    value
        .parse()
        .ok()
        .filter(|s: &f32| s.is_finite() && *s >= 0.0)
}

impl ScreenshotAt {
    /// A screenshot `seconds` into the run, saved as `file`.
    pub fn new(seconds: f32, file: &str) -> Self {
        Self {
            millis: run_millis(seconds),
            file: file.to_string(),
        }
    }

    /// Seconds into the run, on the simulation clock.
    pub fn seconds(&self) -> f32 {
        self.millis as f32 / 1000.0
    }

    // `SECONDS[:FILE]`, the file named for the moment when left out and given
    // a `.png` extension when it has none. `None` for a moment that is not a
    // number or a file that would leave the screenshot directory.
    fn parse(value: &str) -> Option<Self> {
        let (seconds, file) = match value.split_once(':') {
            Some((seconds, file)) => (seconds, Some(file)),
            None => (value, None),
        };
        let mut shot = Self::new(parse_seconds(seconds)?, "");
        shot.file = match file {
            Some(file) if !crate::screenshot::is_contained_file(file) => return None,
            Some(file)
                if file
                    .rsplit(['/', '\\'])
                    .next()
                    .is_some_and(|f| f.contains('.')) =>
            {
                file.to_string()
            }
            Some(file) => alloc::format!("{file}.png"),
            None => alloc::format!("at-{}.{:03}s.png", shot.millis / 1000, shot.millis % 1000),
        };
        Some(shot)
    }
}

impl LaunchArgs {
    /// A launch that runs the world and asks for nothing else.
    pub const NONE: Self = Self {
        command: LaunchCommand::Run,
        quiet: false,
        graphics_quality: None,
        graphics_api: None,
        window_mode: None,
        graphics_validation: None,
        upscaling: None,
        rt_dynamic: None,
        rt_reflections: None,
        rt_skinned_geometry: None,
        vsync: None,
        dlss_preset: None,
        fixed_frame_rate: None,
        screenshots_dir: None,
        show: ShowFlags::all(),
        limit_frames: None,
        limit_fps: None,
        debug_port: None,
        start_at_millis: None,
        stop_at_millis: None,
        screenshot_at: Vec::new(),
    };

    /// Whether the launch benchmarks its world.
    pub fn benchmarks(&self) -> bool {
        self.command == LaunchCommand::Bench
    }

    /// Seconds into the run, on the simulation clock, the simulation runs
    /// ahead to before it keeps the normal pace; `None` to start at the
    /// normal pace.
    pub fn start_at(&self) -> Option<f32> {
        self.start_at_millis.map(|millis| millis as f32 / 1000.0)
    }

    /// Seconds into the run, on the simulation clock, a benchmark stops at;
    /// `None` to run its every phase.
    pub fn stop_at(&self) -> Option<f32> {
        self.stop_at_millis.map(|millis| millis as f32 / 1000.0)
    }
}
