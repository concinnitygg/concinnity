// Benchmark schema: what a benchmarked run measures, how the camera moves, and
// how long it lasts.

use alloc::string::String;
use alloc::vec::Vec;

use crate::components::{CameraTravel, CameraTurn};

/// A 60 Hz frame, the budget a run counts against unless a world says
/// otherwise.
const DEFAULT_BUDGET_MS: f32 = 1000.0 / 60.0;

// How long a benchmark that names no phases of its own runs for.
const DEFAULT_PHASE_SECONDS: f32 = 10.0;

// The name of the one phase a benchmark that names none of its own runs.
const DEFAULT_PHASE_NAME: &str = "run";

/// Customizes how a world is benchmarked.
///
/// A world is benchmarked when it is launched with `--benchmark`, whether it
/// declares one of these or not. Without one the camera holds the pose the
/// world starts it at for ten seconds after the warm-up. Declaring one does
/// nothing on an ordinary launch, so a world can ship its benchmark beside it.
///
/// A benchmarked run takes the camera away from the player: it holds still
/// through the warm-up, then follows `travel` and `turn` from where it stood.
/// The legs are the same ones a [`CameraTrack`](#cameratrack) takes, and played
/// on the same fixed simulation clock, so a slower machine visits the same
/// poses at the same times and simply samples fewer of them.
///
/// The run is cut into `phases`, named stretches of time that the report
/// gives one by one as well as whole, so a cost that belongs to one stretch
/// says so instead of spreading itself thinly over the average. The run ends
/// when the last phase does: a path shorter than that holds its last pose, and
/// a longer one is cut off.
///
/// The report leads with frame time, because that is the number a player
/// feels, and gives its median and tail percentiles rather than its mean: a
/// mean hides exactly the stutter that matters. Beside it are the frame's CPU
/// work and its GPU time, so a phase reads as CPU-bound or GPU-bound at a
/// glance, and under it the render passes and the systems that owned each,
/// with their shares.
///
/// ```rust
/// # use concinnity_engine::components::{BenchmarkConfig, BenchmarkPhase, CameraTravel};
/// BenchmarkConfig {
///     travel: vec![CameraTravel {
///         direction: [0.0, 0.0, -1.0],
///         distance: 200.0,
///         speed: 10.0,
///         ..Default::default()
///     }],
///     phases: vec![
///         BenchmarkPhase { name: "street".into(), seconds: 12.0 },
///         BenchmarkPhase { name: "plaza".into(), seconds: 8.0 },
///     ],
///     ..Default::default()
/// };
/// ```
#[derive(
    Debug,
    Clone,
    PartialEq,
    serde::Serialize,
    serde::Deserialize,
    crate::ecs::AssetFields,
    crate::ecs::AssetDefault,
)]
#[serde(default)]
pub struct BenchmarkConfig {
    /// Seconds the camera holds its start pose before the first phase, with
    /// every frame drawn in them left out of the measurement.
    ///
    /// Shader compilation, streaming residency, temporal-antialiasing history
    /// and auto-exposure all converge over the opening seconds of a run.
    /// Without a discard, two runs of identical code disagree, so this is what
    /// makes one run comparable with the next.
    #[asset(default = 2.0)]
    pub warmup_seconds: f32,
    /// The frame budget, in milliseconds, that the over-budget count is taken
    /// against. Defaults to a 60 Hz frame.
    #[asset(default = DEFAULT_BUDGET_MS)]
    pub budget_ms: f32,
    /// The frame-rate cap the run holds to, whatever the world or the player's
    /// settings choose. `0` leaves it uncapped, since a capped GPU idles
    /// between frames and lowers its clocks.
    pub fps_cap: u32,
    /// Where the camera starts, or unset to start it where the world does.
    pub start: Option<CameraPose>,
    /// Where the camera goes after the warm-up, leg by leg.
    pub travel: Vec<CameraTravel>,
    /// Where the camera looks after the warm-up, leg by leg. Played against
    /// the same clock as `travel` and independent of it.
    pub turn: Vec<CameraTurn>,
    /// The named stretches the run is measured in, back to back from the end
    /// of the warm-up. Left empty, the run is one ten-second phase.
    pub phases: Vec<BenchmarkPhase>,
    /// Frames saved as PNG images along the way, each the first frame drawn at
    /// or after its moment of the run.
    ///
    /// The moments are counted from the end of the warm-up, on the clock the
    /// camera moves by, so a capture shows the same pose on every run. A
    /// capture is copied out within its frame and written to disk in the
    /// background, so the frames around it are measured like any other.
    pub captures: Vec<FrameCapture>,
}

impl BenchmarkConfig {
    /// The phases the run is measured in: the declared ones, or the one
    /// default phase when none are.
    pub fn resolved_phases(&self) -> Vec<BenchmarkPhase> {
        if self.phases.is_empty() {
            alloc::vec![BenchmarkPhase {
                name: DEFAULT_PHASE_NAME.into(),
                seconds: DEFAULT_PHASE_SECONDS,
            }]
        } else {
            self.phases.clone()
        }
    }
}

/// One named stretch of a [`BenchmarkConfig`](#benchmarkconfig)'s run.
///
/// ```rust
/// # use concinnity_engine::components::BenchmarkPhase;
/// BenchmarkPhase { name: "orbit".into(), seconds: 14.0 };
/// ```
#[derive(
    Debug, Clone, Default, PartialEq, serde::Serialize, serde::Deserialize, crate::ecs::AssetFields,
)]
#[serde(default)]
pub struct BenchmarkPhase {
    /// What the report calls the phase.
    pub name: String,
    /// How long the phase lasts, in seconds of simulated time.
    pub seconds: f32,
}

/// Where a [`BenchmarkConfig`](#benchmarkconfig) starts the camera.
///
/// ```rust
/// # use concinnity_engine::components::CameraPose;
/// CameraPose {
///     position: [0.0, 2.0, 10.0],
///     yaw_deg: 180.0,
///     pitch_deg: -5.0,
/// };
/// ```
#[derive(
    Debug,
    Clone,
    Copy,
    Default,
    PartialEq,
    serde::Serialize,
    serde::Deserialize,
    crate::ecs::AssetFields,
)]
#[serde(default)]
pub struct CameraPose {
    /// World-space position.
    pub position: [f32; 3],
    /// Heading in degrees. `0` looks toward -Z.
    pub yaw_deg: f32,
    /// Elevation in degrees. Positive looks up.
    pub pitch_deg: f32,
}

/// One frame a [`BenchmarkConfig`](#benchmarkconfig) saves as an image.
///
/// ```rust
/// # use concinnity_engine::components::{BenchmarkConfig, FrameCapture};
/// BenchmarkConfig {
///     captures: vec![FrameCapture {
///         seconds: 30.0,
///         path: "captures/sunset.png".to_string(),
///     }],
///     ..Default::default()
/// };
/// ```
#[derive(
    Debug, Clone, Default, PartialEq, serde::Serialize, serde::Deserialize, crate::ecs::AssetFields,
)]
#[serde(default)]
pub struct FrameCapture {
    /// How many seconds after the warm-up the frame is taken.
    pub seconds: f32,
    /// Where the PNG is written. A relative path is taken from the run's
    /// screenshot directory (`screenshots` under the working directory the run
    /// started in, unless the launch names another); an absolute one is used
    /// as given.
    pub path: String,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_benchmark_naming_no_phases_runs_one_ten_second_phase() {
        let phases = BenchmarkConfig::default().resolved_phases();
        assert_eq!(phases.len(), 1);
        assert_eq!(phases[0].name, DEFAULT_PHASE_NAME);
        assert_eq!(phases[0].seconds, DEFAULT_PHASE_SECONDS);
    }

    #[test]
    fn declared_phases_replace_the_default_one() {
        let config = BenchmarkConfig {
            phases: alloc::vec![BenchmarkPhase {
                name: "a".into(),
                seconds: 3.0
            }],
            ..Default::default()
        };
        assert_eq!(config.resolved_phases(), config.phases);
    }

    #[test]
    fn the_defaults_warm_up_uncapped_against_a_60_hz_budget() {
        let config = BenchmarkConfig::default();
        assert_eq!(config.warmup_seconds, 2.0);
        assert_eq!(config.budget_ms, DEFAULT_BUDGET_MS);
        assert_eq!(config.fps_cap, 0);
        assert_eq!(config.start, None);
    }

    #[test]
    fn an_authored_entry_fills_what_it_leaves_out_with_the_defaults() {
        let config: BenchmarkConfig =
            serde_json::from_str(r#"{"phases":[{"name":"x","seconds":4.0}]}"#).unwrap();
        assert_eq!(config.warmup_seconds, 2.0);
        assert_eq!(config.phases[0].seconds, 4.0);
    }
}
