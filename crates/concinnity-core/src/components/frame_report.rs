// Frame-report schema: how a world asks its frames to be measured.

use crate::ecs::asset_id::AssetId;

/// A 60 Hz frame, the budget a report counts against unless a world says
/// otherwise.
const DEFAULT_BUDGET_MS: f32 = 1000.0 / 60.0;

/// Times every frame of a run and prints what they cost when it ends.
///
/// One per world. The report leads with frame time, because that is the number
/// a player feels, and gives its median and tail percentiles rather than its
/// mean: a mean hides exactly the stutter that matters. Beside it are the
/// frame's CPU work and its GPU time, so a stretch reads as CPU-bound or
/// GPU-bound at a glance, and under it the render passes and the systems that
/// owned each, with their shares.
///
/// A run that names segments as it goes is reported segment by segment as well
/// as whole, so a cost that belongs to one stretch of it says so instead of
/// spreading itself thinly over the average.
///
/// ```rust
/// # use concinnity_core::components::FrameReport;
/// FrameReport {
///     warmup_seconds: 3.0,
///     budget_ms: 8.333,
///     ..Default::default()
/// };
/// ```
#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
#[serde(default)]
pub struct FrameReport {
    /// Asset identity; injected via `inject_name`. Not part of `args`.
    #[serde(skip)]
    pub asset_id: AssetId,
    /// Frames earlier than this many seconds into the run are dropped.
    ///
    /// Shader compilation, streaming residency, temporal-antialiasing history
    /// and auto-exposure all converge over the opening seconds of a run.
    /// Without a discard, two runs of identical code disagree, so this is what
    /// makes one run comparable with the next.
    pub warmup_seconds: f32,
    /// The frame budget, in milliseconds, that the over-budget count is taken
    /// against. Defaults to a 60 Hz frame.
    pub budget_ms: f32,
    /// Whether the run ends as soon as the world says the measurement is
    /// complete.
    ///
    /// `true` reports and stops. `false` keeps the world running and reports
    /// when it closes instead, for a world watched rather than left to itself.
    /// A world that never says it is complete runs on either way, and reports
    /// whatever it gathered on the way out.
    ///
    /// A [CameraTrack](#cameratrack) is one thing that says so, when it
    /// reaches the end of its path.
    pub stop_when_complete: bool,
}

impl Default for FrameReport {
    fn default() -> Self {
        Self {
            asset_id: AssetId::default(),
            warmup_seconds: 2.0,
            budget_ms: DEFAULT_BUDGET_MS,
            stop_when_complete: true,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_defaults_measure_a_run_against_a_sixty_hertz_frame() {
        let r = FrameReport::default();
        assert_eq!(r.warmup_seconds, 2.0);
        assert!((r.budget_ms - 16.667).abs() < 1e-3, "{}", r.budget_ms);
        // A world declaring one is asking to be measured, so it ends when the
        // measurement does.
        assert!(r.stop_when_complete);
    }

    #[test]
    fn an_authored_report_parses_and_round_trips_through_postcard() {
        let r: FrameReport = serde_json::from_str(
            r#"{"warmup_seconds":4,"budget_ms":8.333,"stop_when_complete":false}"#,
        )
        .unwrap();
        assert_eq!(r.warmup_seconds, 4.0);
        assert_eq!(r.budget_ms, 8.333);
        assert!(!r.stop_when_complete);

        let bytes = postcard::to_allocvec(&r).unwrap();
        let back: FrameReport = postcard::from_bytes(&bytes).unwrap();
        assert_eq!(back, r);
    }

    #[test]
    fn an_omitted_field_keeps_its_default() {
        let r: FrameReport = serde_json::from_str(r#"{"warmup_seconds":0}"#).unwrap();
        assert_eq!(r.warmup_seconds, 0.0);
        assert!((r.budget_ms - 16.667).abs() < 1e-3);
        assert!(r.stop_when_complete);
    }
}
