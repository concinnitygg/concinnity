//! How a failed ray-tracing acceleration-structure update is reported.

use concinnity_core::render::error::RenderResult;
use concinnity_core::render::rt_accel::{FailureStreak, RtUpdate, StreakChange};

/// Log a failed BVH update once when a failure streak begins and once when it
/// ends, rather than on every frame. The renderer keeps the last BVH either way,
/// so a reflection is at most a frame stale while the update fails.
pub(crate) fn report_rt_update(streak: &mut FailureStreak, updated: RenderResult<RtUpdate>) {
    match (streak.record(&updated), updated) {
        (Some(StreakChange::Began), Err(e)) => {
            tracing::warn!("ray-traced reflections: keeping last frame's BVH, update failed: {e}");
        }
        (Some(StreakChange::Ended), _) => {
            tracing::info!("ray-traced reflections: BVH update recovered");
        }
        _ => {}
    }
}
