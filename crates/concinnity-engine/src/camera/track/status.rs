// The tick's position along the camera track, published for readers outside
// the simulation.

use crate::ecs::asset_id::AssetId;

/// Where a world's [`CameraTrack`](crate::components::CameraTrack) has reached
/// this tick, published by
/// [`CameraTrackSystem`](super::CameraTrackSystem).
///
/// A world with no track never publishes one.
#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub(crate) struct CameraTrackStatus {
    /// Seconds of simulated time the track has run for, counted in whole
    /// fixed steps.
    pub elapsed_seconds: f32,
    /// Track time the camera is posed at this frame: up to one fixed step
    /// behind `elapsed_seconds`, blended by the frame's
    /// [`SimTiming::alpha`](crate::ecs::SimTiming::alpha) so the camera moves
    /// as far as the frame took.
    pub pose_seconds: f32,
    /// Seconds the whole track runs for, the longer of its two lists.
    pub duration_seconds: f32,
    /// Whether both lists have run out and the camera is holding its last
    /// pose.
    pub finished: bool,
    /// The prop the track carries, once it has accepted it: the one whose
    /// Transform it owns.
    pub subject: Option<AssetId>,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_default_status_reads_as_a_track_that_has_not_started() {
        let s = CameraTrackStatus::default();
        assert_eq!(s.elapsed_seconds, 0.0);
        assert_eq!(s.pose_seconds, 0.0);
        assert_eq!(s.duration_seconds, 0.0);
        assert!(!s.finished);
    }
}
