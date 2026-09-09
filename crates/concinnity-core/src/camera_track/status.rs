// The tick's position along the camera track, published for readers outside
// the simulation.

/// Where a world's [`CameraTrack`](crate::components::CameraTrack) has reached
/// this tick, published by
/// [`CameraTrackSystem`](super::CameraTrackSystem).
///
/// A world with no track never publishes one. The segment is an index into the
/// track's own `segments` list rather than a name, so publishing it costs no
/// allocation on a tick that changes nothing.
#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub struct CameraTrackStatus {
    /// Seconds of simulated time the track has run for.
    pub elapsed_seconds: f32,
    /// Seconds the whole track runs for, the longer of its two lists.
    pub duration_seconds: f32,
    /// Index into the track's `segments`, or `None` while no leg has opened
    /// one.
    pub segment: Option<u32>,
    /// Whether both lists have run out and the camera is holding its last
    /// pose.
    pub finished: bool,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_default_status_reads_as_a_track_that_has_not_started() {
        let s = CameraTrackStatus::default();
        assert_eq!(s.elapsed_seconds, 0.0);
        assert_eq!(s.duration_seconds, 0.0);
        assert_eq!(s.segment, None);
        assert!(!s.finished);
    }
}
