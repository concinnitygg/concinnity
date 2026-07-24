// src/gfx/streaming_system/stats_log.rs
//
// Change gate for the streaming pools' periodic counter line. The counters are
// sampled on a fixed frame interval, but a settled pool holds the same numbers
// for as long as the world runs, so a sample is logged only when it moved since
// the last one logged.

// Frames between counter samples (~2s near 60 fps).
const SAMPLE_INTERVAL: u64 = 120;

// One pool's last-logged counters, where `T` is that pool's counter tuple.
#[derive(Debug, Default)]
pub(crate) struct StatsHeartbeat<T> {
    last: Option<T>,
}

impl<T: Copy + PartialEq> StatsHeartbeat<T> {
    // The counters to log, or `None` when this frame is not a sample or they
    // have not moved since the last logged one. `counts` is only called on a
    // sample frame, keeping its per-item scan off the other 119.
    pub(crate) fn sample(&mut self, frame: u64, counts: impl FnOnce() -> T) -> Option<T> {
        if !frame.is_multiple_of(SAMPLE_INTERVAL) {
            return None;
        }
        let counts = counts();
        if self.last == Some(counts) {
            return None;
        }
        self.last = Some(counts);
        Some(counts)
    }
}

// The per-pool gates `StreamingState` carries.
#[derive(Debug, Default)]
pub(crate) struct PoolHeartbeats {
    // `(resident, pending, unloaded)` for the texture and mesh pools.
    pub(crate) texture: StatsHeartbeat<(usize, usize, usize)>,
    pub(crate) mesh: StatsHeartbeat<(usize, usize, usize)>,
    // Chunks add the `(full, impostor)` detail split to `(resident, pending)`.
    pub(crate) chunk: StatsHeartbeat<(usize, usize, usize, usize)>,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn emits_the_first_sample() {
        let mut hb = StatsHeartbeat::default();
        assert_eq!(hb.sample(0, || (0, 4, 0)), Some((0, 4, 0)));
    }

    #[test]
    fn skips_frames_off_the_interval() {
        let mut hb = StatsHeartbeat::default();
        for frame in 1..SAMPLE_INTERVAL {
            assert_eq!(hb.sample(frame, || (frame as usize, 0, 0)), None);
        }
    }

    // The counter scan is skipped outright on a non-sample frame.
    #[test]
    fn does_not_read_the_counters_off_the_interval() {
        let mut hb = StatsHeartbeat::default();
        let mut reads = 0;
        for frame in 0..SAMPLE_INTERVAL * 2 {
            hb.sample(frame, || {
                reads += 1;
                (0, 0, 0)
            });
        }
        assert_eq!(reads, 2);
    }

    #[test]
    fn skips_a_repeated_sample() {
        let mut hb = StatsHeartbeat::default();
        assert!(hb.sample(SAMPLE_INTERVAL, || (399, 0, 0)).is_some());
        for tick in 2..10 {
            assert_eq!(hb.sample(SAMPLE_INTERVAL * tick, || (399, 0, 0)), None);
        }
    }

    #[test]
    fn emits_again_once_the_counters_move() {
        let mut hb = StatsHeartbeat::default();
        assert!(hb.sample(SAMPLE_INTERVAL, || (0, 8, 0)).is_some());
        assert_eq!(hb.sample(SAMPLE_INTERVAL * 2, || (0, 8, 0)), None);
        assert_eq!(
            hb.sample(SAMPLE_INTERVAL * 3, || (8, 0, 0)),
            Some((8, 0, 0))
        );
    }

    // Movement between two samples is invisible: the gate compares against the
    // last sample it logged, not the last one it saw.
    #[test]
    fn returns_to_the_logged_sample() {
        let mut hb = StatsHeartbeat::default();
        assert!(hb.sample(SAMPLE_INTERVAL, || (4, 0, 0)).is_some());
        assert_eq!(hb.sample(SAMPLE_INTERVAL + 1, || (0, 4, 0)), None);
        assert_eq!(hb.sample(SAMPLE_INTERVAL * 2, || (4, 0, 0)), None);
    }

    #[test]
    fn tracks_each_pool_independently() {
        let mut hb = PoolHeartbeats::default();
        assert!(hb.texture.sample(SAMPLE_INTERVAL, || (399, 0, 0)).is_some());
        assert!(hb.mesh.sample(SAMPLE_INTERVAL, || (399, 0, 0)).is_some());
        assert_eq!(hb.texture.sample(SAMPLE_INTERVAL * 2, || (399, 0, 0)), None);
        assert!(
            hb.mesh
                .sample(SAMPLE_INTERVAL * 2, || (1610, 0, 0))
                .is_some()
        );
        assert!(hb.chunk.sample(SAMPLE_INTERVAL, || (12, 8, 4, 0)).is_some());
    }
}
