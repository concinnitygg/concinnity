//! When a pipelined loop lets the simulation step: just in time behind the
//! GPU, or overlapping the previous frame's submit.

// A frame that waited this long on the GPU was GPU-bound.
const GPU_BOUND_WAIT_US: u32 = 1_000;
// Frames in a row on the other side of the bound before the timing switches,
// so a frame that happens to straddle it does not flip the loop's shape.
const SWITCH_AFTER_FRAMES: u32 = 4;

/// When the render half lets the next step go: just in time, once the next
/// frame's GPU slot is reserved, while frames wait on the GPU; or as soon as
/// the current snapshot arrives while they do not, so the step overlaps that
/// frame's submit instead of queueing behind it.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct GoTiming {
    just_in_time: bool,
    // Consecutive frames that disagreed with the current timing.
    against: u32,
}

impl Default for GoTiming {
    fn default() -> Self {
        Self {
            just_in_time: true,
            against: 0,
        }
    }
}

impl GoTiming {
    /// Whether the next step waits for its frame's GPU slot.
    pub fn just_in_time(self) -> bool {
        self.just_in_time
    }

    /// Account for a frame that spent `gpu_wait_us` blocked on the GPU.
    pub fn record(&mut self, gpu_wait_us: u32) {
        let gpu_bound = gpu_wait_us >= GPU_BOUND_WAIT_US;
        if gpu_bound == self.just_in_time {
            self.against = 0;
            return;
        }
        self.against += 1;
        if self.against >= SWITCH_AFTER_FRAMES {
            self.just_in_time = gpu_bound;
            self.against = 0;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use alloc::vec;

    fn after(waits: &[u32]) -> GoTiming {
        let mut timing = GoTiming::default();
        for &wait in waits {
            timing.record(wait);
        }
        timing
    }

    #[test]
    fn a_gpu_bound_loop_stays_just_in_time() {
        assert!(after(&[20_000; 10]).just_in_time());
    }

    #[test]
    fn a_loop_that_never_waits_overlaps_after_a_few_frames() {
        let waits = [0; SWITCH_AFTER_FRAMES as usize];
        assert!(after(&waits[..waits.len() - 1]).just_in_time());
        assert!(!after(&waits).just_in_time());
    }

    #[test]
    fn a_loop_that_starts_waiting_goes_back_to_just_in_time() {
        let mut waits = vec![0; SWITCH_AFTER_FRAMES as usize];
        waits.extend([GPU_BOUND_WAIT_US; SWITCH_AFTER_FRAMES as usize]);
        assert!(after(&waits).just_in_time());
    }

    // One frame across the bound does not flip the timing.
    #[test]
    fn a_lone_frame_across_the_bound_changes_nothing() {
        let mut waits = vec![0; SWITCH_AFTER_FRAMES as usize];
        waits.push(GPU_BOUND_WAIT_US);
        waits.extend([0; SWITCH_AFTER_FRAMES as usize - 1]);
        assert!(!after(&waits).just_in_time());
    }
}
