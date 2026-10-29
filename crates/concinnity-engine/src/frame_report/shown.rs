// Pairs each completed frame's GPU-done time with the camera pose its tick
// left, so motion is measured against the cadence frames finished at rather
// than the cadence the simulation thread happened to sample.

use alloc::collections::VecDeque;
use concinnity_core::math::SaturatingFrom;

use crate::profile::GpuDone;

// Ticks remembered while their frames are still on the GPU. Frames finish a
// few ticks after they were simulated; this is well past that.
const REMEMBERED_TICKS: usize = 16;

/// One newly completed frame, against the one completed before it.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
#[expect(clippy::struct_field_names, reason = "the shared suffix is the unit")]
pub(crate) struct Shown {
    /// Microseconds of run time the camera's pose advanced between the two.
    pub step_us: u32,
    /// Microseconds between the two frames' GPU completions; zero for the
    /// first completion, which has nothing to measure against.
    pub interval_us: u32,
    /// Microseconds from this frame's input sample to its GPU completion.
    pub latency_us: u32,
}

#[derive(Debug, Default)]
pub(crate) struct ShownFrames {
    // (input sample time naming the tick's frame, the pose that tick left).
    poses: VecDeque<(u64, f32)>,
    // The last completion paired: (its input sample time, done time, pose).
    last: Option<(u64, u64, f32)>,
}

impl ShownFrames {
    /// The tick whose input was sampled at `input_sampled_us` left the camera
    /// at `pose_seconds` of run time.
    pub(crate) fn posed(&mut self, input_sampled_us: u64, pose_seconds: f32) {
        if self
            .poses
            .back()
            .is_some_and(|(at, _)| *at == input_sampled_us)
        {
            return;
        }
        if self.poses.len() == REMEMBERED_TICKS {
            self.poses.pop_front();
        }
        self.poses.push_back((input_sampled_us, pose_seconds));
    }

    /// The backend reports `done` as its latest completed frame. `None` when
    /// it was already counted or names a tick no longer remembered.
    pub(crate) fn completed(&mut self, done: GpuDone) -> Option<Shown> {
        if self
            .last
            .is_some_and(|(at, _, _)| at == done.input_sampled_us)
        {
            return None;
        }
        let pose = self
            .poses
            .iter()
            .find(|(at, _)| *at == done.input_sampled_us)
            .map(|(_, pose)| *pose)?;
        let micros = |us: f64| {
            u32::saturating_from(
                concinnity_core::math::round_f64(us).clamp(0.0, f64::from(u32::MAX)),
            )
        };
        let (step_us, interval_us) = match self.last {
            Some((_, last_done, last_pose)) => (
                micros(f64::from(pose - last_pose) * 1e6),
                micros(done.done_us.saturating_sub(last_done) as f64),
            ),
            None => (0, 0),
        };
        self.last = Some((done.input_sampled_us, done.done_us, pose));
        Some(Shown {
            step_us,
            interval_us,
            latency_us: done.latency_us(),
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn done(input_sampled_us: u64, done_us: u64) -> GpuDone {
        GpuDone {
            input_sampled_us,
            done_us,
        }
    }

    #[test]
    fn the_first_completion_measures_latency_only() {
        let mut shown = ShownFrames::default();
        shown.posed(100, 1.0);
        let first = shown
            .completed(done(100, 45_100))
            .expect("a new completion");
        assert_eq!(
            first,
            Shown {
                step_us: 0,
                interval_us: 0,
                latency_us: 45_000
            }
        );
    }

    #[test]
    fn consecutive_completions_measure_pose_against_gpu_interval() {
        let mut shown = ShownFrames::default();
        shown.posed(100, 1.0);
        shown.posed(22_100, 1.022);
        shown.completed(done(100, 45_100));
        let second = shown
            .completed(done(22_100, 70_100))
            .expect("a new completion");
        assert_eq!(second.step_us, 22_000);
        assert_eq!(second.interval_us, 25_000);
        assert_eq!(second.latency_us, 48_000);
    }

    #[test]
    fn a_completion_reported_twice_counts_once() {
        let mut shown = ShownFrames::default();
        shown.posed(100, 1.0);
        assert!(shown.completed(done(100, 45_100)).is_some());
        assert!(shown.completed(done(100, 45_100)).is_none());
    }

    // A frame finished between two reads still counts: the next completion
    // spans both, pose and time alike.
    #[test]
    fn a_skipped_completion_is_spanned_by_the_next() {
        let mut shown = ShownFrames::default();
        for (n, at) in [100, 22_100, 44_100].into_iter().enumerate() {
            shown.posed(at, 1.0 + n as f32 * 0.022);
        }
        shown.completed(done(100, 45_100));
        let spanned = shown
            .completed(done(44_100, 89_100))
            .expect("a new completion");
        assert_eq!(spanned.interval_us, 44_000);
        assert!(
            (i64::from(spanned.step_us) - 44_000).abs() <= 1,
            "{spanned:?}"
        );
    }

    #[test]
    fn a_tick_long_forgotten_is_not_paired() {
        let mut shown = ShownFrames::default();
        shown.posed(1, 0.0);
        for n in 0..REMEMBERED_TICKS as u64 {
            shown.posed(100 + n, 1.0);
        }
        assert!(shown.completed(done(1, 50_000)).is_none());
    }
}
