// Which frames a report saves as images, and when.

use concinnity_core::components::FrameCapture;

/// The captures a report has yet to take, handed out as the run reaches them.
#[derive(Debug, Default)]
pub(crate) struct CaptureSchedule {
    // Latest first, so the next one due is at the end. Captures declared for
    // the same moment keep their declared order.
    waiting: Vec<FrameCapture>,
}

impl CaptureSchedule {
    pub(crate) fn new(captures: &[FrameCapture]) -> Self {
        let mut waiting = captures.to_vec();
        waiting.sort_by(|a, b| a.seconds.total_cmp(&b.seconds));
        waiting.reverse();
        Self { waiting }
    }

    /// The capture due once the run has reached `run_seconds`, if any. Each
    /// is handed out once, and at most one per frame, so captures that fall
    /// due together go out on successive frames.
    pub(crate) fn due(&mut self, run_seconds: f32) -> Option<FrameCapture> {
        if self.waiting.last()?.seconds <= run_seconds {
            self.waiting.pop()
        } else {
            None
        }
    }

    /// The captures the run never reached, earliest first.
    pub(crate) fn missed(&self) -> impl Iterator<Item = &FrameCapture> {
        self.waiting.iter().rev()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn capture(seconds: f32, path: &str) -> FrameCapture {
        FrameCapture {
            seconds,
            path: path.to_string(),
        }
    }

    // What each frame, at the given run times, hands out.
    fn fired(schedule: &mut CaptureSchedule, frames: &[f32]) -> Vec<Option<String>> {
        frames
            .iter()
            .map(|&t| schedule.due(t).map(|c| c.path))
            .collect()
    }

    #[test]
    fn a_capture_fires_on_the_first_frame_at_or_after_its_moment() {
        let mut schedule = CaptureSchedule::new(&[capture(1.0, "a.png")]);
        assert_eq!(
            fired(&mut schedule, &[0.5, 0.98, 1.02, 1.5]),
            [None, None, Some("a.png".to_string()), None]
        );
    }

    #[test]
    fn a_frame_landing_exactly_on_the_moment_takes_it() {
        let mut schedule = CaptureSchedule::new(&[capture(2.0, "a.png")]);
        assert_eq!(schedule.due(2.0).map(|c| c.path).as_deref(), Some("a.png"));
    }

    #[test]
    fn each_capture_fires_once_in_time_order_whatever_the_declared_order() {
        let mut schedule =
            CaptureSchedule::new(&[capture(3.0, "late.png"), capture(1.0, "early.png")]);
        assert_eq!(
            fired(&mut schedule, &[1.0, 2.0, 3.0, 4.0]),
            [
                Some("early.png".to_string()),
                None,
                Some("late.png".to_string()),
                None
            ]
        );
        assert_eq!(schedule.missed().count(), 0);
    }

    #[test]
    fn captures_due_together_go_out_on_successive_frames() {
        // A long frame can step over several moments at once; each still gets
        // a frame of its own, in declared order where they tie.
        let mut schedule = CaptureSchedule::new(&[
            capture(1.0, "a.png"),
            capture(1.0, "b.png"),
            capture(1.5, "c.png"),
        ]);
        assert_eq!(
            fired(&mut schedule, &[2.0, 2.1, 2.2, 2.3]),
            [
                Some("a.png".to_string()),
                Some("b.png".to_string()),
                Some("c.png".to_string()),
                None
            ]
        );
    }

    #[test]
    fn a_capture_the_run_never_reaches_is_reported_missed() {
        let mut schedule =
            CaptureSchedule::new(&[capture(9.0, "never.png"), capture(1.0, "a.png")]);
        fired(&mut schedule, &[1.0, 5.0]);
        let missed: Vec<&str> = schedule.missed().map(|c| c.path.as_str()).collect();
        assert_eq!(missed, ["never.png"]);
    }

    #[test]
    fn no_captures_fire_nothing() {
        let mut schedule = CaptureSchedule::new(&[]);
        assert_eq!(fired(&mut schedule, &[0.0, 100.0]), [None, None]);
    }
}
