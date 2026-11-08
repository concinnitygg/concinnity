// The shots a run has yet to take, handed out as the clock reaches them.

use alloc::string::String;
use alloc::vec::Vec;

/// One shot to take: when, into what file, and whether the HUD shows.
#[derive(Debug, Clone, PartialEq)]
pub(super) struct Shot {
    pub(super) seconds: f32,
    pub(super) file: String,
    pub(super) hud: bool,
}

#[derive(Debug, Default)]
pub(super) struct ShotSchedule {
    // Latest first, so the next one due is at the end. Shots declared for the
    // same moment keep their declared order.
    waiting: Vec<Shot>,
}

impl ShotSchedule {
    pub(super) fn new(mut shots: Vec<Shot>) -> Self {
        shots.sort_by(|a, b| a.seconds.total_cmp(&b.seconds));
        shots.reverse();
        Self { waiting: shots }
    }

    /// The shot due once the clock has reached `seconds`, if any. Each is
    /// handed out once, and at most one per frame, so shots that fall due
    /// together go out on successive frames.
    pub(super) fn due(&mut self, seconds: f32) -> Option<Shot> {
        if self.waiting.last()?.seconds <= seconds {
            self.waiting.pop()
        } else {
            None
        }
    }

    /// Take the shots due after `seconds` out of the schedule, earliest first.
    pub(super) fn take_after(&mut self, seconds: f32) -> impl Iterator<Item = Shot> + '_ {
        let late = self
            .waiting
            .iter()
            .take_while(|s| s.seconds > seconds)
            .count();
        self.waiting.drain(..late).rev()
    }

    /// The shots not yet taken, earliest first.
    pub(super) fn waiting(&self) -> impl Iterator<Item = &Shot> {
        self.waiting.iter().rev()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use alloc::string::ToString;
    use alloc::vec;

    fn shot(seconds: f32, file: &str) -> Shot {
        Shot {
            seconds,
            file: file.to_string(),
            hud: true,
        }
    }

    fn fired(schedule: &mut ShotSchedule, frames: &[f32]) -> Vec<Option<String>> {
        frames
            .iter()
            .map(|&t| schedule.due(t).map(|s| s.file))
            .collect()
    }

    #[test]
    fn a_shot_fires_once_on_the_first_frame_at_or_after_its_moment() {
        let mut schedule = ShotSchedule::new(vec![shot(1.0, "a.png")]);
        assert_eq!(
            fired(&mut schedule, &[0.5, 1.2, 1.4]),
            [None, Some("a.png".to_string()), None]
        );
    }

    #[test]
    fn shots_go_out_in_time_order_one_a_frame() {
        let mut schedule = ShotSchedule::new(vec![
            shot(3.0, "late.png"),
            shot(1.0, "first.png"),
            shot(1.0, "second.png"),
        ]);
        assert_eq!(
            fired(&mut schedule, &[5.0, 5.0, 5.0, 5.0]),
            [
                Some("first.png".to_string()),
                Some("second.png".to_string()),
                Some("late.png".to_string()),
                None
            ]
        );
    }

    #[test]
    fn shots_past_an_end_are_taken_out() {
        let mut schedule = ShotSchedule::new(vec![
            shot(9.0, "c.png"),
            shot(1.0, "a.png"),
            shot(7.0, "b.png"),
        ]);
        let late: Vec<String> = schedule.take_after(5.0).map(|s| s.file).collect();
        assert_eq!(late, ["b.png", "c.png"]);
        let left: Vec<&str> = schedule.waiting().map(|s| s.file.as_str()).collect();
        assert_eq!(left, ["a.png"]);
    }

    #[test]
    fn shots_never_reached_are_still_waiting() {
        let mut schedule = ShotSchedule::new(vec![shot(9.0, "never.png"), shot(1.0, "a.png")]);
        fired(&mut schedule, &[2.0]);
        let left: Vec<&str> = schedule.waiting().map(|s| s.file.as_str()).collect();
        assert_eq!(left, ["never.png"]);
    }
}
