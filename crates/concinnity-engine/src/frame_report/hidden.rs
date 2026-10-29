// Frames the window could not show, which the measurement leaves out.

use alloc::vec::Vec;

/// Frames left out once the window shows again: the first spans the pacing a
/// hidden window runs under, and the GPU timings arrive frames behind.
pub(crate) const RESTORE_DROPPED_FRAMES: u32 = 4;

/// Which frames ran while the window was minimized or hidden, when the backend
/// draws nothing and the wall time is only the background pacing, and how
/// many of them fell where the run is measured, by phase.
#[derive(Debug, Default)]
pub(crate) struct HiddenFrames {
    // Frames still to leave out behind the latest hidden one.
    settling: u32,
    // Frames left out past the warm-up, by phase, in the order first seen.
    counts: Vec<(Option<u32>, u32)>,
}

impl HiddenFrames {
    /// Whether a frame drawn in `phase` is left out, given whether the
    /// window was `hidden` for it. Only frames `past_warmup` are tallied,
    /// since the warm-up is dropped anyway.
    pub(crate) fn leaves_out(
        &mut self,
        hidden: bool,
        phase: Option<u32>,
        past_warmup: bool,
    ) -> bool {
        if hidden {
            self.settling = RESTORE_DROPPED_FRAMES;
        } else if self.settling > 0 {
            self.settling -= 1;
        } else {
            return false;
        }
        if past_warmup {
            match self.counts.iter_mut().find(|(s, _)| *s == phase) {
                Some((_, count)) => *count += 1,
                None => self.counts.push((phase, 1)),
            }
        }
        true
    }

    /// Frames left out past the warm-up.
    pub(crate) fn total(&self) -> u32 {
        self.counts.iter().map(|(_, count)| count).sum()
    }

    /// Frames left out past the warm-up, by phase, in the order first seen.
    pub(crate) fn by_phase(&self) -> impl Iterator<Item = (Option<u32>, u32)> + '_ {
        self.counts.iter().copied()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_window_that_stays_shown_leaves_nothing_out() {
        let mut hidden = HiddenFrames::default();
        for _ in 0..10 {
            assert!(!hidden.leaves_out(false, Some(0), true));
        }
        assert_eq!(hidden.total(), 0);
    }

    #[test]
    fn hidden_frames_and_those_settling_after_them_are_left_out() {
        let mut hidden = HiddenFrames::default();
        assert!(!hidden.leaves_out(false, Some(0), true));
        assert!(hidden.leaves_out(true, Some(0), true));
        assert!(hidden.leaves_out(true, Some(0), true));
        for _ in 0..RESTORE_DROPPED_FRAMES {
            assert!(hidden.leaves_out(false, Some(0), true));
        }
        assert!(!hidden.leaves_out(false, Some(0), true));
        assert_eq!(hidden.total(), 2 + RESTORE_DROPPED_FRAMES);
    }

    #[test]
    fn hiding_again_while_settling_restarts_the_settle() {
        let mut hidden = HiddenFrames::default();
        hidden.leaves_out(true, None, true);
        hidden.leaves_out(false, None, true);
        hidden.leaves_out(true, None, true);
        for _ in 0..RESTORE_DROPPED_FRAMES {
            assert!(hidden.leaves_out(false, None, true));
        }
        assert!(!hidden.leaves_out(false, None, true));
    }

    #[test]
    fn frames_are_tallied_by_phase_in_the_order_first_seen() {
        let mut hidden = HiddenFrames::default();
        hidden.leaves_out(true, Some(2), true);
        hidden.leaves_out(true, Some(0), true);
        hidden.leaves_out(true, Some(2), true);
        let tally: Vec<_> = hidden.by_phase().collect();
        assert_eq!(tally, [(Some(2), 2), (Some(0), 1)]);
    }

    #[test]
    fn the_warm_up_is_left_out_but_not_tallied() {
        let mut hidden = HiddenFrames::default();
        assert!(hidden.leaves_out(true, Some(0), false));
        assert!(hidden.leaves_out(false, Some(0), false));
        assert_eq!(hidden.total(), 0);
        assert!(hidden.leaves_out(false, Some(0), true));
        assert_eq!(hidden.total(), 1);
    }
}
