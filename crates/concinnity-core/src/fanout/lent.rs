use core::fmt;

use super::FanOut;
use super::ScheduleMode;

/// A host's pool as a system holds it: the pool each schedule mode runs on.
///
/// A serial schedule picks a single-worker pool rather than none, so the
/// determinism oracle runs the same code path a parallel one does.
pub struct Lent<P: 'static> {
    pick: fn(ScheduleMode) -> &'static P,
}

impl<P: FanOut> Lent<P> {
    /// Lend the pool `pick` names for each schedule mode.
    pub const fn new(pick: fn(ScheduleMode) -> &'static P) -> Self {
        Self { pick }
    }

    /// The pool `mode` runs on.
    pub fn pool(&self, mode: ScheduleMode) -> &'static P {
        (self.pick)(mode)
    }
}

impl<P> Clone for Lent<P> {
    fn clone(&self) -> Self {
        *self
    }
}

impl<P> Copy for Lent<P> {}

impl<P> fmt::Debug for Lent<P> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str("Lent")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    use crate::fanout::test_pools::Reversed;

    static WIDE: Reversed = Reversed(8);
    static NARROW: Reversed = Reversed(1);

    fn pick(mode: ScheduleMode) -> &'static Reversed {
        match mode {
            ScheduleMode::Parallel => &WIDE,
            ScheduleMode::Serial => &NARROW,
        }
    }

    #[test]
    fn each_mode_reaches_the_pool_it_names() {
        let lent = Lent::new(pick);
        assert_eq!(lent.pool(ScheduleMode::Parallel).workers(), 8);
        assert_eq!(lent.pool(ScheduleMode::Serial).workers(), 1);
    }
}
