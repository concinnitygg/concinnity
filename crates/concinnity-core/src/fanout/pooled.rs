use core::fmt;

use super::ScheduleMode;
use super::{Fanout, Inline, Lent};
use crate::fanout::FanOutTiming;

/// A pool as a system holds it: erased over the host's pool type, typed by the
/// items the system hands out, picked per schedule mode.
pub trait Pooled<T>: fmt::Debug + Send + Sync {
    /// Units of work the pool `mode` picks runs at once.
    fn workers(&self, mode: ScheduleMode) -> usize;

    /// Run `body` over every item on the pool `mode` picks.
    fn for_each(&self, mode: ScheduleMode, items: &mut [T], body: &(dyn Fn(&mut T) + Send + Sync));

    /// `for_each`, reporting how long the caller waited against the work.
    fn for_each_timed(
        &self,
        mode: ScheduleMode,
        items: &mut [T],
        body: &(dyn Fn(&mut T) + Send + Sync),
    ) -> FanOutTiming;
}

impl<T: Send> Pooled<T> for Inline {
    fn workers(&self, _mode: ScheduleMode) -> usize {
        1
    }

    fn for_each(
        &self,
        _mode: ScheduleMode,
        items: &mut [T],
        body: &(dyn Fn(&mut T) + Send + Sync),
    ) {
        Fanout::for_each(self, items, body);
    }

    fn for_each_timed(
        &self,
        _mode: ScheduleMode,
        items: &mut [T],
        body: &(dyn Fn(&mut T) + Send + Sync),
    ) -> FanOutTiming {
        Fanout::for_each_timed(self, items, body)
    }
}

impl<P: Fanout, T: Send> Pooled<T> for Lent<P> {
    fn workers(&self, mode: ScheduleMode) -> usize {
        self.pool(mode).workers()
    }

    fn for_each(&self, mode: ScheduleMode, items: &mut [T], body: &(dyn Fn(&mut T) + Send + Sync)) {
        self.pool(mode).for_each(items, body);
    }

    fn for_each_timed(
        &self,
        mode: ScheduleMode,
        items: &mut [T],
        body: &(dyn Fn(&mut T) + Send + Sync),
    ) -> FanOutTiming {
        self.pool(mode).for_each_timed(items, body)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::fanout::test_pools::Reversed;
    use alloc::vec::Vec;

    static REVERSED: Reversed = Reversed(1);

    // Both forms visit every item once; a lent pool reports its own width.
    #[test]
    fn inline_and_lent_pools_visit_every_item() {
        let lent = Lent::new(|_| &REVERSED);
        let pools: [&dyn Pooled<u32>; 2] = [&Inline, &lent];
        for pool in pools {
            let mut items: Vec<u32> = (0..5).collect();
            pool.for_each(ScheduleMode::Parallel, &mut items, &|v| *v += 10);
            pool.for_each_timed(ScheduleMode::Serial, &mut items, &|v| *v += 10);
            assert_eq!(items, [20, 21, 22, 23, 24]);
        }
        assert_eq!(Pooled::<u32>::workers(&Inline, ScheduleMode::Parallel), 1);
        assert_eq!(Pooled::<u32>::workers(&lent, ScheduleMode::Parallel), 1);
    }
}
