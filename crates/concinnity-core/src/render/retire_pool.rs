//! A frame-tagged deferred-free pool for GPU resources the CPU has replaced
//! while an in-flight frame may still read them.
//!
//! The replaced handle is parked with the tick it was retired on and handed back
//! only once `depth` more ticks have passed, by which point the frames-in-flight
//! fence guarantees every frame that could still reference it has retired on the
//! GPU. Dropping what comes back frees it; a backend whose handles need an
//! explicit destroy takes them out with [`RetirePool::pop_due`].
//!
//! This deliberately does not key storage by `frame % depth` the way a per-frame
//! ring does: a ring is only safe when every slot is rewritten every frame, and a
//! retired resource is written once and then only waited out.

use alloc::collections::VecDeque;

/// Parked resources, each with the tick it was retired on.
pub struct RetirePool<T> {
    // `(retired_at, item)`, pushed in nondecreasing tick order.
    pending: VecDeque<(u64, T)>,
}

impl<T> Default for RetirePool<T> {
    fn default() -> Self {
        Self::new()
    }
}

impl<T> RetirePool<T> {
    /// An empty pool.
    pub fn new() -> Self {
        Self {
            pending: VecDeque::new(),
        }
    }

    /// Park `item`, retired on tick `retired_at`. Ticks must not decrease from
    /// one push to the next.
    pub fn push(&mut self, retired_at: u64, item: T) {
        self.pending.push_back((retired_at, item));
    }

    /// The oldest parked item whose window has closed by tick `now`
    /// (`retired_at + depth <= now`), or `None` when none has.
    pub fn pop_due(&mut self, now: u64, depth: u64) -> Option<T> {
        let &(retired_at, _) = self.pending.front()?;
        if retired_at.saturating_add(depth) <= now {
            self.pending.pop_front().map(|(_, item)| item)
        } else {
            None
        }
    }

    /// Drop every item whose window has closed by tick `now`.
    pub fn collect(&mut self, now: u64, depth: u64) {
        while self.pop_due(now, depth).is_some() {}
    }

    /// Everything still parked, regardless of its window. Only for a caller that
    /// has already idled the device.
    pub fn drain(&mut self) -> impl Iterator<Item = T> + '_ {
        self.pending.drain(..).map(|(_, item)| item)
    }

    /// How many items are still parked.
    pub fn len(&self) -> usize {
        self.pending.len()
    }

    /// Whether nothing is parked.
    pub fn is_empty(&self) -> bool {
        self.pending.is_empty()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use alloc::vec::Vec;

    #[test]
    fn an_item_is_held_for_depth_ticks() {
        let mut pool = RetirePool::new();
        pool.push(10, 'a');
        for now in 10..13 {
            pool.collect(now, 3);
            assert_eq!(pool.len(), 1, "tick {now} is inside the window");
        }
        pool.collect(13, 3);
        assert!(pool.is_empty());
    }

    #[test]
    fn depth_one_frees_on_the_next_tick() {
        let mut pool = RetirePool::new();
        pool.push(4, 'a');
        assert_eq!(pool.pop_due(4, 1), None);
        assert_eq!(pool.pop_due(5, 1), Some('a'));
    }

    #[test]
    fn items_come_back_oldest_first_and_stop_at_the_first_live_one() {
        let mut pool = RetirePool::new();
        pool.push(1, 'a');
        pool.push(1, 'b');
        pool.push(2, 'c');
        let mut due = Vec::new();
        while let Some(item) = pool.pop_due(3, 2) {
            due.push(item);
        }
        assert_eq!(due, ['a', 'b']);
        assert_eq!(pool.len(), 1);
    }

    #[test]
    fn collect_is_idempotent_and_safe_when_empty() {
        let mut pool: RetirePool<u32> = RetirePool::new();
        pool.collect(100, 3);
        pool.push(0, 7);
        pool.collect(3, 3);
        pool.collect(3, 3);
        assert!(pool.is_empty());
    }

    #[test]
    fn drain_hands_back_everything() {
        let mut pool = RetirePool::new();
        pool.push(0, 1);
        pool.push(9, 2);
        assert_eq!(pool.drain().collect::<Vec<_>>(), [1, 2]);
        assert!(pool.is_empty());
    }
}
