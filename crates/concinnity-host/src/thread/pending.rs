//! A count of work handed to another thread and not yet finished, which a
//! caller can wait to drain.

use std::sync::{Condvar, Mutex, MutexGuard, PoisonError};

/// Work queued and not yet finished, so a caller can wait it out.
#[derive(Debug, Default)]
pub struct Pending {
    count: Mutex<usize>,
    drained: Condvar,
}

impl Pending {
    /// One more piece of work is queued.
    pub fn add(&self) {
        *self.count() += 1;
    }

    /// One piece of work finished, waking every waiter once none remain.
    pub fn done(&self) {
        let mut count = self.count();
        *count = count.saturating_sub(1);
        if *count == 0 {
            self.drained.notify_all();
        }
    }

    /// Block until every queued piece of work has finished.
    pub fn wait(&self) {
        let mut count = self.count();
        while *count > 0 {
            count = self
                .drained
                .wait(count)
                .unwrap_or_else(PoisonError::into_inner);
        }
    }

    // A panic elsewhere cannot leave the count unusable: it is only ever
    // incremented or decremented whole.
    fn count(&self) -> MutexGuard<'_, usize> {
        self.count.lock().unwrap_or_else(PoisonError::into_inner)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Arc;

    #[test]
    fn nothing_queued_waits_for_nothing() {
        Pending::default().wait();
    }

    #[test]
    fn a_wait_returns_once_every_piece_is_done() {
        let pending = Arc::new(Pending::default());
        pending.add();
        pending.add();
        let worker = Arc::clone(&pending);
        let finished = std::thread::spawn(move || {
            worker.done();
            worker.done();
        });
        pending.wait();
        finished.join().unwrap();
        assert_eq!(*pending.count(), 0);
    }

    // A stray finish never takes the count below zero, so a later add still
    // holds a waiter.
    #[test]
    fn an_unmatched_done_leaves_the_count_at_zero() {
        let pending = Pending::default();
        pending.done();
        assert_eq!(*pending.count(), 0);
        pending.add();
        assert_eq!(*pending.count(), 1);
    }
}
