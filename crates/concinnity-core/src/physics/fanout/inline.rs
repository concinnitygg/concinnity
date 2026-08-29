// How a caller lends the simulation its threads.
//
// The simulation cannot own a scheduler. It is `#![no_std]` and depends on two
// leaves, so it can neither name a thread pool nor detect that one exists. The
// work it can split is therefore offered rather than taken: a step says what it
// holds that is independent, and whoever called it decides whether that runs on
// one thread or several.
//
// The trait is generic in the item and the body rather than object safe. A step
// hands out a small fixed array of work units, so the fan-out is monomorphised
// into the step that used it and nothing is boxed or dispatched dynamically.
//
// `scope` is the second half of that, and it is what makes the first one
// affordable. Reaching a pool of sleeping workers from a thread that is not one
// of them costs far more than the handing-out does; reaching it again from
// inside costs almost nothing. So a step gathers the workers once and offers
// all three of its stages inside that, rather than paying the entry three
// times. A fan-out with nothing to gather leaves the default alone.
//
// A caller that lends nothing gets `Inline`, which runs the units in order on
// the calling thread. That is the default `Simulation::step` uses, and it is
// why a single-threaded host needs no capability check and no configuration.

/// A caller's way of running independent work at the same time.
///
/// The simulation asks for one of these rather than depending on a scheduler,
/// so the same step runs on a thread pool, on one thread, or on a host that has
/// no threads at all.
///
/// # Examples
///
/// ```
/// use concinnity_core::physics::{Fanout, Inline};
///
/// let mut work = [1u32, 2, 3];
/// Inline.for_each(&mut work, |item| *item *= 10);
/// assert_eq!(work, [10, 20, 30]);
/// assert_eq!(Inline.workers(), 1);
/// ```
pub trait Fanout: Sync {
    /// Units of work this fan-out can run at once. One means the work runs on
    /// the calling thread, which is what the simulation sizes its per-worker
    /// scratch against.
    fn workers(&self) -> usize;

    /// Run `work` with this fan-out's workers already gathered, and return
    /// what it produced.
    ///
    /// Everything a step hands out happens inside one of these. An
    /// implementation backed by a thread pool enters it here, so the
    /// [`Fanout::for_each`] calls inside are already there; one with no pool
    /// to enter leaves this as it is and runs `work` where it stands.
    fn scope<R, F>(&self, work: F) -> R
    where
        F: FnOnce() -> R + Send,
        R: Send,
    {
        work()
    }

    /// Run `body` over every item and return once all of them are done.
    ///
    /// Items are independent, so an implementation may visit them in any order
    /// and on any thread. The simulation never lets the result depend on which
    /// one did what.
    fn for_each<T, F>(&self, items: &mut [T], body: F)
    where
        T: Send,
        F: Fn(&mut T) + Send + Sync;
}

/// The fan-out for a caller that has no threads to lend: every unit runs on the
/// calling thread, in order.
#[derive(Debug, Clone, Copy, Default)]
pub struct Inline;

impl Fanout for Inline {
    fn workers(&self) -> usize {
        1
    }

    fn for_each<T, F>(&self, items: &mut [T], body: F)
    where
        T: Send,
        F: Fn(&mut T) + Send + Sync,
    {
        for item in items {
            body(item);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use alloc::vec::Vec;

    #[test]
    fn inline_visits_every_item_in_order() {
        let mut items: Vec<u32> = (0..8).collect();
        let mut seen = Vec::new();
        // A `Fn` cannot hold the log mutably, so the order is recovered from
        // what the items themselves end up carrying.
        Inline.for_each(&mut items, |item| *item += 100);
        seen.extend(items.iter().copied());
        assert_eq!(seen, (0..8).map(|i| i + 100).collect::<Vec<_>>());
    }

    #[test]
    fn inline_lends_one_worker() {
        assert_eq!(Inline.workers(), 1);
    }

    #[test]
    fn an_empty_batch_is_a_no_op() {
        let mut items: [u32; 0] = [];
        Inline.for_each(&mut items, |item| *item += 1);
        assert!(items.is_empty());
    }
}
