//! The monotonic clock the frame loop times systems with, published as a
//! resource by the host that has one.
//!
//! Reading a wall clock needs an operating system, so the loop names a function
//! pointer instead of a platform type. A world running without one (a headless
//! or embedded host) records zero micros per system rather than losing the
//! profile entirely.

/// A monotonic microsecond source, installed as a world resource by the host.
///
/// `World::step` reads it once per tick and brackets each system with it. Only
/// differences are used, so the epoch is the host's to choose; it must be
/// monotonic within a process.
pub struct Clock(pub fn() -> u64);

impl core::fmt::Debug for Clock {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("Clock").finish_non_exhaustive()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ecs::World;

    fn ticking() -> u64 {
        7
    }

    // The clock is an ordinary resource: the host installs it, the loop reads
    // it back through the same map every other protocol resource uses.
    #[test]
    fn a_installed_clock_reads_back() {
        let mut world = World::new();
        assert!(world.resource::<Clock>().is_none());
        world.insert_resource(Clock(ticking));
        assert_eq!((world.resource::<Clock>().expect("installed").0)(), 7);
    }
}
