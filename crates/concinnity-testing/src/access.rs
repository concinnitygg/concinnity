//! The one lock over this test binary's process-global state.
//!
//! Cargo runs a binary's tests on parallel threads, so anything process-wide is
//! shared by every test running at that moment. A test that reaches such a
//! state takes [`exclusive`] and runs alone against every other holder in the
//! binary. This generalises the discipline the engine's development flags
//! already used, so that every crate reaches the same lock instead of each
//! keeping its own.
//!
//! A mutex rather than a reader/writer lock: `RwLock` blocks a new reader while
//! a writer is queued, so a thread taking the read guard twice waits on itself
//! and the test hangs rather than fails.
//!
//! The guard is not reentrant. A test that takes it must not take it again,
//! on its own thread or on one it spawns and waits for.
//! `tests/global_state_discipline.rs` in the root package is what checks that.

use std::sync::{Mutex, MutexGuard};

// One per test binary. Cargo gives each crate's tests their own process, so
// this never spans crates: it serialises the tests of one binary against each
// other, which is the only place they can collide.
static ACCESS: Mutex<()> = Mutex::new(());

/// Permission to reach process-global state, held alone.
///
/// Hold it for as long as the access matters, not just across the access
/// itself: a value read and then acted on is still in use until the action is
/// done.
pub struct ExclusiveAccess {
    _guard: MutexGuard<'static, ()>,
}

/// Take exclusive access, for a test that reaches a global.
///
/// Poison is ignored: the test that panicked holding this has already failed,
/// and erroring every later lock buries that failure under a cascade.
pub fn exclusive() -> ExclusiveAccess {
    ExclusiveAccess {
        _guard: ACCESS.lock().unwrap_or_else(|e| e.into_inner()),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_poisoned_lock_is_still_taken() {
        let panicked = std::panic::catch_unwind(|| {
            let _guard = exclusive();
            panic!("poison it");
        });
        assert!(panicked.is_err());

        // The point: this neither deadlocks nor unwraps on the poison.
        drop(exclusive());
    }
}
