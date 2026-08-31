//! The one lock over this test binary's process-global state.
//!
//! Cargo runs a binary's tests on parallel threads, so anything process-wide is
//! shared by every test running at that moment. Most tests only *read* that
//! state, and readers do not conflict with each other -- serialising them all
//! would cost the suite its parallelism for nothing.
//!
//! So the lock is a reader/writer one. A test that reads a global takes
//! [`shared`] and still runs beside every other reader; a test that writes one
//! takes [`exclusive`] and runs alone. This generalises the discipline the
//! engine's development flags already used, so that every crate reaches the
//! same lock instead of each keeping its own.
//!
//! Neither guard is reentrant: a test that takes one must not take the other,
//! and must not take the same one twice. `tests/global_state_discipline.rs` in
//! the root package is what checks that.

use std::sync::{RwLock, RwLockReadGuard, RwLockWriteGuard};

// One per test binary. Cargo gives each crate's tests their own process, so
// this never spans crates: it serialises the tests of one binary against each
// other, which is the only place they can collide.
static ACCESS: RwLock<()> = RwLock::new(());

/// Permission to read process-global state, shared with other readers.
///
/// Hold it for as long as the read matters, not just across the read itself:
/// a value read and then acted on is still a read until the action is done.
pub struct SharedAccess {
    _guard: RwLockReadGuard<'static, ()>,
}

/// Permission to write process-global state, held alone.
pub struct ExclusiveAccess {
    _guard: RwLockWriteGuard<'static, ()>,
}

/// Take shared access, for a test whose code path reads a global.
///
/// Poison is ignored: the test that panicked holding this has already failed,
/// and erroring every later lock buries that failure under a cascade.
pub fn shared() -> SharedAccess {
    SharedAccess {
        _guard: ACCESS.read().unwrap_or_else(|e| e.into_inner()),
    }
}

/// Take exclusive access, for a test that writes a global.
///
/// Poison is ignored, for the reason [`shared`] gives.
pub fn exclusive() -> ExclusiveAccess {
    ExclusiveAccess {
        _guard: ACCESS.write().unwrap_or_else(|e| e.into_inner()),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::thread;

    #[test]
    fn readers_do_not_block_each_other() {
        let first = shared();
        let second = shared();
        drop((first, second));
    }

    #[test]
    fn a_writer_waits_for_the_readers_to_finish() {
        static SEEN: AtomicUsize = AtomicUsize::new(0);

        let reader = shared();
        let writer = thread::spawn(|| {
            let _w = exclusive();
            SEEN.store(1, Ordering::SeqCst);
        });

        // The writer cannot have run while the read guard is alive. This is a
        // liveness check, not a timing one: the reader is dropped immediately
        // after, so a failure here means the lock is not exclusive at all.
        assert_eq!(
            SEEN.load(Ordering::SeqCst),
            0,
            "a writer ran beside a reader"
        );
        drop(reader);

        writer
            .join()
            .expect("the writer finishes once the reader drops");
        assert_eq!(SEEN.load(Ordering::SeqCst), 1);
    }

    #[test]
    fn a_poisoned_lock_is_still_taken() {
        let panicked = std::panic::catch_unwind(|| {
            let _guard = exclusive();
            panic!("poison it");
        });
        assert!(panicked.is_err());

        // The point: neither of these deadlocks or unwraps on the poison.
        drop(exclusive());
        drop(shared());
    }
}
