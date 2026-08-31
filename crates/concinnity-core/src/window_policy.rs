//! Whether this process is allowed to open a window.
//!
//! A backend stands up its window from deep inside device initialisation, and
//! a process that reaches that point without an operating system to drive it
//! does not fail: it blocks on an event loop that never ends. Under a test
//! harness that is a hang with no failing assertion to read.
//!
//! The policy is a process-wide latch a caller sets before it runs anything.
//! Default is permissive, so a shipped binary behaves exactly as it did; a
//! headless host forbids windows up front and gets a panic naming the backend
//! instead of a hang.

use core::sync::atomic::{AtomicBool, Ordering};

static FORBIDDEN: AtomicBool = AtomicBool::new(false);

/// Forbid window creation for the rest of the process, until [`allow_windows`].
///
/// A backend that reaches its window call after this panics rather than
/// standing one up.
pub fn forbid_windows() {
    FORBIDDEN.store(true, Ordering::SeqCst);
}

/// Lift the ban set by [`forbid_windows`], restoring the default.
pub fn allow_windows() {
    FORBIDDEN.store(false, Ordering::SeqCst);
}

/// Whether window creation is currently forbidden.
pub fn windows_forbidden() -> bool {
    FORBIDDEN.load(Ordering::SeqCst)
}

/// Panic if windows are forbidden, naming `backend` as what tried to open one.
///
/// Called at each backend's window entry point, before any operating-system
/// resource is taken.
pub fn assert_windows_allowed(backend: &str) {
    assert!(
        !windows_forbidden(),
        "{backend} tried to open a window in a process that forbids them; \
         run the world on the headless loop instead"
    );
}

#[cfg(test)]
mod tests {
    use alloc::string::String;

    use super::*;

    // The latch is process-global, so the two tests that move it share a lock
    // and put it back. Everything else in the crate reads the default.
    static LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());

    #[test]
    fn the_default_permits_a_window() {
        let _guard = LOCK.lock().unwrap_or_else(|e| e.into_inner());
        allow_windows();
        assert!(!windows_forbidden());
        assert_windows_allowed("a backend");
    }

    #[test]
    fn a_forbidden_process_panics_naming_the_backend() {
        let _guard = LOCK.lock().unwrap_or_else(|e| e.into_inner());
        forbid_windows();
        assert!(windows_forbidden());

        let panicked = std::panic::catch_unwind(|| assert_windows_allowed("TestBackend"));
        allow_windows();

        let payload = panicked.expect_err("a forbidden window creation panics");
        let msg = payload
            .downcast_ref::<String>()
            .expect("the panic carries its message");
        assert!(msg.contains("TestBackend"), "{msg}");
        assert!(msg.contains("headless"), "{msg}");
    }
}
