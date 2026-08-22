//! Cooperative shutdown signal shared across threads. Clones observe the same
//! flag: any clone's `cancel()` is visible to every other clone's
//! `is_cancelled()`. Cancellation is one-way and sticky.

use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};

/// Cloneable one-shot cancellation flag. The run loop polls `is_cancelled`
/// each tick; signal handlers and debug servers call `cancel` to stop it.
#[derive(Debug, Clone, Default)]
pub struct ShutdownToken(Arc<AtomicBool>);

impl ShutdownToken {
    /// A fresh, uncancelled token.
    pub fn new() -> Self {
        Self::default()
    }

    /// Signal shutdown to every clone of this token.
    pub fn cancel(&self) {
        self.0.store(true, Ordering::Release);
    }

    /// Whether any clone has signalled shutdown.
    pub fn is_cancelled(&self) -> bool {
        self.0.load(Ordering::Acquire)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn starts_uncancelled_and_sticks_once_cancelled() {
        let token = ShutdownToken::new();
        assert!(!token.is_cancelled());
        token.cancel();
        assert!(token.is_cancelled());
        token.cancel();
        assert!(token.is_cancelled());
    }

    #[test]
    fn clones_share_the_flag() {
        let token = ShutdownToken::new();
        let clone = token.clone();
        clone.cancel();
        assert!(token.is_cancelled());
    }

    #[test]
    fn cancel_crosses_threads() {
        let token = ShutdownToken::new();
        let clone = token.clone();
        std::thread::spawn(move || clone.cancel())
            .join()
            .expect("cancel thread panicked");
        assert!(token.is_cancelled());
    }
}
