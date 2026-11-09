//! Built-in shader hot reload as a backend tracks it: whether built-in shader
//! sources resolve disk-first, the flag a reload request raises, and the
//! template generation that keeps a pipeline built from older templates from
//! being installed after a reload.

use alloc::sync::Arc;
use core::sync::atomic::{AtomicBool, Ordering};

/// A backend's shader hot-reload state.
pub struct ShaderReload {
    /// True when built-in shader sources resolve from disk first.
    pub enabled: bool,
    /// Bumped by every engine-template reload, so a world Shader pipeline a
    /// worker built from the templates before it is never installed after it.
    pub generation: u64,
    pending: Option<Arc<AtomicBool>>,
}

impl ShaderReload {
    /// Reload state for a backend, with a request flag only when `enabled`.
    pub fn new(enabled: bool) -> Self {
        Self {
            enabled,
            generation: 0,
            pending: enabled.then(|| Arc::new(AtomicBool::new(false))),
        }
    }

    /// The flag a reload request sets, shared with whoever raises it. `None`
    /// when reload is off.
    pub fn flag(&self) -> Option<Arc<AtomicBool>> {
        self.pending.as_ref().map(Arc::clone)
    }

    /// True when a reload has been requested. Always false when reload is
    /// off, so a shipped build never enters the reload branch.
    pub fn requested(&self) -> bool {
        self.pending
            .as_ref()
            .is_some_and(|f| f.load(Ordering::SeqCst))
    }

    /// Lower the request flag, whatever the reload's outcome, so a failed
    /// rebuild does not retry every frame.
    pub fn clear(&self) {
        if let Some(flag) = &self.pending {
            flag.store(false, Ordering::SeqCst);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_disabled_reload_has_no_flag_and_never_fires() {
        let reload = ShaderReload::new(false);
        assert!(reload.flag().is_none());
        assert!(!reload.requested());
    }

    #[test]
    fn a_raised_flag_is_requested_until_cleared() {
        let reload = ShaderReload::new(true);
        reload
            .flag()
            .expect("enabled")
            .store(true, Ordering::SeqCst);
        assert!(reload.requested());
        reload.clear();
        assert!(!reload.requested());
    }
}
