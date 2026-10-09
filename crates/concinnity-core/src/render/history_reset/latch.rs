// The per-upscaler record of whether its next dispatch starts its history
// over.

use core::sync::atomic::{AtomicU8, Ordering};

/// Whether a temporal upscaler discards its history on its next dispatch:
/// pending from creation until the first dispatch, and again after each
/// [`request`](Self::request). A request made while the creation reset is
/// still pending folds into it. Atomic, so a backend may request on one thread
/// and dispatch on another.
#[derive(Debug)]
pub struct UpscalerResetLatch(AtomicU8);

const PENDING_NONE: u8 = 0;
const PENDING_CREATED: u8 = 1;
const PENDING_REQUESTED: u8 = 2;

/// What one dispatch found pending.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ConsumedReset {
    /// Nothing: the history carries on.
    None,
    /// The upscaler was just created or rebuilt.
    Created,
    /// A frame asked for its history to be dropped.
    Requested,
}

impl ConsumedReset {
    /// Whether this dispatch discards the history.
    pub const fn discards(self) -> bool {
        !matches!(self, ConsumedReset::None)
    }
}

impl Default for UpscalerResetLatch {
    fn default() -> Self {
        Self(AtomicU8::new(PENDING_CREATED))
    }
}

impl UpscalerResetLatch {
    /// Discard the history on the next dispatch.
    pub fn request(&self) {
        let _ = self.0.compare_exchange(
            PENDING_NONE,
            PENDING_REQUESTED,
            Ordering::AcqRel,
            Ordering::Acquire,
        );
    }

    /// The upscaler was rebuilt, so its next dispatch starts over as if new.
    pub fn rebuilt(&self) {
        self.0.store(PENDING_CREATED, Ordering::Release);
    }

    /// What this dispatch finds pending, clearing it.
    pub fn take(&self) -> ConsumedReset {
        match self.0.swap(PENDING_NONE, Ordering::AcqRel) {
            PENDING_CREATED => ConsumedReset::Created,
            PENDING_REQUESTED => ConsumedReset::Requested,
            _ => ConsumedReset::None,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_latch_resets_on_creation_and_on_each_request() {
        let latch = UpscalerResetLatch::default();
        assert_eq!(latch.take(), ConsumedReset::Created);
        assert_eq!(latch.take(), ConsumedReset::None);
        latch.request();
        latch.request();
        assert_eq!(latch.take(), ConsumedReset::Requested);
        assert!(!latch.take().discards());
    }

    // A request while the creation reset is pending, or after a rebuild, folds
    // into it; a rebuild overrides a pending request.
    #[test]
    fn creation_absorbs_a_pending_request() {
        let latch = UpscalerResetLatch::default();
        latch.request();
        assert_eq!(latch.take(), ConsumedReset::Created);
        latch.request();
        latch.rebuilt();
        assert_eq!(latch.take(), ConsumedReset::Created);
        latch.rebuilt();
        latch.request();
        assert_eq!(latch.take(), ConsumedReset::Created);
    }

    #[test]
    fn every_consumed_reset_but_none_discards() {
        assert!(ConsumedReset::Requested.discards());
        assert!(ConsumedReset::Created.discards());
        assert!(!ConsumedReset::None.discards());
    }
}
