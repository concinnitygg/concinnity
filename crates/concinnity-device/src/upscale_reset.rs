//! Consuming an upscaler's history reset, the same on every backend.

use concinnity_core::render::history_reset::{ConsumedReset, UpscalerResetLatch};

/// Whether this dispatch discards the upscaler's history. A reset a frame
/// requested logs one line, which is what tells a run the reset reached the
/// upscaler; a creation or rebuild reset is silent.
pub(crate) fn consume(latch: &UpscalerResetLatch) -> bool {
    let consumed = latch.take();
    if consumed == ConsumedReset::Requested {
        tracing::info!("temporal upscaling: history reset");
    }
    consumed.discards()
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Arc;
    use std::sync::atomic::{AtomicUsize, Ordering};

    // Runs `consume` once and counts the info events it emits. A live
    // subscriber is required: tracing skips the call when nothing listens.
    fn consume_logged(latch: &UpscalerResetLatch) -> (bool, usize) {
        struct Counter(Arc<AtomicUsize>);
        impl tracing::Subscriber for Counter {
            fn enabled(&self, meta: &tracing::Metadata<'_>) -> bool {
                *meta.level() == tracing::Level::INFO
            }
            fn new_span(&self, _: &tracing::span::Attributes<'_>) -> tracing::span::Id {
                tracing::span::Id::from_u64(1)
            }
            fn record(&self, _: &tracing::span::Id, _: &tracing::span::Record<'_>) {}
            fn record_follows_from(&self, _: &tracing::span::Id, _: &tracing::span::Id) {}
            fn event(&self, _: &tracing::Event<'_>) {
                self.0.fetch_add(1, Ordering::Relaxed);
            }
            fn enter(&self, _: &tracing::span::Id) {}
            fn exit(&self, _: &tracing::span::Id) {}
        }
        let count = Arc::new(AtomicUsize::new(0));
        let discards = tracing::subscriber::with_default(Counter(count.clone()), || consume(latch));
        (discards, count.load(Ordering::Relaxed))
    }

    #[test]
    fn only_a_requested_reset_logs() {
        let latch = UpscalerResetLatch::default();
        assert_eq!(consume_logged(&latch), (true, 0), "creation is silent");
        assert_eq!(consume_logged(&latch), (false, 0), "nothing pending");
        latch.request();
        assert_eq!(consume_logged(&latch), (true, 1), "a request logs once");
        latch.rebuilt();
        assert_eq!(consume_logged(&latch), (true, 0), "a rebuild is silent");
    }
}
