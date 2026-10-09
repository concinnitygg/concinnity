// Behavior reports, written to the log.

use concinnity_core::behavior::BehaviorReporter;

#[derive(Debug)]
pub(crate) struct Log;

impl BehaviorReporter for Log {
    fn warn(&self, message: &str) {
        tracing::warn!("behavior: {message}");
    }
}
