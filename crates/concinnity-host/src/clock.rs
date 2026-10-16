//! The process-wide monotonic clock that times are compared on across crates
//! and threads: an input sample taken on one thread and a GPU completion
//! observed on another land on the same axis.

use std::sync::OnceLock;
use std::time::Instant;

/// Microseconds since the first call in this process. Only differences mean
/// anything.
pub fn monotonic_micros() -> u64 {
    micros_at(Instant::now())
}

/// `at` on the [`monotonic_micros`] axis; an instant before the first call
/// reads as zero.
pub fn micros_at(at: Instant) -> u64 {
    static EPOCH: OnceLock<Instant> = OnceLock::new();
    let epoch = *EPOCH.get_or_init(Instant::now);
    at.saturating_duration_since(epoch).as_micros() as u64
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_clock_never_goes_backwards() {
        let first = monotonic_micros();
        assert!(monotonic_micros() >= first);
    }

    #[test]
    fn an_instant_maps_onto_the_same_axis() {
        let before = monotonic_micros();
        let at = Instant::now();
        let after = monotonic_micros();
        let mapped = micros_at(at);
        assert!(
            (before..=after).contains(&mapped),
            "{before} <= {mapped} <= {after}"
        );
    }
}
