//! How a caller runs the independent rows an environment-map convolution
//! decomposes into.
//!
//! The rows share nothing: each reads only the immutable source and writes only
//! its own texels, so they can be worked through in any order and on any thread.
//! This crate owns no thread pool, so the schedule is the caller's to supply.
//! `concinnity_host::thread` owns the engine's pool; a caller without one uses
//! [`Serial`].

/// Runs a set of independent work items to completion, in any order.
///
/// The method is generic rather than object-safe so one scheduler serves every
/// item type; pass an implementor by reference.
pub trait RowScheduler {
    /// Apply `compute` to every item in `items`, then return.
    fn run<T: Send>(&self, items: &mut [T], compute: &(dyn Fn(&mut T) + Send + Sync));
}

/// Runs every item on the calling thread, in order.
pub struct Serial;

impl RowScheduler for Serial {
    fn run<T: Send>(&self, items: &mut [T], compute: &(dyn Fn(&mut T) + Send + Sync)) {
        items.iter_mut().for_each(compute);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use alloc::vec::Vec;

    #[test]
    fn serial_touches_every_item_once() {
        let mut items: Vec<u32> = (0..8).collect();
        Serial.run(&mut items, &|v| *v *= 2);
        assert_eq!(items, (0..8).map(|v| v * 2).collect::<Vec<_>>());
    }

    #[test]
    fn serial_over_an_empty_set_is_a_no_op() {
        let mut items: Vec<u32> = Vec::new();
        Serial.run(&mut items, &|v| *v += 1);
        assert!(items.is_empty());
    }
}
