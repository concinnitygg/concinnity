//! The main-pass pipelines of a world's material-referenced shaders, one slot
//! per shader bucket past the default, generic over a backend's pipeline type.
//!
//! Bucket 0 is the world's default program, drawn under the main pass's own
//! pipeline, so it is always resident and has no slot here. Bucket `b` reads
//! slot `b - 1`, which stays empty while the scene owning its Shader has not
//! pinned.

use alloc::format;
use alloc::string::ToString;
use alloc::vec::Vec;

use crate::render::error::{RenderError, RenderResult};

/// One optional pipeline per shader bucket past the default.
#[derive(Debug)]
pub struct WorldPipelines<P> {
    slots: Vec<Option<P>>,
}

impl<P> Default for WorldPipelines<P> {
    fn default() -> Self {
        Self { slots: Vec::new() }
    }
}

impl<P> WorldPipelines<P> {
    /// A table over `slots`, slot `i` holding bucket `i + 1`.
    pub fn new(slots: Vec<Option<P>>) -> Self {
        Self { slots }
    }

    /// Shader buckets, the default included.
    pub fn bucket_count(&self) -> usize {
        1 + self.slots.len()
    }

    /// The slot holding `bucket`'s pipeline. A bucket outside the table is a
    /// scene-authoring mistake, not a device failure, so it errors as
    /// [`RenderError::Other`].
    pub fn slot(&self, bucket: u32) -> RenderResult<usize> {
        let slot = (bucket as usize).checked_sub(1).ok_or_else(|| {
            RenderError::Other("shader bucket 0 is the world default program".to_string())
        })?;
        if slot >= self.slots.len() {
            return Err(RenderError::Other(format!(
                "shader bucket {bucket} is past the world's {} shader pipeline(s)",
                self.slots.len()
            )));
        }
        Ok(slot)
    }

    /// Whether `bucket`'s draws can render: the default always can, every
    /// other bucket needs its pipeline installed.
    pub fn resident(&self, bucket: usize) -> bool {
        bucket == 0 || self.get(bucket).is_some()
    }

    /// `bucket`'s installed pipeline.
    pub fn get(&self, bucket: usize) -> Option<&P> {
        self.slots.get(bucket.checked_sub(1)?)?.as_ref()
    }

    /// Install `pipeline` for `bucket`, handing back the one it displaces.
    pub fn install(&mut self, bucket: u32, pipeline: P) -> RenderResult<Option<P>> {
        let slot = self.slot(bucket)?;
        Ok(self.slots[slot].replace(pipeline))
    }

    /// Empty `bucket`'s slot, handing back what it held. A bucket outside the
    /// table holds nothing.
    pub fn evict(&mut self, bucket: u32) -> Option<P> {
        let slot = self.slot(bucket).ok()?;
        self.slots[slot].take()
    }

    /// Every bucket past the default whose pipeline is installed, ascending.
    pub fn resident_buckets(&self) -> impl Iterator<Item = usize> + '_ {
        self.slots
            .iter()
            .enumerate()
            .filter(|(_, p)| p.is_some())
            .map(|(slot, _)| slot + 1)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use alloc::vec;

    // Bucket 1 installed, bucket 2 deferred, bucket 3 installed.
    fn table() -> WorldPipelines<&'static str> {
        WorldPipelines::new(vec![Some("one"), None, Some("three")])
    }

    #[test]
    fn buckets_count_the_default() {
        assert_eq!(table().bucket_count(), 4);
        assert_eq!(WorldPipelines::<()>::default().bucket_count(), 1);
    }

    #[test]
    fn a_bucket_reads_the_slot_before_it() {
        let t = table();
        assert_eq!(t.slot(1).unwrap(), 0);
        assert_eq!(t.slot(3).unwrap(), 2);
        assert_eq!(t.get(3), Some(&"three"));
    }

    #[test]
    fn the_default_and_buckets_past_the_table_have_no_slot() {
        let t = table();
        for bucket in [0, 4, u32::MAX] {
            assert!(
                matches!(t.slot(bucket), Err(RenderError::Other(_))),
                "{bucket}"
            );
        }
        assert_eq!(t.get(0), None);
        assert_eq!(t.get(9), None);
    }

    #[test]
    fn the_default_is_always_resident_and_a_deferred_bucket_is_not() {
        let t = table();
        assert!(t.resident(0));
        assert!(t.resident(1));
        assert!(!t.resident(2));
        assert!(!t.resident(4));
        assert_eq!(t.resident_buckets().collect::<Vec<_>>(), [1, 3]);
    }

    #[test]
    fn install_and_evict_hand_back_what_they_replace() {
        let mut t = table();
        assert_eq!(t.install(2, "two").unwrap(), None);
        assert_eq!(t.install(1, "uno").unwrap(), Some("one"));
        assert_eq!(t.resident_buckets().collect::<Vec<_>>(), [1, 2, 3]);
        assert_eq!(t.evict(3), Some("three"));
        assert_eq!(t.evict(3), None);
        assert_eq!(t.evict(0), None);
        assert_eq!(t.evict(7), None);
        assert!(t.install(0, "zero").is_err());
        assert_eq!(t.resident_buckets().collect::<Vec<_>>(), [1, 2]);
    }
}
