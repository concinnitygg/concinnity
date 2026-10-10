//! What the renderer reads of a texture beside its compiled image.

use alloc::collections::BTreeSet;
use alloc::vec::Vec;

use super::TextureTable;
use crate::blob::{FrameError, decode_exact};

/// The runtime fields a texture's resource record carries beside its compiled
/// image, which the cook decides from how the world uses the texture.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct TextureRecord {
    /// The world reads the texture only at cook time, decoded into another
    /// asset's payload (a terrain's heightmap or a grass layer's density
    /// mask), so nothing samples it on the GPU and the renderer keeps it out of
    /// its texture pool.
    pub cook_only: bool,
}

impl TextureRecord {
    /// The record's bytes, as a resource record's `data_bytes` carries them.
    pub fn to_bytes(&self) -> Vec<u8> {
        postcard::to_allocvec(self).unwrap_or_default()
    }

    /// Decode a resource record's `data_bytes`. A record with no runtime
    /// fields holds the defaults.
    pub fn from_bytes(bytes: &[u8]) -> Result<Self, FrameError> {
        match bytes {
            [] => Ok(Self::default()),
            _ => decode_exact(bytes),
        }
    }
}

impl TextureTable {
    /// The runtime fields of the texture at `handle`; the defaults when there
    /// is no such texture.
    pub fn record(&self, handle: usize) -> Result<TextureRecord, FrameError> {
        self.0
            .get(handle)
            .map_or(Ok(TextureRecord::default()), |e| {
                TextureRecord::from_bytes(&e.data_bytes)
            })
    }

    /// The handles of every texture only the cook reads.
    pub fn cook_only_slots(&self) -> Result<BTreeSet<usize>, FrameError> {
        let mut slots = BTreeSet::new();
        for (slot, entry) in self.0.iter().enumerate() {
            if TextureRecord::from_bytes(&entry.data_bytes)?.cook_only {
                slots.insert(slot);
            }
        }
        Ok(slots)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::resource::ResourceEntry;
    use alloc::vec;

    fn entry(data_bytes: Vec<u8>) -> ResourceEntry {
        ResourceEntry {
            data_bytes,
            ..Default::default()
        }
    }

    #[test]
    fn a_record_round_trips_through_its_bytes() {
        for cook_only in [true, false] {
            let record = TextureRecord { cook_only };
            assert_eq!(TextureRecord::from_bytes(&record.to_bytes()), Ok(record));
        }
    }

    // A texture compiled before the record existed, or one the cook found a
    // GPU use for, carries no bytes and stays in the pool.
    #[test]
    fn an_empty_record_is_sampled_on_the_gpu() {
        assert_eq!(TextureRecord::from_bytes(&[]), Ok(TextureRecord::default()));
        assert!(!TextureRecord::default().cook_only);
    }

    #[test]
    fn the_table_lists_its_cook_only_slots() {
        let cook_only = TextureRecord { cook_only: true }.to_bytes();
        let table = TextureTable(vec![
            entry(Vec::new()),
            entry(cook_only.clone()),
            entry(TextureRecord::default().to_bytes()),
            entry(cook_only),
        ]);
        assert_eq!(table.cook_only_slots(), Ok(BTreeSet::from([1, 3])));
        assert!(table.record(1).unwrap().cook_only);
        assert!(!table.record(9).unwrap().cook_only);
    }

    #[test]
    fn a_malformed_record_is_an_error() {
        let table = TextureTable(vec![entry(vec![7, 7, 7])]);
        assert!(table.cook_only_slots().is_err());
    }
}
