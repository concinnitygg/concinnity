// The cache segment's metadata: an index of regenerable artifacts, addressed by
// producer and key rather than by filename. A segment is one container holding
// every entry, so the index is what keeps two producers -- and two adapters of
// one producer -- out of each other's bytes.

use alloc::string::String;
use alloc::vec::Vec;

use serde::{Deserialize, Serialize};

use crate::blob::kind::BlobKind;

/// The four magic bytes a cache segment starts with, the [`BlobKind`] magic of
/// [`CacheMeta`].
pub const CACHE_MAGIC: [u8; 4] = *b"CNC\0";

/// The validity token a cache segment's header carries: the layout version of
/// [`CacheMeta`] and of the payload addressing its entries use. postcard cannot
/// tell a layout change from valid bytes, so a segment stamped with another
/// value is regenerated whole rather than decoded.
pub const CACHE_SEGMENT_VERSION: u32 = 2;

/// Which producer an entry belongs to.
///
/// The discriminant is part of the segment bytes and
/// [`CACHE_SEGMENT_VERSION`] guards them, so a new producer takes the next
/// value rather than reshaping the index.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum CacheEntryKind {
    /// A driver pipeline blob: a serialized `VkPipelineCache` or a D3D12
    /// pipeline library, machine code for the one adapter its key names.
    Pipeline,
    /// A compiled shader binary (backend IR: DXBC, SPIR-V, a metallib), keyed
    /// by a digest of everything the compile was a function of.
    Shader,
    /// A compiled asset payload, keyed by a digest of the args and source
    /// files the compile read.
    Payload,
    /// The asset entries a scene import expands to, keyed by a digest of the
    /// source file and the import options.
    Expansion,
    /// A baked asset preview: a PNG keyed by a digest of what it depicts, plus
    /// the one entry holding the asset-name-to-key map over the whole set.
    Thumbnail,
}

/// One cached artifact: which producer owns it, what it is valid for, and where
/// its bytes sit in the payload section.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct CacheEntry {
    /// The producer that wrote the entry.
    pub kind: CacheEntryKind,
    /// The producer's key, which doubles as the entry's validity: a lookup
    /// naming another key misses. A pipeline blob keys on the adapter it is
    /// machine code for; a producer whose artifacts are valid anywhere keys on
    /// content alone.
    pub key: String,
    /// Byte offset of the entry's bytes within the payload section.
    pub offset: u64,
    /// Byte length of the entry's bytes.
    pub len: u64,
}

/// A cache segment's metadata block: the index of everything its payload
/// section holds.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct CacheMeta {
    /// Names the host shader toolchain that produced the segment's entries,
    /// empty when nothing stamped it. An entry is a function of its source
    /// rather than of what compiled it, so an external compiler upgrade moves
    /// no key: a segment naming another toolchain is discarded instead.
    pub toolchain: String,
    /// The segment's entries, in payload order.
    pub entries: Vec<CacheEntry>,
}

impl CacheMeta {
    /// The entry `kind` stored under `key`, if the segment holds one.
    pub fn find(&self, kind: CacheEntryKind, key: &str) -> Option<&CacheEntry> {
        self.entries.iter().find(|e| e.kind == kind && e.key == key)
    }
}

impl BlobKind for CacheMeta {
    const MAGIC: [u8; 4] = CACHE_MAGIC;
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::blob::schema::BlobMeta;
    use crate::blob::{BlobError, encode_cnb, parse_cnb};
    use alloc::vec;

    fn entry(kind: CacheEntryKind, key: &str, offset: u64, len: u64) -> CacheEntry {
        CacheEntry {
            kind,
            key: String::from(key),
            offset,
            len,
        }
    }

    #[test]
    fn a_segment_round_trips_its_index_and_payload() {
        let meta = CacheMeta {
            toolchain: String::from("slang 2026.1"),
            entries: vec![
                entry(CacheEntryKind::Pipeline, "vk-aa", 0, 3),
                entry(CacheEntryKind::Shader, "deadbeef", 3, 2),
            ],
        };
        let image = encode_cnb(CACHE_SEGMENT_VERSION, &meta, &[1, 2, 3, 4, 5]).unwrap();
        let (got, payload_start) =
            parse_cnb::<CacheMeta>(CACHE_SEGMENT_VERSION, &image).expect("parse");
        assert_eq!(got, meta);
        assert_eq!(&image[payload_start..], &[1, 2, 3, 4, 5]);
    }

    // Two kinds of container must never be read as each other, which is the
    // whole reason the magic hangs off the meta type.
    #[test]
    fn a_world_blob_is_not_a_cache_segment() {
        assert_ne!(CacheMeta::MAGIC, BlobMeta::MAGIC);
        let world = encode_cnb(CACHE_SEGMENT_VERSION, &BlobMeta::default(), &[]).unwrap();
        assert_eq!(
            parse_cnb::<CacheMeta>(CACHE_SEGMENT_VERSION, &world),
            Err(BlobError::BadMagic)
        );
    }

    // A segment written under another index layout is not decoded into a
    // plausible-looking index; it is rejected so the caller regenerates it.
    #[test]
    fn a_segment_of_another_version_is_rejected() {
        let image = encode_cnb(CACHE_SEGMENT_VERSION - 1, &CacheMeta::default(), &[]).unwrap();
        assert_eq!(
            parse_cnb::<CacheMeta>(CACHE_SEGMENT_VERSION, &image),
            Err(BlobError::ValidityMismatch(CACHE_SEGMENT_VERSION - 1))
        );
    }

    // The build segment holds its own kinds in the same index, which is what
    // keeps a payload and an expansion that hash alike out of each other's
    // bytes now that no source hash separates their key spaces.
    #[test]
    fn a_kind_separates_two_entries_sharing_one_key() {
        let meta = CacheMeta {
            toolchain: String::new(),
            entries: vec![
                entry(CacheEntryKind::Payload, "cafe", 0, 1),
                entry(CacheEntryKind::Expansion, "cafe", 1, 1),
            ],
        };
        assert_eq!(
            meta.find(CacheEntryKind::Payload, "cafe"),
            Some(&meta.entries[0])
        );
        assert_eq!(
            meta.find(CacheEntryKind::Expansion, "cafe"),
            Some(&meta.entries[1])
        );
    }

    #[test]
    fn lookup_matches_on_both_kind_and_key() {
        let meta = CacheMeta {
            toolchain: String::new(),
            entries: vec![entry(CacheEntryKind::Pipeline, "vk-aa", 0, 3)],
        };
        assert_eq!(
            meta.find(CacheEntryKind::Pipeline, "vk-aa"),
            Some(&meta.entries[0])
        );
        assert_eq!(meta.find(CacheEntryKind::Pipeline, "vk-bb"), None);
        // Two producers share one index, so a key must not match across kinds.
        assert_eq!(meta.find(CacheEntryKind::Shader, "vk-aa"), None);
    }
}
