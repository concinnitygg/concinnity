//! The .cnb blob container format.
//!
//! Layout of a blob binary:
//!
//!   `[ 4 bytes magic ][ 4 bytes schema version ][ 8 bytes meta_len ][ meta_bytes ][ payload_bytes ... ]`
//!
//! The header is fixed at 16 bytes. `meta_len` is the byte length of the
//! postcard-serialized `BlobMeta` that follows: the component defs stream and
//! the resource records stream. Everything after meta_len + meta_bytes is the
//! raw payload section, addressed by the (blob_index, offset, length) fields
//! inside each BlobAssetDef / ResourceRecord.
//!
//! Blob 0 is the primary blob. It always holds the full metadata and may also
//! hold payload bytes for assets packed before the size ceiling. Overflow
//! payloads spill into blobs 1, 2, ... as needed. All blobs share the same
//! header format; only blob 0 carries non-empty metadata.
//!
//! This module owns the format contract and nothing else: the record schema,
//! the header consts, the version, and the pure bytes <-> metadata transforms.
//! It performs no I/O and holds no residency policy, so it never learns where
//! blob files live or which of them are resident. Callers own both:
//! `concinnity_host::store` reads the state root's `data/` layout into `BlobData`,
//! concinnity-cook writes what `encode_cnb` returns. Being I/O-free is what
//! lets a no_std client runtime decode blobs with its own byte source.
//!
//! Nothing here needs to change when a new asset type is added.

mod encode;
mod error;
mod frame;
mod parse;
mod schema;

pub use encode::encode_cnb;
pub use error::BlobError;
pub use frame::{FrameError, decode_exact};
pub use parse::{parse_cnb, parse_payload_section_start, payload_section};
pub use schema::{
    AssetKind, BlobAssetDef, BlobMeta, MeshBoundsRecord, PhysicsBudgetRecord, ResourceKind,
    ResourceRecord, SceneGroup, WorldManifest,
};

// The identity and payload-address types the records carry, owned by the
// components module.
pub use crate::ecs::PayloadLocator;
pub use crate::ecs::asset_id::AssetId;

/// The four magic bytes every `.cnb` blob starts with.
pub const BLOB_MAGIC: [u8; 4] = *b"CNB\0";
/// Fixed blob header size: magic (4) + schema version (4) + `meta_len` (8).
pub const HEADER_SIZE: usize = 16; // magic(4) + schema version(4) + meta_len(8)
