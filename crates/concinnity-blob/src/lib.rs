// concinnity-blob: the .cnb blob container format.
//
// Layout of a blob binary:
//
//   [ 4 bytes magic ][ 4 bytes version ][ 8 bytes meta_len ][ meta_bytes ][ payload_bytes ... ]
//
// The header is fixed at 16 bytes. `meta_len` is the byte length of the
// postcard-serialized `BlobMeta` that follows: the component defs stream and
// the resource records stream. Everything after meta_len + meta_bytes is the
// raw payload section, addressed by the (blob_index, offset, length) fields
// inside each BlobAssetDef / ResourceRecord.
//
// Blob 0 is the primary blob. It always holds the full metadata and may also
// hold payload bytes for assets packed before the size ceiling. Overflow
// payloads spill into blobs 1, 2, ... as needed. All blobs share the same
// header format; only blob 0 carries non-empty metadata.
//
// This crate owns the format contract and nothing else: the record schema, the
// header consts, the version, and the pure bytes <-> metadata transforms. It
// performs no I/O and holds no residency policy, so it never learns where blob
// files live or which of them are resident. Callers own both: concinnity-core
// reads the `.concinnity/data/` layout into `BlobData`, concinnity-cook writes
// what `encode_cnb` returns.
//
// Being I/O-free makes the crate `#![no_std]` (using only `core` + `alloc`), so
// a no_std client runtime can decode blobs with its own byte source.
//
// This crate never needs to change when a new asset type is added.

#![no_std]

extern crate alloc;

// The test harness still wants std.
#[cfg(test)]
#[macro_use]
extern crate std;

mod encode;
mod error;
mod parse;
mod schema;

pub use encode::encode_cnb;
pub use error::BlobError;
pub use parse::{parse_cnb, parse_payload_section_start, payload_section};
pub use schema::{
    AssetKind, BlobAssetDef, BlobMeta, MeshBoundsRecord, ResourceKind, ResourceRecord, SceneGroup,
    WorldManifest,
};

// The identity and payload-address types the records carry, owned by the
// schema crate.
pub use concinnity_asset::{AssetId, PayloadLocator};

pub const BLOB_MAGIC: [u8; 4] = *b"CNB\0";
// SCHEMA_HASH: derived by build.rs from the postcard-visible schema sources,
// so any change to them is caught by the load check.
include!(concat!(env!("OUT_DIR"), "/schema_hash.rs"));
pub const HEADER_SIZE: usize = 16; // magic(4) + schema hash(4) + meta_len(8)
