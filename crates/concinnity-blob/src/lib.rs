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
// This crate owns the format contract: the record schema, the header consts,
// the version, and the read half. Blob data is read-only at runtime -- the
// symmetric encode half (`write_cnb`) sits behind the `write` feature and only
// the compile pipeline enables it. The crate knows nothing about where blob
// files live: callers supply explicit paths (the `.concinnity/data/` layout is
// concinnity-core's).
//
// This crate never needs to change when a new asset type is added.

mod read;
mod schema;
#[cfg(feature = "write")]
mod write;

pub use read::{BlobData, BlobError, load_raw, payload_section_start, read_cnb};
pub use schema::{AssetKind, BlobAssetDef, BlobMeta, ResourceKind, ResourceRecord};
#[cfg(feature = "write")]
pub use write::write_cnb;

// The identity and payload-address types the records carry, owned by the
// schema crate.
pub use concinnity_asset::{AssetId, PayloadLocator};

pub const BLOB_MAGIC: [u8; 4] = *b"CNB\0";
// Bump on any postcard-visible schema change so a stale blob fails the version
// check with a clear "rebuild" error instead of mis-decoding. v2: every record
// is baked -- `RecordKind` and BlobAssetDef's `record` field left the schema.
// v3: `args_bytes` / resource `data_bytes` are postcard-encoded components, not
// JSON. v4: View became Screen (stack, input policy, layer fields) and element
// `view` refs became `screen`.
pub const BLOB_VERSION: u32 = 4;
pub const HEADER_SIZE: usize = 16; // magic(4) + version(4) + meta_len(8)
