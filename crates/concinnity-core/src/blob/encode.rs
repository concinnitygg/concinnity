// The blob encode half. Writing the bytes out is the caller's job: only
// concinnity-cook packs blobs, and it already owns the packing policy (payload
// distribution across overflow blobs, the size ceiling) and the output paths.

use alloc::vec::Vec;

use crate::blob::error::BlobError;
use crate::blob::schema::BlobMeta;
use crate::blob::{BLOB_MAGIC, HEADER_SIZE};

/// Encode a blob image: the 16-byte header, the postcard-serialized metadata
/// block, then the raw payload section.
///
/// `schema_version` is stamped into the header and checked on load. The
/// container does not own the asset types travelling inside it, so the caller
/// supplies the version; runtime callers pass
/// `concinnity_core::SCHEMA_VERSION`.
pub fn encode_cnb(
    schema_version: u32,
    meta: &BlobMeta,
    payload: &[u8],
) -> Result<Vec<u8>, BlobError> {
    let meta_bytes: Vec<u8> = postcard::to_allocvec(meta).map_err(|_| BlobError::Encode)?;

    let mut data = Vec::with_capacity(HEADER_SIZE + meta_bytes.len() + payload.len());
    data.extend_from_slice(&BLOB_MAGIC);
    data.extend_from_slice(&schema_version.to_le_bytes());
    data.extend_from_slice(&(meta_bytes.len() as u64).to_le_bytes());
    data.extend_from_slice(&meta_bytes);
    data.extend_from_slice(payload);
    Ok(data)
}
