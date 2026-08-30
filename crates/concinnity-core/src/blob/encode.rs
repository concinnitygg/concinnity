// The blob encode half. Writing the bytes out is the caller's job: only
// concinnity-cook packs blobs, and it already owns the packing policy (payload
// distribution across overflow blobs, the size ceiling) and the output paths.

use alloc::vec::Vec;

use crate::blob::HEADER_SIZE;
use crate::blob::error::BlobError;
use crate::blob::kind::BlobKind;

/// Encode a blob image: the 16-byte header, the postcard-serialized metadata
/// block, then the raw payload section.
///
/// The header's magic comes from `K`, so the bytes declare which kind of
/// container they are.
///
/// `validity` is the token stamped into the header and required to match on
/// parse. What it means belongs to the kind: for the cooked world's
/// [`BlobMeta`](crate::blob::BlobMeta) it is a schema version, and runtime
/// callers pass `concinnity_core::SCHEMA_VERSION`.
pub fn encode_cnb<K: BlobKind>(
    validity: u32,
    meta: &K,
    payload: &[u8],
) -> Result<Vec<u8>, BlobError> {
    let mut data = encode_cnb_prefix(validity, meta)?;
    data.reserve(payload.len());
    data.extend_from_slice(payload);
    Ok(data)
}

/// Everything an image carries before its payload section: the 16-byte header
/// and the metadata block.
///
/// For a writer that streams its payload rather than holding it in memory. A
/// build cache segment is the case that needs it: its payload can run to
/// hundreds of megabytes, most of them copied from the segment it replaces, so
/// the bytes go to the file as they are produced and only this prefix is built
/// as a value.
pub fn encode_cnb_prefix<K: BlobKind>(validity: u32, meta: &K) -> Result<Vec<u8>, BlobError> {
    let meta_bytes: Vec<u8> = postcard::to_allocvec(meta).map_err(|_| BlobError::Encode)?;

    let mut data = Vec::with_capacity(HEADER_SIZE + meta_bytes.len());
    data.extend_from_slice(&K::MAGIC);
    data.extend_from_slice(&validity.to_le_bytes());
    data.extend_from_slice(&(meta_bytes.len() as u64).to_le_bytes());
    data.extend_from_slice(&meta_bytes);
    Ok(data)
}
