// Runtime blob access: the `.concinnity/data/` path layout plus thin CnResult
// wrappers over the concinnity-blob format crate, which owns the record
// schema, the header, the version, and the read half. The WRITE half is
// feature-gated inside that crate and enabled only by concinnity-cook -- the
// runtime treats blob data as read-only.
use crate::result::CnResult;

pub use crate::ecs::{BlobAssetDef, BlobMeta, ResourceRecord};
pub use concinnity_blob::{BLOB_MAGIC, BLOB_VERSION, BlobData, BlobError, HEADER_SIZE};

// Format a blob file path for a given index under `.concinnity/data/`. Blob 0
// is the primary blob (the metadata block plus the first payload section);
// higher indices are overflow payload blobs. The format crate is path-agnostic;
// this layout knowledge stays here.
pub fn blob_path(index: u32) -> String {
    crate::paths::data_dir()
        .join(index.to_string())
        .to_string_lossy()
        .into_owned()
}

// Read and deserialize a blob's metadata section (component defs + resource
// records). Returns (meta, payload_start_offset).
pub fn read_cnb(path: &str) -> Result<(BlobMeta, usize), CnResult> {
    concinnity_blob::read_cnb(path).map_err(Into::into)
}

// Byte offset within a blob file at which its payload section begins. Reads
// only the header; the disk-backed streaming source uses it to turn a
// `PayloadLocator` offset into an absolute file offset.
// Used only by the Metal-driven disk-backed streaming source for now
// (Vulkan/DirectX streaming catch-up is a follow-up).
#[cfg_attr(not(target_os = "macos"), allow(dead_code))]
pub fn payload_section_start(path: &str) -> Result<u64, CnResult> {
    concinnity_blob::payload_section_start(path).map_err(Into::into)
}

// Load the primary blob's defs and the `BlobData` payload store from the
// `.concinnity/data/` layout, without resolving defs into runtime `Asset`s
// (that resolution depends on the client runtime registry, so it lives in the
// client `blob::load` shim).
pub fn load_raw() -> Result<(Vec<BlobAssetDef>, Vec<ResourceRecord>, BlobData), CnResult> {
    concinnity_blob::load_raw(blob_path).map_err(Into::into)
}

// Load defs without resolving (for callers that apply overlays first)
#[allow(dead_code)]
pub fn load_defs() -> Result<Vec<BlobAssetDef>, CnResult> {
    read_cnb(&blob_path(0)).map(|(meta, _)| meta.defs)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn blob_path_appends_the_index() {
        // Tolerant of whatever the process-global data-dir anchor currently is:
        // only the trailing index and distinctness are asserted.
        assert!(blob_path(5).ends_with('5'));
        assert_ne!(blob_path(0), blob_path(1));
    }

    #[test]
    fn blob_errors_fold_onto_file_io() {
        assert_eq!(CnResult::from(BlobError::Io), CnResult::FileIo);
        assert_eq!(CnResult::from(BlobError::Format), CnResult::FileIo);
    }
}
