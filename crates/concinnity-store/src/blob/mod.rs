//! Runtime blob access: the `.concinnity/data/` path layout, the payload
//! residency store, and all blob file I/O. The concinnity-blob crate owns the
//! format contract (schema, header, version, bytes <-> metadata) and is
//! deliberately I/O-free, so every read below is `fs` here plus a pure parse
//! there. Blob data is read-only at runtime; concinnity-cook writes what
//! `concinnity_blob::encode_cnb` returns.
use std::fs;
use std::io::Read;

use concinnity_blob::BlobError;
use concinnity_core::result::CnResult;

mod data;

pub use concinnity_blob::{BLOB_MAGIC, HEADER_SIZE, WorldManifest};
pub use concinnity_core::SCHEMA_HASH;
pub use concinnity_core::ecs::{BlobAssetDef, BlobMeta, ResourceRecord};
pub use data::BlobData;

/// Format a blob file path for a given index under `.concinnity/data/`. Blob 0
/// is the primary blob (the metadata block plus the first payload section);
/// higher indices are overflow payload blobs. The format crate is path-agnostic;
/// this layout knowledge stays here.
pub fn blob_path(index: u32) -> String {
    crate::paths::data_dir()
        .join(index.to_string())
        .to_string_lossy()
        .into_owned()
}

/// Read and deserialize a blob's metadata section (component defs + resource
/// records). Returns (meta, payload_start_offset).
pub fn read_cnb(path: &str) -> Result<(BlobMeta, usize), CnResult> {
    let data = read_file(path)?;
    concinnity_blob::parse_cnb(SCHEMA_HASH, &data).map_err(|e| report(path, e))
}

/// Byte offset within a blob file at which its payload section begins. Reads
/// only the header; the disk-backed streaming source uses it to turn a
/// `PayloadLocator` offset into an absolute file offset.
/// Used only by the Metal-driven disk-backed streaming source for now
/// (Vulkan/DirectX streaming catch-up is a follow-up).
pub fn payload_section_start(path: &str) -> Result<u64, CnResult> {
    let mut file = fs::File::open(path).map_err(|e| {
        tracing::error!("Failed to open {}: {}", path, e);
        CnResult::FileIo
    })?;
    let mut header = [0u8; HEADER_SIZE];
    file.read_exact(&mut header).map_err(|e| {
        tracing::error!("Failed to read header of {}: {}", path, e);
        CnResult::FileIo
    })?;
    concinnity_blob::parse_payload_section_start(&header).map_err(|e| report(path, e))
}

// Read just the payload section of a blob file into memory.
fn read_payload_section(path: &str) -> Result<Vec<u8>, CnResult> {
    let data = read_file(path)?;
    Ok(concinnity_blob::payload_section(&data).to_vec())
}

fn read_file(path: &str) -> Result<Vec<u8>, CnResult> {
    fs::read(path).map_err(|e| {
        tracing::error!("Failed to read {}: {}", path, e);
        CnResult::FileIo
    })
}

// Log a format failure against the file it came from. The format crate has no
// path to name, so the diagnostic belongs here.
fn report(path: &str, e: BlobError) -> CnResult {
    match e {
        BlobError::TooShort => tracing::error!("{}: file too short", path),
        BlobError::BadMagic => tracing::error!("{}: bad magic", path),
        BlobError::SchemaMismatch(_) => tracing::error!(
            "{}: world data was built by a different version of the engine",
            path
        ),
        BlobError::TruncatedMeta => tracing::error!("{}: truncated metadata section", path),
        BlobError::Decode => tracing::error!("{}: failed to deserialize metadata", path),
        BlobError::Encode => tracing::error!("{}: failed to serialize metadata", path),
    }
    CnResult::FileIo
}

/// Load the primary blob's metadata and the `BlobData` payload store from the
/// `.concinnity/data/` layout, without resolving defs into runtime `Asset`s
/// (that resolution depends on the client runtime registry, so it lives in the
/// client `blob::load` shim).
///
/// Only blob 0's payload section is read here; overflow blobs (named by the
/// manifest's `max_blob_index`) start unloaded and `BlobData::read()` pulls
/// each from disk the first time a locator needs it.
pub fn load_raw() -> Result<(BlobMeta, BlobData), CnResult> {
    load_raw_from(blob_path)
}

// `load_raw` against an injected layout, so the eager/deferred split can be
// exercised without the process-global data-dir anchor.
fn load_raw_from(blob_path: impl Fn(u32) -> String) -> Result<(BlobMeta, BlobData), CnResult> {
    let (meta, _payload_start) = read_cnb(&blob_path(0))?;

    // Cook derives the manifest from the very streams it summarizes, so a
    // mismatch means a corrupt or hand-edited blob.
    debug_assert_eq!(
        meta.manifest,
        WorldManifest::from_records(&meta.defs, &meta.resources),
        "blob manifest does not match its record streams"
    );

    let blob0_payload = read_payload_section(&blob_path(0))?;
    tracing::debug!("Loaded blob 0 payload ({} bytes)", blob0_payload.len());
    let overflow_paths = (1..=meta.manifest.max_blob_index).map(blob_path).collect();

    let blob_data = BlobData::from_blob_files(blob0_payload, overflow_paths);
    Ok((meta, blob_data))
}

/// Number of Texture resource records in the primary blob's metadata, read
/// without loading any payload. This is the compiled world's texture-table
/// length; `cn export` uses it to precompile the built-in shaders whose bindless
/// texture pool is sized per world.
pub fn texture_resource_count() -> Result<usize, CnResult> {
    let (meta, _) = read_cnb(&blob_path(0))?;
    let tag = concinnity_core::ecs::ResourceKind::Texture as u8;
    Ok(meta
        .resources
        .iter()
        .filter(|r| r.resource_kind == tag)
        .count())
}

/// Load defs without resolving (for callers that apply overlays first)
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
    fn format_failures_fold_onto_file_io() {
        assert_eq!(report("x.cnb", BlobError::BadMagic), CnResult::FileIo);
        assert_eq!(
            report("x.cnb", BlobError::SchemaMismatch(99)),
            CnResult::FileIo
        );
    }

    #[test]
    fn read_cnb_errors_on_a_missing_file() {
        assert_eq!(
            read_cnb("/nonexistent/cn/blob/path.cnb"),
            Err(CnResult::FileIo)
        );
    }

    #[test]
    fn read_payload_section_returns_empty_for_a_short_file() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("short").to_string_lossy().into_owned();
        std::fs::write(&path, vec![0u8; HEADER_SIZE - 1]).unwrap();
        assert!(read_payload_section(&path).unwrap().is_empty());
    }

    #[test]
    fn payload_section_start_skips_header_and_meta() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("0").to_string_lossy().into_owned();
        let image = concinnity_blob::encode_cnb(SCHEMA_HASH, &BlobMeta::default(), b"payloadbytes")
            .unwrap();
        std::fs::write(&path, &image).unwrap();

        let start = payload_section_start(&path).expect("section start");
        assert_eq!(&image[start as usize..], b"payloadbytes");
    }

    #[test]
    fn payload_section_start_rejects_bad_magic() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("bad").to_string_lossy().into_owned();
        std::fs::write(&path, vec![0u8; HEADER_SIZE]).unwrap();
        assert_eq!(payload_section_start(&path), Err(CnResult::FileIo));
    }

    #[test]
    fn load_raw_reads_blob0_eagerly_and_defers_overflow() {
        use concinnity_core::ecs::{AssetKind, PayloadLocator};

        let dir = tempfile::tempdir().unwrap();
        let path_for = |idx: u32| {
            dir.path()
                .join(idx.to_string())
                .to_string_lossy()
                .into_owned()
        };

        // Blob 0: one def whose payload lives in overflow blob 1. The manifest
        // is derived exactly as cook derives it; `load_raw` trusts its
        // `max_blob_index` to name the overflow file.
        let defs = vec![BlobAssetDef {
            name: None,
            kind: AssetKind::Component,
            discriminant: 1,
            args_bytes: Vec::new(),
            payload: Some(PayloadLocator {
                blob_index: 1,
                offset: 0,
                len: 8,
            }),
        }];
        let meta = BlobMeta {
            manifest: WorldManifest::from_records(&defs, &[]),
            defs,
            resources: Vec::new(),
            scene_groups: Vec::new(),
            mesh_bounds: Vec::new(),
            physics_budget: None,
        };
        std::fs::write(
            path_for(0),
            concinnity_blob::encode_cnb(SCHEMA_HASH, &meta, b"primary").unwrap(),
        )
        .unwrap();
        std::fs::write(
            path_for(1),
            concinnity_blob::encode_cnb(SCHEMA_HASH, &BlobMeta::default(), b"overflow").unwrap(),
        )
        .unwrap();

        let (meta, mut bd) = load_raw_from(path_for).expect("load");
        assert_eq!(meta.defs.len(), 1);
        assert!(meta.resources.is_empty());
        assert_eq!(meta.manifest.component_counts, vec![(1, 1)]);
        assert!(bd.disk_backed());
        // Blob 0 resident, blob 1 deferred until its first read.
        assert!(bd.is_loaded(0));
        assert!(!bd.is_loaded(1));
        let loc = meta.defs[0].payload.clone().unwrap();
        assert_eq!(bd.read(&loc).expect("overflow read"), b"overflow");
        assert!(bd.is_loaded(1));
    }
}
