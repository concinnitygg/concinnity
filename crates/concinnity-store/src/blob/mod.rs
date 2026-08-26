//! Runtime blob access: the state root's `data/` path layout, the payload
//! residency store, and all blob file I/O. The concinnity-blob crate owns the
//! format contract (schema, header, version, bytes <-> metadata) and is
//! deliberately I/O-free, so every read below is `fs` here plus a pure parse
//! there. Blob data is read-only at runtime; concinnity-cook writes what
//! `concinnity_core::blob::encode_cnb` returns.
use std::fs;
use std::io::Read;
use std::path::{Path, PathBuf};
use std::sync::{Mutex, OnceLock};

pub use concinnity_core::blob::{BLOB_MAGIC, HEADER_SIZE, WorldManifest};
use concinnity_core::blob::{BlobError, parse_cnb, parse_payload_section_start, payload_section};
use concinnity_core::result::CnResult;

mod data;

pub use concinnity_core::SCHEMA_HASH;
pub use concinnity_core::ecs::{BlobAssetDef, BlobMeta, ResourceRecord};
pub use data::BlobData;

// Where the blob files live when a host named a primary blob directly, rather
// than taking the state root's `data/` layout. Holds that file's path; every
// overflow blob is its sibling named by index.
fn primary_override() -> &'static Mutex<Option<PathBuf>> {
    static PRIMARY: OnceLock<Mutex<Option<PathBuf>>> = OnceLock::new();
    PRIMARY.get_or_init(|| Mutex::new(None))
}

/// Format a blob file path for a given index. Blob 0 is the primary blob (the
/// metadata block plus the first payload section); higher indices are overflow
/// payload blobs, which are always siblings of blob 0. The format crate is
/// path-agnostic; this layout knowledge stays here.
///
/// Blob 0 is `<state dir>/data/0` unless [`load_raw_at`] anchored the layout on
/// a blob file named directly, in which case that file is blob 0 and its
/// directory holds the rest. `None` when neither applies: no host installed a
/// state root and none named a blob, so there is no layout to resolve against.
pub fn blob_path(index: u32) -> Option<String> {
    let primary = primary_override().lock().unwrap().clone();
    resolve_blob_path(
        primary.as_deref(),
        crate::paths::data_dir().as_deref(),
        index,
    )
}

// Pure resolution split out so the two layouts are unit-testable without the
// process-global anchor or the process-global state root.
fn resolve_blob_path(
    primary: Option<&Path>,
    data_dir: Option<&Path>,
    index: u32,
) -> Option<String> {
    let path = match primary {
        Some(p) if index == 0 => p.to_path_buf(),
        Some(p) => p
            .parent()
            .map_or_else(|| PathBuf::from("."), Path::to_path_buf)
            .join(index.to_string()),
        None => data_dir?.join(index.to_string()),
    };
    Some(path.to_string_lossy().into_owned())
}

/// Read and deserialize a blob's metadata section (component defs + resource
/// records). Returns (meta, payload_start_offset).
pub fn read_cnb(path: &str) -> Result<(BlobMeta, usize), CnResult> {
    let data = read_file(path)?;
    parse_cnb(SCHEMA_HASH, &data).map_err(|e| report(path, e))
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
    parse_payload_section_start(&header).map_err(|e| report(path, e))
}

// Read just the payload section of a blob file into memory.
fn read_payload_section(path: &str) -> Result<Vec<u8>, CnResult> {
    let data = read_file(path)?;
    Ok(payload_section(&data).to_vec())
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

/// Load the blob file at `primary` and the payload store around it, anchoring
/// the process's blob layout on it: that file is blob 0 and its siblings named
/// by index are the overflow payload blobs, so a world written to `data/0`
/// reads `data/1`, `data/2`, ... beside it. The anchor outlives the call
/// because payloads stream off disk long after startup.
pub fn load_raw_at(primary: &Path) -> Result<(BlobMeta, BlobData), CnResult> {
    *primary_override().lock().unwrap() = Some(primary.to_path_buf());
    load_raw()
}

/// Load the primary blob's metadata and the `BlobData` payload store from the
/// state root's `data/` layout, without resolving defs into runtime `Asset`s
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
fn load_raw_from(
    blob_path: impl Fn(u32) -> Option<String>,
) -> Result<(BlobMeta, BlobData), CnResult> {
    let (meta, _payload_start) = read_cnb(&blob_path(0).ok_or(CnResult::NoStateRoot)?)?;

    // Cook derives the manifest from the very streams it summarizes, so a
    // mismatch means a corrupt or hand-edited blob.
    debug_assert_eq!(
        meta.manifest,
        WorldManifest::from_records(&meta.defs, &meta.resources),
        "blob manifest does not match its record streams"
    );

    let blob0_payload = read_payload_section(&blob_path(0).ok_or(CnResult::NoStateRoot)?)?;
    tracing::debug!("Loaded blob 0 payload ({} bytes)", blob0_payload.len());
    let overflow_paths = (1..=meta.manifest.max_blob_index)
        .map(|i| blob_path(i).ok_or(CnResult::NoStateRoot))
        .collect::<Result<Vec<_>, _>>()?;

    let blob_data = BlobData::from_blob_files(blob0_payload, overflow_paths);
    Ok((meta, blob_data))
}

/// Number of Texture resource records in the primary blob's metadata, read
/// without loading any payload. This is the compiled world's texture-table
/// length; `cn export` uses it to precompile the built-in shaders whose bindless
/// texture pool is sized per world.
pub fn texture_resource_count() -> Result<usize, CnResult> {
    let (meta, _) = read_cnb(&blob_path(0).ok_or(CnResult::NoStateRoot)?)?;
    let tag = concinnity_core::ecs::ResourceKind::Texture as u8;
    Ok(meta
        .resources
        .iter()
        .filter(|r| r.resource_kind == tag)
        .count())
}

/// Load defs without resolving (for callers that apply overlays first)
pub fn load_defs() -> Result<Vec<BlobAssetDef>, CnResult> {
    read_cnb(&blob_path(0).ok_or(CnResult::NoStateRoot)?).map(|(meta, _)| meta.defs)
}

#[cfg(test)]
mod tests {
    use super::*;
    use concinnity_core::blob::encode_cnb;

    #[test]
    fn an_anchored_primary_owns_blob_zero_and_its_siblings() {
        // Blob 0 is the file named verbatim (whatever it is called); every
        // overflow blob is its sibling named by index. Built through `join` so
        // the separator is the platform's.
        let primary = Path::new("out").join("blobs").join("0");
        assert_eq!(
            resolve_blob_path(Some(&primary), None, 0).as_deref(),
            Some(&*primary.to_string_lossy())
        );
        assert_eq!(
            resolve_blob_path(Some(&primary), None, 2),
            Some(
                Path::new("out")
                    .join("blobs")
                    .join("2")
                    .to_string_lossy()
                    .into_owned()
            )
        );

        // A bare file name hangs its siblings off the working directory.
        assert_eq!(
            resolve_blob_path(Some(Path::new("0")), None, 1).as_deref(),
            Some("1")
        );
    }

    #[test]
    fn without_an_anchor_blobs_sit_under_the_data_dir() {
        let data = Path::new("/proj").join("data");
        assert_eq!(
            resolve_blob_path(None, Some(&data), 3),
            Some(data.join("3").to_string_lossy().into_owned())
        );
    }

    // With neither an anchor nor a state root there is no layout to resolve
    // against, which is what turns a blob read into `NoStateRoot` rather than a
    // read of some path relative to the working directory.
    #[test]
    fn without_an_anchor_or_a_state_root_there_is_no_path() {
        assert_eq!(resolve_blob_path(None, None, 0), None);
        assert_eq!(resolve_blob_path(None, None, 3), None);
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
        let image = encode_cnb(SCHEMA_HASH, &BlobMeta::default(), b"payloadbytes").unwrap();
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
            Some(
                dir.path()
                    .join(idx.to_string())
                    .to_string_lossy()
                    .into_owned(),
            )
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
            path_for(0).unwrap(),
            encode_cnb(SCHEMA_HASH, &meta, b"primary").unwrap(),
        )
        .unwrap();
        std::fs::write(
            path_for(1).unwrap(),
            encode_cnb(SCHEMA_HASH, &BlobMeta::default(), b"overflow").unwrap(),
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

    // A layout that resolves to nothing is the uninstalled-state-root case, and
    // it has to name itself rather than folding onto a file-not-found.
    #[test]
    fn load_raw_without_a_layout_reports_no_state_root() {
        assert_eq!(load_raw_from(|_| None).err(), Some(CnResult::NoStateRoot));
    }
}
