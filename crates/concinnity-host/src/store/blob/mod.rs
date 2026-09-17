//! Runtime blob access: the state root's `data/` path layout, the payload
//! residency store, and all blob file I/O. `concinnity_core::blob` owns the
//! format contract (schema, header, version, bytes <-> metadata) and is
//! deliberately I/O-free, so every read below is `fs` here plus a pure parse
//! there. Blob data is read-only at runtime; concinnity-cook writes what
//! `concinnity_core::blob::encode_cnb` returns.
use std::fs;
use std::io::Read;
use std::path::{Path, PathBuf};
use std::sync::{Mutex, OnceLock};

pub use concinnity_core::blob::{BLOB_MAGIC, HEADER_SIZE, WorldManifest};
use concinnity_core::blob::{parse_cnb, parse_payload_section_start, payload_section};

mod data;
mod error;

pub use error::BlobLoadError;

pub use concinnity_core::SCHEMA_VERSION;
pub use concinnity_core::ecs::{BlobAssetDef, BlobMeta, ResourceRecord};
pub use data::BlobData;

// The primary blob this process reads, named by whatever anchored it. Process
// state because payloads stream off disk long after startup: a locator resolved
// mid-frame has no caller to carry the layout down from.
fn anchored_primary() -> &'static Mutex<Option<PathBuf>> {
    static PRIMARY: OnceLock<Mutex<Option<PathBuf>>> = OnceLock::new();
    PRIMARY.get_or_init(|| Mutex::new(None))
}

// Anchor the process's blob layout on `primary`: that file is blob 0 and its
// siblings named by index are the overflow payload blobs. Reached only through
// `load_raw_at`, so a process addresses the blobs it actually opened.
fn anchor(primary: &Path) {
    *anchored_primary().lock().unwrap() = Some(primary.to_path_buf());
}

// Blob 0 is the primary blob; every other index names an overflow sibling.
const PRIMARY_INDEX: u32 = 0;

/// The primary blob inside `data_dir`: blob 0, holding the metadata block plus
/// the first payload section. What a host opens when it reads a state tree's
/// `data/`, and what a build writes there.
pub fn primary_in(data_dir: &Path) -> PathBuf {
    data_dir.join(PRIMARY_INDEX.to_string())
}

/// Format a blob file path for a given index. Blob 0 is the primary blob
/// [`load_raw_at`] opened (the metadata block plus the first payload section);
/// higher indices are overflow payload blobs, which are always its siblings.
/// The format crate is path-agnostic; this layout knowledge stays here.
///
/// `None` before any load, so there is no layout to resolve against.
pub fn blob_path(index: u32) -> Option<String> {
    let primary = anchored_primary().lock().unwrap().clone();
    resolve_blob_path(primary.as_deref(), index)
}

// Pure resolution split out so the sibling naming is unit-testable without the
// process-global anchor.
fn resolve_blob_path(primary: Option<&Path>, index: u32) -> Option<String> {
    let primary = primary?;
    let path = if index == PRIMARY_INDEX {
        primary.to_path_buf()
    } else {
        primary
            .parent()
            .map_or_else(|| PathBuf::from("."), Path::to_path_buf)
            .join(index.to_string())
    };
    Some(path.to_string_lossy().into_owned())
}

/// Read and deserialize a blob's metadata section (component defs + resource
/// records). Returns (meta, payload_start_offset).
pub fn read_cnb(path: &str) -> Result<(BlobMeta, usize), BlobLoadError> {
    let data = read_file(path)?;
    parse_cnb::<BlobMeta>(SCHEMA_VERSION, &data)
        .map_err(|source| BlobLoadError::format(path, source))
}

/// Byte offset within a blob file at which its payload section begins. Reads
/// only the header; the disk-backed streaming source uses it to turn a
/// `PayloadLocator` offset into an absolute file offset.
/// Used only by the Metal-driven disk-backed streaming source for now
/// (Vulkan/DirectX streaming catch-up is a follow-up).
pub fn payload_section_start(path: &str) -> Result<u64, BlobLoadError> {
    let mut file = fs::File::open(path).map_err(|source| BlobLoadError::io(path, source))?;
    let mut header = [0u8; HEADER_SIZE];
    file.read_exact(&mut header)
        .map_err(|source| BlobLoadError::io(path, source))?;
    parse_payload_section_start::<BlobMeta>(&header)
        .map_err(|source| BlobLoadError::format(path, source))
}

// Read just the payload section of a blob file into memory.
fn read_payload_section(path: &str) -> Result<Vec<u8>, BlobLoadError> {
    let data = read_file(path)?;
    Ok(payload_section(&data).to_vec())
}

fn read_file(path: &str) -> Result<Vec<u8>, BlobLoadError> {
    fs::read(path).map_err(|source| BlobLoadError::io(path, source))
}

/// Load the blob file at `primary` and the payload store around it, anchoring
/// the process's blob layout on it, so a world written to `data/0` reads
/// `data/1`, `data/2`, ... beside it. The anchor outlives the call because
/// payloads stream off disk long after startup: a locator resolved mid-frame
/// has no caller to carry the layout down from.
///
/// Only blob 0's payload section is read here; overflow blobs (named by the
/// manifest's `max_blob_index`) start unloaded and `BlobData::read()` pulls
/// each from disk the first time a locator needs it. Defs are not resolved into
/// runtime `Asset`s: that resolution depends on the client runtime registry, so
/// it lives in the client `blob::load` shim.
pub fn load_raw_at(primary: &Path) -> Result<(BlobMeta, BlobData), BlobLoadError> {
    anchor(primary);
    load_raw_from(blob_path)
}

// `load_raw_at` against an injected layout, so the eager/deferred split can be
// exercised without the process-global data-dir anchor.
fn load_raw_from(
    blob_path: impl Fn(u32) -> Option<String>,
) -> Result<(BlobMeta, BlobData), BlobLoadError> {
    let (meta, _payload_start) = read_cnb(&blob_path(0).ok_or(BlobLoadError::NoStateRoot)?)?;

    // Cook derives the manifest from the very streams it summarizes, so a
    // mismatch means a corrupt or hand-edited blob.
    debug_assert_eq!(
        meta.manifest,
        WorldManifest::from_records(&meta.defs, &meta.resources),
        "blob manifest does not match its record streams"
    );

    let blob0_payload = read_payload_section(&blob_path(0).ok_or(BlobLoadError::NoStateRoot)?)?;
    tracing::debug!("Loaded blob 0 payload ({} bytes)", blob0_payload.len());
    let overflow_paths = (1..=meta.manifest.max_blob_index)
        .map(|i| blob_path(i).ok_or(BlobLoadError::NoStateRoot))
        .collect::<Result<Vec<_>, _>>()?;

    let blob_data = BlobData::from_blob_files(blob0_payload, overflow_paths);
    Ok((meta, blob_data))
}

#[cfg(test)]
mod tests {
    use super::*;
    use concinnity_core::blob::{BlobError, encode_cnb};

    #[test]
    fn an_anchored_primary_owns_blob_zero_and_its_siblings() {
        // Blob 0 is the file named verbatim (whatever it is called); every
        // overflow blob is its sibling named by index. Built through `join` so
        // the separator is the platform's.
        let primary = Path::new("out").join("blobs").join("0");
        assert_eq!(
            resolve_blob_path(Some(&primary), 0).as_deref(),
            Some(&*primary.to_string_lossy())
        );
        assert_eq!(
            resolve_blob_path(Some(&primary), 2),
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
            resolve_blob_path(Some(Path::new("0")), 1).as_deref(),
            Some("1")
        );

        // A state tree's `data/` is just a primary named `<tree>/data/0`, so
        // the overflow blobs land beside it the same way.
        let tree = crate::store::paths::StateTree::at(Path::new("/proj"));
        let data = tree.data_dir().join("0");
        assert_eq!(
            resolve_blob_path(Some(&data), 3),
            Some(tree.data_dir().join("3").to_string_lossy().into_owned())
        );
    }

    // With nothing anchored there is no layout to resolve against, which is
    // what turns a blob read into `NoStateRoot` rather than a read of some path
    // relative to the working directory.
    #[test]
    fn without_an_anchor_there_is_no_path() {
        assert_eq!(resolve_blob_path(None, 0), None);
        assert_eq!(resolve_blob_path(None, 3), None);
    }

    // A file that was read but cannot be used is not a disk failure, and the
    // reader is the only place that knows which file the bytes came from, so
    // the failure has to name both the path and what the format rejected.
    #[test]
    fn a_format_failure_names_the_file_and_keeps_its_cause() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("0").to_string_lossy().into_owned();
        std::fs::write(&path, vec![0xabu8; HEADER_SIZE * 2]).unwrap();

        let error = read_cnb(&path).unwrap_err();
        assert!(
            matches!(
                error,
                BlobLoadError::Format {
                    source: BlobError::BadMagic,
                    ..
                }
            ),
            "{error:?}"
        );
        assert!(error.to_string().contains(&path), "{error}");
        assert!(
            std::error::Error::source(&error).is_some(),
            "the format error stays reachable"
        );
    }

    // The other half of the split the flat code could not express: a file that
    // was never read at all.
    #[test]
    fn a_missing_file_reports_the_io_failure_that_found_it() {
        let error = read_cnb("/nonexistent/cn/blob/path.cnb").unwrap_err();
        let BlobLoadError::Io { source, .. } = &error else {
            panic!("expected an io failure, got {error:?}");
        };
        assert_eq!(source.kind(), std::io::ErrorKind::NotFound);
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
        let image = encode_cnb(SCHEMA_VERSION, &BlobMeta::default(), b"payloadbytes").unwrap();
        std::fs::write(&path, &image).unwrap();

        let start = payload_section_start(&path).expect("section start");
        assert_eq!(&image[start as usize..], b"payloadbytes");
    }

    #[test]
    fn payload_section_start_rejects_bad_magic() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("bad").to_string_lossy().into_owned();
        std::fs::write(&path, vec![0u8; HEADER_SIZE]).unwrap();
        assert!(matches!(
            payload_section_start(&path),
            Err(BlobLoadError::Format {
                source: BlobError::BadMagic,
                ..
            })
        ));
    }

    #[test]
    fn load_raw_reads_blob0_eagerly_and_defers_overflow() {
        use concinnity_core::ecs::PayloadLocator;

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
        // is derived exactly as cook derives it; `load_raw_from` trusts its
        // `max_blob_index` to name the overflow file.
        let defs = vec![BlobAssetDef {
            name: None,
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
            encode_cnb(SCHEMA_VERSION, &meta, b"primary").unwrap(),
        )
        .unwrap();
        std::fs::write(
            path_for(1).unwrap(),
            encode_cnb(SCHEMA_VERSION, &BlobMeta::default(), b"overflow").unwrap(),
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
        assert!(matches!(
            load_raw_from(|_| None),
            Err(BlobLoadError::NoStateRoot)
        ));
    }
}
