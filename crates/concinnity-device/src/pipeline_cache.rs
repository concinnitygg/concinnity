// Disk persistence for driver pipeline blobs: a serialized VkPipelineCache or
// a D3D12 pipeline library, one file per adapter under `pipeline-cache/`.
//
// These blobs are machine code tied to one GPU and driver, so unlike the
// shader cache there is no bundled tier and no cross-machine reuse; the driver
// (or the backend's own header check) rejects a stale blob and the launch
// falls back to building pipelines cold. Every operation here is best-effort:
// an unreadable, oversized, or rejected file is deleted and treated as absent,
// never surfaced as an init failure.

use std::path::PathBuf;
use std::sync::atomic::{AtomicU64, Ordering};

// Neither VkPipelineCache nor a D3D12 pipeline library evicts internally, so a
// long-lived checkout accumulates entries for every edited shader. Past this
// cap the file is deleted and rebuilt cold rather than growing forever.
const FILE_BUDGET_BYTES: u64 = 32 * 1024 * 1024;

static CREATED: AtomicU64 = AtomicU64::new(0);
static CREATE_MICROS: AtomicU64 = AtomicU64::new(0);

// Record one pipeline creation for the init tally.
pub(crate) fn note_creation(micros: u64) {
    CREATED.fetch_add(1, Ordering::Relaxed);
    CREATE_MICROS.fetch_add(micros, Ordering::Relaxed);
}

// Log the pipeline-creation cost of a renderer init and whether the disk blob
// was used. Pipelines built lazily after init (wireframe twins, world shader
// buckets) land after this tally, so it is a snapshot rather than a total.
pub(crate) fn report_init(disk: &str) {
    let (created, micros) = (
        CREATED.load(Ordering::Relaxed),
        CREATE_MICROS.load(Ordering::Relaxed),
    );
    if created == 0 {
        return;
    }
    tracing::info!(
        "pipeline cache: {created} pipelines created ({:.0} ms) at renderer init, disk blob {disk}",
        micros as f64 / 1000.0
    );
}

// Off under `cargo test` so the suite never writes into a developer's state
// dir nor warms a run from a previous test's blob.
fn enabled() -> bool {
    !cfg!(test)
}

fn file_path(name: &str) -> PathBuf {
    concinnity_store::paths::pipeline_cache_dir().join(name)
}

// Read the persisted blob for `name`. An empty or over-budget file is deleted
// and reads as absent.
pub(crate) fn load(name: &str) -> Option<Vec<u8>> {
    if !enabled() {
        return None;
    }
    load_in(&file_path(name))
}

fn load_in(path: &std::path::Path) -> Option<Vec<u8>> {
    let bytes = std::fs::read(path).ok()?;
    if bytes.is_empty() || bytes.len() as u64 > FILE_BUDGET_BYTES {
        let _ = std::fs::remove_file(path);
        return None;
    }
    Some(bytes)
}

// Persist `bytes` for `name` when they hold more than `previous` (the blob
// this run loaded or last wrote). Growth is the only reliable "new content"
// signal: a driver cache only accumulates, but its serialization is not
// deterministic (MoltenVK shuffles entry order run to run), so a byte compare
// would rewrite an unchanged cache every launch. Returns whether a write
// happened, for the caller's bookkeeping.
pub(crate) fn store_if_grown(name: &str, bytes: &[u8], previous: Option<&[u8]>) -> bool {
    if !enabled() {
        return false;
    }
    store_in(&file_path(name), bytes, previous)
}

fn store_in(path: &std::path::Path, bytes: &[u8], previous: Option<&[u8]>) -> bool {
    if bytes.is_empty() || bytes.len() <= previous.map_or(0, <[u8]>::len) {
        return false;
    }
    let Some(dir) = path.parent() else {
        return false;
    };
    if std::fs::create_dir_all(dir).is_err() {
        return false;
    }
    // Process-unique temp plus rename, so a concurrent reader (a second `cn
    // debug` on the same checkout) never observes a partial blob.
    let tmp = path.with_extension(format!("{}.tmp", std::process::id()));
    if std::fs::write(&tmp, bytes).is_err() {
        return false;
    }
    if std::fs::rename(&tmp, path).is_err() {
        let _ = std::fs::remove_file(&tmp);
        return false;
    }
    true
}

// Drop the persisted blob for `name`; used when the driver rejects it.
pub(crate) fn delete(name: &str) {
    let _ = std::fs::remove_file(file_path(name));
}

#[cfg(test)]
mod tests {
    use super::*;

    fn temp_file(tag: &str) -> PathBuf {
        std::env::temp_dir().join(format!("cn_pso_{tag}_{}", std::process::id()))
    }

    #[test]
    fn a_blob_round_trips_through_a_file() {
        let path = temp_file("roundtrip").join("adapter.bin");
        let _ = std::fs::remove_dir_all(path.parent().unwrap());
        assert!(store_in(&path, &[1, 2, 3], None));
        assert_eq!(load_in(&path), Some(vec![1, 2, 3]));
        let leftovers = std::fs::read_dir(path.parent().unwrap())
            .unwrap()
            .flatten()
            .filter(|e| e.path().extension().is_some_and(|x| x == "tmp"))
            .count();
        assert_eq!(leftovers, 0, "temp files must not survive a store");
        std::fs::remove_dir_all(path.parent().unwrap()).unwrap();
    }

    #[test]
    fn only_growth_rewrites_the_blob() {
        let path = temp_file("growth").join("adapter.bin");
        let _ = std::fs::remove_dir_all(path.parent().unwrap());
        assert!(store_in(&path, &[5, 5], None));
        let first_write = std::fs::metadata(&path).unwrap().modified().unwrap();
        // Same or reshuffled content of equal length is not new content: the
        // driver's serialization is nondeterministic, so equal-length bytes
        // must leave the file untouched (byte-stable warm launches).
        assert!(!store_in(&path, &[5, 5], Some(&[5, 5])));
        assert!(!store_in(&path, &[6, 5], Some(&[5, 5])), "reshuffled");
        assert!(!store_in(&path, &[5], Some(&[5, 5])), "shrunk");
        assert_eq!(
            std::fs::metadata(&path).unwrap().modified().unwrap(),
            first_write
        );
        assert!(store_in(&path, &[5, 5, 6], Some(&[5, 5])), "growth writes");
        std::fs::remove_dir_all(path.parent().unwrap()).unwrap();
    }

    #[test]
    fn an_empty_blob_is_neither_stored_nor_loaded() {
        let path = temp_file("empty").join("adapter.bin");
        let _ = std::fs::remove_dir_all(path.parent().unwrap());
        assert!(!store_in(&path, &[], None));
        std::fs::create_dir_all(path.parent().unwrap()).unwrap();
        std::fs::write(&path, []).unwrap();
        assert_eq!(load_in(&path), None);
        assert!(!path.exists(), "a truncated file is deleted on load");
        std::fs::remove_dir_all(path.parent().unwrap()).unwrap();
    }

    #[test]
    fn an_over_budget_blob_is_deleted_on_load() {
        let path = temp_file("budget").join("adapter.bin");
        let _ = std::fs::remove_dir_all(path.parent().unwrap());
        std::fs::create_dir_all(path.parent().unwrap()).unwrap();
        std::fs::write(&path, vec![0u8; (FILE_BUDGET_BYTES + 1) as usize]).unwrap();
        assert_eq!(load_in(&path), None);
        assert!(!path.exists());
        std::fs::remove_dir_all(path.parent().unwrap()).unwrap();
    }
}
