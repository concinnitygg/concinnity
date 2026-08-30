// Disk persistence for driver pipeline blobs: a serialized VkPipelineCache or
// a D3D12 pipeline library, one entry per adapter in the runtime cache segment.
//
// These blobs are machine code tied to one GPU and driver, so unlike the
// shader cache there is no bundled tier and no cross-machine reuse; the entry
// key is the adapter the blob was built on, and the driver (or the backend's
// own header check) rejects a stale blob so the launch falls back to building
// pipelines cold. Every operation here is best-effort: an unreadable,
// oversized, or rejected entry is dropped and treated as absent, never
// surfaced as an init failure.
//
// A store lands in the segment held in memory and reaches disk at the next
// checkpoint, so the two serializations a session makes (end of init, then
// teardown) cost one write between them rather than one each.

use std::sync::atomic::{AtomicU64, Ordering};

use concinnity_core::blob::CacheEntryKind;

// Neither VkPipelineCache nor a D3D12 pipeline library evicts internally, so a
// long-lived checkout accumulates entries for every edited shader. Past this
// cap the entry is dropped and rebuilt cold rather than growing forever.
const FILE_BUDGET_BYTES: u64 = 32 * 1024 * 1024;

const KIND: CacheEntryKind = CacheEntryKind::Pipeline;

static CREATED: AtomicU64 = AtomicU64::new(0);
static CREATE_MICROS: AtomicU64 = AtomicU64::new(0);

// Record one pipeline creation for the init tally.
pub(crate) fn note_creation(micros: u64) {
    CREATED.fetch_add(1, Ordering::Relaxed);
    CREATE_MICROS.fetch_add(micros, Ordering::Relaxed);
}

// Log the pipeline-creation cost of a renderer init and whether the disk entry
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

// Read the persisted blob for `key`. An empty or over-budget entry is dropped
// and reads as absent.
pub(crate) fn load(key: &str) -> Option<Vec<u8>> {
    let bytes = crate::runtime_cache::load(KIND, key)?;
    if within_budget(&bytes) {
        Some(bytes)
    } else {
        delete(key);
        None
    }
}

// Hold `bytes` for `key` until the next checkpoint. The segment keeps whichever
// blob is larger, which is the growth check this needs: a driver cache only
// accumulates, but its serialization is not deterministic (MoltenVK shuffles
// entry order run to run), so a byte compare would rewrite an unchanged cache
// every launch. Returns whether the segment took them.
pub(crate) fn store(key: &str, bytes: &[u8]) -> bool {
    crate::runtime_cache::store(KIND, key, bytes)
}

// Drop the persisted blob for `key`; used when the driver rejects it.
pub(crate) fn delete(key: &str) {
    crate::runtime_cache::delete(KIND, key);
}

// Whether an entry is worth keeping: a truncated write reads back empty, and a
// cache past the budget is rebuilt cold instead of growing forever.
fn within_budget(bytes: &[u8]) -> bool {
    !bytes.is_empty() && bytes.len() as u64 <= FILE_BUDGET_BYTES
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn an_empty_or_over_budget_blob_is_not_kept() {
        assert!(within_budget(&[1]));
        assert!(within_budget(&vec![0u8; FILE_BUDGET_BYTES as usize]));
        assert!(!within_budget(&[]), "a truncated entry");
        assert!(!within_budget(&vec![0u8; FILE_BUDGET_BYTES as usize + 1]));
    }

    // Under `cargo test` nothing reaches the state dir, in either direction.
    #[test]
    fn the_cache_is_off_under_test() {
        assert!(!crate::runtime_cache::enabled());
        assert_eq!(load("vk-probe"), None);
        assert!(!store("vk-probe", &[1, 2, 3]));
    }
}
