//! The runtime cache segment: regenerable artifacts the running application
//! writes for its own later launches, all of them in `cache/0`.
//!
//! One file per writer role is the rule the layout is built on. The application
//! writes this segment and nothing else, a build writes its own, so a build
//! running against a live editor never touches the file the editor is writing.
//! Within the segment an index keyed by producer and key separates the entries,
//! which is what lets two adapters of one producer share a file, and the shader
//! cache share it with the driver pipeline blobs.
//!
//! Naming that file is not this module's business: a host [anchors] the one it
//! wants, having resolved it from its own state tree. Until one does, every
//! operation here is a miss.
//!
//! [anchors]: anchor
//!
//! The file is touched twice: once when the first lookup reads it, and once per
//! [`flush`]. Everything between is memory, so a producer that stores in a loop
//! costs one write rather than one per entry. A crash before a flush costs the
//! recompute of whatever had not been written, which is the same price deleting
//! `cache/` already carries.
//!
//! Every operation is best-effort: a miss, an unreadable segment, or a failed
//! write leaves the caller to produce the artifact the slow way. Two
//! applications running against one checkout do share this file, and the later
//! flush wins; what the loser had cached is recomputed on its next launch.
//!
//! The container format is `concinnity_core::blob`, which is I/O-free; the file
//! reads and writes live in `segment`.

mod segment;

use std::path::PathBuf;
use std::sync::{Mutex, MutexGuard, OnceLock};

pub use concinnity_core::blob::CacheEntryKind;
pub use segment::Segment;

/// How much payload the segment may hold. Every shader edit orphans the
/// artifact it replaces and neither driver pipeline blob evicts internally, so
/// a long-lived checkout would otherwise accumulate forever. Generous next to
/// the ~100 live entries one build needs; [`flush`] evicts down to it.
pub const CACHE_BUDGET_BYTES: u64 = 64 * 1024 * 1024;

/// The bytes `kind` stored under `key`, or `None` when there is no such entry
/// (or nothing anchored a segment to look in).
pub fn load(kind: CacheEntryKind, key: &str) -> Option<Vec<u8>> {
    with(|segment| segment.get(kind, key).map(<[u8]>::to_vec)).flatten()
}

/// Hold `bytes` under `key` until the next [`flush`], reporting whether the
/// segment took them: an entry already holding at least as many bytes is left
/// alone, so a driver blob whose serialization only reshuffles does not make
/// the flush rewrite the file.
pub fn store(kind: CacheEntryKind, key: &str, bytes: &[u8]) -> bool {
    with(|segment| segment.put(kind, key, bytes)).unwrap_or(false)
}

/// Drop `key`'s entry, for a caller whose artifact turned out unusable.
pub fn delete(kind: CacheEntryKind, key: &str) {
    with(|segment| segment.remove(kind, key));
}

/// Adopt `id` as the host shader toolchain the segment's entries were produced
/// by, discarding every entry when the segment names another one. Reports
/// whether it discarded, which the caller logs.
///
/// An artifact is a function of its source, not of what compiled it, so an
/// external compiler upgrade (or one shadowed by another install earlier on
/// PATH) moves no key: without this its predecessor's output would be replayed
/// forever.
pub fn verify_toolchain(id: &str) -> bool {
    with(|segment| segment.adopt_toolchain(id)).unwrap_or(false)
}

/// Write the segment to disk, if anything changed it since it was read, and
/// report whether the file was written. Called when the work producing entries
/// finishes -- the end of a renderer init, a clean shutdown -- never per entry.
pub fn flush() -> bool {
    match lock().as_mut() {
        Some(loaded) => loaded.segment.write_to(&loaded.path, CACHE_BUDGET_BYTES),
        None => false,
    }
}

// The segment file this process was told about. A host resolves it from its own
// [`StateTree`](super::paths::StateTree), so nothing here knows what a cache
// path looks like. Process state because artifacts are produced deep inside a
// renderer init, with no caller to carry the path down from.
fn anchored() -> &'static Mutex<Option<PathBuf>> {
    static ANCHOR: OnceLock<Mutex<Option<PathBuf>>> = OnceLock::new();
    ANCHOR.get_or_init(|| Mutex::new(None))
}

/// Point the runtime cache at `path` for the rest of the process, or until
/// another anchor replaces it. Until a host calls this every operation above is
/// a miss: artifacts are produced fresh rather than warmed from disk.
pub fn anchor<P: Into<PathBuf>>(path: P) {
    *anchored().lock().unwrap() = Some(path.into());
}

/// Drop the anchor, leaving the process with no cache to warm from. The segment
/// read under it is written back to its own file first, so a later [`flush`]
/// has nothing left to write.
pub fn clear_anchor() {
    *anchored().lock().unwrap() = None;
    if let Some(mut previous) = lock().take() {
        previous
            .segment
            .write_to(&previous.path, CACHE_BUDGET_BYTES);
    }
}

// The file this run writes, when one is anchored.
fn writable_path() -> Option<PathBuf> {
    anchored().lock().unwrap().clone()
}

// The segment this process read, and the file it came from.
struct Loaded {
    path: PathBuf,
    segment: Segment,
}

// Run `f` against the loaded segment, reading the file on the first call.
// `None` when nothing anchored the cache, which turns every operation above
// into a miss: artifacts are produced fresh rather than warmed from disk.
fn with<R>(f: impl FnOnce(&mut Segment) -> R) -> Option<R> {
    let path = writable_path()?;
    let mut held = lock();
    if held.as_ref().is_some_and(|loaded| loaded.path != path)
        && let Some(mut previous) = held.take()
    {
        // A host moved the writable state root after this segment was read (a
        // world's own `home` overriding the launcher's). What it holds belongs
        // to the old root, so write it back there before reading the new one.
        previous
            .segment
            .write_to(&previous.path, CACHE_BUDGET_BYTES);
    }
    let loaded = held.get_or_insert_with(|| Loaded {
        segment: Segment::read_from(&path),
        path,
    });
    Some(f(&mut loaded.segment))
}

// Serializes this process's access to the one segment it holds. Lookups take it
// too: a lookup marks the entry it found as one this run needs, so eviction
// spares it.
fn lock() -> MutexGuard<'static, Option<Loaded>> {
    static LOADED: Mutex<Option<Loaded>> = Mutex::new(None);
    LOADED.lock().unwrap_or_else(|e| e.into_inner())
}

#[cfg(test)]
mod tests {
    use super::*;

    // Dropping the anchor writes back what was read under it, and leaves
    // nothing for a later flush or store to reach.
    #[test]
    fn clearing_the_anchor_writes_the_segment_back_and_detaches_it() {
        // The anchor and the segment behind it are process state, so the two
        // tests that move them cannot run beside each other.
        let _guard = concinnity_testing::exclusive();
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("0");
        anchor(path.clone());
        assert!(store(CacheEntryKind::Shader, "k", b"v"));

        clear_anchor();
        assert_eq!(
            Segment::read_from(&path).get(CacheEntryKind::Shader, "k"),
            Some(&b"v"[..])
        );
        assert!(!flush(), "nothing is left to write");
        assert!(!store(CacheEntryKind::Shader, "k", b"v"));
    }

    // An anchor is one named file and nothing more: a host is free to point it
    // anywhere, and what it reads under one root does not follow it to another.
    #[test]
    fn a_moved_anchor_writes_the_old_file_back_before_reading_the_new_one() {
        let _guard = concinnity_testing::exclusive();
        let dir = tempfile::tempdir().unwrap();
        let (first, second) = (dir.path().join("a"), dir.path().join("b"));
        anchor(first.clone());
        assert!(store(CacheEntryKind::Shader, "k", b"v"));

        anchor(second.clone());
        assert_eq!(load(CacheEntryKind::Shader, "k"), None);
        assert_eq!(
            Segment::read_from(&first).get(CacheEntryKind::Shader, "k"),
            Some(&b"v"[..])
        );
        clear_anchor();
    }
}
