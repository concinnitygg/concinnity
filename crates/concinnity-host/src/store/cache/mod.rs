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
//! Which files those are is not this module's business: a host [anchors] the
//! two it wants, having resolved them from its own state tree. Until one does,
//! every operation here is a miss.
//!
//! A lookup has a second tier behind it: the segment a bundle ships, which `cn
//! export` warms so a player's first launch does not pay the compile. A host
//! resolves it against the content root rather than the writable one, so the
//! two are one file except on an install that cannot write beside its data --
//! which is the whole reason the tier exists.
//!
//! [anchors]: anchor
//!
//! The file is touched twice: once when the first lookup reads it, and once per
//! [`flush`]. The bundled tier is read once and never written. Everything
//! between is memory, so a producer that stores in a loop costs one write
//! rather than one per entry. A crash before a flush costs the recompute of
//! whatever had not been written, which is the same price deleting `cache/`
//! already carries.
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

/// The same lookup against the read-only segment a bundle ships, for a caller
/// [`load`] missed. Read once like the writable one, so a run that consults it
/// fifty times reads the file once.
///
/// Reports a miss when the bundle is one the application can write to: both
/// roles then name one file, the writable tier already holds its entries, and
/// answering from a second copy would only risk [`flush`] writing back a view
/// that was never the shipped one.
pub fn load_bundled(kind: CacheEntryKind, key: &str) -> Option<Vec<u8>> {
    bundled(|segment| segment.get(kind, key).map(<[u8]>::to_vec)).flatten()
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

/// The two segment files a run consults, named by whatever anchored them.
///
/// A host builds this from its [`StateTree`](super::paths::StateTree) --
/// `runtime_cache_path` and `bundled_runtime_cache_path` -- so nothing here
/// knows what a cache path looks like. Process state because artifacts are
/// produced deep inside a renderer init, with no caller to carry the paths
/// down from.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CacheAnchor {
    writable: PathBuf,
    bundled: Option<PathBuf>,
}

impl CacheAnchor {
    /// An anchor naming the file this run writes.
    pub fn new<P: Into<PathBuf>>(writable: P) -> Self {
        Self {
            writable: writable.into(),
            bundled: None,
        }
    }

    /// Also read the read-only segment a bundle ships, for the tier behind a
    /// miss. Ignored when it names the very file this run writes.
    #[must_use]
    pub fn with_bundled<P: Into<PathBuf>>(mut self, bundled: P) -> Self {
        self.bundled = Some(bundled.into());
        self
    }
}

// The segment files this process was told about.
fn anchored() -> &'static Mutex<Option<CacheAnchor>> {
    static ANCHOR: OnceLock<Mutex<Option<CacheAnchor>>> = OnceLock::new();
    ANCHOR.get_or_init(|| Mutex::new(None))
}

/// Point the runtime cache at the files `anchor` names for the rest of the
/// process, or until another anchor replaces it. Until a host calls this every
/// operation above is a miss: artifacts are produced fresh rather than warmed
/// from disk.
pub fn anchor(anchor: CacheAnchor) {
    *anchored().lock().unwrap() = Some(anchor);
}

/// Drop the anchor, leaving the process with no cache to warm from.
pub fn clear_anchor() {
    *anchored().lock().unwrap() = None;
}

// The file this run writes, when one is anchored.
fn writable_path() -> Option<PathBuf> {
    anchored()
        .lock()
        .unwrap()
        .as_ref()
        .map(|a| a.writable.clone())
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

// Run `f` against the bundled segment, reading the file on the first call.
// Nothing writes this tier, so a root move just drops what was read rather
// than writing it back.
fn bundled<R>(f: impl FnOnce(&mut Segment) -> R) -> Option<R> {
    let held = anchored().lock().unwrap().clone()?;
    let path = shipped_path(held.bundled?, Some(held.writable))?;
    static LOADED: Mutex<Option<Loaded>> = Mutex::new(None);
    let mut held = LOADED.lock().unwrap_or_else(|e| e.into_inner());
    if held.as_ref().is_some_and(|loaded| loaded.path != path) {
        *held = None;
    }
    let loaded = held.get_or_insert_with(|| Loaded {
        segment: Segment::read_from(&path),
        path,
    });
    Some(f(&mut loaded.segment))
}

// `bundled` unless the application writes that same file, in which case the
// writable tier is already serving its entries and reading a second copy would
// only put a stale view in front of what `flush` writes back.
fn shipped_path(shipped: PathBuf, writable: Option<PathBuf>) -> Option<PathBuf> {
    (writable.as_deref() != Some(shipped.as_path())).then_some(shipped)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::Path;

    // The portable-folder case: one file in both roles, so only the writable
    // tier reads it and the shipped entries ride its flush back to disk.
    #[test]
    fn a_writable_bundle_has_no_second_tier() {
        let one = PathBuf::from("/bundle/cache/0");
        assert_eq!(shipped_path(one.clone(), Some(one.clone())), None);
    }

    // A read-only install: the two roots diverge, so the shipped segment is a
    // tier of its own.
    #[test]
    fn a_read_only_install_reads_the_shipped_segment() {
        let shipped = PathBuf::from("/opt/app/cache/0");
        let writable = PathBuf::from("/home/u/.local/share/app/cache/0");
        assert_eq!(
            shipped_path(shipped.clone(), Some(writable)),
            Some(shipped.clone())
        );
        // No writable root at all leaves the shipped one readable.
        assert_eq!(shipped_path(shipped.clone(), None), Some(shipped));
    }

    // The anchor is two named files and nothing more: it carries no layout, so
    // a host is free to point the two tiers at unrelated places.
    #[test]
    fn an_anchor_names_the_files_it_was_given() {
        let plain = CacheAnchor::new("/run/segment");
        assert_eq!(plain.writable, Path::new("/run/segment"));
        assert_eq!(plain.bundled, None);

        let tiered = CacheAnchor::new("/run/segment").with_bundled("/opt/shipped");
        assert_eq!(tiered.writable, Path::new("/run/segment"));
        assert_eq!(tiered.bundled.as_deref(), Some(Path::new("/opt/shipped")));
    }

    // The tree is what a host builds an anchor from, and the portable layout
    // (one folder) is the case where both tiers name one file.
    #[test]
    fn a_tree_builds_the_anchor_its_layout_implies() {
        let tree = super::super::paths::StateTree::at("/bundle");
        let anchor = CacheAnchor::new(tree.runtime_cache_path())
            .with_bundled(tree.bundled_runtime_cache_path());
        assert_eq!(
            shipped_path(anchor.bundled.clone().unwrap(), Some(anchor.writable)),
            None,
            "one file in both roles has no second tier"
        );
    }
}
