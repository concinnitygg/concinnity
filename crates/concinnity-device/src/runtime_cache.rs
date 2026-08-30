// The runtime cache segment (`cache/0`) as this crate sees it: the compiled
// shader binaries and driver pipeline blobs a renderer init produces, the
// moments they reach disk, and the read-only copy of the same segment a bundle
// ships.
//
// The segment is read once, on the first lookup, and written at a checkpoint.
// Nothing here writes per entry: an init that compiles fifty shaders stores
// fifty times into memory and touches the file once, which is what keeps the
// cache cheaper than the compiles it saves.
//
// Both producers route through this module rather than through
// `concinnity_host` directly, so the cargo-test opt-out sits in one place:
// under test nothing is read and nothing is written, which keeps the suite out
// of a developer's state directory and keeps a previous run's artifact from
// masking a compile change.

use concinnity_core::blob::CacheEntryKind;
use concinnity_host::store::cache;

// Off under `cargo test`, where every operation below is a no-op.
pub(crate) fn enabled() -> bool {
    !cfg!(test)
}

// The bytes stored for `key`, or `None` when the segment holds no such entry.
pub(crate) fn load(kind: CacheEntryKind, key: &str) -> Option<Vec<u8>> {
    enabled().then(|| cache::load(kind, key))?
}

// The same lookup against the read-only segment a bundle ships, for a caller
// `load` missed. Read once, like the writable one: a lookup here parses
// nothing, so an init consulting it fifty times still costs one file read.
pub(crate) fn load_bundled(kind: CacheEntryKind, key: &str) -> Option<Vec<u8>> {
    enabled().then(|| cache::load_bundled(kind, key))?
}

// Hold `bytes` under `key` until the next checkpoint, reporting whether the
// segment took them.
pub(crate) fn store(kind: CacheEntryKind, key: &str, bytes: &[u8]) -> bool {
    enabled() && cache::store(kind, key, bytes)
}

// Drop `key`'s entry, for a caller whose artifact turned out unusable. Only the
// pipeline blobs have one: a shader artifact is content-addressed, so a bad
// entry is a bad key rather than one to withdraw.
#[cfg(any(backend_dx, backend_vk))]
pub(crate) fn delete(kind: CacheEntryKind, key: &str) {
    if enabled() {
        cache::delete(kind, key);
    }
}

// Adopt `id` as the host shader toolchain the segment's entries were produced
// by, reporting whether entries another toolchain produced were discarded.
pub(crate) fn verify_toolchain(id: &str) -> bool {
    enabled() && cache::verify_toolchain(id)
}

// Write everything produced since the last checkpoint. Called at the end of a
// renderer init, so a crash later in the session cannot lose the warm-up, and
// again at teardown for what was built lazily after it.
pub(crate) fn checkpoint() {
    if enabled() {
        cache::flush();
    }
}
