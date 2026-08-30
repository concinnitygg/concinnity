//! The build cache: what a cook produced for its own later runs, all of it in
//! `cache/1`.
//!
//! Some assets are expensive to compile -- the EnvironmentMap IBL convolution
//! alone is hundreds of millions of float ops per build -- and a scene import
//! re-parses a source file that may run to gigabytes. Both are deterministic
//! functions of a small set of inputs, so the inputs are hashed into a key
//! (`key`) and the output is stored under it. A later build that produces the
//! same key reuses what is stored instead of doing the work again. The baked
//! asset previews ([`thumbnails`]) ride the same segment on the same terms:
//! cook renders them, so cook stores them, though the editor is what reads
//! them back.
//!
//! One file per writer role is the rule the layout is built on. A build writes
//! this segment and nothing else, the running application writes its own, so a
//! cook against a live editor never touches the file that editor is writing.
//!
//! The file is touched at the two moments the design allows and no others: the
//! index is read when the first lookup needs it, and the segment is replaced by
//! `flush` when the work producing it finishes. In between, a hit seeks to
//! the one entry it wants and a store lands in memory, so a compile that stores
//! for every asset costs one write rather than one per asset. That is also what
//! makes the concurrent-store race structurally impossible: nothing writes the
//! file while the compile is running.
//!
//! What produced an entry is not part of its key. The identity of the cook
//! binary rides the segment header instead (`identity`), so a segment an
//! older binary wrote is dropped whole rather than replayed against code that
//! moved.
//!
//! Every operation is best-effort: a miss, an unreadable segment, or a failed
//! write all leave the caller to compile normally, so the cache can never break
//! or corrupt a build. Deleting `cache/` at any point costs recomputation and
//! nothing else.

mod identity;
mod key;
mod segment;
pub mod thumbnails;

use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::{Arc, Mutex, MutexGuard};

pub(crate) use concinnity_core::blob::CacheEntryKind;
pub(crate) use key::{bake_key, expand_key, payload_key};

use segment::Index;

/// Read the entry `kind` stored under `key`, if the segment holds one.
pub(crate) fn load(kind: CacheEntryKind, key: &str) -> Option<Vec<u8>> {
    // Disabled under `cargo test` so the suite neither creates stray segments
    // nor lets a stale entry mask a change to a compile path. What a hit has to
    // reproduce is covered out of process instead, by driving the binary twice
    // over one world.
    if cfg!(test) {
        return None;
    }
    let index = {
        let mut held = lock();
        let loaded = open(&mut held)?;
        // A key stored earlier in this same build is served from memory: the
        // compile is parallel and two assets with identical inputs share a key,
        // so the second is a hit against bytes the file does not hold yet.
        if let Some(bytes) = loaded.stored.get(&(kind, key.to_owned())) {
            return Some(bytes.to_vec());
        }
        Arc::clone(&loaded.index)
    };
    index.get(kind, key)
}

/// Whether the segment already holds the entry `kind` stored under `key`,
/// without reading its bytes. What a producer asks before doing the work an
/// entry would save.
pub(crate) fn contains(kind: CacheEntryKind, key: &str) -> bool {
    if cfg!(test) {
        return false;
    }
    let mut held = lock();
    let Some(loaded) = open(&mut held) else {
        return false;
    };
    loaded.stored.contains_key(&(kind, key.to_owned())) || loaded.index.contains(kind, key)
}

/// Hold `bytes` as `key`'s entry until the next [`flush`].
pub(crate) fn store(kind: CacheEntryKind, key: &str, bytes: &[u8]) {
    if cfg!(test) {
        return;
    }
    let mut held = lock();
    let Some(loaded) = open(&mut held) else {
        return;
    };
    loaded
        .stored
        .insert((kind, key.to_owned()), bytes.to_vec().into());
}

/// Write the segment, carrying through what it already held, and report whether
/// the file was written. Called when the work producing entries finishes -- the
/// end of an expansion, the end of a compile -- never per entry.
///
/// A build that stored nothing writes nothing: the file it would produce is the
/// one already there.
pub(crate) fn flush() -> bool {
    let mut held = lock();
    let Some(loaded) = held.take() else {
        return false;
    };
    if loaded.stored.is_empty() {
        *held = Some(loaded);
        return false;
    }
    write(&loaded)
}

// The segment this process read, the file it came from, and what this build has
// produced for it.
struct Loaded {
    path: PathBuf,
    token: u32,
    index: Arc<Index>,
    stored: HashMap<(CacheEntryKind, String), Arc<[u8]>>,
}

// Replace `loaded`'s file with what it now holds: what the index still
// addresses, plus the entries this build produced.
fn write(loaded: &Loaded) -> bool {
    let mut stored: Vec<(CacheEntryKind, &str, &[u8])> = loaded
        .stored
        .iter()
        .map(|((kind, key), bytes)| (*kind, key.as_str(), &**bytes))
        .collect();
    // Hash iteration order must not reach the file: two builds that stored the
    // same entries write the same bytes.
    stored.sort_by(|a, b| (a.1, a.0 as u8).cmp(&(b.1, b.0 as u8)));
    segment::write(&loaded.path, &loaded.index, &stored, loaded.token)
}

// The segment for the installed state root, reading its index on the first call.
// `None` when no host installed a state root, or when the running binary cannot
// be identified; both turn every operation above into a miss, so payloads are
// compiled fresh rather than warmed from a segment nothing can invalidate.
fn open<'a>(held: &'a mut MutexGuard<'static, Option<Loaded>>) -> Option<&'a mut Loaded> {
    let path = crate::paths::build_cache_path()?;
    let token = identity::token()?;
    if held.as_ref().is_some_and(|loaded| loaded.path != path) {
        // A host moved the state root after this segment was read (a world's
        // own `home` overriding the CLI's). What this build produced belongs to
        // the old root, so write it back there before reading the new one.
        if let Some(previous) = held.take() {
            write(&previous);
        }
    }
    Some(held.get_or_insert_with(|| Loaded {
        index: Arc::new(Index::read(&path, token)),
        path,
        token,
        stored: HashMap::new(),
    }))
}

// Serializes this process's access to the one segment it holds. Held only
// across the index lookup and the store, never across the file read a hit does.
fn lock() -> MutexGuard<'static, Option<Loaded>> {
    static LOADED: Mutex<Option<Loaded>> = Mutex::new(None);
    LOADED.lock().unwrap_or_else(|e| e.into_inner())
}
