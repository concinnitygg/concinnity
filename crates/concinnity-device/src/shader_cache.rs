// Content-addressed cache for the shader binaries compiled after build time.
//
// The DirectX and Vulkan backends compile every built-in shader from embedded
// source at renderer init, and that compile dominates startup: 993 ms of a
// 1.58 s release init on DirectX (45 FXC invocations), and 369 ms on Vulkan (53
// shaderc invocations). Metal precompiles its built-ins into the binary but
// assembles the raymarch libraries around world-authored SdfVolume fragments
// at init, and caches those metallibs here (see `metal::msl_cache`). The
// output is a pure function of the source text, the entry point, the compile
// target, and the compiler options, none of which change between runs of an
// unedited binary -- so the second run of a given build has no reason to
// compile anything.
//
// Each artifact is stored under the hex digest of those inputs, which makes the
// entry self-validating: a shader edit, a flag change, or a debug/release switch
// all produce a different key and simply miss rather than replaying stale bytes.
// Keying on the *assembled* source is what lets this cover the runtime-templated
// shaders (`{POOL_SIZE}`, `{MAX_PROBES}`, the probe_common injection, the
// `CULL_PHASE2` / `SHADOW_CULL` variants) that a build-time table would have had
// to enumerate by hand.
//
// Every operation is best-effort: a miss, an unreadable entry, or a failed write
// all fall back to compiling normally, so the cache can never break a run.
// Deleting the directory is the way to force a full recompile, and is what a
// host toolchain upgrade whose output differs for identical source wants.

use sha2::{Digest, Sha256};
use std::path::{Path, PathBuf};
use std::sync::OnceLock;
use std::sync::atomic::{AtomicU64, Ordering};

// SHADER_COMPILE_SOURCE_HASH: derived by build.rs from the modules that decide
// how an artifact is produced, so a change to a compiler invocation or to what
// `cached` stores orphans every entry it would otherwise serve stale. `prune`
// reclaims the orphans.
include!(concat!(env!("OUT_DIR"), "/shader_compile_source_hash.rs"));

// Keep the cache from growing without bound: every shader edit orphans the
// previous artifact, and a long-lived checkout would otherwise accumulate them
// forever. Generous next to the ~100 live entries a single build needs.
const CACHE_BUDGET_BYTES: u64 = 64 * 1024 * 1024;

// The inputs a compiled shader artifact is a function of. `compiler` separates
// the toolchains (FXC's DXBC must never be served to a Vulkan build); `options`
// carries whatever flag word or option discriminator the caller's compiler takes.
pub(crate) struct Key<'a> {
    pub compiler: &'a str,
    pub source: &'a str,
    pub entry: &'a str,
    pub target: &'a str,
    pub options: u64,
}

impl Key<'_> {
    // Hex SHA-256 over every field, each length-prefixed so no two distinct
    // key tuples can concatenate to the same byte stream.
    fn digest(&self) -> String {
        let mut h = Sha256::new();
        h.update(SHADER_COMPILE_SOURCE_HASH.to_le_bytes());
        h.update(concinnity_slang::SOURCE_HASH.to_le_bytes());
        for part in [self.compiler, self.source, self.entry, self.target] {
            h.update((part.len() as u64).to_le_bytes());
            h.update(part.as_bytes());
        }
        h.update(self.options.to_le_bytes());
        format!("{:x}", h.finalize())
    }
}

static HITS: AtomicU64 = AtomicU64::new(0);
static MISSES: AtomicU64 = AtomicU64::new(0);
static COMPILE_MICROS: AtomicU64 = AtomicU64::new(0);

// Return the cached artifact for `key`, else run `compile`, store the result,
// and return it. `label` names the shader in the miss log only.
pub(crate) fn cached(
    key: &Key<'_>,
    label: &str,
    compile: impl FnOnce() -> Result<Vec<u8>, String>,
) -> Result<Vec<u8>, String> {
    if !enabled() {
        return compile();
    }
    verify_toolchain();
    let digest = key.digest();
    if let Some(bytes) = load(&digest) {
        HITS.fetch_add(1, Ordering::Relaxed);
        return Ok(bytes);
    }
    let started = std::time::Instant::now();
    let bytes = compile()?;
    let micros = started.elapsed().as_micros() as u64;
    MISSES.fetch_add(1, Ordering::Relaxed);
    COMPILE_MICROS.fetch_add(micros, Ordering::Relaxed);
    tracing::debug!(
        "shader cache miss: {} {} ({:.1} ms)",
        key.entry,
        label,
        micros as f64 / 1000.0
    );
    store(&digest, &bytes);
    Ok(bytes)
}

// How `ensure_in` satisfied a request: the artifact was already in the target
// directory, was copied over from this machine's local cache tiers, or had to
// be compiled fresh.
#[cfg(any(backend_dx, backend_vk))]
#[derive(Debug, PartialEq, Eq)]
pub(crate) enum Ensured {
    Present,
    Copied,
    Compiled,
}

// Make sure the artifact for `key` exists in `dir` (a bundle's shader-cache/),
// compiling only when neither `dir` nor the local cache tiers already hold it.
// A fresh compile is also stored locally, so repeated exports stay warm. Used
// by the export-time precompile; the runtime path stays on `cached`.
#[cfg(any(backend_dx, backend_vk))]
pub(crate) fn ensure_in(
    dir: &Path,
    key: &Key<'_>,
    compile: impl FnOnce() -> Result<Vec<u8>, String>,
) -> Result<Ensured, String> {
    if enabled() {
        verify_toolchain();
    }
    let digest = key.digest();
    if load_in(dir, &digest).is_some() {
        return Ok(Ensured::Present);
    }
    if enabled()
        && let Some(bytes) = load(&digest)
    {
        store_in(dir, &digest, &bytes);
        return Ok(Ensured::Copied);
    }
    let bytes = compile()?;
    if bytes.is_empty() {
        return Err("compile produced an empty artifact".to_string());
    }
    store_in(dir, &digest, &bytes);
    if enabled() {
        store(&digest, &bytes);
    }
    Ok(Ensured::Compiled)
}

// Log what the cache did during a renderer init, and reclaim orphaned entries.
// Called once per backend init. Shaders built lazily after it (the skinned-mesh
// pipelines on first upload, a world shader bucket on scene pin) are cached the
// same way but land after this tally, so it is a snapshot rather than a total.
pub(crate) fn report_init_and_prune() {
    let (hits, misses, micros) = (
        HITS.load(Ordering::Relaxed),
        MISSES.load(Ordering::Relaxed),
        COMPILE_MICROS.load(Ordering::Relaxed),
    );
    if hits + misses == 0 {
        return;
    }
    tracing::info!(
        "shader cache: {hits} reused, {misses} compiled ({:.0} ms) at renderer init",
        micros as f64 / 1000.0
    );
    if misses > 0 {
        prune(&cache_dir(), CACHE_BUDGET_BYTES);
    }
}

// Off under `cargo test` so the suite neither writes into a developer's state dir
// nor lets an entry from a previous run mask a compile change.
fn enabled() -> bool {
    !cfg!(test)
}

fn cache_dir() -> PathBuf {
    concinnity_store::paths::shader_cache_dir()
}

// Scratch directory for runtime slangc invocations (the compiler works on
// files; source and artifact are removed after each compile). Lives beside
// the cache so it stays inside the engine's state root. Entries in `prune`'s
// walk are filtered to files, so the subdirectory never confuses eviction.
pub(crate) fn slang_work_dir() -> PathBuf {
    cache_dir().join("slang-work")
}

// Names the shader toolchains this directory's entries were produced by. Not a
// digest, so it can never collide with one.
const TOOLCHAIN_STAMP: &str = "toolchain";

// Discard the writable tier when the shader toolchain changes. An entry is a
// function of its source, not of what compiled it, and slangc is an external
// binary that can be upgraded -- or shadowed by another install earlier on
// PATH -- without a byte of source moving; without this, that upgrade never
// takes effect and the old compiler's output is replayed forever. The
// read-only tier is deliberately left alone: a bundle ships it on purpose, and
// it is what a host with no compiler of its own has to run from.
//
// Costs one `slangc -version` per process, which is why it is a `OnceLock`.
fn verify_toolchain() {
    static ONCE: OnceLock<()> = OnceLock::new();
    ONCE.get_or_init(|| {
        let dir = cache_dir();
        let current = concinnity_slang::compiler_id();
        let stamp = dir.join(TOOLCHAIN_STAMP);
        if std::fs::read_to_string(&stamp).is_ok_and(|found| found == current) {
            return;
        }
        discard_entries(&dir);
        if std::fs::create_dir_all(&dir).is_ok() {
            let _ = std::fs::write(&stamp, current);
        }
    });
}

// Drop every artifact in `dir`, leaving subdirectories (the slang work dir) be.
fn discard_entries(dir: &Path) {
    let Ok(entries) = std::fs::read_dir(dir) else {
        return;
    };
    for entry in entries.flatten() {
        if entry.metadata().is_ok_and(|meta| meta.is_file()) {
            let _ = std::fs::remove_file(entry.path());
        }
    }
}

fn load(digest: &str) -> Option<Vec<u8>> {
    dirs_to_read(
        cache_dir(),
        concinnity_store::paths::bundled_shader_cache_dir(),
    )
    .iter()
    .find_map(|dir| load_in(dir, digest))
}

// Directories to search, in order: the writable dir this process stores into,
// then the read-only tier a bundle ships. Split from `load` so the de-duplication
// is unit-testable: the two resolve to the same path unless a read-only install
// redirected writable state, and searching it twice would be wasted syscalls.
fn dirs_to_read(writable: PathBuf, bundled: PathBuf) -> Vec<PathBuf> {
    if bundled == writable {
        vec![writable]
    } else {
        vec![writable, bundled]
    }
}

fn load_in(dir: &Path, digest: &str) -> Option<Vec<u8>> {
    let bytes = std::fs::read(dir.join(digest)).ok()?;
    // A zero-length artifact is never a legitimate compile result; treat it as
    // a miss so a truncated entry recompiles instead of failing pipeline
    // creation with an empty bytecode blob.
    (!bytes.is_empty()).then_some(bytes)
}

fn store(digest: &str, bytes: &[u8]) {
    store_in(&cache_dir(), digest, bytes);
}

// Write via a process-unique temp file and rename, so a concurrent reader (a
// second `cn debug` on the same checkout) never observes a partial artifact.
fn store_in(dir: &Path, digest: &str, bytes: &[u8]) {
    if std::fs::create_dir_all(dir).is_err() {
        return;
    }
    let tmp = dir.join(format!("{digest}.{}.tmp", std::process::id()));
    if std::fs::write(&tmp, bytes).is_ok() && std::fs::rename(&tmp, dir.join(digest)).is_err() {
        let _ = std::fs::remove_file(&tmp);
    }
}

// Delete oldest-first until the directory fits `budget`. Content-addressed
// entries are interchangeable, so evicting the least recently written one is
// both safe and a decent proxy for least recently useful.
fn prune(dir: &Path, budget: u64) {
    let Ok(entries) = std::fs::read_dir(dir) else {
        return;
    };
    let mut listing: Vec<(std::time::SystemTime, u64, PathBuf)> = entries
        .flatten()
        .filter_map(|e| {
            let meta = e.metadata().ok()?;
            // The stamp is not an artifact, and evicting it would make the next
            // run read the directory as another toolchain's and discard it all.
            if !meta.is_file() || e.file_name() == TOOLCHAIN_STAMP {
                return None;
            }
            Some((meta.modified().ok()?, meta.len(), e.path()))
        })
        .collect();
    for path in evictions(&mut listing, budget) {
        let _ = std::fs::remove_file(path);
    }
}

// Oldest entries to drop so the total fits `budget`. Split from the filesystem
// walk so the policy is unit-testable.
fn evictions(
    listing: &mut [(std::time::SystemTime, u64, PathBuf)],
    budget: u64,
) -> Vec<std::path::PathBuf> {
    let mut total: u64 = listing.iter().map(|(_, len, _)| len).sum();
    if total <= budget {
        return Vec::new();
    }
    listing.sort_by_key(|(modified, _, _)| *modified);
    let mut doomed = Vec::new();
    for (_, len, path) in listing.iter() {
        if total <= budget {
            break;
        }
        total -= len;
        doomed.push(path.clone());
    }
    doomed
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::{Duration, SystemTime};

    fn key<'a>(source: &'a str, entry: &'a str, target: &'a str, options: u64) -> Key<'a> {
        Key {
            compiler: "fxc",
            source,
            entry,
            target,
            options,
        }
    }

    #[test]
    fn digest_is_stable_for_identical_inputs() {
        let a = key("float4 main() { return 0; }", "main", "ps_5_1", 7);
        let b = key("float4 main() { return 0; }", "main", "ps_5_1", 7);
        assert_eq!(a.digest(), b.digest());
    }

    #[test]
    fn every_field_changes_the_digest() {
        let base = key("src", "main", "ps_5_1", 1).digest();
        assert_ne!(base, key("other", "main", "ps_5_1", 1).digest(), "source");
        assert_ne!(base, key("src", "main2", "ps_5_1", 1).digest(), "entry");
        assert_ne!(base, key("src", "main", "vs_5_1", 1).digest(), "target");
        assert_ne!(base, key("src", "main", "ps_5_1", 2).digest(), "options");
        let mut other_compiler = key("src", "main", "ps_5_1", 1);
        other_compiler.compiler = "glsl";
        assert_ne!(base, other_compiler.digest(), "compiler");
    }

    // Length prefixes must keep adjacent fields from running together: without
    // them ("ab", "c") and ("a", "bc") would hash alike.
    #[test]
    fn field_boundaries_cannot_be_confused() {
        assert_ne!(
            key("ab", "c", "t", 0).digest(),
            key("a", "bc", "t", 0).digest()
        );
    }

    #[test]
    fn store_then_load_round_trips_through_a_directory() {
        let dir = std::env::temp_dir().join(format!("cn_shader_cache_{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        store_in(&dir, "deadbeef", &[1, 2, 3, 4]);
        assert_eq!(
            std::fs::read(dir.join("deadbeef")).ok(),
            Some(vec![1, 2, 3, 4])
        );
        // Temp files must not survive a successful store.
        let leftovers = std::fs::read_dir(&dir)
            .unwrap()
            .flatten()
            .filter(|e| e.path().extension().is_some_and(|x| x == "tmp"))
            .count();
        assert_eq!(leftovers, 0);
        std::fs::remove_dir_all(&dir).unwrap();
    }

    // A toolchain change drops the entries but not the slang work directory, so
    // a compile already writing into it is not pulled out from under itself.
    #[test]
    fn discarding_entries_spares_subdirectories() {
        let tmp = std::env::temp_dir().join(format!("cn_sc_discard_{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&tmp);
        store_in(&tmp, "cafe", &[1, 2]);
        store_in(&tmp, "f00d", &[3, 4]);
        let work = tmp.join("slang-work");
        std::fs::create_dir_all(&work).unwrap();
        std::fs::write(work.join("in-flight.slang"), "x").unwrap();

        discard_entries(&tmp);

        assert!(load_in(&tmp, "cafe").is_none());
        assert!(load_in(&tmp, "f00d").is_none());
        assert!(work.join("in-flight.slang").exists());
        std::fs::remove_dir_all(&tmp).unwrap();
    }

    // Evicting the stamp would make the next run read the directory as another
    // toolchain's and throw away everything the eviction just made room for.
    #[test]
    fn pruning_never_evicts_the_toolchain_stamp() {
        let tmp = std::env::temp_dir().join(format!("cn_sc_stamp_{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&tmp);
        std::fs::create_dir_all(&tmp).unwrap();
        // Oldest file in the directory, so an LRU walk would reach it first.
        std::fs::write(tmp.join(TOOLCHAIN_STAMP), "slang 2026.1").unwrap();
        store_in(&tmp, "cafe", &[0u8; 512]);

        prune(&tmp, 16);

        assert!(tmp.join(TOOLCHAIN_STAMP).exists());
        assert!(load_in(&tmp, "cafe").is_none(), "artifact should evict");
        std::fs::remove_dir_all(&tmp).unwrap();
    }

    #[test]
    fn a_shared_path_is_searched_once() {
        let p = PathBuf::from("/state/shader-cache");
        assert_eq!(dirs_to_read(p.clone(), p.clone()), vec![p]);
    }

    #[test]
    fn a_read_only_install_searches_writable_then_bundled() {
        let writable = PathBuf::from("/user/appdata/shader-cache");
        let bundled = PathBuf::from("/program files/game/shader-cache");
        assert_eq!(
            dirs_to_read(writable.clone(), bundled.clone()),
            vec![writable, bundled]
        );
    }

    // A bundle ships its artifacts read-only; a player's first launch must find
    // them there even though nothing has been written to the writable dir yet.
    #[test]
    fn an_artifact_is_found_in_the_bundled_tier() {
        let tmp = std::env::temp_dir().join(format!("cn_sc_tiers_{}", std::process::id()));
        let writable = tmp.join("writable");
        let bundled = tmp.join("bundled");
        let _ = std::fs::remove_dir_all(&tmp);
        store_in(&bundled, "cafe", &[9, 9]);

        let dirs = dirs_to_read(writable.clone(), bundled.clone());
        assert_eq!(
            dirs.iter().find_map(|d| load_in(d, "cafe")),
            Some(vec![9, 9])
        );
        // The writable tier wins when both hold the key, so a locally recompiled
        // artifact is never shadowed by a stale bundled one.
        store_in(&writable, "cafe", &[1, 1]);
        assert_eq!(
            dirs.iter().find_map(|d| load_in(d, "cafe")),
            Some(vec![1, 1])
        );
        std::fs::remove_dir_all(&tmp).unwrap();
    }

    #[test]
    fn a_truncated_artifact_reads_as_a_miss() {
        let tmp = std::env::temp_dir().join(format!("cn_sc_trunc_{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&tmp);
        store_in(&tmp, "empty", &[]);
        assert_eq!(load_in(&tmp, "empty"), None);
        std::fs::remove_dir_all(&tmp).unwrap();
    }

    fn entry(secs: u64, len: u64, name: &str) -> (SystemTime, u64, PathBuf) {
        (
            SystemTime::UNIX_EPOCH + Duration::from_secs(secs),
            len,
            PathBuf::from(name),
        )
    }

    #[test]
    fn nothing_is_evicted_under_budget() {
        let mut listing = [entry(1, 10, "a"), entry(2, 10, "b")];
        assert!(evictions(&mut listing, 100).is_empty());
    }

    #[test]
    fn eviction_drops_oldest_first_until_it_fits() {
        let mut listing = [
            entry(3, 40, "newest"),
            entry(1, 40, "oldest"),
            entry(2, 40, "middle"),
        ];
        let doomed = evictions(&mut listing, 80);
        assert_eq!(doomed, [PathBuf::from("oldest")]);
    }

    #[test]
    fn eviction_can_clear_everything_for_a_zero_budget() {
        let mut listing = [entry(1, 40, "a"), entry(2, 40, "b")];
        assert_eq!(evictions(&mut listing, 0).len(), 2);
    }

    #[cfg(any(backend_dx, backend_vk))]
    #[test]
    fn ensure_in_compiles_once_then_finds_the_artifact_present() {
        let dir = std::env::temp_dir().join(format!("cn_sc_ensure_{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        let k = key("ensure src", "main", "ps_5_1", 3);

        let first = ensure_in(&dir, &k, || Ok(vec![7, 7, 7])).unwrap();
        assert_eq!(first, Ensured::Compiled);
        assert_eq!(load_in(&dir, &k.digest()), Some(vec![7, 7, 7]));

        // The second request must be served from `dir` without recompiling.
        let second = ensure_in(&dir, &k, || panic!("must not recompile")).unwrap();
        assert_eq!(second, Ensured::Present);
        std::fs::remove_dir_all(&dir).unwrap();
    }

    #[cfg(any(backend_dx, backend_vk))]
    #[test]
    fn ensure_in_propagates_a_compile_error_and_stores_nothing() {
        let dir = std::env::temp_dir().join(format!("cn_sc_ensure_err_{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        let k = key("bad src", "main", "ps_5_1", 0);
        assert!(ensure_in(&dir, &k, || Err("boom".to_string())).is_err());
        assert!(ensure_in(&dir, &k, || Ok(Vec::new())).is_err(), "empty");
        assert_eq!(load_in(&dir, &k.digest()), None);
        let _ = std::fs::remove_dir_all(&dir);
    }
}
