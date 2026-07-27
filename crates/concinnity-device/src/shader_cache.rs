// Content-addressed cache for the engine's built-in shader binaries.
//
// The DirectX and Vulkan backends compile every built-in shader from embedded
// source at renderer init, and that compile dominates startup: 993 ms of a
// 1.58 s release init on DirectX (45 FXC invocations), and 369 ms on Vulkan (53
// shaderc invocations). The output is a pure function of the source text, the
// entry point, the compile target, and the compiler options, none of which
// change between runs of an unedited binary -- so the second run of a given
// build has no reason to compile anything.
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
// all fall back to compiling normally, so the cache can never break a run. Set
// `CN_SHADER_CACHE=0` to bypass it entirely.

use sha2::{Digest, Sha256};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};

// Bump when a stored artifact's meaning changes without a corresponding change
// to the hashed inputs -- a new compiler toolchain whose output differs for
// identical source, or a change to what `cached` stores. A bump orphans every
// existing entry, which `prune` then reclaims.
const CACHE_FORMAT_VERSION: u32 = 1;

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
        h.update(CACHE_FORMAT_VERSION.to_le_bytes());
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

fn enabled() -> bool {
    // Off under `cargo test` so the suite neither writes into a developer's
    // state dir nor lets an entry from a previous run mask a compile change.
    !cfg!(test) && std::env::var("CN_SHADER_CACHE").as_deref() != Ok("0")
}

fn cache_dir() -> PathBuf {
    concinnity_core::paths::shader_cache_dir()
}

fn load(digest: &str) -> Option<Vec<u8>> {
    let bytes = std::fs::read(cache_dir().join(digest)).ok()?;
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
            if !meta.is_file() {
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
}
