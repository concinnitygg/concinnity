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
// Artifacts live in the runtime cache segment, so an init that misses fifty
// times writes one file at its checkpoint rather than fifty as it goes -- see
// `crate::runtime_cache`. A bundle ships a segment of the same kind, warmed by
// `cn export` and read after the writable one; both tiers are read once, so a
// lookup in either is a memory lookup.
//
// Every operation is best-effort: a miss, an unreadable entry, or a failed write
// all fall back to compiling normally, so the cache can never break a run.
// Deleting `cache/` is the way to force a full recompile, and is what a host
// toolchain upgrade whose output differs for identical source wants.

use concinnity_core::blob::CacheEntryKind;
#[cfg(any(backend_dx, backend_vk))]
use concinnity_host::store::cache::Segment;
use sha2::{Digest, Sha256};
use std::sync::OnceLock;
use std::sync::atomic::{AtomicU64, Ordering};

// SHADER_COMPILE_SOURCE_HASH: derived by build.rs from the modules that decide
// how an artifact is produced, so a change to a compiler invocation or to what
// `cached` stores orphans every entry it would otherwise serve stale. The
// segment's own budget reclaims the orphans.
include!(concat!(env!("OUT_DIR"), "/shader_compile_source_hash.rs"));

const KIND: CacheEntryKind = CacheEntryKind::Shader;

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
// segment, was copied over from this machine's own cache tiers, or had to be
// compiled fresh.
#[cfg(any(backend_dx, backend_vk))]
#[derive(Debug, PartialEq, Eq)]
pub(crate) enum Ensured {
    Present,
    Copied,
    Compiled,
}

// Make sure the artifact for `key` is in `bundle` (the segment `cn export`
// ships), compiling only when neither it nor this machine's own cache tiers
// already hold it. A fresh compile is also stored locally, so repeated exports
// stay warm. `bundle` accumulates in memory and its caller writes it once, so
// warming a hundred artifacts costs one file write.
//
// Used by the export-time precompile; the runtime path stays on `cached`.
#[cfg(any(backend_dx, backend_vk))]
pub(crate) fn ensure_in(
    bundle: &mut Segment,
    key: &Key<'_>,
    compile: impl FnOnce() -> Result<Vec<u8>, String>,
) -> Result<Ensured, String> {
    if enabled() {
        verify_toolchain();
    }
    let digest = key.digest();
    if bundle.get(KIND, &digest).is_some_and(|b| !b.is_empty()) {
        return Ok(Ensured::Present);
    }
    if enabled()
        && let Some(bytes) = load(&digest)
    {
        bundle.put(KIND, &digest, &bytes);
        return Ok(Ensured::Copied);
    }
    let bytes = compile()?;
    if bytes.is_empty() {
        return Err("compile produced an empty artifact".to_string());
    }
    bundle.put(KIND, &digest, &bytes);
    if enabled() {
        store(&digest, &bytes);
    }
    Ok(Ensured::Compiled)
}

// Log what the cache did during a renderer init. Called once per backend init.
// Shaders built lazily after it (the skinned-mesh pipelines on first upload, a
// world shader bucket on scene pin) are cached the same way but land after this
// tally, so it is a snapshot rather than a total.
pub(crate) fn report_init() {
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
}

// Off under `cargo test` so the suite neither writes into a developer's state dir
// nor lets an entry from a previous run mask a compile change.
fn enabled() -> bool {
    crate::runtime_cache::enabled()
}

// Discard the segment's artifacts when the shader toolchain changes. An entry
// is a function of its source, not of what compiled it, and slangc is an
// external binary that can be upgraded -- or shadowed by another install
// earlier on PATH -- without a byte of source moving; without this, that
// upgrade never takes effect and the old compiler's output is replayed forever.
// The bundled tier is deliberately left alone: a bundle ships it on purpose,
// and it is what a host with no compiler of its own has to run from. `cn
// export` leaves the segment it warms unstamped for the same reason -- on a
// writable bundle that file is also the one this stamps, and a player whose
// slangc differs from the exporter's must keep the shipped artifacts rather
// than discard them on its first launch.
//
// Costs one `slangc -version` per process, which is why it is a `OnceLock`.
fn verify_toolchain() {
    static ONCE: OnceLock<()> = OnceLock::new();
    ONCE.get_or_init(|| {
        let current = concinnity_slang::compiler_id();
        if crate::runtime_cache::verify_toolchain(current) {
            tracing::info!("shader cache: {current} did not write it, discarding entries");
        }
    });
}

// The segment this process writes, then the read-only one a bundle ships. Both
// are in memory by the time a second lookup arrives, so the tiering costs no
// I/O per shader. They only diverge for a bundle: a dev checkout ships no
// artifacts, and a read-only install cannot write beside the ones it has.
//
// A zero-length artifact is never a legitimate compile result, so a hand-edited
// or truncated entry reads as a miss and recompiles rather than failing
// pipeline creation with an empty bytecode blob.
fn load(digest: &str) -> Option<Vec<u8>> {
    let usable = |bytes: Vec<u8>| (!bytes.is_empty()).then_some(bytes);
    crate::runtime_cache::load(KIND, digest)
        .and_then(usable)
        .or_else(|| crate::runtime_cache::load_bundled(KIND, digest).and_then(usable))
}

fn store(digest: &str, bytes: &[u8]) {
    crate::runtime_cache::store(KIND, digest, bytes);
}

#[cfg(test)]
mod tests {
    use super::*;

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

    // Under `cargo test` nothing reaches the state dir, in either direction,
    // and that covers the bundled tier too.
    #[test]
    fn the_cache_is_off_under_test() {
        assert!(!enabled());
        assert_eq!(load("deadbeef"), None);
    }

    // The segment `cn export` ships: warmed in memory here, written by the
    // precompile once, and read back the way a player's first launch reads it.
    #[cfg(any(backend_dx, backend_vk))]
    #[test]
    fn ensure_in_compiles_once_then_finds_the_artifact_present() {
        let tree = concinnity_testing::TempTree::new();
        let path = concinnity_host::store::paths::StateTree::at(tree.path()).runtime_cache_path();
        let mut bundle = Segment::read_from(&path);
        let k = key("ensure src", "main", "ps_5_1", 3);

        let first = ensure_in(&mut bundle, &k, || Ok(vec![7, 7, 7])).unwrap();
        assert_eq!(first, Ensured::Compiled);

        // The second request must be served from the segment without
        // recompiling, and without having touched the filesystem yet.
        let second = ensure_in(&mut bundle, &k, || panic!("must not recompile")).unwrap();
        assert_eq!(second, Ensured::Present);
        assert!(!path.exists(), "warming is memory until the caller writes");

        assert!(bundle.write_to(&path, 1 << 20));
        let mut shipped = Segment::read_from(&path);
        assert_eq!(shipped.get(KIND, &k.digest()), Some(&[7, 7, 7][..]));
    }

    #[cfg(any(backend_dx, backend_vk))]
    #[test]
    fn ensure_in_propagates_a_compile_error_and_stores_nothing() {
        let mut bundle = Segment::read_from(std::path::Path::new("/nonexistent/cache/0"));
        let k = key("bad src", "main", "ps_5_1", 0);
        assert!(ensure_in(&mut bundle, &k, || Err("boom".to_string())).is_err());
        assert!(
            ensure_in(&mut bundle, &k, || Ok(Vec::new())).is_err(),
            "empty"
        );
        assert_eq!(bundle.get(KIND, &k.digest()), None);
    }
}
