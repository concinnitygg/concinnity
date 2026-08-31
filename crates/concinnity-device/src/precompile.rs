//! Export-time compilation of the engine's built-in shaders. The DirectX and
//! Vulkan backends declare their compile set as static data (each backend's
//! builtins.rs); this module iterates those declarations and makes sure every
//! enumerable variant's artifact is in the runtime cache segment a bundle
//! ships. Compilation is pure CPU (FXC / DXC / shaderc need no GPU device), so
//! this runs inside `cn export` with no window, no adapter, and no child
//! process. Renderer init compiles through the same declarations and the same
//! cache keys, so a shipped bundle's first launch reuses every artifact written
//! here; anything not enumerable (a world-authored SdfVolume fragment, an
//! unusual runtime parameter) still compiles at init exactly as before.
//!
//! The segment is built in memory and written once when the run finishes, so
//! warming a hundred artifacts costs one file write rather than a hundred.

use std::path::Path;

use concinnity_host::store::cache::{CACHE_BUDGET_BYTES, Segment};
use concinnity_host::store::paths::StateTree;

use crate::shader_cache::Ensured;

/// Outcome of a built-in shader precompile: how many artifacts were already in
/// place or copied from the local cache, how many compiled fresh, and the
/// programs that failed (with their compile diagnostics). Failures do not
/// abort the run -- the affected shader falls back to compiling at the
/// bundle's first launch.
#[derive(Debug, Default)]
pub struct Report {
    /// Artifacts already in place or copied from the local cache.
    pub reused: usize,
    /// Artifacts compiled fresh this run.
    pub compiled: usize,
    /// Programs that failed, with their compile diagnostics.
    pub failed: Vec<String>,
}

impl Report {
    pub(crate) fn record(&mut self, label: &str, result: Result<Ensured, String>) {
        match result {
            Ok(Ensured::Present) | Ok(Ensured::Copied) => self.reused += 1,
            Ok(Ensured::Compiled) => self.compiled += 1,
            Err(e) => self.failed.push(format!("{label}: {e}")),
        }
    }

    /// Total artifacts now in the bundle's segment.
    pub fn cached(&self) -> usize {
        self.reused + self.compiled
    }
}

/// Compile every enumerable built-in shader variant into the runtime cache
/// segment under `state_dir`, which is a bundle's state root.
///
/// Nothing about the exported world enters this: every variant a backend can
/// take is a property of the device the bundle eventually runs on -- its MSAA
/// mode, its probe cube-array length, whether it seats the bindless pool at its
/// ceiling -- so each is baked at what a desktop driver affords, and a device
/// that differs misses those entries and compiles at first launch.
///
/// The segment is left unstamped by the host toolchain that warmed it: a
/// player's own slangc is whatever it is, and a shipped artifact is a function
/// of its source rather than of what compiled it.
pub fn precompile_builtin_shaders(state_dir: &Path) -> Report {
    let path = StateTree::at(state_dir).bundled_runtime_cache_path();
    let mut bundle = Segment::read_from(&path);
    let mut report = Report::default();
    #[cfg(backend_dx)]
    {
        crate::directx::builtins::precompile(&mut bundle, &mut report);
        crate::directx::slang_builtins::precompile(&mut bundle, &mut report);
    }
    #[cfg(backend_vk)]
    crate::vulkan::builtins::precompile(&mut bundle, &mut report);
    bundle.write_to(&path, CACHE_BUDGET_BYTES);
    // `ensure_in` also keeps a copy in this machine's own cache segment, so
    // repeated exports stay warm. That copy is memory until a checkpoint.
    crate::runtime_cache::checkpoint();
    report
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn report_tallies_reuse_compile_and_failure() {
        let mut r = Report::default();
        r.record("a", Ok(Ensured::Present));
        r.record("b", Ok(Ensured::Copied));
        r.record("c", Ok(Ensured::Compiled));
        r.record("d ps_5_1", Err("boom".to_string()));
        assert_eq!(r.reused, 2);
        assert_eq!(r.compiled, 1);
        assert_eq!(r.cached(), 3);
        assert_eq!(r.failed, vec!["d ps_5_1: boom".to_string()]);
    }
}
