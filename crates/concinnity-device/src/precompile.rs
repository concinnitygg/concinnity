// src/precompile.rs
//
// Export-time compilation of the engine's built-in shaders. The DirectX and
// Vulkan backends declare their compile set as static data (each backend's
// builtins.rs); this module iterates those declarations and makes sure every
// enumerable variant's artifact exists in a bundle's shader-cache/ directory.
// Compilation is pure CPU (FXC / DXC / shaderc need no GPU device), so this
// runs inside `cn export` with no window, no adapter, and no child process.
// Renderer init compiles through the same declarations and the same cache
// keys, so a shipped bundle's first launch reuses every artifact written here;
// anything not enumerable (a world-authored SdfVolume fragment, an unusual
// runtime parameter) still compiles at init exactly as before.

use std::path::Path;

use crate::shader_cache::Ensured;

/// Outcome of a built-in shader precompile: how many artifacts were already in
/// place or copied from the local cache, how many compiled fresh, and the
/// programs that failed (with their compile diagnostics). Failures do not
/// abort the run -- the affected shader falls back to compiling at the
/// bundle's first launch.
#[derive(Debug, Default)]
pub struct Report {
    pub reused: usize,
    pub compiled: usize,
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

    /// Total artifacts now present in the output directory.
    pub fn cached(&self) -> usize {
        self.reused + self.compiled
    }
}

/// Compile every enumerable built-in shader variant into `out_dir` (a bundle's
/// `shader-cache/`). `texture_count` is the exported world's texture-table
/// length, which sizes the Vulkan bindless texture pool baked into its
/// pool-sized shaders; DirectX ignores it (its pools are unbounded arrays).
pub fn precompile_builtin_shaders(out_dir: &Path, texture_count: usize) -> Report {
    let mut report = Report::default();
    #[cfg(not(backend_vk))]
    let _ = texture_count;
    #[cfg(backend_dx)]
    crate::directx::builtins::precompile(out_dir, &mut report);
    #[cfg(backend_vk)]
    crate::vulkan::builtins::precompile(out_dir, texture_count, &mut report);
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
