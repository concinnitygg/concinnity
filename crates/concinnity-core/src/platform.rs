// The shader `Platform` selector: which shader source language the running
// backend consumes. Pure (no I/O, cfg-resolved), so it sits in the runtime
// foundation rather than the build module -- the engine picks a `Shader` stage's
// current-platform source at runtime, and the build pipeline reuses the same
// selection at compile time. Re-exported as `crate::build::Platform` for the
// build-side callers.

// Shader source language families supported by the engine. Each variant
// matches one render backend: Metal, HLSL (DirectX), or GLSL (Vulkan).
//
// A given build only ever constructs the variant for its own backend (see
// `current`), so the other two read as never-constructed; `key` still matches
// all three, so the type stays whole.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[allow(dead_code)]
pub enum Platform {
    Metal,
    Hlsl,
    Glsl,
}

impl Platform {
    // The shader platform the current binary's rendering backend was built
    // for. Resolved from the backend cfg (see build.rs), not the target OS, so
    // a Windows Vulkan build correctly selects GLSL rather than HLSL.
    pub fn current() -> Self {
        #[cfg(backend_metal)]
        {
            Platform::Metal
        }
        #[cfg(backend_dx)]
        {
            Platform::Hlsl
        }
        #[cfg(backend_vk)]
        {
            Platform::Glsl
        }
    }

    // String key used in the `sources` map of a `Shader` stage.
    pub fn key(self) -> &'static str {
        match self {
            Platform::Metal => "metal",
            Platform::Hlsl => "hlsl",
            Platform::Glsl => "glsl",
        }
    }

    // Whether a shader source with the given file extension is usable on this
    // platform. The matching extension (`metal` / `hlsl` / `glsl`) is accepted;
    // a non-matching shader extension is rejected so a single-path source
    // authored for one backend doesn't get fed to another; an unknown
    // extension is accepted by default (the build step surfaces a real compile
    // error later if the file truly can't be built).
    //
    // Shared by the per-platform source selection of `Shader` stages and
    // `SdfVolume` so both apply identical fallback rules.
    pub fn accepts_ext(self, ext: &str) -> bool {
        match (ext, self) {
            ("metal", Platform::Metal) => true,
            ("hlsl", Platform::Hlsl) => true,
            ("glsl", Platform::Glsl) => true,
            _ if matches!(ext, "metal" | "hlsl" | "glsl") => false,
            _ => true,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn platform_key_and_accepts_ext_cover_all_variants() {
        assert_eq!(Platform::Metal.key(), "metal");
        assert_eq!(Platform::Hlsl.key(), "hlsl");
        assert_eq!(Platform::Glsl.key(), "glsl");

        // The matching extension is accepted; another backend's shader
        // extension is rejected; an unknown extension is accepted by default.
        assert!(Platform::Metal.accepts_ext("metal"));
        assert!(!Platform::Metal.accepts_ext("hlsl"));
        assert!(!Platform::Metal.accepts_ext("glsl"));
        assert!(Platform::Hlsl.accepts_ext("hlsl"));
        assert!(!Platform::Hlsl.accepts_ext("metal"));
        assert!(Platform::Glsl.accepts_ext("glsl"));
        assert!(!Platform::Glsl.accepts_ext("hlsl"));
        assert!(Platform::Metal.accepts_ext("txt"));
    }
}
