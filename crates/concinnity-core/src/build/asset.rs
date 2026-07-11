// src/build/asset.rs
//
// The shader `Platform` selector and the `SourceBacked` trait. These stay in
// the runtime foundation because the engine reads a `ShaderStage`'s
// current-platform source at runtime (e.g. the DirectX backend checks whether
// the main shader is the built-in default before choosing its bindless path).
// The build-time context `BuildCtx` and the `BuildAsset` compile trait live in
// `concinnity-cook`.

use crate::ecs::Component;

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

    // String key used in the `sources` map of `ShaderStage`.
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
    // Shared by the per-platform source selection of `ShaderStage` and
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

// A component that points at a source file on disk. Implementations expose
// "here's my source path for this platform" without the build pipeline
// having to know which JSON key the asset uses to store it (`source` vs
// `path` vs the per-platform `sources` map).
//
// Returns `None` when the asset has no source on the given platform: for
// example, a `Texture` that uses a procedural generator instead of a file,
// or a `ShaderStage` whose `sources` map has no entry for the platform.
pub trait SourceBacked: Component {
    fn source_path(args: &serde_json::Value, platform: Platform) -> Option<String>;
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
