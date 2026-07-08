// src/build/asset.rs
//
// `BuildAsset` is the build-time counterpart to `Component`. Components whose
// `PAYLOAD = AssetPayload::Compiled` implement this trait to turn their args
// into a binary payload (mesh vertices, shader bytecode, decoded image, etc.).
// The build pipeline calls `<T as BuildAsset>::compile_payload` for each
// declared asset and packs the resulting bytes into a blob.

use crate::ecs::Component;
use crate::world::WorldJsonlAsset;

// Build-time context handed to each `BuildAsset` impl.
pub struct BuildCtx<'a> {
    // The asset's declared name (used in error messages and as a key for
    // build-time intermediates such as compiled shader filenames).
    pub name: &'a str,
    // Optional directory of user-supplied artifacts (e.g. account-uploaded
    // shader source files) consulted when resolving bare filenames.
    pub artifacts_dir: Option<&'a str>,
    // All sibling assets declared in the same world. Used by types like
    // `VoxelChunk` that need to resolve cross-asset references (palette).
    pub all_assets: &'a [WorldJsonlAsset],
}

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
