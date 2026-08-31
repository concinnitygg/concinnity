//! The shader `Platform` vocabulary: which shader source language a rendering
//! backend consumes. The enum is pure data with no ambient resolution of its
//! own, so it sits in the runtime foundation and every caller states the
//! platform it means -- the engine names the backend it was built for, and the
//! build pipeline is told the backend it cooks for.

/// Shader source language families supported by the engine. Each variant
/// matches one render backend: Metal, HLSL (DirectX), or GLSL (Vulkan).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Platform {
    /// Metal Shading Language, for the Metal backend.
    Metal,
    /// HLSL, for the DirectX backend.
    Hlsl,
    /// GLSL, for the Vulkan backend.
    Glsl,
}

impl Platform {
    /// String key used in the `sources` map of a `Shader` stage.
    pub fn key(self) -> &'static str {
        match self {
            Platform::Metal => "metal",
            Platform::Hlsl => "hlsl",
            Platform::Glsl => "glsl",
        }
    }

    /// Whether a shader source with the given file extension is usable on this
    /// platform. The matching extension (`metal` / `hlsl` / `glsl`) is accepted;
    /// a non-matching shader extension is rejected so a single-path source
    /// authored for one backend doesn't get fed to another; an unknown
    /// extension is accepted by default (the build step surfaces a real compile
    /// error later if the file truly can't be built).
    ///
    /// Shared by the per-platform source selection of `Shader` stages and
    /// `SdfVolume` so both apply identical fallback rules.
    pub fn accepts_ext(self, ext: &str) -> bool {
        !matches!(ext, "metal" | "hlsl" | "glsl") || ext == self.key()
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
