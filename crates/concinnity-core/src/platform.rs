//! The shader `Platform` vocabulary: which compiled shader form a rendering
//! backend consumes. The enum is pure data with no ambient resolution of its
//! own, so it sits in the runtime foundation and every caller states the
//! platform it means -- the engine names the backend it was built for, and the
//! build pipeline is told the backend it cooks for.

/// The shader targets the engine compiles for. Each variant matches one render
/// backend: Metal (MSL), DirectX (DXIL), or Vulkan (SPIR-V).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Platform {
    /// The Metal backend.
    Metal,
    /// The DirectX backend.
    Hlsl,
    /// The Vulkan backend.
    Glsl,
}

impl Platform {
    /// The short name a cook cache key and an export stamp record the backend under.
    pub fn key(self) -> &'static str {
        match self {
            Platform::Metal => "metal",
            Platform::Hlsl => "hlsl",
            Platform::Glsl => "glsl",
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn every_platform_has_a_distinct_key() {
        assert_eq!(Platform::Metal.key(), "metal");
        assert_eq!(Platform::Hlsl.key(), "hlsl");
        assert_eq!(Platform::Glsl.key(), "glsl");
    }
}
