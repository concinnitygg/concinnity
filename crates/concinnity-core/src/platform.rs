//! The shader `Platform` vocabulary: which rendering backend a compiled shader
//! is for. The enum is pure data with no ambient resolution of its own, so it
//! sits in the runtime foundation and every caller states the platform it
//! means -- the engine names the backend it was built for, and the build
//! pipeline is told the backend it cooks for.

/// The rendering backends the engine compiles shaders for: Metal (MSL),
/// DirectX (DXIL), or Vulkan (SPIR-V).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Platform {
    /// The Metal backend.
    Metal,
    /// The DirectX backend.
    DirectX,
    /// The Vulkan backend.
    Vulkan,
}

impl Platform {
    /// Every platform.
    pub const ALL: [Platform; 3] = [Platform::Metal, Platform::DirectX, Platform::Vulkan];

    /// The short name a cook cache key and an export stamp record the backend under.
    pub fn key(self) -> &'static str {
        match self {
            Platform::Metal => "metal",
            Platform::DirectX => "directx",
            Platform::Vulkan => "vulkan",
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn every_platform_has_a_distinct_key() {
        assert_eq!(Platform::Metal.key(), "metal");
        assert_eq!(Platform::DirectX.key(), "directx");
        assert_eq!(Platform::Vulkan.key(), "vulkan");
    }
}
