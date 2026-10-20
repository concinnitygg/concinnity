//! The shader `Platform` vocabulary: which rendering backend a compiled shader
//! is for, and [`PlatformSet`] for a cook that targets several at once. The
//! enum is pure data with no ambient resolution of its own, so it sits in the
//! runtime foundation and every caller states the platform it means: the
//! engine names the backend it was built for, and the build pipeline is told
//! the backend it cooks for.

use alloc::string::String;
use alloc::vec::Vec;

pub mod select;

/// The rendering backends the engine compiles shaders for: Metal (MSL),
/// DirectX (DXIL), or Vulkan (SPIR-V).
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "lowercase")]
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
    pub const ALL: [Self; 3] = [Self::Metal, Self::DirectX, Self::Vulkan];

    /// The short name a cook cache key and an export stamp record the backend under.
    pub fn key(self) -> &'static str {
        match self {
            Self::Metal => "metal",
            Self::DirectX => "directx",
            Self::Vulkan => "vulkan",
        }
    }

    /// The graphics API's name as a player reads it in the settings menu.
    pub fn label(self) -> &'static str {
        match self {
            Self::Metal => "Metal",
            Self::DirectX => "DirectX 12",
            Self::Vulkan => "Vulkan",
        }
    }

    /// The platform `key` names, if any.
    pub fn from_key(key: &str) -> Option<Self> {
        Self::ALL.into_iter().find(|p| p.key() == key)
    }

    const fn bit(self) -> u8 {
        1 << self as u8
    }
}

/// A set of platforms, such as every backend a build compiles in, which a cook
/// compiles shaders for side by side.
///
/// ```rust
/// # use concinnity_engine::platform::{Platform, PlatformSet};
/// let set: PlatformSet = [Platform::DirectX, Platform::Vulkan].into_iter().collect();
/// assert!(set.contains(Platform::Vulkan));
/// assert_eq!(set.key(), "directx+vulkan");
/// ```
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct PlatformSet(u8);

impl PlatformSet {
    /// The set holding only `platform`.
    pub const fn single(platform: Platform) -> Self {
        Self(platform.bit())
    }

    /// Whether `platform` is in the set.
    pub const fn contains(self, platform: Platform) -> bool {
        self.0 & platform.bit() != 0
    }

    /// Whether the set holds no platform.
    pub const fn is_empty(self) -> bool {
        self.0 == 0
    }

    /// Whether every platform in the set is also in `other`.
    pub const fn is_subset(self, other: Self) -> bool {
        self.0 & !other.0 == 0
    }

    /// The platforms in the set, in [`Platform::ALL`] order.
    pub fn iter(self) -> impl Iterator<Item = Platform> {
        Platform::ALL.into_iter().filter(move |p| self.contains(*p))
    }

    /// Every member's [`Platform::key`], joined by `+`: what a cook cache key
    /// records the set under.
    pub fn key(self) -> String {
        self.iter().map(Platform::key).collect::<Vec<_>>().join("+")
    }
}

impl From<Platform> for PlatformSet {
    fn from(platform: Platform) -> Self {
        Self::single(platform)
    }
}

impl FromIterator<Platform> for PlatformSet {
    fn from_iter<I: IntoIterator<Item = Platform>>(iter: I) -> Self {
        Self(iter.into_iter().fold(0, |bits, p| bits | p.bit()))
    }
}

/// The graphics APIs this build can run, native first, published as a world
/// resource by the host that compiled them in.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CompiledPlatforms(pub &'static [Platform]);

/// Where this launch stands on the graphics API chosen in the settings menu,
/// published as a world resource by the host that selected the API.
#[derive(Debug, Clone, Copy)]
pub struct GraphicsApiStatus {
    /// The API this launch renders with.
    pub current: Platform,
    /// The API this launch gave up on because the last launch on it never drew
    /// a frame.
    pub reverted: Option<Platform>,
    /// Whether this launch is trying a newly chosen API that has not yet
    /// drawn: the settings menu clears the persisted trial after a frame.
    pub confirm_pending: bool,
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

    // The persisted settings store a platform under the same name a cache key
    // and an export stamp use.
    #[test]
    fn a_platform_serializes_as_its_key() {
        for platform in Platform::ALL {
            let json = serde_json::to_string(&platform).unwrap();
            assert_eq!(json, alloc::format!("\"{}\"", platform.key()));
            assert_eq!(serde_json::from_str::<Platform>(&json).unwrap(), platform);
        }
    }

    #[test]
    fn a_key_names_its_platform_back() {
        for platform in Platform::ALL {
            assert_eq!(Platform::from_key(platform.key()), Some(platform));
        }
        assert_eq!(Platform::from_key("opengl"), None);
    }

    #[test]
    fn a_set_holds_exactly_what_it_was_built_from() {
        let set: PlatformSet = [Platform::Vulkan, Platform::Metal, Platform::Vulkan]
            .into_iter()
            .collect();
        assert_eq!(
            set.iter().collect::<Vec<_>>(),
            [Platform::Metal, Platform::Vulkan]
        );
        assert!(!set.contains(Platform::DirectX));
        assert!(PlatformSet::default().is_empty());
        assert_eq!(PlatformSet::from(Platform::DirectX).key(), "directx");
    }

    #[test]
    fn subset_is_membership_of_every_member() {
        let dx = PlatformSet::single(Platform::DirectX);
        let dx_vk: PlatformSet = [Platform::DirectX, Platform::Vulkan].into_iter().collect();
        assert!(dx.is_subset(dx_vk));
        assert!(!dx_vk.is_subset(dx));
        assert!(PlatformSet::default().is_subset(dx));
    }
}
