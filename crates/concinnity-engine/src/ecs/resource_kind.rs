// The kinds of resource asset the registry names, each with its own table.

use crate::error::AssetError;

/// The kinds of resource the runtime keeps in per-kind tables, one dense handle
/// space per kind. The `#[repr(u8)]` discriminant is the resource stream's
/// `resource_kind` tag (like `ComponentTag` for components); cook writes it and
/// the runtime selects the table by it. Order is the assignment order cook uses.
/// Serialized as that `u8`, so an unknown tag fails to deserialize.
#[repr(u8)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, serde::Serialize, serde::Deserialize)]
#[serde(try_from = "u8", into = "u8")]
pub enum ResourceKind {
    /// Static mesh geometry.
    Mesh,
    /// A 2D texture.
    Texture,
    /// A baked material.
    Material,
    /// A font atlas.
    Font,
    /// A decoded audio clip.
    AudioClip,
    /// A cubemap texture.
    CubemapTexture,
    /// A prefiltered environment map.
    EnvironmentMap,
    /// A color lookup table.
    ColorLut,
    /// Skinned mesh geometry.
    SkinnedMesh,
}

impl From<ResourceKind> for u8 {
    fn from(kind: ResourceKind) -> Self {
        kind as Self
    }
}

impl TryFrom<u8> for ResourceKind {
    type Error = AssetError;

    fn try_from(tag: u8) -> Result<Self, AssetError> {
        Ok(match tag {
            0 => Self::Mesh,
            1 => Self::Texture,
            2 => Self::Material,
            3 => Self::Font,
            4 => Self::AudioClip,
            5 => Self::CubemapTexture,
            6 => Self::EnvironmentMap,
            7 => Self::ColorLut,
            8 => Self::SkinnedMesh,
            _ => return Err(AssetError::UnknownResourceKind { tag }),
        })
    }
}
