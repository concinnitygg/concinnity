// Room authoring schema. The runtime `Room` component lives in core.

use crate::{AssetId, de_opt_asset_ref};
use alloc::vec::Vec;

/// Authored fields of a `Room`; the resolved dimensions and payload locator are
/// runtime state.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
#[serde(default)]
pub struct RoomArgs {
    /// Half the room's width along X, in world units. Ignored when `size` is set.
    pub half_width: f32,
    /// Half the room's depth along Z, in world units. Ignored when `size` is set.
    pub half_depth: f32,
    /// Floor-to-ceiling height in world units. Ignored when `size` is set.
    pub ceiling_height: f32,
    /// Shorthand for the full dimensions `[width, depth, height]`. When set, it
    /// overrides `half_width`, `half_depth`, and `ceiling_height`.
    pub size: Option<[f32; 3]>,
    /// [Texture](#texture) applied to all surfaces. Falls back to `wall_texture`
    /// when unset. Generator names such as `"brick"` or `"concrete"` resolve to
    /// a matching texture at build time.
    #[serde(deserialize_with = "de_opt_asset_ref")]
    pub texture: Option<AssetId>,
    /// [Texture](#texture) for the walls. Currently all surfaces share one
    /// texture; per-surface texturing is reserved for a future update.
    #[serde(deserialize_with = "de_opt_asset_ref")]
    pub wall_texture: Option<AssetId>,
    /// [Texture](#texture) for the floor (see `wall_texture`).
    #[serde(deserialize_with = "de_opt_asset_ref")]
    pub floor_texture: Option<AssetId>,
    /// [Texture](#texture) for the ceiling (see `wall_texture`).
    #[serde(deserialize_with = "de_opt_asset_ref")]
    pub ceiling_texture: Option<AssetId>,
    /// Number of level-of-detail versions to generate, including the original.
    /// `1` (the default) generates no alternates.
    pub lod_levels: u32,
    /// Camera distances at which to switch to each lower-detail version. Empty
    /// lets the build choose defaults.
    #[serde(default)]
    pub lod_distances: Vec<f32>,
}

impl Default for RoomArgs {
    fn default() -> Self {
        Self {
            half_width: 8.0,
            half_depth: 10.0,
            ceiling_height: 3.5,
            size: None,
            texture: None,
            wall_texture: None,
            floor_texture: None,
            ceiling_texture: None,
            lod_levels: 1,
            lod_distances: Vec::new(),
        }
    }
}
