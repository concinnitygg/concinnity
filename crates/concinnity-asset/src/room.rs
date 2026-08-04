// Room authoring schema. The runtime `Room` component lives in core.

use crate::{TextureHandle, de_opt_texture_handle};
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
    #[serde(deserialize_with = "de_opt_texture_handle")]
    pub texture: Option<TextureHandle>,
    /// [Texture](#texture) for the walls. Currently all surfaces share one
    /// texture; per-surface texturing is reserved for a future update.
    #[serde(deserialize_with = "de_opt_texture_handle")]
    pub wall_texture: Option<TextureHandle>,
    /// [Texture](#texture) for the floor (see `wall_texture`).
    #[serde(deserialize_with = "de_opt_texture_handle")]
    pub floor_texture: Option<TextureHandle>,
    /// [Texture](#texture) for the ceiling (see `wall_texture`).
    #[serde(deserialize_with = "de_opt_texture_handle")]
    pub ceiling_texture: Option<TextureHandle>,
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_blank_room_is_an_untextured_box_at_the_default_dimensions() {
        let r = RoomArgs::default();
        assert_eq!(r.half_width, 8.0);
        assert_eq!(r.half_depth, 10.0);
        assert_eq!(r.ceiling_height, 3.5);
        // `size` overrides the three dimensions above when set.
        assert_eq!(r.size, None);
        assert!(r.texture.is_none());
        assert!(r.wall_texture.is_none());
        assert!(r.floor_texture.is_none());
        assert!(r.ceiling_texture.is_none());
        assert_eq!(r.lod_levels, 1);
        assert!(r.lod_distances.is_empty());
    }

    #[test]
    fn each_surface_takes_its_own_texture_and_falls_back_to_the_shared_one() {
        crate::test_support::install_resolvers();
        let r: RoomArgs = serde_json::from_str(
            r#"{"texture":"tex_base","wall_texture":"tex_brick","floor_texture":"tex_stone"}"#,
        )
        .unwrap();
        assert_eq!(r.texture, Some(TextureHandle(8)));
        assert_eq!(r.wall_texture, Some(TextureHandle(9)));
        assert_eq!(r.floor_texture, Some(TextureHandle(9)));
        // The ceiling was not named, so it falls back to the shared texture.
        assert_eq!(r.ceiling_texture, None);
    }

    #[test]
    fn an_authored_room_round_trips_through_postcard() {
        let r: RoomArgs =
            serde_json::from_str(r#"{"size":[20,4,30],"lod_levels":2,"lod_distances":[25]}"#)
                .unwrap();
        assert_eq!(r.size, Some([20.0, 4.0, 30.0]));

        let bytes = postcard::to_allocvec(&r).unwrap();
        let back: RoomArgs = postcard::from_bytes(&bytes).unwrap();
        assert_eq!(back.size, Some([20.0, 4.0, 30.0]));
        assert_eq!(back.lod_levels, 2);
        assert_eq!(back.lod_distances, [25.0]);
        // The half-extent fields keep their defaults; `size` takes precedence.
        assert_eq!(back.half_width, 8.0);
    }
}
