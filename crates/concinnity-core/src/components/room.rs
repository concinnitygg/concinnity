// src/components/room.rs
//
// The `Room` asset: the authored args a world declares, and the runtime
// component they bake into.

use crate::ecs::TextureHandle;
use crate::ecs::asset_id::AssetId;
use crate::ecs::de_opt_texture_handle;
use crate::ecs::{Component, PayloadLocator};
use alloc::vec::Vec;

/// Authored fields of a `Room`; the resolved dimensions and payload locator are
/// runtime state.
///
/// ```rust
/// # use concinnity_core::components::cook::Room as RoomArgs;
/// RoomArgs {
///     size: Some([16.0, 20.0, 3.5]),
///     ..Default::default()
/// };
/// ```
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

/// A self-contained room (floor, ceiling, four walls), with optional texturing.
///
/// Prefer `Room` over a [ProceduralMesh](#proceduralmesh) (generator `"room"`) +
/// [Prop](#prop) pair for a shorter declaration. The room is placed at the world
/// origin.
///
/// Dimensions can be given as `size: [width, depth, height]` (full extents) or
/// as `half_width`, `half_depth`, and `ceiling_height` individually.
///
/// `texture`, `wall_texture`, `floor_texture`, and `ceiling_texture` are checked
/// in that order; the first set value wins. Generator names such as `"brick"` or
/// `"concrete"` resolve to a matching [Texture](#texture) at build time.
#[derive(Debug, serde::Serialize, serde::Deserialize)]
pub struct Room {
    /// Assigned by the loader; not authored.
    pub asset_id: AssetId,
    /// Half the room's width in world units.
    pub half_width: f32,
    /// Half the room's depth in world units.
    pub half_depth: f32,
    /// Floor-to-ceiling height in world units.
    pub ceiling_height: f32,
    /// Texture applied to every surface unless a surface overrides it.
    pub texture: Option<TextureHandle>,
    /// Texture for the four walls.
    pub wall_texture: Option<TextureHandle>,
    /// Texture for the floor.
    pub floor_texture: Option<TextureHandle>,
    /// Texture for the ceiling.
    pub ceiling_texture: Option<TextureHandle>,
    /// The generated geometry's place in the blob, injected at load.
    pub locator: Option<PayloadLocator>,
}

impl Room {
    /// Returns the first set texture reference across all texture fields.
    pub fn effective_texture(&self) -> Option<TextureHandle> {
        [
            self.texture,
            self.wall_texture,
            self.floor_texture,
            self.ceiling_texture,
        ]
        .into_iter()
        .flatten()
        .next()
    }
}

impl Room {
    /// Translate the authored args into the runtime room: resolve the `size`
    /// shorthand into half extents. Run by cook at build time (the baked blob
    /// record carries the result).
    pub fn bake(args: RoomArgs) -> Self {
        let (half_width, half_depth, ceiling_height) = if let Some([w, d, h]) = args.size {
            (w / 2.0, d / 2.0, h)
        } else {
            (args.half_width, args.half_depth, args.ceiling_height)
        };
        Self {
            asset_id: AssetId::default(),
            half_width,
            half_depth,
            ceiling_height,
            texture: args.texture,
            wall_texture: args.wall_texture,
            floor_texture: args.floor_texture,
            ceiling_texture: args.ceiling_texture,
            locator: None,
        }
    }
}

impl Component for Room {
    const NAME: &'static str = "Room";

    fn from_baked(bytes: &[u8]) -> Result<Self, crate::result::CnResult> {
        Ok(crate::blob::decode_exact(bytes)?)
    }

    fn inject_locator(&mut self, locator: PayloadLocator) {
        self.locator = Some(locator);
    }

    fn inject_name(&mut self, id: AssetId) {
        self.asset_id = id;
    }
}

#[cfg(test)]
mod runtime_tests {
    use super::*;

    #[test]
    fn effective_texture_returns_texture_field_first() {
        let room = Room {
            asset_id: AssetId::default(),
            half_width: 8.0,
            half_depth: 10.0,
            ceiling_height: 3.5,
            texture: Some(TextureHandle(1)),
            wall_texture: Some(TextureHandle(2)),
            floor_texture: None,
            ceiling_texture: None,
            locator: None,
        };
        assert_eq!(room.effective_texture(), Some(TextureHandle(1)));
    }

    #[test]
    fn effective_texture_falls_back_to_wall_texture() {
        let room = Room {
            asset_id: AssetId::default(),
            half_width: 8.0,
            half_depth: 10.0,
            ceiling_height: 3.5,
            texture: None,
            wall_texture: Some(TextureHandle(7)),
            floor_texture: None,
            ceiling_texture: None,
            locator: None,
        };
        assert_eq!(room.effective_texture(), Some(TextureHandle(7)));
    }

    #[test]
    fn effective_texture_returns_none_when_all_unset() {
        let room = Room::bake(RoomArgs::default());
        assert_eq!(room.effective_texture(), None);
    }

    #[test]
    fn from_args_resolves_size_shorthand() {
        let args = RoomArgs {
            size: Some([16.0, 20.0, 3.5]),
            ..RoomArgs::default()
        };
        let room = Room::bake(args);
        assert_eq!(room.half_width, 8.0);
        assert_eq!(room.half_depth, 10.0);
        assert_eq!(room.ceiling_height, 3.5);
    }

    #[test]
    fn from_args_uses_explicit_half_extents_when_no_size() {
        let args = RoomArgs {
            half_width: 5.0,
            half_depth: 7.0,
            ceiling_height: 4.0,
            size: None,
            ..RoomArgs::default()
        };
        let room = Room::bake(args);
        assert_eq!(room.half_width, 5.0);
        assert_eq!(room.half_depth, 7.0);
        assert_eq!(room.ceiling_height, 4.0);
    }
}
