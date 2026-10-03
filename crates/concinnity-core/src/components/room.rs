//! The `Room` asset: the authored args a world declares, and the runtime
//! component they bake into.

use crate::ecs::PayloadLocator;
use crate::ecs::TextureHandle;
use alloc::vec::Vec;

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
///
/// ```rust
/// # use concinnity_core::components::cook::Room as RoomArgs;
/// RoomArgs {
///     size: Some([16.0, 20.0, 3.5]),
///     ..Default::default()
/// };
/// ```
#[derive(
    Debug,
    Clone,
    serde::Serialize,
    serde::Deserialize,
    crate::ecs::AssetFields,
    crate::ecs::AssetDefault,
)]
#[serde(default)]
pub struct RoomArgs {
    /// Half the room's width along X, in world units. Ignored when `size` is set.
    #[asset(default = 8.0)]
    pub half_width: f32,
    /// Half the room's depth along Z, in world units. Ignored when `size` is set.
    #[asset(default = 10.0)]
    pub half_depth: f32,
    /// Floor-to-ceiling height in world units. Ignored when `size` is set.
    #[asset(default = 3.5)]
    pub ceiling_height: f32,
    /// Shorthand for the full dimensions `[width, depth, height]`. When set, it
    /// overrides `half_width`, `half_depth`, and `ceiling_height`.
    pub size: Option<[f32; 3]>,
    /// [Texture](#texture) applied to all surfaces. Falls back to `wall_texture`
    /// when unset. Generator names such as `"brick"` or `"concrete"` resolve to
    /// a matching texture at build time.
    pub texture: Option<TextureHandle>,
    /// [Texture](#texture) for the walls. Currently all surfaces share one
    /// texture; per-surface texturing is reserved for a future update.
    pub wall_texture: Option<TextureHandle>,
    /// [Texture](#texture) for the floor (see `wall_texture`).
    pub floor_texture: Option<TextureHandle>,
    /// [Texture](#texture) for the ceiling (see `wall_texture`).
    pub ceiling_texture: Option<TextureHandle>,
    /// Number of level-of-detail versions to generate, including the original.
    /// `1` (the default) generates no alternates.
    #[asset(default = 1)]
    pub lod_levels: u32,
    /// Camera distances at which to switch to each lower-detail version. Empty
    /// lets the build choose defaults.
    pub lod_distances: Vec<f32>,
}

/// The runtime `Room`: the dimensions resolved from
/// [`cook::Room`](crate::components::cook::Room)'s authored fields, and its payload
/// locator.
#[derive(Debug, serde::Serialize, serde::Deserialize)]
pub struct Room {
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

#[cfg(test)]
mod runtime_tests {
    use super::*;

    #[test]
    fn effective_texture_returns_texture_field_first() {
        let room = Room {
            half_width: 8.0,
            half_depth: 10.0,
            ceiling_height: 3.5,
            texture: Some(TextureHandle::new(1)),
            wall_texture: Some(TextureHandle::new(2)),
            floor_texture: None,
            ceiling_texture: None,
            locator: None,
        };
        assert_eq!(room.effective_texture(), Some(TextureHandle::new(1)));
    }

    #[test]
    fn effective_texture_falls_back_to_wall_texture() {
        let room = Room {
            half_width: 8.0,
            half_depth: 10.0,
            ceiling_height: 3.5,
            texture: None,
            wall_texture: Some(TextureHandle::new(7)),
            floor_texture: None,
            ceiling_texture: None,
            locator: None,
        };
        assert_eq!(room.effective_texture(), Some(TextureHandle::new(7)));
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
