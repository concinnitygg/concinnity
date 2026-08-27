// src/components/room.rs
//
// Runtime `Room` component. Its authored args live in the schema crate
// (concinnity_asset::room).

use concinnity_asset::cook;

use crate::ecs::asset_id::AssetId;
use crate::ecs::{Component, PayloadLocator, TextureHandle};

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
    pub fn bake(args: cook::Room) -> Self {
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
mod tests {
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
        let room = Room::bake(cook::Room::default());
        assert_eq!(room.effective_texture(), None);
    }

    #[test]
    fn from_args_resolves_size_shorthand() {
        let args = cook::Room {
            size: Some([16.0, 20.0, 3.5]),
            ..cook::Room::default()
        };
        let room = Room::bake(args);
        assert_eq!(room.half_width, 8.0);
        assert_eq!(room.half_depth, 10.0);
        assert_eq!(room.ceiling_height, 3.5);
    }

    #[test]
    fn from_args_uses_explicit_half_extents_when_no_size() {
        let args = cook::Room {
            half_width: 5.0,
            half_depth: 7.0,
            ceiling_height: 4.0,
            size: None,
            ..cook::Room::default()
        };
        let room = Room::bake(args);
        assert_eq!(room.half_width, 5.0);
        assert_eq!(room.half_depth, 7.0);
        assert_eq!(room.ceiling_height, 4.0);
    }
}
