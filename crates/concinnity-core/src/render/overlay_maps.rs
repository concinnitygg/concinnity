//! The side tables the overlay builders take alongside the components they
//! draw.
//!
//! A text label and a sprite are both authored as a component plus per-asset
//! state the component does not carry: the scissor rectangle a screen imposes
//! on it, the layer it draws in, the atlas slot its texture streamed into. The
//! engine assembles these once per frame and hands the same tables to every
//! builder, so they are named once here.

use crate::ecs::TextureHandle;
use crate::ecs::asset_id::AssetId;
use hashbrown::HashMap;

/// Scissor rectangle per overlay asset, `[x, y, width, height]` in the
/// reference canvas the sprite was authored in. An asset absent from the table
/// is unclipped.
pub type ClipRects = HashMap<AssetId, [f32; 4]>;

/// Draw layer per overlay asset. An asset absent from the table is layer 0.
pub type OverlayLayers = HashMap<AssetId, i32>;

/// Slot in the backend's atlas pool per streamed texture. A textured sprite
/// whose texture is absent falls back to a solid fill.
pub type TextureSlots = HashMap<TextureHandle, usize>;
