//! The side tables the overlay builders take alongside the components they
//! draw.
//!
//! A text label and a sprite are both authored as a component plus per-asset
//! state the component does not carry: the scissor rectangle a screen imposes
//! on it, the layer it draws in, the atlas slot its texture streamed into. The
//! engine assembles these once per frame and hands the same tables to every
//! builder, so they are named once here.

use crate::components::ScrollPanel;
use crate::ecs::asset_id::AssetId;
use crate::ecs::{FontHandle, OverlayImage, PipelineContext, TextureHandle};
use alloc::vec::Vec;
use hashbrown::HashMap;

/// Scissor rectangle per overlay asset, `[x, y, width, height]` in the
/// reference canvas the sprite was authored in. An asset absent from the table
/// is unclipped.
pub type ClipRects = HashMap<AssetId, [f32; 4]>;

/// Draw layer per overlay asset. An asset absent from the table is layer 0.
pub type OverlayLayers = HashMap<AssetId, i32>;

/// Where one overlay element's draw call lands, resolved by the caller from
/// the element's asset id through [`ClipRects`] and [`OverlayLayers`].
#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub struct Placement {
    /// Scissor band `[x, y, width, height]` in the reference canvas; `None`
    /// draws unclipped.
    pub clip: Option<[f32; 4]>,
    /// Draw layer; 0 unless a screen or override lifts the element.
    pub layer: i32,
}

impl Placement {
    /// The placement of the element with `id`, read from the frame's tables.
    pub fn of(id: Option<AssetId>, clips: &ClipRects, layers: &OverlayLayers) -> Self {
        let Some(id) = id else {
            return Self::default();
        };
        Self {
            clip: clips.get(&id).copied(),
            layer: layers.get(&id).copied().unwrap_or(0),
        }
    }
}

/// Slot in the backend's atlas pool per streamed texture. A textured sprite
/// whose texture is absent falls back to a solid fill.
pub type TextureSlots = HashMap<TextureHandle, usize>;

/// One RGBA8 atlas in the pool: width, height and row-major pixels.
pub type Atlas = (u32, u32, Vec<u8>);

/// The shared text / sprite atlas pool graphics init assembles, in slot order:
/// each Font's atlas at its handle's slot, then the built-in fallback face, then
/// each sprite texture and overlay image with the slot it landed in.
#[derive(Debug, Default)]
pub struct AtlasPool {
    /// The atlases in slot order.
    pub atlases: Vec<Atlas>,
    /// The slot of each sprite texture and overlay image.
    pub sprite_slots: TextureSlots,
}

impl AtlasPool {
    /// The handle the next font face added is named by: its slot.
    pub fn next_font(&self) -> FontHandle {
        FontHandle::new(self.atlases.len() as u32)
    }

    /// Add a font face's atlas, returning the handle that names it. Fonts are
    /// added first, in handle order.
    pub fn push_font(&mut self, atlas: Atlas) -> FontHandle {
        let handle = self.next_font();
        self.atlases.push(atlas);
        handle
    }

    /// Add a sprite texture's pixels under `texture`.
    pub fn push_sprite(&mut self, texture: TextureHandle, atlas: Atlas) {
        self.sprite_slots.insert(texture, self.atlases.len());
        self.atlases.push(atlas);
    }

    /// Add a tool-provided overlay image under its reserved handle. A malformed
    /// image is skipped with a warning.
    pub fn push_overlay(&mut self, image: &OverlayImage) {
        if !image.is_well_formed() {
            tracing::warn!(
                "graphics: overlay image {:?} byte length mismatch; skipped",
                image.handle
            );
            return;
        }
        self.push_sprite(
            image.handle,
            (image.width, image.height, image.rgba.clone()),
        );
    }
}

/// Every world `ScrollPanel`'s clip band (reference space), keyed by each
/// element listed in any of its rows, so the draw path scissors scroll content
/// to its panel and off-band rows do not bleed over the chrome.
pub fn scroll_clip_rects(ctx: &PipelineContext<'_>) -> ClipRects {
    let mut clips = ClipRects::new();
    for panel in ctx.query::<ScrollPanel>() {
        let band = [panel.x, panel.y, panel.width, panel.height];
        for row in &panel.rows {
            for &id in &row.elements {
                clips.insert(id.id(), band);
            }
        }
    }
    clips
}

#[cfg(test)]
mod tests {
    use super::*;
    use alloc::vec;

    // Fonts take the leading slots in handle order; sprites and overlay images
    // follow, each mapped to its slot, and a malformed image takes none.
    #[test]
    fn the_atlas_pool_lays_fonts_first_and_maps_each_sprite_to_its_slot() {
        let mut pool = AtlasPool::default();
        assert_eq!(pool.push_font((1, 1, vec![0; 4])), FontHandle::new(0));
        assert_eq!(pool.push_font((1, 1, vec![0; 4])), FontHandle::new(1));
        pool.push_sprite(TextureHandle::new(7), (2, 1, vec![0; 8]));
        pool.push_overlay(&OverlayImage {
            handle: TextureHandle::new(900),
            width: 2,
            height: 2,
            rgba: vec![0; 3],
        });
        pool.push_overlay(&OverlayImage {
            handle: TextureHandle::new(901),
            width: 1,
            height: 1,
            rgba: vec![0; 4],
        });
        assert_eq!(pool.atlases.len(), 4);
        assert_eq!(pool.sprite_slots[&TextureHandle::new(7)], 2);
        assert_eq!(pool.sprite_slots[&TextureHandle::new(901)], 3);
        assert!(!pool.sprite_slots.contains_key(&TextureHandle::new(900)));
    }
    use crate::ecs::World;

    // Every element of every row clips to its own panel's band.
    #[test]
    fn each_listed_element_clips_to_its_panel() {
        use crate::components::ScrollRow;
        use crate::ecs::Ref;
        use alloc::vec;
        let mut world = World::new();
        world.push(ScrollPanel {
            x: 10.0,
            y: 20.0,
            width: 300.0,
            height: 200.0,
            rows: vec![ScrollRow {
                elements: vec![Ref::new(AssetId(3)), Ref::new(AssetId(4))],
                ..Default::default()
            }],
            ..Default::default()
        });
        let clips = scroll_clip_rects(&world.context());
        assert_eq!(clips.len(), 2);
        assert_eq!(clips[&AssetId(3)], [10.0, 20.0, 300.0, 200.0]);
        assert_eq!(clips[&AssetId(4)], [10.0, 20.0, 300.0, 200.0]);
    }
}
