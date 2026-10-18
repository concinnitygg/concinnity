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
use crate::ecs::{PipelineContext, TextureHandle};
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

/// Every world `ScrollPanel`'s clip band (reference space), keyed by each
/// element listed in any of its rows, so the draw path scissors scroll content
/// to its panel and off-band rows do not bleed over the chrome.
pub fn scroll_clip_rects(ctx: &PipelineContext) -> ClipRects {
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
