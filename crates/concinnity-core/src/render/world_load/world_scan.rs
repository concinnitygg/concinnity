//! What a world's components ask of its graphics load: a fallback font face,
//! the HUD chips the frame anchors, the color LUT's blend, frame captures, and
//! the planet's grass.

use alloc::vec::Vec;
use hashbrown::HashMap;

use crate::components::{
    DebugHud, Grass, HitRegion, KeyBinding, Planet, StatHud, TextInput, TextLabel, UiAction,
};
use crate::ecs::asset_id::AssetId;
use crate::ecs::{PipelineContext, Ref};
use crate::render::grass::PlanetGrass;
use crate::resource::ColorLutTable;

/// Whether any text in the world names no Font, and so has no face to draw
/// with unless one is registered as the fallback.
pub fn font_less_text(ctx: &PipelineContext) -> bool {
    ctx.query::<TextLabel>().any(|l| l.font.is_none())
        || ctx.query::<TextInput>().any(|t| t.font.is_none())
}

/// The DebugHud chip ids (cursor, camera, sys, passes: the top-right stack
/// order) and the StatHud chip ids (fps, gpu wait, vram, ram, ev, edr, track:
/// the top-left strip order) the frame step anchors. Passes is last because its
/// height follows the frame's step count.
pub fn capture_hud_chips(ctx: &PipelineContext) -> (Vec<AssetId>, Vec<AssetId>) {
    let debug = ctx
        .query::<DebugHud>()
        .next()
        .map(|d| {
            [d.mouse_label, d.camera_label, d.sys_label, d.passes_label]
                .into_iter()
                .flatten()
                .map(Ref::id)
                .collect()
        })
        .unwrap_or_default();
    let stat = ctx
        .query::<StatHud>()
        .next()
        .map(|s| {
            [
                s.fps_label,
                s.gpu_wait_label,
                s.vram_label,
                s.ram_label,
                s.ev_label,
                s.edr_label,
                s.track_label,
            ]
            .into_iter()
            .flatten()
            .map(Ref::id)
            .collect()
        })
        .unwrap_or_default();
    (debug, stat)
}

/// The blend strength of the world's first `ColorLut`, read from its compiled
/// payload, or `None` without one.
pub fn color_lut_strength(ctx: &mut PipelineContext) -> Option<f32> {
    let locator = ctx.resource::<ColorLutTable>()?.locator(0)?;
    let bytes = ctx.read_payload(&locator).ok()?;
    crate::bake::color_lut::strength(bytes).ok()
}

/// Whether the world saves frames as it goes, from its benchmark or from a
/// `screenshot` action, which needs the presented frame readable just as an
/// exit screenshot does.
pub fn declares_frame_captures(ctx: &PipelineContext) -> bool {
    let screenshot = |action: Option<&UiAction>| matches!(action, Some(UiAction::Screenshot(_)));
    ctx.resource::<crate::benchmark::BenchmarkRun>()
        .is_some_and(|run| !run.config.captures.is_empty())
        || ctx
            .query::<KeyBinding>()
            .any(|kb| screenshot(kb.action.as_ref()))
        || ctx
            .query::<HitRegion>()
            .any(|r| screenshot(r.action.as_ref()))
}

/// The grass `planet`'s layers grow, each layer's `Grass` looked up in `grass`
/// by id; `None` when no layer grows anything. A layer naming no known
/// `Grass` grows nothing. A density mask is not applied on a planet, whose
/// density comes from its meadow alone; one is reported and left unused.
pub fn planet_grass(planet: &Planet, grass: &HashMap<AssetId, Grass>) -> Option<PlanetGrass> {
    let layers: Vec<Grass> = planet
        .layers
        .iter()
        .enumerate()
        .filter_map(|(i, layer)| {
            if layer.density_mask.is_some() {
                tracing::warn!(
                    "Planet layer {i} sets a density_mask, which a Planet does not \
                     support; the layer grows over the planet's meadow without it"
                );
            }
            grass.get(&layer.grass.id()).cloned()
        })
        .collect();
    PlanetGrass::resolve(planet.shape(), &layers)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::components::TerrainLayer;
    use crate::ecs::World;
    use alloc::vec;

    #[test]
    fn layers_resolve_through_their_grass() {
        let grass: HashMap<AssetId, Grass> = [(AssetId(0), Grass::default())].into();
        let layer = |id: u32| TerrainLayer {
            grass: Ref::new(AssetId(id)),
            density_mask: None,
        };
        let planet = Planet {
            layers: vec![layer(0), layer(1)],
            ..Planet::default()
        };
        let resolved = planet_grass(&planet, &grass).expect("one layer grows");
        assert_eq!(resolved.layers.len(), 1);
        assert_eq!(resolved.shape, planet.shape());
        let bare = Planet {
            layers: vec![layer(1)],
            ..Planet::default()
        };
        assert!(planet_grass(&bare, &grass).is_none());
        assert!(planet_grass(&Planet::default(), &grass).is_none());
    }

    // Text without a Font needs the fallback face; text naming one does not.
    #[test]
    fn only_font_less_text_needs_the_fallback() {
        let mut world = World::new();
        assert!(!font_less_text(&world.context()));
        world.push(TextLabel::default());
        assert!(font_less_text(&world.context()));
    }

    #[test]
    fn a_world_without_huds_anchors_no_chips() {
        let mut world = World::new();
        assert_eq!(
            capture_hud_chips(&world.context()),
            (Vec::new(), Vec::new())
        );
    }

    // A screenshot key binding is a frame capture; any other action is not.
    #[test]
    fn a_screenshot_binding_declares_frame_captures() {
        let mut world = World::new();
        assert!(!declares_frame_captures(&world.context()));
        world.push(KeyBinding {
            action: Some(UiAction::Screenshot(Default::default())),
            ..Default::default()
        });
        assert!(declares_frame_captures(&world.context()));
    }
}
