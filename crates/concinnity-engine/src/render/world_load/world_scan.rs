//! What a world's components ask of its graphics load: a fallback font face,
//! the HUD chips the frame anchors, the color LUT's blend, frame captures, the
//! planet's grass, and how it holds the cursor.

use alloc::vec::Vec;
use hashbrown::HashMap;

use crate::components::{
    Camera3D, DebugHud, Grass, GroundRule, HitRegion, KeyBinding, Planet, Screen, StatHud,
    TextInput, TextLabel, UiAction,
};
use crate::ecs::asset_id::AssetId;
use crate::ecs::{MenuOverride, PipelineContext, Ref};
use crate::render::environment::grass::PlanetGrass;
use crate::resource::ColorLutTable;

/// How a world holds the cursor from its first frame.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CursorPolicy {
    /// A camera world with UI the cursor is freed for (a `HitRegion` or a
    /// Screen): capture follows the menu frame by frame.
    pub menu_mode: bool,
    /// A plain first-person world captures the cursor at startup, unless a
    /// menu or editor driver (a `MenuOverride`) owns capture: an editor reruns
    /// the load on every rebuild, and a grab each time would re-hide its cursor.
    pub capture_at_start: bool,
}

/// The cursor policy for the world. `KeyBindings` alone need no free cursor,
/// and a UI-only world (no camera) stays free-cursor.
pub fn cursor_policy(ctx: &PipelineContext<'_>) -> CursorPolicy {
    let has_ui =
        ctx.query::<HitRegion>().next().is_some() || ctx.query::<Screen>().next().is_some();
    let has_camera = ctx.query::<Camera3D>().next().is_some();
    CursorPolicy {
        menu_mode: has_camera && has_ui,
        capture_at_start: has_camera && !has_ui && ctx.resource::<MenuOverride>().is_none(),
    }
}

/// Whether any text in the world names no Font, and so has no face to draw
/// with unless one is registered as the fallback.
pub fn font_less_text(ctx: &PipelineContext<'_>) -> bool {
    ctx.query::<TextLabel>().any(|l| l.font.is_none())
        || ctx.query::<TextInput>().any(|t| t.font.is_none())
}

/// The `DebugHud` chip ids (cursor, camera, sys, passes: the top-right stack
/// order) and the `StatHud` chip ids, its top row then its benchmark row, each
/// in left-to-right strip order, that the frame step anchors. Passes is last
/// because its height follows the frame's step count.
pub fn capture_hud_chips(ctx: &PipelineContext<'_>) -> (Vec<AssetId>, [Vec<AssetId>; 2]) {
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
    let ids = |row: &[Option<Ref<TextLabel>>]| row.iter().flatten().map(|r| r.id()).collect();
    let stat = ctx
        .query::<StatHud>()
        .next()
        .map(|s| [ids(&s.top_row()), ids(&s.bench_row())])
        .unwrap_or_default();
    (debug, stat)
}

/// The blend strength of the world's first `ColorLut`, read from its compiled
/// payload, or `None` without one.
pub fn color_lut_strength(ctx: &mut PipelineContext<'_>) -> Option<f32> {
    let locator = ctx.resource::<ColorLutTable>()?.locator(0)?;
    let bytes = ctx.read_payload(&locator).ok()?;
    crate::bake::color_lut::strength(bytes).ok()
}

/// Whether the world saves frames as it goes, from the run's screenshots or
/// from a `screenshot` action, which needs the presented frame readable just as an
/// exit screenshot does.
pub fn declares_frame_captures(ctx: &PipelineContext<'_>) -> bool {
    let screenshot = |action: Option<&UiAction>| matches!(action, Some(UiAction::Screenshot(_)));
    crate::screenshot::takes_screenshots(ctx)
        || ctx
            .query::<KeyBinding>()
            .any(|kb| screenshot(kb.action.as_ref()))
        || ctx
            .query::<HitRegion>()
            .any(|r| screenshot(r.action.as_ref()))
}

/// The grass `planet`'s layers grow, each layer's `Grass` looked up in `grass`
/// by id and grown where its rule holds; `None` when no layer grows anything.
/// A layer naming no known `Grass` grows nothing. A density mask is not
/// applied on a planet, whose density comes from its meadow alone; one is
/// reported and left unused.
pub fn planet_grass(planet: &Planet, grass: &HashMap<AssetId, Grass>) -> Option<PlanetGrass> {
    let layers: Vec<(Grass, GroundRule)> = planet
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
            grass
                .get(&layer.grass.id())
                .map(|g| (g.clone(), layer.rule))
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

    // A camera alone captures at start; UI beside it makes menu mode; a menu
    // driver keeps the cursor free; a UI-only world neither.
    #[test]
    fn the_cursor_follows_camera_ui_and_driver() {
        let policy = |camera: bool, ui: bool, driven: bool| {
            let mut world = World::new();
            if camera {
                world.push(Camera3D::bake(crate::components::cook::Camera3D::default()));
            }
            if ui {
                world.push(Screen::default());
            }
            if driven {
                world.insert_resource(MenuOverride::default());
            }
            cursor_policy(&world.context())
        };
        let p = policy(true, false, false);
        assert!(p.capture_at_start && !p.menu_mode);
        let p = policy(true, true, false);
        assert!(!p.capture_at_start && p.menu_mode);
        assert!(!policy(true, false, true).capture_at_start);
        assert_eq!(
            policy(false, true, false),
            CursorPolicy {
                menu_mode: false,
                capture_at_start: false
            }
        );
    }

    #[test]
    fn layers_resolve_through_their_grass() {
        let grass: HashMap<AssetId, Grass> = [(AssetId(0), Grass::default())].into();
        let layer = |id: u32| TerrainLayer {
            grass: Ref::new(AssetId(id)),
            density_mask: None,
            ..TerrainLayer::default()
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
            (Vec::new(), [Vec::new(), Vec::new()])
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
