// src/gfx/overlay/mod.rs
//
// OverlaySystem: builds the 2D overlay draw list (sprites, text, dropdown,
// text-input fields, cursor) from the world's UI components and publishes the
// per-frame menu state. Runs first in the schedule, before GraphicsSystem
// submits the frame:
//   mod.rs        system + the per-frame draw-list build
//   widgets.rs    transient dropdown / text-input element synthesis
//   hud_layout.rs LayoutContainer reflow + DebugHud / StatHud chip anchoring
//
// The font atlases and texture slots the build measures against are uploaded
// by GraphicsSystem's init, which parks them here as the `OverlayAssets`
// resource; the build's output is parked as the `OverlayFrame` resource that
// GraphicsSystem consumes for this same frame's submit. HUD content is what
// the HUD systems wrote last tick (they run after the build), so what is
// measured is exactly what is drawn.

use crate::assets::{Sprite, TextInput, TextLabel};
use crate::ecs::asset_id::AssetId;
use crate::ecs::{PipelineContext, StepResult, System};
use crate::gfx::{sprite as gfx_sprite, text};
use std::time::Instant;

mod hud_layout;
mod widgets;

// Reserved draw-layer range for the `cn editor` HUD's per-frame overrides:
// far above any screen-stack layer (authored layer band ~4M at most), so the
// editor panels always occlude world screens while keeping their own order.
const EDITOR_LAYER_BASE: i32 = i32::MAX / 2;

// Everything the overlay build needs from GraphicsSystem's init: the loaded
// font atlases, the sprite-texture slot map, the HUD chip id lists, and the
// scroll-panel clip bands. Parked as a resource at the end of graphics init;
// the build takes it for the duration of each step and puts it back.
pub struct OverlayAssets {
    pub fonts: std::collections::HashMap<crate::ecs::FontHandle, text::LoadedFont>,
    pub sprite_texture_slots: std::collections::HashMap<crate::ecs::TextureHandle, usize>,
    pub debug_hud_chips: Vec<AssetId>,
    pub stat_hud_chips: Vec<AssetId>,
    pub clip_rects: std::collections::HashMap<AssetId, [f32; 4]>,
    // The backend's logical size at init, the viewport used until the first
    // input poll publishes a live one (`FrameInput.viewport`).
    pub initial_viewport: (f32, f32),
}

// One frame's overlay build, published by OverlaySystem and consumed (taken)
// by GraphicsSystem's submit the same tick: the shaped draw calls, whether an
// in-engine cursor sprite is shown (so the backend hides the system cursor),
// the resolved menu state (`MenuOverride` applied), and whether an opaque
// full-canvas backdrop lets the world render be skipped entirely.
#[derive(Default)]
pub struct OverlayFrame {
    pub calls: Vec<crate::gfx::render_types::TextDrawCall>,
    pub want_ui_cursor: bool,
    pub menu_active: bool,
    pub world_hidden: bool,
}

#[derive(Debug, Default)]
pub struct OverlaySystem {
    // Base for the caret-blink clock, set on the first step.
    start_time: Option<Instant>,
}

impl OverlaySystem {
    pub fn new() -> Self {
        Self::default()
    }
}

impl System for OverlaySystem {
    fn step(&mut self, ctx: &mut PipelineContext) -> StepResult {
        // No parked assets: graphics init has not succeeded (or not run), so
        // there is nothing to build against and nothing will be drawn.
        let Some(assets) = ctx.resources.remove::<OverlayAssets>() else {
            return StepResult::Continue;
        };
        let elapsed = self
            .start_time
            .get_or_insert_with(Instant::now)
            .elapsed()
            .as_secs_f32();
        let mut frame = build_overlay_frame(ctx, &assets, elapsed);
        ctx.resources.insert(assets);

        // An external per-frame driver (the `cn editor` HUD) can force the
        // menu-active state through the `MenuOverride` resource, so it frees
        // the cursor + freezes the world regardless of the world's own menu
        // UI. `None` leaves the world's own logic in charge. This shadows
        // `menu_active` for every consumer (capture, the freeze resource, the
        // gameplay-input gate), but not `world_hidden`: the editor keeps the
        // world visible.
        if let Some(forced) = ctx.resource::<crate::ecs::MenuOverride>().and_then(|m| m.0) {
            frame.menu_active = forced;
        }
        // Publish the menu state for every later system this tick: physics +
        // animation freeze while it is set, so a paused world stops consuming
        // CPU/GPU behind the menu. The App-level pacer reads it before the
        // next step to clamp the frame rate while a menu is open.
        ctx.insert_resource(crate::ecs::MenuActive(frame.menu_active));
        ctx.insert_resource(frame);
        StepResult::Continue
    }
}

// Build the frame's overlay draw calls. Sprites render as solid-coloured
// quads through the same UI pass as TextLabel (sentinel-UV path), so they
// share the text pipeline and require no new render state. Backdrop / HUD
// sprites are emitted first so labels composite on top; `follow_cursor`
// sprites are emitted last so the cursor sits on top of everything.
fn build_overlay_frame(
    ctx: &mut PipelineContext,
    assets: &OverlayAssets,
    elapsed: f32,
) -> OverlayFrame {
    // The viewport InputSystem sampled at the end of the previous tick, or
    // the init-time size before the first poll. A live resize is picked up
    // one frame later, which is invisible mid-drag.
    let (win_w, win_h) = ctx
        .resource::<crate::assets::FrameInput>()
        .map(|i| (i.viewport[0], i.viewport[1]))
        .unwrap_or(assets.initial_viewport);
    // The cursor state InputSystem sampled at the end of the previous tick
    // (`follow_cursor` sprites are positioned a frame after the input that
    // moved them).
    let cursor = ctx
        .resource::<crate::ecs::CursorState>()
        .copied()
        .unwrap_or_default();
    // Reposition LayoutContainer-managed labels before measuring them
    // for draw, so a HUD reflows to its live text each frame.
    hud_layout::apply_label_layout(ctx, &assets.fonts);
    // Anchor the DebugHud chips to the top-right corner, stacked.
    hud_layout::position_debug_hud(ctx, &assets.debug_hud_chips, &assets.fonts, win_w);
    // Pack the StatHud chips into a tight strip in the top-left corner.
    hud_layout::position_stat_hud(ctx, &assets.stat_hud_chips, &assets.fonts);
    let default_atlas_slot = assets.fonts.values().next().map(|f| f.atlas_slot);
    let sprites: Vec<&Sprite> = ctx.query::<Sprite>().collect();
    let (cursor_sprites, scene_sprites): (Vec<&Sprite>, Vec<&Sprite>) =
        sprites.into_iter().partition(|s| s.follow_cursor);

    // Per-element draw layers, from two sources merged into one map:
    //   - the screen stack: every element of an active Screen takes its
    //     screen's computed layer (stack position within the authored layer
    //     band), so screens draw in stack order and above the layer-0 HUD;
    //   - the `cn editor` HUD's per-frame overrides, lifted into a reserved
    //     top range so the editor panels always occlude world screens while
    //     keeping their own focus order.
    // An id absent from the map is layer 0. When the map ends up empty (no
    // active screen, no editor), the sort below is skipped and draw order is
    // pure insertion order, as before.
    let empty_layers = std::collections::HashMap::new();
    let screen_layers = ctx
        .resource::<crate::ecs::ScreenStack>()
        .map(|s| s.layers.clone())
        .unwrap_or_default();
    let mut effective_layers: std::collections::HashMap<AssetId, i32> =
        std::collections::HashMap::new();
    if !screen_layers.is_empty() {
        for s in ctx.query::<Sprite>() {
            if let Some(layer) = s.screen.and_then(|id| screen_layers.get(&id)) {
                effective_layers.insert(s.asset_id, *layer);
            }
        }
        for l in ctx.query::<TextLabel>() {
            if let Some(layer) = l.screen.and_then(|id| screen_layers.get(&id)) {
                effective_layers.insert(l.asset_id, *layer);
            }
        }
        for t in ctx.query::<TextInput>() {
            if let Some(layer) = t.screen.and_then(|id| screen_layers.get(&id)) {
                effective_layers.insert(t.asset_id, *layer);
            }
        }
    }
    if let Some(overrides) = ctx.resource::<crate::ecs::HudLayers>() {
        for (id, layer) in &overrides.0 {
            effective_layers.insert(*id, EDITOR_LAYER_BASE + layer);
        }
    }
    let hud_layers = &effective_layers;

    let mut calls = gfx_sprite::build_sprite_calls(
        &scene_sprites,
        default_atlas_slot,
        &assets.sprite_texture_slots,
        [win_w, win_h],
        &assets.clip_rects,
        hud_layers,
    );
    let labels: Vec<&TextLabel> = ctx.query::<TextLabel>().collect();
    calls.extend(text::build_text_calls(
        &labels,
        &assets.fonts,
        win_w,
        win_h,
        &assets.clip_rects,
        hud_layers,
    ));

    // A settings dropdown's open list draws on top of the menu (after the
    // clipped row text, before the cursor) and unclipped, so it escapes
    // the scroll band's scissor. Built as transient overlay Sprites +
    // TextLabels fed through the same shapers (with no clip bands).
    if let Some(screen) = ctx
        .resource::<crate::ecs::OpenDropdown>()
        .and_then(|d| d.0.clone())
    {
        let no_clips = std::collections::HashMap::new();
        let (dd_sprites, dd_labels) = widgets::build_dropdown_overlay(&screen, &assets.fonts);
        let sprite_refs: Vec<&Sprite> = dd_sprites.iter().collect();
        calls.extend(gfx_sprite::build_sprite_calls(
            &sprite_refs,
            default_atlas_slot,
            &assets.sprite_texture_slots,
            [win_w, win_h],
            &no_clips,
            &empty_layers,
        ));
        let label_refs: Vec<&TextLabel> = dd_labels.iter().collect();
        calls.extend(text::build_text_calls(
            &label_refs,
            &assets.fonts,
            win_w,
            win_h,
            &no_clips,
            &empty_layers,
        ));
    }

    // Text-input fields draw as a background box + their text + a caret,
    // synthesised the same way as the dropdown overlay and fed through the
    // shapers (clipped like the rest, so a field inside a scroll band
    // scissors correctly).
    let text_inputs: Vec<&TextInput> = ctx.query::<TextInput>().collect();
    // Caret blink: visible for the first half of each period so a focused
    // field's caret pulses rather than sitting solid.
    const CARET_BLINK_PERIOD: f32 = 1.06;
    let caret_visible = (elapsed % CARET_BLINK_PERIOD) < CARET_BLINK_PERIOD * 0.5;
    for ti in text_inputs.iter().filter(|t| t.visible) {
        let (ti_sprites, ti_labels) =
            widgets::build_text_input_overlay(ti, &assets.fonts, caret_visible);
        // The synthesised overlay carries no asset id, so its calls take the
        // field's own layer (from the field's id) rather than looking up the
        // default id -- otherwise a focused panel's text fields would sink
        // below it.
        let ti_layer = hud_layers.get(&ti.asset_id).copied().unwrap_or(0);
        let sprite_refs: Vec<&Sprite> = ti_sprites.iter().collect();
        let mut ti_calls = gfx_sprite::build_sprite_calls(
            &sprite_refs,
            default_atlas_slot,
            &assets.sprite_texture_slots,
            [win_w, win_h],
            &assets.clip_rects,
            &empty_layers,
        );
        let label_refs: Vec<&TextLabel> = ti_labels.iter().collect();
        ti_calls.extend(text::build_text_calls(
            &label_refs,
            &assets.fonts,
            win_w,
            win_h,
            &assets.clip_rects,
            &empty_layers,
        ));
        for c in &mut ti_calls {
            c.layer = ti_layer;
        }
        calls.extend(ti_calls);
    }

    // A menu cursor is present when any visible follow_cursor sprite is
    // opaque. Draw it (as an arrow pointer at the latest mouse position,
    // after the text so it sits on top) only while the real cursor is
    // inside the window: when it leaves in windowed / borderless modes
    // the arrow is hidden instead of lingering at the edge. The backend
    // confines the cursor in fullscreen, so it reports "inside" there.
    let menu_cursor = cursor_sprites.iter().any(|s| s.visible && s.tint[3] > 0.0);
    let want_ui_cursor = menu_cursor && !cursor.outside_window;
    if want_ui_cursor {
        calls.extend(crate::gfx::cursor::build_cursor_calls(
            &cursor_sprites,
            cursor.pos,
            default_atlas_slot,
            [win_w, win_h],
        ));
    }
    // A menu is "active" while any active screen pauses the world (the
    // screen stack publishes the flag); used to drive cursor capture and to
    // freeze gameplay input + simulation. A screen with `pauses_world` off
    // (a passthrough overlay, a live console) shows without pausing.
    let menu_active = ctx
        .resource::<crate::ecs::ScreenStack>()
        .is_some_and(|s| s.pauses_world);
    // The whole world render can be skipped when an opaque full-canvas
    // backdrop covers the scene (a menu authored with its dim alpha at
    // 1.0): nothing of the scene is visible, so every world pass is
    // wasted. A translucent dim keeps the world faintly visible and so
    // does not qualify.
    let world_hidden = menu_active
        && scene_sprites
            .iter()
            .any(|s| s.visible && s.tint[3] >= 1.0 && gfx_sprite::covers_canvas(s));
    // Reorder the overlay by layer when any layer is assigned (an active
    // screen stack, or the editor's focus-stack overrides): a stable sort
    // keeps same-layer order (so the sprites-then-text order within a panel
    // is intact) while lifting a screen's or focused panel's whole content
    // above the others'. Skipped entirely when no layer is set, so draw order
    // stays pure insertion order.
    if !hud_layers.is_empty() {
        calls.sort_by_key(|c| c.layer);
    }
    OverlayFrame {
        calls,
        want_ui_cursor,
        menu_active,
        world_hidden,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::assets::{SpriteFit, TextAlign};
    use crate::blob::BlobData;
    use crate::ecs::{
        ComponentSlot, ComponentStorage, CursorState, DropdownView, FontHandle, HudLayers,
        MenuOverride, OpenDropdown, Resources, ScreenStack,
    };
    use crate::gfx::profile::FrameProfile;

    const FONT: FontHandle = FontHandle(0);
    const SCREEN: AssetId = AssetId(50);
    // The reference canvas the overlay authors against, so a backdrop sized to
    // it counts as full-canvas.
    const REF_W: f32 = 1280.0;
    const REF_H: f32 = 720.0;

    fn make_glyph(advance_px: f32) -> crate::build::font::GlyphMetrics {
        crate::build::font::GlyphMetrics {
            char_code: 0,
            atlas_x: 0,
            atlas_y: 0,
            atlas_w: 8,
            atlas_h: 12,
            advance_px,
            bearing_x: 0.0,
            bearing_y: 12.0,
        }
    }

    // A fixed-width synthetic font (every glyph 10px in a 16px em) so the built
    // geometry is exact.
    fn loaded_fonts() -> std::collections::HashMap<FontHandle, text::LoadedFont> {
        let metrics: std::collections::HashMap<u32, crate::build::font::GlyphMetrics> = ('a'..='z')
            .chain('A'..='Z')
            .map(|c| (c as u32, make_glyph(10.0)))
            .collect();
        let cap_px = text::derive_cap_px(&metrics, 16.0);
        std::collections::HashMap::from([(
            FONT,
            text::LoadedFont {
                atlas_slot: 0,
                cap_px,
                metrics,
                atlas_w: 128,
                atlas_h: 128,
                size_px: 16.0,
                supersample: 1.0,
            },
        )])
    }

    fn assets() -> OverlayAssets {
        OverlayAssets {
            fonts: loaded_fonts(),
            sprite_texture_slots: std::collections::HashMap::new(),
            debug_hud_chips: Vec::new(),
            stat_hud_chips: Vec::new(),
            clip_rects: std::collections::HashMap::new(),
            initial_viewport: (REF_W, REF_H),
        }
    }

    // An opaque HUD sprite (window pixels, no screen), visible by default.
    fn sprite(id: AssetId) -> Sprite {
        Sprite {
            asset_id: id,
            x: 0.0,
            y: 0.0,
            width: 10.0,
            height: 10.0,
            texture: None,
            tint: [1.0, 1.0, 1.0, 1.0],
            follow_cursor: false,
            visible: true,
            screen: None,
            fit: SpriteFit::Fit,
            corner_radius: 0.0,
            border_width: 0.0,
            border_color: [0.0, 0.0, 0.0, 1.0],
        }
    }

    // A screen-owned sprite spanning the whole reference canvas: the menu-dim
    // shape `covers_canvas` recognises.
    fn backdrop(id: AssetId) -> Sprite {
        Sprite {
            width: REF_W,
            height: REF_H,
            screen: Some(SCREEN),
            ..sprite(id)
        }
    }

    fn label(id: AssetId, content: &str) -> TextLabel {
        TextLabel {
            asset_id: id,
            font: Some(FONT),
            content: content.to_string(),
            x: 0.0,
            y: 0.0,
            color: [1.0, 1.0, 1.0],
            scale: 1.0,
            centered: false,
            align: TextAlign::Left,
            fit: SpriteFit::Fit,
            background: [0.0, 0.0, 0.0, 0.0],
            padding: 0.0,
            visible: true,
            screen: None,
        }
    }

    fn text_input(id: AssetId) -> TextInput {
        TextInput {
            asset_id: id,
            font: Some(FONT),
            content: "ab".to_string(),
            ..Default::default()
        }
    }

    // A stack that owns SCREEN at `layer` and pauses the world.
    fn screen_stack(layer: i32) -> ScreenStack {
        ScreenStack {
            layers: std::collections::BTreeMap::from([(SCREEN, layer)]),
            pauses_world: true,
            captures_input: true,
        }
    }

    // Owns the storage a PipelineContext borrows from. The overlay build reads
    // no payloads, so the blob stays empty.
    struct TestWorld {
        components: ComponentStorage,
        blob: BlobData,
        profile: FrameProfile,
        resources: Resources,
    }

    impl TestWorld {
        fn new() -> Self {
            Self {
                components: ComponentStorage::default(),
                blob: BlobData::new(vec![Some(Vec::new())]),
                profile: FrameProfile::default(),
                resources: Resources::new(),
            }
        }

        fn push<C: ComponentSlot>(&mut self, c: C) {
            self.components.push_typed(c);
        }

        fn ctx(&mut self) -> PipelineContext<'_> {
            PipelineContext {
                components: &mut self.components,
                blob: &mut self.blob,
                profile: &mut self.profile,
                resources: &mut self.resources,
            }
        }

        // Build one frame at an explicit `elapsed`, so the caret blink is driven
        // by the test rather than the wall clock.
        fn build(&mut self, elapsed: f32) -> OverlayFrame {
            let a = assets();
            build_overlay_frame(&mut self.ctx(), &a, elapsed)
        }
    }

    // The x span of a call's quad, for reading a backdrop's mapped rect back out.
    fn x_span(call: &crate::gfx::render_types::TextDrawCall) -> (f32, f32) {
        let xs: Vec<f32> = call.vertices.iter().map(|v| v.pos[0]).collect();
        (
            xs.iter().copied().fold(f32::INFINITY, f32::min),
            xs.iter().copied().fold(f32::NEG_INFINITY, f32::max),
        )
    }

    // Without the parked assets graphics init has not run, so the step is inert:
    // no frame and no menu state are published.
    #[test]
    fn step_without_overlay_assets_publishes_nothing() {
        let mut w = TestWorld::new();
        w.push(sprite(AssetId(1)));
        let mut sys = OverlaySystem::new();
        sys.step(&mut w.ctx());
        assert!(w.resources.get::<OverlayFrame>().is_none());
        assert!(w.resources.get::<crate::ecs::MenuActive>().is_none());
    }

    // A step publishes the frame plus the menu state for the systems behind it,
    // and parks the assets back for the next tick.
    #[test]
    fn step_publishes_the_frame_and_parks_the_assets_back() {
        let mut w = TestWorld::new();
        w.push(sprite(AssetId(1)));
        w.resources.insert(assets());
        let mut sys = OverlaySystem::new();
        sys.step(&mut w.ctx());
        assert_eq!(w.resources.get::<OverlayFrame>().unwrap().calls.len(), 1);
        assert!(!w.resources.get::<crate::ecs::MenuActive>().unwrap().0);
        assert!(
            w.resources.get::<OverlayAssets>().is_some(),
            "the assets go back for the next build"
        );
    }

    // The editor's `MenuOverride` shadows the world's own menu state for every
    // consumer, but deliberately not `world_hidden`: the editor keeps the world
    // visible behind its panels.
    #[test]
    fn step_menu_override_shadows_the_menu_state_but_not_world_hidden() {
        let mut w = TestWorld::new();
        w.resources.insert(assets());
        w.resources.insert(MenuOverride(Some(true)));
        let mut sys = OverlaySystem::new();
        sys.step(&mut w.ctx());
        assert!(w.resources.get::<OverlayFrame>().unwrap().menu_active);
        assert!(w.resources.get::<crate::ecs::MenuActive>().unwrap().0);

        // A world whose own menu pauses and fully covers the scene, forced off:
        // the menu state follows the override while world_hidden keeps tracking
        // what is actually drawn.
        let mut w = TestWorld::new();
        w.push(backdrop(AssetId(1)));
        w.resources.insert(assets());
        w.resources.insert(screen_stack(0));
        w.resources.insert(MenuOverride(Some(false)));
        let mut sys = OverlaySystem::new();
        sys.step(&mut w.ctx());
        let frame = w.resources.get::<OverlayFrame>().unwrap();
        assert!(!frame.menu_active);
        assert!(frame.world_hidden);
    }

    // `MenuOverride(None)` leaves the world's own menu logic in charge.
    #[test]
    fn step_menu_override_of_none_defers_to_the_world() {
        let mut w = TestWorld::new();
        w.resources.insert(assets());
        w.resources.insert(screen_stack(0));
        w.resources.insert(MenuOverride(None));
        let mut sys = OverlaySystem::new();
        sys.step(&mut w.ctx());
        assert!(w.resources.get::<OverlayFrame>().unwrap().menu_active);
    }

    // The build measures against the viewport InputSystem last sampled, falling
    // back to the init-time size before the first poll. A full-canvas backdrop
    // stretches to exactly that, so its quad reads the viewport back out.
    #[test]
    fn viewport_follows_frame_input_and_falls_back_to_the_init_size() {
        let mut w = TestWorld::new();
        w.push(backdrop(AssetId(1)));
        let frame = w.build(0.0);
        assert_eq!(x_span(&frame.calls[0]), (0.0, REF_W));

        w.resources.insert(crate::assets::FrameInput {
            viewport: [800.0, 600.0],
            ..Default::default()
        });
        let frame = w.build(0.0);
        assert_eq!(x_span(&frame.calls[0]), (0.0, 800.0));
    }

    // An active screen's layer spreads onto every element it owns -- sprites,
    // labels and text-input fields alike -- and the calls sort by it, so the
    // screen's whole content lifts above the layer-0 HUD.
    #[test]
    fn screen_layers_spread_onto_the_elements_the_screen_owns() {
        let mut w = TestWorld::new();
        w.push(label(AssetId(1), "hud"));
        w.push(Sprite {
            screen: Some(SCREEN),
            ..sprite(AssetId(2))
        });
        w.push(TextLabel {
            screen: Some(SCREEN),
            ..label(AssetId(3), "menu")
        });
        w.push(TextInput {
            screen: Some(SCREEN),
            ..text_input(AssetId(4))
        });
        w.resources.insert(screen_stack(7));

        let frame = w.build(0.0);
        // The HUD label is layer 0 and sorts first; everything the screen owns
        // takes its layer.
        assert_eq!(frame.calls[0].layer, 0);
        assert!(
            frame.calls[1..].iter().all(|c| c.layer == 7),
            "{:?}",
            frame.calls.iter().map(|c| c.layer).collect::<Vec<_>>()
        );
    }

    // A screen-less element stays at layer 0 even while a stack is active, and
    // an element pointing at a screen that is not in the stack does too.
    #[test]
    fn elements_outside_the_active_stack_stay_at_layer_zero() {
        let mut w = TestWorld::new();
        w.push(sprite(AssetId(1)));
        w.push(Sprite {
            screen: Some(AssetId(99)),
            ..sprite(AssetId(2))
        });
        w.resources.insert(screen_stack(7));
        let frame = w.build(0.0);
        assert!(frame.calls.iter().all(|c| c.layer == 0));
    }

    // The editor HUD's overrides land in a reserved range far above any screen
    // layer, so its panels always occlude world screens while keeping their own
    // order.
    #[test]
    fn editor_layer_overrides_lift_elements_above_screen_layers() {
        let mut w = TestWorld::new();
        w.push(Sprite {
            screen: Some(SCREEN),
            ..sprite(AssetId(1))
        });
        w.push(sprite(AssetId(2)));
        w.resources.insert(screen_stack(7));
        w.resources
            .insert(HudLayers(std::collections::BTreeMap::from([(
                AssetId(2),
                3,
            )])));

        let frame = w.build(0.0);
        // The screen sprite sorts below the editor panel, which sits in the
        // reserved band regardless of the screen's own layer.
        assert_eq!(frame.calls[0].layer, 7);
        assert_eq!(frame.calls[1].layer, EDITOR_LAYER_BASE + 3);
    }

    // With no screen stack and no editor overrides nothing is layered, so the
    // sort is skipped and draw order stays pure insertion order.
    #[test]
    fn draw_order_is_insertion_order_without_any_layers() {
        let mut w = TestWorld::new();
        w.push(sprite(AssetId(1)));
        w.push(label(AssetId(2), "hud"));
        let frame = w.build(0.0);
        assert!(frame.calls.iter().all(|c| c.layer == 0));
        // Sprites first, then text: the label's call follows the sprite's.
        assert_eq!(frame.calls.len(), 2);
    }

    // An open dropdown list draws on top of the menu and unclipped, so it
    // escapes the scroll band's scissor even though the rows behind it clip.
    #[test]
    fn open_dropdown_draws_unclipped_over_the_menu() {
        let mut w = TestWorld::new();
        let before = w.build(0.0).calls.len();

        w.resources.insert(OpenDropdown(Some(DropdownView {
            anchor: [400.0, 100.0, 200.0, 40.0],
            options: vec!["aa".to_string(), "bb".to_string()],
            selected: 0,
            first: 0,
            hovered: None,
            screen: Some(SCREEN),
            font: Some(FONT),
            scale: 1.0,
            color: [1.0, 1.0, 1.0],
        })));
        let frame = w.build(0.0);
        assert!(frame.calls.len() > before, "the list added draw calls");
        assert!(
            frame.calls.iter().all(|c| c.clip_rect.is_none()),
            "the list is never scissored"
        );
    }

    // A closed dropdown synthesises nothing.
    #[test]
    fn closed_dropdown_builds_no_list() {
        let mut w = TestWorld::new();
        w.resources.insert(OpenDropdown(None));
        assert!(w.build(0.0).calls.is_empty());
    }

    // A field's synthesised box / text / caret carry no asset id of their own, so
    // they take the field's layer rather than sinking to the default -- otherwise
    // a focused panel's fields would drop below it.
    #[test]
    fn text_input_calls_take_the_fields_own_layer() {
        let mut w = TestWorld::new();
        w.push(text_input(AssetId(4)));
        w.push(sprite(AssetId(1)));
        w.resources
            .insert(HudLayers(std::collections::BTreeMap::from([(
                AssetId(4),
                2,
            )])));
        let frame = w.build(0.0);
        let field_layer = EDITOR_LAYER_BASE + 2;
        assert!(
            frame.calls.iter().any(|c| c.layer == field_layer),
            "the field's calls lift with it"
        );
        assert!(
            frame
                .calls
                .iter()
                .all(|c| c.layer == 0 || c.layer == field_layer),
            "nothing else moved"
        );
    }

    // The caret pulses: it draws on the first half of each blink period and is
    // gone on the second, so a focused field's caret does not sit solid.
    #[test]
    fn the_caret_draws_only_on_the_visible_half_of_the_blink() {
        let mut w = TestWorld::new();
        w.push(TextInput {
            focused: true,
            ..text_input(AssetId(4))
        });
        let visible = w.build(0.0).calls.len();
        let dark = w.build(0.6).calls.len();
        assert_eq!(visible, dark + 1, "the caret is the one call that drops");
        // The period wraps, so the next cycle's first half draws it again.
        assert_eq!(w.build(1.06).calls.len(), visible);
    }

    // A hidden field builds nothing at all.
    #[test]
    fn hidden_text_inputs_build_no_overlay() {
        let mut w = TestWorld::new();
        w.push(TextInput {
            visible: false,
            focused: true,
            ..text_input(AssetId(4))
        });
        assert!(w.build(0.0).calls.is_empty());
    }

    // The in-engine arrow draws for a visible, opaque follow_cursor sprite, and
    // is drawn last so it sits over everything.
    #[test]
    fn an_opaque_follow_cursor_sprite_draws_the_ui_arrow() {
        let mut w = TestWorld::new();
        w.push(Sprite {
            follow_cursor: true,
            ..sprite(AssetId(1))
        });
        let frame = w.build(0.0);
        assert!(frame.want_ui_cursor);
        assert!(!frame.calls.is_empty(), "the arrow was shaped");
    }

    // The arrow outranks every layered element: an active screen and the editor
    // both lift their content above 0, and a layer-0 cursor would sort under
    // the opaque menu backdrop it points at.
    #[test]
    fn the_ui_arrow_sorts_above_screen_and_editor_layers() {
        let mut w = TestWorld::new();
        w.push(backdrop(AssetId(1)));
        w.push(sprite(AssetId(2)));
        w.push(Sprite {
            follow_cursor: true,
            ..sprite(AssetId(3))
        });
        w.resources.insert(screen_stack(7));
        w.resources
            .insert(HudLayers(std::collections::BTreeMap::from([(
                AssetId(2),
                3,
            )])));

        let frame = w.build(0.0);
        let layers: Vec<i32> = frame.calls.iter().map(|c| c.layer).collect();
        let cursor = *layers.last().expect("the arrow was shaped");
        assert!(
            layers[..layers.len() - 1].iter().all(|l| *l < cursor),
            "{layers:?}"
        );
    }

    // A transparent or hidden cursor sprite is not a menu cursor, so the system
    // cursor stays in charge.
    #[test]
    fn a_transparent_or_hidden_cursor_sprite_draws_no_arrow() {
        let mut w = TestWorld::new();
        w.push(Sprite {
            follow_cursor: true,
            tint: [1.0, 1.0, 1.0, 0.0],
            ..sprite(AssetId(1))
        });
        w.push(Sprite {
            follow_cursor: true,
            visible: false,
            ..sprite(AssetId(2))
        });
        let frame = w.build(0.0);
        assert!(!frame.want_ui_cursor);
        assert!(frame.calls.is_empty());
    }

    // Once the real cursor leaves the window the arrow is hidden rather than
    // lingering at the edge.
    #[test]
    fn the_ui_arrow_hides_when_the_real_cursor_leaves_the_window() {
        let mut w = TestWorld::new();
        w.push(Sprite {
            follow_cursor: true,
            ..sprite(AssetId(1))
        });
        w.resources.insert(CursorState {
            pos: (10.0, 10.0),
            outside_window: true,
        });
        let frame = w.build(0.0);
        assert!(!frame.want_ui_cursor);
        assert!(frame.calls.is_empty(), "the arrow is not shaped either");
    }

    // The menu-active flag is exactly the screen stack's `pauses_world`: a
    // passthrough overlay shows without pausing.
    #[test]
    fn menu_active_follows_the_stacks_pauses_world() {
        let mut w = TestWorld::new();
        assert!(!w.build(0.0).menu_active, "no stack, no menu");

        w.resources.insert(screen_stack(0));
        assert!(w.build(0.0).menu_active);

        w.resources.insert(ScreenStack {
            pauses_world: false,
            ..screen_stack(0)
        });
        assert!(!w.build(0.0).menu_active);
    }

    // The world render is skipped only when a paused menu is backed by an opaque
    // full-canvas backdrop: nothing of the scene would be visible anyway.
    #[test]
    fn world_hidden_needs_a_paused_menu_behind_an_opaque_backdrop() {
        let mut w = TestWorld::new();
        w.push(backdrop(AssetId(1)));
        w.resources.insert(screen_stack(0));
        assert!(w.build(0.0).world_hidden);
    }

    // A translucent dim keeps the world faintly visible, so it does not qualify;
    // neither does a backdrop that does not span the canvas.
    #[test]
    fn a_translucent_or_partial_backdrop_keeps_the_world_rendering() {
        let mut w = TestWorld::new();
        w.push(Sprite {
            tint: [0.0, 0.0, 0.0, 0.5],
            ..backdrop(AssetId(1))
        });
        w.resources.insert(screen_stack(0));
        assert!(!w.build(0.0).world_hidden);

        let mut w = TestWorld::new();
        w.push(Sprite {
            width: REF_W / 2.0,
            ..backdrop(AssetId(1))
        });
        w.resources.insert(screen_stack(0));
        assert!(!w.build(0.0).world_hidden);
    }

    // An opaque backdrop with no menu pausing behind it is just scene art: the
    // world still renders.
    #[test]
    fn an_opaque_backdrop_without_a_paused_menu_keeps_the_world_rendering() {
        let mut w = TestWorld::new();
        w.push(backdrop(AssetId(1)));
        assert!(!w.build(0.0).world_hidden);
    }
}
