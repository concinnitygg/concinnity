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
