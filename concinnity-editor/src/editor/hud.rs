// src/editor/hud.rs
//
// The editor HUD's per-frame geometry, hit-testing, and layout. This lives in
// the editor crate (not in a client ECS system) so no editor code is compiled
// into the shipped runtime: the HUD is driven from the editor's `DebugHook`
// tick, which runs only under `cn editor`.
//
// The buttons are plain `Sprite` + `TextLabel` components (injected by
// `inject.rs` at reserved ids). Each frame the hook re-anchors them flush to the
// window's top-right corner from the live viewport and hit-tests clicks against
// their window-space rects. Running in the tick (before the world step) means
// the repositioning applies the same frame GraphicsSystem draws it.

use crate::assets::{FrameInput, Sprite, TextAlign, TextLabel};
use crate::ecs::World;
use crate::ecs::asset_id::AssetId;

// Reserved asset-id range for the runtime-injected editor HUD. Interned ids are
// dense from 0 and a real world never approaches this range, so a fixed high
// base is collision-free without scanning the world. These ids are never
// interned and never serialized to a blob.
const ID_BASE: u32 = 0x3000_0000;
pub(crate) const SAVE_BUTTON: AssetId = AssetId(ID_BASE);
pub(crate) const SAVE_LABEL: AssetId = AssetId(ID_BASE + 1);
pub(crate) const ADD_BUTTON: AssetId = AssetId(ID_BASE + 2);
pub(crate) const ADD_LABEL: AssetId = AssetId(ID_BASE + 3);

// Button geometry, in window pixels. The SAVE button is a flush-cornered square;
// the add button sits to its left with a small gap. Zero margin keeps SAVE hard
// against the window's top-right corner.
pub(crate) const BTN_H: f32 = 88.0;
const SAVE_W: f32 = 88.0;
const ADD_W: f32 = 132.0;
const GAP: f32 = 8.0;

// Vertical offset of a button's label from the button top, chosen to sit the
// ~20px HUD font roughly on the button's vertical center without measuring the
// glyphs here (the font metrics live on GraphicsSystem).
pub(crate) const LABEL_TOP: f32 = BTN_H * 0.5 - 10.0;

// SAVE button fill with unsaved edits (active) vs none (inert); the white label
// text dims in the inert state. The add button keeps a fixed fill.
const SAVE_TINT_ACTIVE: [f32; 4] = [0.82, 0.14, 0.16, 1.0];
const SAVE_TINT_INERT: [f32; 4] = [0.26, 0.26, 0.30, 1.0];
const ADD_TINT: [f32; 4] = [0.20, 0.34, 0.52, 1.0];
const LABEL_ACTIVE: [f32; 3] = [1.0, 1.0, 1.0];
const LABEL_INERT: [f32; 3] = [0.55, 0.55, 0.58];

// A click the HUD resolved to one of its buttons.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum HudAction {
    Save,
    Add,
}

// The SAVE and add-asset button rects (`[x, y, w, h]`, window pixels) for a
// window `vw` pixels wide. Pure: the single source of truth both the layout
// pass and the hit test derive from.
pub(crate) fn layout(vw: f32) -> ([f32; 4], [f32; 4]) {
    let save = [vw - SAVE_W, 0.0, SAVE_W, BTN_H];
    let add = [vw - SAVE_W - GAP - ADD_W, 0.0, ADD_W, BTN_H];
    (save, add)
}

fn point_in(x: f32, y: f32, rect: [f32; 4]) -> bool {
    x >= rect[0] && x < rect[0] + rect[2] && y >= rect[1] && y < rect[1] + rect[3]
}

// Resolve a click at `(mx, my)` against the button layout for a `vw`-wide
// window. SAVE only reports when there are edits to persist (`dirty`); the add
// button always reports. Returns `None` for no click, an off-button click, or a
// SAVE click with nothing to save. Pure -- the hook maps the action to a method.
pub(crate) fn hit_test(mx: f32, my: f32, clicked: bool, dirty: bool, vw: f32) -> Option<HudAction> {
    if !clicked || vw <= 0.0 {
        return None;
    }
    let (save, add) = layout(vw);
    if dirty && point_in(mx, my, save) {
        Some(HudAction::Save)
    } else if point_in(mx, my, add) {
        Some(HudAction::Add)
    } else {
        None
    }
}

// Re-anchor the injected HUD sprites/labels to the top-right corner from the
// live viewport and colour the SAVE button by `dirty`. Reads the window width
// from the most recent `FrameInput` (published by GraphicsSystem); a no-op until
// one exists (frame 0) or on a zero-width window.
pub(crate) fn apply_layout(world: &mut World, dirty: bool) {
    let Some(vw) = world.query::<FrameInput>().last().map(|i| i.viewport[0]) else {
        return;
    };
    if vw <= 0.0 {
        return;
    }
    let (save, add) = layout(vw);
    let save_tint = if dirty {
        SAVE_TINT_ACTIVE
    } else {
        SAVE_TINT_INERT
    };
    let save_color = if dirty { LABEL_ACTIVE } else { LABEL_INERT };

    place_sprite(world, SAVE_BUTTON, save, save_tint);
    place_sprite(world, ADD_BUTTON, add, ADD_TINT);
    place_label(world, SAVE_LABEL, save, save_color);
    place_label(world, ADD_LABEL, add, LABEL_ACTIVE);
}

// Move + resize the Sprite with `id` to `rect` and set its tint, if present.
fn place_sprite(world: &mut World, id: AssetId, rect: [f32; 4], tint: [f32; 4]) {
    for s in world.query_mut::<Sprite>() {
        if s.asset_id == id {
            s.x = rect[0];
            s.y = rect[1];
            s.width = rect[2];
            s.height = rect[3];
            s.tint = tint;
            s.visible = true;
            break;
        }
    }
}

// Center the TextLabel with `id` horizontally over `rect` and set its colour.
fn place_label(world: &mut World, id: AssetId, rect: [f32; 4], color: [f32; 3]) {
    for l in world.query_mut::<TextLabel>() {
        if l.asset_id == id {
            l.x = rect[0] + rect[2] * 0.5;
            l.y = rect[1] + LABEL_TOP;
            l.align = TextAlign::Center;
            l.color = color;
            l.visible = true;
            break;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // Build a world holding the two button sprites, their labels, and a
    // FrameInput of the given window width.
    fn hud_world(vw: f32) -> World {
        let mut world = World::new_empty();
        for id in [SAVE_BUTTON, ADD_BUTTON] {
            world.add_component(Sprite {
                asset_id: id,
                ..Default::default()
            });
        }
        for id in [SAVE_LABEL, ADD_LABEL] {
            world.add_component(TextLabel {
                asset_id: id,
                ..Default::default()
            });
        }
        world.add_component(FrameInput {
            viewport: [vw, 720.0],
            ..Default::default()
        });
        world
    }

    fn sprite(world: &World, id: AssetId) -> Sprite {
        world
            .query::<Sprite>()
            .find(|s| s.asset_id == id)
            .cloned()
            .expect("sprite present")
    }

    fn label(world: &World, id: AssetId) -> TextLabel {
        world
            .query::<TextLabel>()
            .find(|l| l.asset_id == id)
            .cloned()
            .expect("label present")
    }

    // The SAVE rect is a flush top-right square; the add rect sits to its left
    // with the gap and does not overlap.
    #[test]
    fn layout_anchors_save_flush_top_right() {
        let (save, add) = layout(1280.0);
        assert_eq!(save[0] + save[2], 1280.0, "SAVE right edge at window width");
        assert_eq!(save[1], 0.0, "SAVE top edge at window top");
        assert_eq!(save, [1192.0, 0.0, 88.0, 88.0]);
        assert!(add[0] + add[2] <= save[0], "add button left of SAVE");
        assert_eq!(add[0] + add[2], save[0] - GAP, "gap between add and SAVE");
    }

    // The layout follows the window width so a resize re-anchors flush.
    #[test]
    fn layout_tracks_window_width() {
        let (save, _) = layout(1024.0);
        assert_eq!(save[0] + save[2], 1024.0);
        assert_eq!(save[0], 936.0);
    }

    #[test]
    fn hit_test_save_fires_only_when_dirty() {
        // A point inside the SAVE square at a 1280-wide window.
        let (sx, sy) = (1240.0, 40.0);
        assert_eq!(
            hit_test(sx, sy, true, true, 1280.0),
            Some(HudAction::Save),
            "dirty SAVE click fires"
        );
        assert_eq!(
            hit_test(sx, sy, true, false, 1280.0),
            None,
            "clean SAVE click is inert"
        );
    }

    #[test]
    fn hit_test_add_fires_regardless_of_dirty() {
        // The add rect spans ~[1052, 0, 132, 88] at 1280 wide.
        let (ax, ay) = (1100.0, 40.0);
        assert_eq!(hit_test(ax, ay, true, false, 1280.0), Some(HudAction::Add));
        assert_eq!(hit_test(ax, ay, true, true, 1280.0), Some(HudAction::Add));
    }

    #[test]
    fn hit_test_ignores_non_clicks_and_misses() {
        // No click: never fires even over a button.
        assert_eq!(hit_test(1240.0, 40.0, false, true, 1280.0), None);
        // A click in the empty middle of the screen hits nothing.
        assert_eq!(hit_test(640.0, 360.0, true, true, 1280.0), None);
        // A zero-width window (pre-first-frame) never resolves a button.
        assert_eq!(hit_test(0.0, 0.0, true, true, 0.0), None);
    }

    // apply_layout anchors the sprites to the true corner and colours SAVE red
    // (white label) while dirty.
    #[test]
    fn apply_layout_anchors_and_colours_dirty() {
        let mut world = hud_world(1024.0);
        apply_layout(&mut world, true);

        let save = sprite(&world, SAVE_BUTTON);
        assert_eq!(save.x + save.width, 1024.0, "SAVE flush to window right");
        assert_eq!(save.tint, SAVE_TINT_ACTIVE, "SAVE red when dirty");
        assert!(save.visible);
        // The label centers over the button and reads active white.
        let lbl = label(&world, SAVE_LABEL);
        assert_eq!(lbl.x, save.x + save.width * 0.5);
        assert_eq!(lbl.align, TextAlign::Center);
        assert_eq!(lbl.color, LABEL_ACTIVE);
        // The add button anchors to the left of SAVE.
        let add = sprite(&world, ADD_BUTTON);
        assert!(add.x + add.width <= save.x);
    }

    // With no edits, SAVE greys out and its label dims.
    #[test]
    fn apply_layout_greys_save_when_clean() {
        let mut world = hud_world(1280.0);
        apply_layout(&mut world, false);
        assert_eq!(sprite(&world, SAVE_BUTTON).tint, SAVE_TINT_INERT);
        assert_eq!(label(&world, SAVE_LABEL).color, LABEL_INERT);
    }

    // No FrameInput yet (frame 0): apply_layout is a no-op, leaving the injected
    // placeholder positions untouched rather than moving to a bogus corner.
    #[test]
    fn apply_layout_noops_without_frame_input() {
        let mut world = World::new_empty();
        world.add_component(Sprite {
            asset_id: SAVE_BUTTON,
            x: 7.0,
            ..Default::default()
        });
        apply_layout(&mut world, true);
        assert_eq!(
            sprite(&world, SAVE_BUTTON).x,
            7.0,
            "untouched without input"
        );
    }
}
