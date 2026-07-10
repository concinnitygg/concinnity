// src/editor/hud.rs
//
// The editor HUD's top bar: the SAVE and View buttons. This lives in the editor
// crate (not in a client ECS system) so no editor code is compiled into the
// shipped runtime: the HUD is driven from the editor's `DebugHook` tick, which
// runs only under `cn editor`.
//
// These are plain `Sprite` + `TextLabel` components (injected by `inject.rs` at
// reserved ids). Each frame the hook re-anchors the buttons flush to the window's
// top-right corner from the live viewport and hit-tests clicks. SAVE persists +
// live-swaps the world; View opens / closes the View panel (`view.rs`), which in
// turn toggles the Assets, Preview, and Templates panels. Running in the tick
// (before the world step) means the layout applies the same frame GraphicsSystem
// draws it. The whole HUD toggles with F1 (see `hook.rs`).

use super::widget::{self, place_sprite, point_in};
use crate::assets::{FrameInput, TextAlign};
use crate::ecs::World;
use crate::ecs::asset_id::AssetId;

// Reserved asset-id range for the runtime-injected editor HUD. Interned ids are
// dense from 0 and a real world never approaches this range, so a fixed high
// base is collision-free without scanning the world. These ids are never
// interned and never serialized to a blob. (The panel families live past
// `ID_BASE + 0x40`; keep the two ranges disjoint.)
const ID_BASE: u32 = 0x3000_0000;
pub(crate) const SAVE_BUTTON: AssetId = AssetId(ID_BASE);
pub(crate) const SAVE_LABEL: AssetId = AssetId(ID_BASE + 1);
pub(crate) const VIEW_BUTTON: AssetId = AssetId(ID_BASE + 2);
pub(crate) const VIEW_LABEL: AssetId = AssetId(ID_BASE + 3);

// Button geometry, in window pixels. SAVE is a flush-cornered square; View sits
// to its left. Zero margin keeps SAVE hard against the window's top-right corner.
pub(crate) const BTN_H: f32 = 88.0;
const SAVE_W: f32 = 88.0;
const VIEW_W: f32 = 132.0;
const GAP: f32 = 8.0;

// Vertical offset of a button's label from the button top, chosen to sit the
// ~20px HUD font roughly on the button's vertical center without measuring the
// glyphs here (the font metrics live on GraphicsSystem).
pub(crate) const LABEL_TOP: f32 = BTN_H * 0.5 - 10.0;

const SAVE_TINT_ACTIVE: [f32; 4] = [0.82, 0.14, 0.16, 1.0];
const SAVE_TINT_INERT: [f32; 4] = [0.26, 0.26, 0.30, 1.0];
const VIEW_TINT: [f32; 4] = [0.20, 0.34, 0.52, 1.0];
const VIEW_TINT_OPEN: [f32; 4] = [0.30, 0.48, 0.68, 1.0];
const LABEL_ACTIVE: [f32; 3] = [1.0, 1.0, 1.0];
const LABEL_INERT: [f32; 3] = [0.55, 0.55, 0.58];

// Per-frame top-bar state the hook hands to `apply_layout`.
#[derive(Debug, Clone, Copy)]
pub(crate) struct HudState {
    // Are there unsaved edits (SAVE active)?
    pub dirty: bool,
    // Is the View panel open (brightens the View button)?
    pub view_open: bool,
    // Is the whole HUD shown (F1 toggle)?
    pub visible: bool,
}

// A click the top bar resolved to one of its controls.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum HudAction {
    // The SAVE button, while there are edits to persist.
    Save,
    // The View button: open / close the View panel.
    ToggleView,
}

// The SAVE and View button rects (`[x, y, w, h]`, window pixels) for a `vw`-wide
// window. Pure: the layout pass and the hit test both derive from it.
pub(crate) fn layout(vw: f32) -> ([f32; 4], [f32; 4]) {
    let save = [vw - SAVE_W, 0.0, SAVE_W, BTN_H];
    let view = [save[0] - GAP - VIEW_W, 0.0, VIEW_W, BTN_H];
    (save, view)
}

// The y where the body region (the floating panels' default anchors) begins:
// directly below the top buttons.
pub(crate) fn body_top() -> f32 {
    BTN_H
}

// Resolve a top-bar click at `(mx, my)` for a `vw`-wide window. Pure -- the hook
// maps the action to a method and updates its own flags. Returns `None` for a
// click the top bar does not own (the hook then offers it to the panels).
pub(crate) fn hit_test(mx: f32, my: f32, clicked: bool, dirty: bool, vw: f32) -> Option<HudAction> {
    if !clicked || vw <= 0.0 {
        return None;
    }
    let (save, view) = layout(vw);
    if dirty && point_in(mx, my, save) {
        Some(HudAction::Save)
    } else if point_in(mx, my, view) {
        Some(HudAction::ToggleView)
    } else {
        None
    }
}

// Re-anchor the top bar to the top-right corner from the live viewport and colour
// the SAVE + View buttons by state. Hides the entire HUD when `state.visible` is
// false (F1). A no-op until a `FrameInput` exists (frame 0) or a zero-width
// window.
pub(crate) fn apply_layout(world: &mut World, state: HudState) {
    if !state.visible {
        hide_all(world);
        return;
    }
    let Some(input) = world.query::<FrameInput>().last().cloned() else {
        return;
    };
    let vw = input.viewport[0];
    if vw <= 0.0 {
        return;
    }
    let (save, view) = layout(vw);
    let save_tint = if state.dirty {
        SAVE_TINT_ACTIVE
    } else {
        SAVE_TINT_INERT
    };
    let save_color = if state.dirty {
        LABEL_ACTIVE
    } else {
        LABEL_INERT
    };
    let view_tint = if state.view_open {
        VIEW_TINT_OPEN
    } else {
        VIEW_TINT
    };

    place_sprite(world, SAVE_BUTTON, save, save_tint, true);
    place_sprite(world, VIEW_BUTTON, view, view_tint, true);
    place_label(
        world,
        SAVE_LABEL,
        centered(save),
        save_color,
        TextAlign::Center,
        true,
    );
    place_label(
        world,
        VIEW_LABEL,
        centered(view),
        LABEL_ACTIVE,
        TextAlign::Center,
        true,
    );
}

// Every injected top-bar sprite / label id, so the F1-hidden pass can blank it.
fn all_sprite_ids() -> Vec<AssetId> {
    vec![SAVE_BUTTON, VIEW_BUTTON]
}
fn all_label_ids() -> Vec<AssetId> {
    vec![SAVE_LABEL, VIEW_LABEL]
}

// Every top-bar element id (sprites + labels), so the hook can pin the whole bar
// to the top draw layer above the floating panels.
pub(crate) fn all_ids() -> Vec<AssetId> {
    all_sprite_ids()
        .into_iter()
        .chain(all_label_ids())
        .collect()
}

fn hide_all(world: &mut World) {
    for id in all_sprite_ids() {
        widget::set_sprite_visible(world, id, false);
    }
    for id in all_label_ids() {
        widget::set_label_visible(world, id, false);
    }
}

fn centered(rect: [f32; 4]) -> [f32; 2] {
    [rect[0] + rect[2] * 0.5, rect[1] + LABEL_TOP]
}

// Position + colour + show/hide a fixed-content label (a top-bar button).
fn place_label(
    world: &mut World,
    id: AssetId,
    pos: [f32; 2],
    color: [f32; 3],
    align: TextAlign,
    visible: bool,
) {
    if let Some(l) = widget::label_mut(world, id) {
        l.x = pos[0];
        l.y = pos[1];
        l.align = align;
        l.color = color;
        l.visible = visible;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::assets::{Sprite, TextLabel};

    fn state(dirty: bool, view: bool, visible: bool) -> HudState {
        HudState {
            dirty,
            view_open: view,
            visible,
        }
    }

    fn hud_world(vw: f32, mouse: (f32, f32)) -> World {
        let mut world = World::new_empty();
        for id in all_sprite_ids() {
            world.add_component(Sprite {
                asset_id: id,
                ..Default::default()
            });
        }
        for id in all_label_ids() {
            world.add_component(TextLabel {
                asset_id: id,
                ..Default::default()
            });
        }
        world.add_component(FrameInput {
            viewport: [vw, 720.0],
            mouse_x: mouse.0,
            mouse_y: mouse.1,
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

    // The two buttons pack right-to-left from the corner without overlapping.
    #[test]
    fn layout_packs_two_buttons_from_the_corner() {
        let (save, view) = layout(1280.0);
        assert_eq!(save[0] + save[2], 1280.0, "SAVE flush to the window right");
        assert_eq!(view[0] + view[2], save[0] - GAP, "View left of SAVE");
        assert_eq!(body_top(), BTN_H);
    }

    #[test]
    fn hit_test_resolves_each_control() {
        let (save, view) = layout(1280.0);
        let mid = |r: [f32; 4]| (r[0] + r[2] * 0.5, r[1] + r[3] * 0.5);
        let (sx, sy) = mid(save);
        assert_eq!(hit_test(sx, sy, true, true, 1280.0), Some(HudAction::Save));
        assert_eq!(
            hit_test(sx, sy, true, false, 1280.0),
            None,
            "clean SAVE inert"
        );
        let (vx, vy) = mid(view);
        assert_eq!(
            hit_test(vx, vy, true, false, 1280.0),
            Some(HudAction::ToggleView)
        );
        // Below the buttons is not top-bar territory: the click falls through.
        assert_eq!(hit_test(1180.0, BTN_H + 10.0, true, false, 1280.0), None);
    }

    #[test]
    fn hit_test_ignores_non_clicks_and_zero_width() {
        assert_eq!(hit_test(1240.0, 40.0, false, true, 1280.0), None);
        assert_eq!(hit_test(0.0, 0.0, true, true, 0.0), None);
    }

    // Buttons shown, SAVE coloured by dirty.
    #[test]
    fn apply_layout_shows_buttons() {
        let mut world = hud_world(1024.0, (0.0, 0.0));
        apply_layout(&mut world, state(true, false, true));
        assert_eq!(sprite(&world, SAVE_BUTTON).tint, SAVE_TINT_ACTIVE);
        for id in [SAVE_BUTTON, VIEW_BUTTON] {
            assert!(sprite(&world, id).visible, "{id:?} shown");
        }
    }

    // The View button brightens while the View panel is open.
    #[test]
    fn apply_layout_marks_open_view() {
        let mut world = hud_world(1280.0, (0.0, 0.0));
        apply_layout(&mut world, state(false, true, true));
        assert_eq!(sprite(&world, VIEW_BUTTON).tint, VIEW_TINT_OPEN);
    }

    // F1 hidden: every top-bar element is blanked.
    #[test]
    fn apply_layout_hidden_blanks_everything() {
        let mut world = hud_world(1280.0, (0.0, 0.0));
        apply_layout(&mut world, state(true, true, true));
        apply_layout(&mut world, state(true, true, false));
        for id in all_sprite_ids() {
            assert!(!sprite(&world, id).visible, "sprite {id:?} hidden");
        }
    }
}
