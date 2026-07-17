// src/editor/hud.rs
//
// The editor HUD's top bar: a slim full-width strip in the shared chrome tint
// (matching the floating panels, like a modern macOS toolbar) holding the Save
// and View buttons as compact rounded chips at its right end. This lives in the
// editor crate (not in a client ECS system) so no editor code is compiled into
// the shipped runtime: the HUD is driven from the editor's `DebugHook` tick,
// which runs only under `cn editor`.
//
// These are plain `Sprite` + `TextLabel` components (injected by `inject.rs` at
// reserved ids). Each frame the hook re-anchors the bar to the window width
// from the live viewport and hit-tests clicks. Save persists + live-swaps the
// world; View opens / closes the View panel (`view.rs`), which in turn toggles
// the Assets, Preview, and Templates panels. Running in the tick (before the
// world step) means the layout applies the same frame GraphicsSystem draws it.
// The whole HUD toggles with F1 (see `hook.rs`).

use super::registry::ID_BASE;
use super::theme;
use super::widget::{self, place_rounded, place_sprite, point_in};
use crate::assets::{FrameInput, TextAlign};
use crate::ecs::World;
use crate::ecs::asset_id::AssetId;
pub(crate) const SAVE_BUTTON: AssetId = AssetId(ID_BASE);
pub(crate) const SAVE_LABEL: AssetId = AssetId(ID_BASE + 1);
pub(crate) const VIEW_BUTTON: AssetId = AssetId(ID_BASE + 2);
pub(crate) const VIEW_LABEL: AssetId = AssetId(ID_BASE + 3);
pub(crate) const BAR_BG: AssetId = AssetId(ID_BASE + 4);

// Bar + button geometry, in window pixels. The bar spans the window top edge;
// the chips sit vertically centered at its right end. On macOS the window's
// traffic-light buttons float over the bar's left end (the editor drops the OS
// title bar there), so the bar doubles as their backdrop.
pub(crate) const BAR_H: f32 = 40.0;
pub(crate) const BTN_H: f32 = 26.0;
const SAVE_W: f32 = 64.0;
const VIEW_W: f32 = 72.0;
const GAP: f32 = 8.0;
const MARGIN: f32 = 8.0;

// Vertical offset of a chip's label from the chip top, centering the scaled
// editor text without measuring glyphs here (the font metrics live on
// GraphicsSystem).
pub(crate) const LABEL_TOP: f32 = BTN_H * 0.5 - theme::TEXT_HALF;

const SAVE_TINT_ACTIVE: [f32; 4] = [0.72, 0.18, 0.22, 1.0];
const LABEL_ACTIVE: [f32; 3] = [1.0, 1.0, 1.0];

// Per-frame top-bar state the hook hands to `apply_layout`.
#[derive(Debug, Clone, Copy)]
pub(crate) struct HudState {
    // Are there unsaved edits (Save active)?
    pub dirty: bool,
    // Is the View panel open (accents the View chip)?
    pub view_open: bool,
    // Is the whole HUD shown (F1 toggle)?
    pub visible: bool,
}

// A click the top bar resolved to one of its controls.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum HudAction {
    // The Save button, while there are edits to persist.
    Save,
    // The View button: open / close the View panel.
    ToggleView,
    // A click on the bar that hits no chip: swallowed so it cannot reach the
    // world behind the bar.
    Consume,
}

// The Save and View chip rects (`[x, y, w, h]`, window pixels) for a `vw`-wide
// window. Pure: the layout pass and the hit test both derive from it.
pub(crate) fn layout(vw: f32) -> ([f32; 4], [f32; 4]) {
    let y = (BAR_H - BTN_H) * 0.5;
    let save = [vw - MARGIN - SAVE_W, y, SAVE_W, BTN_H];
    let view = [save[0] - GAP - VIEW_W, y, VIEW_W, BTN_H];
    (save, view)
}

// The y where the body region (the floating panels' default anchors) begins:
// just below the bar.
pub(crate) fn body_top() -> f32 {
    BAR_H + 8.0
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
    } else if my < BAR_H && mx >= 0.0 && mx < vw {
        // The bar itself: swallow the click so it cannot fall through to the
        // world (and the hook dismisses any open overlays).
        Some(HudAction::Consume)
    } else {
        None
    }
}

// Re-anchor the bar to the window width from the live viewport and colour the
// Save + View chips by state. Hides the entire HUD when `state.visible` is
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
        theme::BUTTON_TINT
    };
    let save_color = if state.dirty {
        LABEL_ACTIVE
    } else {
        theme::LABEL_DIM
    };
    let view_tint = if state.view_open {
        theme::ACCENT_TINT
    } else {
        theme::BUTTON_TINT
    };

    place_sprite(
        world,
        BAR_BG,
        [0.0, 0.0, vw, BAR_H],
        theme::CHROME_TINT,
        true,
    );
    place_rounded(
        world,
        SAVE_BUTTON,
        save,
        save_tint,
        theme::CONTROL_RADIUS,
        true,
    );
    place_rounded(
        world,
        VIEW_BUTTON,
        view,
        view_tint,
        theme::CONTROL_RADIUS,
        true,
    );
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
    vec![BAR_BG, SAVE_BUTTON, VIEW_BUTTON]
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

// Position + colour + show/hide a fixed-content label (a top-bar chip).
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

    // The two chips pack right-to-left inside the bar without overlapping.
    #[test]
    fn layout_packs_two_chips_inside_the_bar() {
        let (save, view) = layout(1280.0);
        assert_eq!(save[0] + save[2], 1280.0 - MARGIN, "Save inset from right");
        assert_eq!(view[0] + view[2], save[0] - GAP, "View left of Save");
        assert!(
            save[1] > 0.0 && save[1] + save[3] < BAR_H,
            "centered in bar"
        );
        assert!(body_top() > BAR_H, "panels anchor below the bar");
    }

    #[test]
    fn hit_test_resolves_each_control() {
        let (save, view) = layout(1280.0);
        let mid = |r: [f32; 4]| (r[0] + r[2] * 0.5, r[1] + r[3] * 0.5);
        let (sx, sy) = mid(save);
        assert_eq!(hit_test(sx, sy, true, true, 1280.0), Some(HudAction::Save));
        assert_eq!(
            hit_test(sx, sy, true, false, 1280.0),
            Some(HudAction::Consume),
            "a clean Save chip is inert; the bar still swallows the click"
        );
        let (vx, vy) = mid(view);
        assert_eq!(
            hit_test(vx, vy, true, false, 1280.0),
            Some(HudAction::ToggleView)
        );
        // Empty bar area: swallowed, never reaching the world behind the bar.
        assert_eq!(
            hit_test(100.0, BAR_H * 0.5, true, false, 1280.0),
            Some(HudAction::Consume)
        );
        // Below the bar is not top-bar territory: the click falls through.
        assert_eq!(hit_test(1180.0, BAR_H + 10.0, true, false, 1280.0), None);
    }

    #[test]
    fn hit_test_ignores_non_clicks_and_zero_width() {
        assert_eq!(hit_test(1240.0, 20.0, false, true, 1280.0), None);
        assert_eq!(hit_test(0.0, 0.0, true, true, 0.0), None);
    }

    // The bar spans the window and the chips are shown, Save coloured by dirty.
    #[test]
    fn apply_layout_shows_bar_and_chips() {
        let mut world = hud_world(1024.0, (0.0, 0.0));
        apply_layout(&mut world, state(true, false, true));
        let bar = sprite(&world, BAR_BG);
        assert!(bar.visible);
        assert_eq!((bar.x, bar.width, bar.height), (0.0, 1024.0, BAR_H));
        assert_eq!(bar.tint, theme::CHROME_TINT, "bar matches the panel chrome");
        assert_eq!(sprite(&world, SAVE_BUTTON).tint, SAVE_TINT_ACTIVE);
        assert!(
            sprite(&world, SAVE_BUTTON).corner_radius > 0.0,
            "chips are rounded"
        );
        for id in [BAR_BG, SAVE_BUTTON, VIEW_BUTTON] {
            assert!(sprite(&world, id).visible, "{id:?} shown");
        }
    }

    // A clean Save chip recedes to the neutral chip tint.
    #[test]
    fn apply_layout_dims_a_clean_save() {
        let mut world = hud_world(1280.0, (0.0, 0.0));
        apply_layout(&mut world, state(false, false, true));
        assert_eq!(sprite(&world, SAVE_BUTTON).tint, theme::BUTTON_TINT);
    }

    // The View chip takes the accent while the View panel is open.
    #[test]
    fn apply_layout_marks_open_view() {
        let mut world = hud_world(1280.0, (0.0, 0.0));
        apply_layout(&mut world, state(false, true, true));
        assert_eq!(sprite(&world, VIEW_BUTTON).tint, theme::ACCENT_TINT);
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
