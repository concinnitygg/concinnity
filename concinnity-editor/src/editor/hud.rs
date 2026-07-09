// src/editor/hud.rs
//
// The editor HUD's top bar: the SAVE / Assets / Templates buttons plus the
// Templates dropdown. This lives in the editor crate (not in a client ECS
// system) so no editor code is compiled into the shipped runtime: the HUD is
// driven from the editor's `DebugHook` tick, which runs only under `cn editor`.
//
// These are plain `Sprite` + `TextLabel` components (injected by `inject.rs` at
// reserved ids). Each frame the hook re-anchors the controls flush to the
// window's top-right corner from the live viewport, shows / hides the Templates
// dropdown, and hit-tests clicks. The Assets button opens the browse/add panel
// handled by `panel.rs` (a floating panel that defaults to sitting below this
// bar); the capture checkbox lives on the floating Preview panel (`preview.rs`).
// Running in the tick (before the world step) means the layout applies the same
// frame GraphicsSystem draws it. The whole HUD toggles with F1 (see `hook.rs`).

use super::widget::{self, place_sprite, point_in};
use crate::assets::{FrameInput, TextAlign};
use crate::ecs::World;
use crate::ecs::asset_id::AssetId;

// How many Templates dropdown rows there are, and the title of row `i`.
pub(crate) fn templates_len() -> usize {
    concinnity_templates::TEMPLATES.len()
}
fn template_title(i: usize) -> &'static str {
    concinnity_templates::TEMPLATES[i].title
}

// The number of dropdown rows to inject (the Templates menu is the only one that
// uses this shared family; the Assets panel has its own).
pub(crate) fn max_rows() -> usize {
    templates_len()
}

// Reserved asset-id range for the runtime-injected editor HUD. Interned ids are
// dense from 0 and a real world never approaches this range, so a fixed high
// base is collision-free without scanning the world. These ids are never
// interned and never serialized to a blob. (The `panel.rs` families live past
// `ID_BASE + 0x40`; keep the two ranges disjoint.)
const ID_BASE: u32 = 0x3000_0000;
pub(crate) const SAVE_BUTTON: AssetId = AssetId(ID_BASE);
pub(crate) const SAVE_LABEL: AssetId = AssetId(ID_BASE + 1);
pub(crate) const ASSETS_BUTTON: AssetId = AssetId(ID_BASE + 2);
pub(crate) const ASSETS_LABEL: AssetId = AssetId(ID_BASE + 3);
pub(crate) const TPL_BUTTON: AssetId = AssetId(ID_BASE + 4);
pub(crate) const TPL_LABEL: AssetId = AssetId(ID_BASE + 5);

// Templates dropdown row elements, in two contiguous sub-ranges.
pub(crate) fn dropdown_bg(i: usize) -> AssetId {
    AssetId(ID_BASE + 16 + i as u32)
}
pub(crate) fn dropdown_label(i: usize) -> AssetId {
    AssetId(ID_BASE + 32 + i as u32)
}

// Button geometry, in window pixels. SAVE is a flush-cornered square; Assets and
// Templates sit to its left. Zero margin keeps SAVE hard against the window's
// top-right corner. Below the buttons: the Templates dropdown (when open),
// right-aligned.
pub(crate) const BTN_H: f32 = 88.0;
const SAVE_W: f32 = 88.0;
const ASSETS_W: f32 = 132.0;
const TPL_W: f32 = 132.0;
const GAP: f32 = 8.0;
const DROP_W: f32 = 200.0;
const ROW_H: f32 = 40.0;

// Vertical offset of a button's label from the button top, chosen to sit the
// ~20px HUD font roughly on the button's vertical center without measuring the
// glyphs here (the font metrics live on GraphicsSystem).
pub(crate) const LABEL_TOP: f32 = BTN_H * 0.5 - 10.0;
const ROW_LABEL_TOP: f32 = ROW_H * 0.5 - 10.0;

const SAVE_TINT_ACTIVE: [f32; 4] = [0.82, 0.14, 0.16, 1.0];
const SAVE_TINT_INERT: [f32; 4] = [0.26, 0.26, 0.30, 1.0];
const ASSETS_TINT: [f32; 4] = [0.20, 0.34, 0.52, 1.0];
const ASSETS_TINT_OPEN: [f32; 4] = [0.30, 0.48, 0.68, 1.0];
const TPL_TINT: [f32; 4] = [0.28, 0.24, 0.44, 1.0];
const ROW_TINT: [f32; 4] = [0.14, 0.14, 0.17, 0.96];
const ROW_TINT_HOVER: [f32; 4] = [0.24, 0.26, 0.34, 0.98];
const LABEL_ACTIVE: [f32; 3] = [1.0, 1.0, 1.0];
const LABEL_INERT: [f32; 3] = [0.55, 0.55, 0.58];
const ROW_LABEL: [f32; 3] = [0.90, 0.90, 0.92];

// Per-frame top-bar state the hook hands to `apply_layout`.
#[derive(Debug, Clone, Copy)]
pub(crate) struct HudState {
    // Are there unsaved edits (SAVE active)?
    pub dirty: bool,
    // Is the Templates dropdown open?
    pub templates_open: bool,
    // Is the Assets panel open (brightens the Assets button)?
    pub panel_open: bool,
    // Is the whole HUD shown (F1 toggle)?
    pub visible: bool,
}

// A click the top bar resolved to one of its controls.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum HudAction {
    // The SAVE button, while there are edits to persist.
    Save,
    // The Assets button: open / close the browse-and-add panel.
    ToggleAssets,
    // The Templates button: open / close the templates dropdown.
    ToggleTemplates,
    // A Templates dropdown row (index): apply that template and close.
    PickTemplate(usize),
    // A click while the Templates dropdown is open that is not a row: close it.
    CloseTemplates,
}

// The SAVE, Assets, and Templates button rects (`[x, y, w, h]`, window pixels)
// for a `vw`-wide window. Pure: the layout pass and the hit test both derive
// from it.
pub(crate) fn layout(vw: f32) -> ([f32; 4], [f32; 4], [f32; 4]) {
    let save = [vw - SAVE_W, 0.0, SAVE_W, BTN_H];
    let assets = [save[0] - GAP - ASSETS_W, 0.0, ASSETS_W, BTN_H];
    let tpl = [assets[0] - GAP - TPL_W, 0.0, TPL_W, BTN_H];
    (save, assets, tpl)
}

// The y where the body region (the Assets panel's default anchor or the
// Templates dropdown) begins: directly below the top buttons.
pub(crate) fn body_top() -> f32 {
    BTN_H
}

// The rect of Templates dropdown row `i`, stacked below the buttons.
pub(crate) fn dropdown_row_rect(vw: f32, i: usize) -> [f32; 4] {
    [vw - DROP_W, BTN_H + i as f32 * ROW_H, DROP_W, ROW_H]
}

// The Templates dropdown row index under `(mx, my)`, bounded by the row count.
fn template_row_at(mx: f32, my: f32, vw: f32) -> Option<usize> {
    (0..templates_len()).find(|&i| point_in(mx, my, dropdown_row_rect(vw, i)))
}

// Resolve a top-bar click at `(mx, my)` for a `vw`-wide window. Pure -- the hook
// maps the action to a method and updates its own flags. Returns `None` for a
// click the top bar does not own (the hook then offers it to the panel).
//
// While the Templates dropdown is open it captures the body region: a row click
// picks it and any other click dismisses it.
pub(crate) fn hit_test(
    mx: f32,
    my: f32,
    clicked: bool,
    dirty: bool,
    templates_open: bool,
    vw: f32,
) -> Option<HudAction> {
    if !clicked || vw <= 0.0 {
        return None;
    }
    if templates_open {
        return match template_row_at(mx, my, vw) {
            Some(i) => Some(HudAction::PickTemplate(i)),
            None => Some(HudAction::CloseTemplates),
        };
    }
    let (save, assets, tpl) = layout(vw);
    if dirty && point_in(mx, my, save) {
        Some(HudAction::Save)
    } else if point_in(mx, my, assets) {
        Some(HudAction::ToggleAssets)
    } else if point_in(mx, my, tpl) {
        Some(HudAction::ToggleTemplates)
    } else {
        None
    }
}

// Re-anchor the top bar to the top-right corner from the live viewport, colour
// the SAVE button by state, and show the Templates dropdown rows when open.
// Hides the entire HUD when `state.visible` is false (F1). A no-op until a
// `FrameInput` exists (frame 0) or a zero-width window.
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
    let (save, assets, tpl) = layout(vw);
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
    let assets_tint = if state.panel_open {
        ASSETS_TINT_OPEN
    } else {
        ASSETS_TINT
    };

    place_sprite(world, SAVE_BUTTON, save, save_tint, true);
    place_sprite(world, ASSETS_BUTTON, assets, assets_tint, true);
    place_sprite(world, TPL_BUTTON, tpl, TPL_TINT, true);
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
        ASSETS_LABEL,
        centered(assets),
        LABEL_ACTIVE,
        TextAlign::Center,
        true,
    );
    place_label(
        world,
        TPL_LABEL,
        centered(tpl),
        LABEL_ACTIVE,
        TextAlign::Center,
        true,
    );

    // Templates dropdown rows: hide them all, then show them (with the hovered
    // row highlighted) when the dropdown is open.
    for i in 0..max_rows() {
        place_sprite(world, dropdown_bg(i), [0.0; 4], ROW_TINT, false);
        set_row_label(world, dropdown_label(i), [0.0, 0.0], "", false);
    }
    if state.templates_open {
        for i in 0..templates_len() {
            let rect = dropdown_row_rect(vw, i);
            let hovered = point_in(input.mouse_x, input.mouse_y, rect);
            let tint = if hovered { ROW_TINT_HOVER } else { ROW_TINT };
            place_sprite(world, dropdown_bg(i), rect, tint, true);
            set_row_label(
                world,
                dropdown_label(i),
                [rect[0] + 12.0, rect[1] + ROW_LABEL_TOP],
                template_title(i),
                true,
            );
        }
    }
}

// Every injected top-bar sprite / label id, so the F1-hidden pass can blank it.
fn all_sprite_ids() -> Vec<AssetId> {
    let mut ids = vec![SAVE_BUTTON, ASSETS_BUTTON, TPL_BUTTON];
    ids.extend((0..max_rows()).map(dropdown_bg));
    ids
}
fn all_label_ids() -> Vec<AssetId> {
    let mut ids = vec![SAVE_LABEL, ASSETS_LABEL, TPL_LABEL];
    ids.extend((0..max_rows()).map(dropdown_label));
    ids
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

// Position + colour + show/hide a fixed-content label (a button or the checkbox).
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

// Position a dropdown row's label and set its content (written per frame from the
// Templates list).
fn set_row_label(world: &mut World, id: AssetId, pos: [f32; 2], content: &str, visible: bool) {
    if let Some(l) = widget::label_mut(world, id) {
        l.x = pos[0];
        l.y = pos[1];
        l.align = TextAlign::Left;
        l.color = ROW_LABEL;
        l.visible = visible;
        if visible {
            l.content = content.to_string();
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::assets::{Sprite, TextLabel};

    fn state(dirty: bool, templates: bool, panel: bool, visible: bool) -> HudState {
        HudState {
            dirty,
            templates_open: templates,
            panel_open: panel,
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

    fn label(world: &World, id: AssetId) -> TextLabel {
        world
            .query::<TextLabel>()
            .find(|l| l.asset_id == id)
            .cloned()
            .expect("label present")
    }

    // The three buttons pack right-to-left from the corner without overlapping.
    #[test]
    fn layout_packs_three_buttons_from_the_corner() {
        let (save, assets, tpl) = layout(1280.0);
        assert_eq!(save[0] + save[2], 1280.0, "SAVE flush to the window right");
        assert_eq!(assets[0] + assets[2], save[0] - GAP, "Assets left of SAVE");
        assert_eq!(tpl[0] + tpl[2], assets[0] - GAP, "Templates left of Assets");
    }

    #[test]
    fn dropdown_stacks_directly_below_buttons() {
        assert_eq!(dropdown_row_rect(1280.0, 0)[1], BTN_H);
        assert_eq!(dropdown_row_rect(1280.0, 1)[1], BTN_H + ROW_H);
        assert_eq!(body_top(), BTN_H);
    }

    #[test]
    fn hit_test_closed_resolves_each_control() {
        let (save, assets, tpl) = layout(1280.0);
        let mid = |r: [f32; 4]| (r[0] + r[2] * 0.5, r[1] + r[3] * 0.5);
        let (sx, sy) = mid(save);
        assert_eq!(
            hit_test(sx, sy, true, true, false, 1280.0),
            Some(HudAction::Save)
        );
        assert_eq!(
            hit_test(sx, sy, true, false, false, 1280.0),
            None,
            "clean SAVE inert"
        );
        let (ax, ay) = mid(assets);
        assert_eq!(
            hit_test(ax, ay, true, false, false, 1280.0),
            Some(HudAction::ToggleAssets)
        );
        let (tx, ty) = mid(tpl);
        assert_eq!(
            hit_test(tx, ty, true, false, false, 1280.0),
            Some(HudAction::ToggleTemplates)
        );
        // Below the buttons is no longer top-bar territory (the capture checkbox
        // moved to the Preview panel): the click falls through.
        assert_eq!(
            hit_test(1180.0, BTN_H + 10.0, true, false, false, 1280.0),
            None
        );
    }

    #[test]
    fn hit_test_open_templates_picks_row_or_closes() {
        let r1 = dropdown_row_rect(1280.0, 1);
        assert_eq!(
            hit_test(r1[0] + 10.0, r1[1] + 10.0, true, false, true, 1280.0),
            Some(HudAction::PickTemplate(1))
        );
        // A row index past the Templates menu is not a row -> dismiss.
        let past = dropdown_row_rect(1280.0, templates_len());
        assert_eq!(
            hit_test(past[0] + 10.0, past[1] + 10.0, true, false, true, 1280.0),
            Some(HudAction::CloseTemplates)
        );
        assert_eq!(
            hit_test(640.0, 360.0, true, false, true, 1280.0),
            Some(HudAction::CloseTemplates)
        );
    }

    #[test]
    fn hit_test_ignores_non_clicks_and_zero_width() {
        assert_eq!(hit_test(1240.0, 40.0, false, true, false, 1280.0), None);
        assert_eq!(hit_test(0.0, 0.0, true, true, true, 0.0), None);
    }

    // Closed: buttons shown, SAVE coloured by dirty, all rows hidden.
    #[test]
    fn apply_layout_closed_shows_buttons_hides_rows() {
        let mut world = hud_world(1024.0, (0.0, 0.0));
        apply_layout(&mut world, state(true, false, false, true));
        assert_eq!(sprite(&world, SAVE_BUTTON).tint, SAVE_TINT_ACTIVE);
        for id in [SAVE_BUTTON, ASSETS_BUTTON, TPL_BUTTON] {
            assert!(sprite(&world, id).visible, "{id:?} shown");
        }
        for i in 0..max_rows() {
            assert!(!sprite(&world, dropdown_bg(i)).visible, "row {i} hidden");
        }
    }

    // The Assets button brightens while the panel is open.
    #[test]
    fn apply_layout_marks_open_panel() {
        let mut world = hud_world(1280.0, (0.0, 0.0));
        apply_layout(&mut world, state(false, false, true, true));
        assert_eq!(sprite(&world, ASSETS_BUTTON).tint, ASSETS_TINT_OPEN);
    }

    // Open Templates: rows show the template titles; hovered row highlighted.
    #[test]
    fn apply_layout_templates_menu_shows_title_rows() {
        let r0 = dropdown_row_rect(1280.0, 0);
        let mut world = hud_world(1280.0, (r0[0] + 10.0, r0[1] + 10.0));
        apply_layout(&mut world, state(false, true, false, true));
        for i in 0..templates_len() {
            assert!(sprite(&world, dropdown_bg(i)).visible, "tpl row {i} shown");
            assert_eq!(
                label(&world, dropdown_label(i)).content,
                concinnity_templates::TEMPLATES[i].title
            );
        }
        assert_eq!(
            sprite(&world, dropdown_bg(0)).tint,
            ROW_TINT_HOVER,
            "row 0 hovered"
        );
    }

    // F1 hidden: every top-bar element is blanked.
    #[test]
    fn apply_layout_hidden_blanks_everything() {
        let mut world = hud_world(1280.0, (0.0, 0.0));
        apply_layout(&mut world, state(true, true, true, true));
        apply_layout(&mut world, state(true, true, true, false));
        for id in all_sprite_ids() {
            assert!(!sprite(&world, id).visible, "sprite {id:?} hidden");
        }
    }
}
