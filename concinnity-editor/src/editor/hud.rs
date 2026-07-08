// src/editor/hud.rs
//
// The editor HUD's per-frame geometry, hit-testing, and layout. This lives in
// the editor crate (not in a client ECS system) so no editor code is compiled
// into the shipped runtime: the HUD is driven from the editor's `DebugHook`
// tick, which runs only under `cn editor`.
//
// The HUD is plain `Sprite` + `TextLabel` components (injected by `inject.rs` at
// reserved ids). Each frame the hook re-anchors the controls flush to the
// window's top-right corner from the live viewport, shows/hides the Add or
// Templates dropdown (one shared row range, at most one open), reflects the
// capture-checkbox state, and hit-tests clicks. Running in the tick (before the
// world step) means the layout applies the same frame GraphicsSystem draws it.
// The whole HUD toggles with F1 (see `hook.rs`).

use crate::assets::{FrameInput, Sprite, TextAlign, TextLabel};
use crate::ecs::World;
use crate::ecs::asset_id::AssetId;

// The asset types the Add dropdown offers. All are External (user-declarable),
// standalone (no required cross-references), and naturally multi-instance, so
// each recompiles cleanly when added with default args to any rendering world.
pub(crate) const ADD_TYPES: &[&str] = &[
    "PointLight",
    "DirectionalLight",
    "ParticleEmitter",
    "Decal",
    "ReflectionProbe",
];

// The two dropdown menus the HUD can open.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum Menu {
    // Add a single asset (rows are `ADD_TYPES`).
    Add,
    // Apply an engine-owned content template (rows are the template titles).
    Templates,
}

// How many rows the given menu shows.
pub(crate) fn menu_len(menu: Menu) -> usize {
    match menu {
        Menu::Add => ADD_TYPES.len(),
        Menu::Templates => concinnity_templates::TEMPLATES.len(),
    }
}

// The label of row `i` of the given menu.
fn menu_item(menu: Menu, i: usize) -> &'static str {
    match menu {
        Menu::Add => ADD_TYPES[i],
        Menu::Templates => concinnity_templates::TEMPLATES[i].title,
    }
}

// The number of dropdown rows to inject: enough for whichever menu is longest.
pub(crate) fn max_rows() -> usize {
    ADD_TYPES.len().max(concinnity_templates::TEMPLATES.len())
}

// Reserved asset-id range for the runtime-injected editor HUD. Interned ids are
// dense from 0 and a real world never approaches this range, so a fixed high
// base is collision-free without scanning the world. These ids are never
// interned and never serialized to a blob.
const ID_BASE: u32 = 0x3000_0000;
pub(crate) const SAVE_BUTTON: AssetId = AssetId(ID_BASE);
pub(crate) const SAVE_LABEL: AssetId = AssetId(ID_BASE + 1);
pub(crate) const ADD_BUTTON: AssetId = AssetId(ID_BASE + 2);
pub(crate) const ADD_LABEL: AssetId = AssetId(ID_BASE + 3);
pub(crate) const TPL_BUTTON: AssetId = AssetId(ID_BASE + 4);
pub(crate) const TPL_LABEL: AssetId = AssetId(ID_BASE + 5);
// The capture checkbox: a row background, the box indicator, and its label.
pub(crate) const CHECK_BG: AssetId = AssetId(ID_BASE + 6);
pub(crate) const CHECK_BOX: AssetId = AssetId(ID_BASE + 7);
pub(crate) const CHECK_LABEL: AssetId = AssetId(ID_BASE + 8);

// Dropdown row elements (shared by both menus) in two contiguous sub-ranges.
pub(crate) fn dropdown_bg(i: usize) -> AssetId {
    AssetId(ID_BASE + 16 + i as u32)
}
pub(crate) fn dropdown_label(i: usize) -> AssetId {
    AssetId(ID_BASE + 32 + i as u32)
}

// Button geometry, in window pixels. SAVE is a flush-cornered square; Add and
// Templates sit to its left. Zero margin keeps SAVE hard against the window's
// top-right corner. Below the buttons: the capture checkbox, then (when open)
// the dropdown, all right-aligned.
pub(crate) const BTN_H: f32 = 88.0;
const SAVE_W: f32 = 88.0;
const ADD_W: f32 = 132.0;
const TPL_W: f32 = 132.0;
const GAP: f32 = 8.0;
const DROP_W: f32 = 200.0;
const ROW_H: f32 = 40.0;
const CHECK_H: f32 = 34.0;
const BOX_SIZE: f32 = 20.0;

// Vertical offset of a button's label from the button top, chosen to sit the
// ~20px HUD font roughly on the button's vertical center without measuring the
// glyphs here (the font metrics live on GraphicsSystem).
pub(crate) const LABEL_TOP: f32 = BTN_H * 0.5 - 10.0;
const ROW_LABEL_TOP: f32 = ROW_H * 0.5 - 10.0;

const SAVE_TINT_ACTIVE: [f32; 4] = [0.82, 0.14, 0.16, 1.0];
const SAVE_TINT_INERT: [f32; 4] = [0.26, 0.26, 0.30, 1.0];
const ADD_TINT: [f32; 4] = [0.20, 0.34, 0.52, 1.0];
const TPL_TINT: [f32; 4] = [0.28, 0.24, 0.44, 1.0];
const ROW_TINT: [f32; 4] = [0.14, 0.14, 0.17, 0.96];
const ROW_TINT_HOVER: [f32; 4] = [0.24, 0.26, 0.34, 0.98];
const CHECK_BG_TINT: [f32; 4] = [0.12, 0.12, 0.15, 0.92];
const BOX_TINT_ON: [f32; 4] = [0.30, 0.66, 0.34, 1.0];
const BOX_TINT_OFF: [f32; 4] = [0.30, 0.30, 0.34, 1.0];
const LABEL_ACTIVE: [f32; 3] = [1.0, 1.0, 1.0];
const LABEL_INERT: [f32; 3] = [0.55, 0.55, 0.58];
const ROW_LABEL: [f32; 3] = [0.90, 0.90, 0.92];

// Per-frame HUD state the hook hands to `apply_layout`.
#[derive(Debug, Clone, Copy)]
pub(crate) struct HudState {
    // Are there unsaved edits (SAVE active)?
    pub dirty: bool,
    // Which dropdown is open, if any.
    pub menu: Option<Menu>,
    // Does the world currently hold the cursor (checkbox ticked / play mode)?
    pub world_capture: bool,
    // Is the whole HUD shown (F1 toggle)?
    pub visible: bool,
}

// A click the HUD resolved to one of its controls.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum HudAction {
    // The SAVE button, while there are edits to persist.
    Save,
    // The Add or Templates button, while closed: open that dropdown.
    OpenMenu(Menu),
    // A dropdown row (index into the open menu): act on it and close.
    Pick(usize),
    // The capture checkbox: hand the cursor to / take it from the world.
    ToggleCapture,
    // Any click while a dropdown is open that is not a row: close it.
    CloseMenu,
}

// The SAVE, Add, and Templates button rects (`[x, y, w, h]`, window pixels) for
// a `vw`-wide window. Pure: the layout pass and the hit test both derive from it.
pub(crate) fn layout(vw: f32) -> ([f32; 4], [f32; 4], [f32; 4]) {
    let save = [vw - SAVE_W, 0.0, SAVE_W, BTN_H];
    let add = [save[0] - GAP - ADD_W, 0.0, ADD_W, BTN_H];
    let tpl = [add[0] - GAP - TPL_W, 0.0, TPL_W, BTN_H];
    (save, add, tpl)
}

// The capture-checkbox row rect, directly below the buttons.
pub(crate) fn checkbox_rect(vw: f32) -> [f32; 4] {
    [vw - DROP_W, BTN_H, DROP_W, CHECK_H]
}

// The rect of dropdown row `i`, stacked below the checkbox, right-aligned.
pub(crate) fn dropdown_row_rect(vw: f32, i: usize) -> [f32; 4] {
    [
        vw - DROP_W,
        BTN_H + CHECK_H + i as f32 * ROW_H,
        DROP_W,
        ROW_H,
    ]
}

fn point_in(x: f32, y: f32, rect: [f32; 4]) -> bool {
    x >= rect[0] && x < rect[0] + rect[2] && y >= rect[1] && y < rect[1] + rect[3]
}

// The dropdown row index under `(mx, my)`, bounded by the open menu's `count`.
fn row_at(mx: f32, my: f32, vw: f32, count: usize) -> Option<usize> {
    (0..count).find(|&i| point_in(mx, my, dropdown_row_rect(vw, i)))
}

// Resolve a click at `(mx, my)` against the HUD for a `vw`-wide window, given the
// dirty state and which menu (if any) is open. Pure -- the hook maps the action
// to a method and updates its own flags. Returns `None` for a no-op click.
//
// While a menu is open, a row click picks it and any other click dismisses it.
// While closed, SAVE fires only when dirty, the Add / Templates buttons open
// their menus, and the checkbox toggles cursor ownership.
pub(crate) fn hit_test(
    mx: f32,
    my: f32,
    clicked: bool,
    dirty: bool,
    open: Option<Menu>,
    vw: f32,
) -> Option<HudAction> {
    if !clicked || vw <= 0.0 {
        return None;
    }
    if let Some(menu) = open {
        return match row_at(mx, my, vw, menu_len(menu)) {
            Some(i) => Some(HudAction::Pick(i)),
            None => Some(HudAction::CloseMenu),
        };
    }
    let (save, add, tpl) = layout(vw);
    if dirty && point_in(mx, my, save) {
        Some(HudAction::Save)
    } else if point_in(mx, my, add) {
        Some(HudAction::OpenMenu(Menu::Add))
    } else if point_in(mx, my, tpl) {
        Some(HudAction::OpenMenu(Menu::Templates))
    } else if point_in(mx, my, checkbox_rect(vw)) {
        Some(HudAction::ToggleCapture)
    } else {
        None
    }
}

// Re-anchor the injected HUD to the top-right corner from the live viewport,
// colour the SAVE button + capture box by state, and show the open dropdown's
// rows (with their menu's labels). Hides the entire HUD when `state.visible` is
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
    let (save, add, tpl) = layout(vw);
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

    place_sprite(world, SAVE_BUTTON, save, save_tint, true);
    place_sprite(world, ADD_BUTTON, add, ADD_TINT, true);
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
        ADD_LABEL,
        centered(add),
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

    // Capture checkbox: row background, box indicator (green when the world holds
    // the cursor), and label.
    let check = checkbox_rect(vw);
    let box_tint = if state.world_capture {
        BOX_TINT_ON
    } else {
        BOX_TINT_OFF
    };
    place_sprite(world, CHECK_BG, check, CHECK_BG_TINT, true);
    place_sprite(
        world,
        CHECK_BOX,
        [
            check[0] + 8.0,
            check[1] + (CHECK_H - BOX_SIZE) * 0.5,
            BOX_SIZE,
            BOX_SIZE,
        ],
        box_tint,
        true,
    );
    place_label(
        world,
        CHECK_LABEL,
        [check[0] + 36.0, check[1] + CHECK_H * 0.5 - 10.0],
        ROW_LABEL,
        TextAlign::Left,
        true,
    );

    // Dropdown rows: hide them all, then show the open menu's, labelled from that
    // menu and with the hovered row highlighted.
    for i in 0..max_rows() {
        place_sprite(world, dropdown_bg(i), check, ROW_TINT, false);
        set_row_label(world, dropdown_label(i), [0.0, 0.0], "", false);
    }
    if let Some(menu) = state.menu {
        for i in 0..menu_len(menu) {
            let rect = dropdown_row_rect(vw, i);
            let hovered = point_in(input.mouse_x, input.mouse_y, rect);
            let tint = if hovered { ROW_TINT_HOVER } else { ROW_TINT };
            place_sprite(world, dropdown_bg(i), rect, tint, true);
            set_row_label(
                world,
                dropdown_label(i),
                [rect[0] + 12.0, rect[1] + ROW_LABEL_TOP],
                menu_item(menu, i),
                true,
            );
        }
    }
}

// Every injected sprite / label id, so the F1-hidden pass can blank the HUD.
fn all_sprite_ids() -> Vec<AssetId> {
    let mut ids = vec![SAVE_BUTTON, ADD_BUTTON, TPL_BUTTON, CHECK_BG, CHECK_BOX];
    ids.extend((0..max_rows()).map(dropdown_bg));
    ids
}
fn all_label_ids() -> Vec<AssetId> {
    let mut ids = vec![SAVE_LABEL, ADD_LABEL, TPL_LABEL, CHECK_LABEL];
    ids.extend((0..max_rows()).map(dropdown_label));
    ids
}

fn hide_all(world: &mut World) {
    for id in all_sprite_ids() {
        for s in world.query_mut::<Sprite>() {
            if s.asset_id == id {
                s.visible = false;
                break;
            }
        }
    }
    for id in all_label_ids() {
        for l in world.query_mut::<TextLabel>() {
            if l.asset_id == id {
                l.visible = false;
                break;
            }
        }
    }
}

fn centered(rect: [f32; 4]) -> [f32; 2] {
    [rect[0] + rect[2] * 0.5, rect[1] + LABEL_TOP]
}

// Move + resize the Sprite with `id` to `rect`, set its tint + visibility.
fn place_sprite(world: &mut World, id: AssetId, rect: [f32; 4], tint: [f32; 4], visible: bool) {
    for s in world.query_mut::<Sprite>() {
        if s.asset_id == id {
            s.x = rect[0];
            s.y = rect[1];
            s.width = rect[2];
            s.height = rect[3];
            s.tint = tint;
            s.visible = visible;
            break;
        }
    }
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
    for l in world.query_mut::<TextLabel>() {
        if l.asset_id == id {
            l.x = pos[0];
            l.y = pos[1];
            l.align = align;
            l.color = color;
            l.visible = visible;
            break;
        }
    }
}

// Position a dropdown row's label and set its content (the rows are shared
// between menus, so the content is written per frame from the open menu).
fn set_row_label(world: &mut World, id: AssetId, pos: [f32; 2], content: &str, visible: bool) {
    for l in world.query_mut::<TextLabel>() {
        if l.asset_id == id {
            l.x = pos[0];
            l.y = pos[1];
            l.align = TextAlign::Left;
            l.color = ROW_LABEL;
            l.visible = visible;
            if visible {
                l.content = content.to_string();
            }
            break;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn state(dirty: bool, menu: Option<Menu>, capture: bool, visible: bool) -> HudState {
        HudState {
            dirty,
            menu,
            world_capture: capture,
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
        let (save, add, tpl) = layout(1280.0);
        assert_eq!(save[0] + save[2], 1280.0, "SAVE flush to the window right");
        assert_eq!(add[0] + add[2], save[0] - GAP, "Add left of SAVE");
        assert_eq!(tpl[0] + tpl[2], add[0] - GAP, "Templates left of Add");
    }

    #[test]
    fn checkbox_and_dropdown_stack_below_buttons() {
        assert_eq!(checkbox_rect(1280.0)[1], BTN_H);
        assert_eq!(dropdown_row_rect(1280.0, 0)[1], BTN_H + CHECK_H);
    }

    #[test]
    fn hit_test_closed_resolves_each_control() {
        let (save, add, tpl) = layout(1280.0);
        let mid = |r: [f32; 4]| (r[0] + r[2] * 0.5, r[1] + r[3] * 0.5);
        let (sx, sy) = mid(save);
        assert_eq!(
            hit_test(sx, sy, true, true, None, 1280.0),
            Some(HudAction::Save)
        );
        assert_eq!(
            hit_test(sx, sy, true, false, None, 1280.0),
            None,
            "clean SAVE inert"
        );
        let (ax, ay) = mid(add);
        assert_eq!(
            hit_test(ax, ay, true, false, None, 1280.0),
            Some(HudAction::OpenMenu(Menu::Add))
        );
        let (tx, ty) = mid(tpl);
        assert_eq!(
            hit_test(tx, ty, true, false, None, 1280.0),
            Some(HudAction::OpenMenu(Menu::Templates))
        );
        let chk = checkbox_rect(1280.0);
        assert_eq!(
            hit_test(chk[0] + 10.0, chk[1] + 10.0, true, false, None, 1280.0),
            Some(HudAction::ToggleCapture)
        );
    }

    #[test]
    fn hit_test_open_picks_row_or_closes() {
        let r2 = dropdown_row_rect(1280.0, 2);
        assert_eq!(
            hit_test(
                r2[0] + 10.0,
                r2[1] + 10.0,
                true,
                false,
                Some(Menu::Add),
                1280.0
            ),
            Some(HudAction::Pick(2))
        );
        // A row index past the Templates menu length is not a row -> dismiss.
        let past = dropdown_row_rect(1280.0, menu_len(Menu::Templates));
        assert_eq!(
            hit_test(
                past[0] + 10.0,
                past[1] + 10.0,
                true,
                false,
                Some(Menu::Templates),
                1280.0
            ),
            Some(HudAction::CloseMenu)
        );
        assert_eq!(
            hit_test(640.0, 360.0, true, false, Some(Menu::Add), 1280.0),
            Some(HudAction::CloseMenu)
        );
    }

    #[test]
    fn hit_test_ignores_non_clicks_and_zero_width() {
        assert_eq!(hit_test(1240.0, 40.0, false, true, None, 1280.0), None);
        assert_eq!(hit_test(0.0, 0.0, true, true, Some(Menu::Add), 0.0), None);
    }

    // Closed: buttons + checkbox shown, SAVE coloured by dirty, all rows hidden.
    #[test]
    fn apply_layout_closed_shows_buttons_hides_rows() {
        let mut world = hud_world(1024.0, (0.0, 0.0));
        apply_layout(&mut world, state(true, None, false, true));
        assert_eq!(sprite(&world, SAVE_BUTTON).tint, SAVE_TINT_ACTIVE);
        assert_eq!(sprite(&world, CHECK_BOX).tint, BOX_TINT_OFF);
        for id in [SAVE_BUTTON, ADD_BUTTON, TPL_BUTTON, CHECK_BG] {
            assert!(sprite(&world, id).visible, "{id:?} shown");
        }
        for i in 0..max_rows() {
            assert!(!sprite(&world, dropdown_bg(i)).visible, "row {i} hidden");
        }
    }

    // Open Add: rows show the type names; hovered row highlighted.
    #[test]
    fn apply_layout_add_menu_shows_type_rows() {
        let r0 = dropdown_row_rect(1280.0, 0);
        let mut world = hud_world(1280.0, (r0[0] + 10.0, r0[1] + 10.0));
        apply_layout(&mut world, state(false, Some(Menu::Add), false, true));
        for (i, ty) in ADD_TYPES.iter().enumerate() {
            assert!(sprite(&world, dropdown_bg(i)).visible, "add row {i} shown");
            assert_eq!(&label(&world, dropdown_label(i)).content, ty);
        }
        assert_eq!(
            sprite(&world, dropdown_bg(0)).tint,
            ROW_TINT_HOVER,
            "row 0 hovered"
        );
    }

    // Open Templates: rows show the template titles; extra shared rows stay
    // hidden when the templates menu is shorter than ADD_TYPES.
    #[test]
    fn apply_layout_templates_menu_shows_title_rows() {
        let mut world = hud_world(1280.0, (0.0, 0.0));
        apply_layout(&mut world, state(false, Some(Menu::Templates), false, true));
        let n = menu_len(Menu::Templates);
        for i in 0..n {
            assert!(sprite(&world, dropdown_bg(i)).visible, "tpl row {i} shown");
            assert_eq!(
                label(&world, dropdown_label(i)).content,
                concinnity_templates::TEMPLATES[i].title
            );
        }
        for i in n..max_rows() {
            assert!(
                !sprite(&world, dropdown_bg(i)).visible,
                "extra row {i} hidden"
            );
        }
    }

    // F1 hidden: every HUD element is blanked.
    #[test]
    fn apply_layout_hidden_blanks_everything() {
        let mut world = hud_world(1280.0, (0.0, 0.0));
        apply_layout(&mut world, state(true, Some(Menu::Add), true, true));
        apply_layout(&mut world, state(true, Some(Menu::Add), true, false));
        for id in all_sprite_ids() {
            assert!(!sprite(&world, id).visible, "sprite {id:?} hidden");
        }
    }
}
