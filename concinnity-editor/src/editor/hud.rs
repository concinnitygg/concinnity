// src/editor/hud.rs
//
// The editor HUD's per-frame geometry, hit-testing, and layout. This lives in
// the editor crate (not in a client ECS system) so no editor code is compiled
// into the shipped runtime: the HUD is driven from the editor's `DebugHook`
// tick, which runs only under `cn editor`.
//
// The HUD is plain `Sprite` + `TextLabel` components (injected by `inject.rs` at
// reserved ids). Each frame the hook re-anchors the SAVE + Add buttons flush to
// the window's top-right corner from the live viewport, shows/hides the
// add-asset dropdown, and hit-tests clicks against their window-space rects.
// Running in the tick (before the world step) means the layout applies the same
// frame GraphicsSystem draws it.

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

// Reserved asset-id range for the runtime-injected editor HUD. Interned ids are
// dense from 0 and a real world never approaches this range, so a fixed high
// base is collision-free without scanning the world. These ids are never
// interned and never serialized to a blob.
const ID_BASE: u32 = 0x3000_0000;
pub(crate) const SAVE_BUTTON: AssetId = AssetId(ID_BASE);
pub(crate) const SAVE_LABEL: AssetId = AssetId(ID_BASE + 1);
pub(crate) const ADD_BUTTON: AssetId = AssetId(ID_BASE + 2);
pub(crate) const ADD_LABEL: AssetId = AssetId(ID_BASE + 3);

// Dropdown row elements occupy two contiguous sub-ranges past the buttons, one
// per offered type. `ADD_TYPES` is far shorter than the 16-id gap between them.
pub(crate) fn dropdown_bg(i: usize) -> AssetId {
    AssetId(ID_BASE + 16 + i as u32)
}
pub(crate) fn dropdown_label(i: usize) -> AssetId {
    AssetId(ID_BASE + 32 + i as u32)
}

// Button geometry, in window pixels. The SAVE button is a flush-cornered square;
// the Add button sits to its left with a small gap. Zero margin keeps SAVE hard
// against the window's top-right corner. The dropdown drops down under the
// buttons, right-aligned to the window edge.
pub(crate) const BTN_H: f32 = 88.0;
const SAVE_W: f32 = 88.0;
const ADD_W: f32 = 132.0;
const GAP: f32 = 8.0;
const DROP_W: f32 = 200.0;
const ROW_H: f32 = 40.0;

// Vertical offset of a button's label from the button top, chosen to sit the
// ~20px HUD font roughly on the button's vertical center without measuring the
// glyphs here (the font metrics live on GraphicsSystem).
pub(crate) const LABEL_TOP: f32 = BTN_H * 0.5 - 10.0;
const ROW_LABEL_TOP: f32 = ROW_H * 0.5 - 10.0;
const ROW_LABEL_PAD: f32 = 12.0;

// SAVE button fill with unsaved edits (active) vs none (inert); the white label
// text dims in the inert state. The Add button keeps a fixed fill.
const SAVE_TINT_ACTIVE: [f32; 4] = [0.82, 0.14, 0.16, 1.0];
const SAVE_TINT_INERT: [f32; 4] = [0.26, 0.26, 0.30, 1.0];
const ADD_TINT: [f32; 4] = [0.20, 0.34, 0.52, 1.0];
const ROW_TINT: [f32; 4] = [0.14, 0.14, 0.17, 0.96];
const ROW_TINT_HOVER: [f32; 4] = [0.24, 0.26, 0.34, 0.98];
const LABEL_ACTIVE: [f32; 3] = [1.0, 1.0, 1.0];
const LABEL_INERT: [f32; 3] = [0.55, 0.55, 0.58];
const ROW_LABEL: [f32; 3] = [0.90, 0.90, 0.92];

// A click the HUD resolved to one of its controls.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum HudAction {
    // The SAVE button, while there are edits to persist.
    Save,
    // The Add button, while the dropdown is closed: open it.
    OpenMenu,
    // A dropdown row (index into `ADD_TYPES`): add that type and close.
    PickType(usize),
    // Any click while the dropdown is open that is not a row: close it.
    CloseMenu,
}

// The SAVE and Add button rects (`[x, y, w, h]`, window pixels) for a window
// `vw` pixels wide. Pure: the single source of truth the layout pass and the hit
// test both derive from.
pub(crate) fn layout(vw: f32) -> ([f32; 4], [f32; 4]) {
    let save = [vw - SAVE_W, 0.0, SAVE_W, BTN_H];
    let add = [vw - SAVE_W - GAP - ADD_W, 0.0, ADD_W, BTN_H];
    (save, add)
}

// The rect of dropdown row `i`, stacked downward under the buttons and
// right-aligned to the window edge.
pub(crate) fn dropdown_row_rect(vw: f32, i: usize) -> [f32; 4] {
    [vw - DROP_W, BTN_H + i as f32 * ROW_H, DROP_W, ROW_H]
}

fn point_in(x: f32, y: f32, rect: [f32; 4]) -> bool {
    x >= rect[0] && x < rect[0] + rect[2] && y >= rect[1] && y < rect[1] + rect[3]
}

// The dropdown row index under `(mx, my)`, or `None`.
fn row_at(mx: f32, my: f32, vw: f32) -> Option<usize> {
    (0..ADD_TYPES.len()).find(|&i| point_in(mx, my, dropdown_row_rect(vw, i)))
}

// Resolve a click at `(mx, my)` against the HUD for a `vw`-wide window, given the
// current dirty + open state. Pure -- the hook maps the action to a method and
// updates its own open flag. Returns `None` for a no-op click.
//
// While open, a row click picks that type and any other click closes the menu
// (standard click-outside dismissal); while closed, SAVE fires only when dirty
// and the Add button opens the menu.
pub(crate) fn hit_test(
    mx: f32,
    my: f32,
    clicked: bool,
    dirty: bool,
    open: bool,
    vw: f32,
) -> Option<HudAction> {
    if !clicked || vw <= 0.0 {
        return None;
    }
    if open {
        return match row_at(mx, my, vw) {
            Some(i) => Some(HudAction::PickType(i)),
            None => Some(HudAction::CloseMenu),
        };
    }
    let (save, add) = layout(vw);
    if dirty && point_in(mx, my, save) {
        Some(HudAction::Save)
    } else if point_in(mx, my, add) {
        Some(HudAction::OpenMenu)
    } else {
        None
    }
}

// Re-anchor the injected HUD to the top-right corner from the live viewport,
// colour the SAVE button by `dirty`, and show/hide the dropdown by `open`
// (highlighting the row under the cursor). A no-op until a `FrameInput` exists
// (frame 0) or on a zero-width window.
pub(crate) fn apply_layout(world: &mut World, dirty: bool, open: bool) {
    let Some(input) = world.query::<FrameInput>().last().cloned() else {
        return;
    };
    let vw = input.viewport[0];
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

    place_sprite(world, SAVE_BUTTON, save, save_tint, true);
    place_sprite(world, ADD_BUTTON, add, ADD_TINT, true);
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

    // Dropdown rows: shown + positioned while open (hovered row highlighted),
    // hidden otherwise.
    let hovered = open
        .then(|| row_at(input.mouse_x, input.mouse_y, vw))
        .flatten();
    for i in 0..ADD_TYPES.len() {
        let rect = dropdown_row_rect(vw, i);
        let tint = if hovered == Some(i) {
            ROW_TINT_HOVER
        } else {
            ROW_TINT
        };
        place_sprite(world, dropdown_bg(i), rect, tint, open);
        let label_pos = [rect[0] + ROW_LABEL_PAD, rect[1] + ROW_LABEL_TOP];
        place_label_at(
            world,
            dropdown_label(i),
            label_pos,
            ROW_LABEL,
            TextAlign::Left,
            open,
        );
    }
}

// The label origin that centers text over `rect` at the button label height.
fn centered(rect: [f32; 4]) -> [f32; 2] {
    [rect[0] + rect[2] * 0.5, rect[1] + LABEL_TOP]
}

// Move + resize the Sprite with `id` to `rect`, set its tint, and set its
// visibility, if present.
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

fn place_label(
    world: &mut World,
    id: AssetId,
    pos: [f32; 2],
    color: [f32; 3],
    align: TextAlign,
    visible: bool,
) {
    place_label_at(world, id, pos, color, align, visible);
}

// Position the TextLabel with `id` at `pos`, set its colour/alignment, and set
// its visibility, if present.
fn place_label_at(
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

#[cfg(test)]
mod tests {
    use super::*;

    // Build a world holding the buttons, dropdown rows, their labels, and a
    // FrameInput of the given window width + cursor position.
    fn hud_world(vw: f32, mouse: (f32, f32)) -> World {
        let mut world = World::new_empty();
        let mut sprite_ids = vec![SAVE_BUTTON, ADD_BUTTON];
        let mut label_ids = vec![SAVE_LABEL, ADD_LABEL];
        for i in 0..ADD_TYPES.len() {
            sprite_ids.push(dropdown_bg(i));
            label_ids.push(dropdown_label(i));
        }
        for id in sprite_ids {
            world.add_component(Sprite {
                asset_id: id,
                ..Default::default()
            });
        }
        for id in label_ids {
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

    #[test]
    fn layout_anchors_save_flush_top_right() {
        let (save, add) = layout(1280.0);
        assert_eq!(save[0] + save[2], 1280.0, "SAVE right edge at window width");
        assert_eq!(save[1], 0.0, "SAVE top edge at window top");
        assert_eq!(save, [1192.0, 0.0, 88.0, 88.0]);
        assert!(add[0] + add[2] <= save[0], "Add button left of SAVE");
        assert_eq!(add[0] + add[2], save[0] - GAP, "gap between Add and SAVE");
    }

    #[test]
    fn layout_tracks_window_width() {
        let (save, _) = layout(1024.0);
        assert_eq!(save[0] + save[2], 1024.0);
        assert_eq!(save[0], 936.0);
    }

    // Dropdown rows stack below the buttons, right-aligned, non-overlapping.
    #[test]
    fn dropdown_rows_stack_below_buttons() {
        let r0 = dropdown_row_rect(1280.0, 0);
        let r1 = dropdown_row_rect(1280.0, 1);
        assert_eq!(r0[1], BTN_H, "first row starts below the buttons");
        assert_eq!(r1[1], BTN_H + ROW_H, "rows stack by row height");
        assert_eq!(r0[0] + r0[2], 1280.0, "rows right-aligned to window");
    }

    #[test]
    fn hit_test_closed_save_fires_only_when_dirty() {
        let (sx, sy) = (1240.0, 40.0); // inside SAVE at 1280 wide
        assert_eq!(
            hit_test(sx, sy, true, true, false, 1280.0),
            Some(HudAction::Save)
        );
        assert_eq!(hit_test(sx, sy, true, false, false, 1280.0), None);
    }

    #[test]
    fn hit_test_closed_add_opens_menu() {
        let (ax, ay) = (1100.0, 40.0); // inside Add at 1280 wide
        assert_eq!(
            hit_test(ax, ay, true, false, false, 1280.0),
            Some(HudAction::OpenMenu)
        );
    }

    // While open, a click on row i picks that type; a click off the rows closes.
    #[test]
    fn hit_test_open_picks_row_or_closes() {
        let r2 = dropdown_row_rect(1280.0, 2);
        let (mx, my) = (r2[0] + 10.0, r2[1] + 10.0);
        assert_eq!(
            hit_test(mx, my, true, false, true, 1280.0),
            Some(HudAction::PickType(2))
        );
        // A click in the empty middle of the screen dismisses the menu.
        assert_eq!(
            hit_test(640.0, 360.0, true, false, true, 1280.0),
            Some(HudAction::CloseMenu)
        );
        // Even a click on the SAVE button just closes while open (no save).
        assert_eq!(
            hit_test(1240.0, 40.0, true, true, true, 1280.0),
            Some(HudAction::CloseMenu)
        );
    }

    #[test]
    fn hit_test_ignores_non_clicks_and_zero_width() {
        assert_eq!(hit_test(1240.0, 40.0, false, true, false, 1280.0), None);
        assert_eq!(hit_test(0.0, 0.0, true, true, true, 0.0), None);
    }

    // Closed: buttons anchor to the corner, SAVE reds when dirty, and every
    // dropdown row is hidden.
    #[test]
    fn apply_layout_closed_hides_rows_and_colours_save() {
        let mut world = hud_world(1024.0, (0.0, 0.0));
        apply_layout(&mut world, true, false);

        let save = sprite(&world, SAVE_BUTTON);
        assert_eq!(save.x + save.width, 1024.0, "SAVE flush right");
        assert_eq!(save.tint, SAVE_TINT_ACTIVE, "SAVE red when dirty");
        assert_eq!(label(&world, SAVE_LABEL).color, LABEL_ACTIVE);
        for i in 0..ADD_TYPES.len() {
            assert!(!sprite(&world, dropdown_bg(i)).visible, "row {i} hidden");
            assert!(!label(&world, dropdown_label(i)).visible);
        }
    }

    #[test]
    fn apply_layout_clean_greys_save() {
        let mut world = hud_world(1280.0, (0.0, 0.0));
        apply_layout(&mut world, false, false);
        assert_eq!(sprite(&world, SAVE_BUTTON).tint, SAVE_TINT_INERT);
        assert_eq!(label(&world, SAVE_LABEL).color, LABEL_INERT);
    }

    // Open: rows are shown + stacked, and the row under the cursor is
    // highlighted while the others use the base tint.
    #[test]
    fn apply_layout_open_shows_rows_and_highlights_hovered() {
        let r1 = dropdown_row_rect(1280.0, 1);
        let mut world = hud_world(1280.0, (r1[0] + 20.0, r1[1] + 20.0));
        apply_layout(&mut world, false, true);

        for i in 0..ADD_TYPES.len() {
            let bg = sprite(&world, dropdown_bg(i));
            assert!(bg.visible, "row {i} shown while open");
            assert_eq!(bg.y, BTN_H + i as f32 * ROW_H, "row {i} stacked");
            let expect = if i == 1 { ROW_TINT_HOVER } else { ROW_TINT };
            assert_eq!(bg.tint, expect, "row {i} tint (hovered=1)");
            assert!(label(&world, dropdown_label(i)).visible);
        }
    }

    #[test]
    fn apply_layout_noops_without_frame_input() {
        let mut world = World::new_empty();
        world.add_component(Sprite {
            asset_id: SAVE_BUTTON,
            x: 7.0,
            ..Default::default()
        });
        apply_layout(&mut world, true, false);
        assert_eq!(
            sprite(&world, SAVE_BUTTON).x,
            7.0,
            "untouched without input"
        );
    }
}
