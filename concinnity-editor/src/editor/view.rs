// src/editor/view.rs
//
// The editor "View" panel: the hub that toggles the visibility of the other
// floating editor panels (Assets, Preview, Templates). Like the rest of the
// editor HUD it is plain `Sprite` / `TextLabel` components at reserved ids
// (injected by `inject.rs`), driven each frame by the editor hook -- nothing here
// reaches the shipped runtime. It shares the floating-panel chrome (`widget.rs`):
// a draggable title bar the hook moves, and the hook owns the position. Each row
// is a checkbox that reflects, and toggles, one panel's shown state; the top-bar
// "View" button opens / closes this panel.

use super::widget::{self, place_sprite, point_in};
use crate::assets::TextAlign;
use crate::ecs::World;
use crate::ecs::asset_id::AssetId;

// Reserved ids, past the Preview panel's family (`hud.rs` / `preview.rs`; the
// Preview panel lives at `ID_BASE + 0x400`). Keep the ranges disjoint.
const VIEW: u32 = 0x3000_0000 + 0x500;
pub(crate) const TITLE_BG: AssetId = AssetId(VIEW);
pub(crate) const TITLE_LABEL: AssetId = AssetId(VIEW + 1);

// The toggle rows, one per controllable panel: a background, a checkbox, and a
// label each (three contiguous sub-ranges).
pub(crate) fn row_bg(i: usize) -> AssetId {
    AssetId(VIEW + 0x10 + i as u32)
}
pub(crate) fn check_box(i: usize) -> AssetId {
    AssetId(VIEW + 0x20 + i as u32)
}
pub(crate) fn row_label(i: usize) -> AssetId {
    AssetId(VIEW + 0x30 + i as u32)
}

// The toggle rows, in order: their captions and which panel each controls (the
// hook maps `Toggle(i)` back through this order).
pub(crate) const ROWS: usize = 3;
const ROW_LABELS: [&str; ROWS] = ["Assets", "Preview", "Templates"];

// Geometry, in window pixels. Every rect derives from the panel origin `o` (the
// title bar's top-left), like the other floating panels.
const VIEW_W: f32 = 200.0;
const ROW_H: f32 = 34.0;
const BOX_SIZE: f32 = 20.0;

const ROW_TINT: [f32; 4] = [0.12, 0.12, 0.15, 0.92];
const ROW_TINT_HOVER: [f32; 4] = [0.20, 0.24, 0.34, 0.96];
const BOX_TINT_ON: [f32; 4] = [0.30, 0.66, 0.34, 1.0];
const BOX_TINT_OFF: [f32; 4] = [0.30, 0.30, 0.34, 1.0];
const LABEL: [f32; 3] = [0.90, 0.90, 0.92];

// Which panels are currently shown, so each row's checkbox reflects its state.
#[derive(Debug, Clone, Copy)]
pub(crate) struct ViewState {
    pub assets: bool,
    pub preview: bool,
    pub templates: bool,
}

impl ViewState {
    // Whether the panel controlled by row `i` is shown (matches `ROW_LABELS`).
    fn on(&self, i: usize) -> bool {
        match i {
            0 => self.assets,
            1 => self.preview,
            2 => self.templates,
            _ => false,
        }
    }
}

// A resolved View-panel click.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum ViewAction {
    // Toggle the panel controlled by row `i` (0 Assets, 1 Preview, 2 Templates).
    Toggle(usize),
    // A click elsewhere on the panel: swallowed so it cannot reach the world.
    Consume,
}

// Where the panel sits until the user drags it: the window's top-left, below the
// Preview panel's default anchor so the two do not overlap at launch.
pub(crate) fn default_origin() -> [f32; 2] {
    [8.0, 8.0 + widget::TITLE_H + ROW_H + 8.0]
}

// The panel's fixed footprint, for the hook's drag clamp.
pub(crate) fn size() -> [f32; 2] {
    [VIEW_W, widget::TITLE_H + ROWS as f32 * ROW_H]
}

// The panel outer rect (title bar + the toggle rows).
pub(crate) fn panel_rect(o: [f32; 2]) -> [f32; 4] {
    let s = size();
    [o[0], o[1], s[0], s[1]]
}

// The draggable title bar across the panel top.
pub(crate) fn title_rect(o: [f32; 2]) -> [f32; 4] {
    [o[0], o[1], VIEW_W, widget::TITLE_H]
}

// Toggle row `i`, stacked below the title bar.
fn row_rect(o: [f32; 2], i: usize) -> [f32; 4] {
    [
        o[0],
        o[1] + widget::TITLE_H + i as f32 * ROW_H,
        VIEW_W,
        ROW_H,
    ]
}

// Resolve a click at `(mx, my)` against the panel at origin `o`. `None` means the
// click missed the panel. Title-bar presses never reach this: the hook intercepts
// them first to start a drag.
pub(crate) fn hit_test(mx: f32, my: f32, o: [f32; 2]) -> Option<ViewAction> {
    for i in 0..ROWS {
        if point_in(mx, my, row_rect(o, i)) {
            return Some(ViewAction::Toggle(i));
        }
    }
    point_in(mx, my, panel_rect(o)).then_some(ViewAction::Consume)
}

// Position + show the panel at origin `o`, colouring each row's checkbox by its
// panel's shown state (green when shown) and highlighting the hovered row.
pub(crate) fn apply(world: &mut World, o: [f32; 2], state: ViewState, mouse: [f32; 2]) {
    widget::place_title(world, TITLE_BG, TITLE_LABEL, title_rect(o), "View");
    for (i, caption) in ROW_LABELS.iter().enumerate() {
        let row = row_rect(o, i);
        let hovered = point_in(mouse[0], mouse[1], row);
        let tint = if hovered { ROW_TINT_HOVER } else { ROW_TINT };
        place_sprite(world, row_bg(i), row, tint, true);
        let box_tint = if state.on(i) {
            BOX_TINT_ON
        } else {
            BOX_TINT_OFF
        };
        place_sprite(
            world,
            check_box(i),
            [
                row[0] + 8.0,
                row[1] + (ROW_H - BOX_SIZE) * 0.5,
                BOX_SIZE,
                BOX_SIZE,
            ],
            box_tint,
            true,
        );
        if let Some(l) = widget::label_mut(world, row_label(i)) {
            l.x = row[0] + 36.0;
            l.y = row[1] + ROW_H * 0.5 - 10.0;
            l.align = TextAlign::Left;
            l.color = LABEL;
            l.visible = true;
            l.content = caption.to_string();
        }
    }
}

// Hide every panel element (the F1-hidden pass, or when the panel is toggled off).
pub(crate) fn hide_all(world: &mut World) {
    for id in all_sprite_ids() {
        widget::set_sprite_visible(world, id, false);
    }
    for id in all_label_ids() {
        widget::set_label_visible(world, id, false);
    }
}

// Every panel sprite / label id, for injection and the hidden pass.
pub(crate) fn all_sprite_ids() -> Vec<AssetId> {
    let mut ids = vec![TITLE_BG];
    ids.extend((0..ROWS).map(row_bg));
    ids.extend((0..ROWS).map(check_box));
    ids
}
pub(crate) fn all_label_ids() -> Vec<AssetId> {
    let mut ids = vec![TITLE_LABEL];
    ids.extend((0..ROWS).map(row_label));
    ids
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::assets::{Sprite, TextLabel};

    fn injected_world() -> World {
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
        world
    }

    fn sprite(world: &World, id: AssetId) -> Sprite {
        world
            .query::<Sprite>()
            .find(|s| s.asset_id == id)
            .cloned()
            .unwrap()
    }

    fn state(assets: bool, preview: bool, templates: bool) -> ViewState {
        ViewState {
            assets,
            preview,
            templates,
        }
    }

    #[test]
    fn rows_stack_below_the_title_bar() {
        let o = [40.0, 60.0];
        assert_eq!(title_rect(o), [40.0, 60.0, VIEW_W, widget::TITLE_H]);
        assert_eq!(row_rect(o, 0)[1], 60.0 + widget::TITLE_H);
        assert_eq!(row_rect(o, 1)[1], 60.0 + widget::TITLE_H + ROW_H);
        let p = panel_rect(o);
        assert_eq!(
            p[3],
            widget::TITLE_H + ROWS as f32 * ROW_H,
            "panel is the title bar plus its rows"
        );
    }

    #[test]
    fn hit_test_resolves_each_row_and_swallows_the_rest() {
        let o = default_origin();
        for i in 0..ROWS {
            let r = row_rect(o, i);
            assert_eq!(
                hit_test(r[0] + 10.0, r[1] + 10.0, o),
                Some(ViewAction::Toggle(i))
            );
        }
        // A title-bar click (not a drag) is swallowed here.
        let t = title_rect(o);
        assert_eq!(
            hit_test(t[0] + 5.0, t[1] + 5.0, o),
            Some(ViewAction::Consume)
        );
        assert_eq!(hit_test(2000.0, 2000.0, o), None, "a miss falls through");
    }

    #[test]
    fn apply_shows_rows_and_reflects_state() {
        let mut world = injected_world();
        let o = default_origin();
        apply(&mut world, o, state(true, false, true), [0.0, 0.0]);
        let title = world
            .query::<TextLabel>()
            .find(|l| l.asset_id == TITLE_LABEL)
            .unwrap();
        assert!(title.visible && title.content == "View");
        // Row captions match the panel order.
        for (i, want) in ROW_LABELS.iter().enumerate() {
            let l = world
                .query::<TextLabel>()
                .find(|l| l.asset_id == row_label(i))
                .unwrap();
            assert_eq!(&l.content, want);
        }
        // Checkboxes reflect the state: Assets on, Preview off, Templates on.
        assert_eq!(sprite(&world, check_box(0)).tint, BOX_TINT_ON);
        assert_eq!(sprite(&world, check_box(1)).tint, BOX_TINT_OFF);
        assert_eq!(sprite(&world, check_box(2)).tint, BOX_TINT_ON);
    }

    #[test]
    fn hovered_row_is_highlighted() {
        let mut world = injected_world();
        let o = default_origin();
        let r1 = row_rect(o, 1);
        apply(
            &mut world,
            o,
            state(false, false, false),
            [r1[0] + 10.0, r1[1] + 10.0],
        );
        assert_eq!(sprite(&world, row_bg(1)).tint, ROW_TINT_HOVER);
        assert_eq!(sprite(&world, row_bg(0)).tint, ROW_TINT);
    }

    #[test]
    fn hide_all_blanks_every_element() {
        let mut world = injected_world();
        apply(
            &mut world,
            default_origin(),
            state(true, true, true),
            [0.0, 0.0],
        );
        hide_all(&mut world);
        assert!(world.query::<Sprite>().all(|s| !s.visible));
        assert!(world.query::<TextLabel>().all(|l| !l.visible));
    }
}
