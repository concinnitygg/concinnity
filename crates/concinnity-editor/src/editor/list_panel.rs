// src/editor/list_panel.rs
//
// Shared chrome for the editor's simple "row list" floating panels (Preview,
// View, Templates). Each is one rounded surface: a draggable title area with a
// close button over a vertical stack of fixed-height rows, and each row is a
// hover / selection highlight plus a label with an optional checkbox. The three
// panels differ only in their reserved-id base, their width, and how a row
// index maps to their own action; the row geometry, the id-family layout, the
// per-row draw, and the hit-test / hide bookkeeping are identical, so they live
// here once. (The Assets and Template detail panels use the richer grouped list
// in `asset_list.rs` instead.)

use super::theme;
use super::widget::{self, place_rounded, point_in};
use crate::assets::TextAlign;
use crate::ecs::World;
use crate::ecs::asset_id::AssetId;

// Row geometry, in window pixels.
pub(crate) const ROW_H: f32 = 28.0;
const BOX_SIZE: f32 = 16.0;
const PAD: f32 = 8.0;
// A checkbox row insets its label past the box.
const CHECK_LABEL_INSET: f32 = 32.0;
const LABEL_TOP: f32 = ROW_H * 0.5 - theme::TEXT_HALF;
// Breathing room between the last row and the panel's rounded bottom edge.
const BOTTOM_PAD: f32 = 6.0;

// Row highlight tints: nothing while idle (the panel surface shows through),
// the shared hover / selected highlights otherwise. A checkbox reflects its own
// on / off state separately.
const ROW_TINT: [f32; 4] = [0.0, 0.0, 0.0, 0.0];
const BOX_TINT_ON: [f32; 4] = [0.30, 0.66, 0.34, 1.0];
const BOX_TINT_OFF: [f32; 4] = [0.30, 0.30, 0.34, 1.0];

// The reserved-id layout every row-list panel follows, offset from its `base`:
// the panel surface + title label + close button, then three contiguous per-row
// sub-ranges (row highlight, checkbox, row label). `const fn` so a panel can
// name a fixed id (`const PANEL_BG = list_panel::panel_bg(BASE)`).
pub(crate) const fn panel_bg(base: u32) -> AssetId {
    AssetId(base)
}
pub(crate) const fn title_label(base: u32) -> AssetId {
    AssetId(base + 1)
}
pub(crate) const fn close_bg(base: u32) -> AssetId {
    AssetId(base + 2)
}
pub(crate) const fn close_label(base: u32) -> AssetId {
    AssetId(base + 3)
}
pub(crate) const fn row_bg(base: u32, i: usize) -> AssetId {
    AssetId(base + 0x10 + i as u32)
}
pub(crate) const fn check_box(base: u32, i: usize) -> AssetId {
    AssetId(base + 0x20 + i as u32)
}
pub(crate) const fn row_label(base: u32, i: usize) -> AssetId {
    AssetId(base + 0x30 + i as u32)
}

// The draggable title bar across the panel top.
pub(crate) fn title_rect(o: [f32; 2], w: f32) -> [f32; 4] {
    [o[0], o[1], w, widget::TITLE_H]
}

// The "X" close button at the title bar's right end.
pub(crate) fn close_rect(o: [f32; 2], w: f32) -> [f32; 4] {
    widget::close_rect(title_rect(o, w))
}

// Row `i`, stacked below the title bar.
pub(crate) fn row_rect(o: [f32; 2], w: f32, i: usize) -> [f32; 4] {
    [o[0], o[1] + widget::TITLE_H + i as f32 * ROW_H, w, ROW_H]
}

// The panel's footprint (title bar plus `rows` rows plus the bottom pad).
pub(crate) fn size(w: f32, rows: usize) -> [f32; 2] {
    [w, widget::TITLE_H + rows as f32 * ROW_H + BOTTOM_PAD]
}

// The panel outer rect at origin `o`.
pub(crate) fn panel_rect(o: [f32; 2], w: f32, rows: usize) -> [f32; 4] {
    let s = size(w, rows);
    [o[0], o[1], s[0], s[1]]
}

// One row to draw: its caption, an optional checkbox (`Some(on)` draws a box
// tinted by its state; `None` is a label-only row), and whether it is the
// currently selected row (highlighted even without a hover).
pub(crate) struct Row {
    pub caption: String,
    pub check: Option<bool>,
    pub selected: bool,
}

impl Row {
    // A label-only row (Templates).
    pub(crate) fn label(caption: impl Into<String>) -> Self {
        Self {
            caption: caption.into(),
            check: None,
            selected: false,
        }
    }

    // A checkbox row reflecting `on` (Preview / View).
    pub(crate) fn checkbox(caption: impl Into<String>, on: bool) -> Self {
        Self {
            caption: caption.into(),
            check: Some(on),
            selected: false,
        }
    }

    // Mark this row selected (drawn highlighted while idle).
    pub(crate) fn select(mut self, selected: bool) -> Self {
        self.selected = selected;
        self
    }
}

// Position + show the whole panel at origin `o`: the rounded panel surface, the
// heading, the hover-tinted close button, and each row's highlight, optional
// checkbox, and label. `mouse` drives the hover highlight and the close-button
// tint.
pub(crate) fn apply(
    world: &mut World,
    base: u32,
    o: [f32; 2],
    w: f32,
    heading: &str,
    rows: &[Row],
    mouse: [f32; 2],
) {
    widget::place_panel(world, panel_bg(base), panel_rect(o, w, rows.len()));
    let title = title_rect(o, w);
    widget::place_heading(world, title_label(base), title, heading);
    let close_hover = point_in(mouse[0], mouse[1], close_rect(o, w));
    widget::place_close(world, close_bg(base), close_label(base), title, close_hover);
    for (i, row) in rows.iter().enumerate() {
        let r = row_rect(o, w, i);
        let hovered = point_in(mouse[0], mouse[1], r);
        let tint = if hovered {
            theme::HOVER_TINT
        } else if row.selected {
            theme::SELECTED_TINT
        } else {
            ROW_TINT
        };
        place_rounded(
            world,
            row_bg(base, i),
            theme::highlight_rect(r),
            tint,
            theme::CONTROL_RADIUS,
            true,
        );
        let label_x = match row.check {
            Some(on) => {
                let box_tint = if on { BOX_TINT_ON } else { BOX_TINT_OFF };
                place_rounded(
                    world,
                    check_box(base, i),
                    [
                        r[0] + PAD,
                        r[1] + (ROW_H - BOX_SIZE) * 0.5,
                        BOX_SIZE,
                        BOX_SIZE,
                    ],
                    box_tint,
                    4.0,
                    true,
                );
                r[0] + CHECK_LABEL_INSET
            }
            None => r[0] + PAD,
        };
        if let Some(l) = widget::label_mut(world, row_label(base, i)) {
            l.x = label_x;
            l.y = r[1] + LABEL_TOP;
            l.align = TextAlign::Left;
            l.color = theme::LABEL;
            l.visible = true;
            l.content = row.caption.clone();
        }
    }
}

// The row index at `(mx, my)`, or `None` if the point is outside every row. A
// caller then falls back to a panel-wide "consume" hit if it wants to swallow
// clicks that land on the panel but miss a row.
pub(crate) fn hit_row(mx: f32, my: f32, o: [f32; 2], w: f32, rows: usize) -> Option<usize> {
    (0..rows).find(|&i| point_in(mx, my, row_rect(o, w, i)))
}

// The sprite ids a row-list panel injects: the panel surface, the close-button
// background, every row highlight, and (when the rows carry checkboxes) every
// checkbox.
pub(crate) fn all_sprite_ids(base: u32, rows: usize, checkboxes: bool) -> Vec<AssetId> {
    let mut ids = vec![panel_bg(base), close_bg(base)];
    ids.extend((0..rows).map(|i| row_bg(base, i)));
    if checkboxes {
        ids.extend((0..rows).map(|i| check_box(base, i)));
    }
    ids
}

// The label ids a row-list panel injects: the title / close labels and every row
// label.
pub(crate) fn all_label_ids(base: u32, rows: usize) -> Vec<AssetId> {
    let mut ids = vec![title_label(base), close_label(base)];
    ids.extend((0..rows).map(|i| row_label(base, i)));
    ids
}

// Hide every listed element (the F1-hidden pass, or when a panel is toggled off).
pub(crate) fn hide_all(world: &mut World, sprite_ids: &[AssetId], label_ids: &[AssetId]) {
    for &id in sprite_ids {
        widget::set_sprite_visible(world, id, false);
    }
    for &id in label_ids {
        widget::set_label_visible(world, id, false);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::assets::{Sprite, TextLabel};

    // A scratch family well clear of every real allocation in `registry.rs`.
    const BASE: u32 = 0x3000_0000 + 0x1F00;

    fn injected_world(rows: usize) -> World {
        let mut world = World::new_empty();
        for id in all_sprite_ids(BASE, rows, true) {
            world.add_component(Sprite {
                asset_id: id,
                ..Default::default()
            });
        }
        for id in all_label_ids(BASE, rows) {
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

    #[test]
    fn geometry_stacks_rows_below_the_title_bar() {
        let o = [40.0, 60.0];
        let w = 200.0;
        assert_eq!(title_rect(o, w), [40.0, 60.0, w, widget::TITLE_H]);
        assert_eq!(row_rect(o, w, 0)[1], 60.0 + widget::TITLE_H);
        assert_eq!(row_rect(o, w, 1)[1], 60.0 + widget::TITLE_H + ROW_H);
        assert_eq!(size(w, 3)[1], widget::TITLE_H + 3.0 * ROW_H + BOTTOM_PAD);
        assert_eq!(
            panel_rect(o, w, 2)[3],
            widget::TITLE_H + 2.0 * ROW_H + BOTTOM_PAD
        );
    }

    #[test]
    fn id_family_is_disjoint_across_sub_ranges() {
        assert_eq!(panel_bg(BASE), AssetId(BASE));
        assert_eq!(close_bg(BASE), AssetId(BASE + 2));
        assert_eq!(row_bg(BASE, 0), AssetId(BASE + 0x10));
        assert_eq!(check_box(BASE, 0), AssetId(BASE + 0x20));
        assert_eq!(row_label(BASE, 0), AssetId(BASE + 0x30));
    }

    // The whole panel is one rounded chrome surface; rows highlight over it.
    #[test]
    fn apply_draws_the_rounded_panel_surface() {
        let mut world = injected_world(1);
        apply(
            &mut world,
            BASE,
            [20.0, 30.0],
            200.0,
            "Panel",
            &[Row::label("a")],
            [0.0, 0.0],
        );
        let bg = sprite(&world, panel_bg(BASE));
        assert!(bg.visible);
        assert_eq!((bg.x, bg.y), (20.0, 30.0));
        assert_eq!((bg.width, bg.height), (200.0, size(200.0, 1)[1]));
        assert_eq!(bg.corner_radius, theme::PANEL_RADIUS);
    }

    #[test]
    fn hit_row_resolves_a_row_or_misses() {
        let o = [10.0, 10.0];
        let w = 200.0;
        let r1 = row_rect(o, w, 1);
        assert_eq!(hit_row(r1[0] + 5.0, r1[1] + 5.0, o, w, 3), Some(1));
        // Above the first row (in the title bar) and below the last row: misses.
        assert_eq!(hit_row(o[0] + 5.0, o[1] + 5.0, o, w, 3), None);
        assert_eq!(hit_row(o[0] + 5.0, 10_000.0, o, w, 3), None);
    }

    #[test]
    fn apply_draws_title_close_and_a_checkbox_row() {
        let mut world = injected_world(1);
        let o = [20.0, 20.0];
        let w = 200.0;
        apply(
            &mut world,
            BASE,
            o,
            w,
            "Panel",
            &[Row::checkbox("Toggle", true)],
            [0.0, 0.0],
        );
        let title = world
            .query::<TextLabel>()
            .find(|l| l.asset_id == title_label(BASE))
            .unwrap();
        assert!(title.visible && title.content == "Panel");
        // The close button always shows its "X".
        let close = world
            .query::<TextLabel>()
            .find(|l| l.asset_id == close_label(BASE))
            .unwrap();
        assert!(close.visible && close.content == "X");
        // The checkbox is green while on and the label is inset past the box.
        assert_eq!(sprite(&world, check_box(BASE, 0)).tint, BOX_TINT_ON);
        let label = world
            .query::<TextLabel>()
            .find(|l| l.asset_id == row_label(BASE, 0))
            .unwrap();
        assert_eq!(label.content, "Toggle");
        assert_eq!(label.x, o[0] + CHECK_LABEL_INSET);
        // Off flips the checkbox tint.
        apply(
            &mut world,
            BASE,
            o,
            w,
            "Panel",
            &[Row::checkbox("Toggle", false)],
            [0.0, 0.0],
        );
        assert_eq!(sprite(&world, check_box(BASE, 0)).tint, BOX_TINT_OFF);
    }

    #[test]
    fn label_only_row_insets_by_pad_not_a_box() {
        // A label-only panel injects no checkbox ids (`all_sprite_ids(.., false)`),
        // so its rows have nothing to inset past: the label sits at PAD.
        let mut world = World::new_empty();
        for id in all_sprite_ids(BASE, 1, false) {
            world.add_component(Sprite {
                asset_id: id,
                ..Default::default()
            });
        }
        for id in all_label_ids(BASE, 1) {
            world.add_component(TextLabel {
                asset_id: id,
                ..Default::default()
            });
        }
        let o = [20.0, 20.0];
        apply(
            &mut world,
            BASE,
            o,
            200.0,
            "Panel",
            &[Row::label("Just text")],
            [0.0, 0.0],
        );
        let label = world
            .query::<TextLabel>()
            .find(|l| l.asset_id == row_label(BASE, 0))
            .unwrap();
        assert_eq!(
            label.x,
            o[0] + PAD,
            "label-only rows inset by PAD, not a box"
        );
    }

    #[test]
    fn hovered_and_selected_rows_are_tinted() {
        let mut world = injected_world(2);
        let o = [20.0, 20.0];
        let w = 200.0;
        let r0 = row_rect(o, w, 0);
        apply(
            &mut world,
            BASE,
            o,
            w,
            "Panel",
            &[Row::label("a"), Row::label("b").select(true)],
            [r0[0] + 5.0, r0[1] + 5.0],
        );
        let hovered = sprite(&world, row_bg(BASE, 0));
        assert_eq!(hovered.tint, theme::HOVER_TINT);
        assert!(
            hovered.height < ROW_H && hovered.corner_radius > 0.0,
            "the highlight is an inset rounded pill, not a full-width band"
        );
        assert_eq!(
            sprite(&world, row_bg(BASE, 1)).tint,
            theme::SELECTED_TINT,
            "the selected row is highlighted without a hover"
        );
        // An idle, unselected row draws no highlight at all.
        apply(
            &mut world,
            BASE,
            o,
            w,
            "Panel",
            &[Row::label("a"), Row::label("b")],
            [0.0, 0.0],
        );
        assert_eq!(sprite(&world, row_bg(BASE, 0)).tint[3], 0.0);
    }

    #[test]
    fn hide_all_blanks_every_listed_element() {
        let mut world = injected_world(2);
        apply(
            &mut world,
            BASE,
            [20.0, 20.0],
            200.0,
            "Panel",
            &[Row::checkbox("a", true), Row::checkbox("b", false)],
            [0.0, 0.0],
        );
        hide_all(
            &mut world,
            &all_sprite_ids(BASE, 2, true),
            &all_label_ids(BASE, 2),
        );
        assert!(world.query::<Sprite>().all(|s| !s.visible));
        assert!(world.query::<TextLabel>().all(|l| !l.visible));
    }
}
