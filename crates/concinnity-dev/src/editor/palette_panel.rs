// src/editor/palette_panel.rs
//
// The command palette's layout half: a centered floating panel with one text
// input and a ranked result window under it, each row a caption, a dimmed
// hint, and its category tag. The data model and ranking live in
// `editor/palette/`; the drive in `hook/palette_edit.rs`.

use super::registry::{self, PanelKey};
use super::theme;
use super::widget::{self, point_in};
use crate::ecs::World;
use crate::ecs::asset_id::AssetId;

const BASE: u32 = registry::base(PanelKey::Palette);
pub(crate) const PANEL_BG: AssetId = AssetId(BASE);
pub(crate) const TITLE_LABEL: AssetId = AssetId(BASE + 2);
pub(crate) const CLOSE_BG: AssetId = AssetId(BASE + 3);
pub(crate) const CLOSE_LABEL: AssetId = AssetId(BASE + 4);
pub(crate) const INPUT: AssetId = AssetId(BASE + 5);

fn row_bg(i: usize) -> AssetId {
    AssetId(BASE + 0x10 + i as u32)
}
fn row_label(i: usize) -> AssetId {
    AssetId(BASE + 0x40 + i as u32)
}
fn row_hint(i: usize) -> AssetId {
    AssetId(BASE + 0x70 + i as u32)
}
fn row_tag(i: usize) -> AssetId {
    AssetId(BASE + 0xA0 + i as u32)
}

// Geometry, in window pixels. Every rect derives from the panel origin `o`, so
// dragging the title bar moves the whole panel.
pub(crate) const PALETTE_W: f32 = 560.0;
const PAD: f32 = 10.0;
const INPUT_H: f32 = 28.0;
const ROW_H: f32 = 24.0;
// Visible result rows; a longer match list scrolls.
pub(crate) const ROW_POOL: usize = 10;
// Character budgets for the three row columns at the fixed width.
const CAPTION_CHARS: usize = 26;
const HINT_CHARS: usize = 24;
// Column offsets inside a row.
const HINT_X: f32 = 240.0;
const TAG_W: f32 = 70.0;

// One visible result row, borrowed from the hook's item list.
pub(crate) struct PaletteRow<'a> {
    pub caption: &'a str,
    pub hint: &'a str,
    pub tag: &'a str,
}

// The per-frame view the hook assembles. `rows` is the visible window (already
// cut by `scroll`); `selected` counts filtered matches, like the scroll.
pub(crate) struct PaletteView<'a> {
    pub rows: Vec<PaletteRow<'a>>,
    pub selected: usize,
    pub scroll: usize,
    pub total: usize,
    // Whether the input asserts keyboard focus this frame.
    pub focus: bool,
    pub mouse: [f32; 2],
}

// A resolved palette-body click.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum PaletteHit {
    // Give the input keyboard focus (it normally holds it anyway).
    FocusInput,
    // Commit visible row `slot`.
    Row(usize),
    // A click elsewhere on the panel: swallowed so it cannot reach the world.
    Consume,
}

// Centered near the top of the viewport, launcher-style.
pub(crate) fn default_origin(vp: [f32; 2]) -> [f32; 2] {
    let s = size();
    [
        ((vp[0] - s[0]) * 0.5).max(0.0),
        (vp[1] * 0.18).max(super::hud::body_top()),
    ]
}

pub(crate) fn size() -> [f32; 2] {
    [
        PALETTE_W,
        widget::TITLE_H + INPUT_H + ROW_POOL as f32 * ROW_H + 2.0 * PAD,
    ]
}

pub(crate) fn input_rect(o: [f32; 2]) -> [f32; 4] {
    [
        o[0] + PAD,
        o[1] + widget::TITLE_H + PAD * 0.5,
        PALETTE_W - 2.0 * PAD,
        INPUT_H,
    ]
}

fn rows_top(o: [f32; 2]) -> f32 {
    o[1] + widget::TITLE_H + INPUT_H + PAD
}

pub(crate) fn row_rect(o: [f32; 2], slot: usize) -> [f32; 4] {
    [
        o[0] + PAD * 0.5,
        rows_top(o) + slot as f32 * ROW_H,
        PALETTE_W - PAD,
        ROW_H,
    ]
}

// Whether the cursor is over the scrollable result window (for wheel routing).
pub(crate) fn cursor_over_rows(mx: f32, my: f32, o: [f32; 2]) -> bool {
    let top = rows_top(o);
    mx >= o[0] && mx < o[0] + PALETTE_W && my >= top && my < top + ROW_POOL as f32 * ROW_H
}

// Resolve a click at `(mx, my)` against the panel at origin `o`. `None` means
// the click missed the panel; the title bar never reaches this.
pub(crate) fn hit_test(view: &PaletteView, mx: f32, my: f32, o: [f32; 2]) -> Option<PaletteHit> {
    if point_in(mx, my, input_rect(o)) {
        return Some(PaletteHit::FocusInput);
    }
    for slot in 0..view.rows.len() {
        if point_in(mx, my, row_rect(o, slot)) {
            return Some(PaletteHit::Row(slot));
        }
    }
    point_in(mx, my, widget::outer_rect(o, size())).then_some(PaletteHit::Consume)
}

// Position + show the panel (`Some(view)`), or blank every element (`None`).
pub(crate) fn apply(world: &mut World, view: Option<&PaletteView>, o: [f32; 2]) {
    let Some(view) = view else {
        hide_all(world);
        return;
    };
    widget::place_panel(world, PANEL_BG, widget::outer_rect(o, size()));
    let title = widget::title_rect(o, PALETTE_W);
    widget::place_heading(world, TITLE_LABEL, title, "Palette");
    let close_hover = point_in(view.mouse[0], view.mouse[1], widget::close_rect(title));
    widget::place_close(world, CLOSE_BG, CLOSE_LABEL, title, close_hover);
    widget::show_field(world, INPUT, input_rect(o), view.focus);

    for slot in 0..ROW_POOL {
        let Some(row) = view.rows.get(slot) else {
            widget::set_sprite_visible(world, row_bg(slot), false);
            widget::set_label_visible(world, row_hint(slot), false);
            widget::set_label_visible(world, row_tag(slot), false);
            if slot == 0 && view.total == 0 {
                empty_row(world, o);
            } else {
                widget::set_label_visible(world, row_label(slot), false);
            }
            continue;
        };
        let r = row_rect(o, slot);
        let tint = if view.scroll + slot == view.selected {
            theme::SELECTED_TINT
        } else if point_in(view.mouse[0], view.mouse[1], r) {
            theme::HOVER_TINT
        } else {
            [0.0; 4]
        };
        widget::place_rounded(
            world,
            row_bg(slot),
            theme::highlight_rect(r),
            tint,
            theme::CONTROL_RADIUS,
            tint[3] > 0.0,
        );
        let text_y = r[1] + ROW_H * 0.5 - theme::TEXT_HALF;
        widget::place_left_label(
            world,
            row_label(slot),
            [r[0] + 8.0, text_y],
            &widget::clip_text(row.caption, CAPTION_CHARS),
            theme::LABEL,
            true,
        );
        widget::place_left_label(
            world,
            row_hint(slot),
            [r[0] + HINT_X, text_y],
            &widget::clip_text(row.hint, HINT_CHARS),
            theme::LABEL_DIM,
            true,
        );
        widget::place_left_label(
            world,
            row_tag(slot),
            [r[0] + r[2] - TAG_W, text_y],
            row.tag,
            theme::LABEL_DIM,
            true,
        );
    }
}

// The result window with nothing kept: one dim row saying so, instead of a
// silently blank body.
fn empty_row(world: &mut World, o: [f32; 2]) {
    let r = row_rect(o, 0);
    widget::place_left_label(
        world,
        row_label(0),
        [r[0] + 8.0, r[1] + ROW_H * 0.5 - theme::TEXT_HALF],
        "no match",
        theme::LABEL_DIM,
        true,
    );
}

// Hide every panel element, blurring the input so a hidden field cannot keep
// keyboard focus.
pub(crate) fn hide_all(world: &mut World) {
    widget::hide_all(world, &all_sprite_ids(), &all_label_ids(), &all_field_ids());
}

pub(crate) fn all_sprite_ids() -> Vec<AssetId> {
    let mut ids = vec![PANEL_BG, CLOSE_BG];
    ids.extend((0..ROW_POOL).map(row_bg));
    ids
}

pub(crate) fn all_label_ids() -> Vec<AssetId> {
    let mut ids = vec![TITLE_LABEL, CLOSE_LABEL];
    for i in 0..ROW_POOL {
        ids.push(row_label(i));
        ids.push(row_hint(i));
        ids.push(row_tag(i));
    }
    ids
}

pub(crate) fn all_field_ids() -> Vec<AssetId> {
    vec![INPUT]
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::components::{Sprite, TextInput, TextLabel};

    fn injected_world() -> World {
        let mut world = World::new();
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
        for id in all_field_ids() {
            world.add_component(TextInput {
                asset_id: id,
                ..Default::default()
            });
        }
        world
    }

    fn rows(n: usize) -> Vec<(String, String)> {
        (0..n)
            .map(|i| (format!("item {i}"), format!("hint {i}")))
            .collect()
    }

    fn view<'a>(backing: &'a [(String, String)], selected: usize) -> PaletteView<'a> {
        PaletteView {
            rows: backing
                .iter()
                .map(|(c, h)| PaletteRow {
                    caption: c,
                    hint: h,
                    tag: "entity",
                })
                .collect(),
            selected,
            scroll: 0,
            total: backing.len(),
            focus: true,
            mouse: [0.0, 0.0],
        }
    }

    #[test]
    fn hit_test_resolves_rows_input_or_swallows() {
        let o = [40.0, 40.0];
        let backing = rows(3);
        let v = view(&backing, 0);
        let i = input_rect(o);
        assert_eq!(
            hit_test(&v, i[0] + 5.0, i[1] + 5.0, o),
            Some(PaletteHit::FocusInput)
        );
        let r1 = row_rect(o, 1);
        assert_eq!(
            hit_test(&v, r1[0] + 5.0, r1[1] + 5.0, o),
            Some(PaletteHit::Row(1))
        );
        // An empty slot below the last row is panel chrome, not a row.
        let r5 = row_rect(o, 5);
        assert_eq!(
            hit_test(&v, r5[0] + 5.0, r5[1] + 5.0, o),
            Some(PaletteHit::Consume)
        );
        assert_eq!(hit_test(&v, 5000.0, 5000.0, o), None);
    }

    #[test]
    fn rows_stack_under_the_input_inside_the_panel() {
        let o = [0.0, 0.0];
        let i = input_rect(o);
        let first = row_rect(o, 0);
        assert!(first[1] >= i[1] + i[3], "rows start below the input");
        let last = row_rect(o, ROW_POOL - 1);
        let p = widget::outer_rect(o, size());
        assert!(
            last[1] + last[3] <= p[1] + p[3],
            "rows stay inside the panel"
        );
        assert!(cursor_over_rows(10.0, first[1] + 5.0, o));
        assert!(!cursor_over_rows(10.0, i[1] + 5.0, o));
    }

    #[test]
    fn apply_draws_the_window_and_highlights_the_selection() {
        let mut world = injected_world();
        let backing = rows(3);
        apply(&mut world, Some(&view(&backing, 1)), [20.0, 20.0]);
        let caption = world
            .query::<TextLabel>()
            .find(|l| l.asset_id == row_label(0))
            .unwrap();
        assert!(caption.visible && caption.content == "item 0");
        let tag = world
            .query::<TextLabel>()
            .find(|l| l.asset_id == row_tag(0))
            .unwrap();
        assert!(tag.visible && tag.content == "entity");
        // Only the selected row keeps a background with the cursor away.
        let lit: Vec<usize> = (0..ROW_POOL)
            .filter(|&i| {
                world
                    .query::<Sprite>()
                    .find(|s| s.asset_id == row_bg(i))
                    .is_some_and(|s| s.visible)
            })
            .collect();
        assert_eq!(lit, vec![1]);
        // Slots past the window are blank.
        assert!(
            !world
                .query::<TextLabel>()
                .find(|l| l.asset_id == row_label(3))
                .unwrap()
                .visible
        );
        let input = world.query::<TextInput>().next().unwrap();
        assert!(input.visible && input.focused);
    }

    #[test]
    fn an_empty_window_says_no_match() {
        let mut world = injected_world();
        let backing = rows(0);
        apply(&mut world, Some(&view(&backing, 0)), [20.0, 20.0]);
        let first = world
            .query::<TextLabel>()
            .find(|l| l.asset_id == row_label(0))
            .unwrap();
        assert!(first.visible && first.content == "no match");
    }

    #[test]
    fn hide_all_blanks_every_element() {
        let mut world = injected_world();
        let backing = rows(ROW_POOL);
        apply(&mut world, Some(&view(&backing, 0)), [20.0, 20.0]);
        apply(&mut world, None, [0.0, 0.0]);
        assert!(world.query::<Sprite>().all(|s| !s.visible));
        assert!(world.query::<TextLabel>().all(|l| !l.visible));
        let input = world.query::<TextInput>().next().unwrap();
        assert!(!input.visible && !input.focused);
    }
}
