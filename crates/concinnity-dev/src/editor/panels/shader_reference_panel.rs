//! The reference column's layout half (see `shader_reference.rs` for the
//! rows): a well at the right of the Shader source panel, one row per group
//! heading or name, names in the code face, the hovered row highlighted.

use concinnity_core::components::TextAlign;
use concinnity_core::ecs::World;
use concinnity_core::ecs::asset_id::AssetId;

use super::registry::{self, PanelKey};
use super::shader_reference::RefRow;
use crate::editor::text_area::layout::Metrics;
use crate::editor::theme;
use crate::editor::widget::{self, place_rounded, point_in};

const BASE: u32 = registry::base(PanelKey::ShaderSource);
const COLUMN_BG: AssetId = AssetId(BASE + 0x300);
const POOL: usize = 48;
fn row_bg(i: usize) -> AssetId {
    AssetId(BASE + 0x340 + i as u32)
}
fn heading_label(i: usize) -> AssetId {
    AssetId(BASE + 0x380 + i as u32)
}
fn name_label(i: usize) -> AssetId {
    AssetId(BASE + 0x3C0 + i as u32)
}

pub(crate) const COLUMN_W: f32 = 250.0;
const ROW_H: f32 = 22.0;
const INSET: f32 = 4.0;
const TEXT_PAD: f32 = 8.0;
const WELL_TINT: [f32; 4] = [0.08, 0.08, 0.10, 1.0];
const HEADING_COLOR: [f32; 3] = [0.58, 0.66, 0.80];

// The per-frame view the hook assembles.
pub(crate) struct RefView<'a> {
    pub rows: &'a [RefRow<'a>],
    pub scroll: usize,
    pub mouse: [f32; 2],
}

// How many rows the column at `rect` shows.
pub(crate) fn rows_shown(rect: [f32; 4]) -> usize {
    (((rect[3] - 2.0 * INSET) / ROW_H).floor().max(1.0) as usize).min(POOL)
}

fn row_rect(rect: [f32; 4], slot: usize) -> [f32; 4] {
    [
        rect[0],
        rect[1] + INSET + slot as f32 * ROW_H,
        rect[2],
        ROW_H,
    ]
}

// The shown row under a point, counted from the first shown row.
pub(crate) fn slot_at(rect: [f32; 4], mx: f32, my: f32) -> Option<usize> {
    if !point_in(mx, my, rect) || my < rect[1] + INSET {
        return None;
    }
    let slot = ((my - rect[1] - INSET) / ROW_H).floor() as usize;
    (slot < rows_shown(rect)).then_some(slot)
}

// The row under the cursor.
pub(crate) fn hovered<'v, 'a>(view: &'v RefView<'a>, rect: [f32; 4]) -> Option<&'v RefRow<'a>> {
    let slot = slot_at(rect, view.mouse[0], view.mouse[1])?;
    view.rows.get(view.scroll + slot)
}

// Position and show the column in `rect` (`Some(view)`), or blank it.
pub(crate) fn place(world: &mut World, view: Option<&RefView>, rect: [f32; 4]) {
    let Some(view) = view else {
        return hide_all(world);
    };
    place_rounded(
        world,
        COLUMN_BG,
        rect,
        WELL_TINT,
        theme::CONTROL_RADIUS,
        true,
    );
    let m = Metrics::code();
    let shown = rows_shown(rect);
    let hover = slot_at(rect, view.mouse[0], view.mouse[1]);
    for slot in 0..POOL {
        let row = (slot < shown)
            .then(|| view.rows.get(view.scroll + slot))
            .flatten();
        let r = row_rect(rect, slot);
        let lit = row.is_some() && hover == Some(slot);
        place_rounded(
            world,
            row_bg(slot),
            theme::highlight_rect(r),
            theme::HOVER_TINT,
            theme::CONTROL_RADIUS,
            lit,
        );
        let heading = matches!(row, Some(RefRow::Heading { .. }));
        let name = matches!(row, Some(RefRow::Name(_)));
        let text = row.map(RefRow::text).unwrap_or_default();
        if let Some(l) = widget::label_mut(world, heading_label(slot)) {
            l.x = r[0] + TEXT_PAD;
            l.y = r[1] + ROW_H * 0.5 - theme::TEXT_HALF;
            l.align = TextAlign::Left;
            l.color = HEADING_COLOR;
            l.visible = heading;
            l.content = if heading { text.clone() } else { String::new() };
        }
        if let Some(l) = widget::label_mut(world, name_label(slot)) {
            l.x = r[0] + TEXT_PAD * 2.0;
            l.y = r[1] + (ROW_H - m.text_px) * 0.5;
            l.align = TextAlign::Left;
            l.scale = m.label_scale;
            l.color = theme::CODE_ENGINE;
            l.wrap_width = (r[2] - TEXT_PAD * 3.0).max(0.0);
            l.max_lines = 1;
            l.visible = name;
            l.content = if name { text } else { String::new() };
        }
    }
}

pub(crate) fn hide_all(world: &mut World) {
    let mut labels = label_ids();
    labels.extend(code_label_ids());
    widget::hide_all(world, &sprite_ids(), &labels, &[]);
}

pub(crate) fn sprite_ids() -> Vec<AssetId> {
    let mut ids = vec![COLUMN_BG];
    ids.extend((0..POOL).map(row_bg));
    ids
}

pub(crate) fn label_ids() -> Vec<AssetId> {
    (0..POOL).map(heading_label).collect()
}

// The names, drawn in the code face.
pub(crate) fn code_label_ids() -> Vec<AssetId> {
    (0..POOL).map(name_label).collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::editor::panels::shader_reference::Reference;
    use concinnity_core::components::{Sprite, TextLabel};
    use concinnity_core::render::shader_programs::vocabulary::ENTRIES;

    const RECT: [f32; 4] = [100.0, 50.0, COLUMN_W, 10.0 * ROW_H + 2.0 * INSET];

    fn injected_world() -> World {
        let mut labels = label_ids();
        labels.extend(code_label_ids());
        crate::test_support::injected_world(&sprite_ids(), &labels, &[])
    }

    fn label(world: &World, id: AssetId) -> TextLabel {
        world.get_by_id::<TextLabel>(id).cloned().unwrap()
    }

    #[test]
    fn rows_fill_the_column_from_the_scroll_window() {
        let mut world = injected_world();
        let rows = Reference::default().rows(ENTRIES);
        let view = RefView {
            rows: &rows,
            scroll: 1,
            mouse: [0.0, 0.0],
        };
        place(&mut world, Some(&view), RECT);
        assert_eq!(rows_shown(RECT), 10);
        let first = label(&world, name_label(0));
        assert!(first.visible && first.content == "shade_surface()");
        assert!(!label(&world, heading_label(0)).visible);
        assert!(!label(&world, name_label(10)).visible, "past the rect");
    }

    #[test]
    fn a_heading_draws_in_the_heading_face_and_hover_lights_its_row() {
        let mut world = injected_world();
        let rows = Reference::default().rows(ENTRIES);
        let r = row_rect(RECT, 0);
        let view = RefView {
            rows: &rows,
            scroll: 0,
            mouse: [r[0] + 5.0, r[1] + 5.0],
        };
        place(&mut world, Some(&view), RECT);
        let heading = label(&world, heading_label(0));
        assert!(heading.visible && heading.content.starts_with("- Helpers"));
        assert!(!label(&world, name_label(0)).visible);
        assert!(world.get_by_id::<Sprite>(row_bg(0)).unwrap().visible);
        assert!(!world.get_by_id::<Sprite>(row_bg(1)).unwrap().visible);
        assert_eq!(hovered(&view, RECT), Some(&rows[0]));
    }

    #[test]
    fn slot_at_maps_points_to_shown_rows() {
        let r = row_rect(RECT, 3);
        assert_eq!(slot_at(RECT, r[0] + 1.0, r[1] + 1.0), Some(3));
        assert_eq!(slot_at(RECT, RECT[0] - 1.0, r[1] + 1.0), None);
        assert_eq!(
            slot_at(RECT, RECT[0] + 1.0, RECT[1] + 1.0),
            None,
            "the inset"
        );
    }

    #[test]
    fn hide_all_blanks_the_column() {
        let mut world = injected_world();
        let rows = Reference::default().rows(ENTRIES);
        let view = RefView {
            rows: &rows,
            scroll: 0,
            mouse: [0.0, 0.0],
        };
        place(&mut world, Some(&view), RECT);
        place(&mut world, None, RECT);
        assert!(world.query::<Sprite>().all(|s| !s.visible));
        assert!(world.query::<TextLabel>().all(|l| !l.visible));
    }
}
