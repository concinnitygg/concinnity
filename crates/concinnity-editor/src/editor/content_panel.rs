// src/editor/content_panel.rs
//
// The Content panel: a thumbnail grid over the world's visual assets
// (textures, materials, meshes, models, environment maps, prefabs). Cells show
// the baked thumbnail when one exists (`editor/thumbs.rs`) and a typed icon
// chip otherwise; a search field ranks matches (`editor/filter.rs`) and a type
// chip cycles the kind filter. Pure geometry + draw here; the item assembly
// and click handling live in `hook/content_edit.rs`.

use super::billboards;
use super::registry::{self, PanelKey};
use super::theme;
use super::thumbs::{self, Thumb};
use super::widget::{self, point_in};
use crate::ecs::World;
use crate::ecs::asset_id::AssetId;

const BASE: u32 = registry::base(PanelKey::Content);

pub(crate) const PANEL_BG: AssetId = AssetId(BASE);
const TITLE_LABEL: AssetId = AssetId(BASE + 1);
const CLOSE_BG: AssetId = AssetId(BASE + 2);
const CLOSE_LABEL: AssetId = AssetId(BASE + 3);
const TYPE_BG: AssetId = AssetId(BASE + 4);
const TYPE_LABEL: AssetId = AssetId(BASE + 5);
const STATUS_LABEL: AssetId = AssetId(BASE + 6);
pub(crate) const SEARCH_INPUT: AssetId = AssetId(BASE + 7);

const fn cell_bg(i: usize) -> AssetId {
    AssetId(BASE + 0x20 + i as u32)
}
const fn cell_thumb(i: usize) -> AssetId {
    AssetId(BASE + 0x40 + i as u32)
}
const fn cell_glyph(i: usize) -> AssetId {
    AssetId(BASE + 0x60 + i as u32)
}
const fn cell_name(i: usize) -> AssetId {
    AssetId(BASE + 0x80 + i as u32)
}

pub(crate) const COLS: usize = 4;
pub(crate) const GRID_ROWS: usize = 4;
pub(crate) const CELLS: usize = COLS * GRID_ROWS;
const CELL_W: f32 = 92.0;
const CELL_H: f32 = 112.0;
const THUMB_H: f32 = 84.0;
const GAP: f32 = 6.0;
const PAD: f32 = 8.0;
// The search field + type chip row below the title bar.
const CTRL_H: f32 = 30.0;
const STATUS_H: f32 = 20.0;

pub(crate) fn size() -> [f32; 2] {
    [
        PAD * 2.0 + COLS as f32 * CELL_W + (COLS as f32 - 1.0) * GAP,
        widget::TITLE_H + CTRL_H + GRID_ROWS as f32 * (CELL_H + GAP) + STATUS_H,
    ]
}

pub(crate) fn default_origin(vp: [f32; 2]) -> [f32; 2] {
    // Right side, clear of the Assets panel's left anchor.
    [(vp[0] - size()[0] - 16.0).max(0.0), super::hud::body_top()]
}

pub(crate) fn panel_rect(o: [f32; 2]) -> [f32; 4] {
    widget::outer_rect(o, size())
}

fn search_rect(o: [f32; 2]) -> [f32; 4] {
    let w = size()[0];
    [
        o[0] + PAD,
        o[1] + widget::TITLE_H + 3.0,
        w - PAD * 2.0 - 96.0 - GAP,
        CTRL_H - 6.0,
    ]
}

fn type_rect(o: [f32; 2]) -> [f32; 4] {
    let w = size()[0];
    [
        o[0] + w - PAD - 96.0,
        o[1] + widget::TITLE_H + 3.0,
        96.0,
        CTRL_H - 6.0,
    ]
}

pub(crate) fn cell_rect(o: [f32; 2], slot: usize) -> [f32; 4] {
    let (col, row) = (slot % COLS, slot / COLS);
    [
        o[0] + PAD + col as f32 * (CELL_W + GAP),
        o[1] + widget::TITLE_H + CTRL_H + row as f32 * (CELL_H + GAP),
        CELL_W,
        CELL_H,
    ]
}

fn thumb_rect(cell: [f32; 4]) -> [f32; 4] {
    [cell[0] + 4.0, cell[1] + 4.0, cell[2] - 8.0, THUMB_H]
}

fn status_rect(o: [f32; 2]) -> [f32; 4] {
    let s = size();
    [
        o[0] + PAD,
        o[1] + s[1] - STATUS_H + 2.0,
        s[0] - PAD * 2.0,
        widget::LINE_H,
    ]
}

// A resolved Content-panel click.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum ContentAction {
    FocusSearch,
    CycleType,
    SelectCell(usize),
    // A click elsewhere on the panel: swallowed (and blurs the search field).
    Consume,
}

pub(crate) fn hit_test(mx: f32, my: f32, o: [f32; 2], shown: usize) -> Option<ContentAction> {
    if point_in(mx, my, search_rect(o)) {
        return Some(ContentAction::FocusSearch);
    }
    if point_in(mx, my, type_rect(o)) {
        return Some(ContentAction::CycleType);
    }
    for slot in 0..shown.min(CELLS) {
        if point_in(mx, my, cell_rect(o, slot)) {
            return Some(ContentAction::SelectCell(slot));
        }
    }
    point_in(mx, my, panel_rect(o)).then_some(ContentAction::Consume)
}

// One grid cell to draw.
pub(crate) struct CellView {
    pub name: String,
    pub asset_type: String,
    pub thumb: Option<Thumb>,
    pub selected: bool,
}

pub(crate) struct ContentView<'a> {
    pub cells: &'a [CellView],
    // Items matching the filters, beyond the visible window.
    pub total: usize,
    pub type_caption: &'a str,
    pub search_focus: bool,
    pub mouse: [f32; 2],
}

pub(crate) fn apply(world: &mut World, view: &ContentView, o: [f32; 2]) {
    widget::place_panel(world, PANEL_BG, panel_rect(o));
    let title = widget::title_rect(o, size()[0]);
    widget::place_heading(world, TITLE_LABEL, title, "Content");
    let close_hover = point_in(view.mouse[0], view.mouse[1], widget::close_rect(title));
    widget::place_close(world, CLOSE_BG, CLOSE_LABEL, title, close_hover);
    widget::show_field(world, SEARCH_INPUT, search_rect(o), view.search_focus);
    let tr = type_rect(o);
    widget::place_rounded(
        world,
        TYPE_BG,
        tr,
        theme::BUTTON_TINT,
        theme::CONTROL_RADIUS,
        true,
    );
    widget::place_message(
        world,
        TYPE_LABEL,
        [
            tr[0] + 6.0,
            tr[1] + (tr[3] - widget::LINE_H) * 0.5,
            tr[2] - 12.0,
            widget::LINE_H,
        ],
        view.type_caption,
        theme::LABEL,
        true,
    );
    for slot in 0..CELLS {
        match view.cells.get(slot) {
            Some(cell) => place_cell(world, slot, cell, o, view.mouse),
            None => hide_cell(world, slot),
        }
    }
    let status = if view.total > view.cells.len() {
        format!("{} of {} assets", view.cells.len(), view.total)
    } else {
        format!("{} asset(s)", view.total)
    };
    widget::place_message(
        world,
        STATUS_LABEL,
        status_rect(o),
        &status,
        theme::LABEL_DIM,
        true,
    );
}

fn place_cell(world: &mut World, slot: usize, cell: &CellView, o: [f32; 2], mouse: [f32; 2]) {
    let rect = cell_rect(o, slot);
    let hovered = point_in(mouse[0], mouse[1], rect);
    let tint = if cell.selected {
        theme::SELECTED_TINT
    } else if hovered {
        theme::HOVER_TINT
    } else {
        theme::BUTTON_TINT
    };
    widget::place_rounded(
        world,
        cell_bg(slot),
        rect,
        tint,
        theme::CONTROL_RADIUS,
        true,
    );
    let area = thumb_rect(rect);
    match cell.thumb {
        Some(thumb) => {
            let fitted = thumbs::fit_rect(area, thumb.width, thumb.height);
            if let Some(s) = widget::sprite_mut(world, cell_thumb(slot)) {
                s.x = fitted[0];
                s.y = fitted[1];
                s.width = fitted[2];
                s.height = fitted[3];
                s.texture = Some(thumb.handle);
                s.tint = [1.0, 1.0, 1.0, 1.0];
                s.corner_radius = 0.0;
                s.border_width = 0.0;
                s.visible = true;
            }
            widget::set_label_visible(world, cell_glyph(slot), false);
        }
        None => {
            // The typed icon chip: the billboard glyph + hue for the type.
            widget::set_sprite_visible(world, cell_thumb(slot), false);
            if let Some(l) = widget::label_mut(world, cell_glyph(slot)) {
                let tint = billboards::tint(&cell.asset_type);
                l.content = billboards::glyph(&cell.asset_type);
                l.color = [tint[0], tint[1], tint[2]];
                l.x = area[0] + area[2] * 0.5;
                l.y = area[1] + (area[3] - widget::LINE_H) * 0.5;
                l.align = crate::assets::TextAlign::Center;
                l.visible = true;
            }
        }
    }
    widget::place_message(
        world,
        cell_name(slot),
        [
            rect[0] + 4.0,
            rect[1] + 6.0 + THUMB_H,
            rect[2] - 8.0,
            widget::LINE_H,
        ],
        &cell.name,
        theme::LABEL,
        true,
    );
}

fn hide_cell(world: &mut World, slot: usize) {
    widget::set_sprite_visible(world, cell_bg(slot), false);
    widget::set_sprite_visible(world, cell_thumb(slot), false);
    widget::set_label_visible(world, cell_glyph(slot), false);
    widget::set_label_visible(world, cell_name(slot), false);
}

pub(crate) fn all_sprite_ids() -> Vec<AssetId> {
    let mut ids = vec![PANEL_BG, CLOSE_BG, TYPE_BG];
    ids.extend((0..CELLS).map(cell_bg));
    ids.extend((0..CELLS).map(cell_thumb));
    ids
}

pub(crate) fn all_label_ids() -> Vec<AssetId> {
    let mut ids = vec![TITLE_LABEL, CLOSE_LABEL, TYPE_LABEL, STATUS_LABEL];
    ids.extend((0..CELLS).map(cell_glyph));
    ids.extend((0..CELLS).map(cell_name));
    ids
}

pub(crate) fn all_field_ids() -> Vec<(AssetId, &'static str)> {
    vec![(SEARCH_INPUT, "search")]
}

pub(crate) fn hide_all(world: &mut World) {
    widget::hide_all(world, &all_sprite_ids(), &all_label_ids(), &[SEARCH_INPUT]);
}

// Whether the wheel over `(mx, my)` belongs to the grid body.
pub(crate) fn cursor_over_body(mx: f32, my: f32, o: [f32; 2]) -> bool {
    let r = panel_rect(o);
    point_in(mx, my, [r[0], o[1] + widget::TITLE_H + CTRL_H, r[2], r[3]])
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn geometry_grids_cells_inside_the_panel() {
        let o = [100.0, 50.0];
        let panel = panel_rect(o);
        for slot in 0..CELLS {
            let c = cell_rect(o, slot);
            assert!(
                c[0] >= panel[0] && c[0] + c[2] <= panel[0] + panel[2],
                "{slot}"
            );
            assert!(c[1] + c[3] <= panel[1] + panel[3], "{slot}");
        }
        let a = cell_rect(o, 0);
        let b = cell_rect(o, 1);
        assert!(b[0] > a[0] + a[2], "columns do not overlap");
        let below = cell_rect(o, COLS);
        assert!(below[1] > a[1] + a[3], "rows do not overlap");
    }

    #[test]
    fn hit_test_resolves_controls_cells_and_consumes_the_rest() {
        let o = [100.0, 50.0];
        let s = search_rect(o);
        assert_eq!(
            hit_test(s[0] + 5.0, s[1] + 5.0, o, 0),
            Some(ContentAction::FocusSearch)
        );
        let t = type_rect(o);
        assert_eq!(
            hit_test(t[0] + 5.0, t[1] + 5.0, o, 0),
            Some(ContentAction::CycleType)
        );
        let c = cell_rect(o, 5);
        assert_eq!(
            hit_test(c[0] + 5.0, c[1] + 5.0, o, 16),
            Some(ContentAction::SelectCell(5))
        );
        assert_eq!(
            hit_test(c[0] + 5.0, c[1] + 5.0, o, 3),
            Some(ContentAction::Consume),
            "an empty slot swallows the click"
        );
        assert_eq!(hit_test(2000.0, 2000.0, o, 16), None);
    }
}
