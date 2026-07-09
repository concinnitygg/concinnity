// src/editor/panel.rs
//
// The editor "Assets" panel: browse the world's existing assets (grouped by
// type, filtered by type) and add or edit them. Like the rest of the editor HUD
// it is plain `Sprite` / `TextLabel` / `TextInput` components at reserved ids
// (injected by `inject.rs`), driven each frame by the editor hook -- nothing here
// reaches the shipped runtime. This module owns the panel's pure geometry, its
// click resolution, and the per-frame layout that shows / positions the elements;
// the hook owns the state and the option lists.
//
// The panel opens below the top bar's capture row. Its header is a square "+"
// (add) button and a combo area. The combo area is a dropdown that, when opened,
// turns into a filter text field with an option list floating below it:
//   * Filter  -- opened by clicking the combo: the list is the asset types
//                present in the world; picking one filters the browse list.
//   * Picker  -- opened by clicking "+": the list is the addable asset types;
//                picking one opens the add form.
// The single `FILTER_INPUT` field lives in the combo area and filters whichever
// option list is open. The body below the header is one of:
//   * List    -- the world's existing assets, grouped by type (a type sub-header
//                then its indented names), scrollable. Hovering a name reveals a
//                triple-dot button opening a small Edit / Delete menu.
//   * AddForm -- a prefilled name field plus Add / Cancel (name-only for now),
//                for either a new asset or a rename of an existing one.
// The two typed fields (`FILTER_INPUT`, `NAME_INPUT`) are real `TextInput` assets
// edited by the engine's text-input system; the hook reads them back.

use crate::assets::{Sprite, TextAlign, TextInput, TextLabel};
use crate::ecs::World;
use crate::ecs::asset_id::AssetId;

use super::hud;

// The addable asset types the "+" picker offers. All are External
// (user-declarable), standalone (no required cross-references), and naturally
// multi-instance, so each recompiles cleanly when added with default args to any
// rendering world. Broadening this to every registry-addable type awaits
// per-type add validation, so the picker is deliberately curated (not silently
// capped): unlisted types are simply not offered here yet.
pub(crate) const ADD_TYPES: &[&str] = &[
    "PointLight",
    "DirectionalLight",
    "ParticleEmitter",
    "Decal",
    "ReflectionProbe",
];

// The label of the "all assets" filter option (the default), shown first in the
// combo's filter list and as the combo's text when no type filter is active.
pub(crate) const ALL_LABEL: &str = "Assets";

// Visible rows in the body before it scrolls (shared by the grouped list and the
// combo option list).
pub(crate) const MAX_ROWS: usize = 12;

// Which body the panel is showing.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum Mode {
    // The world's existing assets, grouped by type and filtered by the combo.
    List,
    // The name-first add / edit form for the chosen type.
    AddForm,
}

// The combo (header dropdown) state: closed, or open in one of two flavours that
// share the header filter field and the floating option list.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum Combo {
    // Just the browse label; the body list shows.
    Closed,
    // Filtering the browse list by type (opened from the combo button).
    Filter,
    // Picking a type to add (opened from the "+" button).
    Picker,
}

// One rendered browse-list row: a type sub-header, or an indented asset name that
// carries the index of its entry (for the Edit / Delete menu).
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct ListRow {
    pub is_header: bool,
    pub text: String,
    // The `entries` index for a name row; `None` for a header.
    pub entry: Option<usize>,
}

// Reserved asset-id families for the panel, offset past the top-bar HUD's ids
// (see `hud.rs`; the top bar uses `ID_BASE + 0..0x23`). Interned world ids never
// reach this range and these are never serialized. Offsets are ordered so a
// higher id draws later (on top): the panel background is lowest, the browse rows
// sit above it, and the floating combo / triple-dot / row menu sit above those.
const PANEL: u32 = 0x3000_0000 + 0x40;
pub(crate) const PANEL_BG: AssetId = AssetId(PANEL);
pub(crate) const PLUS_BG: AssetId = AssetId(PANEL + 1);
pub(crate) const PLUS_LABEL: AssetId = AssetId(PANEL + 2);
pub(crate) const TYPEDROP_BG: AssetId = AssetId(PANEL + 3);
pub(crate) const TYPEDROP_LABEL: AssetId = AssetId(PANEL + 4);
pub(crate) const FILTER_INPUT: AssetId = AssetId(PANEL + 5);
pub(crate) const NAME_INPUT: AssetId = AssetId(PANEL + 6);
pub(crate) const FORM_TITLE: AssetId = AssetId(PANEL + 7);
pub(crate) const FORMADD_BG: AssetId = AssetId(PANEL + 8);
pub(crate) const FORMADD_LABEL: AssetId = AssetId(PANEL + 9);
pub(crate) const FORMCANCEL_BG: AssetId = AssetId(PANEL + 10);
pub(crate) const FORMCANCEL_LABEL: AssetId = AssetId(PANEL + 11);
pub(crate) const EMPTY_LABEL: AssetId = AssetId(PANEL + 12);
pub(crate) const LIST_TRACK: AssetId = AssetId(PANEL + 0x58);
pub(crate) const LIST_THUMB: AssetId = AssetId(PANEL + 0x59);
pub(crate) const COMBO_BG: AssetId = AssetId(PANEL + 0x5A);
pub(crate) const DOT_BG: AssetId = AssetId(PANEL + 0xA0);
pub(crate) const DOT1: AssetId = AssetId(PANEL + 0xA1);
pub(crate) const DOT2: AssetId = AssetId(PANEL + 0xA2);
pub(crate) const DOT3: AssetId = AssetId(PANEL + 0xA3);
pub(crate) const MENU_BG: AssetId = AssetId(PANEL + 0xB0);
pub(crate) const MENU_EDIT_BG: AssetId = AssetId(PANEL + 0xB1);
pub(crate) const MENU_EDIT_LABEL: AssetId = AssetId(PANEL + 0xB2);
pub(crate) const MENU_DELETE_BG: AssetId = AssetId(PANEL + 0xB3);
pub(crate) const MENU_DELETE_LABEL: AssetId = AssetId(PANEL + 0xB4);

pub(crate) fn list_row_bg(i: usize) -> AssetId {
    AssetId(PANEL + 0x20 + i as u32)
}
pub(crate) fn list_row_label(i: usize) -> AssetId {
    AssetId(PANEL + 0x40 + i as u32)
}
pub(crate) fn combo_row_bg(i: usize) -> AssetId {
    AssetId(PANEL + 0x60 + i as u32)
}
pub(crate) fn combo_row_label(i: usize) -> AssetId {
    AssetId(PANEL + 0x80 + i as u32)
}

// Geometry, in window pixels. The panel is a right-aligned column below the top
// bar's capture row.
const PANEL_W: f32 = 320.0;
const HEADER_H: f32 = 40.0;
const ROW_H: f32 = 34.0;
const PAD: f32 = 8.0;
const GAP: f32 = 6.0;
const LINE: f32 = 26.0;
const SCROLLBAR_W: f32 = 5.0;
const ROW_LABEL_TOP: f32 = ROW_H * 0.5 - 10.0;
// Extra left inset for an asset name under its type sub-header.
const INDENT: f32 = 16.0;
// The triple-dot button on a hovered name row.
const DOT_SZ: f32 = 24.0;
// The floating Edit / Delete menu.
const MENU_W: f32 = 132.0;
const MENU_ROW_H: f32 = 30.0;

const PANEL_BG_TINT: [f32; 4] = [0.09, 0.09, 0.12, 0.97];
const PLUS_TINT: [f32; 4] = [0.20, 0.44, 0.30, 1.0];
const TYPEDROP_TINT: [f32; 4] = [0.18, 0.20, 0.28, 1.0];
const ROW_TINT: [f32; 4] = [0.13, 0.13, 0.16, 0.0];
const ROW_TINT_HOVER: [f32; 4] = [0.22, 0.26, 0.36, 0.98];
const HEADER_ROW_TINT: [f32; 4] = [0.0, 0.0, 0.0, 0.0];
const OPTION_TINT: [f32; 4] = [0.16, 0.16, 0.20, 1.0];
const OPTION_TINT_HOVER: [f32; 4] = [0.24, 0.28, 0.40, 1.0];
const OPTION_TINT_SELECTED: [f32; 4] = [0.16, 0.22, 0.34, 1.0];
const COMBO_BG_TINT: [f32; 4] = [0.10, 0.10, 0.13, 1.0];
const BTN_TINT: [f32; 4] = [0.22, 0.40, 0.56, 1.0];
const CANCEL_TINT: [f32; 4] = [0.30, 0.30, 0.34, 1.0];
const TRACK_TINT: [f32; 4] = [0.12, 0.12, 0.15, 0.9];
const THUMB_TINT: [f32; 4] = [0.40, 0.44, 0.56, 0.95];
const DOT_BG_TINT: [f32; 4] = [0.30, 0.34, 0.46, 0.95];
const DOT_TINT: [f32; 4] = [0.90, 0.92, 0.96, 1.0];
const MENU_BG_TINT: [f32; 4] = [0.14, 0.14, 0.18, 1.0];
const MENU_ROW_TINT: [f32; 4] = [0.16, 0.16, 0.20, 1.0];
const MENU_ROW_HOVER: [f32; 4] = [0.26, 0.30, 0.42, 1.0];
const LABEL: [f32; 3] = [0.90, 0.90, 0.92];
const LABEL_DIM: [f32; 3] = [0.60, 0.60, 0.66];
const LABEL_WHITE: [f32; 3] = [1.0, 1.0, 1.0];
const HEADER_LABEL: [f32; 3] = [0.58, 0.66, 0.80];
const DELETE_LABEL: [f32; 3] = [0.95, 0.60, 0.58];

// The panel outer rect (header + body).
pub(crate) fn panel_rect(vw: f32) -> [f32; 4] {
    [
        vw - PANEL_W,
        hud::body_top(),
        PANEL_W,
        HEADER_H + MAX_ROWS as f32 * ROW_H,
    ]
}

// The square "+" add button (panel header, left).
pub(crate) fn plus_rect(vw: f32) -> [f32; 4] {
    [vw - PANEL_W, hud::body_top(), HEADER_H, HEADER_H]
}

// The combo area (panel header, filling the rest of the row): the browse-filter
// label when closed, the filter text field when open.
pub(crate) fn combo_rect(vw: f32) -> [f32; 4] {
    [
        vw - PANEL_W + HEADER_H + GAP,
        hud::body_top(),
        PANEL_W - HEADER_H - GAP,
        HEADER_H,
    ]
}

// The filter text field, centred in the combo area.
pub(crate) fn filter_input_rect(vw: f32) -> [f32; 4] {
    let c = combo_rect(vw);
    [c[0], c[1] + (HEADER_H - ROW_H) * 0.5, c[2], ROW_H]
}

// Where the body (below the header) begins.
fn body_y() -> f32 {
    hud::body_top() + HEADER_H
}

// A body row `i` spanning the panel width (list or combo option).
pub(crate) fn list_row_rect(vw: f32, i: usize) -> [f32; 4] {
    [vw - PANEL_W, body_y() + i as f32 * ROW_H, PANEL_W, ROW_H]
}
pub(crate) fn combo_option_rect(vw: f32, i: usize) -> [f32; 4] {
    list_row_rect(vw, i)
}

// The triple-dot button rect at the right of a name row, left of the scrollbar.
fn dot_rect(row: [f32; 4]) -> [f32; 4] {
    [
        row[0] + row[2] - DOT_SZ - SCROLLBAR_W - 4.0,
        row[1] + (ROW_H - DOT_SZ) * 0.5,
        DOT_SZ,
        DOT_SZ,
    ]
}

// The Edit / Delete menu, floating just below a name row at visible index `vr`.
// Returns (background, edit row, delete row).
fn menu_rects(vw: f32, vr: usize) -> ([f32; 4], [f32; 4], [f32; 4]) {
    let x = vw - MENU_W - SCROLLBAR_W - 2.0;
    let top = body_y() + vr as f32 * ROW_H + ROW_H;
    let edit = [x, top, MENU_W, MENU_ROW_H];
    let delete = [x, top + MENU_ROW_H, MENU_W, MENU_ROW_H];
    let bg = [x, top, MENU_W, 2.0 * MENU_ROW_H];
    (bg, edit, delete)
}

// AddForm rects: a heading, the name field, then the Add / Cancel buttons.
pub(crate) fn name_input_rect(vw: f32) -> [f32; 4] {
    [
        vw - PANEL_W + PAD,
        body_y() + PAD + LINE,
        PANEL_W - 2.0 * PAD,
        ROW_H,
    ]
}
pub(crate) fn form_add_rect(vw: f32) -> [f32; 4] {
    let n = name_input_rect(vw);
    let w = (PANEL_W - 2.0 * PAD - GAP) / 2.0;
    [n[0], n[1] + ROW_H + GAP, w, ROW_H]
}
pub(crate) fn form_cancel_rect(vw: f32) -> [f32; 4] {
    let a = form_add_rect(vw);
    [a[0] + a[2] + GAP, a[1], a[2], ROW_H]
}

fn point_in(x: f32, y: f32, r: [f32; 4]) -> bool {
    x >= r[0] && x < r[0] + r[2] && y >= r[1] && y < r[1] + r[3]
}

// A resolved panel click. Option picks carry an index into the hook's current
// combo option list (the hook maps it to a filter or a type). Row-menu picks
// carry the entry index the menu belongs to.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum PanelAction {
    // The "+" button: open the type picker (from list) or close it.
    TogglePicker,
    // The combo button: open the type-filter dropdown (or close it).
    ToggleFilter,
    // Choose combo option `i` (the hook reads the open flavour to interpret it).
    PickOption(usize),
    // A name row's triple-dot: open its Edit / Delete menu (carries the entry).
    OpenRowMenu(usize),
    // The open row menu's Edit / Delete rows.
    RowEdit,
    RowDelete,
    // Confirm / cancel the add-or-edit form.
    ConfirmAdd,
    CancelForm,
    // Dismiss any open overlay (combo / row menu) without picking.
    CloseOverlays,
    // A click inside the panel that hits no control (swallowed so it does not
    // fall through to the world; a text-field click resolves to this too, so the
    // engine's text-input system takes focus).
    Consume,
}

// The per-frame data the hook hands to `apply` / `hit_test`.
pub(crate) struct PanelView<'a> {
    pub mode: Mode,
    pub combo: Combo,
    // The combo button's text ("Assets" or the active type), shown when closed.
    pub filter_label: &'a str,
    // The floating combo options (already narrowed by the typed field).
    pub combo_options: &'a [String],
    // Index of the highlighted option (the active filter), for the Filter flavour.
    pub combo_selected: Option<usize>,
    pub combo_scroll: usize,
    // The grouped browse rows (type sub-headers + indented names).
    pub list_rows: &'a [ListRow],
    pub list_scroll: usize,
    // The entry index whose Edit / Delete menu is open, if any.
    pub row_menu: Option<usize>,
    // Heading shown over the add form (e.g. "New PointLight" / "Edit lamp").
    pub form_title: &'a str,
    pub mouse: [f32; 2],
}

// The visible row (0..MAX_ROWS) currently showing entry `entry`, if any.
fn visible_row_of(view: &PanelView, entry: usize) -> Option<usize> {
    let scroll = view.list_scroll.min(view.list_rows.len().saturating_sub(1));
    for r in 0..MAX_ROWS {
        let idx = scroll + r;
        if idx >= view.list_rows.len() {
            break;
        }
        if view.list_rows[idx].entry == Some(entry) {
            return Some(r);
        }
    }
    None
}

// Resolve a click against the open panel. `None` means the click missed the
// panel entirely (the caller lets it fall through). Text-field clicks resolve to
// `Consume` (swallowed here; the engine's text-input system focuses the field
// from the same input).
pub(crate) fn hit_test(view: &PanelView, mx: f32, my: f32, vw: f32) -> Option<PanelAction> {
    if vw <= 0.0 {
        return None;
    }

    // An open row menu is modal over the panel: its rows pick, anything else
    // dismisses it.
    if let Some(entry) = view.row_menu {
        if let Some(vr) = visible_row_of(view, entry) {
            let (_, edit, delete) = menu_rects(vw, vr);
            if point_in(mx, my, edit) {
                return Some(PanelAction::RowEdit);
            }
            if point_in(mx, my, delete) {
                return Some(PanelAction::RowDelete);
            }
        }
        return Some(PanelAction::CloseOverlays);
    }

    // An open combo captures the header field + its floating options.
    if view.combo != Combo::Closed {
        if point_in(mx, my, plus_rect(vw)) {
            return Some(PanelAction::TogglePicker);
        }
        if point_in(mx, my, combo_rect(vw)) {
            return Some(PanelAction::Consume);
        }
        let scroll = view
            .combo_scroll
            .min(view.combo_options.len().saturating_sub(1));
        for r in 0..MAX_ROWS {
            let idx = scroll + r;
            if idx >= view.combo_options.len() {
                break;
            }
            if point_in(mx, my, combo_option_rect(vw, r)) {
                return Some(PanelAction::PickOption(idx));
            }
        }
        return Some(PanelAction::CloseOverlays);
    }

    // Combo closed: clicks outside the panel fall through (the top bar handles
    // the region above `body_top`; the world gets the rest).
    if !point_in(mx, my, panel_rect(vw)) {
        return None;
    }
    if point_in(mx, my, plus_rect(vw)) {
        return Some(PanelAction::TogglePicker);
    }
    if point_in(mx, my, combo_rect(vw)) {
        return Some(PanelAction::ToggleFilter);
    }

    match view.mode {
        Mode::List => {
            let scroll = view.list_scroll.min(view.list_rows.len().saturating_sub(1));
            for r in 0..MAX_ROWS {
                let idx = scroll + r;
                if idx >= view.list_rows.len() {
                    break;
                }
                let Some(entry) = view.list_rows[idx].entry else {
                    continue;
                };
                let rect = list_row_rect(vw, r);
                if point_in(mx, my, rect) {
                    if point_in(mx, my, dot_rect(rect)) {
                        return Some(PanelAction::OpenRowMenu(entry));
                    }
                    return Some(PanelAction::Consume);
                }
            }
            Some(PanelAction::Consume)
        }
        Mode::AddForm => {
            if point_in(mx, my, form_add_rect(vw)) {
                Some(PanelAction::ConfirmAdd)
            } else if point_in(mx, my, form_cancel_rect(vw)) {
                Some(PanelAction::CancelForm)
            } else {
                // The name field, or empty form space: swallow (the field focuses
                // itself from the same click).
                Some(PanelAction::Consume)
            }
        }
    }
}

// Position + show the panel's elements for this frame, or hide them all when the
// panel is closed (`view` is `None`).
pub(crate) fn apply(world: &mut World, view: Option<&PanelView>, vw: f32) {
    let Some(view) = view else {
        hide_all(world);
        return;
    };
    if vw <= 0.0 {
        return;
    }

    // Blank everything, then re-show what this frame needs.
    hide_all(world);

    place_sprite(world, PANEL_BG, panel_rect(vw), PANEL_BG_TINT, true);
    place_sprite(world, PLUS_BG, plus_rect(vw), PLUS_TINT, true);
    place_center_label(world, PLUS_LABEL, plus_rect(vw), "+", LABEL_WHITE, true);

    // The combo area: the browse label when closed, the filter field when open.
    if view.combo == Combo::Closed {
        place_sprite(world, TYPEDROP_BG, combo_rect(vw), TYPEDROP_TINT, true);
        let td = combo_rect(vw);
        place_left_label(
            world,
            TYPEDROP_LABEL,
            [td[0] + PAD, td[1] + HEADER_H * 0.5 - 10.0],
            view.filter_label,
            LABEL,
            true,
        );
    } else {
        show_field(world, FILTER_INPUT, filter_input_rect(vw));
    }

    // The body.
    match (view.mode, view.combo) {
        (Mode::AddForm, _) => layout_form(world, view, vw),
        (Mode::List, Combo::Closed) => layout_list(world, view, vw),
        (Mode::List, _) => layout_combo(world, view, vw),
    }
}

fn layout_list(world: &mut World, view: &PanelView, vw: f32) {
    if view.list_rows.is_empty() {
        place_left_label(
            world,
            EMPTY_LABEL,
            [vw - PANEL_W + PAD, body_y() + PAD],
            "No matching assets",
            LABEL_DIM,
            true,
        );
        return;
    }
    let total = view.list_rows.len();
    let scroll = view.list_scroll.min(total.saturating_sub(1));
    let mut hovered_row = None;
    let mut menu_row = None;
    for r in 0..MAX_ROWS {
        let idx = scroll + r;
        if idx >= total {
            break;
        }
        let row = &view.list_rows[idx];
        let rect = list_row_rect(vw, r);
        if row.is_header {
            place_sprite(world, list_row_bg(r), rect, HEADER_ROW_TINT, true);
            set_row_label(
                world,
                list_row_label(r),
                [rect[0] + PAD, rect[1] + ROW_LABEL_TOP],
                &row.text,
                HEADER_LABEL,
                true,
            );
            continue;
        }
        let hovered = point_in(view.mouse[0], view.mouse[1], rect);
        if hovered {
            hovered_row = Some(r);
        }
        if row.entry.is_some() && view.row_menu == row.entry {
            menu_row = Some(r);
        }
        let tint = if hovered { ROW_TINT_HOVER } else { ROW_TINT };
        place_sprite(world, list_row_bg(r), rect, tint, true);
        set_row_label(
            world,
            list_row_label(r),
            [rect[0] + PAD + INDENT, rect[1] + ROW_LABEL_TOP],
            &row.text,
            LABEL,
            true,
        );
    }
    // The triple-dot follows the row whose menu is open, else the hovered row.
    if let Some(r) = menu_row.or(hovered_row) {
        place_dot(world, list_row_rect(vw, r));
    }
    if let Some(r) = menu_row {
        layout_row_menu(world, view, vw, r);
    }
    layout_scrollbar(world, total, scroll, vw);
}

fn layout_combo(world: &mut World, view: &PanelView, vw: f32) {
    let total = view.combo_options.len();
    let scroll = view.combo_scroll.min(total.saturating_sub(1));
    let shown = total.saturating_sub(scroll).clamp(1, MAX_ROWS);
    let backing = [vw - PANEL_W, body_y(), PANEL_W, shown as f32 * ROW_H + PAD];
    place_sprite(world, COMBO_BG, backing, COMBO_BG_TINT, true);
    if total == 0 {
        place_left_label(
            world,
            EMPTY_LABEL,
            [vw - PANEL_W + PAD, body_y() + PAD],
            "No matching types",
            LABEL_DIM,
            true,
        );
        return;
    }
    for r in 0..MAX_ROWS {
        let idx = scroll + r;
        if idx >= total {
            break;
        }
        let rect = combo_option_rect(vw, r);
        let hovered = point_in(view.mouse[0], view.mouse[1], rect);
        let tint = if hovered {
            OPTION_TINT_HOVER
        } else if view.combo_selected == Some(idx) {
            OPTION_TINT_SELECTED
        } else {
            OPTION_TINT
        };
        place_sprite(world, combo_row_bg(r), rect, tint, true);
        set_row_label(
            world,
            combo_row_label(r),
            [rect[0] + PAD, rect[1] + ROW_LABEL_TOP],
            &view.combo_options[idx],
            LABEL,
            true,
        );
    }
    layout_scrollbar(world, total, scroll, vw);
}

// The three stacked dots of the triple-dot button, on a subtle hover square.
fn place_dot(world: &mut World, row: [f32; 4]) {
    let d = dot_rect(row);
    place_sprite(world, DOT_BG, d, DOT_BG_TINT, true);
    let cx = d[0] + d[2] * 0.5;
    let cy = d[1] + d[3] * 0.5;
    let s = 3.5;
    let gap = 3.5;
    for (id, dy) in [(DOT1, -gap - s), (DOT2, -s * 0.5), (DOT3, gap)] {
        place_sprite(world, id, [cx - s * 0.5, cy + dy, s, s], DOT_TINT, true);
    }
}

fn layout_row_menu(world: &mut World, view: &PanelView, vw: f32, vr: usize) {
    let (bg, edit, delete) = menu_rects(vw, vr);
    place_sprite(world, MENU_BG, bg, MENU_BG_TINT, true);
    let edit_hover = point_in(view.mouse[0], view.mouse[1], edit);
    place_sprite(
        world,
        MENU_EDIT_BG,
        edit,
        if edit_hover {
            MENU_ROW_HOVER
        } else {
            MENU_ROW_TINT
        },
        true,
    );
    place_left_label(
        world,
        MENU_EDIT_LABEL,
        [edit[0] + PAD, edit[1] + MENU_ROW_H * 0.5 - 10.0],
        "Edit",
        LABEL,
        true,
    );
    let del_hover = point_in(view.mouse[0], view.mouse[1], delete);
    place_sprite(
        world,
        MENU_DELETE_BG,
        delete,
        if del_hover {
            MENU_ROW_HOVER
        } else {
            MENU_ROW_TINT
        },
        true,
    );
    place_left_label(
        world,
        MENU_DELETE_LABEL,
        [delete[0] + PAD, delete[1] + MENU_ROW_H * 0.5 - 10.0],
        "Delete",
        DELETE_LABEL,
        true,
    );
}

fn layout_form(world: &mut World, view: &PanelView, vw: f32) {
    place_left_label(
        world,
        FORM_TITLE,
        [vw - PANEL_W + PAD, body_y() + PAD],
        view.form_title,
        LABEL_WHITE,
        true,
    );
    show_field(world, NAME_INPUT, name_input_rect(vw));

    let add = form_add_rect(vw);
    let cancel = form_cancel_rect(vw);
    let add_hover = point_in(view.mouse[0], view.mouse[1], add);
    place_sprite(
        world,
        FORMADD_BG,
        add,
        if add_hover {
            OPTION_TINT_HOVER
        } else {
            BTN_TINT
        },
        true,
    );
    place_center_label(world, FORMADD_LABEL, add, "Add", LABEL_WHITE, true);
    place_sprite(world, FORMCANCEL_BG, cancel, CANCEL_TINT, true);
    place_center_label(world, FORMCANCEL_LABEL, cancel, "Cancel", LABEL, true);
}

// A simple non-interactive scrollbar thumb sizing the visible window against the
// total, shown only when the body overflows.
fn layout_scrollbar(world: &mut World, total: usize, scroll: usize, vw: f32) {
    if total <= MAX_ROWS {
        return;
    }
    let track_h = MAX_ROWS as f32 * ROW_H;
    let track = [vw - SCROLLBAR_W, body_y(), SCROLLBAR_W, track_h];
    place_sprite(world, LIST_TRACK, track, TRACK_TINT, true);
    let frac_visible = MAX_ROWS as f32 / total as f32;
    let thumb_h = (track_h * frac_visible).max(20.0);
    let max_scroll = (total - MAX_ROWS) as f32;
    let t = if max_scroll > 0.0 {
        scroll as f32 / max_scroll
    } else {
        0.0
    };
    let thumb_y = body_y() + t * (track_h - thumb_h);
    place_sprite(
        world,
        LIST_THUMB,
        [vw - SCROLLBAR_W, thumb_y, SCROLLBAR_W, thumb_h],
        THUMB_TINT,
        true,
    );
}

// Every panel sprite id, so the closed / hidden pass can blank the whole panel
// (and `inject.rs` can create exactly this set).
pub(crate) fn all_sprite_ids() -> Vec<AssetId> {
    let mut ids = vec![
        PANEL_BG,
        PLUS_BG,
        TYPEDROP_BG,
        FORMADD_BG,
        FORMCANCEL_BG,
        LIST_TRACK,
        LIST_THUMB,
        COMBO_BG,
        DOT_BG,
        DOT1,
        DOT2,
        DOT3,
        MENU_BG,
        MENU_EDIT_BG,
        MENU_DELETE_BG,
    ];
    ids.extend((0..MAX_ROWS).map(list_row_bg));
    ids.extend((0..MAX_ROWS).map(combo_row_bg));
    ids
}
pub(crate) fn all_label_ids() -> Vec<AssetId> {
    let mut ids = vec![
        PLUS_LABEL,
        TYPEDROP_LABEL,
        FORMADD_LABEL,
        FORMCANCEL_LABEL,
        FORM_TITLE,
        EMPTY_LABEL,
        MENU_EDIT_LABEL,
        MENU_DELETE_LABEL,
    ];
    ids.extend((0..MAX_ROWS).map(list_row_label));
    ids.extend((0..MAX_ROWS).map(combo_row_label));
    ids
}

// Hide every panel element, including the two typed fields (and blur them so a
// hidden field cannot keep keyboard focus).
pub(crate) fn hide_all(world: &mut World) {
    for id in all_sprite_ids() {
        set_sprite_visible(world, id, false);
    }
    for id in all_label_ids() {
        set_label_visible(world, id, false);
    }
    hide_field(world, FILTER_INPUT);
    hide_field(world, NAME_INPUT);
}

// -- Element mutation helpers -------------------------------------------------

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

fn place_center_label(
    world: &mut World,
    id: AssetId,
    rect: [f32; 4],
    content: &str,
    color: [f32; 3],
    visible: bool,
) {
    for l in world.query_mut::<TextLabel>() {
        if l.asset_id == id {
            l.x = rect[0] + rect[2] * 0.5;
            l.y = rect[1] + rect[3] * 0.5 - 10.0;
            l.align = TextAlign::Center;
            l.color = color;
            l.visible = visible;
            l.content = content.to_string();
            break;
        }
    }
}

fn place_left_label(
    world: &mut World,
    id: AssetId,
    pos: [f32; 2],
    content: &str,
    color: [f32; 3],
    visible: bool,
) {
    for l in world.query_mut::<TextLabel>() {
        if l.asset_id == id {
            l.x = pos[0];
            l.y = pos[1];
            l.align = TextAlign::Left;
            l.color = color;
            l.visible = visible;
            l.content = content.to_string();
            break;
        }
    }
}

fn set_row_label(
    world: &mut World,
    id: AssetId,
    pos: [f32; 2],
    content: &str,
    color: [f32; 3],
    visible: bool,
) {
    place_left_label(world, id, pos, content, color, visible);
}

fn set_sprite_visible(world: &mut World, id: AssetId, visible: bool) {
    for s in world.query_mut::<Sprite>() {
        if s.asset_id == id {
            s.visible = visible;
            break;
        }
    }
}
fn set_label_visible(world: &mut World, id: AssetId, visible: bool) {
    for l in world.query_mut::<TextLabel>() {
        if l.asset_id == id {
            l.visible = visible;
            break;
        }
    }
}

// Position, show, and keep focus on the mode's single active field. Focus is
// re-asserted every frame it is shown: the same click that opened the mode (on
// the "+" button, say) lands outside the field, so the engine's text-input
// system would otherwise blur it that frame; re-asserting keeps it typable. The
// content is not touched here (only on the transition), so what is typed stands.
fn show_field(world: &mut World, id: AssetId, rect: [f32; 4]) {
    for t in world.query_mut::<TextInput>() {
        if t.asset_id == id {
            t.x = rect[0];
            t.y = rect[1];
            t.width = rect[2];
            t.height = rect[3];
            t.visible = true;
            t.focused = true;
            break;
        }
    }
}

// Hide + blur a typed field.
fn hide_field(world: &mut World, id: AssetId) {
    for t in world.query_mut::<TextInput>() {
        if t.asset_id == id {
            t.visible = false;
            t.focused = false;
            break;
        }
    }
}

// Set a field's text + caret and give it focus (a mode transition; the hook
// calls this so the field is ready to type into immediately).
pub(crate) fn focus_field_with(world: &mut World, id: AssetId, content: &str) {
    for t in world.query_mut::<TextInput>() {
        if t.asset_id == id {
            t.content = content.to_string();
            t.caret = content.chars().count();
            t.focused = true;
            t.visible = true;
            break;
        }
    }
}

// Read a field's current text.
pub(crate) fn field_text(world: &World, id: AssetId) -> String {
    world
        .query::<TextInput>()
        .find(|t| t.asset_id == id)
        .map(|t| t.content.clone())
        .unwrap_or_default()
}

// Whether the cursor is over the scrollable body area (for wheel scrolling).
pub(crate) fn cursor_over_body(mx: f32, my: f32, vw: f32) -> bool {
    let p = panel_rect(vw);
    mx >= p[0] && mx < p[0] + p[2] && my >= body_y() && my < p[1] + p[3]
}

#[cfg(test)]
mod tests {
    use super::*;

    fn rows(items: &[(bool, &str, Option<usize>)]) -> Vec<ListRow> {
        items
            .iter()
            .map(|(h, t, e)| ListRow {
                is_header: *h,
                text: t.to_string(),
                entry: *e,
            })
            .collect()
    }

    struct Fixture {
        combo_options: Vec<String>,
        list_rows: Vec<ListRow>,
    }

    fn view<'a>(
        fx: &'a Fixture,
        mode: Mode,
        combo: Combo,
        row_menu: Option<usize>,
        mouse: [f32; 2],
    ) -> PanelView<'a> {
        PanelView {
            mode,
            combo,
            filter_label: ALL_LABEL,
            combo_options: &fx.combo_options,
            combo_selected: None,
            combo_scroll: 0,
            list_rows: &fx.list_rows,
            list_scroll: 0,
            row_menu,
            form_title: "New PointLight",
            mouse,
        }
    }

    #[test]
    fn panel_sits_below_the_top_bar_and_is_right_aligned() {
        let p = panel_rect(1280.0);
        assert_eq!(p[0] + p[2], 1280.0, "flush to the window right");
        assert_eq!(p[1], hud::body_top(), "starts below the capture row");
    }

    #[test]
    fn header_plus_and_combo_do_not_overlap() {
        let plus = plus_rect(1280.0);
        let combo = combo_rect(1280.0);
        assert!(plus[0] + plus[2] <= combo[0], "+ is left of the combo");
        assert_eq!(combo[0] + combo[2], 1280.0, "combo reaches the panel right");
    }

    #[test]
    fn plus_toggles_the_picker() {
        let fx = Fixture {
            combo_options: vec![],
            list_rows: vec![],
        };
        let v = view(&fx, Mode::List, Combo::Closed, None, [0.0, 0.0]);
        let plus = plus_rect(1280.0);
        assert_eq!(
            hit_test(&v, plus[0] + 5.0, plus[1] + 5.0, 1280.0),
            Some(PanelAction::TogglePicker)
        );
    }

    #[test]
    fn combo_button_opens_and_picks_an_option() {
        let fx = Fixture {
            combo_options: vec![ALL_LABEL.to_string(), "PointLight".to_string()],
            list_rows: vec![],
        };
        // Closed: clicking the combo opens the filter dropdown.
        let v = view(&fx, Mode::List, Combo::Closed, None, [0.0, 0.0]);
        let c = combo_rect(1280.0);
        assert_eq!(
            hit_test(&v, c[0] + 5.0, c[1] + 5.0, 1280.0),
            Some(PanelAction::ToggleFilter)
        );
        // Open: clicking option row 1 picks it.
        let vo = view(&fx, Mode::List, Combo::Filter, None, [0.0, 0.0]);
        let r1 = combo_option_rect(1280.0, 1);
        assert_eq!(
            hit_test(&vo, r1[0] + 5.0, r1[1] + 5.0, 1280.0),
            Some(PanelAction::PickOption(1))
        );
        // Open: clicking the header field keeps focus (consumed).
        assert_eq!(
            hit_test(&vo, c[0] + 5.0, c[1] + 5.0, 1280.0),
            Some(PanelAction::Consume)
        );
        // Open: a click on empty body space closes it.
        assert_eq!(
            hit_test(&vo, 640.0, 700.0, 1280.0),
            Some(PanelAction::CloseOverlays)
        );
    }

    #[test]
    fn picker_option_maps_to_a_scrolled_index() {
        let opts: Vec<String> = ADD_TYPES.iter().map(|s| s.to_string()).collect();
        let fx = Fixture {
            combo_options: opts,
            list_rows: vec![],
        };
        let mut v = view(&fx, Mode::List, Combo::Picker, None, [0.0, 0.0]);
        v.combo_scroll = 2;
        let r0 = combo_option_rect(1280.0, 0);
        assert_eq!(
            hit_test(&v, r0[0] + 5.0, r0[1] + 5.0, 1280.0),
            Some(PanelAction::PickOption(2)),
            "row 0 maps to option `scroll + 0`"
        );
    }

    #[test]
    fn hovering_a_name_row_dot_opens_its_menu() {
        let fx = Fixture {
            combo_options: vec![],
            list_rows: rows(&[(true, "PointLight", None), (false, "lamp", Some(7))]),
        };
        let v = view(&fx, Mode::List, Combo::Closed, None, [0.0, 0.0]);
        // Row 1 (the name) triple-dot.
        let dot = dot_rect(list_row_rect(1280.0, 1));
        assert_eq!(
            hit_test(&v, dot[0] + 5.0, dot[1] + 5.0, 1280.0),
            Some(PanelAction::OpenRowMenu(7))
        );
        // The header row (row 0) is not interactive: a click consumes.
        let hdr = list_row_rect(1280.0, 0);
        assert_eq!(
            hit_test(&v, hdr[0] + 5.0, hdr[1] + 5.0, 1280.0),
            Some(PanelAction::Consume)
        );
    }

    #[test]
    fn open_row_menu_resolves_edit_and_delete() {
        let fx = Fixture {
            combo_options: vec![],
            list_rows: rows(&[(true, "PointLight", None), (false, "lamp", Some(3))]),
        };
        let v = view(&fx, Mode::List, Combo::Closed, Some(3), [0.0, 0.0]);
        let (_, edit, delete) = menu_rects(1280.0, 1);
        assert_eq!(
            hit_test(&v, edit[0] + 5.0, edit[1] + 5.0, 1280.0),
            Some(PanelAction::RowEdit)
        );
        assert_eq!(
            hit_test(&v, delete[0] + 5.0, delete[1] + 5.0, 1280.0),
            Some(PanelAction::RowDelete)
        );
        // A click off the menu dismisses it.
        assert_eq!(
            hit_test(&v, 640.0, 700.0, 1280.0),
            Some(PanelAction::CloseOverlays)
        );
    }

    #[test]
    fn form_buttons_resolve() {
        let fx = Fixture {
            combo_options: vec![],
            list_rows: vec![],
        };
        let v = view(&fx, Mode::AddForm, Combo::Closed, None, [0.0, 0.0]);
        let add = form_add_rect(1280.0);
        let cancel = form_cancel_rect(1280.0);
        assert_eq!(
            hit_test(&v, add[0] + 5.0, add[1] + 5.0, 1280.0),
            Some(PanelAction::ConfirmAdd)
        );
        assert_eq!(
            hit_test(&v, cancel[0] + 5.0, cancel[1] + 5.0, 1280.0),
            Some(PanelAction::CancelForm)
        );
    }

    #[test]
    fn clicks_outside_the_panel_fall_through_when_closed() {
        let fx = Fixture {
            combo_options: vec![],
            list_rows: vec![],
        };
        let v = view(&fx, Mode::List, Combo::Closed, None, [0.0, 0.0]);
        assert_eq!(hit_test(&v, 10.0, 400.0, 1280.0), None);
    }

    #[test]
    fn name_field_click_is_consumed_for_the_text_system() {
        let fx = Fixture {
            combo_options: vec![],
            list_rows: vec![],
        };
        let v = view(&fx, Mode::AddForm, Combo::Closed, None, [0.0, 0.0]);
        let f = name_input_rect(1280.0);
        assert_eq!(
            hit_test(&v, f[0] + 5.0, f[1] + 5.0, 1280.0),
            Some(PanelAction::Consume)
        );
    }
}
