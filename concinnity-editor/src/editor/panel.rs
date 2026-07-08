// src/editor/panel.rs
//
// The editor "Assets" panel: browse the world's existing assets (filtered by
// type) and add new ones. Like the rest of the editor HUD it is plain
// `Sprite` / `TextLabel` / `TextInput` components at reserved ids (injected by
// `inject.rs`), driven each frame by the editor hook -- nothing here reaches the
// shipped runtime. This module owns the panel's pure geometry, its click
// resolution, and the per-frame layout that shows / positions the elements; the
// hook owns the state and the option lists.
//
// The panel opens below the top bar's capture row. Its header is a square "+"
// (add) button and a type-filter dropdown; its body is one of three modes:
//   * List      -- the existing entries matching the type filter, scrollable.
//   * TypePicker -- a typed filter field over an autocomplete list of the
//                   addable types (the "+" flow).
//   * AddForm   -- a prefilled name field plus Add / Cancel (name-only for now).
// The two typed fields (`FILTER_INPUT`, `NAME_INPUT`) are real `TextInput`
// assets edited by the engine's text-input system; the hook reads them back.

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
// type-filter dropdown and as its button text when no type filter is active.
pub(crate) const ALL_LABEL: &str = "Assets";

// Visible rows in the body before it scrolls. The picker reserves its top row
// for the filter field, so it shows one fewer.
pub(crate) const MAX_ROWS: usize = 12;
pub(crate) const PICKER_ROWS: usize = MAX_ROWS - 1;

// Which body the panel is showing.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum Mode {
    // The world's existing assets, filtered by the type dropdown.
    List,
    // A typed filter over the addable-type autocomplete (the "+" flow).
    TypePicker,
    // The name-first add form for the picked type.
    AddForm,
}

// Reserved asset-id families for the panel, offset past the top-bar HUD's ids
// (see `hud.rs`; the top bar uses `ID_BASE + 0..0x23`). Interned world ids never
// reach this range and these are never serialized.
const PANEL: u32 = 0x3000_0000 + 0x40;
pub(crate) const PANEL_BG: AssetId = AssetId(PANEL);
pub(crate) const PLUS_BG: AssetId = AssetId(PANEL + 1);
pub(crate) const PLUS_LABEL: AssetId = AssetId(PANEL + 2);
pub(crate) const TYPEDROP_BG: AssetId = AssetId(PANEL + 3);
pub(crate) const TYPEDROP_LABEL: AssetId = AssetId(PANEL + 4);
pub(crate) const FILTER_INPUT: AssetId = AssetId(PANEL + 5);
pub(crate) const NAME_INPUT: AssetId = AssetId(PANEL + 6);
pub(crate) const FORMADD_BG: AssetId = AssetId(PANEL + 7);
pub(crate) const FORMADD_LABEL: AssetId = AssetId(PANEL + 8);
pub(crate) const FORMCANCEL_BG: AssetId = AssetId(PANEL + 9);
pub(crate) const FORMCANCEL_LABEL: AssetId = AssetId(PANEL + 10);
pub(crate) const FORM_TITLE: AssetId = AssetId(PANEL + 11);
pub(crate) const LIST_TRACK: AssetId = AssetId(PANEL + 12);
pub(crate) const LIST_THUMB: AssetId = AssetId(PANEL + 13);
pub(crate) const EMPTY_LABEL: AssetId = AssetId(PANEL + 14);

pub(crate) fn list_row_bg(i: usize) -> AssetId {
    AssetId(PANEL + 0x20 + i as u32)
}
pub(crate) fn list_row_label(i: usize) -> AssetId {
    AssetId(PANEL + 0x40 + i as u32)
}
pub(crate) fn filter_row_bg(i: usize) -> AssetId {
    AssetId(PANEL + 0x60 + i as u32)
}
pub(crate) fn filter_row_label(i: usize) -> AssetId {
    AssetId(PANEL + 0x80 + i as u32)
}
pub(crate) fn picker_row_bg(i: usize) -> AssetId {
    AssetId(PANEL + 0xA0 + i as u32)
}
pub(crate) fn picker_row_label(i: usize) -> AssetId {
    AssetId(PANEL + 0xC0 + i as u32)
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

const PANEL_BG_TINT: [f32; 4] = [0.09, 0.09, 0.12, 0.97];
const PLUS_TINT: [f32; 4] = [0.20, 0.44, 0.30, 1.0];
const TYPEDROP_TINT: [f32; 4] = [0.18, 0.20, 0.28, 1.0];
const ROW_TINT: [f32; 4] = [0.13, 0.13, 0.16, 0.0];
const ROW_TINT_HOVER: [f32; 4] = [0.22, 0.26, 0.36, 0.98];
const OPTION_TINT: [f32; 4] = [0.14, 0.14, 0.18, 0.99];
const OPTION_TINT_HOVER: [f32; 4] = [0.24, 0.28, 0.40, 1.0];
const OPTION_TINT_SELECTED: [f32; 4] = [0.16, 0.22, 0.34, 1.0];
const BTN_TINT: [f32; 4] = [0.22, 0.40, 0.56, 1.0];
const CANCEL_TINT: [f32; 4] = [0.30, 0.30, 0.34, 1.0];
const TRACK_TINT: [f32; 4] = [0.12, 0.12, 0.15, 0.9];
const THUMB_TINT: [f32; 4] = [0.40, 0.44, 0.56, 0.95];
const LABEL: [f32; 3] = [0.90, 0.90, 0.92];
const LABEL_DIM: [f32; 3] = [0.60, 0.60, 0.66];
const LABEL_WHITE: [f32; 3] = [1.0, 1.0, 1.0];

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

// The type-filter dropdown button (panel header, filling the rest of the row).
pub(crate) fn typedrop_rect(vw: f32) -> [f32; 4] {
    [
        vw - PANEL_W + HEADER_H + GAP,
        hud::body_top(),
        PANEL_W - HEADER_H - GAP,
        HEADER_H,
    ]
}

// Where the body (below the header) begins.
fn body_y() -> f32 {
    hud::body_top() + HEADER_H
}

// A body row `i` spanning the panel width (List mode).
pub(crate) fn list_row_rect(vw: f32, i: usize) -> [f32; 4] {
    [vw - PANEL_W, body_y() + i as f32 * ROW_H, PANEL_W, ROW_H]
}

// A type-filter dropdown option `i`, floating below the dropdown button over the
// body area.
pub(crate) fn filter_option_rect(vw: f32, i: usize) -> [f32; 4] {
    let td = typedrop_rect(vw);
    [td[0], body_y() + i as f32 * ROW_H, td[2], ROW_H]
}

// The typed filter field at the top of the TypePicker body.
pub(crate) fn filter_input_rect(vw: f32) -> [f32; 4] {
    [
        vw - PANEL_W + PAD,
        body_y() + PAD,
        PANEL_W - 2.0 * PAD,
        ROW_H,
    ]
}

// A picker autocomplete row `i`, below the filter field.
pub(crate) fn picker_row_rect(vw: f32, i: usize) -> [f32; 4] {
    [
        vw - PANEL_W,
        body_y() + ROW_H + PAD + i as f32 * ROW_H,
        PANEL_W,
        ROW_H,
    ]
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

// A resolved panel click. Row picks carry an index into the hook's current
// option list (the hook maps it to the concrete type / filter).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum PanelAction {
    // Toggle the "+" flow: open the type picker (from List) or return to the
    // list (from the picker / form).
    TogglePicker,
    // Open / close the type-filter dropdown.
    ToggleTypeDropdown,
    // Choose type-filter option `i` (0 is the "all" option).
    PickFilter(usize),
    // Choose addable-type option `i` in the picker -> AddForm.
    PickType(usize),
    // Confirm / cancel the add form.
    ConfirmAdd,
    CancelForm,
    // A click inside the panel that hits no control (swallowed so it does not
    // fall through to the world; a text-field click resolves to this too, so the
    // engine's text-input system takes focus).
    Consume,
}

// The per-frame data the hook hands to `apply` / `hit_test`.
pub(crate) struct PanelView<'a> {
    pub mode: Mode,
    pub type_dropdown_open: bool,
    // The filter button's text ("Assets" or the active type).
    pub filter_label: &'a str,
    // The filter dropdown options (index 0 is `ALL_LABEL`).
    pub filter_options: &'a [String],
    // Index of the active filter option, for highlighting.
    pub filter_selected: usize,
    // Existing entries matching the filter, as (name, type).
    pub list_items: &'a [(String, String)],
    pub list_scroll: usize,
    // The addable-type options (already filtered by the typed field).
    pub picker_options: &'a [String],
    pub picker_scroll: usize,
    // Heading shown over the add form (e.g. "New PointLight").
    pub form_title: &'a str,
    pub mouse: [f32; 2],
}

// Resolve a click against the open panel. `None` means the click missed the
// panel entirely (the caller lets it fall through). Text-field clicks resolve to
// `Consume` (swallowed here; the engine's text-input system focuses the field
// from the same input).
pub(crate) fn hit_test(view: &PanelView, mx: f32, my: f32, vw: f32) -> Option<PanelAction> {
    if vw <= 0.0 {
        return None;
    }

    // While the type-filter dropdown is open it captures the body: an option row
    // picks it, anything else closes it.
    if view.type_dropdown_open {
        if point_in(mx, my, typedrop_rect(vw)) {
            return Some(PanelAction::ToggleTypeDropdown);
        }
        for i in 0..view.filter_options.len() {
            if point_in(mx, my, filter_option_rect(vw, i)) {
                return Some(PanelAction::PickFilter(i));
            }
        }
        return Some(PanelAction::ToggleTypeDropdown);
    }

    // Clicks outside the panel fall through (the top bar handles the region above
    // `body_top`; the world gets the rest).
    if !point_in(mx, my, panel_rect(vw)) {
        return None;
    }

    // Header controls (all modes).
    if point_in(mx, my, plus_rect(vw)) {
        return Some(PanelAction::TogglePicker);
    }
    if point_in(mx, my, typedrop_rect(vw)) {
        return Some(PanelAction::ToggleTypeDropdown);
    }

    match view.mode {
        Mode::List => Some(PanelAction::Consume),
        Mode::TypePicker => {
            if point_in(mx, my, filter_input_rect(vw)) {
                return Some(PanelAction::Consume);
            }
            for i in 0..view.picker_options.len().min(PICKER_ROWS) {
                if point_in(mx, my, picker_row_rect(vw, i)) {
                    return Some(PanelAction::PickType(view.picker_scroll + i));
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

    place_sprite(world, PANEL_BG, panel_rect(vw), PANEL_BG_TINT, true);
    place_sprite(world, PLUS_BG, plus_rect(vw), PLUS_TINT, true);
    place_center_label(world, PLUS_LABEL, plus_rect(vw), "+", LABEL_WHITE, true);
    place_sprite(world, TYPEDROP_BG, typedrop_rect(vw), TYPEDROP_TINT, true);
    let td = typedrop_rect(vw);
    place_left_label(
        world,
        TYPEDROP_LABEL,
        [td[0] + PAD, td[1] + HEADER_H * 0.5 - 10.0],
        view.filter_label,
        LABEL,
        true,
    );

    // Reset every body family + field to hidden; the active mode re-shows its own.
    hide_body(world);

    match view.mode {
        Mode::List => layout_list(world, view, vw),
        Mode::TypePicker => layout_picker(world, view, vw),
        Mode::AddForm => layout_form(world, view, vw),
    }

    // The type-filter dropdown floats over whatever body is shown.
    if view.type_dropdown_open {
        layout_filter_dropdown(world, view, vw);
    }
}

fn layout_list(world: &mut World, view: &PanelView, vw: f32) {
    if view.list_items.is_empty() {
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
    let total = view.list_items.len();
    let scroll = view.list_scroll.min(total.saturating_sub(1));
    for row in 0..MAX_ROWS {
        let idx = scroll + row;
        if idx >= total {
            break;
        }
        let rect = list_row_rect(vw, row);
        let hovered = point_in(view.mouse[0], view.mouse[1], rect);
        let tint = if hovered { ROW_TINT_HOVER } else { ROW_TINT };
        place_sprite(world, list_row_bg(row), rect, tint, true);
        let (name, ty) = &view.list_items[idx];
        set_row_label(
            world,
            list_row_label(row),
            [rect[0] + PAD, rect[1] + ROW_LABEL_TOP],
            &format!("{name}   ({ty})"),
            LABEL,
            true,
        );
    }
    layout_scrollbar(world, total, scroll, vw);
}

fn layout_picker(world: &mut World, view: &PanelView, vw: f32) {
    show_field(world, FILTER_INPUT, filter_input_rect(vw));
    let total = view.picker_options.len();
    let scroll = view.picker_scroll.min(total.saturating_sub(1));
    for row in 0..PICKER_ROWS {
        let idx = scroll + row;
        if idx >= total {
            break;
        }
        let rect = picker_row_rect(vw, row);
        let hovered = point_in(view.mouse[0], view.mouse[1], rect);
        let tint = if hovered {
            OPTION_TINT_HOVER
        } else {
            OPTION_TINT
        };
        place_sprite(world, picker_row_bg(row), rect, tint, true);
        set_row_label(
            world,
            picker_row_label(row),
            [rect[0] + PAD, rect[1] + ROW_LABEL_TOP],
            &view.picker_options[idx],
            LABEL,
            true,
        );
    }
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

fn layout_filter_dropdown(world: &mut World, view: &PanelView, vw: f32) {
    for i in 0..view.filter_options.len().min(MAX_ROWS) {
        let rect = filter_option_rect(vw, i);
        let hovered = point_in(view.mouse[0], view.mouse[1], rect);
        let tint = if hovered {
            OPTION_TINT_HOVER
        } else if i == view.filter_selected {
            OPTION_TINT_SELECTED
        } else {
            OPTION_TINT
        };
        place_sprite(world, filter_row_bg(i), rect, tint, true);
        set_row_label(
            world,
            filter_row_label(i),
            [rect[0] + PAD, rect[1] + ROW_LABEL_TOP],
            &view.filter_options[i],
            LABEL,
            true,
        );
    }
}

// A simple non-interactive scrollbar thumb sizing the visible window against the
// total, shown only when the list overflows.
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

// Every panel sprite id, so the closed / hidden pass can blank the whole panel.
fn panel_sprite_ids() -> Vec<AssetId> {
    let mut ids = vec![
        PANEL_BG,
        PLUS_BG,
        TYPEDROP_BG,
        FORMADD_BG,
        FORMCANCEL_BG,
        LIST_TRACK,
        LIST_THUMB,
    ];
    ids.extend((0..MAX_ROWS).map(list_row_bg));
    ids.extend((0..MAX_ROWS).map(filter_row_bg));
    ids.extend((0..PICKER_ROWS).map(picker_row_bg));
    ids
}
fn panel_label_ids() -> Vec<AssetId> {
    let mut ids = vec![
        PLUS_LABEL,
        TYPEDROP_LABEL,
        FORMADD_LABEL,
        FORMCANCEL_LABEL,
        FORM_TITLE,
        EMPTY_LABEL,
    ];
    ids.extend((0..MAX_ROWS).map(list_row_label));
    ids.extend((0..MAX_ROWS).map(filter_row_label));
    ids.extend((0..PICKER_ROWS).map(picker_row_label));
    ids
}

// Hide every panel element, including the two typed fields (and blur them so a
// hidden field cannot keep keyboard focus).
pub(crate) fn hide_all(world: &mut World) {
    for id in panel_sprite_ids() {
        set_sprite_visible(world, id, false);
    }
    for id in panel_label_ids() {
        set_label_visible(world, id, false);
    }
    hide_field(world, FILTER_INPUT);
    hide_field(world, NAME_INPUT);
}

// Hide the body families + fields (but keep the panel bg + header), so the
// active mode can re-show only its own rows each frame.
fn hide_body(world: &mut World) {
    for id in panel_sprite_ids() {
        if matches!(id, PANEL_BG | PLUS_BG | TYPEDROP_BG) {
            continue;
        }
        set_sprite_visible(world, id, false);
    }
    for id in panel_label_ids() {
        if matches!(id, PLUS_LABEL | TYPEDROP_LABEL) {
            continue;
        }
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

    fn view<'a>(
        mode: Mode,
        dropdown: bool,
        filter_options: &'a [String],
        list_items: &'a [(String, String)],
        picker_options: &'a [String],
        mouse: [f32; 2],
    ) -> PanelView<'a> {
        PanelView {
            mode,
            type_dropdown_open: dropdown,
            filter_label: ALL_LABEL,
            filter_options,
            filter_selected: 0,
            list_items,
            list_scroll: 0,
            picker_options,
            picker_scroll: 0,
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
    fn header_plus_and_dropdown_do_not_overlap() {
        let plus = plus_rect(1280.0);
        let td = typedrop_rect(1280.0);
        assert!(plus[0] + plus[2] <= td[0], "+ is left of the dropdown");
        assert_eq!(td[0] + td[2], 1280.0, "dropdown reaches the panel right");
    }

    #[test]
    fn plus_toggles_the_picker() {
        let opts = vec![ALL_LABEL.to_string()];
        let v = view(Mode::List, false, &opts, &[], &[], [0.0, 0.0]);
        let plus = plus_rect(1280.0);
        assert_eq!(
            hit_test(&v, plus[0] + 5.0, plus[1] + 5.0, 1280.0),
            Some(PanelAction::TogglePicker)
        );
    }

    #[test]
    fn type_dropdown_opens_and_picks_an_option() {
        let opts = vec![ALL_LABEL.to_string(), "PointLight".to_string()];
        // Closed: clicking the button opens it.
        let v = view(Mode::List, false, &opts, &[], &[], [0.0, 0.0]);
        let td = typedrop_rect(1280.0);
        assert_eq!(
            hit_test(&v, td[0] + 5.0, td[1] + 5.0, 1280.0),
            Some(PanelAction::ToggleTypeDropdown)
        );
        // Open: clicking option row 1 picks it.
        let vo = view(Mode::List, true, &opts, &[], &[], [0.0, 0.0]);
        let r1 = filter_option_rect(1280.0, 1);
        assert_eq!(
            hit_test(&vo, r1[0] + 5.0, r1[1] + 5.0, 1280.0),
            Some(PanelAction::PickFilter(1))
        );
        // Open: a click elsewhere closes it.
        assert_eq!(
            hit_test(&vo, 10.0, 10.0, 1280.0),
            Some(PanelAction::ToggleTypeDropdown)
        );
    }

    #[test]
    fn picker_row_maps_to_a_scrolled_type_index() {
        let opts = vec![ALL_LABEL.to_string()];
        let picker: Vec<String> = ADD_TYPES.iter().map(|s| s.to_string()).collect();
        let mut v = view(Mode::TypePicker, false, &opts, &[], &picker, [0.0, 0.0]);
        v.picker_scroll = 2;
        let r0 = picker_row_rect(1280.0, 0);
        assert_eq!(
            hit_test(&v, r0[0] + 5.0, r0[1] + 5.0, 1280.0),
            Some(PanelAction::PickType(2)),
            "row 0 maps to option `scroll + 0`"
        );
    }

    #[test]
    fn form_buttons_resolve() {
        let opts = vec![ALL_LABEL.to_string()];
        let v = view(Mode::AddForm, false, &opts, &[], &[], [0.0, 0.0]);
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
    fn clicks_outside_the_panel_fall_through() {
        let opts = vec![ALL_LABEL.to_string()];
        let v = view(Mode::List, false, &opts, &[], &[], [0.0, 0.0]);
        assert_eq!(hit_test(&v, 10.0, 400.0, 1280.0), None);
    }

    #[test]
    fn name_field_click_is_consumed_for_the_text_system() {
        let opts = vec![ALL_LABEL.to_string()];
        let v = view(Mode::AddForm, false, &opts, &[], &[], [0.0, 0.0]);
        let f = name_input_rect(1280.0);
        assert_eq!(
            hit_test(&v, f[0] + 5.0, f[1] + 5.0, 1280.0),
            Some(PanelAction::Consume)
        );
    }
}
