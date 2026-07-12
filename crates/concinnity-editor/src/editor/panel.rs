// src/editor/panel.rs
//
// The editor "Assets" panel: browse the world's existing assets (grouped by
// type, filtered by type). Like the rest of the editor HUD it is plain
// `Sprite` / `TextLabel` / `TextInput` components at reserved ids (injected by
// `inject.rs`), driven each frame by the editor hook -- nothing here reaches the
// shipped runtime. This module owns the panel's pure geometry, its click
// resolution, and the per-frame layout that shows / positions the elements; the
// hook owns the state and the option lists.
//
// The panel is a floating column: a draggable title bar ("Assets") across its
// top, defaulting to below the top bar's buttons; the hook owns its position and
// clamps a drag so the panel stays fully on screen. Under the title bar the
// header is a square "+" (add) button and a combo area. While the type picker is
// open, the "+" becomes a gray "X" that returns to the browse list. The combo
// area is a dropdown that, when opened, turns into a filter text field
// (`FILTER_INPUT`, a real `TextInput`) with an option list floating below it:
//   * Filter  -- opened by clicking the combo: the list is the asset types
//                present in the world; picking one filters the browse list.
//   * Picker  -- opened by clicking "+": the list is the addable asset types;
//                picking one opens the add form.
// The body below the header is the browse list: the world's existing assets,
// grouped by type (a type sub-header then its indented names), scrollable.
// Clicking a name opens that asset's add / edit form -- a separate floating
// panel (`form_panel.rs`) -- and the name row stays highlighted while its form
// is open. Hovering a name also reveals a triple-dot button opening a small
// Delete menu.

use crate::assets::TextAlign;
use crate::ecs::World;
use crate::ecs::asset_id::AssetId;

use super::hud;
use super::widget::{self, place_sprite, point_in};

// The grouped browse list (row model, grouping, geometry, base style, and the
// per-row + scrollbar draw) is shared with the Template detail panel; only the
// interactive chrome below (hover / selected tints, triple-dot, Delete) is
// Assets-panel-specific. `MAX_ROWS` + `ListRow` are re-exported so the hook keeps
// referring to them as `panel::MAX_ROWS` / `panel::ListRow`.
use super::asset_list::{
    self, HEADER_ROW_TINT, LABEL, PAD, ROW_H, ROW_LABEL_TOP, ROW_TINT, SCROLLBAR_W,
};
pub(crate) use super::asset_list::{ListRow, MAX_ROWS};

// The addable asset types the "+" picker offers: every External (user-declarable)
// type that (a) recompiles cleanly when added with default args and (b) is useful
// when added blank. `add_types_cook_with_default_args` guards (a) by running every
// entry through the real cook pipeline; `add_types_are_the_curated_blank_useful_addable_set`
// guards the whole boundary -- every addable-and-cooks-blank type must be either
// here or in that test's EXCLUDED list (with a reason), so a newly registered type
// is a deliberate choice, never a silent omission. The mix spans lights / scene
// effects, procedural geometry, a camera, referenced library assets (Material /
// Font / BlockType), a UI layer (View), UI widgets, UI interaction (HitRegion) +
// input (KeyBinding), a HUD element (FpsCounter), and audio. Most entries are
// naturally multi-instance; a few are effectively singletons where a second
// instance is harmlessly ignored (only the first enabled VolumetricFog draws; the
// first Camera3D is the active one), which is fine because the common action is
// adding the first one to a world that lacks it.
// The types held back cook blank but are not useful blank: world-config singletons
// (GraphicsConfig, PhysicsConfig, Window, Application, ...) want an edit-or-add flow
// rather than a blind append; engine-injected HUDs (DebugHud / StatHud) are added by
// `cn build`, not by hand; and types defined by a nested array or a source file
// (Model, Scene, Story, LayoutContainer's rows, ...) are inert until the nested /
// source form controls exist. Types that cannot even cook blank -- needing a source
// or a required cross-reference (Mesh / AudioClip / Joint) -- can never be offered.
pub(crate) const ADD_TYPES: &[&str] = &[
    // Lights + scene effects.
    "PointLight",
    "DirectionalLight",
    "ParticleEmitter",
    "Decal",
    "ReflectionProbe",
    "VolumetricFog",
    "GlassPanel",
    "WaterSurface",
    // Geometry + camera.
    "Room",
    "Camera3D",
    // Library assets (referenced by other assets, by name).
    "Material",
    "Font",
    "BlockType",
    // UI structure + widgets.
    "View",
    "Sprite",
    "TextLabel",
    "TextInput",
    // UI interaction + input.
    "HitRegion",
    "KeyBinding",
    // UI HUD.
    "FpsCounter",
    // Audio.
    "AudioEmitter",
    "AudioCue",
];

// World-config singletons: exactly one instance belongs to a world. They are
// offered in the "+" picker alongside `ADD_TYPES`, but picking one EDITS the
// world's existing instance when it has one and only ADDS when it does not (the
// hook's `open_form` is handed the existing entry's index) -- an edit-or-add flow,
// never a blind second append. Held out of `ADD_TYPES` so the plain add path can
// keep assuming multi-instance. Like the addables, each must cook blank (guarded).
pub(crate) const CONFIG_TYPES: &[&str] = &[
    "GraphicsConfig",
    "PhysicsConfig",
    "PostProcessConfig",
    "StreamingConfig",
    "Window",
    "Application",
];

// Whether `ty` is a world-config singleton (edit-or-add rather than blind append).
pub(crate) fn is_singleton(ty: &str) -> bool {
    CONFIG_TYPES.contains(&ty)
}

// Every type the "+" picker offers: the multi-instance addables plus the config
// singletons.
pub(crate) fn picker_types() -> impl Iterator<Item = &'static str> {
    ADD_TYPES.iter().chain(CONFIG_TYPES.iter()).copied()
}

// The label of the "all assets" filter option (the default), shown first in the
// combo's filter list and as the combo's text when no type filter is active.
pub(crate) const ALL_LABEL: &str = "All";

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

// Reserved asset-id families for the panel, offset past the top-bar HUD's ids
// (see `hud.rs`; the top bar uses `ID_BASE + 0..0x23`). Interned world ids never
// reach this range and these are never serialized. The asset-id VALUE does not
// affect draw order (the overlay draws in component-insertion order, not by id);
// z-order is set by the sequence in `all_sprite_ids` / `all_label_ids`, which
// `inject.rs` inserts in that order. The id values only need to be distinct.
const PANEL: u32 = 0x3000_0000 + 0x40;
pub(crate) const PANEL_BG: AssetId = AssetId(PANEL);
pub(crate) const PLUS_BG: AssetId = AssetId(PANEL + 1);
pub(crate) const PLUS_LABEL: AssetId = AssetId(PANEL + 2);
pub(crate) const TYPEDROP_BG: AssetId = AssetId(PANEL + 3);
pub(crate) const TYPEDROP_LABEL: AssetId = AssetId(PANEL + 4);
pub(crate) const FILTER_INPUT: AssetId = AssetId(PANEL + 5);
pub(crate) const EMPTY_LABEL: AssetId = AssetId(PANEL + 12);
pub(crate) const LIST_TRACK: AssetId = AssetId(PANEL + 0x58);
pub(crate) const LIST_THUMB: AssetId = AssetId(PANEL + 0x59);
pub(crate) const COMBO_BG: AssetId = AssetId(PANEL + 0x5A);
// The draggable title bar across the panel top.
pub(crate) const TITLE_BG: AssetId = AssetId(PANEL + 0x5B);
pub(crate) const TITLE_LABEL: AssetId = AssetId(PANEL + 0x5C);
// The "X" close button in the title bar's top-right corner.
pub(crate) const CLOSE_BG: AssetId = AssetId(PANEL + 0x5D);
pub(crate) const CLOSE_LABEL: AssetId = AssetId(PANEL + 0x5E);
pub(crate) const DOT_BG: AssetId = AssetId(PANEL + 0xA0);
pub(crate) const DOT1: AssetId = AssetId(PANEL + 0xA1);
pub(crate) const DOT2: AssetId = AssetId(PANEL + 0xA2);
pub(crate) const DOT3: AssetId = AssetId(PANEL + 0xA3);
pub(crate) const MENU_BG: AssetId = AssetId(PANEL + 0xB0);
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

// Geometry, in window pixels. Every rect derives from the panel's origin `o`
// (its title bar's top-left corner), so dragging the title bar moves the whole
// panel; the hook owns the origin.
pub(crate) const PANEL_W: f32 = 320.0;
const HEADER_H: f32 = 40.0;
const GAP: f32 = 6.0;
// The triple-dot button on a hovered name row.
const DOT_SZ: f32 = 24.0;
// The floating Delete menu.
const MENU_W: f32 = 132.0;
const MENU_ROW_H: f32 = 30.0;

const PANEL_BG_TINT: [f32; 4] = [0.09, 0.09, 0.12, 0.97];
const PLUS_TINT: [f32; 4] = [0.20, 0.44, 0.30, 1.0];
const TYPEDROP_TINT: [f32; 4] = [0.18, 0.20, 0.28, 1.0];
// Interactive name-row tints, layered over the shared base `ROW_TINT`.
const ROW_TINT_HOVER: [f32; 4] = [0.22, 0.26, 0.36, 0.98];
const ROW_TINT_SELECTED: [f32; 4] = [0.16, 0.22, 0.34, 1.0];
const OPTION_TINT: [f32; 4] = [0.16, 0.16, 0.20, 1.0];
const OPTION_TINT_HOVER: [f32; 4] = [0.24, 0.28, 0.40, 1.0];
const OPTION_TINT_SELECTED: [f32; 4] = [0.16, 0.22, 0.34, 1.0];
const COMBO_BG_TINT: [f32; 4] = [0.10, 0.10, 0.13, 1.0];
const CANCEL_TINT: [f32; 4] = [0.30, 0.30, 0.34, 1.0];
const DOT_BG_TINT: [f32; 4] = [0.30, 0.34, 0.46, 0.95];
const DOT_TINT: [f32; 4] = [0.90, 0.92, 0.96, 1.0];
const MENU_BG_TINT: [f32; 4] = [0.14, 0.14, 0.18, 1.0];
const MENU_ROW_TINT: [f32; 4] = [0.16, 0.16, 0.20, 1.0];
const MENU_ROW_HOVER: [f32; 4] = [0.26, 0.30, 0.42, 1.0];
const LABEL_DIM: [f32; 3] = [0.60, 0.60, 0.66];
const LABEL_WHITE: [f32; 3] = [1.0, 1.0, 1.0];
const DELETE_LABEL: [f32; 3] = [0.95, 0.60, 0.58];

// Where the panel sits until the user drags it: right-aligned below the top bar.
pub(crate) fn default_origin(vw: f32) -> [f32; 2] {
    [vw - PANEL_W, hud::body_top()]
}

// The panel's fixed footprint, for the hook's drag clamp.
pub(crate) fn size() -> [f32; 2] {
    let r = panel_rect([0.0, 0.0]);
    [r[2], r[3]]
}

// The panel outer rect (title bar + header + body) at origin `o`.
pub(crate) fn panel_rect(o: [f32; 2]) -> [f32; 4] {
    [
        o[0],
        o[1],
        PANEL_W,
        widget::TITLE_H + HEADER_H + MAX_ROWS as f32 * ROW_H,
    ]
}

// The draggable title bar across the panel top.
pub(crate) fn title_rect(o: [f32; 2]) -> [f32; 4] {
    [o[0], o[1], PANEL_W, widget::TITLE_H]
}

// The "X" close button in the title bar's top-right corner.
pub(crate) fn close_rect(o: [f32; 2]) -> [f32; 4] {
    widget::close_rect(title_rect(o))
}

// The square "+" add button (panel header below the title bar, left).
pub(crate) fn plus_rect(o: [f32; 2]) -> [f32; 4] {
    [o[0], o[1] + widget::TITLE_H, HEADER_H, HEADER_H]
}

// The combo area (panel header, filling the rest of the row): the browse-filter
// label when closed, the filter text field when open.
pub(crate) fn combo_rect(o: [f32; 2]) -> [f32; 4] {
    [
        o[0] + HEADER_H + GAP,
        o[1] + widget::TITLE_H,
        PANEL_W - HEADER_H - GAP,
        HEADER_H,
    ]
}

// The filter text field, centred in the combo area.
pub(crate) fn filter_input_rect(o: [f32; 2]) -> [f32; 4] {
    let c = combo_rect(o);
    [c[0], c[1] + (HEADER_H - ROW_H) * 0.5, c[2], ROW_H]
}

// Where the body (below the header) begins.
fn body_y(o: [f32; 2]) -> f32 {
    o[1] + widget::TITLE_H + HEADER_H
}

// A body row `i` spanning the panel width (list or combo option).
pub(crate) fn list_row_rect(o: [f32; 2], i: usize) -> [f32; 4] {
    [o[0], body_y(o) + i as f32 * ROW_H, PANEL_W, ROW_H]
}
pub(crate) fn combo_option_rect(o: [f32; 2], i: usize) -> [f32; 4] {
    list_row_rect(o, i)
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

// The row menu, floating just below a name row at visible index `vr`. A single
// Delete row (a name-row click opens the edit form, so Edit is redundant here).
// Returns (background, delete row).
fn menu_rects(o: [f32; 2], vr: usize) -> ([f32; 4], [f32; 4]) {
    let x = o[0] + PANEL_W - MENU_W - SCROLLBAR_W - 2.0;
    let top = body_y(o) + vr as f32 * ROW_H + ROW_H;
    let delete = [x, top, MENU_W, MENU_ROW_H];
    let bg = [x, top, MENU_W, MENU_ROW_H];
    (bg, delete)
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
    // A name row's body: open that entry's edit form (carries the entry index).
    OpenEntry(usize),
    // A name row's triple-dot: open its Delete menu (carries the entry).
    OpenRowMenu(usize),
    // The open row menu's Delete row.
    RowDelete,
    // Dismiss any open overlay (combo / row menu) without picking.
    CloseOverlays,
    // A click inside the panel that hits no control (swallowed so it does not
    // fall through to the world; a text-field click resolves to this too, so the
    // engine's text-input system takes focus).
    Consume,
}

// The per-frame data the hook hands to `apply` / `hit_test`.
pub(crate) struct PanelView<'a> {
    pub combo: Combo,
    // The combo button's text ("All" or the active type), shown when closed.
    pub filter_label: &'a str,
    // The floating combo options (already narrowed by the typed field).
    pub combo_options: &'a [String],
    // Index of the highlighted option (the active filter), for the Filter flavour.
    pub combo_selected: Option<usize>,
    pub combo_scroll: usize,
    // The grouped browse rows (type sub-headers + indented names).
    pub list_rows: &'a [ListRow],
    pub list_scroll: usize,
    // The entry index whose Delete menu is open, if any.
    pub row_menu: Option<usize>,
    // The entry whose edit form is open, if any: its name row stays highlighted.
    pub selected: Option<usize>,
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

// Resolve a click against the open panel at origin `o`. `None` means the click
// missed the panel entirely (the caller lets it fall through). Text-field clicks
// resolve to `Consume` (swallowed here; the engine's text-input system focuses
// the field from the same input). Title-bar presses never reach this: the hook
// intercepts them first to start a drag.
pub(crate) fn hit_test(view: &PanelView, mx: f32, my: f32, o: [f32; 2]) -> Option<PanelAction> {
    // An open row menu is modal over the panel: its rows pick, anything else
    // dismisses it.
    if let Some(entry) = view.row_menu {
        if let Some(vr) = visible_row_of(view, entry) {
            let (_, delete) = menu_rects(o, vr);
            if point_in(mx, my, delete) {
                return Some(PanelAction::RowDelete);
            }
        }
        return Some(PanelAction::CloseOverlays);
    }

    // An open combo captures the header field + its floating options.
    if view.combo != Combo::Closed {
        if point_in(mx, my, plus_rect(o)) {
            return Some(PanelAction::TogglePicker);
        }
        if point_in(mx, my, combo_rect(o)) {
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
            if point_in(mx, my, combo_option_rect(o, r)) {
                return Some(PanelAction::PickOption(idx));
            }
        }
        return Some(PanelAction::CloseOverlays);
    }

    // Combo closed: clicks outside the panel fall through (the hook's caller
    // decides who else wants them).
    if !point_in(mx, my, panel_rect(o)) {
        return None;
    }
    if point_in(mx, my, plus_rect(o)) {
        return Some(PanelAction::TogglePicker);
    }
    if point_in(mx, my, combo_rect(o)) {
        return Some(PanelAction::ToggleFilter);
    }

    // The browse list: a name row's triple-dot opens its menu; the rest of the
    // row opens that entry's edit form.
    let scroll = view.list_scroll.min(view.list_rows.len().saturating_sub(1));
    for r in 0..MAX_ROWS {
        let idx = scroll + r;
        if idx >= view.list_rows.len() {
            break;
        }
        let Some(entry) = view.list_rows[idx].entry else {
            continue;
        };
        let rect = list_row_rect(o, r);
        if point_in(mx, my, rect) {
            if point_in(mx, my, dot_rect(rect)) {
                return Some(PanelAction::OpenRowMenu(entry));
            }
            return Some(PanelAction::OpenEntry(entry));
        }
    }
    Some(PanelAction::Consume)
}

// Position + show the panel's elements for this frame at origin `o`, or hide
// them all when the panel is closed (`view` is `None`).
pub(crate) fn apply(world: &mut World, view: Option<&PanelView>, o: [f32; 2]) {
    let Some(view) = view else {
        hide_all(world);
        return;
    };

    // Blank everything, then re-show what this frame needs.
    hide_all(world);

    place_sprite(world, PANEL_BG, panel_rect(o), PANEL_BG_TINT, true);
    widget::place_title(world, TITLE_BG, TITLE_LABEL, title_rect(o), "Assets");
    let close_hover = point_in(view.mouse[0], view.mouse[1], close_rect(o));
    widget::place_close(world, CLOSE_BG, CLOSE_LABEL, title_rect(o), close_hover);

    // The "+" add button. While the type picker is open it becomes a gray "X"
    // that returns to the browse list (the previous focus).
    let (glyph, tint) = if view.combo == Combo::Picker {
        ("X", CANCEL_TINT)
    } else {
        ("+", PLUS_TINT)
    };
    place_sprite(world, PLUS_BG, plus_rect(o), tint, true);
    place_plus_glyph(world, plus_rect(o), glyph);

    // The combo area: the browse label when closed, the filter field when open.
    if view.combo == Combo::Closed {
        place_sprite(world, TYPEDROP_BG, combo_rect(o), TYPEDROP_TINT, true);
        let td = combo_rect(o);
        place_left_label(
            world,
            TYPEDROP_LABEL,
            [td[0] + PAD, td[1] + HEADER_H * 0.5 - 10.0],
            view.filter_label,
            LABEL,
            true,
        );
        layout_list(world, view, o);
    } else {
        widget::show_field(world, FILTER_INPUT, filter_input_rect(o), true);
        layout_combo(world, view, o);
    }
}

// The "+" / "X" glyph, centered in the add button and drawn a step larger than
// the body text (the box itself stays a HEADER_H square).
const PLUS_SCALE: f32 = 1.3;
fn place_plus_glyph(world: &mut World, rect: [f32; 4], glyph: &str) {
    if let Some(l) = widget::label_mut(world, PLUS_LABEL) {
        l.x = rect[0] + rect[2] * 0.5;
        l.y = rect[1] + rect[3] * 0.5 - 10.0 * PLUS_SCALE;
        l.align = TextAlign::Center;
        l.color = LABEL_WHITE;
        l.scale = PLUS_SCALE;
        l.visible = true;
        l.content = glyph.to_string();
    }
}

fn layout_list(world: &mut World, view: &PanelView, o: [f32; 2]) {
    if view.list_rows.is_empty() {
        place_left_label(
            world,
            EMPTY_LABEL,
            [o[0] + PAD, body_y(o) + PAD],
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
        let rect = list_row_rect(o, r);
        if row.is_header {
            asset_list::place_row(
                world,
                list_row_bg(r),
                list_row_label(r),
                row,
                rect,
                HEADER_ROW_TINT,
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
        // The row whose edit form is open stays highlighted (hover still wins so
        // the cursor keeps its feedback).
        let selected = row.entry.is_some() && view.selected == row.entry;
        let tint = if hovered {
            ROW_TINT_HOVER
        } else if selected {
            ROW_TINT_SELECTED
        } else {
            ROW_TINT
        };
        asset_list::place_row(world, list_row_bg(r), list_row_label(r), row, rect, tint);
    }
    // The triple-dot follows the row whose menu is open, else the hovered row.
    // Its background box shows only when the dots themselves are hovered, or when
    // the menu is open on that row (a "clicked" state); a plain row-hover shows
    // the bare white dots.
    if let Some(r) = menu_row.or(hovered_row) {
        let rect = list_row_rect(o, r);
        let over_dots = point_in(view.mouse[0], view.mouse[1], dot_rect(rect));
        let show_box = menu_row == Some(r) || over_dots;
        place_dot(world, rect, show_box);
    }
    if let Some(r) = menu_row {
        layout_row_menu(world, view, o, r);
    }
    asset_list::layout_scrollbar(
        world,
        LIST_TRACK,
        LIST_THUMB,
        total,
        scroll,
        o[0] + PANEL_W,
        body_y(o),
    );
}

fn layout_combo(world: &mut World, view: &PanelView, o: [f32; 2]) {
    let total = view.combo_options.len();
    let scroll = view.combo_scroll.min(total.saturating_sub(1));
    let shown = total.saturating_sub(scroll).clamp(1, MAX_ROWS);
    let backing = [o[0], body_y(o), PANEL_W, shown as f32 * ROW_H + PAD];
    place_sprite(world, COMBO_BG, backing, COMBO_BG_TINT, true);
    if total == 0 {
        place_left_label(
            world,
            EMPTY_LABEL,
            [o[0] + PAD, body_y(o) + PAD],
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
        let rect = combo_option_rect(o, r);
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
    asset_list::layout_scrollbar(
        world,
        LIST_TRACK,
        LIST_THUMB,
        total,
        scroll,
        o[0] + PANEL_W,
        body_y(o),
    );
}

// The three stacked white dots of the triple-dot button. The background box is
// shown only when `show_box` (the dots are hovered, or the menu is open); a plain
// row-hover shows the bare dots. `DOT_BG` is left hidden (blanked by `hide_all`)
// when `show_box` is false.
fn place_dot(world: &mut World, row: [f32; 4], show_box: bool) {
    let d = dot_rect(row);
    if show_box {
        place_sprite(world, DOT_BG, d, DOT_BG_TINT, true);
    }
    let cx = d[0] + d[2] * 0.5;
    let cy = d[1] + d[3] * 0.5;
    let s = 3.5;
    let gap = 3.5;
    for (id, dy) in [(DOT1, -gap - s), (DOT2, -s * 0.5), (DOT3, gap)] {
        place_sprite(world, id, [cx - s * 0.5, cy + dy, s, s], DOT_TINT, true);
    }
}

fn layout_row_menu(world: &mut World, view: &PanelView, o: [f32; 2], vr: usize) {
    let (bg, delete) = menu_rects(o, vr);
    place_sprite(world, MENU_BG, bg, MENU_BG_TINT, true);
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

// Every panel sprite id, so the closed / hidden pass can blank the whole panel
// (and `inject.rs` can create exactly this set). THE ORDER OF THIS VEC IS THE DRAW
// ORDER: `inject.rs` adds the panel's Sprites in this sequence, and the overlay
// draws components in insertion (component-column) order -- NOT by asset id -- so
// later entries paint on top. Bottom-to-top: panel background, header chrome, the
// combo backing (under its option rows), the row families, then the floating
// overlays (scrollbar, triple-dot, row menu), which must sit ABOVE the row
// backgrounds so a hovered row's fill cannot cover them.
pub(crate) fn all_sprite_ids() -> Vec<AssetId> {
    let mut ids = vec![PANEL_BG, TITLE_BG, CLOSE_BG, PLUS_BG, TYPEDROP_BG, COMBO_BG];
    ids.extend((0..MAX_ROWS).map(list_row_bg));
    ids.extend((0..MAX_ROWS).map(combo_row_bg));
    ids.extend([
        LIST_TRACK,
        LIST_THUMB,
        DOT_BG,
        DOT1,
        DOT2,
        DOT3,
        MENU_BG,
        MENU_DELETE_BG,
    ]);
    ids
}
// Same draw-order contract as `all_sprite_ids` (all TextLabels draw after all
// Sprites, so any label sits above the sprite chrome; among labels this Vec's
// order decides who wins). The row-menu captions come last so they draw above the
// row labels the menu floats over.
pub(crate) fn all_label_ids() -> Vec<AssetId> {
    let mut ids = vec![
        TITLE_LABEL,
        CLOSE_LABEL,
        PLUS_LABEL,
        TYPEDROP_LABEL,
        EMPTY_LABEL,
    ];
    ids.extend((0..MAX_ROWS).map(list_row_label));
    ids.extend((0..MAX_ROWS).map(combo_row_label));
    ids.extend([MENU_DELETE_LABEL]);
    ids
}

// Every typed field the panel injects: just the combo's filter field (the form's
// inputs belong to `form_panel.rs`).
pub(crate) fn all_field_ids() -> Vec<AssetId> {
    vec![FILTER_INPUT]
}

// Hide every panel element, including the typed fields (and blur them so a hidden
// field cannot keep keyboard focus).
pub(crate) fn hide_all(world: &mut World) {
    for id in all_sprite_ids() {
        widget::set_sprite_visible(world, id, false);
    }
    for id in all_label_ids() {
        widget::set_label_visible(world, id, false);
    }
    for id in all_field_ids() {
        widget::hide_field(world, id);
    }
}

// -- Element mutation helpers -------------------------------------------------
//
// The reserved-id lookups (`widget::sprite_mut` / `label_mut` / `input_mut`) and
// `place_sprite` / visibility setters live in `widget.rs`, shared with `hud.rs`.
// These wrap them with the panel's label / field conventions.

fn place_left_label(
    world: &mut World,
    id: AssetId,
    pos: [f32; 2],
    content: &str,
    color: [f32; 3],
    visible: bool,
) {
    if let Some(l) = widget::label_mut(world, id) {
        l.x = pos[0];
        l.y = pos[1];
        l.align = TextAlign::Left;
        l.color = color;
        l.visible = visible;
        l.content = content.to_string();
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

// Whether the cursor is over the scrollable body area (for wheel scrolling).
pub(crate) fn cursor_over_body(mx: f32, my: f32, o: [f32; 2]) -> bool {
    let p = panel_rect(o);
    mx >= p[0] && mx < p[0] + p[2] && my >= body_y(o) && my < p[1] + p[3]
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::assets::{Sprite, TextInput, TextLabel};

    // Point the cook's `.concinnity/` (its content-addressed cache) at a private
    // temp dir for the whole test process, so the cook-based tests below never read
    // or write the working directory (the shader compile itself already uses a unique
    // temp path). Set once per process; the shared cache is race-tolerant.
    fn isolate_state_dir() {
        use std::sync::Once;
        static ONCE: Once = Once::new();
        ONCE.call_once(|| {
            let dir = std::env::temp_dir().join(format!("cn-editor-tests-{}", std::process::id()));
            let _ = std::fs::create_dir_all(&dir);
            concinnity_core::paths::set_root(dir);
        });
    }

    // The panel origin the tests lay out at: the default anchor in a 1280-wide
    // window.
    fn test_origin() -> [f32; 2] {
        default_origin(1280.0)
    }

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
        combo: Combo,
        row_menu: Option<usize>,
        mouse: [f32; 2],
    ) -> PanelView<'a> {
        PanelView {
            combo,
            filter_label: ALL_LABEL,
            combo_options: &fx.combo_options,
            combo_selected: None,
            combo_scroll: 0,
            list_rows: &fx.list_rows,
            list_scroll: 0,
            row_menu,
            selected: None,
            mouse,
        }
    }

    // A world with every panel element injected (hidden), for driving `apply`.
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
        for id in all_field_ids() {
            world.add_component(TextInput {
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
            .expect("sprite present")
    }

    fn sprite_visible(world: &World, id: AssetId) -> bool {
        sprite(world, id).visible
    }

    #[test]
    fn default_origin_sits_below_the_top_bar_right_aligned() {
        let o = test_origin();
        let p = panel_rect(o);
        assert_eq!(p[0] + p[2], 1280.0, "flush to the window right");
        assert_eq!(p[1], hud::body_top(), "starts below the top-bar buttons");
        // The whole panel follows its origin: dragged elsewhere, every rect moves.
        let moved = panel_rect([40.0, 60.0]);
        assert_eq!((moved[0], moved[1]), (40.0, 60.0));
        assert_eq!((moved[2], moved[3]), (p[2], p[3]));
        assert_eq!(size(), [p[2], p[3]], "the drag-clamp footprint matches");
    }

    #[test]
    fn title_bar_spans_the_panel_top_and_header_sits_below_it() {
        let o = test_origin();
        let title = title_rect(o);
        assert_eq!(title, [o[0], o[1], PANEL_W, widget::TITLE_H]);
        let plus = plus_rect(o);
        assert_eq!(
            plus[1],
            o[1] + widget::TITLE_H,
            "the header row starts below the title bar"
        );
        let combo = combo_rect(o);
        assert!(plus[0] + plus[2] <= combo[0], "+ is left of the combo");
        assert_eq!(
            combo[0] + combo[2],
            o[0] + PANEL_W,
            "combo reaches the panel right"
        );
    }

    #[test]
    fn plus_toggles_the_picker() {
        let fx = Fixture {
            combo_options: vec![],
            list_rows: vec![],
        };
        let v = view(&fx, Combo::Closed, None, [0.0, 0.0]);
        let plus = plus_rect(test_origin());
        assert_eq!(
            hit_test(&v, plus[0] + 5.0, plus[1] + 5.0, test_origin()),
            Some(PanelAction::TogglePicker)
        );
    }

    // The header button renders as a green "+" while browsing and as a gray "X"
    // (return to the list) while the type picker is open.
    #[test]
    fn plus_renders_as_an_x_while_picker_is_open() {
        let fx = Fixture {
            combo_options: vec!["PointLight".to_string()],
            list_rows: vec![],
        };
        let glyph = |world: &World| {
            world
                .query::<TextLabel>()
                .find(|l| l.asset_id == PLUS_LABEL)
                .unwrap()
                .clone()
        };
        let mut world = injected_world();
        let o = test_origin();
        apply(
            &mut world,
            Some(&view(&fx, Combo::Closed, None, [0.0, 0.0])),
            o,
        );
        let l = glyph(&world);
        assert_eq!(l.content, "+");
        assert_eq!(sprite(&world, PLUS_BG).tint, PLUS_TINT);
        assert!(
            l.scale > 1.0,
            "the glyph draws larger than the body text (the box is unchanged)"
        );
        apply(
            &mut world,
            Some(&view(&fx, Combo::Picker, None, [0.0, 0.0])),
            o,
        );
        assert_eq!(glyph(&world).content, "X");
        assert_eq!(
            sprite(&world, PLUS_BG).tint,
            CANCEL_TINT,
            "gray while the picker is open"
        );
    }

    // The title bar is drawn with the panel's heading; a click on it reaches no
    // control (the hook intercepts it for a drag before this hit test, and the
    // fall-through here just consumes it).
    #[test]
    fn title_bar_renders_heading_and_consumes_clicks() {
        let fx = Fixture {
            combo_options: vec![],
            list_rows: vec![],
        };
        let mut world = injected_world();
        let o = test_origin();
        apply(
            &mut world,
            Some(&view(&fx, Combo::Closed, None, [0.0, 0.0])),
            o,
        );
        let title = world
            .query::<TextLabel>()
            .find(|l| l.asset_id == TITLE_LABEL)
            .unwrap();
        assert!(title.visible);
        assert_eq!(title.content, "Assets");
        assert!(sprite_visible(&world, TITLE_BG));
        let t = title_rect(o);
        let v = view(&fx, Combo::Closed, None, [0.0, 0.0]);
        assert_eq!(
            hit_test(&v, t[0] + 5.0, t[1] + 5.0, o),
            Some(PanelAction::Consume)
        );
    }

    #[test]
    fn combo_button_opens_and_picks_an_option() {
        let fx = Fixture {
            combo_options: vec![ALL_LABEL.to_string(), "PointLight".to_string()],
            list_rows: vec![],
        };
        // Closed: clicking the combo opens the filter dropdown.
        let o = test_origin();
        let v = view(&fx, Combo::Closed, None, [0.0, 0.0]);
        let c = combo_rect(o);
        assert_eq!(
            hit_test(&v, c[0] + 5.0, c[1] + 5.0, o),
            Some(PanelAction::ToggleFilter)
        );
        // Open: clicking option row 1 picks it.
        let vo = view(&fx, Combo::Filter, None, [0.0, 0.0]);
        let r1 = combo_option_rect(o, 1);
        assert_eq!(
            hit_test(&vo, r1[0] + 5.0, r1[1] + 5.0, o),
            Some(PanelAction::PickOption(1))
        );
        // Open: clicking the header field keeps focus (consumed).
        assert_eq!(
            hit_test(&vo, c[0] + 5.0, c[1] + 5.0, o),
            Some(PanelAction::Consume)
        );
        // Open: a click on empty body space closes it.
        assert_eq!(
            hit_test(&vo, 640.0, 700.0, o),
            Some(PanelAction::CloseOverlays)
        );
    }

    #[test]
    fn combo_open_plus_returns_to_the_list() {
        // With the picker open, the header "+" turns into an "X" that toggles
        // back to the browse list.
        let fx = Fixture {
            combo_options: vec!["PointLight".to_string()],
            list_rows: vec![],
        };
        let o = test_origin();
        let vo = view(&fx, Combo::Filter, None, [0.0, 0.0]);
        let plus = plus_rect(o);
        assert_eq!(
            hit_test(&vo, plus[0] + 5.0, plus[1] + 5.0, o),
            Some(PanelAction::TogglePicker)
        );
    }

    #[test]
    fn row_menu_dismisses_when_its_entry_is_off_window() {
        // An open row menu whose entry scrolled out of the visible window has no
        // hittable Delete row, so any click just closes the overlay.
        let fx = Fixture {
            combo_options: vec![],
            list_rows: vec![ListRow {
                is_header: false,
                text: "only".to_string(),
                entry: Some(0),
            }],
        };
        let o = test_origin();
        let v = view(&fx, Combo::Closed, Some(9), [0.0, 0.0]);
        assert_eq!(
            hit_test(&v, 640.0, 700.0, o),
            Some(PanelAction::CloseOverlays)
        );
    }

    #[test]
    fn combo_shows_an_empty_state_with_no_matching_options() {
        let fx = Fixture {
            combo_options: vec![],
            list_rows: vec![],
        };
        let mut world = injected_world();
        let o = test_origin();
        apply(
            &mut world,
            Some(&view(&fx, Combo::Filter, None, [0.0, 0.0])),
            o,
        );
        let empty = world
            .query::<TextLabel>()
            .find(|l| l.asset_id == EMPTY_LABEL)
            .unwrap();
        assert!(empty.visible);
        assert_eq!(empty.content, "No matching types");
    }

    #[test]
    fn combo_marks_the_selected_option() {
        let fx = Fixture {
            combo_options: vec!["All".to_string(), "PointLight".to_string()],
            list_rows: vec![],
        };
        let mut world = injected_world();
        let o = test_origin();
        // The active filter is option 1; the mouse rests far from the options so
        // the selected (not hovered) branch renders it.
        let v = PanelView {
            combo: Combo::Filter,
            filter_label: ALL_LABEL,
            combo_options: &fx.combo_options,
            combo_selected: Some(1),
            combo_scroll: 0,
            list_rows: &fx.list_rows,
            list_scroll: 0,
            row_menu: None,
            selected: None,
            mouse: [0.0, 0.0],
        };
        apply(&mut world, Some(&v), o);
        let label = world
            .query::<TextLabel>()
            .find(|l| l.asset_id == combo_row_label(1))
            .unwrap();
        assert!(label.visible);
        assert_eq!(label.content, "PointLight");
    }

    #[test]
    fn picker_option_maps_to_a_scrolled_index() {
        let opts: Vec<String> = ADD_TYPES.iter().map(|s| s.to_string()).collect();
        let fx = Fixture {
            combo_options: opts,
            list_rows: vec![],
        };
        let mut v = view(&fx, Combo::Picker, None, [0.0, 0.0]);
        v.combo_scroll = 2;
        let r0 = combo_option_rect(test_origin(), 0);
        assert_eq!(
            hit_test(&v, r0[0] + 5.0, r0[1] + 5.0, test_origin()),
            Some(PanelAction::PickOption(2)),
            "row 0 maps to option `scroll + 0`"
        );
    }

    // Clicking a name row's body opens that entry's edit form; the triple-dot
    // still opens the row menu; a header row is inert.
    #[test]
    fn name_row_click_opens_the_entry_and_dots_open_the_menu() {
        let fx = Fixture {
            combo_options: vec![],
            list_rows: rows(&[(true, "PointLight", None), (false, "lamp", Some(7))]),
        };
        let o = test_origin();
        let v = view(&fx, Combo::Closed, None, [0.0, 0.0]);
        let name = list_row_rect(o, 1);
        assert_eq!(
            hit_test(&v, name[0] + 5.0, name[1] + 5.0, o),
            Some(PanelAction::OpenEntry(7)),
            "the row body opens the entry's form"
        );
        let dot = dot_rect(name);
        assert_eq!(
            hit_test(&v, dot[0] + 5.0, dot[1] + 5.0, o),
            Some(PanelAction::OpenRowMenu(7))
        );
        let hdr = list_row_rect(o, 0);
        assert_eq!(
            hit_test(&v, hdr[0] + 5.0, hdr[1] + 5.0, o),
            Some(PanelAction::Consume)
        );
    }

    // The entry whose edit form is open keeps its row highlighted.
    #[test]
    fn selected_entry_row_stays_highlighted() {
        let fx = Fixture {
            combo_options: vec![],
            list_rows: rows(&[
                (true, "PointLight", None),
                (false, "lamp", Some(0)),
                (false, "spot", Some(1)),
            ]),
        };
        let mut world = injected_world();
        let o = test_origin();
        let mut v = view(&fx, Combo::Closed, None, [0.0, 0.0]);
        v.selected = Some(0);
        apply(&mut world, Some(&v), o);
        assert_eq!(
            sprite(&world, list_row_bg(1)).tint,
            ROW_TINT_SELECTED,
            "the edited entry's row is highlighted"
        );
        assert_eq!(
            sprite(&world, list_row_bg(2)).tint,
            ROW_TINT,
            "other rows keep the plain tint"
        );
    }

    #[test]
    fn open_row_menu_resolves_delete() {
        let fx = Fixture {
            combo_options: vec![],
            list_rows: rows(&[(true, "PointLight", None), (false, "lamp", Some(3))]),
        };
        let o = test_origin();
        let v = view(&fx, Combo::Closed, Some(3), [0.0, 0.0]);
        let (_, delete) = menu_rects(o, 1);
        assert_eq!(
            hit_test(&v, delete[0] + 5.0, delete[1] + 5.0, o),
            Some(PanelAction::RowDelete)
        );
        // A click off the menu dismisses it.
        assert_eq!(
            hit_test(&v, 640.0, 700.0, o),
            Some(PanelAction::CloseOverlays)
        );
    }

    #[test]
    fn clicks_outside_the_panel_fall_through() {
        let fx = Fixture {
            combo_options: vec![],
            list_rows: vec![],
        };
        let v = view(&fx, Combo::Closed, None, [0.0, 0.0]);
        assert_eq!(hit_test(&v, 10.0, 400.0, test_origin()), None);
    }

    // Draw order is component-insertion order, and injection follows these Vecs,
    // so the floating overlays (dots, row menu) must come AFTER every row family or
    // a hovered row's fill paints over them.
    #[test]
    fn floating_overlays_are_injected_after_the_row_families() {
        let sprites = all_sprite_ids();
        let spos = |id: AssetId| sprites.iter().position(|&x| x == id).unwrap();
        let last_row_bg = (0..MAX_ROWS)
            .map(list_row_bg)
            .chain((0..MAX_ROWS).map(combo_row_bg))
            .map(spos)
            .max()
            .unwrap();
        for overlay in [DOT_BG, DOT1, DOT2, DOT3, MENU_BG, MENU_DELETE_BG] {
            assert!(
                spos(overlay) > last_row_bg,
                "{overlay:?} must draw above the row backgrounds"
            );
        }
        // The combo backing stays below its own option rows.
        let first_combo = (0..MAX_ROWS).map(combo_row_bg).map(spos).min().unwrap();
        assert!(spos(COMBO_BG) < first_combo, "combo backing under its rows");

        // The row-menu captions draw above the row labels the menu floats over.
        let labels = all_label_ids();
        let lpos = |id: AssetId| labels.iter().position(|&x| x == id).unwrap();
        let last_row_label = (0..MAX_ROWS)
            .map(list_row_label)
            .chain((0..MAX_ROWS).map(combo_row_label))
            .map(lpos)
            .max()
            .unwrap();
        assert!(
            lpos(MENU_DELETE_LABEL) > last_row_label,
            "the Delete caption draws above the row labels"
        );
    }

    // The white dots show whenever the row is hovered; the background box shows
    // only when the dots themselves are hovered, or the menu is open on that row.
    #[test]
    fn triple_dot_box_shows_only_over_dots_or_with_the_menu_open() {
        let o = test_origin();
        let fx = Fixture {
            combo_options: vec![],
            list_rows: rows(&[(true, "PointLight", None), (false, "lamp", Some(0))]),
        };
        let name = list_row_rect(o, 1);
        let dot = dot_rect(name);
        let mut world = injected_world();

        // Hover the row body (left of the dots): bare dots, no box.
        let over_body = [name[0] + PAD + asset_list::INDENT, name[1] + 5.0];
        apply(
            &mut world,
            Some(&view(&fx, Combo::Closed, None, over_body)),
            o,
        );
        assert!(sprite_visible(&world, DOT1), "white dots show on row hover");
        assert!(
            !sprite_visible(&world, DOT_BG),
            "no box on a plain row hover"
        );

        // Hover the dots: the box appears.
        let over_dots = [dot[0] + 5.0, dot[1] + 5.0];
        apply(
            &mut world,
            Some(&view(&fx, Combo::Closed, None, over_dots)),
            o,
        );
        assert!(
            sprite_visible(&world, DOT_BG),
            "box shows when dots hovered"
        );

        // Menu open: the box shows without any hover, and the menu shows.
        apply(
            &mut world,
            Some(&view(&fx, Combo::Closed, Some(0), [0.0, 0.0])),
            o,
        );
        assert!(
            sprite_visible(&world, DOT_BG),
            "box shows with the menu open"
        );
        assert!(sprite_visible(&world, DOT1), "dots show with the menu open");
        assert!(sprite_visible(&world, MENU_BG), "the row menu shows");
    }

    // Every offered add-type is a real External type whose default args cook in a
    // minimal rendering world. This is the guard the curated list leans on: a type
    // that needs a source file or a required cross-reference (Mesh, AudioClip,
    // Joint, ...) fails here and must not be listed.
    #[test]
    fn add_types_cook_with_default_args() {
        isolate_state_dir();
        for ty in picker_types() {
            // Most add types are components; Font (and future resources) are
            // addable-blank resource assets, External by construction.
            if let Some(ct) = concinnity_cook::ComponentType::parse(ty) {
                assert!(ct.addable(), "{ty} must be External / addable");
            } else {
                assert!(
                    concinnity_cook::resource_handles::ResourceAssetType::parse(ty).is_some(),
                    "{ty} must be a known component or resource asset type"
                );
            }
            let world = format!(
                "{{\"name\":\"gfx\",\"type\":\"GraphicsConfig\",\"args\":{{}}}}\n\
                 {{\"name\":\"probe\",\"type\":\"{ty}\",\"args\":{{}}}}\n"
            );
            crate::build_pipeline_from_str(&world, None)
                .unwrap_or_else(|e| panic!("{ty} must cook with default args: {e}"));
        }
    }

    // The set of offered types is exactly the addable types that cook with default
    // args AND are useful when added blank: no world-config singleton (those want an
    // edit-or-add flow), no engine-injected HUD, and no type whose value is a nested
    // array / source file it can't be given here. Enforced so a newly-registered
    // addable-and-blank-useful type is a deliberate ADD_TYPES choice, not forgotten.
    #[test]
    fn add_types_are_the_curated_blank_useful_addable_set() {
        isolate_state_dir();
        use concinnity_cook::ComponentType;
        // Types that cook blank but are deliberately NOT offered, each for a reason
        // above. Keeping this explicit means the assertion below flags anything new.
        const EXCLUDED: &[&str] = &[
            // Engine-injected HUDs (added by `cn build`, not by hand).
            "DebugHud",
            "StatHud",
            // Defined by nested content / a source the scalar form can't supply, so
            // a blank instance is inert.
            "Animation",
            "File",
            "LayoutContainer",
            "Model",
            "PropBody",
            "RigidBody",
            "Scene",
            "SceneReel",
            "ScrollPanel",
            "Spawner",
            "Story",
        ];
        // The config singletons are offered too (edit-or-add), so "offered" spans
        // both lists.
        let offered: std::collections::HashSet<&str> = picker_types().collect();
        let excluded: std::collections::HashSet<&str> = EXCLUDED.iter().copied().collect();
        for (_t, reg) in ComponentType::addable_types() {
            let ty = reg.type_name;
            let cooks = crate::build_pipeline_from_str(
                &format!(
                    "{{\"name\":\"gfx\",\"type\":\"GraphicsConfig\",\"args\":{{}}}}\n\
                     {{\"name\":\"probe\",\"type\":\"{ty}\",\"args\":{{}}}}\n"
                ),
                None,
            )
            .is_ok();
            if !cooks {
                // Needs a source / required reference: never offerable blank, and it
                // does not belong in the EXCLUDED (cooks-but-curated-out) list.
                assert!(
                    !offered.contains(ty),
                    "{ty} cannot cook blank yet is offered"
                );
                continue;
            }
            assert!(
                offered.contains(ty) ^ excluded.contains(ty),
                "{ty} cooks blank: add it to ADD_TYPES or to the EXCLUDED list (with a reason), not both/neither"
            );
        }
    }
}
