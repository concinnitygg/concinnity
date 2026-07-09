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

use crate::assets::TextAlign;
use crate::ecs::World;
use crate::ecs::asset_id::AssetId;

use super::form::{self, FieldKind, FormField};
use super::hud;
use super::widget::{self, place_sprite, point_in};

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
pub(crate) const ALL_LABEL: &str = "Assets";

// Visible rows in the body before it scrolls (shared by the grouped list and the
// combo option list).
pub(crate) const MAX_ROWS: usize = 12;

// An enum / ref field with this many variants or fewer cycles in place on click
// (fast for small sets); more than this opens a floating value dropdown anchored
// below the field (a long list is tedious to cycle through). A reference field's
// count includes its `(none)` option.
pub(crate) const CYCLE_MAX: usize = 5;

// Visible rows in an open field-value dropdown before it scrolls.
pub(crate) const MAX_DROP_ROWS: usize = 8;
// A field-value dropdown option row's height.
const DROP_ROW_H: f32 = 28.0;

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
pub(crate) const NAME_INPUT: AssetId = AssetId(PANEL + 6);
pub(crate) const FORM_TITLE: AssetId = AssetId(PANEL + 7);
pub(crate) const FORMADD_BG: AssetId = AssetId(PANEL + 8);
pub(crate) const FORMADD_LABEL: AssetId = AssetId(PANEL + 9);
pub(crate) const FORMCANCEL_BG: AssetId = AssetId(PANEL + 10);
pub(crate) const FORMCANCEL_LABEL: AssetId = AssetId(PANEL + 11);
pub(crate) const EMPTY_LABEL: AssetId = AssetId(PANEL + 12);
pub(crate) const FORM_STATUS: AssetId = AssetId(PANEL + 13);
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
// Add / edit form pool. Row 0 is the asset name (label + `NAME_INPUT`); rows
// 1..=N are the editable arg fields. `form_row_label(i)` is the caption for row
// `i`; `form_input(j)` is the text control for arg field `j` (row `j + 1`);
// `form_toggle_bg(j)` is the checkbox box for a bool arg field `j`.
pub(crate) fn form_row_label(i: usize) -> AssetId {
    AssetId(PANEL + 0x100 + i as u32)
}
pub(crate) fn form_input(j: usize) -> AssetId {
    AssetId(PANEL + 0x120 + j as u32)
}
pub(crate) fn form_toggle_bg(j: usize) -> AssetId {
    AssetId(PANEL + 0x140 + j as u32)
}
// Colour preview swatch for a colour-vector arg field `j` (drawn over the right
// end of its text control, AddForm only).
pub(crate) fn form_swatch(j: usize) -> AssetId {
    AssetId(PANEL + 0x160 + j as u32)
}
// The current-variant caption of an enum arg field `j` (drawn over its cycling
// button, which reuses `form_toggle_bg(j)`; AddForm only).
pub(crate) fn form_enum_label(j: usize) -> AssetId {
    AssetId(PANEL + 0x180 + j as u32)
}

// Which of the form's inputs holds keyboard focus (re-asserted each frame so the
// opening click cannot blur it; only the focused input re-asserts, so N text
// fields do not fight).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum FormFocus {
    Name,
    Field(usize),
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
// Add / edit form rows: a compact row per field, a label column on the left.
const FIELD_H: f32 = 30.0;
const LABEL_COL: f32 = 108.0;
// Left inset per nesting level for a flattened nested (dotted-path) field's leaf
// caption, so sub-object fields read as indented under their parent.
const NEST_INDENT: f32 = 8.0;

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
const CHECK_ON: [f32; 4] = [0.30, 0.66, 0.34, 1.0];
const CHECK_OFF: [f32; 4] = [0.30, 0.30, 0.34, 1.0];
const LABEL: [f32; 3] = [0.90, 0.90, 0.92];
const LABEL_DIM: [f32; 3] = [0.60, 0.60, 0.66];
const LABEL_WHITE: [f32; 3] = [1.0, 1.0, 1.0];
const HEADER_LABEL: [f32; 3] = [0.58, 0.66, 0.80];
const DELETE_LABEL: [f32; 3] = [0.95, 0.60, 0.58];
const ERROR_LABEL: [f32; 3] = [0.95, 0.55, 0.55];

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

// AddForm layout: a heading, then stacked rows (row 0 = name, rows 1..=N = arg
// fields), then the Add / Cancel buttons, then a status line.
fn form_rows_top() -> f32 {
    body_y() + PAD + LINE
}
// The full-width rect of form row `i` (0 = name).
fn form_row_rect(vw: f32, i: usize) -> [f32; 4] {
    [
        vw - PANEL_W,
        form_rows_top() + i as f32 * FIELD_H,
        PANEL_W,
        FIELD_H,
    ]
}
// The control (text field / checkbox area) on the right of form row `i`.
pub(crate) fn form_control_rect(vw: f32, i: usize) -> [f32; 4] {
    let r = form_row_rect(vw, i);
    [
        r[0] + LABEL_COL,
        r[1] + 2.0,
        PANEL_W - LABEL_COL - PAD,
        FIELD_H - 6.0,
    ]
}
// The checkbox box for a bool field on form row `i`.
pub(crate) fn form_toggle_rect(vw: f32, i: usize) -> [f32; 4] {
    let c = form_control_rect(vw, i);
    let s = FIELD_H - 10.0;
    [c[0], c[1], s, s]
}

// An open value dropdown for the enum / ref field on form row `row`: it floats
// directly below that field's control, aligned to it, `shown` options tall.
fn field_option_rect(vw: f32, row: usize, r: usize) -> [f32; 4] {
    let c = form_control_rect(vw, row);
    [c[0], c[1] + c[3] + r as f32 * DROP_ROW_H, c[2], DROP_ROW_H]
}
fn field_dropdown_backing(vw: f32, row: usize, shown: usize) -> [f32; 4] {
    let c = form_control_rect(vw, row);
    [c[0], c[1] + c[3], c[2], shown as f32 * DROP_ROW_H + 4.0]
}

// Whether two rects overlap (touching edges do not count).
fn rects_intersect(a: [f32; 4], b: [f32; 4]) -> bool {
    a[0] < b[0] + b[2] && b[0] < a[0] + a[2] && a[1] < b[1] + b[3] && b[1] < a[1] + a[3]
}
fn form_buttons_y(n_rows: usize) -> f32 {
    form_rows_top() + n_rows as f32 * FIELD_H + GAP
}
pub(crate) fn form_add_rect(vw: f32, n_rows: usize) -> [f32; 4] {
    let w = (PANEL_W - 2.0 * PAD - GAP) / 2.0;
    [vw - PANEL_W + PAD, form_buttons_y(n_rows), w, ROW_H]
}
pub(crate) fn form_cancel_rect(vw: f32, n_rows: usize) -> [f32; 4] {
    let a = form_add_rect(vw, n_rows);
    [a[0] + a[2] + GAP, a[1], a[2], ROW_H]
}
// The y just past the form's last element (buttons + a status line's worth of
// room), so the panel background and the click bounds can grow to contain a tall
// form instead of letting its lower rows fall outside the fixed panel.
fn form_bottom(n_rows: usize) -> f32 {
    form_buttons_y(n_rows) + ROW_H + GAP + FIELD_H
}
// The panel outer rect, grown when needed to contain the `n_rows`-row form.
fn form_panel_rect(vw: f32, n_rows: usize) -> [f32; 4] {
    let base = panel_rect(vw);
    let bottom = form_bottom(n_rows).max(base[1] + base[3]);
    [base[0], base[1], base[2], bottom - base[1]]
}
// The panel's outer rect for `view` (grown for a tall form), used for the
// background sprite and the fall-through hit-test bounds.
fn outer_rect(view: &PanelView, vw: f32) -> [f32; 4] {
    if view.mode == Mode::AddForm {
        form_panel_rect(vw, view.form_fields.len() + 1)
    } else {
        panel_rect(vw)
    }
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
    // Focus the form's name field, or its arg text field `i`.
    FocusName,
    FocusField(usize),
    // Toggle the form's bool arg field `i`.
    ToggleField(usize),
    // Advance the form's enum arg field `i` to its next variant (small sets).
    CycleField(usize),
    // Open (or, if already open, close) the value dropdown for enum / ref arg
    // field `i` (large sets, chosen over cycling).
    OpenFieldDropdown(usize),
    // Pick option `i` from the open field-value dropdown for the current field.
    PickFieldOption(usize),
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
    // The editable arg fields of the add / edit form (empty outside AddForm).
    pub form_fields: &'a [FormField],
    // Which form input holds keyboard focus.
    pub form_focus: FormFocus,
    // The form arg field whose value dropdown is open, if any (enum / ref with a
    // large variant set). Its options float below the field.
    pub field_dropdown: Option<usize>,
    // First visible option of the open value dropdown.
    pub field_dropdown_scroll: usize,
    // A validation error to surface under the form, if the last Add failed.
    pub form_error: Option<&'a str>,
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

    // An open field-value dropdown is modal over the panel, like the combo: its
    // option rows pick, anything else dismisses it.
    if view.mode == Mode::AddForm
        && let Some(open) = view.field_dropdown
        && let Some(field) = view.form_fields.get(open)
    {
        let row = open + 1;
        let total = field.variants.len();
        let scroll = view.field_dropdown_scroll.min(total.saturating_sub(1));
        for r in 0..MAX_DROP_ROWS {
            let idx = scroll + r;
            if idx >= total {
                break;
            }
            if point_in(mx, my, field_option_rect(vw, row, r)) {
                return Some(PanelAction::PickFieldOption(idx));
            }
        }
        return Some(PanelAction::CloseOverlays);
    }

    // Combo closed: clicks outside the panel fall through (the top bar handles
    // the region above `body_top`; the world gets the rest). A tall form grows the
    // bounds so its lower rows / buttons stay clickable.
    if !point_in(mx, my, outer_rect(view, vw)) {
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
            let n_rows = view.form_fields.len() + 1;
            if point_in(mx, my, form_add_rect(vw, n_rows)) {
                return Some(PanelAction::ConfirmAdd);
            }
            if point_in(mx, my, form_cancel_rect(vw, n_rows)) {
                return Some(PanelAction::CancelForm);
            }
            // The name field (row 0).
            if point_in(mx, my, form_control_rect(vw, 0)) {
                return Some(PanelAction::FocusName);
            }
            // Arg field controls: a bool toggles, an enum cycles, a text field
            // takes focus.
            for (j, f) in view.form_fields.iter().enumerate() {
                let row = j + 1;
                match f.kind {
                    FieldKind::Bool => {
                        if point_in(mx, my, form_toggle_rect(vw, row)) {
                            return Some(PanelAction::ToggleField(j));
                        }
                    }
                    FieldKind::Enum | FieldKind::Ref { .. } => {
                        if point_in(mx, my, form_control_rect(vw, row)) {
                            // A small variant set cycles in place; a large one opens
                            // a floating dropdown instead of forcing many clicks.
                            if f.variants.len() > CYCLE_MAX {
                                return Some(PanelAction::OpenFieldDropdown(j));
                            }
                            return Some(PanelAction::CycleField(j));
                        }
                    }
                    _ => {
                        if point_in(mx, my, form_control_rect(vw, row)) {
                            return Some(PanelAction::FocusField(j));
                        }
                    }
                }
            }
            // Empty form space: swallow so it does not fall through to the world.
            Some(PanelAction::Consume)
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

    place_sprite(world, PANEL_BG, outer_rect(view, vw), PANEL_BG_TINT, true);
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
        show_field(world, FILTER_INPUT, filter_input_rect(vw), true);
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
    // Its background box shows only when the dots themselves are hovered, or when
    // the menu is open on that row (a "clicked" state); a plain row-hover shows
    // the bare white dots.
    if let Some(r) = menu_row.or(hovered_row) {
        let rect = list_row_rect(vw, r);
        let over_dots = point_in(view.mouse[0], view.mouse[1], dot_rect(rect));
        let show_box = menu_row == Some(r) || over_dots;
        place_dot(world, rect, show_box);
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

    // Row 0: the asset name.
    let name_row = form_row_rect(vw, 0);
    place_left_label(
        world,
        form_row_label(0),
        [name_row[0] + PAD, name_row[1] + FIELD_H * 0.5 - 10.0],
        "name",
        LABEL,
        true,
    );
    show_field(
        world,
        NAME_INPUT,
        form_control_rect(vw, 0),
        view.form_focus == FormFocus::Name,
    );

    // An open value dropdown floats over the rows below its field. Those rows are
    // left fully hidden this frame (both the control and its caption): a covered
    // VISIBLE `TextInput` would synth an opaque background box over the dropdown (it
    // is synthesised only for visible inputs, see graphics_system/frame.rs), and a
    // long caption could overflow across it. The dropdown paints its own opaque list
    // in this space, so nothing beneath it draws.
    let drop_backing = view.field_dropdown.and_then(|open| {
        let field = view.form_fields.get(open)?;
        let total = field.variants.len();
        let scroll = view.field_dropdown_scroll.min(total.saturating_sub(1));
        let shown = total.saturating_sub(scroll).clamp(1, MAX_DROP_ROWS);
        Some(field_dropdown_backing(vw, open + 1, shown))
    });

    // Rows 1..=N: the editable arg fields (a text field or a checkbox each).
    for (j, field) in view.form_fields.iter().enumerate() {
        let row = j + 1;
        let rect = form_row_rect(vw, row);
        if drop_backing.is_some_and(|d| rects_intersect(form_control_rect(vw, row), d)) {
            continue;
        }
        // A nested (dotted-path) field shows its indented leaf name -- the full path
        // would overflow the label column.
        let depth = field.key.matches('.').count();
        let leaf = field.key.rsplit('.').next().unwrap_or(field.key.as_str());
        place_left_label(
            world,
            form_row_label(row),
            [
                rect[0] + PAD + depth as f32 * NEST_INDENT,
                rect[1] + FIELD_H * 0.5 - 10.0,
            ],
            leaf,
            LABEL,
            true,
        );
        match field.kind {
            FieldKind::Bool => {
                let t = form_toggle_rect(vw, row);
                let tint = if field.boolval { CHECK_ON } else { CHECK_OFF };
                place_sprite(world, form_toggle_bg(j), t, tint, true);
            }
            FieldKind::Enum | FieldKind::Ref { .. } => {
                // A cycling button spanning the control: its background reuses the
                // bool checkbox sprite, captioned with the current selection (an
                // enum variant, or a referenced asset name / `(none)`).
                let c = form_control_rect(vw, row);
                let hover = point_in(view.mouse[0], view.mouse[1], c);
                let tint = if hover {
                    OPTION_TINT_HOVER
                } else {
                    TYPEDROP_TINT
                };
                place_sprite(world, form_toggle_bg(j), c, tint, true);
                let value = field
                    .variants
                    .get(field.variant_idx)
                    .map(String::as_str)
                    .unwrap_or("");
                place_center_label(world, form_enum_label(j), c, value, LABEL, true);
            }
            kind => {
                let control = form_control_rect(vw, row);
                // A colour vector reserves a right-hand strip for a live preview
                // swatch and narrows its field to clear it. The swatch must sit
                // OUTSIDE the field rect: a `TextInput`'s opaque background box is
                // synthesised after every authored sprite (the swatch included, see
                // graphics_system/frame.rs), so a swatch drawn under the field would
                // be painted over.
                let swatch = matches!(kind, FieldKind::Vec { color: true, .. }).then(|| {
                    let s = FIELD_H - 12.0;
                    [
                        control[0] + control[2] - s,
                        control[1] + (control[3] - s) * 0.5,
                        s,
                        s,
                    ]
                });
                let field = match swatch {
                    Some(sw) => [control[0], control[1], sw[0] - control[0] - 4.0, control[3]],
                    None => control,
                };
                show_field(
                    world,
                    form_input(j),
                    field,
                    view.form_focus == FormFocus::Field(j),
                );
                if let Some(sw) = swatch {
                    let rgb = swatch_rgb(&field_text(world, form_input(j)));
                    place_sprite(world, form_swatch(j), sw, rgb, true);
                }
            }
        }
    }

    // Add / Cancel below the last row.
    let n_rows = view.form_fields.len() + 1;
    let add = form_add_rect(vw, n_rows);
    let cancel = form_cancel_rect(vw, n_rows);
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

    // A validation error, if the last Add was rejected.
    if let Some(err) = view.form_error {
        place_left_label(
            world,
            FORM_STATUS,
            [add[0], add[1] + ROW_H + GAP],
            err,
            ERROR_LABEL,
            true,
        );
    }

    // The open value dropdown draws last so it floats over the form below it.
    if let Some(open) = view.field_dropdown {
        layout_field_dropdown(world, view, vw, open);
    }
}

// The floating value dropdown for an open enum / ref field: an opaque backing plus
// its option rows (the current selection highlighted), reusing the combo option
// pool (idle in AddForm) and, when it overflows, the list scrollbar.
fn layout_field_dropdown(world: &mut World, view: &PanelView, vw: f32, open: usize) {
    let Some(field) = view.form_fields.get(open) else {
        return;
    };
    let row = open + 1;
    let total = field.variants.len();
    let scroll = view.field_dropdown_scroll.min(total.saturating_sub(1));
    let shown = total.saturating_sub(scroll).clamp(1, MAX_DROP_ROWS);
    place_sprite(
        world,
        COMBO_BG,
        field_dropdown_backing(vw, row, shown),
        COMBO_BG_TINT,
        true,
    );
    for r in 0..MAX_DROP_ROWS {
        let idx = scroll + r;
        if idx >= total {
            break;
        }
        let rect = field_option_rect(vw, row, r);
        let hovered = point_in(view.mouse[0], view.mouse[1], rect);
        let tint = if hovered {
            OPTION_TINT_HOVER
        } else if idx == field.variant_idx {
            OPTION_TINT_SELECTED
        } else {
            OPTION_TINT
        };
        place_sprite(world, combo_row_bg(r), rect, tint, true);
        set_row_label(
            world,
            combo_row_label(r),
            [rect[0] + PAD, rect[1] + DROP_ROW_H * 0.5 - 10.0],
            &field.variants[idx],
            LABEL,
            true,
        );
    }
    // A thin scrollbar down the dropdown's right edge when it overflows.
    if total > MAX_DROP_ROWS {
        let back = field_dropdown_backing(vw, row, shown);
        let track = [
            back[0] + back[2] - SCROLLBAR_W,
            back[1],
            SCROLLBAR_W,
            back[3],
        ];
        place_sprite(world, LIST_TRACK, track, TRACK_TINT, true);
        let frac = MAX_DROP_ROWS as f32 / total as f32;
        let thumb_h = (back[3] * frac).max(20.0);
        let max_scroll = (total - MAX_DROP_ROWS) as f32;
        let t = if max_scroll > 0.0 {
            scroll as f32 / max_scroll
        } else {
            0.0
        };
        let thumb_y = back[1] + t * (back[3] - thumb_h);
        place_sprite(
            world,
            LIST_THUMB,
            [track[0], thumb_y, SCROLLBAR_W, thumb_h],
            THUMB_TINT,
            true,
        );
    }
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
// (and `inject.rs` can create exactly this set). THE ORDER OF THIS VEC IS THE DRAW
// ORDER: `inject.rs` adds the panel's Sprites in this sequence, and the overlay
// draws components in insertion (component-column) order -- NOT by asset id -- so
// later entries paint on top. Bottom-to-top: panel background, header chrome, the
// combo backing (under its option rows), the row families, then the floating
// overlays (scrollbar, triple-dot, row menu), which must sit ABOVE the row
// backgrounds so a hovered row's fill cannot cover them.
pub(crate) fn all_sprite_ids() -> Vec<AssetId> {
    let mut ids = vec![
        PANEL_BG,
        PLUS_BG,
        TYPEDROP_BG,
        FORMADD_BG,
        FORMCANCEL_BG,
        COMBO_BG,
    ];
    ids.extend((0..MAX_ROWS).map(list_row_bg));
    ids.extend((0..MAX_ROWS).map(combo_row_bg));
    // Form field checkboxes + colour swatches (AddForm only, no list overlap).
    ids.extend((0..form::MAX_FIELDS).map(form_toggle_bg));
    ids.extend((0..form::MAX_FIELDS).map(form_swatch));
    ids.extend([
        LIST_TRACK,
        LIST_THUMB,
        DOT_BG,
        DOT1,
        DOT2,
        DOT3,
        MENU_BG,
        MENU_EDIT_BG,
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
        PLUS_LABEL,
        TYPEDROP_LABEL,
        FORMADD_LABEL,
        FORMCANCEL_LABEL,
        FORM_TITLE,
        EMPTY_LABEL,
    ];
    ids.extend((0..MAX_ROWS).map(list_row_label));
    ids.extend((0..MAX_ROWS).map(combo_row_label));
    // Form row captions (name row 0 + one per arg field) and the status line.
    ids.extend((0..=form::MAX_FIELDS).map(form_row_label));
    // Enum arg-field value captions (drawn over their cycling buttons).
    ids.extend((0..form::MAX_FIELDS).map(form_enum_label));
    ids.push(FORM_STATUS);
    ids.extend([MENU_EDIT_LABEL, MENU_DELETE_LABEL]);
    ids
}

// Every typed field the panel injects: the combo filter, the form name, and the
// form's arg-field text inputs.
pub(crate) fn all_field_ids() -> Vec<AssetId> {
    let mut ids = vec![FILTER_INPUT, NAME_INPUT];
    ids.extend((0..form::MAX_FIELDS).map(form_input));
    ids
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
        hide_field(world, id);
    }
}

// -- Element mutation helpers -------------------------------------------------
//
// The reserved-id lookups (`widget::sprite_mut` / `label_mut` / `input_mut`) and
// `place_sprite` / visibility setters live in `widget.rs`, shared with `hud.rs`.
// These wrap them with the panel's label / field conventions.

fn place_center_label(
    world: &mut World,
    id: AssetId,
    rect: [f32; 4],
    content: &str,
    color: [f32; 3],
    visible: bool,
) {
    if let Some(l) = widget::label_mut(world, id) {
        l.x = rect[0] + rect[2] * 0.5;
        l.y = rect[1] + rect[3] * 0.5 - 10.0;
        l.align = TextAlign::Center;
        l.color = color;
        l.visible = visible;
        l.content = content.to_string();
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

// Position + show a field, setting its focus explicitly. The focused field
// re-asserts `focused = true` every frame: the click that opened the mode (on the
// "+" button, say) lands outside the field, so the engine's text-input system
// would otherwise blur it that frame; re-asserting keeps it typable. Non-focused
// fields are shown with `focused = false` so several fields can coexist without
// fighting for the keyboard. The content is not touched here (only on a
// transition), so what is typed stands.
fn show_field(world: &mut World, id: AssetId, rect: [f32; 4], focused: bool) {
    if let Some(t) = widget::input_mut(world, id) {
        t.x = rect[0];
        t.y = rect[1];
        t.width = rect[2];
        t.height = rect[3];
        t.visible = true;
        t.focused = focused;
    }
}

// Hide + blur a typed field.
fn hide_field(world: &mut World, id: AssetId) {
    if let Some(t) = widget::input_mut(world, id) {
        t.visible = false;
        t.focused = false;
    }
}

// Set a field's text + caret and give it focus (a mode transition; the hook
// calls this so the field is ready to type into immediately).
pub(crate) fn focus_field_with(world: &mut World, id: AssetId, content: &str) {
    if let Some(t) = widget::input_mut(world, id) {
        t.content = content.to_string();
        t.caret = content.chars().count();
        t.focused = true;
        t.visible = true;
    }
}

// Seed a field's text + caret without changing focus (the layout decides focus
// from `FormFocus`). Used to pre-fill the form's arg inputs on open.
pub(crate) fn seed_field(world: &mut World, id: AssetId, content: &str) {
    if let Some(t) = widget::input_mut(world, id) {
        t.content = content.to_string();
        t.caret = content.chars().count();
    }
}

// Read a field's current text.
pub(crate) fn field_text(world: &World, id: AssetId) -> String {
    widget::input(world, id)
        .map(|t| t.content.clone())
        .unwrap_or_default()
}

// Parse a colour field's live text ("r, g, b" / "r, g, b, a") into an opaque RGB
// tint for its preview swatch, clamped to the displayable 0..=1 range. Falls back
// to a neutral dark swatch until three components parse, so the swatch is always
// visible while typing.
fn swatch_rgb(text: &str) -> [f32; 4] {
    let nums: Vec<f32> = text
        .split(|c: char| c == ',' || c.is_whitespace())
        .filter(|s| !s.is_empty())
        .filter_map(|s| s.parse::<f32>().ok())
        .take(3)
        .collect();
    if nums.len() < 3 {
        return [0.15, 0.15, 0.18, 1.0];
    }
    [
        nums[0].clamp(0.0, 1.0),
        nums[1].clamp(0.0, 1.0),
        nums[2].clamp(0.0, 1.0),
        1.0,
    ]
}

// Whether the cursor is over the scrollable body area (for wheel scrolling).
pub(crate) fn cursor_over_body(mx: f32, my: f32, vw: f32) -> bool {
    let p = panel_rect(vw);
    mx >= p[0] && mx < p[0] + p[2] && my >= body_y() && my < p[1] + p[3]
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::assets::{Sprite, TextInput, TextLabel};

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
            form_fields: &[],
            form_focus: FormFocus::Name,
            field_dropdown: None,
            field_dropdown_scroll: 0,
            form_error: None,
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
        // No arg fields -> one row (the name).
        let add = form_add_rect(1280.0, 1);
        let cancel = form_cancel_rect(1280.0, 1);
        assert_eq!(
            hit_test(&v, add[0] + 5.0, add[1] + 5.0, 1280.0),
            Some(PanelAction::ConfirmAdd)
        );
        assert_eq!(
            hit_test(&v, cancel[0] + 5.0, cancel[1] + 5.0, 1280.0),
            Some(PanelAction::CancelForm)
        );
    }

    // A full form must not overflow the panel: its Add button stays clickable
    // (inside the grown bounds) and within the enlarged background.
    #[test]
    fn a_full_form_keeps_its_buttons_inside_the_panel() {
        let fields: Vec<FormField> = (0..form::MAX_FIELDS)
            .map(|i| FormField {
                key: format!("f{i}"),
                kind: FieldKind::Float,
                initial: "0".into(),
                boolval: false,
                variants: Vec::new(),
                variant_idx: 0,
            })
            .collect();
        let v = PanelView {
            mode: Mode::AddForm,
            combo: Combo::Closed,
            filter_label: ALL_LABEL,
            combo_options: &[],
            combo_selected: None,
            combo_scroll: 0,
            list_rows: &[],
            list_scroll: 0,
            row_menu: None,
            form_title: "New X",
            form_fields: &fields,
            form_focus: FormFocus::Name,
            field_dropdown: None,
            field_dropdown_scroll: 0,
            form_error: None,
            mouse: [0.0, 0.0],
        };
        let vw = 1280.0;
        let n_rows = fields.len() + 1;
        let add = form_add_rect(vw, n_rows);
        assert_eq!(
            hit_test(&v, add[0] + 5.0, add[1] + 5.0, vw),
            Some(PanelAction::ConfirmAdd),
            "the Add button of a tall form still resolves (grown bounds)"
        );
        let outer = outer_rect(&v, vw);
        assert!(
            add[1] + add[3] <= outer[1] + outer[3],
            "the Add button sits inside the grown panel background"
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
    fn name_field_click_focuses_the_name_for_the_text_system() {
        let fx = Fixture {
            combo_options: vec![],
            list_rows: vec![],
        };
        let v = view(&fx, Mode::AddForm, Combo::Closed, None, [0.0, 0.0]);
        let f = form_control_rect(1280.0, 0);
        assert_eq!(
            hit_test(&v, f[0] + 5.0, f[1] + 5.0, 1280.0),
            Some(PanelAction::FocusName)
        );
    }

    // Draw order is component-insertion order, and injection follows these Vecs,
    // so the floating overlays (dots, row menu) must come AFTER every row family or
    // a hovered row's fill paints over them (the bug this round fixed).
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
        for overlay in [
            DOT_BG,
            DOT1,
            DOT2,
            DOT3,
            MENU_BG,
            MENU_EDIT_BG,
            MENU_DELETE_BG,
        ] {
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
        for cap in [MENU_EDIT_LABEL, MENU_DELETE_LABEL] {
            assert!(lpos(cap) > last_row_label, "{cap:?} above the row labels");
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

    fn sprite_visible(world: &World, id: AssetId) -> bool {
        world
            .query::<Sprite>()
            .find(|s| s.asset_id == id)
            .unwrap()
            .visible
    }

    // Build an AddForm view over a single arg field, for the swatch tests.
    fn form_view<'a>(fields: &'a [FormField]) -> PanelView<'a> {
        PanelView {
            mode: Mode::AddForm,
            combo: Combo::Closed,
            filter_label: ALL_LABEL,
            combo_options: &[],
            combo_selected: None,
            combo_scroll: 0,
            list_rows: &[],
            list_scroll: 0,
            row_menu: None,
            form_title: "New asset",
            form_fields: fields,
            form_focus: FormFocus::Name,
            field_dropdown: None,
            field_dropdown_scroll: 0,
            form_error: None,
            mouse: [0.0, 0.0],
        }
    }

    // A ref field with `n` options past `(none)`, for the dropdown tests. With
    // `n > CYCLE_MAX` it opens a dropdown rather than cycling.
    fn ref_field(n: usize) -> FormField {
        let mut variants = vec![form::NONE_LABEL.to_string()];
        variants.extend((0..n).map(|i| format!("tex_{i}")));
        FormField {
            key: "texture".into(),
            kind: FieldKind::Ref { target: "Texture" },
            initial: String::new(),
            boolval: false,
            variants,
            variant_idx: 0,
        }
    }

    // A colour-vector arg field renders its text control plus a live preview
    // swatch tinted from the field's current text.
    #[test]
    fn colour_vector_field_shows_a_live_swatch() {
        let vw = 1280.0;
        let mut world = injected_world();
        for t in world.query_mut::<TextInput>() {
            if t.asset_id == form_input(0) {
                t.content = "1, 0, 0".into();
            }
        }
        let fields = [FormField {
            key: "color".into(),
            kind: FieldKind::Vec {
                len: 3,
                color: true,
            },
            initial: "1, 0, 0".into(),
            boolval: false,
            variants: Vec::new(),
            variant_idx: 0,
        }];
        apply(&mut world, Some(&form_view(&fields)), vw);
        let sw = world
            .query::<Sprite>()
            .find(|s| s.asset_id == form_swatch(0))
            .unwrap();
        assert!(sw.visible, "the colour field draws a swatch");
        assert_eq!(sw.tint, [1.0, 0.0, 0.0, 1.0], "swatch tint is the RGB text");
        let ti = world
            .query::<TextInput>()
            .find(|t| t.asset_id == form_input(0))
            .unwrap();
        assert!(ti.visible, "the editable text field still shows");
        // The field's opaque background box is drawn after every authored sprite
        // (including the swatch), so the swatch must sit clear of the field rect or
        // it would be painted over and never seen.
        assert!(
            sw.x >= ti.x + ti.width,
            "the swatch sits outside the field's opaque background box"
        );
    }

    // A plain (non-colour) vector shows its text field but no swatch.
    #[test]
    fn plain_vector_field_has_no_swatch() {
        let vw = 1280.0;
        let mut world = injected_world();
        let fields = [FormField {
            key: "position".into(),
            kind: FieldKind::Vec {
                len: 3,
                color: false,
            },
            initial: "0, 0, 0".into(),
            boolval: false,
            variants: Vec::new(),
            variant_idx: 0,
        }];
        apply(&mut world, Some(&form_view(&fields)), vw);
        assert!(
            !world
                .query::<Sprite>()
                .find(|s| s.asset_id == form_swatch(0))
                .unwrap()
                .visible,
            "a non-colour vector has no swatch"
        );
        assert!(
            world
                .query::<TextInput>()
                .find(|t| t.asset_id == form_input(0))
                .unwrap()
                .visible
        );
    }

    // An enum arg field renders as a cycling button (reusing the toggle sprite)
    // captioned with the current variant, keeps its text input hidden, and
    // hit-tests to a cycle.
    #[test]
    fn enum_field_renders_a_cycling_button_and_hit_tests_to_cycle() {
        let vw = 1280.0;
        let mut world = injected_world();
        let fields = [FormField {
            key: "align".into(),
            kind: FieldKind::Enum,
            initial: String::new(),
            boolval: false,
            variants: vec!["left".into(), "center".into(), "right".into()],
            variant_idx: 1,
        }];
        let v = form_view(&fields);
        apply(&mut world, Some(&v), vw);
        assert!(
            sprite_visible(&world, form_toggle_bg(0)),
            "cycling button shows"
        );
        let cap = world
            .query::<TextLabel>()
            .find(|l| l.asset_id == form_enum_label(0))
            .unwrap();
        assert!(
            cap.visible && cap.content == "center",
            "shows the current variant"
        );
        assert!(
            !world
                .query::<TextInput>()
                .find(|t| t.asset_id == form_input(0))
                .unwrap()
                .visible,
            "an enum has no editable text field"
        );
        // A click on the control cycles rather than focusing.
        let c = form_control_rect(vw, 1);
        assert_eq!(
            hit_test(&v, c[0] + 5.0, c[1] + 5.0, vw),
            Some(PanelAction::CycleField(0))
        );
    }

    // A reference field renders through the same cycle-button path as an enum,
    // showing the current selection and hit-testing to a cycle.
    #[test]
    fn ref_field_renders_a_cycling_button_and_hit_tests_to_cycle() {
        let vw = 1280.0;
        let mut world = injected_world();
        let fields = [FormField {
            key: "texture".into(),
            kind: FieldKind::Ref { target: "Texture" },
            initial: String::new(),
            boolval: false,
            variants: vec!["(none)".into(), "grass".into()],
            variant_idx: 1,
        }];
        let v = form_view(&fields);
        apply(&mut world, Some(&v), vw);
        assert!(
            sprite_visible(&world, form_toggle_bg(0)),
            "cycle button shows"
        );
        let cap = world
            .query::<TextLabel>()
            .find(|l| l.asset_id == form_enum_label(0))
            .unwrap();
        assert!(
            cap.visible && cap.content == "grass",
            "shows the referenced name"
        );
        let c = form_control_rect(vw, 1);
        assert_eq!(
            hit_test(&v, c[0] + 5.0, c[1] + 5.0, vw),
            Some(PanelAction::CycleField(0))
        );
    }

    // A ref field with more than `CYCLE_MAX` options opens a dropdown on click
    // rather than cycling (cycling a long list is tedious); a small one still
    // cycles.
    #[test]
    fn large_choice_field_opens_a_dropdown_small_one_cycles() {
        let vw = 1280.0;
        let big = [ref_field(CYCLE_MAX + 1)]; // (none) + CYCLE_MAX+1 = over the cap
        let v = form_view(&big);
        let c = form_control_rect(vw, 1);
        assert_eq!(
            hit_test(&v, c[0] + 5.0, c[1] + 5.0, vw),
            Some(PanelAction::OpenFieldDropdown(0)),
            "a large variant set opens a dropdown"
        );
        let small = [ref_field(1)]; // (none) + 1 = 2 variants, within the cap
        let vs = form_view(&small);
        assert_eq!(
            hit_test(&vs, c[0] + 5.0, c[1] + 5.0, vw),
            Some(PanelAction::CycleField(0)),
            "a small variant set still cycles"
        );
    }

    // An open value dropdown is modal: its option rows pick, anything else closes.
    #[test]
    fn open_field_dropdown_picks_options_and_is_modal() {
        let vw = 1280.0;
        let fields = [ref_field(CYCLE_MAX + 3)];
        let mut v = form_view(&fields);
        v.field_dropdown = Some(0);
        // Option row 2 (index 2 into variants) picks that option.
        let opt = field_option_rect(vw, 1, 2);
        assert_eq!(
            hit_test(&v, opt[0] + 5.0, opt[1] + 5.0, vw),
            Some(PanelAction::PickFieldOption(2))
        );
        // A click off the option list dismisses it (e.g. up on the name row).
        let name = form_control_rect(vw, 0);
        assert_eq!(
            hit_test(&v, name[0] + 5.0, name[1] + 5.0, vw),
            Some(PanelAction::CloseOverlays)
        );
    }

    // A scrolled dropdown maps a visible row to `scroll + row`.
    #[test]
    fn open_field_dropdown_maps_a_scrolled_row() {
        let vw = 1280.0;
        let fields = [ref_field(20)];
        let mut v = form_view(&fields);
        v.field_dropdown = Some(0);
        v.field_dropdown_scroll = 3;
        let r0 = field_option_rect(vw, 1, 0);
        assert_eq!(
            hit_test(&v, r0[0] + 5.0, r0[1] + 5.0, vw),
            Some(PanelAction::PickFieldOption(3)),
            "visible row 0 is option `scroll + 0`"
        );
    }

    // Laying out an open dropdown shows its backing + option rows and hides the
    // form controls it floats over (so their opaque boxes cannot paint over it),
    // while leaving the opening field's own button visible.
    #[test]
    fn open_dropdown_hides_covered_controls_and_shows_options() {
        let vw = 1280.0;
        let mut world = injected_world();
        let fields = [
            ref_field(CYCLE_MAX + 2),
            FormField {
                key: "intensity".into(),
                kind: FieldKind::Float,
                initial: "1".into(),
                boolval: false,
                variants: Vec::new(),
                variant_idx: 0,
            },
        ];
        let mut v = form_view(&fields);
        v.field_dropdown = Some(0);
        apply(&mut world, Some(&v), vw);

        // The dropdown backing and its first option row (the current `(none)`)
        // show.
        assert!(sprite_visible(&world, COMBO_BG), "dropdown backing shows");
        assert!(sprite_visible(&world, combo_row_bg(0)), "option row shows");
        let opt0 = world
            .query::<TextLabel>()
            .find(|l| l.asset_id == combo_row_label(0))
            .unwrap();
        assert!(opt0.visible && opt0.content == form::NONE_LABEL);
        // The opening ref field's own button stays visible (dropdown is below it).
        assert!(sprite_visible(&world, form_toggle_bg(0)));
        // The following float field's whole row is hidden while covered: a visible
        // input would synth an opaque box over the list, and a long caption could
        // overflow across it, so both the text field and its caption are suppressed.
        assert!(
            !world
                .query::<TextInput>()
                .find(|t| t.asset_id == form_input(1))
                .unwrap()
                .visible,
            "the covered text field is hidden while the dropdown is open"
        );
        assert!(
            !world
                .query::<TextLabel>()
                .find(|l| l.asset_id == form_row_label(2))
                .unwrap()
                .visible,
            "the covered row's caption is hidden too (no overflow over the list)"
        );
    }

    // Every offered add-type is a real External type whose default args cook in a
    // minimal rendering world. This is the guard the curated list leans on: a type
    // that needs a source file or a required cross-reference (Mesh, AudioClip,
    // Joint, ...) fails here and must not be listed.
    #[test]
    fn add_types_cook_with_default_args() {
        for ty in picker_types() {
            let ct = crate::ecs::ComponentType::parse(ty)
                .unwrap_or_else(|| panic!("{ty} is a real component type"));
            assert!(ct.addable(), "{ty} must be External / addable");
            let world = format!(
                "{{\"name\":\"gfx\",\"type\":\"GraphicsConfig\",\"args\":{{}}}}\n\
                 {{\"name\":\"probe\",\"type\":\"{ty}\",\"args\":{{}}}}\n"
            );
            concinnity_app::build_pipeline_from_str(&world, None)
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
        use crate::ecs::ComponentType;
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
            let cooks = concinnity_app::build_pipeline_from_str(
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

    // The white dots show whenever the row is hovered; the background box shows
    // only when the dots themselves are hovered, or the menu is open on that row.
    #[test]
    fn triple_dot_box_shows_only_over_dots_or_with_the_menu_open() {
        let vw = 1280.0;
        let fx = Fixture {
            combo_options: vec![],
            list_rows: rows(&[(true, "PointLight", None), (false, "lamp", Some(0))]),
        };
        let name = list_row_rect(vw, 1);
        let dot = dot_rect(name);
        let mut world = injected_world();

        // Hover the row body (left of the dots): bare dots, no box.
        let over_body = [name[0] + PAD + INDENT, name[1] + 5.0];
        apply(
            &mut world,
            Some(&view(&fx, Mode::List, Combo::Closed, None, over_body)),
            vw,
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
            Some(&view(&fx, Mode::List, Combo::Closed, None, over_dots)),
            vw,
        );
        assert!(
            sprite_visible(&world, DOT_BG),
            "box shows when dots hovered"
        );

        // Menu open: the box shows without any hover, and the menu shows.
        apply(
            &mut world,
            Some(&view(&fx, Mode::List, Combo::Closed, Some(0), [0.0, 0.0])),
            vw,
        );
        assert!(
            sprite_visible(&world, DOT_BG),
            "box shows with the menu open"
        );
        assert!(sprite_visible(&world, DOT1), "dots show with the menu open");
        assert!(sprite_visible(&world, MENU_BG), "the row menu shows");
    }
}
