// src/editor/panel.rs
//
// The editor "Assets" panel: every asset of the expanded world as one
// collapsible tree grouped by origin (built by `asset_tree.rs`), a search field
// over it, and the per-row editor-session hide / lock toggles. Like the rest of
// the editor HUD it is plain `Sprite` / `TextLabel` / `TextInput` components at
// reserved ids (injected by `inject.rs`), driven each frame by the editor hook --
// nothing here reaches the shipped runtime. This module owns the panel's pure
// geometry, its click resolution, and the per-frame layout that shows /
// positions the elements; the hook owns the state and the option list.
//
// The panel is a floating column: a draggable title bar ("Assets") across its
// top, defaulting to below the top bar's buttons; the hook owns its position and
// clamps a drag so the panel stays fully on screen. Under the title bar the
// header is a square "+" (add) button and the search field, then a status line
// (the asset count, or a cook failure), then the tree.
//
// Clicking a name selects the asset in the viewport and opens its add / edit
// form -- a separate floating panel (`form_panel.rs`). An asset the build
// generates has no world.jsonl line of its own; its form is seeded from the
// entry the expansion produced and only confirming it appends that line, which
// then overrides the expansion. The passes that emit unconditionally (menu,
// story, and prefab primitives) cannot be overridden that way, so their rows
// select but open no form. Hovering a name also reveals a triple-dot button
// opening a small Delete menu.
//
// While the type picker is open, the "+" becomes a gray "X" that returns to the
// tree and the search field narrows the picker's option list, which floats over
// the body; picking a type opens the add form.

use std::collections::BTreeSet;
use std::sync::OnceLock;

use concinnity_cook::ComponentType;
use concinnity_cook::resource_handles::ResourceAssetType;

use crate::assets::TextAlign;
use crate::ecs::World;
use crate::ecs::asset_id::AssetId;

use super::asset_tree::{Badge, TreeRow};
use super::hud;
use super::registry::{self, PanelKey};
use super::selection::Selection;
use super::theme;
use super::widget::{self, place_rounded, place_sprite, point_in};

// The addable asset types the "+" picker offers for a plain add: every type
// flagged `useful_blank` in the registry (both the component and resource-asset
// lists). The flag marks a type that (a) recompiles cleanly when added with
// default args and (b) is useful when added blank; the choices are guarded here
// by `add_types_cook_with_default_args` (runs every offered type through the
// real cook pipeline) and `add_types_are_the_curated_blank_useful_addable_set`
// (every addable-and-cooks-blank type must be flagged or in that test's
// EXCLUDED list with a reason, so a newly registered type is a deliberate
// choice, never a silent omission). Types that cook blank but stay unflagged:
// world-config singletons want the edit-or-add flow below, engine-injected
// HUDs (DebugHud / StatHud) are added by `cn build`, and types defined by a
// nested array or a source file (Model, Scene, Story, ...) are inert until the
// nested / source form controls exist. Types that cannot even cook blank --
// needing a source or a required cross-reference (Mesh / AudioClip / Joint) --
// can never be flagged.
pub(crate) fn add_types() -> impl Iterator<Item = &'static str> {
    static TYPES: OnceLock<Vec<&'static str>> = OnceLock::new();
    TYPES
        .get_or_init(|| {
            let components = ComponentType::all()
                .iter()
                .filter(|t| t.useful_blank())
                .map(|t| t.as_str());
            let resources = ResourceAssetType::all()
                .iter()
                .filter(|t| t.useful_blank())
                .map(|t| t.as_str());
            components.chain(resources).collect()
        })
        .iter()
        .copied()
}

// The world-config singletons (the registry's `singleton` flag on declarable
// types): exactly one instance belongs to a world. They are offered in the "+"
// picker alongside `add_types`, but picking one EDITS the world's existing
// instance when it has one and only ADDS when it does not (the hook's
// `open_form` is handed the existing entry's index) -- an edit-or-add flow,
// never a blind second append. Held apart from `add_types` so the plain add
// path can keep assuming multi-instance. Like the addables, each must cook
// blank (guarded).
pub(crate) fn config_types() -> impl Iterator<Item = &'static str> {
    static TYPES: OnceLock<Vec<&'static str>> = OnceLock::new();
    TYPES
        .get_or_init(|| {
            ComponentType::all()
                .iter()
                .filter(|t| t.singleton() && t.addable())
                .map(|t| t.as_str())
                .collect()
        })
        .iter()
        .copied()
}

// Whether `ty` is a world-config singleton (edit-or-add rather than blind append).
pub(crate) fn is_singleton(ty: &str) -> bool {
    ComponentType::parse(ty).is_some_and(|t| t.singleton())
}

// Every type the "+" picker offers: the multi-instance addables plus the config
// singletons.
pub(crate) fn picker_types() -> impl Iterator<Item = &'static str> {
    add_types().chain(config_types())
}

// Reserved asset-id family for the panel. The asset-id VALUE does not affect
// draw order (the overlay draws in component-insertion order, not by id);
// z-order is set by the sequence in `all_sprite_ids` / `all_label_ids`, which
// `inject.rs` inserts in that order. The id values only need to be distinct.
const PANEL: u32 = registry::base(PanelKey::Assets);
pub(crate) const PANEL_BG: AssetId = AssetId(PANEL);
pub(crate) const PLUS_BG: AssetId = AssetId(PANEL + 1);
pub(crate) const PLUS_LABEL: AssetId = AssetId(PANEL + 2);
pub(crate) const SEARCH_INPUT: AssetId = AssetId(PANEL + 3);
pub(crate) const STATUS_LABEL: AssetId = AssetId(PANEL + 4);
pub(crate) const EMPTY_LABEL: AssetId = AssetId(PANEL + 5);
pub(crate) const LIST_TRACK: AssetId = AssetId(PANEL + 6);
pub(crate) const LIST_THUMB: AssetId = AssetId(PANEL + 7);
pub(crate) const PICKER_BG: AssetId = AssetId(PANEL + 8);
// The draggable title bar's heading (the bar is the panel surface itself).
pub(crate) const TITLE_LABEL: AssetId = AssetId(PANEL + 9);
// The "X" close button in the title bar's top-right corner.
pub(crate) const CLOSE_BG: AssetId = AssetId(PANEL + 10);
pub(crate) const CLOSE_LABEL: AssetId = AssetId(PANEL + 11);
pub(crate) const DOT_BG: AssetId = AssetId(PANEL + 12);
pub(crate) const DOT1: AssetId = AssetId(PANEL + 13);
pub(crate) const DOT2: AssetId = AssetId(PANEL + 14);
pub(crate) const DOT3: AssetId = AssetId(PANEL + 15);
pub(crate) const MENU_BG: AssetId = AssetId(PANEL + 16);
pub(crate) const MENU_DELETE_BG: AssetId = AssetId(PANEL + 17);
pub(crate) const MENU_DELETE_LABEL: AssetId = AssetId(PANEL + 18);

pub(crate) fn row_bg(slot: usize) -> AssetId {
    AssetId(PANEL + 0x20 + slot as u32)
}
pub(crate) fn name_label(slot: usize) -> AssetId {
    AssetId(PANEL + 0x40 + slot as u32)
}
pub(crate) fn type_label(slot: usize) -> AssetId {
    AssetId(PANEL + 0x60 + slot as u32)
}
pub(crate) fn eye_box(slot: usize) -> AssetId {
    AssetId(PANEL + 0x80 + slot as u32)
}
pub(crate) fn lock_box(slot: usize) -> AssetId {
    AssetId(PANEL + 0xA0 + slot as u32)
}
pub(crate) fn picker_row_bg(slot: usize) -> AssetId {
    AssetId(PANEL + 0xC0 + slot as u32)
}
pub(crate) fn picker_row_label(slot: usize) -> AssetId {
    AssetId(PANEL + 0xE0 + slot as u32)
}

// Geometry, in window pixels. Every rect derives from the panel's origin `o`
// (its title bar's top-left corner), so dragging the title bar moves the whole
// panel; the hook owns the origin.
pub(crate) const PANEL_W: f32 = 500.0;
const PAD: f32 = 10.0;
const HEADER_H: f32 = 36.0;
const STATUS_H: f32 = 20.0;
pub(crate) const ROW_H: f32 = 26.0;
const SCROLLBAR_W: f32 = 5.0;
const GAP: f32 = 6.0;
// The eye / lock toggle squares.
const BOX_SIZE: f32 = 14.0;
// An asset row's name is inset under its group header.
const ASSET_INDENT: f32 = 24.0;
// Visible rows in the body before it scrolls, shared by the tree and the
// picker's option list.
pub(crate) const ROW_POOL: usize = 14;
// The asset name, and the type that reads right-aligned beside it (the
// triple-dot takes the type's slot over while the row is hovered).
const MAX_NAME_CHARS: usize = 24;
const MAX_TYPE_CHARS: usize = 18;
// The triple-dot button on a hovered name row.
const DOT_SZ: f32 = 20.0;
// The floating Delete menu.
const MENU_W: f32 = 132.0;
const MENU_ROW_H: f32 = 26.0;

const PLUS_TINT: [f32; 4] = [0.20, 0.44, 0.30, 1.0];
const CANCEL_TINT: [f32; 4] = [0.30, 0.30, 0.34, 1.0];
const ROW_TINT: [f32; 4] = [0.0, 0.0, 0.0, 0.0];
const ROW_TINT_HOVER: [f32; 4] = theme::HOVER_TINT;
const ROW_TINT_SELECTED: [f32; 4] = theme::SELECTED_TINT;
const TRACK_TINT: [f32; 4] = [0.12, 0.12, 0.15, 0.9];
const THUMB_TINT: [f32; 4] = [0.40, 0.44, 0.56, 0.95];
// The eye box: green while the asset renders, dim once hidden.
const EYE_TINT_ON: [f32; 4] = [0.30, 0.66, 0.34, 1.0];
const EYE_TINT_OFF: [f32; 4] = [0.30, 0.30, 0.34, 1.0];
// The lock box: amber while locked, dim while pickable.
const LOCK_TINT_ON: [f32; 4] = [0.78, 0.56, 0.22, 1.0];
const LOCK_TINT_OFF: [f32; 4] = [0.30, 0.30, 0.34, 1.0];
const PICKER_BG_TINT: [f32; 4] = [0.09, 0.09, 0.12, 1.0];
const OPTION_TINT: [f32; 4] = [0.16, 0.16, 0.20, 0.0];
const OPTION_TINT_HOVER: [f32; 4] = theme::HOVER_TINT;
const DOT_BG_TINT: [f32; 4] = [0.30, 0.34, 0.46, 0.95];
const DOT_TINT: [f32; 4] = [0.90, 0.92, 0.96, 1.0];
const MENU_BG_TINT: [f32; 4] = [0.15, 0.15, 0.19, 1.0];
const MENU_ROW_TINT: [f32; 4] = [0.0, 0.0, 0.0, 0.0];
const MENU_ROW_HOVER: [f32; 4] = theme::HOVER_TINT;
const LABEL: [f32; 3] = theme::LABEL;
const LABEL_DIM: [f32; 3] = theme::LABEL_DIM;
const LABEL_WHITE: [f32; 3] = [1.0, 1.0, 1.0];
const DELETE_LABEL: [f32; 3] = [0.95, 0.60, 0.58];
const ERROR_LABEL: [f32; 3] = [0.95, 0.55, 0.55];

// Per-badge type-label colours, so provenance still reads at a glance now that
// the row's right slot carries the asset type rather than a badge caption.
fn badge_color(badge: Badge) -> [f32; 3] {
    match badge {
        Badge::Authored => theme::LABEL_DIM,
        Badge::Imported => [0.45, 0.72, 0.62],
        Badge::Injected => [0.70, 0.58, 0.88],
    }
}

// Where the panel sits until the user drags it: right-aligned below the top bar.
pub(crate) fn default_origin(vw: f32) -> [f32; 2] {
    [vw - PANEL_W, hud::body_top()]
}

// The panel's fixed footprint, for the hook's drag clamp.
pub(crate) fn size() -> [f32; 2] {
    [
        PANEL_W,
        widget::TITLE_H + HEADER_H + STATUS_H + ROW_POOL as f32 * ROW_H + PAD,
    ]
}

pub(crate) fn panel_rect(o: [f32; 2]) -> [f32; 4] {
    let s = size();
    [o[0], o[1], s[0], s[1]]
}

// The draggable title bar across the panel top.
pub(crate) fn title_rect(o: [f32; 2]) -> [f32; 4] {
    [o[0], o[1], PANEL_W, widget::TITLE_H]
}

fn header_y(o: [f32; 2]) -> f32 {
    o[1] + widget::TITLE_H
}

// The square "+" add button at the header's left.
pub(crate) fn plus_rect(o: [f32; 2]) -> [f32; 4] {
    let h = HEADER_H - 8.0;
    [o[0] + PAD, header_y(o) + 4.0, h, h]
}

// The search field, filling the header row right of the "+".
pub(crate) fn search_rect(o: [f32; 2]) -> [f32; 4] {
    let p = plus_rect(o);
    [
        p[0] + p[2] + GAP,
        p[1],
        PANEL_W - PAD - (p[2] + GAP) - PAD,
        p[3],
    ]
}

// Where the tree body begins: below the header and the status line.
fn list_top(o: [f32; 2]) -> f32 {
    header_y(o) + HEADER_H + STATUS_H
}

// Visible row `slot` (0-based within the scroll window), stopping short of the
// scrollbar.
pub(crate) fn row_rect(o: [f32; 2], slot: usize) -> [f32; 4] {
    [
        o[0],
        list_top(o) + slot as f32 * ROW_H,
        PANEL_W - SCROLLBAR_W - 2.0,
        ROW_H,
    ]
}

// The pick-lock toggle, outermost on an asset row.
pub(crate) fn lock_rect(o: [f32; 2], slot: usize) -> [f32; 4] {
    let r = row_rect(o, slot);
    [
        r[0] + r[2] - PAD - BOX_SIZE,
        r[1] + (ROW_H - BOX_SIZE) * 0.5,
        BOX_SIZE,
        BOX_SIZE,
    ]
}

// The hide toggle, just inside the lock.
pub(crate) fn eye_rect(o: [f32; 2], slot: usize) -> [f32; 4] {
    let r = row_rect(o, slot);
    [
        lock_rect(o, slot)[0] - GAP - BOX_SIZE,
        r[1] + (ROW_H - BOX_SIZE) * 0.5,
        BOX_SIZE,
        BOX_SIZE,
    ]
}

// The right edge the asset type is right-aligned against.
fn type_right(o: [f32; 2], slot: usize) -> f32 {
    eye_rect(o, slot)[0] - GAP
}

// The triple-dot button, taking over the type slot while the row is hovered
// (the type label hides for that row, so the two never overlap).
fn dot_rect(o: [f32; 2], slot: usize) -> [f32; 4] {
    let r = row_rect(o, slot);
    [
        type_right(o, slot) - DOT_SZ,
        r[1] + (ROW_H - DOT_SZ) * 0.5,
        DOT_SZ,
        DOT_SZ,
    ]
}

// A picker option row, floating over the tree body.
pub(crate) fn picker_option_rect(o: [f32; 2], slot: usize) -> [f32; 4] {
    [o[0], list_top(o) + slot as f32 * ROW_H, PANEL_W, ROW_H]
}

// The row menu, floating just below the name row at visible slot `slot`. A
// single Delete row (a name-row click opens the edit form, so Edit is redundant
// here). Returns (background, delete row).
fn menu_rects(o: [f32; 2], slot: usize) -> ([f32; 4], [f32; 4]) {
    let x = o[0] + PANEL_W - MENU_W - SCROLLBAR_W - 2.0;
    let top = list_top(o) + slot as f32 * ROW_H + ROW_H;
    let delete = [x, top, MENU_W, MENU_ROW_H];
    (delete, delete)
}

// A resolved panel click. Row picks carry the group and the index within it, so
// the hook resolves the asset (and its promote entry) through the same model the
// rows were flattened from.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum PanelAction {
    // Give the search field keyboard focus.
    FocusSearch,
    // The "+" button: open the type picker (from the tree) or close it.
    TogglePicker,
    // Choose picker option `i`.
    PickOption(usize),
    // A group header: fold / unfold it.
    ToggleGroup(usize),
    // An asset row's body: select it and open its editing surface.
    SelectRow(usize, usize),
    // An asset row's eye: flip the editor-session hide.
    ToggleHide(usize, usize),
    // An asset row's lock: flip the pick lock.
    ToggleLock(usize, usize),
    // An asset row's triple-dot: open its Delete menu.
    OpenRowMenu(usize, usize),
    // The open row menu's Delete row.
    RowDelete,
    // Dismiss any open overlay (picker / row menu) without picking.
    CloseOverlays,
    // A click inside the panel that hits no control (swallowed so it does not
    // fall through to the world; a text-field click resolves to this too, so the
    // engine's text-input system takes focus).
    Consume,
}

// The per-frame data the hook hands to `apply` / `hit_test`.
pub(crate) struct PanelView<'a> {
    // The flattened tree (group headers plus the assets of unfolded groups).
    pub rows: &'a [TreeRow],
    pub scroll: usize,
    // Whether the search field holds keyboard focus.
    pub search_focus: bool,
    // The floating type-picker options, already narrowed by the search field;
    // `None` while the picker is closed.
    pub picker_options: Option<&'a [String]>,
    pub picker_scroll: usize,
    // The viewport selection the rows mirror, and the session hide / lock sets.
    pub selection: &'a Selection,
    pub hidden: &'a BTreeSet<String>,
    pub locked: &'a BTreeSet<String>,
    // The name whose Delete menu is open, if any.
    pub row_menu: Option<&'a str>,
    // Total assets across every group, for the status line.
    pub total: usize,
    // Why the tree is empty, when a cook of the working entries failed rather
    // than there being nothing to show.
    pub status: Option<&'a str>,
    pub mouse: [f32; 2],
}

impl PanelView<'_> {
    fn picker_open(&self) -> bool {
        self.picker_options.is_some()
    }

    // The visible slot (0..ROW_POOL) currently showing the asset called `name`.
    fn visible_slot_of(&self, name: &str) -> Option<usize> {
        (0..ROW_POOL).find(|slot| {
            matches!(
                self.rows.get(self.scroll + slot),
                Some(TreeRow::Asset { name: n, .. }) if n == name
            )
        })
    }
}

// Resolve a click against the open panel at origin `o`. `None` means the click
// missed the panel entirely (the caller lets it fall through). Text-field clicks
// resolve to `FocusSearch` / `Consume` (swallowed here; the engine's text-input
// system focuses the field from the same input). Title-bar presses never reach
// this: the hook intercepts them first to start a drag.
pub(crate) fn hit_test(view: &PanelView, mx: f32, my: f32, o: [f32; 2]) -> Option<PanelAction> {
    // An open row menu is modal over the panel: its Delete row picks, anything
    // else dismisses it.
    if let Some(name) = view.row_menu {
        if let Some(slot) = view.visible_slot_of(name) {
            let (_, delete) = menu_rects(o, slot);
            if point_in(mx, my, delete) {
                return Some(PanelAction::RowDelete);
            }
        }
        return Some(PanelAction::CloseOverlays);
    }

    // An open picker captures the header field + its floating options.
    if let Some(options) = view.picker_options {
        if point_in(mx, my, plus_rect(o)) {
            return Some(PanelAction::TogglePicker);
        }
        if point_in(mx, my, search_rect(o)) {
            return Some(PanelAction::Consume);
        }
        let scroll = view.picker_scroll.min(options.len().saturating_sub(1));
        for slot in 0..ROW_POOL {
            let idx = scroll + slot;
            if idx >= options.len() {
                break;
            }
            if point_in(mx, my, picker_option_rect(o, slot)) {
                return Some(PanelAction::PickOption(idx));
            }
        }
        return Some(PanelAction::CloseOverlays);
    }

    // Picker closed: clicks outside the panel fall through (the hook's caller
    // decides who else wants them).
    if !point_in(mx, my, panel_rect(o)) {
        return None;
    }
    if point_in(mx, my, plus_rect(o)) {
        return Some(PanelAction::TogglePicker);
    }
    if point_in(mx, my, search_rect(o)) {
        return Some(PanelAction::FocusSearch);
    }

    for slot in 0..ROW_POOL {
        if !point_in(mx, my, row_rect(o, slot)) {
            continue;
        }
        let Some(row) = view.rows.get(view.scroll + slot) else {
            return Some(PanelAction::Consume);
        };
        return Some(match row {
            TreeRow::Header { group, .. } => PanelAction::ToggleGroup(*group),
            TreeRow::Asset {
                group,
                index,
                editable,
                ..
            } => {
                if point_in(mx, my, eye_rect(o, slot)) {
                    PanelAction::ToggleHide(*group, *index)
                } else if point_in(mx, my, lock_rect(o, slot)) {
                    PanelAction::ToggleLock(*group, *index)
                } else if *editable && point_in(mx, my, dot_rect(o, slot)) {
                    PanelAction::OpenRowMenu(*group, *index)
                } else {
                    PanelAction::SelectRow(*group, *index)
                }
            }
        });
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

    widget::place_panel(world, PANEL_BG, panel_rect(o));
    let title = title_rect(o);
    widget::place_heading(world, TITLE_LABEL, title, "Assets");
    let close_hover = point_in(view.mouse[0], view.mouse[1], widget::close_rect(title));
    widget::place_close(world, CLOSE_BG, CLOSE_LABEL, title, close_hover);

    // The "+" add button. While the type picker is open it becomes a gray "X"
    // that returns to the tree (the previous focus).
    let (glyph, tint) = if view.picker_open() {
        ("X", CANCEL_TINT)
    } else {
        ("+", PLUS_TINT)
    };
    place_rounded(
        world,
        PLUS_BG,
        plus_rect(o),
        tint,
        theme::CONTROL_RADIUS,
        true,
    );
    place_plus_glyph(world, plus_rect(o), glyph);
    // The field narrows the picker's options while it is open (and asserts focus
    // then, since the picker is a typed autocomplete), else it filters the tree.
    let focused = view.search_focus || view.picker_open();
    widget::show_field(world, SEARCH_INPUT, search_rect(o), focused);
    layout_status(world, view, o);

    match view.picker_options {
        Some(options) => layout_picker(world, view, options, o),
        None => layout_tree(world, view, o),
    }
}

// The status line: the asset count, or a cook failure in its place.
fn layout_status(world: &mut World, view: &PanelView, o: [f32; 2]) {
    if let Some(l) = widget::label_mut(world, STATUS_LABEL) {
        l.x = o[0] + PAD;
        l.y = header_y(o) + HEADER_H;
        l.align = TextAlign::Left;
        l.visible = true;
        match view.status {
            Some(e) => {
                l.color = ERROR_LABEL;
                l.content = e.to_string();
            }
            None => {
                l.color = LABEL_DIM;
                l.content = format!("Assets ({})", view.total);
            }
        }
    }
}

fn layout_tree(world: &mut World, view: &PanelView, o: [f32; 2]) {
    if view.rows.is_empty() {
        // A world that does not cook has no tree to show; the status line
        // already says which, so this only covers the genuinely empty world.
        if view.status.is_none() {
            place_left_label(
                world,
                EMPTY_LABEL,
                [o[0] + PAD, list_top(o) + PAD],
                "No matching assets",
                LABEL_DIM,
                true,
            );
        }
        return;
    }
    let total = view.rows.len();
    let scroll = view.scroll.min(total.saturating_sub(1));
    let mut menu_slot = None;
    for slot in 0..ROW_POOL {
        let Some(row) = view.rows.get(scroll + slot) else {
            break;
        };
        let r = row_rect(o, slot);
        let hovered = point_in(view.mouse[0], view.mouse[1], r);
        match row {
            TreeRow::Header {
                label, count, open, ..
            } => layout_header_row(world, slot, r, label, *count, *open, hovered),
            TreeRow::Asset {
                name,
                asset_type,
                badge,
                editable,
                ..
            } => {
                if view.row_menu == Some(name.as_str()) {
                    menu_slot = Some(slot);
                }
                let asset = AssetRow {
                    name,
                    asset_type,
                    badge,
                    hovered,
                };
                layout_asset_row(world, view, o, slot, &asset);
                // The triple-dot follows the row whose menu is open, else the
                // hovered row; only an editable asset offers one at all.
                if *editable && (view.row_menu == Some(name.as_str()) || hovered) {
                    let over_dots = point_in(view.mouse[0], view.mouse[1], dot_rect(o, slot));
                    let show_box = view.row_menu == Some(name.as_str()) || over_dots;
                    place_dot(world, o, slot, show_box);
                    // The dots take over the type slot, so that row's type hides.
                    widget::set_label_visible(world, type_label(slot), false);
                }
            }
        }
    }
    if let Some(slot) = menu_slot {
        layout_row_menu(world, view, o, slot);
    }
    layout_scrollbar(world, total, scroll, o);
}

fn layout_header_row(
    world: &mut World,
    slot: usize,
    r: [f32; 4],
    label: &str,
    count: usize,
    open: bool,
    hovered: bool,
) {
    let tint = if hovered { ROW_TINT_HOVER } else { ROW_TINT };
    place_rounded(
        world,
        row_bg(slot),
        theme::highlight_rect(r),
        tint,
        theme::CONTROL_RADIUS,
        true,
    );
    let marker = if open { "-" } else { "+" };
    place_left_label(
        world,
        name_label(slot),
        [r[0] + PAD, r[1] + ROW_H * 0.5 - theme::TEXT_HALF],
        &format!("{marker} {label} ({count})"),
        theme::HEADING,
        true,
    );
}

// One asset row's drawable payload, borrowed out of its `TreeRow::Asset`.
struct AssetRow<'a> {
    name: &'a str,
    asset_type: &'a str,
    badge: &'a Badge,
    hovered: bool,
}

fn layout_asset_row(
    world: &mut World,
    view: &PanelView,
    o: [f32; 2],
    slot: usize,
    asset: &AssetRow,
) {
    let AssetRow {
        name,
        asset_type,
        badge,
        hovered,
    } = *asset;
    let r = row_rect(o, slot);
    let selected = view.selection.contains(name);
    let tint = if hovered {
        ROW_TINT_HOVER
    } else if selected {
        ROW_TINT_SELECTED
    } else {
        ROW_TINT
    };
    place_rounded(
        world,
        row_bg(slot),
        theme::highlight_rect(r),
        tint,
        theme::CONTROL_RADIUS,
        true,
    );
    let hidden = view.hidden.contains(name);
    place_left_label(
        world,
        name_label(slot),
        [r[0] + ASSET_INDENT, r[1] + ROW_H * 0.5 - theme::TEXT_HALF],
        &widget::clip_text(name, MAX_NAME_CHARS),
        // The hidden state dims the name too, so a collapsed object reads as
        // absent at a glance.
        if hidden { LABEL_DIM } else { LABEL },
        true,
    );
    if let Some(l) = widget::label_mut(world, type_label(slot)) {
        l.x = type_right(o, slot);
        l.y = r[1] + ROW_H * 0.5 - theme::TEXT_HALF;
        l.align = TextAlign::Right;
        l.color = badge_color(*badge);
        l.visible = true;
        l.content = widget::clip_text(asset_type, MAX_TYPE_CHARS);
    }
    let eye = if hidden { EYE_TINT_OFF } else { EYE_TINT_ON };
    place_rounded(world, eye_box(slot), eye_rect(o, slot), eye, 4.0, true);
    let lock = if view.locked.contains(name) {
        LOCK_TINT_ON
    } else {
        LOCK_TINT_OFF
    };
    place_rounded(world, lock_box(slot), lock_rect(o, slot), lock, 4.0, true);
}

// The floating type-picker list, over the tree body.
fn layout_picker(world: &mut World, view: &PanelView, options: &[String], o: [f32; 2]) {
    let total = options.len();
    let scroll = view.picker_scroll.min(total.saturating_sub(1));
    let shown = total.saturating_sub(scroll).clamp(1, ROW_POOL);
    let backing = [o[0], list_top(o), PANEL_W, shown as f32 * ROW_H + PAD];
    place_sprite(world, PICKER_BG, backing, PICKER_BG_TINT, true);
    if total == 0 {
        place_left_label(
            world,
            EMPTY_LABEL,
            [o[0] + PAD, list_top(o) + PAD],
            "No matching types",
            LABEL_DIM,
            true,
        );
        return;
    }
    for slot in 0..ROW_POOL {
        let idx = scroll + slot;
        if idx >= total {
            break;
        }
        let rect = picker_option_rect(o, slot);
        let hovered = point_in(view.mouse[0], view.mouse[1], rect);
        let tint = if hovered {
            OPTION_TINT_HOVER
        } else {
            OPTION_TINT
        };
        place_rounded(
            world,
            picker_row_bg(slot),
            theme::highlight_rect(rect),
            tint,
            theme::CONTROL_RADIUS,
            true,
        );
        place_left_label(
            world,
            picker_row_label(slot),
            [rect[0] + PAD, rect[1] + ROW_H * 0.5 - theme::TEXT_HALF],
            &options[idx],
            LABEL,
            true,
        );
    }
    layout_scrollbar(world, total, scroll, o);
}

// The "+" / "X" glyph, centered in the add button and drawn a step larger than
// the body text.
const PLUS_SCALE: f32 = 1.15;
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

// The three stacked white dots of the triple-dot button. The background box is
// shown only when `show_box` (the dots are hovered, or the menu is open); a plain
// row-hover shows the bare dots.
fn place_dot(world: &mut World, o: [f32; 2], slot: usize, show_box: bool) {
    let d = dot_rect(o, slot);
    if show_box {
        place_rounded(world, DOT_BG, d, DOT_BG_TINT, theme::CONTROL_RADIUS, true);
    }
    let cx = d[0] + d[2] * 0.5;
    let cy = d[1] + d[3] * 0.5;
    let s = 3.5;
    let gap = 3.5;
    for (id, dy) in [(DOT1, -gap - s), (DOT2, -s * 0.5), (DOT3, gap)] {
        place_sprite(world, id, [cx - s * 0.5, cy + dy, s, s], DOT_TINT, true);
    }
}

fn layout_row_menu(world: &mut World, view: &PanelView, o: [f32; 2], slot: usize) {
    let (bg, delete) = menu_rects(o, slot);
    place_rounded(
        world,
        MENU_BG,
        bg,
        MENU_BG_TINT,
        theme::CONTROL_RADIUS,
        true,
    );
    let del_hover = point_in(view.mouse[0], view.mouse[1], delete);
    place_rounded(
        world,
        MENU_DELETE_BG,
        delete,
        if del_hover {
            MENU_ROW_HOVER
        } else {
            MENU_ROW_TINT
        },
        theme::CONTROL_RADIUS,
        true,
    );
    place_left_label(
        world,
        MENU_DELETE_LABEL,
        [
            delete[0] + PAD,
            delete[1] + MENU_ROW_H * 0.5 - theme::TEXT_HALF,
        ],
        "Delete",
        DELETE_LABEL,
        true,
    );
}

// A simple non-interactive scrollbar sizing the visible window against `total`,
// down the panel's right edge. Shown only when the body overflows.
fn layout_scrollbar(world: &mut World, total: usize, scroll: usize, o: [f32; 2]) {
    if total <= ROW_POOL {
        return;
    }
    let x = o[0] + PANEL_W - SCROLLBAR_W;
    let top = list_top(o);
    let h = ROW_POOL as f32 * ROW_H;
    place_rounded(
        world,
        LIST_TRACK,
        [x, top, SCROLLBAR_W, h],
        TRACK_TINT,
        SCROLLBAR_W * 0.5,
        true,
    );
    let thumb_h = (h * ROW_POOL as f32 / total as f32).max(18.0);
    let max_scroll = (total - ROW_POOL) as f32;
    let off = (h - thumb_h) * (scroll.min(total - ROW_POOL) as f32 / max_scroll);
    place_rounded(
        world,
        LIST_THUMB,
        [x, top + off, SCROLLBAR_W, thumb_h],
        THUMB_TINT,
        SCROLLBAR_W * 0.5,
        true,
    );
}

// Every panel sprite id, so the closed / hidden pass can blank the whole panel
// (and `inject.rs` can create exactly this set). THE ORDER OF THIS VEC IS THE DRAW
// ORDER: `inject.rs` adds the panel's Sprites in this sequence, and the overlay
// draws components in insertion (component-column) order -- NOT by asset id -- so
// later entries paint on top. Bottom-to-top: panel background, header chrome, the
// row families, the picker backing (under its option rows), then the floating
// overlays (scrollbar, triple-dot, row menu), which must sit ABOVE the row
// backgrounds so a hovered row's fill cannot cover them.
pub(crate) fn all_sprite_ids() -> Vec<AssetId> {
    let mut ids = vec![PANEL_BG, CLOSE_BG, PLUS_BG];
    ids.extend((0..ROW_POOL).map(row_bg));
    ids.extend((0..ROW_POOL).map(eye_box));
    ids.extend((0..ROW_POOL).map(lock_box));
    ids.push(PICKER_BG);
    ids.extend((0..ROW_POOL).map(picker_row_bg));
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
// order decides who wins). The row-menu caption comes last so it draws above the
// row labels the menu floats over.
pub(crate) fn all_label_ids() -> Vec<AssetId> {
    let mut ids = vec![
        TITLE_LABEL,
        CLOSE_LABEL,
        PLUS_LABEL,
        STATUS_LABEL,
        EMPTY_LABEL,
    ];
    ids.extend((0..ROW_POOL).map(name_label));
    ids.extend((0..ROW_POOL).map(type_label));
    ids.extend((0..ROW_POOL).map(picker_row_label));
    ids.push(MENU_DELETE_LABEL);
    ids
}

// Every typed field the panel injects: just the search field (the form's inputs
// belong to `form_panel.rs`).
pub(crate) fn all_field_ids() -> Vec<AssetId> {
    vec![SEARCH_INPUT]
}

// Hide every panel element, including the typed field (and blur it so a hidden
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

// Whether the cursor is over the scrollable body area (for wheel scrolling).
pub(crate) fn cursor_over_body(mx: f32, my: f32, o: [f32; 2]) -> bool {
    let p = panel_rect(o);
    mx >= p[0] && mx < p[0] + p[2] && my >= list_top(o) && my < p[1] + p[3]
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

    fn header(group: usize, label: &str, count: usize, open: bool) -> TreeRow {
        TreeRow::Header {
            group,
            label: label.to_string(),
            count,
            open,
        }
    }

    fn asset(group: usize, index: usize, name: &str, editable: bool) -> TreeRow {
        TreeRow::Asset {
            group,
            index,
            name: name.to_string(),
            asset_type: "Material".to_string(),
            badge: Badge::Imported,
            editable,
        }
    }

    struct Fixture {
        rows: Vec<TreeRow>,
        picker_options: Vec<String>,
        selection: Selection,
        hidden: BTreeSet<String>,
        locked: BTreeSet<String>,
    }

    impl Fixture {
        fn new() -> Self {
            Self {
                rows: vec![
                    header(0, "World", 2, true),
                    asset(0, 0, "cam", true),
                    asset(0, 1, "lamp", true),
                    header(1, "fox", 1, false),
                ],
                picker_options: Vec::new(),
                selection: Selection::default(),
                hidden: BTreeSet::new(),
                locked: BTreeSet::new(),
            }
        }

        fn view(&self) -> PanelView<'_> {
            PanelView {
                rows: &self.rows,
                scroll: 0,
                search_focus: false,
                picker_options: None,
                picker_scroll: 0,
                selection: &self.selection,
                hidden: &self.hidden,
                locked: &self.locked,
                row_menu: None,
                total: 3,
                status: None,
                mouse: [0.0, 0.0],
            }
        }

        // A view with the type picker open over the tree.
        fn picker_view(&self) -> PanelView<'_> {
            PanelView {
                picker_options: Some(&self.picker_options),
                ..self.view()
            }
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

    fn label(world: &World, id: AssetId) -> TextLabel {
        world
            .query::<TextLabel>()
            .find(|l| l.asset_id == id)
            .cloned()
            .expect("label present")
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

    // The header stacks under the title bar, the tree under the status line, and
    // every row control stays inside its row without overlapping its neighbour.
    #[test]
    fn geometry_stacks_and_row_controls_stay_inside_the_row() {
        let o = test_origin();
        assert_eq!(title_rect(o), [o[0], o[1], PANEL_W, widget::TITLE_H]);
        let plus = plus_rect(o);
        let search = search_rect(o);
        assert!(plus[0] + plus[2] <= search[0], "+ sits left of the search");
        assert_eq!(
            search[0] + search[2],
            o[0] + PANEL_W - PAD,
            "the search field reaches the panel's right pad"
        );
        assert_eq!(row_rect(o, 0)[1], list_top(o));
        assert_eq!(row_rect(o, 1)[1], list_top(o) + ROW_H);

        let r = row_rect(o, 0);
        let (dots, eye, lock) = (dot_rect(o, 0), eye_rect(o, 0), lock_rect(o, 0));
        assert!(
            dots[0] + dots[2] <= eye[0] + 0.01,
            "dots sit left of the eye"
        );
        assert!(eye[0] + eye[2] + GAP <= lock[0] + 0.01, "eye left of lock");
        assert!(
            lock[0] + lock[2] <= r[0] + r[2],
            "lock stays inside the row"
        );
        assert!(eye[1] >= r[1] && eye[1] + eye[3] <= r[1] + r[3]);
        // The name never runs under the triple-dot slot.
        let name_room = dots[0] - (r[0] + ASSET_INDENT);
        assert!(name_room > 0.0, "the name has room left of the dots");
    }

    #[test]
    fn plus_toggles_the_picker_from_either_body() {
        let f = Fixture::new();
        let o = test_origin();
        let plus = plus_rect(o);
        assert_eq!(
            hit_test(&f.view(), plus[0] + 5.0, plus[1] + 5.0, o),
            Some(PanelAction::TogglePicker)
        );
        // Open, the same button is the "X" that returns to the tree.
        assert_eq!(
            hit_test(&f.picker_view(), plus[0] + 5.0, plus[1] + 5.0, o),
            Some(PanelAction::TogglePicker)
        );
    }

    // The header button renders as a green "+" over the tree and as a gray "X"
    // (return to the tree) while the type picker is open.
    #[test]
    fn plus_renders_as_an_x_while_picker_is_open() {
        let mut f = Fixture::new();
        f.picker_options = vec!["PointLight".to_string()];
        let mut world = injected_world();
        let o = test_origin();
        apply(&mut world, Some(&f.view()), o);
        let l = label(&world, PLUS_LABEL);
        assert_eq!(l.content, "+");
        assert_eq!(sprite(&world, PLUS_BG).tint, PLUS_TINT);
        assert!(
            l.scale > 1.0,
            "the glyph draws larger than the body text (the box is unchanged)"
        );
        apply(&mut world, Some(&f.picker_view()), o);
        assert_eq!(label(&world, PLUS_LABEL).content, "X");
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
        let f = Fixture::new();
        let mut world = injected_world();
        let o = test_origin();
        apply(&mut world, Some(&f.view()), o);
        let title = label(&world, TITLE_LABEL);
        assert!(title.visible);
        assert_eq!(title.content, "Assets");
        assert!(sprite(&world, PANEL_BG).visible);
        let t = title_rect(o);
        assert_eq!(
            hit_test(&f.view(), t[0] + 5.0, t[1] + 5.0, o),
            Some(PanelAction::Consume)
        );
    }

    #[test]
    fn hit_test_resolves_search_headers_rows_and_toggles() {
        let f = Fixture::new();
        let o = test_origin();
        let s = search_rect(o);
        assert_eq!(
            hit_test(&f.view(), s[0] + 5.0, s[1] + 5.0, o),
            Some(PanelAction::FocusSearch)
        );
        let r0 = row_rect(o, 0);
        assert_eq!(
            hit_test(&f.view(), r0[0] + 5.0, r0[1] + 5.0, o),
            Some(PanelAction::ToggleGroup(0))
        );
        let r1 = row_rect(o, 1);
        assert_eq!(
            hit_test(&f.view(), r1[0] + 5.0, r1[1] + 5.0, o),
            Some(PanelAction::SelectRow(0, 0))
        );
        let e = eye_rect(o, 1);
        assert_eq!(
            hit_test(&f.view(), e[0] + 2.0, e[1] + 2.0, o),
            Some(PanelAction::ToggleHide(0, 0))
        );
        let l = lock_rect(o, 1);
        assert_eq!(
            hit_test(&f.view(), l[0] + 2.0, l[1] + 2.0, o),
            Some(PanelAction::ToggleLock(0, 0))
        );
        let d = dot_rect(o, 1);
        assert_eq!(
            hit_test(&f.view(), d[0] + 2.0, d[1] + 2.0, o),
            Some(PanelAction::OpenRowMenu(0, 0))
        );
        // Past the last row the click is swallowed; off the panel it misses.
        let r9 = row_rect(o, 9);
        assert_eq!(
            hit_test(&f.view(), r9[0] + 5.0, r9[1] + 5.0, o),
            Some(PanelAction::Consume)
        );
        assert_eq!(hit_test(&f.view(), 5000.0, 5000.0, o), None);
    }

    // A row the build emits unconditionally cannot be promoted, so its dot slot
    // is inert and the click falls through to a plain select.
    #[test]
    fn a_non_editable_row_offers_no_delete_menu() {
        let mut f = Fixture::new();
        f.rows = vec![
            header(0, "Other expansions", 1, true),
            asset(0, 0, "tab", false),
        ];
        let o = test_origin();
        let d = dot_rect(o, 1);
        assert_eq!(
            hit_test(&f.view(), d[0] + 2.0, d[1] + 2.0, o),
            Some(PanelAction::SelectRow(0, 0))
        );
        let mut world = injected_world();
        let v = PanelView {
            mouse: [d[0] + 2.0, d[1] + 2.0],
            ..f.view()
        };
        apply(&mut world, Some(&v), o);
        assert!(
            !sprite(&world, DOT1).visible,
            "no triple-dot on a row that cannot be promoted or deleted"
        );
    }

    #[test]
    fn scrolled_rows_resolve_through_the_window_offset() {
        let f = Fixture::new();
        let o = test_origin();
        let v = PanelView {
            scroll: 2,
            ..f.view()
        };
        let r0 = row_rect(o, 0);
        assert_eq!(
            hit_test(&v, r0[0] + 5.0, r0[1] + 5.0, o),
            Some(PanelAction::SelectRow(0, 1)),
            "slot 0 shows row 2 under scroll 2"
        );
    }

    #[test]
    fn apply_draws_headers_types_and_toggle_states() {
        let mut world = injected_world();
        let mut f = Fixture::new();
        f.selection.replace("cam".to_string());
        f.hidden.insert("lamp".to_string());
        f.locked.insert("cam".to_string());
        let o = test_origin();
        apply(&mut world, Some(&f.view()), o);

        assert_eq!(label(&world, STATUS_LABEL).content, "Assets (3)");
        assert_eq!(label(&world, name_label(0)).content, "- World (2)");
        assert!(
            !sprite(&world, eye_box(0)).visible,
            "header rows draw no toggles"
        );
        // The folded group header shows its fold marker.
        assert_eq!(label(&world, name_label(3)).content, "+ fox (1)");

        // cam: selected, locked, visible; its type reads in the right slot.
        assert_eq!(sprite(&world, row_bg(1)).tint, ROW_TINT_SELECTED);
        assert_eq!(sprite(&world, eye_box(1)).tint, EYE_TINT_ON);
        assert_eq!(sprite(&world, lock_box(1)).tint, LOCK_TINT_ON);
        let ty = label(&world, type_label(1));
        assert_eq!(ty.content, "Material");
        assert_eq!(ty.align, TextAlign::Right);
        assert_eq!(ty.color, badge_color(Badge::Imported));

        // lamp: hidden dims the name and flips the eye.
        assert_eq!(sprite(&world, eye_box(2)).tint, EYE_TINT_OFF);
        assert_eq!(label(&world, name_label(2)).color, LABEL_DIM);
        assert_eq!(sprite(&world, lock_box(2)).tint, LOCK_TINT_OFF);

        // Empty slots past the tree stay blank.
        assert!(!sprite(&world, row_bg(5)).visible);
    }

    // The search field shows over both bodies; the picker asserts focus into it
    // (it is a typed autocomplete) even when the tree had not been focused.
    #[test]
    fn the_search_field_shows_always_and_the_picker_focuses_it() {
        let mut world = injected_world();
        let mut f = Fixture::new();
        f.picker_options = vec!["PointLight".to_string()];
        let o = test_origin();
        let field = |w: &World| {
            w.query::<TextInput>()
                .find(|t| t.asset_id == SEARCH_INPUT)
                .cloned()
                .unwrap()
        };
        apply(&mut world, Some(&f.view()), o);
        let t = field(&world);
        assert!(t.visible && !t.focused, "shown, unfocused until clicked");

        apply(&mut world, Some(&f.picker_view()), o);
        assert!(field(&world).focused, "the picker types into the field");
    }

    // The triple-dot takes over the type slot on hover, so the two never draw on
    // top of one another.
    #[test]
    fn the_triple_dot_replaces_the_type_on_a_hovered_row() {
        let mut world = injected_world();
        let f = Fixture::new();
        let o = test_origin();
        let r1 = row_rect(o, 1);
        let v = PanelView {
            mouse: [r1[0] + 5.0, r1[1] + 5.0],
            ..f.view()
        };
        apply(&mut world, Some(&v), o);
        assert!(sprite(&world, DOT1).visible, "hover reveals the dots");
        assert!(
            !label(&world, type_label(1)).visible,
            "the hovered row's type yields the slot"
        );
        assert!(
            !sprite(&world, DOT_BG).visible,
            "the box shows only over the dots themselves"
        );
        // Over the dots proper, the background box appears too.
        let d = dot_rect(o, 1);
        let over = PanelView {
            mouse: [d[0] + 2.0, d[1] + 2.0],
            ..f.view()
        };
        apply(&mut world, Some(&over), o);
        assert!(sprite(&world, DOT_BG).visible);
    }

    // The row menu is modal: its Delete row picks, anything else dismisses it.
    #[test]
    fn the_row_menu_picks_delete_and_dismisses_elsewhere() {
        let f = Fixture::new();
        let o = test_origin();
        let v = PanelView {
            row_menu: Some("cam"),
            ..f.view()
        };
        let (_, delete) = menu_rects(o, 1);
        assert_eq!(
            hit_test(&v, delete[0] + 5.0, delete[1] + 5.0, o),
            Some(PanelAction::RowDelete)
        );
        assert_eq!(
            hit_test(&v, o[0] + 5.0, o[1] + 5.0, o),
            Some(PanelAction::CloseOverlays)
        );

        let mut world = injected_world();
        apply(&mut world, Some(&v), o);
        assert!(sprite(&world, MENU_BG).visible);
        assert_eq!(label(&world, MENU_DELETE_LABEL).content, "Delete");
    }

    // An open row menu whose asset scrolled out of the visible window has no
    // hittable Delete row, so any click just closes the overlay.
    #[test]
    fn row_menu_dismisses_when_its_asset_is_off_window() {
        let f = Fixture::new();
        let o = test_origin();
        let v = PanelView {
            row_menu: Some("not_listed"),
            ..f.view()
        };
        let (_, delete) = menu_rects(o, 1);
        assert_eq!(
            hit_test(&v, delete[0] + 5.0, delete[1] + 5.0, o),
            Some(PanelAction::CloseOverlays)
        );
    }

    #[test]
    fn picker_options_pick_and_dismiss() {
        let mut f = Fixture::new();
        f.picker_options = vec!["Decal".to_string(), "PointLight".to_string()];
        let o = test_origin();
        let v = f.picker_view();
        let r1 = picker_option_rect(o, 1);
        assert_eq!(
            hit_test(&v, r1[0] + 5.0, r1[1] + 5.0, o),
            Some(PanelAction::PickOption(1))
        );
        // The header field keeps focus (consumed), and empty body space closes.
        let s = search_rect(o);
        assert_eq!(
            hit_test(&v, s[0] + 5.0, s[1] + 5.0, o),
            Some(PanelAction::Consume)
        );
        let r13 = picker_option_rect(o, 13);
        assert_eq!(
            hit_test(&v, r13[0] + 5.0, r13[1] + 5.0, o),
            Some(PanelAction::CloseOverlays)
        );
    }

    #[test]
    fn picker_shows_an_empty_state_with_no_matching_options() {
        let f = Fixture::new();
        let mut world = injected_world();
        apply(&mut world, Some(&f.picker_view()), test_origin());
        let empty = label(&world, EMPTY_LABEL);
        assert!(empty.visible);
        assert_eq!(empty.content, "No matching types");
    }

    #[test]
    fn a_cook_error_replaces_the_count_line() {
        let mut world = injected_world();
        let f = Fixture::new();
        let v = PanelView {
            rows: &[],
            status: Some("the world does not build"),
            ..f.view()
        };
        apply(&mut world, Some(&v), test_origin());
        let status = label(&world, STATUS_LABEL);
        assert_eq!(status.content, "the world does not build");
        assert_eq!(status.color, ERROR_LABEL);
        assert!(
            !label(&world, EMPTY_LABEL).visible,
            "the status line already explains the empty tree"
        );
    }

    // A search that matches nothing leaves the tree empty with its own message,
    // distinct from a cook failure.
    #[test]
    fn an_empty_tree_without_an_error_says_nothing_matched() {
        let mut world = injected_world();
        let f = Fixture::new();
        let v = PanelView {
            rows: &[],
            ..f.view()
        };
        apply(&mut world, Some(&v), test_origin());
        assert_eq!(label(&world, EMPTY_LABEL).content, "No matching assets");
    }

    #[test]
    fn long_tree_shows_the_scrollbar() {
        let mut world = injected_world();
        let mut f = Fixture::new();
        for i in 0..20 {
            f.rows.push(asset(1, i, &format!("a{i}"), true));
        }
        let v = PanelView {
            scroll: 3,
            ..f.view()
        };
        apply(&mut world, Some(&v), test_origin());
        assert!(sprite(&world, LIST_THUMB).visible);
        // A short tree hides it again.
        let short = Fixture::new();
        apply(&mut world, Some(&short.view()), test_origin());
        assert!(!sprite(&world, LIST_THUMB).visible);
    }

    #[test]
    fn hide_all_blanks_every_element() {
        let mut world = injected_world();
        let f = Fixture::new();
        apply(&mut world, Some(&f.view()), test_origin());
        apply(&mut world, None, [0.0, 0.0]);
        assert!(world.query::<Sprite>().all(|s| !s.visible));
        assert!(world.query::<TextLabel>().all(|l| !l.visible));
        assert!(world.query::<TextInput>().all(|t| !t.visible));
    }

    // The floating overlays are injected after the row families, so a hovered
    // row's fill can never paint over the scrollbar, dots, or row menu.
    #[test]
    fn floating_overlays_are_injected_after_the_row_families() {
        let ids = all_sprite_ids();
        let pos = |id: AssetId| ids.iter().position(|&x| x == id).expect("id listed");
        let last_row = (0..ROW_POOL).map(row_bg).map(pos).max().unwrap();
        for overlay in [LIST_TRACK, LIST_THUMB, DOT_BG, DOT1, MENU_BG] {
            assert!(
                pos(overlay) > last_row,
                "overlays must draw above the row backgrounds"
            );
        }
        // The picker backing sits under its own option rows.
        assert!(pos(PICKER_BG) < (0..ROW_POOL).map(picker_row_bg).map(pos).min().unwrap());
    }

    #[test]
    fn cursor_over_body_covers_the_scrollable_area_only() {
        let o = test_origin();
        assert!(cursor_over_body(o[0] + 5.0, list_top(o) + 5.0, o));
        assert!(
            !cursor_over_body(o[0] + 5.0, o[1] + 5.0, o),
            "the title bar is not the scrollable body"
        );
        assert!(!cursor_over_body(o[0] - 5.0, list_top(o) + 5.0, o));
    }

    // Cook `ty` blank in a minimal world: alongside a GraphicsConfig so the
    // world renders, except for singletons, which cook alone (the fixed
    // GraphicsConfig line would collide with the GraphicsConfig probe under
    // the singleton shape rule; a lone singleton still cooks, as a
    // non-rendering world when it is not itself the renderer config).
    fn cook_blank(ty: &str) -> std::io::Result<()> {
        let world = if is_singleton(ty) {
            format!("{{\"name\":\"probe\",\"type\":\"{ty}\",\"args\":{{}}}}\n")
        } else {
            format!(
                "{{\"name\":\"gfx\",\"type\":\"GraphicsConfig\",\"args\":{{}}}}\n\
                 {{\"name\":\"probe\",\"type\":\"{ty}\",\"args\":{{}}}}\n"
            )
        };
        crate::build_pipeline_from_str(&world, None).map(|_| ())
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
            cook_blank(ty).unwrap_or_else(|e| panic!("{ty} must cook with default args: {e}"));
        }
    }

    // The set of offered types is exactly the addable types that cook with default
    // args AND are useful when added blank: no world-config singleton (those want an
    // edit-or-add flow), no engine-injected HUD, and no type whose value is a nested
    // array / source file it can't be given here. Enforced so a newly-registered
    // addable-and-blank-useful type is a deliberate `useful_blank` flag choice in
    // the registry, not forgotten.
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
            "Reaction",
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
            let cooks = cook_blank(ty).is_ok();
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
                "{ty} cooks blank: flag it `useful_blank` in the registry or add it to the EXCLUDED list (with a reason), not both/neither"
            );
        }
    }
}
