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

use concinnity_cook::resource_handles::ResourceAssetType;
use concinnity_world::registry::ComponentType;

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
// nested / source form controls exist -- Behavior is flagged because it has
// one: its body is authored in the Behavior panel. Types that cannot even cook
// blank --
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
// The visibility eye at a row's head is drawn from two sprites: an outline that
// reshapes from an open lens to a closed lid, and a pupil shown only while open.
// The per-row sub-families are spaced 0x20 apart so each holds up to
// `ROW_POOL_MAX` slots (a resized panel reveals more rows) without overlapping
// the next; the whole set stays within the Assets id window (below the Edit
// family), guarded by `id_families_are_disjoint`.
fn eye_ring(slot: usize) -> AssetId {
    AssetId(PANEL + 0x80 + slot as u32)
}
fn eye_pupil(slot: usize) -> AssetId {
    AssetId(PANEL + 0xA0 + slot as u32)
}
// The pick lock beside the eye is a padlock: a shackle arch behind a solid body.
fn lock_shackle(slot: usize) -> AssetId {
    AssetId(PANEL + 0xC0 + slot as u32)
}
fn lock_body(slot: usize) -> AssetId {
    AssetId(PANEL + 0xE0 + slot as u32)
}
pub(crate) fn picker_row_bg(slot: usize) -> AssetId {
    AssetId(PANEL + 0x100 + slot as u32)
}
pub(crate) fn picker_row_label(slot: usize) -> AssetId {
    AssetId(PANEL + 0x120 + slot as u32)
}

// Geometry, in window pixels. Every rect derives from the panel's origin `o`
// (its title bar's top-left corner), so dragging the title bar moves the whole
// panel; the hook owns the origin.
// The default (and minimum) panel width; the user can widen it past this.
pub(crate) const PANEL_W: f32 = 500.0;
const PAD: f32 = 10.0;
const HEADER_H: f32 = 36.0;
const STATUS_H: f32 = 20.0;
pub(crate) const ROW_H: f32 = 26.0;
const SCROLLBAR_W: f32 = 5.0;
const GAP: f32 = 6.0;
// The square region each row icon draws inside: the visibility eye, then the
// pick lock beside it.
const BOX_SIZE: f32 = 14.0;
// The gap between the eye and the lock at a row's head.
const ICON_GAP: f32 = 5.0;
// An asset row's name is inset past both head icons, so it still reads as
// indented under the group header (whose caption starts at the base pad).
const ASSET_INDENT: f32 = PAD + BOX_SIZE + ICON_GAP + BOX_SIZE + GAP;
// Stroke width for the outlined parts of a row icon (eye lens, lock shackle).
const ICON_STROKE: f32 = 1.5;
// Visible rows in the body at the default height, shared by the tree and the
// picker's option list; resizing the panel taller reveals more, up to the
// injected pool `ROW_POOL_MAX`.
pub(crate) const ROW_POOL: usize = 14;
pub(crate) const ROW_POOL_MAX: usize = 28;
// The asset name, and the type that reads right-aligned beside it, inside what
// the triple-dot's reserved slot leaves. The lock icon widened the row's head,
// so the name clips a step shorter to keep clear of its type caption. This is
// the budget at the default width; widening the panel grows it.
const MAX_NAME_CHARS: usize = 23;
// Approximate body-font advance, for growing the name budget with the width.
const CHAR_W: f32 = 8.5;

// The chrome height above the tree body (title bar, header, status line).
const CHROME_H: f32 = widget::TITLE_H + HEADER_H + STATUS_H;

// The visible row count for panel height `h`, clamped to the injected pool. At
// the default height this is exactly `ROW_POOL`.
pub(crate) fn visible_rows(h: f32) -> usize {
    (((h - CHROME_H - PAD) / ROW_H).floor() as usize).clamp(ROW_POOL, ROW_POOL_MAX)
}

// The asset-name character budget at width `w` (grows past the default with width).
fn name_budget(w: f32) -> usize {
    (MAX_NAME_CHARS as f32 + (w - PANEL_W) / CHAR_W).max(6.0) as usize
}

// The tallest the panel resizes to: the injected row pool shown in full (width
// unbounded, capped only by the screen).
pub(crate) fn max_size() -> [f32; 2] {
    [
        f32::INFINITY,
        size()[1] + (ROW_POOL_MAX - ROW_POOL) as f32 * ROW_H,
    ]
}
// 17 covers the longest registered type name (`PostProcessConfig`), so a
// type caption never clips.
const MAX_TYPE_CHARS: usize = 17;
// The triple-dot button on a hovered name row.
const DOT_SZ: f32 = 20.0;
// The floating row menu.
const MENU_W: f32 = 132.0;
const MENU_ROW_H: f32 = 26.0;

const PLUS_TINT: [f32; 4] = [0.20, 0.44, 0.30, 1.0];
const CANCEL_TINT: [f32; 4] = [0.30, 0.30, 0.34, 1.0];
const ROW_TINT: [f32; 4] = [0.0, 0.0, 0.0, 0.0];
const ROW_TINT_HOVER: [f32; 4] = theme::HOVER_TINT;
const ROW_TINT_SELECTED: [f32; 4] = theme::SELECTED_TINT;
const TRACK_TINT: [f32; 4] = [0.12, 0.12, 0.15, 0.9];
const THUMB_TINT: [f32; 4] = [0.40, 0.44, 0.56, 0.95];
// The eye: a light neutral while the asset renders, dim once hidden.
const EYE_TINT: [f32; 4] = [0.82, 0.84, 0.90, 1.0];
const EYE_TINT_HIDDEN: [f32; 4] = [0.48, 0.49, 0.55, 1.0];
// The lock: amber while the asset is locked out of picking, a faint ghost while
// unlocked (only shown on row hover, hinting the toggle).
const LOCK_TINT: [f32; 4] = [0.90, 0.72, 0.40, 1.0];
const LOCK_TINT_HINT: [f32; 4] = [0.56, 0.57, 0.63, 0.75];
const PICKER_BG_TINT: [f32; 4] = [0.09, 0.09, 0.12, 1.0];
const OPTION_TINT: [f32; 4] = [0.16, 0.16, 0.20, 0.0];
const OPTION_TINT_HOVER: [f32; 4] = theme::HOVER_TINT;
const DOT_BG_TINT: [f32; 4] = [0.30, 0.34, 0.46, 0.95];
const DOT_TINT: [f32; 4] = [0.90, 0.92, 0.96, 1.0];
// Lifted well clear of the panel chrome behind it, and framed by the shared
// hairline border, so the menu reads as a floating surface rather than as text
// sitting loose on the panel.
const MENU_BG_TINT: [f32; 4] = [0.22, 0.23, 0.29, 1.0];
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
        Badge::Overridden => [0.62, 0.76, 1.0],
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

// The square "+" add button at the header's left.
pub(crate) fn plus_rect(o: [f32; 2]) -> [f32; 4] {
    let h = HEADER_H - 8.0;
    [o[0] + PAD, widget::header_y(o, 0.0) + 4.0, h, h]
}

// The search field, filling the header row right of the "+", to the width `w`.
pub(crate) fn search_rect(o: [f32; 2], w: f32) -> [f32; 4] {
    let p = plus_rect(o);
    [p[0] + p[2] + GAP, p[1], w - PAD - (p[2] + GAP) - PAD, p[3]]
}

// Where the tree body begins: below the header and the status line.
fn list_top(o: [f32; 2]) -> f32 {
    widget::header_y(o, 0.0) + HEADER_H + STATUS_H
}

// Visible row `slot` (0-based within the scroll window), stopping short of the
// scrollbar, at the effective width `w`.
pub(crate) fn row_rect(o: [f32; 2], w: f32, slot: usize) -> [f32; 4] {
    [
        o[0],
        list_top(o) + slot as f32 * ROW_H,
        w - SCROLLBAR_W - 2.0,
        ROW_H,
    ]
}

// The visibility eye, at the head of an asset row with the name inset past it.
pub(crate) fn eye_rect(o: [f32; 2], w: f32, slot: usize) -> [f32; 4] {
    let r = row_rect(o, w, slot);
    [
        r[0] + PAD,
        r[1] + (ROW_H - BOX_SIZE) * 0.5,
        BOX_SIZE,
        BOX_SIZE,
    ]
}

// The pick lock, just right of the eye.
pub(crate) fn lock_rect(o: [f32; 2], w: f32, slot: usize) -> [f32; 4] {
    let e = eye_rect(o, w, slot);
    [e[0] + BOX_SIZE + ICON_GAP, e[1], BOX_SIZE, BOX_SIZE]
}

// The triple-dot button, in its own reserved slot at the row's right end. The
// space is held on every row even though the dots only show on hover, so the
// type beside them never has to move or hide to make room.
fn dot_rect(o: [f32; 2], w: f32, slot: usize) -> [f32; 4] {
    let r = row_rect(o, w, slot);
    [
        r[0] + r[2] - PAD - DOT_SZ,
        r[1] + (ROW_H - DOT_SZ) * 0.5,
        DOT_SZ,
        DOT_SZ,
    ]
}

// The right edge the asset type is right-aligned against: just inside the
// triple-dot's reserved slot.
fn type_right(o: [f32; 2], w: f32, slot: usize) -> f32 {
    dot_rect(o, w, slot)[0] - GAP
}

// A picker option row, floating over the tree body, at the effective width `w`.
pub(crate) fn picker_option_rect(o: [f32; 2], w: f32, slot: usize) -> [f32; 4] {
    [o[0], list_top(o) + slot as f32 * ROW_H, w, ROW_H]
}

// The row menu, floating just below the asset row at visible slot `slot`. It
// offers only Delete, so it opens for the authored rows that declare a
// world.jsonl line to remove; a generated asset has none (it is removed by
// editing whatever produced it) and gets no menu. Hide and lock have their own
// row icons, and a row-body click already opens the edit form.
//
// Returns (background, delete row).
fn menu_rects(o: [f32; 2], w: f32, slot: usize) -> ([f32; 4], [f32; 4]) {
    let x = menu_left(o, w);
    let top = list_top(o) + slot as f32 * ROW_H + ROW_H;
    let bg = [x, top, MENU_W, MENU_ROW_H];
    let delete = [x, top, MENU_W, MENU_ROW_H];
    (bg, delete)
}

// The menu's left edge. Only the right-aligned type captions reach under it:
// a row's name is clipped well short of this (`name_captions_clear_the_menu`).
fn menu_left(o: [f32; 2], w: f32) -> f32 {
    o[0] + w - MENU_W - SCROLLBAR_W - 2.0
}

// The single body row the open menu covers, so its type caption can be blanked:
// all TextLabels draw after all Sprites, so the menu's opaque backing cannot
// hide a caption underneath it -- only not drawing it can.
fn menu_covered_slots(slot: usize, rows: usize) -> std::ops::Range<usize> {
    (slot + 1)..(slot + 2).min(rows)
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
    // An asset row's triple-dot: open its menu.
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

    // The visible slot of the asset the open menu belongs to, when that asset is
    // still on screen and has a world.jsonl line to delete: an authored asset's
    // own line, or an overridden instance's patch line (deleting it reverts the
    // instance to its template).
    fn menu_target(&self, rows: usize) -> Option<usize> {
        let name = self.row_menu?;
        (0..rows).find(|&slot| {
            matches!(
                self.rows.get(self.scroll + slot),
                Some(TreeRow::Asset { name: n, badge, .. })
                    if n == name && matches!(badge, Badge::Authored | Badge::Overridden)
            )
        })
    }
}

// Resolve a click against the open panel at origin `o`. `None` means the click
// missed the panel entirely (the caller lets it fall through). Text-field clicks
// resolve to `FocusSearch` / `Consume` (swallowed here; the engine's text-input
// system focuses the field from the same input). Title-bar presses never reach
// this: the hook intercepts them first to start a drag.
pub(crate) fn hit_test(
    view: &PanelView,
    mx: f32,
    my: f32,
    o: [f32; 2],
    s: [f32; 2],
) -> Option<PanelAction> {
    let w = s[0];
    let rows = visible_rows(s[1]);
    // An open row menu is modal over the panel: its Delete row picks, anything
    // else dismisses it.
    if view.row_menu.is_some() {
        if let Some(slot) = view.menu_target(rows) {
            let (_, delete) = menu_rects(o, w, slot);
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
        if point_in(mx, my, search_rect(o, w)) {
            return Some(PanelAction::Consume);
        }
        let scroll = view.picker_scroll.min(options.len().saturating_sub(1));
        for slot in 0..rows {
            let idx = scroll + slot;
            if idx >= options.len() {
                break;
            }
            if point_in(mx, my, picker_option_rect(o, w, slot)) {
                return Some(PanelAction::PickOption(idx));
            }
        }
        return Some(PanelAction::CloseOverlays);
    }

    // Picker closed: clicks outside the panel fall through (the hook's caller
    // decides who else wants them).
    if !point_in(mx, my, widget::outer_rect(o, s)) {
        return None;
    }
    if point_in(mx, my, plus_rect(o)) {
        return Some(PanelAction::TogglePicker);
    }
    if point_in(mx, my, search_rect(o, w)) {
        return Some(PanelAction::FocusSearch);
    }

    for slot in 0..rows {
        if !point_in(mx, my, row_rect(o, w, slot)) {
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
                badge,
                ..
            } => {
                if point_in(mx, my, eye_rect(o, w, slot)) {
                    PanelAction::ToggleHide(*group, *index)
                } else if point_in(mx, my, lock_rect(o, w, slot)) {
                    PanelAction::ToggleLock(*group, *index)
                } else if matches!(badge, Badge::Authored | Badge::Overridden)
                    && point_in(mx, my, dot_rect(o, w, slot))
                {
                    // The menu offers only Delete, so it opens for the rows
                    // with a world.jsonl line to remove (an overridden
                    // instance's Delete reverts it to its template).
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
pub(crate) fn apply(world: &mut World, view: Option<&PanelView>, o: [f32; 2], s: [f32; 2]) {
    let Some(view) = view else {
        hide_all(world);
        return;
    };

    // Blank everything, then re-show what this frame needs.
    hide_all(world);

    let w = s[0];
    widget::place_panel(world, PANEL_BG, widget::outer_rect(o, s));
    let title = widget::title_rect(o, w);
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
    widget::show_field(world, SEARCH_INPUT, search_rect(o, w), focused);
    layout_status(world, view, o);

    match view.picker_options {
        Some(options) => layout_picker(world, view, options, o, s),
        None => layout_tree(world, view, o, s),
    }
}

// The status line: the asset count, or a cook failure in its place.
fn layout_status(world: &mut World, view: &PanelView, o: [f32; 2]) {
    let (color, text) = match view.status {
        Some(e) => (ERROR_LABEL, e.to_string()),
        None => (LABEL_DIM, format!("Assets ({})", view.total)),
    };
    widget::place_message(
        world,
        STATUS_LABEL,
        [
            o[0] + PAD,
            widget::header_y(o, 0.0) + HEADER_H,
            (PANEL_W - 2.0 * PAD).max(0.0),
            widget::LINE_H,
        ],
        &text,
        color,
        true,
    );
}

fn layout_tree(world: &mut World, view: &PanelView, o: [f32; 2], s: [f32; 2]) {
    let w = s[0];
    let rows = visible_rows(s[1]);
    if view.rows.is_empty() {
        // A world that does not cook has no tree to show; the status line
        // already says which, so this only covers the genuinely empty world.
        if view.status.is_none() {
            widget::place_left_label(
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
    for slot in 0..rows {
        let Some(row) = view.rows.get(scroll + slot) else {
            break;
        };
        let r = row_rect(o, w, slot);
        let hovered = point_in(view.mouse[0], view.mouse[1], r);
        match row {
            TreeRow::Header {
                label, count, open, ..
            } => layout_header_row(world, slot, r, label, *count, *open, hovered),
            TreeRow::Asset {
                name,
                asset_type,
                badge,
                ..
            } => {
                let asset = AssetRow {
                    name,
                    asset_type,
                    badge,
                    hovered,
                };
                layout_asset_row(world, view, o, w, slot, &asset);
                // The triple-dot opens the Delete menu, so it shows only on the
                // authored rows that have a line to delete. It follows the row
                // whose menu is open, else the hovered row; its own slot keeps
                // the type beside it reading either way.
                let menu_here = view.row_menu == Some(name.as_str());
                if matches!(badge, Badge::Authored | Badge::Overridden) && (menu_here || hovered) {
                    let over_dots = point_in(view.mouse[0], view.mouse[1], dot_rect(o, w, slot));
                    place_dot(world, o, w, slot, menu_here || over_dots);
                }
            }
        }
    }
    if let Some(slot) = view.menu_target(rows) {
        layout_row_menu(world, view, o, w, rows, slot);
    }
    layout_scrollbar(world, total, scroll, o, w, rows);
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
    widget::place_left_label(
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
    w: f32,
    slot: usize,
    asset: &AssetRow,
) {
    let AssetRow {
        name,
        asset_type,
        badge,
        hovered,
    } = *asset;
    let r = row_rect(o, w, slot);
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
    widget::place_left_label(
        world,
        name_label(slot),
        [r[0] + ASSET_INDENT, r[1] + ROW_H * 0.5 - theme::TEXT_HALF],
        &widget::clip_text(name, name_budget(w)),
        // The hidden state dims the name too, so a collapsed object reads as
        // absent at a glance.
        if hidden { LABEL_DIM } else { LABEL },
        true,
    );
    if let Some(l) = widget::label_mut(world, type_label(slot)) {
        l.x = type_right(o, w, slot);
        l.y = r[1] + ROW_H * 0.5 - theme::TEXT_HALF;
        l.align = TextAlign::Right;
        l.color = badge_color(*badge);
        l.visible = true;
        l.content = widget::clip_text(asset_type, MAX_TYPE_CHARS);
    }
    place_eye(world, eye_rect(o, w, slot), slot, hidden);
    // The lock shows solid while locked, and as a faint hint on row hover so it
    // stays discoverable; otherwise it is blank, leaving only the eye.
    let locked = view.locked.contains(name);
    place_lock(world, lock_rect(o, w, slot), slot, locked, hovered);
}

// A hollow "ring" sprite: a transparent fill with a `color` outline of `border`
// width, rounded to `radius`. Used for the open eye's lens and the lock shackle.
fn place_ring(
    world: &mut World,
    id: AssetId,
    rect: [f32; 4],
    color: [f32; 4],
    radius: f32,
    border: f32,
) {
    if let Some(s) = widget::sprite_mut(world, id) {
        s.x = rect[0];
        s.y = rect[1];
        s.width = rect[2];
        s.height = rect[3];
        s.tint = [0.0, 0.0, 0.0, 0.0];
        s.corner_radius = radius;
        s.border_width = border;
        s.border_color = color;
        s.visible = true;
    }
}

// Draw the visibility eye in `region`: an open lens (a rounded outline with a
// pupil) while the asset renders, a single flat lid line once hidden.
fn place_eye(world: &mut World, region: [f32; 4], slot: usize, hidden: bool) {
    let cx = region[0] + region[2] * 0.5;
    let cy = region[1] + region[3] * 0.5;
    let w = region[2] - 1.0;
    if hidden {
        // Closed: one flat lid bar, pupil blanked.
        let h = 2.5;
        place_rounded(
            world,
            eye_ring(slot),
            [cx - w * 0.5, cy - h * 0.5, w, h],
            EYE_TINT_HIDDEN,
            h * 0.5,
            true,
        );
        widget::set_sprite_visible(world, eye_pupil(slot), false);
    } else {
        // Open: a stadium lens outline with a filled pupil at its centre.
        let h = 8.5;
        place_ring(
            world,
            eye_ring(slot),
            [cx - w * 0.5, cy - h * 0.5, w, h],
            EYE_TINT,
            h * 0.5,
            ICON_STROKE,
        );
        let p = 4.5;
        place_rounded(
            world,
            eye_pupil(slot),
            [cx - p * 0.5, cy - p * 0.5, p, p],
            EYE_TINT,
            p * 0.5,
            true,
        );
    }
}

// Draw the pick lock in `region`: a padlock whose shackle arch sits behind a
// solid body (the body is drawn after, hiding the shackle's lower half). Solid
// amber while `locked`; a faint hint while unlocked but `hovered`; blank
// otherwise.
fn place_lock(world: &mut World, region: [f32; 4], slot: usize, locked: bool, hovered: bool) {
    if !locked && !hovered {
        widget::set_sprite_visible(world, lock_shackle(slot), false);
        widget::set_sprite_visible(world, lock_body(slot), false);
        return;
    }
    let color = if locked { LOCK_TINT } else { LOCK_TINT_HINT };
    let cx = region[0] + region[2] * 0.5;
    let cy = region[1] + region[3] * 0.5;
    let (body_w, body_h) = (11.0, 8.0);
    let (shackle_w, shackle_h) = (7.0, 8.0);
    // The shackle's lower `overlap` tucks behind the body, leaving an arch.
    let overlap = 3.0;
    let top = cy - (shackle_h + body_h - overlap) * 0.5;
    place_ring(
        world,
        lock_shackle(slot),
        [cx - shackle_w * 0.5, top, shackle_w, shackle_h],
        color,
        shackle_w * 0.5,
        ICON_STROKE,
    );
    place_rounded(
        world,
        lock_body(slot),
        [cx - body_w * 0.5, top + shackle_h - overlap, body_w, body_h],
        color,
        2.5,
        true,
    );
}

// The floating type-picker list, over the tree body.
fn layout_picker(
    world: &mut World,
    view: &PanelView,
    options: &[String],
    o: [f32; 2],
    s: [f32; 2],
) {
    let w = s[0];
    let rows = visible_rows(s[1]);
    let total = options.len();
    let scroll = view.picker_scroll.min(total.saturating_sub(1));
    let shown = total.saturating_sub(scroll).clamp(1, rows);
    let backing = [o[0], list_top(o), w, shown as f32 * ROW_H + PAD];
    place_sprite(world, PICKER_BG, backing, PICKER_BG_TINT, true);
    if total == 0 {
        widget::place_left_label(
            world,
            EMPTY_LABEL,
            [o[0] + PAD, list_top(o) + PAD],
            "No matching types",
            LABEL_DIM,
            true,
        );
        return;
    }
    for slot in 0..rows {
        let idx = scroll + slot;
        if idx >= total {
            break;
        }
        let rect = picker_option_rect(o, w, slot);
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
        widget::place_left_label(
            world,
            picker_row_label(slot),
            [rect[0] + PAD, rect[1] + ROW_H * 0.5 - theme::TEXT_HALF],
            &options[idx],
            LABEL,
            true,
        );
    }
    layout_scrollbar(world, total, scroll, o, w, rows);
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
fn place_dot(world: &mut World, o: [f32; 2], w: f32, slot: usize, show_box: bool) {
    let d = dot_rect(o, w, slot);
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

fn layout_row_menu(
    world: &mut World,
    view: &PanelView,
    o: [f32; 2],
    w: f32,
    rows: usize,
    slot: usize,
) {
    let (bg, delete) = menu_rects(o, w, slot);
    // Blank the type caption the menu floats over first: every TextLabel draws
    // after every Sprite, so the opaque backing below cannot cover it. The names
    // stay -- they sit well left of the menu, so hiding them would blank rows the
    // user can still see.
    for covered in menu_covered_slots(slot, rows) {
        widget::set_label_visible(world, type_label(covered), false);
    }
    if let Some(sprite) = widget::sprite_mut(world, MENU_BG) {
        sprite.x = bg[0];
        sprite.y = bg[1];
        sprite.width = bg[2];
        sprite.height = bg[3];
        sprite.tint = MENU_BG_TINT;
        sprite.corner_radius = theme::CONTROL_RADIUS;
        sprite.border_width = theme::PANEL_BORDER_WIDTH;
        sprite.border_color = theme::PANEL_BORDER_TINT;
        sprite.visible = true;
    }
    place_menu_row(
        world,
        (MENU_DELETE_BG, MENU_DELETE_LABEL),
        delete,
        view.mouse,
        "Delete",
        DELETE_LABEL,
    );
}

fn place_menu_row(
    world: &mut World,
    ids: (AssetId, AssetId),
    rect: [f32; 4],
    mouse: [f32; 2],
    caption: &str,
    color: [f32; 3],
) {
    let hovered = point_in(mouse[0], mouse[1], rect);
    place_rounded(
        world,
        ids.0,
        rect,
        if hovered {
            MENU_ROW_HOVER
        } else {
            MENU_ROW_TINT
        },
        theme::CONTROL_RADIUS,
        true,
    );
    widget::place_left_label(
        world,
        ids.1,
        [rect[0] + PAD, rect[1] + MENU_ROW_H * 0.5 - theme::TEXT_HALF],
        caption,
        color,
        true,
    );
}

// A simple non-interactive scrollbar sizing the visible window against `total`,
// down the panel's right edge. Shown only when the body overflows.
fn layout_scrollbar(
    world: &mut World,
    total: usize,
    scroll: usize,
    o: [f32; 2],
    w: f32,
    rows: usize,
) {
    if total <= rows {
        return;
    }
    let x = o[0] + w - SCROLLBAR_W;
    let top = list_top(o);
    let h = rows as f32 * ROW_H;
    place_rounded(
        world,
        LIST_TRACK,
        [x, top, SCROLLBAR_W, h],
        TRACK_TINT,
        SCROLLBAR_W * 0.5,
        true,
    );
    let thumb_h = (h * rows as f32 / total as f32).max(18.0);
    let max_scroll = (total - rows) as f32;
    let off = (h - thumb_h) * (scroll.min(total - rows) as f32 / max_scroll);
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
    ids.extend((0..ROW_POOL_MAX).map(row_bg));
    // The eye then the lock, each family drawn before the next so a lock body
    // paints over its own shackle arch.
    ids.extend((0..ROW_POOL_MAX).map(eye_ring));
    ids.extend((0..ROW_POOL_MAX).map(eye_pupil));
    ids.extend((0..ROW_POOL_MAX).map(lock_shackle));
    ids.extend((0..ROW_POOL_MAX).map(lock_body));
    ids.push(PICKER_BG);
    ids.extend((0..ROW_POOL_MAX).map(picker_row_bg));
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
    ids.extend((0..ROW_POOL_MAX).map(name_label));
    ids.extend((0..ROW_POOL_MAX).map(type_label));
    ids.extend((0..ROW_POOL_MAX).map(picker_row_label));
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
    widget::hide_all(world, &all_sprite_ids(), &all_label_ids(), &all_field_ids());
}

// Whether the cursor is over the scrollable body area (for wheel scrolling).
pub(crate) fn cursor_over_body(mx: f32, my: f32, o: [f32; 2], s: [f32; 2]) -> bool {
    let p = widget::outer_rect(o, s);
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
            concinnity_store::paths::set_root(dir);
        });
    }

    // An upper bound on the body font's per-character advance (it measures
    // ~8.2px), for the geometry guards that keep a clipped caption clear of the
    // chrome beside it.
    const MAX_CHAR_W: f32 = 10.0;

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

    // A generated asset row: no world.jsonl line, so its menu offers no Delete.
    fn asset(group: usize, index: usize, name: &str) -> TreeRow {
        TreeRow::Asset {
            group,
            index,
            name: name.to_string(),
            asset_type: "Material".to_string(),
            badge: Badge::Imported,
        }
    }

    // A row for one of the world's own lines, which the menu can delete.
    fn authored(group: usize, index: usize, name: &str) -> TreeRow {
        match asset(group, index, name) {
            TreeRow::Asset {
                group,
                index,
                name,
                asset_type,
                ..
            } => TreeRow::Asset {
                group,
                index,
                name,
                asset_type,
                badge: Badge::Authored,
            },
            row => row,
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
                    authored(0, 0, "cam"),
                    authored(0, 1, "lamp"),
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
        let p = widget::outer_rect(o, size());
        assert_eq!(p[0] + p[2], 1280.0, "flush to the window right");
        assert_eq!(p[1], hud::body_top(), "starts below the top-bar buttons");
        // The whole panel follows its origin: dragged elsewhere, every rect moves.
        let moved = widget::outer_rect([40.0, 60.0], size());
        assert_eq!((moved[0], moved[1]), (40.0, 60.0));
        assert_eq!((moved[2], moved[3]), (p[2], p[3]));
        assert_eq!(size(), [p[2], p[3]], "the drag-clamp footprint matches");
    }

    // The header stacks under the title bar, the tree under the status line, and
    // every row control stays inside its row without overlapping its neighbour.
    #[test]
    fn geometry_stacks_and_row_controls_stay_inside_the_row() {
        let o = test_origin();
        assert_eq!(
            widget::title_rect(o, PANEL_W),
            [o[0], o[1], PANEL_W, widget::TITLE_H]
        );
        let plus = plus_rect(o);
        let search = search_rect(o, PANEL_W);
        assert!(plus[0] + plus[2] <= search[0], "+ sits left of the search");
        assert_eq!(
            search[0] + search[2],
            o[0] + PANEL_W - PAD,
            "the search field reaches the panel's right pad"
        );
        assert_eq!(row_rect(o, PANEL_W, 0)[1], list_top(o));
        assert_eq!(row_rect(o, PANEL_W, 1)[1], list_top(o) + ROW_H);

        let r = row_rect(o, PANEL_W, 0);
        let (eye, lock, dots) = (
            eye_rect(o, PANEL_W, 0),
            lock_rect(o, PANEL_W, 0),
            dot_rect(o, PANEL_W, 0),
        );
        // The eye heads the row, the lock sits just right of it, and the name is
        // inset past both.
        assert_eq!(eye[0], r[0] + PAD, "the eye sits at the row's head");
        assert_eq!(
            lock[0],
            eye[0] + eye[2] + ICON_GAP,
            "the lock follows the eye"
        );
        assert!(
            lock[0] + lock[2] <= r[0] + ASSET_INDENT,
            "the name starts clear of both head icons"
        );
        assert!(eye[1] >= r[1] && eye[1] + eye[3] <= r[1] + r[3]);
        assert!(lock[1] >= r[1] && lock[1] + lock[3] <= r[1] + r[3]);
        assert!(
            dots[0] + dots[2] <= r[0] + r[2],
            "the triple-dot stays inside the row"
        );
        assert!(
            type_right(o, PANEL_W, 0) <= dots[0],
            "the type is right-aligned clear of the dot slot"
        );
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
            hit_test(&f.view(), plus[0] + 5.0, plus[1] + 5.0, o, size()),
            Some(PanelAction::TogglePicker)
        );
        // Open, the same button is the "X" that returns to the tree.
        assert_eq!(
            hit_test(&f.picker_view(), plus[0] + 5.0, plus[1] + 5.0, o, size()),
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
        apply(&mut world, Some(&f.view()), o, size());
        let l = label(&world, PLUS_LABEL);
        assert_eq!(l.content, "+");
        assert_eq!(sprite(&world, PLUS_BG).tint, PLUS_TINT);
        assert!(
            l.scale > 1.0,
            "the glyph draws larger than the body text (the box is unchanged)"
        );
        apply(&mut world, Some(&f.picker_view()), o, size());
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
        apply(&mut world, Some(&f.view()), o, size());
        let title = label(&world, TITLE_LABEL);
        assert!(title.visible);
        assert_eq!(title.content, "Assets");
        assert!(sprite(&world, PANEL_BG).visible);
        let t = widget::title_rect(o, PANEL_W);
        assert_eq!(
            hit_test(&f.view(), t[0] + 5.0, t[1] + 5.0, o, size()),
            Some(PanelAction::Consume)
        );
    }

    #[test]
    fn hit_test_resolves_search_headers_rows_and_toggles() {
        let f = Fixture::new();
        let o = test_origin();
        let s = search_rect(o, PANEL_W);
        assert_eq!(
            hit_test(&f.view(), s[0] + 5.0, s[1] + 5.0, o, size()),
            Some(PanelAction::FocusSearch)
        );
        let r0 = row_rect(o, PANEL_W, 0);
        assert_eq!(
            hit_test(&f.view(), r0[0] + 5.0, r0[1] + 5.0, o, size()),
            Some(PanelAction::ToggleGroup(0))
        );
        let r1 = row_rect(o, PANEL_W, 1);
        assert_eq!(
            hit_test(&f.view(), r1[0] + 5.0, r1[1] + 5.0, o, size()),
            Some(PanelAction::SelectRow(0, 0))
        );
        let e = eye_rect(o, PANEL_W, 1);
        assert_eq!(
            hit_test(&f.view(), e[0] + 2.0, e[1] + 2.0, o, size()),
            Some(PanelAction::ToggleHide(0, 0))
        );
        let lk = lock_rect(o, PANEL_W, 1);
        assert_eq!(
            hit_test(&f.view(), lk[0] + 2.0, lk[1] + 2.0, o, size()),
            Some(PanelAction::ToggleLock(0, 0))
        );
        // Row 1 is an authored line, so its triple-dot opens the Delete menu.
        let d = dot_rect(o, PANEL_W, 1);
        assert_eq!(
            hit_test(&f.view(), d[0] + 2.0, d[1] + 2.0, o, size()),
            Some(PanelAction::OpenRowMenu(0, 0))
        );
        // Past the last row the click is swallowed; off the panel it misses.
        let r9 = row_rect(o, PANEL_W, 9);
        assert_eq!(
            hit_test(&f.view(), r9[0] + 5.0, r9[1] + 5.0, o, size()),
            Some(PanelAction::Consume)
        );
        assert_eq!(hit_test(&f.view(), 5000.0, 5000.0, o, size()), None);
    }

    // A row the build emits unconditionally has no world.jsonl line, so its
    // triple-dot (Delete only) never appears: the dot area selects the row
    // instead. The lock still applies to anything the viewport can pick, so its
    // icon toggles as on any row.
    #[test]
    fn a_generated_row_has_no_menu_but_still_locks() {
        let mut f = Fixture::new();
        f.rows = vec![header(0, "Other expansions", 1, true), asset(0, 0, "tab")];
        let o = test_origin();
        let d = dot_rect(o, PANEL_W, 1);
        assert_eq!(
            hit_test(&f.view(), d[0] + 2.0, d[1] + 2.0, o, size()),
            Some(PanelAction::SelectRow(0, 0)),
            "a generated row's dot slot selects: it has no Delete menu"
        );
        let lk = lock_rect(o, PANEL_W, 1);
        assert_eq!(
            hit_test(&f.view(), lk[0] + 2.0, lk[1] + 2.0, o, size()),
            Some(PanelAction::ToggleLock(0, 0))
        );

        // Even if a menu were somehow open on it, a generated row draws none.
        let v = PanelView {
            row_menu: Some("tab"),
            ..f.view()
        };
        assert!(
            v.menu_target(ROW_POOL).is_none(),
            "no menu target for a generated row"
        );
        assert_eq!(
            hit_test(&v, o[0] + 5.0, list_top(o) + ROW_H + 5.0, o, size()),
            Some(PanelAction::CloseOverlays)
        );
        let mut world = injected_world();
        apply(&mut world, Some(&v), o, size());
        assert!(!sprite(&world, MENU_BG).visible);
        assert!(!label(&world, MENU_DELETE_LABEL).visible);
    }

    #[test]
    fn scrolled_rows_resolve_through_the_window_offset() {
        let f = Fixture::new();
        let o = test_origin();
        let v = PanelView {
            scroll: 2,
            ..f.view()
        };
        let r0 = row_rect(o, PANEL_W, 0);
        assert_eq!(
            hit_test(&v, r0[0] + 5.0, r0[1] + 5.0, o, size()),
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
        apply(&mut world, Some(&f.view()), o, size());

        assert_eq!(label(&world, STATUS_LABEL).content, "Assets (3)");
        assert_eq!(label(&world, name_label(0)).content, "- World (2)");
        assert!(
            !sprite(&world, eye_ring(0)).visible && !sprite(&world, lock_shackle(0)).visible,
            "header rows draw no toggles"
        );
        // The folded group header shows its fold marker.
        assert_eq!(label(&world, name_label(3)).content, "+ fox (1)");

        // cam: selected, visible, and locked. The open eye is an outlined lens
        // plus a pupil, and the lock draws amber.
        assert_eq!(sprite(&world, row_bg(1)).tint, ROW_TINT_SELECTED);
        assert_eq!(sprite(&world, eye_ring(1)).border_color, EYE_TINT);
        assert!(sprite(&world, eye_pupil(1)).visible);
        assert_eq!(sprite(&world, lock_body(1)).tint, LOCK_TINT);
        assert!(sprite(&world, lock_shackle(1)).visible);
        let ty = label(&world, type_label(1));
        assert_eq!(ty.content, "Material");
        assert_eq!(ty.align, TextAlign::Right);
        assert_eq!(ty.color, badge_color(Badge::Authored));

        // lamp: hidden dims the name and closes the eye (the lid bar, no pupil);
        // unlocked and unhovered, its lock is blank.
        assert_eq!(sprite(&world, eye_ring(2)).tint, EYE_TINT_HIDDEN);
        assert!(!sprite(&world, eye_pupil(2)).visible);
        assert!(!sprite(&world, lock_body(2)).visible);
        assert_eq!(label(&world, name_label(2)).color, LABEL_DIM);

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
        apply(&mut world, Some(&f.view()), o, size());
        let t = field(&world);
        assert!(t.visible && !t.focused, "shown, unfocused until clicked");

        apply(&mut world, Some(&f.picker_view()), o, size());
        assert!(field(&world).focused, "the picker types into the field");
    }

    // The triple-dot has a slot of its own at the row's right end, so revealing
    // it on hover neither moves nor hides the type beside it.
    #[test]
    fn the_triple_dot_keeps_its_own_slot_beside_the_type() {
        let mut world = injected_world();
        let f = Fixture::new();
        let o = test_origin();
        let r1 = row_rect(o, PANEL_W, 1);

        // Unhovered: the type reads, and the dot slot is already reserved.
        apply(&mut world, Some(&f.view()), o, size());
        let resting = label(&world, type_label(1));
        assert!(resting.visible && !sprite(&world, DOT1).visible);

        let v = PanelView {
            mouse: [r1[0] + 5.0, r1[1] + 5.0],
            ..f.view()
        };
        apply(&mut world, Some(&v), o, size());
        assert!(sprite(&world, DOT1).visible, "hover reveals the dots");
        let hovered = label(&world, type_label(1));
        assert!(hovered.visible, "the hovered row's type keeps reading");
        assert_eq!(hovered.x, resting.x, "and does not shift to make room");
        assert!(
            !sprite(&world, DOT_BG).visible,
            "the box shows only over the dots themselves"
        );
        // Over the dots proper, the background box appears too.
        let d = dot_rect(o, PANEL_W, 1);
        let over = PanelView {
            mouse: [d[0] + 2.0, d[1] + 2.0],
            ..f.view()
        };
        apply(&mut world, Some(&over), o, size());
        assert!(sprite(&world, DOT_BG).visible);
    }

    // An authored row's menu offers Delete, and is modal: its row picks,
    // anything else dismisses it.
    #[test]
    fn the_row_menu_picks_delete_and_dismisses_elsewhere() {
        let f = Fixture::new();
        let o = test_origin();
        let v = PanelView {
            row_menu: Some("cam"),
            ..f.view()
        };
        let (_, delete) = menu_rects(o, PANEL_W, 1);
        assert_eq!(
            hit_test(&v, delete[0] + 5.0, delete[1] + 5.0, o, size()),
            Some(PanelAction::RowDelete)
        );
        assert_eq!(
            hit_test(&v, o[0] + 5.0, o[1] + 5.0, o, size()),
            Some(PanelAction::CloseOverlays)
        );

        let mut world = injected_world();
        apply(&mut world, Some(&v), o, size());
        assert!(sprite(&world, MENU_BG).visible);
        assert_eq!(label(&world, MENU_DELETE_LABEL).content, "Delete");
    }

    // Every TextLabel draws after every Sprite, so the menu's opaque backing
    // cannot cover a row caption underneath it: the covered rows must not draw
    // their captions at all, or they read straight through the menu.
    #[test]
    fn the_row_menu_blanks_the_captions_it_covers() {
        let mut world = injected_world();
        let mut f = Fixture::new();
        // Rows past the menu's reach, to check it blanks only what it covers.
        f.rows.push(authored(0, 2, "past_a"));
        f.rows.push(authored(0, 3, "past_b"));
        let o = test_origin();
        let v = PanelView {
            row_menu: Some("cam"),
            ..f.view()
        };
        apply(&mut world, Some(&v), o, size());
        // The menu opens under slot 1 (cam) and is one row tall (Delete), so it
        // covers slot 2 (lamp). Slot 3 is the fox header, slot 4 the past_a row.
        assert!(
            !label(&world, type_label(2)).visible,
            "slot 2's type would read through the menu"
        );
        assert!(
            label(&world, name_label(2)).visible,
            "slot 2's name sits left of the menu and still reads"
        );
        assert!(
            label(&world, type_label(4)).visible,
            "an asset row past the menu keeps its type"
        );
        assert!(label(&world, MENU_DELETE_LABEL).visible);
    }

    // The names are left alone when the menu opens, which is only safe while a
    // clipped name cannot reach the menu's left edge.
    #[test]
    fn name_captions_clear_the_menu() {
        let o = test_origin();
        let name_end = o[0] + ASSET_INDENT + MAX_NAME_CHARS as f32 * MAX_CHAR_W;
        assert!(
            name_end <= menu_left(o, PANEL_W),
            "a full-width name would run under the row menu"
        );
        // The type is clipped to what fits between the name and the dot slot.
        let type_start = type_right(o, PANEL_W, 0) - MAX_TYPE_CHARS as f32 * MAX_CHAR_W;
        assert!(
            name_end <= type_start,
            "a full-width name would collide with its type"
        );
    }

    // An open row menu whose asset scrolled out of the visible window has no
    // hittable rows, so any click just closes the overlay.
    #[test]
    fn row_menu_dismisses_when_its_asset_is_off_window() {
        let f = Fixture::new();
        let o = test_origin();
        let v = PanelView {
            row_menu: Some("not_listed"),
            ..f.view()
        };
        let (_, delete) = menu_rects(o, PANEL_W, 1);
        assert_eq!(
            hit_test(&v, delete[0] + 5.0, delete[1] + 5.0, o, size()),
            Some(PanelAction::CloseOverlays)
        );
    }

    #[test]
    fn picker_options_pick_and_dismiss() {
        let mut f = Fixture::new();
        f.picker_options = vec!["Decal".to_string(), "PointLight".to_string()];
        let o = test_origin();
        let v = f.picker_view();
        let r1 = picker_option_rect(o, PANEL_W, 1);
        assert_eq!(
            hit_test(&v, r1[0] + 5.0, r1[1] + 5.0, o, size()),
            Some(PanelAction::PickOption(1))
        );
        // The header field keeps focus (consumed), and empty body space closes.
        let s = search_rect(o, PANEL_W);
        assert_eq!(
            hit_test(&v, s[0] + 5.0, s[1] + 5.0, o, size()),
            Some(PanelAction::Consume)
        );
        let r13 = picker_option_rect(o, PANEL_W, 13);
        assert_eq!(
            hit_test(&v, r13[0] + 5.0, r13[1] + 5.0, o, size()),
            Some(PanelAction::CloseOverlays)
        );
    }

    #[test]
    fn picker_shows_an_empty_state_with_no_matching_options() {
        let f = Fixture::new();
        let mut world = injected_world();
        apply(&mut world, Some(&f.picker_view()), test_origin(), size());
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
        apply(&mut world, Some(&v), test_origin(), size());
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
        apply(&mut world, Some(&v), test_origin(), size());
        assert_eq!(label(&world, EMPTY_LABEL).content, "No matching assets");
    }

    #[test]
    fn long_tree_shows_the_scrollbar() {
        let mut world = injected_world();
        let mut f = Fixture::new();
        for i in 0..20 {
            f.rows.push(asset(1, i, &format!("a{i}")));
        }
        let v = PanelView {
            scroll: 3,
            ..f.view()
        };
        apply(&mut world, Some(&v), test_origin(), size());
        assert!(sprite(&world, LIST_THUMB).visible);
        // A short tree hides it again.
        let short = Fixture::new();
        apply(&mut world, Some(&short.view()), test_origin(), size());
        assert!(!sprite(&world, LIST_THUMB).visible);
    }

    #[test]
    fn hide_all_blanks_every_element() {
        let mut world = injected_world();
        let f = Fixture::new();
        apply(&mut world, Some(&f.view()), test_origin(), size());
        apply(&mut world, None, [0.0, 0.0], size());
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
        assert!(cursor_over_body(o[0] + 5.0, list_top(o) + 5.0, o, size()));
        assert!(
            !cursor_over_body(o[0] + 5.0, o[1] + 5.0, o, size()),
            "the title bar is not the scrollable body"
        );
        assert!(!cursor_over_body(o[0] - 5.0, list_top(o) + 5.0, o, size()));
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
            if let Some(ct) = concinnity_world::registry::ComponentType::parse(ty) {
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
        use concinnity_world::registry::ComponentType;
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

    // The default height shows exactly ROW_POOL rows; a taller panel reveals more,
    // capped at the injected pool.
    #[test]
    fn taller_size_reveals_more_rows_up_to_the_pool() {
        assert_eq!(visible_rows(size()[1]), ROW_POOL);
        assert_eq!(visible_rows(size()[1] + 5.0 * ROW_H), ROW_POOL + 5);
        assert_eq!(visible_rows(max_size()[1]), ROW_POOL_MAX);
        assert_eq!(visible_rows(10_000.0), ROW_POOL_MAX, "capped at the pool");
    }

    // A wider panel grows the name budget (fewer clipped names); at the default
    // width it is exactly the baseline.
    #[test]
    fn wider_width_grows_the_name_budget() {
        assert_eq!(name_budget(PANEL_W), MAX_NAME_CHARS);
        assert!(name_budget(PANEL_W + 200.0) > MAX_NAME_CHARS);
    }

    // Growing the panel wider stretches the right-anchored elements (the search
    // field's right edge, a row's right edge, the scrollbar) while the left-anchored
    // parts stay put.
    #[test]
    fn wider_width_stretches_right_anchored_geometry() {
        let o = test_origin();
        let wide = PANEL_W + 160.0;
        assert_eq!(
            search_rect(o, wide)[0] + search_rect(o, wide)[2],
            o[0] + wide - PAD,
            "the search field reaches the widened right pad"
        );
        let r = row_rect(o, wide, 0);
        assert_eq!(
            r[2],
            wide - SCROLLBAR_W - 2.0,
            "the row spans the new width"
        );
    }
}
