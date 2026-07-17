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

use super::expanded::ExpandedRow;
use super::hud;
use super::registry::{self, PanelKey};
use super::theme;
use super::widget::{self, place_rounded, place_sprite, point_in};

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
    "Screen",
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

// Which list the panel body shows. Config is the world's own world.jsonl lines
// (the editable set); Expanded is everything the build adds around them, which
// is most of a world's content yet has no line to browse.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub(crate) enum Tab {
    #[default]
    Config,
    Expanded,
}

impl Tab {
    pub(crate) const ALL: [Tab; 2] = [Tab::Config, Tab::Expanded];

    pub(crate) fn label(self) -> &'static str {
        match self {
            Tab::Config => "Config",
            Tab::Expanded => "Expanded",
        }
    }
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

// Reserved asset-id family for the panel. The asset-id VALUE does not affect
// draw order (the overlay draws in component-insertion order, not by id);
// z-order is set by the sequence in `all_sprite_ids` / `all_label_ids`, which
// `inject.rs` inserts in that order. The id values only need to be distinct.
const PANEL: u32 = registry::base(PanelKey::Assets);
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
// The draggable title bar's heading (the bar is the panel surface itself).
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
pub(crate) fn tab_bg(i: usize) -> AssetId {
    AssetId(PANEL + 0xC0 + i as u32)
}
pub(crate) fn tab_label(i: usize) -> AssetId {
    AssetId(PANEL + 0xC4 + i as u32)
}
// The "+" on an Expanded row that can be copied into the config.
pub(crate) fn add_row_bg(i: usize) -> AssetId {
    AssetId(PANEL + 0xD0 + i as u32)
}
pub(crate) fn add_row_label(i: usize) -> AssetId {
    AssetId(PANEL + 0xE0 + i as u32)
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
const HEADER_H: f32 = 36.0;
const TAB_H: f32 = 26.0;
const GAP: f32 = 6.0;
// How far a tab's chip is inset within its strip cell.
const TAB_INSET: f32 = 3.0;
// The "+" that copies an Expanded row into the config.
const ADD_SZ: f32 = 18.0;
// Characters an Expanded row fits at PANEL_W: a header spans the body width, an
// asset row stops short of its "+". Both are free-length text (an import names
// its output after itself), so they clip rather than overrun the panel.
const HEADER_ROW_CHARS: usize = 33;
const ASSET_ROW_CHARS: usize = 29;
// However long the type is, leave enough of the name to tell rows apart.
const MIN_NAME_CHARS: usize = 8;
// The triple-dot button on a hovered name row.
const DOT_SZ: f32 = 22.0;
// The floating Delete menu.
const MENU_W: f32 = 132.0;
const MENU_ROW_H: f32 = 26.0;

const PLUS_TINT: [f32; 4] = [0.20, 0.44, 0.30, 1.0];
const TYPEDROP_TINT: [f32; 4] = theme::BUTTON_TINT;
// Interactive name-row tints, layered over the shared base `ROW_TINT`.
const ROW_TINT_HOVER: [f32; 4] = theme::HOVER_TINT;
const ROW_TINT_SELECTED: [f32; 4] = theme::SELECTED_TINT;
const OPTION_TINT: [f32; 4] = [0.16, 0.16, 0.20, 0.0];
const OPTION_TINT_HOVER: [f32; 4] = theme::HOVER_TINT;
const OPTION_TINT_SELECTED: [f32; 4] = theme::SELECTED_TINT;
const COMBO_BG_TINT: [f32; 4] = [0.09, 0.09, 0.12, 1.0];
const CANCEL_TINT: [f32; 4] = [0.30, 0.30, 0.34, 1.0];
const DOT_BG_TINT: [f32; 4] = [0.30, 0.34, 0.46, 0.95];
const DOT_TINT: [f32; 4] = [0.90, 0.92, 0.96, 1.0];
const TAB_TINT: [f32; 4] = [0.0, 0.0, 0.0, 0.0];
const TAB_TINT_HOVER: [f32; 4] = theme::HOVER_TINT;
const TAB_TINT_ACTIVE: [f32; 4] = theme::SELECTED_TINT;
// The Expanded tab's group headers read as bands, unlike the Config tab's
// transparent type sub-headers, because they are the clickable control there.
const GROUP_ROW_TINT: [f32; 4] = [0.16, 0.17, 0.22, 1.0];
const ADD_TINT: [f32; 4] = [0.20, 0.44, 0.30, 1.0];
const ADD_TINT_HOVER: [f32; 4] = [0.28, 0.60, 0.40, 1.0];
const MENU_BG_TINT: [f32; 4] = [0.15, 0.15, 0.19, 1.0];
const MENU_ROW_TINT: [f32; 4] = [0.0, 0.0, 0.0, 0.0];
const MENU_ROW_HOVER: [f32; 4] = theme::HOVER_TINT;
const LABEL_DIM: [f32; 3] = theme::LABEL_DIM;
const LABEL_WHITE: [f32; 3] = [1.0, 1.0, 1.0];
const DELETE_LABEL: [f32; 3] = [0.95, 0.60, 0.58];

// Where the panel sits until the user drags it: right-aligned below the top bar.
pub(crate) fn default_origin(vw: f32) -> [f32; 2] {
    [vw - PANEL_W, hud::body_top()]
}

// The panel's footprint, for the hook's drag clamp. Tab-dependent: only the
// Config tab carries the add / filter header.
pub(crate) fn size(tab: Tab) -> [f32; 2] {
    let r = panel_rect([0.0, 0.0], tab);
    [r[2], r[3]]
}

// The panel outer rect (title bar + tabs + header + body) at origin `o`.
pub(crate) fn panel_rect(o: [f32; 2], tab: Tab) -> [f32; 4] {
    [
        o[0],
        o[1],
        PANEL_W,
        body_y(o, tab) - o[1] + MAX_ROWS as f32 * ROW_H,
    ]
}

// The tab strip, spanning the panel width below the title bar.
pub(crate) fn tab_strip_rect(o: [f32; 2]) -> [f32; 4] {
    [o[0], o[1] + widget::TITLE_H, PANEL_W, TAB_H]
}

// Tab `i`, splitting the strip evenly.
pub(crate) fn tab_rect(o: [f32; 2], i: usize) -> [f32; 4] {
    let w = PANEL_W / Tab::ALL.len() as f32;
    [o[0] + i as f32 * w, o[1] + widget::TITLE_H, w, TAB_H]
}

// The draggable title bar across the panel top.
pub(crate) fn title_rect(o: [f32; 2]) -> [f32; 4] {
    [o[0], o[1], PANEL_W, widget::TITLE_H]
}

// The "X" close button in the title bar's top-right corner.
pub(crate) fn close_rect(o: [f32; 2]) -> [f32; 4] {
    widget::close_rect(title_rect(o))
}

// Where the Config tab's add / filter header row begins.
fn header_y(o: [f32; 2]) -> f32 {
    o[1] + widget::TITLE_H + TAB_H
}

// The square "+" add button (panel header below the tabs, left).
pub(crate) fn plus_rect(o: [f32; 2]) -> [f32; 4] {
    [o[0], header_y(o), HEADER_H, HEADER_H]
}

// The combo area (panel header, filling the rest of the row): the browse-filter
// label when closed, the filter text field when open.
pub(crate) fn combo_rect(o: [f32; 2]) -> [f32; 4] {
    [
        o[0] + HEADER_H + GAP,
        header_y(o),
        PANEL_W - HEADER_H - GAP,
        HEADER_H,
    ]
}

// The filter text field, centred in the combo area.
pub(crate) fn filter_input_rect(o: [f32; 2]) -> [f32; 4] {
    let c = combo_rect(o);
    [c[0], c[1] + (HEADER_H - ROW_H) * 0.5, c[2], ROW_H]
}

// Where the body begins: below the tabs, and below the add / filter header on
// the Config tab (the Expanded tab has no header, so its body starts higher).
fn body_y(o: [f32; 2], tab: Tab) -> f32 {
    match tab {
        Tab::Config => header_y(o) + HEADER_H,
        Tab::Expanded => o[1] + widget::TITLE_H + TAB_H,
    }
}

// A body row `i` spanning the panel width (list or combo option).
pub(crate) fn list_row_rect(o: [f32; 2], i: usize) -> [f32; 4] {
    [
        o[0],
        body_y(o, Tab::Config) + i as f32 * ROW_H,
        PANEL_W,
        ROW_H,
    ]
}
pub(crate) fn combo_option_rect(o: [f32; 2], i: usize) -> [f32; 4] {
    list_row_rect(o, i)
}

// An Expanded-tab row `i` (its body starts higher than the Config tab's).
pub(crate) fn expanded_row_rect(o: [f32; 2], i: usize) -> [f32; 4] {
    [
        o[0],
        body_y(o, Tab::Expanded) + i as f32 * ROW_H,
        PANEL_W,
        ROW_H,
    ]
}

// The "+" at the right of an addable Expanded row, left of the scrollbar.
pub(crate) fn add_rect(row: [f32; 4]) -> [f32; 4] {
    [
        row[0] + row[2] - ADD_SZ - SCROLLBAR_W - 6.0,
        row[1] + (ROW_H - ADD_SZ) * 0.5,
        ADD_SZ,
        ADD_SZ,
    ]
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
    let top = body_y(o, Tab::Config) + vr as f32 * ROW_H + ROW_H;
    let delete = [x, top, MENU_W, MENU_ROW_H];
    let bg = [x, top, MENU_W, MENU_ROW_H];
    (bg, delete)
}

// A resolved panel click. Option picks carry an index into the hook's current
// combo option list (the hook maps it to a filter or a type). Row-menu picks
// carry the entry index the menu belongs to.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum PanelAction {
    // A tab in the strip below the title bar.
    SwitchTab(Tab),
    // An Expanded-tab group header: collapse / expand it (carries the group).
    ToggleGroup(usize),
    // An Expanded-tab row's "+": copy that asset's entry into the config
    // (carries the group and the index within it).
    AddExpanded(usize, usize),
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
    // Which body list is showing. The combo / row-menu fields below describe the
    // Config tab and are ignored while the Expanded tab is up.
    pub tab: Tab,
    // The Expanded tab's rows (group headers plus the assets of open groups).
    pub expanded_rows: &'a [ExpandedRow],
    pub expanded_scroll: usize,
    // Why the Expanded tab has no rows, when a cook of the working entries
    // failed rather than there being nothing to show.
    pub expanded_status: Option<&'a str>,
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
    // The tab strip is always live, whichever body is up. Checked before the
    // Config tab's modal overlays so a stray combo can never trap the tabs.
    if point_in(mx, my, tab_strip_rect(o)) {
        for (i, tab) in Tab::ALL.iter().enumerate() {
            if point_in(mx, my, tab_rect(o, i)) {
                return Some(PanelAction::SwitchTab(*tab));
            }
        }
        return Some(PanelAction::Consume);
    }

    if view.tab == Tab::Expanded {
        return hit_test_expanded(view, mx, my, o);
    }

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
    if !point_in(mx, my, panel_rect(o, Tab::Config)) {
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

// Resolve a click on the Expanded tab's body: a group header collapses or
// expands it, an addable row's "+" copies that asset into the config, and the
// rest of a row is inert (a generated asset has nothing to edit until it is
// copied, and an already-copied one is edited from the Config tab).
fn hit_test_expanded(view: &PanelView, mx: f32, my: f32, o: [f32; 2]) -> Option<PanelAction> {
    if !point_in(mx, my, panel_rect(o, Tab::Expanded)) {
        return None;
    }
    let scroll = view
        .expanded_scroll
        .min(view.expanded_rows.len().saturating_sub(1));
    for r in 0..MAX_ROWS {
        let idx = scroll + r;
        if idx >= view.expanded_rows.len() {
            break;
        }
        let rect = expanded_row_rect(o, r);
        if !point_in(mx, my, rect) {
            continue;
        }
        return Some(match &view.expanded_rows[idx] {
            ExpandedRow::Header { group, .. } => PanelAction::ToggleGroup(*group),
            ExpandedRow::Asset {
                group,
                index,
                addable,
                ..
            } if *addable && point_in(mx, my, add_rect(rect)) => {
                PanelAction::AddExpanded(*group, *index)
            }
            ExpandedRow::Asset { .. } => PanelAction::Consume,
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

    widget::place_panel(world, PANEL_BG, panel_rect(o, view.tab));
    widget::place_heading(world, TITLE_LABEL, title_rect(o), "Assets");
    let close_hover = point_in(view.mouse[0], view.mouse[1], close_rect(o));
    widget::place_close(world, CLOSE_BG, CLOSE_LABEL, title_rect(o), close_hover);
    layout_tabs(world, view, o);

    if view.tab == Tab::Expanded {
        layout_expanded(world, view, o);
        return;
    }

    // The "+" add button. While the type picker is open it becomes a gray "X"
    // that returns to the browse list (the previous focus).
    let (glyph, tint) = if view.combo == Combo::Picker {
        ("X", CANCEL_TINT)
    } else {
        ("+", PLUS_TINT)
    };
    place_rounded(
        world,
        PLUS_BG,
        theme::highlight_rect(plus_rect(o)),
        tint,
        theme::CONTROL_RADIUS,
        true,
    );
    place_plus_glyph(world, plus_rect(o), glyph);

    // The combo area: the browse label when closed, the filter field when open.
    if view.combo == Combo::Closed {
        let td = combo_rect(o);
        place_rounded(
            world,
            TYPEDROP_BG,
            theme::highlight_rect(td),
            TYPEDROP_TINT,
            theme::CONTROL_RADIUS,
            true,
        );
        place_left_label(
            world,
            TYPEDROP_LABEL,
            [td[0] + PAD, td[1] + HEADER_H * 0.5 - theme::TEXT_HALF],
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

// The tab strip: each tab is a rounded chip inset in its strip cell; the active
// one is lit, the others recede to the panel surface.
fn layout_tabs(world: &mut World, view: &PanelView, o: [f32; 2]) {
    for (i, tab) in Tab::ALL.iter().enumerate() {
        let rect = tab_rect(o, i);
        let active = *tab == view.tab;
        let hovered = point_in(view.mouse[0], view.mouse[1], rect);
        let tint = match (active, hovered) {
            (true, _) => TAB_TINT_ACTIVE,
            (false, true) => TAB_TINT_HOVER,
            (false, false) => TAB_TINT,
        };
        let chip = [
            rect[0] + TAB_INSET,
            rect[1] + TAB_INSET,
            rect[2] - 2.0 * TAB_INSET,
            rect[3] - 2.0 * TAB_INSET,
        ];
        place_rounded(world, tab_bg(i), chip, tint, theme::CONTROL_RADIUS, true);
        if let Some(l) = widget::label_mut(world, tab_label(i)) {
            l.x = rect[0] + rect[2] * 0.5;
            l.y = rect[1] + TAB_H * 0.5 - theme::TEXT_HALF;
            l.align = TextAlign::Center;
            l.color = if active { LABEL_WHITE } else { LABEL_DIM };
            l.visible = true;
            l.content = tab.label().to_string();
        }
    }
}

// The Expanded body: one row per group header, plus the assets of each open
// group. An asset already in the config is dimmed (its own line wins, so there
// is nothing to add); an addable one carries a "+".
fn layout_expanded(world: &mut World, view: &PanelView, o: [f32; 2]) {
    let total = view.expanded_rows.len();
    if total == 0 {
        // A world that does not cook has no expansion to show; say which it is.
        let msg = view
            .expanded_status
            .unwrap_or("The build adds nothing to this world");
        place_left_label(
            world,
            EMPTY_LABEL,
            [o[0] + PAD, body_y(o, Tab::Expanded) + PAD],
            msg,
            LABEL_DIM,
            true,
        );
        return;
    }
    let scroll = view.expanded_scroll.min(total.saturating_sub(1));
    for r in 0..MAX_ROWS {
        let idx = scroll + r;
        if idx >= total {
            break;
        }
        let rect = expanded_row_rect(o, r);
        let hovered = point_in(view.mouse[0], view.mouse[1], rect);
        match &view.expanded_rows[idx] {
            ExpandedRow::Header {
                origin,
                count,
                in_config,
                open,
                ..
            } => {
                let tint = if hovered {
                    ROW_TINT_HOVER
                } else {
                    GROUP_ROW_TINT
                };
                place_rounded(
                    world,
                    list_row_bg(r),
                    theme::highlight_rect(rect),
                    tint,
                    theme::CONTROL_RADIUS,
                    true,
                );
                let caption = group_caption(origin, *count, *in_config, *open);
                place_left_label(
                    world,
                    list_row_label(r),
                    [rect[0] + PAD, rect[1] + ROW_LABEL_TOP],
                    &caption,
                    asset_list::HEADER_LABEL,
                    true,
                );
            }
            ExpandedRow::Asset {
                name,
                asset_type,
                in_config,
                addable,
                ..
            } => {
                let tint = if hovered { ROW_TINT_HOVER } else { ROW_TINT };
                place_rounded(
                    world,
                    list_row_bg(r),
                    theme::highlight_rect(rect),
                    tint,
                    theme::CONTROL_RADIUS,
                    true,
                );
                let color = if *in_config { LABEL_DIM } else { LABEL };
                place_left_label(
                    world,
                    list_row_label(r),
                    [rect[0] + PAD + asset_list::INDENT, rect[1] + ROW_LABEL_TOP],
                    &asset_caption(name, asset_type),
                    color,
                    true,
                );
                if *addable {
                    place_add_button(world, r, rect, view.mouse);
                }
            }
        }
    }
    asset_list::layout_scrollbar(
        world,
        LIST_TRACK,
        LIST_THUMB,
        total,
        scroll,
        o[0] + PANEL_W,
        body_y(o, Tab::Expanded),
    );
}

// A group header's caption: the collapse marker, what produced the group, and
// how many assets it holds (with the count already in the config, when any).
fn group_caption(origin: &str, count: usize, in_config: usize, open: bool) -> String {
    let marker = if open { "-" } else { "+" };
    let tail = if in_config > 0 {
        format!("({}, {} in config)", count, in_config)
    } else {
        format!("({})", count)
    };
    // Clip the origin rather than the counts, which are the part worth keeping.
    let budget = HEADER_ROW_CHARS.saturating_sub(tail.chars().count() + 4);
    format!(
        "{} {}  {}",
        marker,
        widget::clip_text(origin, budget.max(MIN_NAME_CHARS)),
        tail
    )
}

// An asset row's caption: its name and type. Generated names are long (an import
// prefixes every one with its own name), so the name is clipped to what the type
// leaves; unclipped it would run under the "+" and off the panel.
fn asset_caption(name: &str, asset_type: &str) -> String {
    let budget = ASSET_ROW_CHARS.saturating_sub(asset_type.chars().count() + 3);
    format!(
        "{} ({})",
        widget::clip_text(name, budget.max(MIN_NAME_CHARS)),
        asset_type
    )
}

// The "+" that copies one Expanded row's asset into the config.
fn place_add_button(world: &mut World, r: usize, row: [f32; 4], mouse: [f32; 2]) {
    let rect = add_rect(row);
    let hovered = point_in(mouse[0], mouse[1], rect);
    let tint = if hovered { ADD_TINT_HOVER } else { ADD_TINT };
    place_rounded(
        world,
        add_row_bg(r),
        rect,
        tint,
        theme::CONTROL_RADIUS,
        true,
    );
    if let Some(l) = widget::label_mut(world, add_row_label(r)) {
        l.x = rect[0] + rect[2] * 0.5;
        l.y = rect[1] + rect[3] * 0.5 - theme::TEXT_HALF;
        l.align = TextAlign::Center;
        l.color = LABEL_WHITE;
        l.visible = true;
        l.content = "+".to_string();
    }
}

// The "+" / "X" glyph, centered in the add button and drawn a step larger than
// the body text (the box itself stays a HEADER_H square).
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

fn layout_list(world: &mut World, view: &PanelView, o: [f32; 2]) {
    if view.list_rows.is_empty() {
        place_left_label(
            world,
            EMPTY_LABEL,
            [o[0] + PAD, body_y(o, Tab::Config) + PAD],
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
        body_y(o, Tab::Config),
    );
}

fn layout_combo(world: &mut World, view: &PanelView, o: [f32; 2]) {
    let total = view.combo_options.len();
    let scroll = view.combo_scroll.min(total.saturating_sub(1));
    let shown = total.saturating_sub(scroll).clamp(1, MAX_ROWS);
    let backing = [
        o[0],
        body_y(o, Tab::Config),
        PANEL_W,
        shown as f32 * ROW_H + PAD,
    ];
    place_sprite(world, COMBO_BG, backing, COMBO_BG_TINT, true);
    if total == 0 {
        place_left_label(
            world,
            EMPTY_LABEL,
            [o[0] + PAD, body_y(o, Tab::Config) + PAD],
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
        place_rounded(
            world,
            combo_row_bg(r),
            theme::highlight_rect(rect),
            tint,
            theme::CONTROL_RADIUS,
            true,
        );
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
        body_y(o, Tab::Config),
    );
}

// The three stacked white dots of the triple-dot button. The background box is
// shown only when `show_box` (the dots are hovered, or the menu is open); a plain
// row-hover shows the bare dots. `DOT_BG` is left hidden (blanked by `hide_all`)
// when `show_box` is false.
fn place_dot(world: &mut World, row: [f32; 4], show_box: bool) {
    let d = dot_rect(row);
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

fn layout_row_menu(world: &mut World, view: &PanelView, o: [f32; 2], vr: usize) {
    let (bg, delete) = menu_rects(o, vr);
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

// Every panel sprite id, so the closed / hidden pass can blank the whole panel
// (and `inject.rs` can create exactly this set). THE ORDER OF THIS VEC IS THE DRAW
// ORDER: `inject.rs` adds the panel's Sprites in this sequence, and the overlay
// draws components in insertion (component-column) order -- NOT by asset id -- so
// later entries paint on top. Bottom-to-top: panel background, header chrome, the
// combo backing (under its option rows), the row families, then the floating
// overlays (scrollbar, triple-dot, row menu), which must sit ABOVE the row
// backgrounds so a hovered row's fill cannot cover them.
pub(crate) fn all_sprite_ids() -> Vec<AssetId> {
    let mut ids = vec![PANEL_BG, CLOSE_BG, PLUS_BG, TYPEDROP_BG];
    ids.extend((0..Tab::ALL.len()).map(tab_bg));
    ids.push(COMBO_BG);
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
    // The Expanded rows' "+" buttons sit above their row backgrounds.
    ids.extend((0..MAX_ROWS).map(add_row_bg));
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
    ids.extend((0..Tab::ALL.len()).map(tab_label));
    ids.extend((0..MAX_ROWS).map(list_row_label));
    ids.extend((0..MAX_ROWS).map(combo_row_label));
    ids.extend((0..MAX_ROWS).map(add_row_label));
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

// Element mutation helpers
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
pub(crate) fn cursor_over_body(mx: f32, my: f32, o: [f32; 2], tab: Tab) -> bool {
    let p = panel_rect(o, tab);
    mx >= p[0] && mx < p[0] + p[2] && my >= body_y(o, tab) && my < p[1] + p[3]
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

    #[derive(Default)]
    struct Fixture {
        combo_options: Vec<String>,
        list_rows: Vec<ListRow>,
        expanded_rows: Vec<ExpandedRow>,
    }

    fn view<'a>(
        fx: &'a Fixture,
        combo: Combo,
        row_menu: Option<usize>,
        mouse: [f32; 2],
    ) -> PanelView<'a> {
        PanelView {
            tab: Tab::Config,
            expanded_rows: &fx.expanded_rows,
            expanded_scroll: 0,
            expanded_status: None,
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

    // A view showing the Expanded tab's rows.
    fn expanded_view<'a>(fx: &'a Fixture, mouse: [f32; 2]) -> PanelView<'a> {
        PanelView {
            tab: Tab::Expanded,
            ..view(fx, Combo::Closed, None, mouse)
        }
    }

    fn header(
        group: usize,
        origin: &str,
        count: usize,
        in_config: usize,
        open: bool,
    ) -> ExpandedRow {
        ExpandedRow::Header {
            group,
            origin: origin.to_string(),
            count,
            in_config,
            open,
        }
    }

    fn expanded_asset(
        group: usize,
        index: usize,
        name: &str,
        in_config: bool,
        addable: bool,
    ) -> ExpandedRow {
        ExpandedRow::Asset {
            group,
            index,
            name: name.to_string(),
            asset_type: "Material".to_string(),
            in_config,
            addable,
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
        let p = panel_rect(o, Tab::Config);
        assert_eq!(p[0] + p[2], 1280.0, "flush to the window right");
        assert_eq!(p[1], hud::body_top(), "starts below the top-bar buttons");
        // The whole panel follows its origin: dragged elsewhere, every rect moves.
        let moved = panel_rect([40.0, 60.0], Tab::Config);
        assert_eq!((moved[0], moved[1]), (40.0, 60.0));
        assert_eq!((moved[2], moved[3]), (p[2], p[3]));
        assert_eq!(
            size(Tab::Config),
            [p[2], p[3]],
            "the drag-clamp footprint matches"
        );
    }

    #[test]
    fn title_bar_spans_the_panel_top_and_header_sits_below_the_tabs() {
        let o = test_origin();
        let title = title_rect(o);
        assert_eq!(title, [o[0], o[1], PANEL_W, widget::TITLE_H]);
        // The tab strip spans the width directly under the title bar, split
        // evenly between the tabs.
        let strip = tab_strip_rect(o);
        assert_eq!(strip[1], o[1] + widget::TITLE_H);
        assert_eq!(strip[2], PANEL_W);
        assert_eq!(tab_rect(o, 0)[0], o[0]);
        assert_eq!(tab_rect(o, 1)[0], o[0] + PANEL_W * 0.5);
        assert_eq!(
            tab_rect(o, 1)[0] + tab_rect(o, 1)[2],
            o[0] + PANEL_W,
            "the tabs fill the strip"
        );

        let plus = plus_rect(o);
        assert_eq!(
            plus[1],
            strip[1] + strip[3],
            "the header row starts below the tab strip"
        );
        let combo = combo_rect(o);
        assert!(plus[0] + plus[2] <= combo[0], "+ is left of the combo");
        assert_eq!(
            combo[0] + combo[2],
            o[0] + PANEL_W,
            "combo reaches the panel right"
        );
        // And the Config body still sits below that header.
        assert_eq!(list_row_rect(o, 0)[1], plus[1] + plus[3]);
    }

    #[test]
    fn plus_toggles_the_picker() {
        let fx = Fixture {
            combo_options: vec![],
            list_rows: vec![],
            ..Default::default()
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
            ..Default::default()
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
            ..Default::default()
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
        assert!(sprite_visible(&world, PANEL_BG));
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
            ..Default::default()
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
            ..Default::default()
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
            ..Default::default()
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
            ..Default::default()
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
            ..Default::default()
        };
        let mut world = injected_world();
        let o = test_origin();
        // The active filter is option 1; the mouse rests far from the options so
        // the selected (not hovered) branch renders it.
        let v = PanelView {
            combo_selected: Some(1),
            ..view(&fx, Combo::Filter, None, [0.0, 0.0])
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
            ..Default::default()
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
            ..Default::default()
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
            ..Default::default()
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
            ..Default::default()
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
            ..Default::default()
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
            ..Default::default()
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

    // -- Tabs + the Expanded body ------------------------------------------------

    // Each tab in the strip resolves to its own switch, whichever body is up.
    #[test]
    fn the_tab_strip_switches_tabs_from_either_body() {
        let fx = Fixture::default();
        let o = test_origin();
        for (i, tab) in Tab::ALL.iter().enumerate() {
            let r = tab_rect(o, i);
            let point = (r[0] + 5.0, r[1] + 5.0);
            assert_eq!(
                hit_test(
                    &view(&fx, Combo::Closed, None, [0.0, 0.0]),
                    point.0,
                    point.1,
                    o
                ),
                Some(PanelAction::SwitchTab(*tab))
            );
            assert_eq!(
                hit_test(&expanded_view(&fx, [0.0, 0.0]), point.0, point.1, o),
                Some(PanelAction::SwitchTab(*tab)),
                "the strip stays live on the Expanded tab too"
            );
        }
    }

    // The strip is checked before the Config tab's modal overlays, so an open
    // combo cannot trap the user on one tab.
    #[test]
    fn an_open_combo_does_not_swallow_the_tab_strip() {
        let fx = Fixture {
            combo_options: vec!["PointLight".to_string()],
            ..Default::default()
        };
        let o = test_origin();
        let r = tab_rect(o, 1);
        assert_eq!(
            hit_test(
                &view(&fx, Combo::Picker, None, [0.0, 0.0]),
                r[0] + 5.0,
                r[1] + 5.0,
                o
            ),
            Some(PanelAction::SwitchTab(Tab::Expanded))
        );
    }

    // The Expanded tab has no add / filter header, so its body starts higher and
    // the panel is correspondingly shorter.
    #[test]
    fn the_expanded_tab_drops_the_config_header() {
        let o = test_origin();
        let config = panel_rect(o, Tab::Config);
        let expanded = panel_rect(o, Tab::Expanded);
        assert!(
            expanded[3] < config[3],
            "no header row: {expanded:?} vs {config:?}"
        );
        assert_eq!(expanded[2], config[2], "same width");
        // Both bodies still show MAX_ROWS rows.
        assert_eq!(
            expanded_row_rect(o, 0)[1] + MAX_ROWS as f32 * ROW_H,
            expanded[1] + expanded[3]
        );
        // The first Expanded row sits directly below the tab strip.
        let strip = tab_strip_rect(o);
        assert_eq!(expanded_row_rect(o, 0)[1], strip[1] + strip[3]);
        assert_eq!(size(Tab::Expanded), [expanded[2], expanded[3]]);
    }

    // A group header toggles its group; an addable row's "+" copies it; the row
    // body (and a non-addable row's "+" area) is inert.
    #[test]
    fn expanded_rows_toggle_groups_and_add_assets() {
        let fx = Fixture {
            expanded_rows: vec![
                header(0, "fox", 2, 1, true),
                expanded_asset(0, 0, "fox_mat_0", false, true),
                expanded_asset(0, 1, "fox_mat_wood", true, false),
            ],
            ..Default::default()
        };
        let o = test_origin();
        let v = expanded_view(&fx, [0.0, 0.0]);

        let hdr = expanded_row_rect(o, 0);
        assert_eq!(
            hit_test(&v, hdr[0] + 5.0, hdr[1] + 5.0, o),
            Some(PanelAction::ToggleGroup(0))
        );

        let addable = expanded_row_rect(o, 1);
        let plus = add_rect(addable);
        assert_eq!(
            hit_test(&v, plus[0] + 5.0, plus[1] + 5.0, o),
            Some(PanelAction::AddExpanded(0, 0))
        );
        // The rest of that row does nothing: the asset is edited from the Config
        // tab once copied, not from here.
        assert_eq!(
            hit_test(&v, addable[0] + 5.0, addable[1] + 5.0, o),
            Some(PanelAction::Consume)
        );

        // An asset already in the config offers no "+" at all.
        let in_config = expanded_row_rect(o, 2);
        let where_plus_would_be = add_rect(in_config);
        assert_eq!(
            hit_test(
                &v,
                where_plus_would_be[0] + 5.0,
                where_plus_would_be[1] + 5.0,
                o
            ),
            Some(PanelAction::Consume)
        );
    }

    #[test]
    fn expanded_clicks_outside_the_panel_fall_through() {
        let fx = Fixture::default();
        assert_eq!(
            hit_test(&expanded_view(&fx, [0.0, 0.0]), 10.0, 400.0, test_origin()),
            None
        );
    }

    // A header states what produced the group and how many assets it holds; an
    // asset row names its type, dims when already in the config, and shows its
    // "+" only when addable.
    #[test]
    fn expanded_body_draws_headers_dimming_and_add_buttons() {
        let fx = Fixture {
            expanded_rows: vec![
                header(0, "fox", 2, 1, true),
                expanded_asset(0, 0, "fox_mat_0", false, true),
                expanded_asset(0, 1, "fox_mat_wood", true, false),
            ],
            ..Default::default()
        };
        let mut world = injected_world();
        let o = test_origin();
        apply(&mut world, Some(&expanded_view(&fx, [0.0, 0.0])), o);

        let label = |i: usize| {
            world
                .query::<TextLabel>()
                .find(|l| l.asset_id == list_row_label(i))
                .cloned()
                .unwrap()
        };
        // The header names the origin, the count, and what is already copied.
        let h = label(0);
        assert!(h.content.contains("fox"), "{}", h.content);
        assert!(h.content.contains('2'), "{}", h.content);
        assert!(h.content.contains("1 in config"), "{}", h.content);
        assert!(h.content.starts_with('-'), "open marker: {}", h.content);

        // The asset rows carry their type, and the copied one is dimmed.
        assert!(label(1).content.contains("Material"));
        assert_eq!(label(1).color, LABEL);
        assert_eq!(label(2).color, LABEL_DIM, "an already-copied asset dims");

        // Only the addable row draws a "+".
        assert!(sprite_visible(&world, add_row_bg(1)));
        assert!(!sprite_visible(&world, add_row_bg(2)));

        // The Config tab's header chrome is gone.
        assert!(!sprite_visible(&world, PLUS_BG));
        assert!(!sprite_visible(&world, TYPEDROP_BG));
    }

    // A collapsed group draws its marker the other way, so the row reads as
    // openable.
    #[test]
    fn a_collapsed_group_marks_itself_closed() {
        let fx = Fixture {
            expanded_rows: vec![header(0, "fox", 4712, 0, false)],
            ..Default::default()
        };
        let mut world = injected_world();
        apply(
            &mut world,
            Some(&expanded_view(&fx, [0.0, 0.0])),
            test_origin(),
        );
        let h = world
            .query::<TextLabel>()
            .find(|l| l.asset_id == list_row_label(0))
            .unwrap();
        assert!(h.content.starts_with('+'), "closed marker: {}", h.content);
        assert!(h.content.contains("4712"));
        assert!(
            !h.content.contains("in config"),
            "nothing copied yet: {}",
            h.content
        );
    }

    // The active tab is lit and the other recedes.
    #[test]
    fn the_active_tab_is_highlighted() {
        let fx = Fixture::default();
        let mut world = injected_world();
        let o = test_origin();
        apply(
            &mut world,
            Some(&view(&fx, Combo::Closed, None, [0.0, 0.0])),
            o,
        );
        assert_eq!(sprite(&world, tab_bg(0)).tint, TAB_TINT_ACTIVE);
        assert_eq!(sprite(&world, tab_bg(1)).tint, TAB_TINT);
        apply(&mut world, Some(&expanded_view(&fx, [0.0, 0.0])), o);
        assert_eq!(sprite(&world, tab_bg(0)).tint, TAB_TINT);
        assert_eq!(sprite(&world, tab_bg(1)).tint, TAB_TINT_ACTIVE);
    }

    // An empty Expanded tab explains itself; a world that would not cook says so
    // instead of showing an empty list as though there were nothing to add.
    #[test]
    fn the_expanded_empty_state_reports_a_cook_failure() {
        let fx = Fixture::default();
        let mut world = injected_world();
        let o = test_origin();
        apply(&mut world, Some(&expanded_view(&fx, [0.0, 0.0])), o);
        let empty = |world: &World| {
            world
                .query::<TextLabel>()
                .find(|l| l.asset_id == EMPTY_LABEL)
                .cloned()
                .unwrap()
        };
        assert!(empty(&world).content.contains("adds nothing"));

        let v = PanelView {
            expanded_status: Some("boom: bad args"),
            ..expanded_view(&fx, [0.0, 0.0])
        };
        apply(&mut world, Some(&v), o);
        assert_eq!(empty(&world).content, "boom: bad args");
    }

    // Generated names are long (an import prefixes each with its own name), and
    // the panel is narrow: an unclipped caption runs under the "+" and off the
    // panel. The name clips; the type stays, since it is what the row is for.
    #[test]
    fn a_long_asset_caption_clips_and_keeps_its_type() {
        let caption = asset_caption("default_fragment_shader", "ShaderStage");
        assert!(
            caption.chars().count() <= ASSET_ROW_CHARS,
            "{caption} ({} chars)",
            caption.chars().count()
        );
        assert!(caption.ends_with("(ShaderStage)"), "{caption}");
        assert!(caption.starts_with("default_f"), "{caption}");
        assert!(caption.contains("..."), "{caption}");
        // A short name is left alone.
        assert_eq!(asset_caption("hud_font", "Font"), "hud_font (Font)");
    }

    // A type long enough to eat the whole budget still leaves the name legible
    // rather than clipping it away to nothing.
    #[test]
    fn a_very_long_type_still_leaves_a_readable_name() {
        let caption = asset_caption("some_generated_asset", &"T".repeat(40));
        assert!(caption.starts_with("some_..."), "{caption}");
    }

    // The header clips its origin, never the counts.
    #[test]
    fn a_long_group_caption_clips_the_origin_not_the_counts() {
        let caption = group_caption(&"a_very_long_import_name".repeat(3), 4712, 3, false);
        assert!(
            caption.chars().count() <= HEADER_ROW_CHARS,
            "{caption} ({} chars)",
            caption.chars().count()
        );
        assert!(caption.ends_with("(4712, 3 in config)"), "{caption}");
        assert!(caption.starts_with("+ a_very"), "{caption}");
    }

    // Draw order: a row's "+" must sit above the row backgrounds, or a hovered
    // row's fill paints over it.
    #[test]
    fn expanded_add_buttons_draw_above_the_row_backgrounds() {
        let sprites = all_sprite_ids();
        let spos = |id: AssetId| sprites.iter().position(|&x| x == id).unwrap();
        let last_row_bg = (0..MAX_ROWS).map(list_row_bg).map(spos).max().unwrap();
        for i in 0..MAX_ROWS {
            assert!(spos(add_row_bg(i)) > last_row_bg);
        }
    }
}
