// src/editor/hook.rs
//
// The editor's per-frame drive. Implements the run loop's `DebugHook` seam: each
// frame it hit-tests the editor HUD's controls against the live input, mutates
// the working authored entry list, persists on SAVE, drives the world's cursor /
// freeze state, and re-anchors + recolours the HUD. This is the whole editor: it
// lives in the editor crate (never linked by the shipped runtime), so no editor
// code is compiled into a shipped game.
//
// The top bar (`hud.rs`) owns SAVE and the Templates dropdown. The Assets button
// opens the assets panel (`panel.rs`): a search field over every asset of the
// expanded world, grouped by origin into one collapsible tree (`asset_tree.rs`),
// and a "+" that opens a typed autocomplete of the addable types. Clicking a row
// (or picking a type from the "+" picker) opens the add / edit form in its own
// floating panel (`form_panel.rs`). A row the build generates has no world.jsonl
// line of its own: its form is seeded from what the expansion produced, and only
// confirming appends the line -- which then overrides the expansion. The search
// field and the form's name heading are real `TextInput` assets edited by the
// engine's text-input system; the hook reads them back. All three panels
// (Assets, edit form, Preview) are floating: holding their title bars drags them
// (the hook owns each origin, clamped so a panel can never leave the screen).
//
// Cursor control follows the simulation transport (`editor/sim.rs`): while
// Stopped or Paused the editor holds the cursor and the world sits frozen
// (`MenuOverride(Some(true))`); Play hands the cursor to the running world
// (`Some(false)`); Escape pauses and takes it back. Stop restores the authored
// state through the same preview rebuild every committed edit takes. F1 hides /
// shows the whole HUD.
//
// SAVE re-serializes the authored entry list to world.jsonl and recompiles the
// blobs through the validated cook tail (`build_world_to_disk`). A successful
// SAVE then applies the edit to the running world without recreating the window:
// the live render backend is transplanted out of the world and `apply_world_swap`
// rebuilds the recompiled world onto it (carried in as a `PendingBackend`).

use super::asset_tree::{self, TreeGroup, TreeRow};
use super::axes;
use super::behavior_panel::{self, BehaviorAction, BehaviorView, Status, ViewMode};
use super::console::{self, ConsoleSink};
use super::console_panel::{self, ConsoleAction, ConsoleView};
use super::content_panel;
use super::form::{self, FormField};
use super::form_panel::{self, FormAction, FormFocus, FormView};
use super::health::HealthState;
use super::health_panel;
use super::hud::{self, HudAction, HudState};
use super::import_panel::{self, ImportAction, ImportRow, ImportStatus, ImportView};
use super::lighting;
use super::lighting_panel::{self, LightingAction, LightingView};
use super::list_panel::Row;
use super::overrides;
use super::panel::{self, PanelAction, PanelView};
use super::preview::{self, PreviewAction};
use super::registry::{self, PANEL_COUNT, PanelKey};
use super::snap;
use super::story;
use super::story_panel::{self, StoryAction, StoryView};
use super::template_panel::{self, TemplateAction, TemplateView};
use super::templates::{self, TemplatesAction};
use super::variables;
use super::variables_panel::{self, VariablesAction, VariablesView};
use super::view::{self, ViewAction};
use super::view_menu;
use super::widget::{self, point_in};
// Re-exported for the hook's submodules (they reach these editor-level items as
// `super::asset_list` / `super::seeded_content`).
use super::asset_list;
use super::asset_list::ListRow;
use super::billboards;
use super::create_menu;
use super::cursor;
use super::framing;
use super::gizmo;
use super::group_transform;
use super::highlight;
use super::history::History;
use super::marquee;
use super::orbit;
use super::outlines;
use super::resize;
use super::seeded_content;
use super::selection::Selection;
use super::session_store;
use super::sim;
use super::visibility;
use crate::app::state::App;
use crate::assets::FrameInput;
use crate::debug_hook::DebugHook;
use crate::ecs::asset_id::AssetId;
use crate::ecs::{CursorShape, DesiredCursor, HudLayers, MenuOverride, PendingBackend, World};

// Draw layer for the top bar: far above the floating panels' layers (which are a
// small 1..=6 rank), so the bar always sits on top even under a dragged panel.
const TOP_BAR_LAYER: i32 = 1_000;

// An active title-bar drag: the grabbed panel and the cursor's offset from its
// origin at the press, so the panel follows without snapping to the cursor.
#[derive(Debug, Clone, Copy)]
struct Drag {
    key: PanelKey,
    grab: [f32; 2],
}

pub(crate) struct EditorHook {
    // Path to the world.jsonl the edits are written back to.
    world_path: String,
    // The authored entry list (names live here, unlike the compiled blob). Edits
    // mutate this; SAVE serializes it back to `world_path`.
    entries: Vec<serde_json::Value>,
    // Undo/redo stacks over `entries`. `baseline` mirrors `entries` as of the
    // last committed edit (or undo/redo), so when `mark_changed` runs after a
    // mutation it still holds the pre-edit list -- the undo snapshot. `saved`
    // mirrors the on-disk state, so a history jump can recompute `dirty`.
    history: History,
    baseline: Vec<serde_json::Value>,
    saved: Vec<serde_json::Value>,
    // Whether `entries` has changes not yet written to disk.
    dirty: bool,
    // The simulation transport (Play / Pause / Step / Stop). Starts Stopped:
    // the editor owns the cursor at launch so the HUD is immediately usable.
    sim: sim::SimControl,
    // Whether the whole HUD is shown (F1 toggle). Starts shown.
    hud_visible: bool,
    // Whether the world-origin axes draw in the viewport (Preview panel row).
    // Starts on: the axes are the viewport's orientation reference.
    axes_visible: bool,
    // Set whenever the authored entries change (add / edit / delete / template);
    // consumed by `apply_world_swap` to rebuild the live preview world from the
    // in-memory entries. SAVE does not set this -- the preview is already current.
    rebuild_preview: bool,
    // Whether the Templates panel is shown (toggled from the View panel).
    templates_open: bool,
    // The template whose detail panel is open (index into the templates
    // registry); `None` means the detail panel is closed.
    open_template: Option<usize>,
    // First visible row of the Template detail panel's asset list.
    template_list_scroll: usize,
    // Whether the Lighting panel is shown (toggled from the View panel), which
    // text binding holds keyboard focus, and the message from the last rejected
    // Apply.
    lighting_open: bool,
    lighting_focus: Option<usize>,
    lighting_status: Option<String>,
    // The Story panel: shown state, the loaded source's lines / edit line /
    // window scroll, whether the edit line holds keyboard focus, the source
    // path shown in the header, and the last parse / IO error. `story_blur`
    // suppresses the edit line's focus for one frame after a Backspace line
    // join, so the text system does not also apply that Backspace to the
    // freshly joined content.
    story_open: bool,
    story_lines: Vec<String>,
    story_line: usize,
    story_scroll: usize,
    story_focus: bool,
    story_path: String,
    story_status: Option<String>,
    story_blur: bool,
    // The Import panel: shown state, whether the path field holds keyboard
    // focus, the list window scroll, and the last Add's outcome.
    import_open: bool,
    import_focus: bool,
    import_scroll: usize,
    import_status: Option<ImportStatus>,
    // The Console panel: shown state, whether the command line holds keyboard
    // focus (suppressed for one frame after a backtick open so the text
    // system does not type the backtick into it), the log window's scroll
    // position with its pinned-to-bottom flag (pinned auto-scrolls on new
    // lines until the user scrolls up), the shared log sink, and whether a
    // worker is mid-build (the /cook guard).
    console_open: bool,
    console_focus: bool,
    console_blur: bool,
    console_scroll: usize,
    console_pinned: bool,
    console_sink: ConsoleSink,
    console_build_running: std::sync::Arc<std::sync::atomic::AtomicBool>,
    // The Behavior panel: shown state, which of the world's Behavior entries is
    // open (an ordinal into them, so an unrelated add / delete cannot retarget
    // it), the selected outline row, the outline and palette scrolls, whether
    // the palette is up and which of its options the keyboard is on, whether the
    // value field holds keyboard focus, and the world checker's verdict on the
    // open behavior. The name field carries its own focus, and
    // `behavior_remove_armed` is the removal chip waiting on the press that
    // carries it out.
    behavior_open: bool,
    behavior_index: usize,
    behavior_row: Option<usize>,
    behavior_scroll: usize,
    behavior_picking: bool,
    behavior_pick_scroll: usize,
    behavior_pick: usize,
    // The palette's filter text, mirrored off its field once a frame so the
    // presses and draws that follow all narrow by the same query.
    behavior_filter: String,
    behavior_focus: bool,
    behavior_name_focus: bool,
    behavior_remove_armed: bool,
    behavior_status: Option<Status>,
    behavior_mode: ViewMode,
    // The chart's scroll offset, and the anchor an in-flight canvas pan holds.
    behavior_pan: [f32; 2],
    behavior_pan_drag: Option<[f32; 2]>,
    // The Variables panel: shown state, the selected row of the table, the row
    // window's scroll, and which of its two fields holds the keyboard.
    variables_open: bool,
    variables_row: Option<usize>,
    variables_scroll: usize,
    variables_name_focus: bool,
    variables_value_focus: bool,
    // The list member held for a paste, with the kind of list it came out of so
    // it can only land in one of the same kind. Session state, not an edit, and
    // deliberately not cleared by opening another behavior: carrying a node
    // between two of them is most of the point.
    behavior_clip: Option<crate::editor::behavior::clip::Clip>,
    // The overview's selected card. The map's cards stand for whole behaviors
    // and the things they reach rather than for places inside one, so the
    // outline row the other two views share cannot address them.
    behavior_overview_card: Option<usize>,
    // Live-debug state fed by the runtime's execution trace while a play
    // session runs with the Behavior or Variables panel open
    // (`hook/trace_drive.rs`). Pulses cover the OPEN behavior only (paths are
    // per-body); breakpoints are held by behavior NAME + node path so they
    // survive preview rebuilds and body edits shifting node ids.
    behavior_pulses: Vec<crate::editor::behavior::pulse::NodePulse>,
    behavior_breakpoints: Vec<(String, crate::editor::behavior::path::Path)>,
    trace_seen: u64,
    live_vars: Vec<(String, String, String)>,
    live_locals: Vec<(String, String, String)>,
    // Ctrl state sampled from this frame's input, for the panel presses that
    // resolve without direct input access (a Ctrl+click on a chart card
    // toggles its breakpoint).
    ctrl_held: bool,
    // Editor-session hide / lock sets, by NAME (ids drift across preview
    // rebuilds). Hidden assets skip rendering (via the published
    // `EditorHidden` resource); locked ones are skipped by viewport picking.
    // Neither touches the authored entries.
    hidden_assets: std::collections::BTreeSet<String>,
    locked_assets: std::collections::BTreeSet<String>,
    // An active isolate (`hook/hide.rs`): the names kept visible, hiding
    // everything else. Composes with (and never mutates) `hidden_assets`.
    isolate: Option<std::collections::BTreeSet<String>>,
    // Shift state sampled from this frame's input, for the panel presses that
    // resolve without direct input access (the Assets tree's additive select).
    shift_held: bool,
    // Whether the Assets panel is shown (toggled from the View panel).
    panel_open: bool,
    // The Assets panel's body: every asset of the expanded world as one tree
    // grouped by origin. `tree_groups` is the cooked model (it costs a world
    // expansion, so it is recomputed only when `tree_stale` and the panel is
    // up), `tree_unfolded` holds the groups the user unfolded, `row_menu` the
    // name whose Delete menu is open, and `tree_status` carries a cook failure
    // to the status line.
    tree_groups: Vec<TreeGroup>,
    tree_unfolded: Vec<usize>,
    tree_scroll: usize,
    tree_stale: bool,
    tree_status: Option<String>,
    search_focus: bool,
    row_menu: Option<String>,
    // The header "+" type picker: whether its option list is open and how far it
    // is scrolled. While open the search field narrows those options instead of
    // the tree.
    picker_open: bool,
    picker_scroll: usize,
    // The Content panel (the visual-asset thumbnail grid): shown state, the
    // grid's first visible row, the type-chip cycle position (0 = All, i =
    // VISUAL_TYPES[i-1]), whether its search field holds keyboard focus, and
    // an in-flight drag-out placement (`hook/content_drag.rs`), if any.
    content_open: bool,
    content_scroll: usize,
    content_type: usize,
    content_search_focus: bool,
    content_drag: Option<content_drag::ContentDrag>,
    // The right-click "Create here" menu (`hook/create_menu_drive.rs`), if open.
    create_menu: Option<create_menu_drive::CreateMenu>,
    // Whether the Preview panel is shown (starts shown; toggled from the View
    // panel).
    preview_open: bool,
    // Whether the Health panel is shown (starts hidden; toggled from the View
    // panel).
    health_open: bool,
    // The Health panel's sampler. Ticked every frame whether or not the panel is
    // shown, so its rates cover a continuous window.
    health: HealthState,
    // Whether the View panel itself is shown (the top-bar View button toggles it).
    view_open: bool,
    // The Display menu (`hook/view_menu_drive.rs`): open state, the viewport
    // view mode, the show flags, and the editor-side billboard-icons toggle.
    display_menu_open: bool,
    view_mode: view_menu::ViewMode,
    show_flags: view_menu::ShowFlags,
    show_billboards: bool,
    // The Display menu's always-on extent-outline categories (selection
    // outlines regardless; `hook/outline_drive.rs`).
    extent_show: outlines::CategorySet,
    // The type of the open add / edit form; `None` means the form panel is
    // closed.
    selected_type: Option<String>,
    // What confirming the open form commits to: a new asset, an existing line,
    // or the promotion of a generated asset.
    form_target: FormTarget,
    // The editable arg fields of the open form (derived from the type's default
    // args). Empty while the form is closed.
    form_fields: Vec<FormField>,
    // First visible field of the form's scroll window (its physical control pool is
    // fixed size, so a form wider than `form::FIELD_POOL` scrolls). Reset on open /
    // structural change.
    form_scroll: usize,
    // Which form input has keyboard focus.
    form_focus: FormFocus,
    // A validation message from the last rejected Add, shown under the form.
    form_error: Option<String>,
    // The form arg field whose value dropdown is open (a large enum / ref set),
    // and its scroll offset. `None` outside an open dropdown.
    field_dropdown: Option<usize>,
    field_dropdown_scroll: usize,
    // The open form's working args tree: the fields are derived from it, and it is
    // mutated by add / remove (structure) and, on capture, by the controls. Empty
    // outside AddForm.
    form_args: serde_json::Map<String, serde_json::Value>,
    // The template behind the open form when it edits a template-derived asset:
    // confirming writes the minimal patch against this baseline, and the rows
    // show per-field override state. `None` for plain authored / new assets.
    form_template: Option<FormTemplate>,
    // The field whose override menu (Revert / Apply-to-template) is open, and
    // whether the header's entity-level menu is open.
    override_menu: Option<usize>,
    entity_menu_open: bool,
    // Template baselines for every template-derived asset, derived from the
    // working entries. Invalidated by every edit, rebuilt on demand.
    template_index: Option<overrides::TemplateIndex>,
    // The paths of the form's non-colour vector fields currently disclosed into
    // per-element leaves. Cleared when the form opens / closes.
    vec_expanded: std::collections::HashSet<String>,
    // The viewport selection set (`editor/selection.rs`), held by NAME: every
    // live-preview rebuild resets the interner and re-interns names, so a
    // stored AssetId could silently drift to a different asset. Members are
    // re-resolved each frame (`hook/pick.rs`). `pick_last` is the transient
    // repeat-click cycle; `marquee` an in-flight box select.
    selection: Selection,
    pick_last: Option<pick::PickLast>,
    marquee: Option<marquee_drag::MarqueeDrag>,
    // An in-flight Alt+drag tumble around the selection
    // (`hook/orbit_drive.rs`), if any.
    orbit: Option<orbit_drive::OrbitDrag>,
    // The gizmo's edit mode (T/R/S keys) and an active drag
    // (`hook/gizmo_drag.rs`), if any.
    gizmo_mode: gizmo::GizmoMode,
    gizmo_drag: Option<gizmo_drag::GizmoDrag>,
    // Grid / angle snapping for gizmo drags (Preview panel rows + /snap).
    snap: snap::SnapSettings,
    // Whether a drag-out placement orients the drop to the struck surface's
    // face normal (Preview panel row). Session state, off by default.
    align_to_surface: bool,
    // The edit-mode fly camera (`hook/fly.rs`): on/off and the frame clock
    // its integration steps against.
    fly: bool,
    fly_clock: Option<std::time::Instant>,
    // An in-flight framing / bookmark camera glide (`hook/glide_drive.rs`).
    glide: Option<glide_drive::CameraGlide>,
    // Saved camera poses (`hook/bookmarks.rs`), loaded from the per-project
    // session store and persisted on save.
    bookmarks: [Option<framing::CameraPose>; session_store::BOOKMARK_SLOTS],
    // The floating panels' dragged origins, indexed by `PanelKey`; `None` means
    // the panel still sits at its default anchor. Always clamped fully on screen
    // before use.
    positions: [Option<[f32; 2]>; PANEL_COUNT],
    // The user's per-panel size overrides, indexed by `PanelKey`; `None` means the
    // panel is at its content-derived default. Only ever grows a panel past that
    // default (see `effective_size`), and only the resizable panels are set.
    sizes: [Option<[f32; 2]>; PANEL_COUNT],
    // The title-bar drag in progress, if any.
    drag: Option<Drag>,
    // The edge / corner resize in progress, if any.
    resize: Option<resize::Resize>,
    // The floating panels back-to-front: the last entry is the frontmost (drawn on
    // top + first to receive clicks). Dragging or clicking a panel moves it to the
    // end. Its position drives the per-frame `HudLayers` publish so overlapping
    // panels occlude cleanly instead of merging.
    panel_order: Vec<PanelKey>,
}

// The template a form-edited asset derives from.
#[derive(Debug, Clone)]
pub(crate) struct FormTemplate {
    pub name: String,
    // Effective template args: the type's defaults with the generated args
    // merged over them, the baseline a field is inherited from.
    pub baseline: serde_json::Map<String, serde_json::Value>,
    // The authored asset or injection pass that produced the asset.
    pub generated_by: String,
}

// Owned per-tick data backing a `PanelView` (computed from the cooked tree + the
// live search field, then borrowed for both hit-testing and layout).
struct PanelData {
    rows: Vec<TreeRow>,
    picker_options: Option<Vec<String>>,
    form_title: String,
    form_overrides: Option<FormOverridesData>,
}

// Owned per-tick override state backing a `form_panel::OverridesView`.
struct FormOverridesData {
    marks: Vec<overrides::FieldOrigin>,
    count: usize,
    field_menu: Option<(usize, Vec<String>)>,
    entity_menu: Option<Vec<String>>,
}

// What confirming the open add / edit form commits to.
#[derive(Debug, Clone, PartialEq, Default)]
pub(crate) enum FormTarget {
    // A new asset: confirming appends it under a unique name.
    #[default]
    New,
    // Working-entry `idx`: confirming updates that line in place.
    Entry(usize),
    // An asset the build generates, which has no world.jsonl line of its own.
    // The form is seeded from the entry the expansion produced, and confirming
    // appends that line -- which then overrides the expansion, since the cook
    // drops a generated asset in favour of an authored one of the same name and
    // type. Renaming it in the form instead leaves the generated asset in place
    // and adds a separate one, which is the honest reading of a rename.
    Promote(serde_json::Value),
}

impl FormTarget {
    // The working-entry index the form updates in place, if any.
    fn entry(&self) -> Option<usize> {
        match self {
            FormTarget::Entry(i) => Some(*i),
            _ => None,
        }
    }

    // Whether the form is editing an asset that already exists (in the world or
    // in the build), rather than adding a brand-new one.
    fn is_edit(&self) -> bool {
        !matches!(self, FormTarget::New)
    }
}

// Owned per-tick data backing a `TemplateView` (the open template's title,
// description, and grouped asset rows).
#[derive(Default)]
struct TemplateDetailData {
    title: String,
    description: String,
    rows: Vec<ListRow>,
}

// Owned per-tick data backing a `LightingView` (the row list and the
// per-binding fields derived from the current entries).
struct LightingData {
    rows: Vec<lighting::Row>,
    fields: Vec<Option<FormField>>,
}

// Move a scroll offset one row toward the wheel direction, clamped to `max`.
fn scroll_step(cur: usize, delta: f32, max: usize) -> usize {
    if delta > 0.0 {
        (cur + 1).min(max)
    } else {
        cur.saturating_sub(1)
    }
}

// The physical control slot showing logical form field `j` under scroll offset
// `scroll`, or `None` when the field is outside the visible window of `window`
// rows. The panel's control pool is slot-indexed, so seeding / reading a field
// goes through its slot.
fn visible_slot(j: usize, scroll: usize, window: usize) -> Option<usize> {
    (j >= scroll && j < scroll + window).then(|| j - scroll)
}

// The first line of a validation error, clipped to fit the panel's status line.
fn short_status(e: &str) -> String {
    let line = e.lines().next().unwrap_or(e);
    let clipped: String = line.chars().take(44).collect();
    if clipped.len() < line.len() {
        format!("{clipped}...")
    } else {
        clipped
    }
}

// The `name` string of an entry, if present.
fn entry_name(e: &serde_json::Value) -> Option<&str> {
    e.get("name").and_then(|v| v.as_str())
}
fn entry_type(e: &serde_json::Value) -> Option<&str> {
    e.get("type").and_then(|v| v.as_str())
}

// The names of the working entries whose type is `ty` (the reference options a
// field targeting that type can pick from).
fn names_of_type(entries: &[serde_json::Value], ty: &str) -> Vec<String> {
    entries
        .iter()
        .filter(|e| entry_type(e) == Some(ty))
        .filter_map(|e| entry_name(e).map(String::from))
        .collect()
}

// Named to avoid colliding with the `use super::asset_tree` module import.
mod asset_tree_edit;
mod axes_drive;
mod behavior_asset;
// Named to avoid colliding with the `use super::behavior_panel` import.
mod behavior_edit;
mod behavior_keys;
mod billboard_drive;
mod bookmarks;
mod browse;
// Named to avoid colliding with the `use super::console` module import.
mod console_edit;
mod content_drag;
mod content_edit;
// Named to avoid colliding with the `use super::create_menu` module import.
mod create_menu_drive;
mod drop_floor;
mod duplicate;
mod editing;
mod edits;
mod fly;
mod gizmo_drag;
mod glide_drive;
mod hide;
mod import_edit;
mod layout;
mod marquee_drag;
mod orbit_drive;
mod outline_drive;
// Named to avoid colliding with the `use super::overrides` module import.
mod override_edit;
mod pick;
// Named to avoid colliding with the `use super::lighting` module import.
mod lighting_edit;
// The per-panel `Panel` impls, reachable by the registry (`editor/registry.rs`).
pub(super) mod panels;
mod routing;
mod select_edit;
mod sim_control;
mod trace_drive;
mod view_menu_drive;
// Named to avoid colliding with the `use super::story` module import.
mod story_edit;
// Named to avoid colliding with the `use super::variables_panel` import.
#[cfg(test)]
mod tests;
mod variables_edit;

impl EditorHook {
    pub(crate) fn new(world_path: String, entries: Vec<serde_json::Value>) -> Self {
        let bookmarks = session_store::load(&session_store::default_path())
            .worlds
            .get(&session_store::world_key(&world_path))
            .map(|w| w.bookmarks)
            .unwrap_or_default();
        Self {
            world_path,
            history: History::default(),
            baseline: entries.clone(),
            saved: entries.clone(),
            entries,
            dirty: false,
            sim: sim::SimControl::default(),
            hud_visible: true,
            axes_visible: true,
            rebuild_preview: false,
            templates_open: false,
            open_template: None,
            template_list_scroll: 0,
            lighting_open: false,
            lighting_focus: None,
            lighting_status: None,
            story_open: false,
            story_lines: vec![String::new()],
            story_line: 0,
            story_scroll: 0,
            story_focus: false,
            story_path: String::new(),
            story_status: None,
            story_blur: false,
            import_open: false,
            import_focus: false,
            import_scroll: 0,
            import_status: None,
            console_open: false,
            console_focus: false,
            console_blur: false,
            console_scroll: 0,
            console_pinned: true,
            console_sink: ConsoleSink::default(),
            console_build_running: std::sync::Arc::new(std::sync::atomic::AtomicBool::new(false)),
            behavior_open: false,
            behavior_index: 0,
            behavior_row: None,
            behavior_scroll: 0,
            behavior_picking: false,
            behavior_pick_scroll: 0,
            behavior_pick: 0,
            behavior_filter: String::new(),
            behavior_focus: false,
            behavior_name_focus: false,
            behavior_remove_armed: false,
            behavior_status: None,
            behavior_mode: ViewMode::default(),
            behavior_pan: [0.0, 0.0],
            behavior_pan_drag: None,
            variables_open: false,
            variables_row: None,
            variables_scroll: 0,
            variables_name_focus: false,
            variables_value_focus: false,
            behavior_clip: None,
            behavior_overview_card: None,
            behavior_pulses: Vec::new(),
            behavior_breakpoints: Vec::new(),
            trace_seen: 0,
            live_vars: Vec::new(),
            live_locals: Vec::new(),
            ctrl_held: false,
            hidden_assets: std::collections::BTreeSet::new(),
            locked_assets: std::collections::BTreeSet::new(),
            isolate: None,
            shift_held: false,
            panel_open: false,
            tree_groups: Vec::new(),
            tree_unfolded: Vec::new(),
            tree_scroll: 0,
            tree_stale: true,
            tree_status: None,
            search_focus: false,
            row_menu: None,
            picker_open: false,
            picker_scroll: 0,
            content_open: false,
            content_scroll: 0,
            content_type: 0,
            content_search_focus: false,
            content_drag: None,
            create_menu: None,
            preview_open: true,
            health_open: false,
            health: HealthState::new(),
            view_open: false,
            display_menu_open: false,
            view_mode: view_menu::ViewMode::default(),
            show_flags: view_menu::ShowFlags::default(),
            show_billboards: true,
            extent_show: outlines::CategorySet::default(),
            selected_type: None,
            form_target: FormTarget::New,
            form_fields: Vec::new(),
            form_scroll: 0,
            form_focus: FormFocus::Name,
            form_error: None,
            field_dropdown: None,
            field_dropdown_scroll: 0,
            form_args: serde_json::Map::new(),
            form_template: None,
            override_menu: None,
            entity_menu_open: false,
            template_index: None,
            vec_expanded: std::collections::HashSet::new(),
            selection: Selection::default(),
            pick_last: None,
            marquee: None,
            orbit: None,
            gizmo_mode: gizmo::GizmoMode::default(),
            gizmo_drag: None,
            snap: snap::SnapSettings::default(),
            align_to_surface: false,
            fly: false,
            fly_clock: None,
            glide: None,
            bookmarks,
            positions: [None; PANEL_COUNT],
            sizes: [None; PANEL_COUNT],
            drag: None,
            resize: None,
            // Back-to-front, matching the injected draw order (registry order:
            // the Template detail panel frontmost, over the Templates list it
            // spawns from).
            panel_order: PanelKey::ALL.to_vec(),
        }
    }
}

impl EditorHook {
    // Replace the standalone sink defaulted by `new` with the shared one the
    // tracing mirror was installed on (see `run_editor`).
    pub(crate) fn with_console_sink(mut self, sink: ConsoleSink) -> Self {
        self.console_sink = sink;
        self
    }

    // Whether any editor text control holds keyboard focus this frame: the
    // Assets panel's search field (or the type picker typing into it), an open
    // edit form (its name / arg inputs always own focus while it shows), or a
    // focused Lighting / Story / Import / Console field. Undo/redo shortcuts
    // stand down while typing.
    // Read the palette's filter off its field. Mirrored onto the hook because
    // the data a press and a draw resolve against is built without world access.
    fn sample_behavior_filter(&mut self, world: &World) {
        let typed = match self.behavior_picking {
            true => widget::field_text(world, behavior_panel::FILTER_INPUT),
            false => String::new(),
        };
        if typed != self.behavior_filter {
            // A narrowed palette is a different list, so the highlight starts
            // again at its best answer rather than keeping a place that may no
            // longer be in it.
            self.behavior_pick = 0;
            self.behavior_pick_scroll = 0;
            self.behavior_filter = typed;
        }
    }

    // The frontmost open floating panel, if any: the keyboard target for
    // per-frame editing keys and the arbiter for shortcuts a panel claims.
    fn frontmost_open_panel(&self) -> Option<PanelKey> {
        self.panel_order
            .iter()
            .rev()
            .copied()
            .find(|&k| registry::panel(k).is_open(self))
    }

    fn text_focus_active(&self) -> bool {
        self.non_console_text_focus() || self.console_focus
    }

    // The same, excluding the console's own command line: the backtick toggle
    // must keep working while the console is being typed into, but stand down
    // while any other field is (a backtick there is just a character).
    fn non_console_text_focus(&self) -> bool {
        self.search_focus
            || self.content_search_focus
            || self.picker_open
            || self.selected_type.is_some()
            || self.lighting_focus.is_some()
            || self.story_focus
            || self.import_focus
            || self.behavior_focus
            || self.behavior_name_focus
            || self.behavior_picking
            || self.variables_name_focus
            || self.variables_value_focus
    }
}

impl DebugHook for EditorHook {
    fn tick(&mut self, world: &mut World) {
        // Bring the Assets tree up to date before anything reads it, so this
        // frame's hit test and draw agree on the rows.
        self.refresh_tree_if_needed();
        // Accumulate the Health panel's per-frame counters and, on its throttled
        // boundary, resample. Unconditional: the rates measure a continuous
        // window, so gating this on the panel being open would make the first
        // window after opening report a partial one.
        self.health.sample(world);
        // Seed Transforms onto the billboard-backed entities before any of
        // this frame's picking or gizmo work resolves them.
        self.seed_billboard_transforms(world);
        // Sampled before anything routes, so this frame's hit tests and draw
        // narrow the palette by the same query.
        self.sample_behavior_filter(world);
        let input = world.query::<FrameInput>().last().cloned();
        if let Some(input) = &input {
            // Sampled for the panel presses that resolve without direct input
            // access (the Assets tree's additive select, the chart's
            // Ctrl+click breakpoint toggle).
            self.shift_held = input.shift;
            self.ctrl_held = input.ctrl;
            // Escape hands the cursor back to the editor: a running world
            // pauses mid-state, the fly camera exits, and the create menu
            // dismisses.
            if input.escape {
                self.sim.pause();
                self.fly = false;
                self.fly_clock = None;
                self.glide = None;
                self.orbit = None;
                self.create_menu = None;
                self.display_menu_open = false;
            }
            // The fly camera integrates before any routing: while it is on
            // the cursor is captured, so no click or HUD press can arrive.
            // An in-flight glide owns the camera instead; deliberate fly
            // input cancels it (see `drive_glide`).
            if self.glide.is_some() {
                self.drive_glide(input, world);
            } else {
                self.drive_fly(input, world);
            }
            // F1 (an edge pulse) toggles the whole HUD.
            if input.hud_toggle {
                self.hud_visible = !self.hud_visible;
            }
            let vp = input.viewport;
            if self.hud_visible && vp[0] > 0.0 {
                // An active title-bar drag follows the cursor; a fresh press (only
                // when no drag is running -- the press that starts one must not
                // also resolve to a control) routes to the bar and the panels.
                self.drive_drag(input, vp);
                // An active edge / corner resize follows the cursor the same way.
                self.drive_resize(input, vp);
                // A chart pan does too, so the canvas tracks the cursor.
                self.drive_behavior_pan(input);
                // An in-flight gizmo drag follows the cursor, cancels on
                // Escape, and commits on release, before any new press routes.
                if self.gizmo_drag.is_some() {
                    self.drive_gizmo_drag(input, vp, world);
                }
                // An in-flight marquee likewise: follow, cancel, or select.
                if self.marquee.is_some() {
                    self.drive_marquee(input, vp, world);
                }
                // An in-flight orbit tumble follows the cursor and ends on
                // release.
                if self.orbit.is_some() {
                    self.drive_orbit(input, world);
                }
                // A drag-out placement from the Content panel: ghost follow,
                // cancel, or commit.
                if self.content_drag.is_some() {
                    self.drive_content_drag(input, vp, world);
                }
                if input.left_click
                    && self.drag.is_none()
                    && self.resize.is_none()
                    && self.gizmo_drag.is_none()
                    && self.marquee.is_none()
                    && self.content_drag.is_none()
                    && self.orbit.is_none()
                {
                    // An open Display or create menu is modal: it takes the
                    // press (a row acts, anything else dismisses) before
                    // normal routing. An Alt+press over the viewport starts
                    // an orbit tumble instead of a pick.
                    let claimed = self.route_display_menu_click(input, vp)
                        || self.route_create_menu_click(input, vp)
                        || (input.alt && self.try_begin_orbit(input, vp, world));
                    if !claimed {
                        self.route_click(input, vp, world);
                    }
                }
                // An unclaimed viewport right press opens the create menu at
                // the cursor, under the same not-mid-gesture guards.
                if input.right_click
                    && self.drag.is_none()
                    && self.resize.is_none()
                    && self.gizmo_drag.is_none()
                    && self.marquee.is_none()
                    && self.content_drag.is_none()
                    && self.orbit.is_none()
                {
                    self.open_create_menu(input, vp, world);
                }
                // T / R / S pick the gizmo's mode (translate / rotate /
                // scale), under the same guards as the history shortcuts.
                if !input.ctrl
                    && !self.sim.playing()
                    && !self.text_focus_active()
                    && self.gizmo_drag.is_none()
                {
                    match input.captured_key {
                        Some(crate::assets::Key::T) => {
                            self.gizmo_mode = gizmo::GizmoMode::Translate;
                        }
                        Some(crate::assets::Key::R) => self.gizmo_mode = gizmo::GizmoMode::Rotate,
                        Some(crate::assets::Key::S) => self.gizmo_mode = gizmo::GizmoMode::Scale,
                        // F frames the selection; Shift+F keeps the old fly
                        // toggle one modifier away.
                        Some(crate::assets::Key::F) => {
                            if input.shift {
                                self.toggle_fly();
                            } else {
                                self.frame_selection(input, world);
                            }
                        }
                        // H hides the selection; Shift+H isolates it.
                        Some(crate::assets::Key::H) => {
                            if input.shift {
                                self.toggle_isolate();
                            } else {
                                self.hide_selected();
                            }
                        }
                        // 1..9 glide back to a saved camera bookmark.
                        Some(key) => {
                            if let Some(slot) = bookmarks::slot_for(key) {
                                self.recall_bookmark(slot, world);
                            }
                        }
                        None => {}
                    }
                }
                // Backtick toggles the console. The flag cleared here is the
                // one-frame focus blur a backtick open sets, so the text
                // system never types that backtick into the command line.
                self.console_blur = false;
                self.drive_console_toggle(input, world);
                // Ctrl+Z / Ctrl+Y step the entry list through the history,
                // unless the world owns the keyboard (play mode), a text
                // field does (its own editing keys must win), or a gizmo drag
                // is mid-flight (its commit has not landed yet). Ctrl+D
                // duplicates the selection and Ctrl+Down drops it to the
                // floor, under the same guards; a frontmost Behavior panel
                // keeps its own Ctrl+D (row duplicate, via frame_keys below).
                if input.ctrl
                    && !self.sim.playing()
                    && !self.text_focus_active()
                    && self.gizmo_drag.is_none()
                {
                    match input.captured_key {
                        Some(crate::assets::Key::Z) => self.undo(world),
                        Some(crate::assets::Key::Y) => self.redo(world),
                        Some(crate::assets::Key::D)
                            if self.frontmost_open_panel() != Some(PanelKey::Behavior) =>
                        {
                            self.duplicate_selection();
                        }
                        Some(crate::assets::Key::Down) => {
                            self.drop_selection_to_floor(world);
                        }
                        // Ctrl+H makes everything visible again.
                        Some(crate::assets::Key::H) => self.unhide_all(),
                        // Ctrl+1..9 save the camera pose to a bookmark.
                        Some(key) => {
                            if let Some(slot) = bookmarks::slot_for(key) {
                                self.save_bookmark(slot, world);
                            }
                        }
                        None => {}
                    }
                }
                // The transport shortcuts, live in every state (pausing a
                // running world is their whole point).
                self.sim_keys(input);
                // Per-frame editing keys go to the frontmost open panel.
                if let Some(key) = self.frontmost_open_panel() {
                    registry::panel(key).frame_keys(self, world, input);
                }
                // Wheel routing: the frontmost scrollable panel under the cursor
                // takes the wheel. An open value dropdown is modal and can extend
                // past the form panel, so it scrolls the form from anywhere while
                // open.
                if input.scroll_delta.abs() > 0.5 {
                    let (mx, my) = (input.mouse_x, input.mouse_y);
                    let form_shown = registry::panel(PanelKey::Edit).is_open(self);
                    if form_shown && self.field_dropdown.is_some() {
                        self.scroll_form(input.scroll_delta, world);
                    } else {
                        let front_to_back: Vec<PanelKey> =
                            self.panel_order.iter().rev().copied().collect();
                        for key in front_to_back {
                            let p = registry::panel(key);
                            if !p.is_open(self) {
                                continue;
                            }
                            let o = self.origin(key, vp);
                            if p.wheel_over(self, world, mx, my, o) {
                                p.scroll(self, world, input.scroll_delta);
                                break;
                            }
                        }
                    }
                }
            }
        }

        // Drive the world's cursor / freeze state from the transport: Stopped /
        // Paused (`Some(true)`) free the cursor and freeze the world; Playing
        // (`Some(false)`) runs it -- as does the one frame a queued Step takes.
        // The fly flag layers on top: navigation input + cursor capture stay
        // live while the frozen world is flown through.
        self.drive_sim(world);
        // Exchange execution-trace state with the behavior system: publish the
        // request while a live-debug panel is open, ingest what last frame's
        // simulated tick reported (pulses, live values, breakpoint hits).
        self.drive_trace(world);
        world.insert_resource(crate::ecs::FlyCam(self.fly && !self.sim.playing()));
        // Publish the editor-session hidden set (manual hides composed with an
        // active isolate, resolved to this world's ids) so the renderer
        // collapses those objects this frame.
        world.insert_resource(crate::ecs::EditorHidden(self.effective_hidden_ids()));
        // Publish the viewport view mode + show flags for this frame's draw.
        world.insert_resource(crate::ecs::ViewOverrides {
            mode: self.view_mode,
            show: self.show_flags,
        });

        // Re-anchor + recolour the top bar, then lay out (or hide) the panels.
        hud::apply_layout(world, self.hud_state());
        let vp = input.as_ref().map(|i| i.viewport).unwrap_or([0.0, 0.0]);
        let mouse = input
            .as_ref()
            .map(|i| [i.mouse_x, i.mouse_y])
            .unwrap_or([0.0, 0.0]);
        // The panels lay out only once the HUD is shown and a real viewport exists
        // (frame 0 keeps the injected-hidden placeholders).
        let shown = self.hud_visible && vp[0] > 0.0;
        // Publish the world-space editor lines (origin axes + extent outlines)
        // for the renderer's line pass, refilling last frame's buffer.
        // Republished every frame (the renderer expands whatever it finds), and
        // empty while everything is toggled off, the HUD is hidden, or the
        // Lines show flag is cleared (which would mask the pass anyway; the
        // empty buffer also skips the CPU generation and ribbon expansion).
        let mut lines = world
            .remove_resource::<crate::ecs::WorldLines>()
            .map(|l| l.0)
            .unwrap_or_default();
        lines.clear();
        if self.show_flags.contains(view_menu::ShowFlags::LINES) {
            self.push_axis_lines(world, &mut lines);
            self.push_extent_lines(world, vp, &mut lines);
        }
        world.insert_resource(crate::ecs::WorldLines(lines));
        // Drive the editor's in-engine cursor. It shows while the editor owns the
        // pointer (edit mode); a captured camera (play / fly) owns the pointer
        // instead, so the sprite hides and no stray arrow lingers over the frozen
        // pointer. Its shape becomes a resize cursor over a resizable panel edge.
        let owns_cursor = vp[0] > 0.0 && !self.sim.playing() && !self.fly;
        cursor::set_visible(world, owns_cursor);
        let shape = if !owns_cursor || !shown {
            CursorShape::Default
        } else if let Some(r) = &self.resize {
            // Keep the resize cursor for the whole drag, even if the pointer
            // drifts off the edge as the panel grows.
            resize::cursor_shape(r.edges)
        } else {
            self.hover_cursor(vp, mouse)
        };
        world.insert_resource(DesiredCursor(shape));
        // Publish the panels' draw layers (focus stack) so the renderer occludes
        // overlaps by focus this frame. Empty while the HUD is hidden, so the
        // renderer skips the overlay sort entirely.
        if shown {
            self.publish_layers(world);
        } else {
            world.insert_resource(HudLayers::default());
        }
        // Lay out every open panel (hiding keeps its state, so toggling back
        // restores the same view). Compound gates live on each panel's
        // `is_open` -- e.g. the edit form shows only while the assets UI is on.
        for p in registry::all() {
            if shown && p.is_open(self) {
                let o = self.origin(p.key(), vp);
                p.draw(self, world, o, mouse);
            } else {
                p.hide(world);
            }
        }
        // The billboards sit at the bottom of the editor overlays; the
        // selection rings ride the picked assets' projected bounds above
        // them, under the panels (their ids take the default draw layer);
        // the gizmo and the marquee rect draw over both.
        self.drive_billboards(world, vp, shown && self.show_billboards);
        self.drive_highlight(world, vp, shown);
        self.drive_gizmo_draw(world, vp, shown);
        self.drive_marquee_draw(world, shown);
        self.drive_create_menu_draw(world, shown, mouse);
        self.drive_display_menu_draw(world, vp, shown, mouse);
    }

    // Rebuild the live preview world from the in-memory entries and swap it under
    // the running render backend, so any authored change (add / edit / delete /
    // template apply) shows immediately without recreating the OS window or
    // touching disk (SAVE owns persistence). Run once per frame by the run loop
    // right after `tick`, whenever an edit flagged `rebuild_preview`.
    //
    // The recompiled world is built FIRST, in a throwaway App; only once that
    // succeeds is the backend transplanted out of the live world. So a rebuild
    // failure leaves the live world -- and its window -- fully intact; the next
    // edit retries. The backend is never dropped on an error path.
    fn apply_world_swap(&mut self, app: &mut App) {
        if !self.rebuild_preview {
            return;
        }
        self.rebuild_preview = false;

        let world = match self.build_preview_world() {
            Ok(w) => w,
            Err(e) => {
                tracing::error!("editor: live preview rebuild failed, keeping current world: {e}");
                return;
            }
        };
        let mut staged = App::new();
        staged.load_world(world);
        // Carry the editor's typed text (an open form's name + fields, the combo
        // filter) across the fresh HUD injection so it is not blanked.
        let fields = Self::field_snapshot(app.world());
        super::inject::editor_hud(staged.world_mut());
        Self::restore_fields(staged.world_mut(), &fields);
        staged
            .world_mut()
            .insert_resource(MenuOverride(Some(!self.sim.playing())));

        let Some(backend) = app.world_mut().take_render_backend() else {
            return;
        };
        staged.world_mut().insert_resource(PendingBackend(backend));
        let new_world = std::mem::replace(staged.world_mut(), World::new_empty());
        app.load_world(new_world);
        if let Err(e) = app.start() {
            tracing::error!("editor: live preview start failed: {e:?}");
        }
    }
}
