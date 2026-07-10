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
// opens the browse panel (`panel.rs`): a combo (dropdown) that filters the
// browse list by type, a "+" that opens a typed autocomplete of the addable
// types, and a browse list grouped by type. Clicking a name (or picking a type
// from the "+" picker) opens the add / edit form in its own floating panel
// (`form_panel.rs`); the browse row of the edited entry stays highlighted. The
// combo's filter field and the form's name heading are real `TextInput` assets
// edited by the engine's text-input system; the hook reads them back. All three
// panels (Assets, edit form, Preview) are floating: holding their title bars
// drags them (the hook owns each origin, clamped so a panel can never leave the
// screen).
//
// Cursor control: the editor holds the cursor by default (edit mode -- cursor
// free, world frozen), publishing `MenuOverride(Some(true))`. Ticking the
// Preview panel's capture checkbox hands the cursor to the world (`Some(false)`);
// Escape takes it back. F1 hides / shows the whole HUD.
//
// SAVE re-serializes the authored entry list to world.jsonl and recompiles the
// blobs through the validated cook tail (`build_world_to_disk`). A successful
// SAVE then applies the edit to the running world without recreating the window:
// the live render backend is transplanted out of the world and `apply_world_swap`
// rebuilds the recompiled world onto it (carried in as a `PendingBackend`).

use super::form::{self, FormField};
use super::form_panel::{self, FormAction, FormFocus, FormView};
use super::hud::{self, HudAction, HudState};
use super::panel::{self, Combo, ListRow, PanelAction, PanelView};
use super::preview::{self, PreviewAction};
use super::template_panel::{self, TemplateAction, TemplateView};
use super::templates::{self, TemplatesAction};
use super::view::{self, ViewAction, ViewState};
use super::widget::{self, point_in};
use crate::app::state::App;
use crate::assets::FrameInput;
use crate::debug_hook::DebugHook;
use crate::ecs::asset_id::AssetId;
use crate::ecs::{HudLayers, MenuOverride, PendingBackend, World};

// Draw layer for the top bar: far above the floating panels' layers (which are a
// small 1..=3 rank), so the bar always sits on top even under a dragged panel.
const TOP_BAR_LAYER: i32 = 1_000;

// Which floating panel a title-bar drag is moving.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum DragTarget {
    Assets,
    Edit,
    Preview,
    View,
    Templates,
    // The Template detail panel spawned by picking a Templates-list row.
    TemplateDetail,
}

// Which scroll region a wheel event lands in (decided by the tick from the
// cursor position, since the browse and form panels can be open at once).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum ScrollTarget {
    // The edit form's field window (or its open value dropdown).
    Form,
    // The Assets panel's browse list / combo options.
    List,
    // The Template detail panel's asset list.
    TemplateList,
}

// An active title-bar drag: the grabbed panel and the cursor's offset from its
// origin at the press, so the panel follows without snapping to the cursor.
#[derive(Debug, Clone, Copy)]
struct Drag {
    target: DragTarget,
    grab: [f32; 2],
}

pub(crate) struct EditorHook {
    // Path to the world.jsonl the edits are written back to.
    world_path: String,
    // The authored entry list (names live here, unlike the compiled blob). Edits
    // mutate this; SAVE serializes it back to `world_path`.
    entries: Vec<serde_json::Value>,
    // Whether `entries` has changes not yet written to disk.
    dirty: bool,
    // Whether the world currently holds the cursor (play mode). Starts false:
    // the editor owns the cursor at launch so the HUD is immediately usable.
    world_capture: bool,
    // Whether the whole HUD is shown (F1 toggle). Starts shown.
    hud_visible: bool,
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
    // Whether the Assets panel is shown (toggled from the View panel).
    panel_open: bool,
    // Whether the Preview panel is shown (starts shown; toggled from the View
    // panel).
    preview_open: bool,
    // Whether the View panel itself is shown (the top-bar View button toggles it).
    view_open: bool,
    // The header combo (dropdown) state: closed, filtering, or picking a type.
    combo: Combo,
    // Active type filter for the browse list, or `None` for "all".
    type_filter: Option<String>,
    // First visible row of the (grouped) browse list / the combo option list.
    list_scroll: usize,
    combo_scroll: usize,
    // The type of the open add / edit form; `None` means the form panel is
    // closed.
    selected_type: Option<String>,
    // When the form is editing an existing entry, its `entries` index (else a new
    // asset is being added).
    editing: Option<usize>,
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
    // The `entries` index whose Delete menu is open, if any.
    row_menu: Option<usize>,
    // The form arg field whose value dropdown is open (a large enum / ref set),
    // and its scroll offset. `None` outside an open dropdown.
    field_dropdown: Option<usize>,
    field_dropdown_scroll: usize,
    // The open form's working args tree: the fields are derived from it, and it is
    // mutated by add / remove (structure) and, on capture, by the controls. Empty
    // outside AddForm.
    form_args: serde_json::Map<String, serde_json::Value>,
    // The paths of the form's non-colour vector fields currently disclosed into
    // per-element leaves. Cleared when the form opens / closes.
    vec_expanded: std::collections::HashSet<String>,
    // The floating panels' dragged origins; `None` means the panel still sits at
    // its default anchor. Always clamped fully on screen before use.
    panel_pos: Option<[f32; 2]>,
    edit_pos: Option<[f32; 2]>,
    preview_pos: Option<[f32; 2]>,
    view_pos: Option<[f32; 2]>,
    templates_pos: Option<[f32; 2]>,
    template_detail_pos: Option<[f32; 2]>,
    // The title-bar drag in progress, if any.
    drag: Option<Drag>,
    // The floating panels back-to-front: the last entry is the frontmost (drawn on
    // top + first to receive clicks). Dragging or clicking a panel moves it to the
    // end. Its position drives the per-frame `HudLayers` publish so overlapping
    // panels occlude cleanly instead of merging.
    panel_order: Vec<DragTarget>,
}

// Owned per-tick data backing a `PanelView` (computed from the entries + the live
// filter field, then borrowed for both hit-testing and layout).
struct PanelData {
    filter_label: String,
    combo_options: Vec<String>,
    combo_selected: Option<usize>,
    list_rows: Vec<ListRow>,
    form_title: String,
}

// Owned per-tick data backing a `TemplateView` (the open template's title,
// description, and grouped asset rows).
#[derive(Default)]
struct TemplateDetailData {
    title: String,
    description: String,
    rows: Vec<ListRow>,
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
// `scroll`, or `None` when the field is outside the visible window. The panel's
// control pool is slot-indexed, so seeding / reading a field goes through its slot.
fn visible_slot(j: usize, scroll: usize) -> Option<usize> {
    (j >= scroll && j < scroll + form::FIELD_POOL).then(|| j - scroll)
}

// The first line of a validation error, clipped to fit the panel's status line.
fn short_error(e: &str) -> String {
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

impl EditorHook {
    pub(crate) fn new(world_path: String, entries: Vec<serde_json::Value>) -> Self {
        Self {
            world_path,
            entries,
            dirty: false,
            world_capture: false,
            hud_visible: true,
            rebuild_preview: false,
            templates_open: false,
            open_template: None,
            template_list_scroll: 0,
            panel_open: false,
            preview_open: true,
            view_open: false,
            combo: Combo::Closed,
            type_filter: None,
            list_scroll: 0,
            combo_scroll: 0,
            selected_type: None,
            editing: None,
            form_fields: Vec::new(),
            form_scroll: 0,
            form_focus: FormFocus::Name,
            form_error: None,
            row_menu: None,
            field_dropdown: None,
            field_dropdown_scroll: 0,
            form_args: serde_json::Map::new(),
            vec_expanded: std::collections::HashSet::new(),
            panel_pos: None,
            edit_pos: None,
            preview_pos: None,
            view_pos: None,
            templates_pos: None,
            template_detail_pos: None,
            drag: None,
            // Back-to-front, matching the injected draw order (the Template detail
            // panel frontmost, over the Templates list it spawns from).
            panel_order: vec![
                DragTarget::Assets,
                DragTarget::Edit,
                DragTarget::Preview,
                DragTarget::View,
                DragTarget::Templates,
                DragTarget::TemplateDetail,
            ],
        }
    }

    // Bring a panel to the front of the focus stack (drawn on top, first to be
    // clicked). A no-op if it is already frontmost.
    fn focus_panel(&mut self, target: DragTarget) {
        self.panel_order.retain(|&p| p != target);
        self.panel_order.push(target);
    }

    // Every injected element id of a panel, for the `HudLayers` layer map.
    fn panel_ids(target: DragTarget) -> Vec<AssetId> {
        match target {
            DragTarget::Assets => panel::all_sprite_ids()
                .into_iter()
                .chain(panel::all_label_ids())
                .chain(panel::all_field_ids())
                .collect(),
            DragTarget::Edit => form_panel::all_sprite_ids()
                .into_iter()
                .chain(form_panel::all_label_ids())
                .chain(form_panel::all_field_ids())
                .collect(),
            DragTarget::Preview => preview::all_sprite_ids()
                .into_iter()
                .chain(preview::all_label_ids())
                .collect(),
            DragTarget::View => view::all_sprite_ids()
                .into_iter()
                .chain(view::all_label_ids())
                .collect(),
            DragTarget::Templates => templates::all_sprite_ids()
                .into_iter()
                .chain(templates::all_label_ids())
                .collect(),
            DragTarget::TemplateDetail => template_panel::all_sprite_ids()
                .into_iter()
                .chain(template_panel::all_label_ids())
                .collect(),
        }
    }

    // The per-frame HUD draw layers: each panel at its focus-stack rank (1..=3,
    // higher = more front), the top bar pinned above them all.
    fn compute_layers(&self) -> std::collections::HashMap<AssetId, i32> {
        let mut layers = std::collections::HashMap::new();
        for (rank, &target) in self.panel_order.iter().enumerate() {
            let layer = rank as i32 + 1;
            for id in Self::panel_ids(target) {
                layers.insert(id, layer);
            }
        }
        for id in hud::all_ids() {
            layers.insert(id, TOP_BAR_LAYER);
        }
        layers
    }

    // Publish the draw layers for the renderer's overlay sort, so a dragged /
    // clicked panel occludes the rest instead of its text bleeding through their
    // backgrounds.
    fn publish_layers(&self, world: &mut World) {
        world.insert_resource(HudLayers(self.compute_layers()));
    }

    // Whether the add / edit form panel is open.
    fn form_open(&self) -> bool {
        self.selected_type.is_some()
    }

    // The Assets panel's top-left for this frame: the dragged position (or the
    // default anchor below the top bar), clamped so the whole panel stays on
    // screen even after a window resize.
    fn panel_origin(&self, vp: [f32; 2]) -> [f32; 2] {
        let pos = self.panel_pos.unwrap_or(panel::default_origin(vp[0]));
        widget::clamp_origin(pos, panel::size(), vp)
    }

    // The edit-form panel's top-left for this frame, clamped at its current
    // height (the field area tracks the open type's field count).
    fn edit_origin(&self, vp: [f32; 2]) -> [f32; 2] {
        let pos = self.edit_pos.unwrap_or(form_panel::default_origin(vp[0]));
        widget::clamp_origin(pos, form_panel::size(self.form_fields.len()), vp)
    }

    // The Preview panel's top-left for this frame, clamped like the others.
    fn preview_origin(&self, vp: [f32; 2]) -> [f32; 2] {
        let pos = self.preview_pos.unwrap_or(preview::default_origin());
        widget::clamp_origin(pos, preview::size(), vp)
    }

    // The View panel's top-left for this frame, clamped like the others.
    fn view_origin(&self, vp: [f32; 2]) -> [f32; 2] {
        let pos = self.view_pos.unwrap_or(view::default_origin());
        widget::clamp_origin(pos, view::size(), vp)
    }

    // The Templates panel's top-left for this frame, clamped like the others.
    fn templates_origin(&self, vp: [f32; 2]) -> [f32; 2] {
        let pos = self
            .templates_pos
            .unwrap_or(templates::default_origin(vp[0]));
        widget::clamp_origin(pos, templates::size(), vp)
    }

    // The Template detail panel's top-left for this frame, clamped at its current
    // height (the list area tracks the open template's asset-row count).
    fn template_detail_origin(&self, i: usize, vp: [f32; 2]) -> [f32; 2] {
        let n = self.template_rows(i).len();
        let pos = self
            .template_detail_pos
            .unwrap_or(template_panel::default_origin(vp[0]));
        widget::clamp_origin(pos, template_panel::size(n), vp)
    }

    // The world-line entries of template `i` (its typed specs via the app bridge).
    fn template_entries(&self, i: usize) -> Vec<serde_json::Value> {
        concinnity_templates::TEMPLATES
            .get(i)
            .map(concinnity_app::world_template_entries)
            .unwrap_or_default()
    }

    // Template `i`'s assets as the shared grouped rows (types + names alphabetical,
    // identical to the Assets panel's list).
    fn template_rows(&self, i: usize) -> Vec<ListRow> {
        super::asset_list::grouped_rows(&self.template_entries(i), None)
    }

    // Which panels are currently shown, so the View panel's checkboxes reflect it.
    fn view_state(&self) -> ViewState {
        ViewState {
            assets: self.panel_open,
            preview: self.preview_open,
            templates: self.templates_open,
        }
    }

    // Whether an entry with this name already exists.
    fn name_taken(&self, n: &str) -> bool {
        self.entries.iter().any(|e| entry_name(e) == Some(n))
    }

    // Whether an entry other than `skip` already has this name (for renames).
    fn name_taken_except(&self, n: &str, skip: usize) -> bool {
        self.entries
            .iter()
            .enumerate()
            .any(|(i, e)| i != skip && entry_name(e) == Some(n))
    }

    // A world-unique name derived from the asset type: `editor_<kind>` plus a
    // numeric suffix bumped until it does not collide with an existing entry.
    fn unique_name(&self, kind: &str) -> String {
        let base = format!("editor_{}", kind.to_ascii_lowercase());
        self.unique_from(&base)
    }

    // `base` if free, else `base_1`, `base_2`, ... until unused.
    fn unique_from(&self, base: &str) -> String {
        if !self.name_taken(base) {
            return base.to_string();
        }
        let mut i = 1;
        loop {
            let candidate = format!("{base}_{i}");
            if !self.name_taken(&candidate) {
                return candidate;
            }
            i += 1;
        }
    }

    // The final name for a new-asset submission: the typed name (trimmed) made
    // unique, or a generated unique name when the field was left blank.
    fn finalize_name(&self, typed: &str, kind: &str) -> String {
        let t = typed.trim();
        if t.is_empty() {
            self.unique_name(kind)
        } else {
            self.unique_from(t)
        }
    }

    // The final name for a rename of entry `idx`: the typed name (trimmed), or a
    // generated one when blank, made unique against the *other* entries.
    fn finalize_rename(&self, typed: &str, idx: usize, kind: &str) -> String {
        let t = typed.trim();
        let base = if t.is_empty() {
            format!("editor_{}", kind.to_ascii_lowercase())
        } else {
            t.to_string()
        };
        if !self.name_taken_except(&base, idx) {
            return base;
        }
        let mut i = 1;
        loop {
            let candidate = format!("{base}_{i}");
            if !self.name_taken_except(&candidate, idx) {
                return candidate;
            }
            i += 1;
        }
    }

    // Record an authored-entry change: the live preview needs a rebuild this frame
    // (`apply_world_swap` reloads the running world from the in-memory entries), and
    // the change is not yet on disk (SAVE clears `dirty`).
    fn mark_changed(&mut self) {
        self.dirty = true;
        self.rebuild_preview = true;
    }

    // SAVE: persist the working entries to disk (world.jsonl + recompiled blobs).
    // The live preview is already up to date (every edit swaps it in), so SAVE is
    // purely persistence -- it does not rebuild or swap the running world. On
    // success the world is clean again; on failure it stays dirty and the next
    // SAVE retries.
    fn save(&mut self) {
        match self.persist() {
            Ok(()) => {
                self.dirty = false;
                tracing::info!("editor: saved {}", self.world_path);
            }
            Err(e) => tracing::error!("editor: save failed: {e}"),
        }
    }

    fn persist(&self) -> std::io::Result<()> {
        self.write_jsonl()?;
        concinnity_app::build_world_to_disk(&self.world_path)?;
        Ok(())
    }

    // Build a ready-to-run world from the in-memory entries, without touching disk
    // (SAVE owns persistence). A GraphicsConfig is seeded when the authored entries
    // alone would not render, so the preview window never goes blank.
    fn build_preview_world(&self) -> std::io::Result<World> {
        let jsonl = concinnity_core::world::write_world_jsonl(&self.entries)
            .map_err(|e| std::io::Error::other(e.to_string()))?;
        match concinnity_app::build_world_from_str(&jsonl) {
            Ok(world) if world.renders() => Ok(world),
            _ => concinnity_app::build_world_from_str(&super::seeded_content(&jsonl)),
        }
    }

    // Snapshot the editor's text-field contents (the combo filter + the form's name
    // heading and arg inputs) by reserved id, so a live rebuild's fresh HUD
    // injection does not blank an open form.
    fn field_snapshot(world: &World) -> Vec<(AssetId, String)> {
        panel::all_field_ids()
            .into_iter()
            .chain(form_panel::all_field_ids())
            .map(|id| (id, widget::field_text(world, id)))
            .collect()
    }

    // Restore a `field_snapshot` into a freshly injected HUD.
    fn restore_fields(world: &mut World, snapshot: &[(AssetId, String)]) {
        for (id, content) in snapshot {
            widget::seed_field(world, *id, content);
        }
    }

    // Write the working entries to world.jsonl atomically (temp file + rename),
    // so a crash mid-write cannot truncate the user's world. Split from `persist`
    // so the serialization is unit-testable without the compile step.
    fn write_jsonl(&self) -> std::io::Result<()> {
        let out = concinnity_core::world::write_world_jsonl(&self.entries)
            .map_err(|e| std::io::Error::other(e.to_string()))?;
        let tmp = format!("{}.tmp", self.world_path);
        std::fs::write(&tmp, out)?;
        std::fs::rename(&tmp, &self.world_path)
    }

    // Apply every entry of engine-owned template `i`, skipping any whose name
    // already exists (so re-applying is idempotent). Marks dirty if anything was
    // added.
    fn apply_template(&mut self, i: usize) {
        let Some(t) = concinnity_templates::TEMPLATES.get(i) else {
            return;
        };
        // The template's typed specs become world-line entries via the app bridge;
        // no JSON string is parsed here.
        let entries = concinnity_app::world_template_entries(t);
        let mut added = 0;
        for entry in entries {
            if entry_name(&entry).is_some_and(|n| self.name_taken(n)) {
                continue;
            }
            self.entries.push(entry);
            added += 1;
        }
        if added > 0 {
            self.mark_changed();
            tracing::info!("editor: applied template '{}' ({added} asset(s))", t.name);
        }
    }

    // -- Option lists (derived from the entries + the live filter field) --------

    // The distinct asset types present in the world, sorted.
    fn distinct_types(&self) -> Vec<String> {
        let mut out: Vec<String> = Vec::new();
        for e in &self.entries {
            if let Some(ty) = entry_type(e)
                && !out.iter().any(|s| s == ty)
            {
                out.push(ty.to_string());
            }
        }
        out.sort();
        out
    }

    // The browse list, grouped by type (the shared model): a sub-header row per
    // type matching the active filter, then an indented row per asset name
    // carrying its entry index (for the Delete / edit menu). Types + names sorted
    // alphabetically, identical to the Template panel's list.
    fn list_rows(&self) -> Vec<ListRow> {
        super::asset_list::grouped_rows(&self.entries, self.type_filter.as_deref())
    }

    // The floating combo option list for the open flavour, narrowed by the typed
    // filter field. Empty when the combo is closed.
    fn combo_options(&self, world: &World) -> Vec<String> {
        let filter = widget::field_text(world, panel::FILTER_INPUT).to_lowercase();
        let matches = |s: &str| filter.is_empty() || s.to_lowercase().contains(&filter);
        match self.combo {
            Combo::Filter => {
                let mut all = vec![panel::ALL_LABEL.to_string()];
                all.extend(self.distinct_types());
                all.into_iter().filter(|o| matches(o)).collect()
            }
            Combo::Picker => {
                let mut opts: Vec<String> = panel::picker_types()
                    .filter(|t| matches(t))
                    .map(|t| t.to_string())
                    .collect();
                // The picker lists the offered types alphabetically (ascending).
                opts.sort();
                opts
            }
            Combo::Closed => Vec::new(),
        }
    }

    fn panel_data(&self, world: &World) -> PanelData {
        let combo_options = self.combo_options(world);
        let combo_selected = match self.combo {
            Combo::Filter => {
                let target = self
                    .type_filter
                    .clone()
                    .unwrap_or_else(|| panel::ALL_LABEL.to_string());
                combo_options.iter().position(|o| o == &target)
            }
            _ => None,
        };
        let form_title = match (&self.editing, &self.selected_type) {
            (Some(_), Some(t)) => format!("Edit {t}"),
            (None, Some(t)) => format!("New {t}"),
            _ => "New asset".to_string(),
        };
        PanelData {
            filter_label: self
                .type_filter
                .clone()
                .unwrap_or_else(|| panel::ALL_LABEL.to_string()),
            combo_options,
            combo_selected,
            list_rows: self.list_rows(),
            form_title,
        }
    }

    fn make_view<'a>(&'a self, d: &'a PanelData, mouse: [f32; 2]) -> PanelView<'a> {
        PanelView {
            combo: self.combo,
            filter_label: &d.filter_label,
            combo_options: &d.combo_options,
            combo_selected: d.combo_selected,
            combo_scroll: self.combo_scroll,
            list_rows: &d.list_rows,
            list_scroll: self.list_scroll,
            row_menu: self.row_menu,
            // The entry whose form is open keeps its browse row highlighted.
            selected: self.editing,
            mouse,
        }
    }

    fn make_form_view<'a>(&'a self, d: &'a PanelData, mouse: [f32; 2]) -> FormView<'a> {
        FormView {
            title: &d.form_title,
            editing: self.editing.is_some(),
            form_fields: &self.form_fields,
            form_scroll: self.form_scroll,
            form_focus: self.form_focus,
            field_dropdown: self.field_dropdown,
            field_dropdown_scroll: self.field_dropdown_scroll,
            form_error: self.form_error.as_deref(),
            mouse,
        }
    }

    // -- Action handling --------------------------------------------------------

    // Route a resolved top-bar click: SAVE persists to disk (the live preview is
    // already current), the View button toggles the View panel.
    fn apply_top(&mut self, action: HudAction) {
        match action {
            HudAction::Save => self.save(),
            HudAction::ToggleView => self.view_open = !self.view_open,
        }
    }

    // Toggle the whole assets UI (browse panel + any open edit form + the browse
    // highlight). Hiding it KEEPS all that state -- panel positions, the open
    // form, the scroll offset, the selection -- so toggling back restores the same
    // view. Only the transient combo / row-menu overlays are dropped.
    fn toggle_assets(&mut self) {
        self.panel_open = !self.panel_open;
        self.combo = Combo::Closed;
        self.row_menu = None;
    }

    // Route a resolved View-panel click: each checkbox toggles one panel's shown
    // state (rows are Assets / Preview / Templates, in order).
    fn apply_view(&mut self, action: ViewAction) {
        match action {
            ViewAction::Toggle(0) => self.toggle_assets(),
            ViewAction::Toggle(1) => self.preview_open = !self.preview_open,
            ViewAction::Toggle(2) => self.templates_open = !self.templates_open,
            ViewAction::Toggle(_) => {}
            ViewAction::Consume => {}
        }
    }

    // Route a resolved Templates-panel click: picking a template opens its detail
    // panel (a preview of the assets it would add, with an Apply button); the
    // Templates list stays open so another can be picked.
    fn apply_templates(&mut self, action: TemplatesAction) {
        match action {
            TemplatesAction::Pick(i) => self.open_template_detail(i),
            TemplatesAction::Consume => {}
        }
    }

    // Open (or re-target) the Template detail panel on template `i`, bringing it to
    // the front of the focus stack.
    fn open_template_detail(&mut self, i: usize) {
        if i >= concinnity_templates::TEMPLATES.len() {
            return;
        }
        self.open_template = Some(i);
        self.template_list_scroll = 0;
        self.focus_panel(DragTarget::TemplateDetail);
    }

    // Close the Template detail panel (its state is transient; the Templates list
    // stays as it was).
    fn close_template_detail(&mut self) {
        self.open_template = None;
        self.template_list_scroll = 0;
    }

    // Route a resolved Template-detail click: Apply layers the template's assets
    // (idempotently) then closes the panel; the "X" just closes it.
    fn apply_template_detail(&mut self, action: TemplateAction) {
        match action {
            TemplateAction::Apply => {
                if let Some(i) = self.open_template {
                    self.apply_template(i);
                }
                self.close_template_detail();
            }
            TemplateAction::Close => self.close_template_detail(),
            TemplateAction::Consume => {}
        }
    }

    // The per-frame Template detail view data (title, description, grouped rows),
    // borrowed for both hit-testing and layout.
    fn template_detail_data(&self, i: usize) -> TemplateDetailData {
        let (title, description) = concinnity_templates::TEMPLATES
            .get(i)
            .map(|t| (format!("Template {}", t.title), t.description.to_string()))
            .unwrap_or_default();
        TemplateDetailData {
            title,
            description,
            rows: self.template_rows(i),
        }
    }

    fn make_template_view<'a>(
        &self,
        d: &'a TemplateDetailData,
        mouse: [f32; 2],
    ) -> TemplateView<'a> {
        TemplateView {
            title: &d.title,
            description: &d.description,
            rows: &d.rows,
            scroll: self.template_list_scroll,
            mouse,
        }
    }

    // Open the combo in `flavour`, clearing and focusing the shared filter field.
    fn open_combo(&mut self, flavour: Combo, world: &mut World) {
        self.combo = flavour;
        self.combo_scroll = 0;
        self.row_menu = None;
        widget::focus_field_with(world, panel::FILTER_INPUT, "");
    }

    // Open the add / edit form for `ty`: derive its editable arg fields from the
    // type's defaults (or the edited entry's current args), seed the name + each
    // text field, and focus the name. `editing` is `Some(idx)` for a rename +
    // arg-edit of an existing entry, `None` for a new asset.
    fn open_form(&mut self, world: &mut World, ty: String, editing: Option<usize>) {
        let seed = editing
            .and_then(|idx| self.entries.get(idx))
            .and_then(|e| e.get("args"))
            .and_then(|v| v.as_object())
            .cloned();
        let name = match editing {
            Some(idx) => self
                .entries
                .get(idx)
                .and_then(entry_name)
                .unwrap_or_default()
                .to_string(),
            None => self.unique_name(&ty),
        };
        // The working args tree: type defaults with the edited entry merged over
        // them. Add / remove and the controls mutate it; the fields are derived from
        // it so a structural change (a grown / shrunk array) re-derives cleanly.
        self.form_args = form::working_args(&ty, seed.as_ref());
        self.form_focus = FormFocus::Name;
        self.form_error = None;
        self.selected_type = Some(ty);
        self.editing = editing;
        self.combo = Combo::Closed;
        self.row_menu = None;
        self.field_dropdown = None;
        self.field_dropdown_scroll = 0;
        self.form_scroll = 0;
        self.vec_expanded.clear();
        // A freshly opened form comes to the front (the click that opened it focused
        // the Assets panel; the form the user is now editing should sit on top).
        self.focus_panel(DragTarget::Edit);
        self.refresh_form(world);
        widget::focus_field_with(world, form_panel::NAME_INPUT, &name);
    }

    // Derive the form's fields from the current working args, fill each reference
    // field's options, and (re-)seed the text controls. Called on open and after a
    // structural change (array add / remove) re-shapes the field list.
    fn refresh_form(&mut self, world: &mut World) {
        let Some(ty) = self.selected_type.clone() else {
            return;
        };
        self.form_fields = form::fields_for_with(&ty, Some(&self.form_args), &self.vec_expanded);
        // Clamp the scroll window to the (possibly changed) field count -- an array
        // shrink can leave `form_scroll` past the new last page.
        let max = self.form_fields.len().saturating_sub(form::FIELD_POOL);
        self.form_scroll = self.form_scroll.min(max);
        // Reference fields pick from the world's existing assets of their target
        // type. Resolve the option lists up front (reads `entries`) so the fill loop
        // does not borrow `self` twice.
        let ref_opts: Vec<(usize, Vec<String>)> = self
            .form_fields
            .iter()
            .enumerate()
            .filter_map(|(i, f)| match f.kind {
                form::FieldKind::Ref { target } => Some((i, names_of_type(&self.entries, target))),
                _ => None,
            })
            .collect();
        for (i, names) in ref_opts {
            form::set_ref_options(&mut self.form_fields[i], &names);
        }
        // Seed only the text controls inside the visible window (their pool is
        // slot-indexed). Bool (checkbox), Enum + Ref (cycle buttons), and Array (a
        // header) have no text input to seed.
        let scroll = self.form_scroll;
        for (j, field) in self.form_fields.iter().enumerate() {
            if !field.kind.has_text_input() {
                continue;
            }
            if let Some(r) = visible_slot(j, scroll) {
                widget::seed_field(world, form_panel::form_input(r), &field.initial);
            }
        }
    }

    // Capture the current control values into the working args, preserving its
    // structure (array lengths). Run before a structural change or commit so edits
    // are not lost when the fields re-derive.
    fn capture_controls(&mut self, world: &World) {
        let Some(ty) = self.selected_type.clone() else {
            return;
        };
        let scroll = self.form_scroll;
        let texts: Vec<String> = self
            .form_fields
            .iter()
            .enumerate()
            .map(|(j, f)| {
                if !f.kind.has_text_input() {
                    // State lives in the field (boolval / variant_idx) or its child
                    // leaves (a disclosed vector), not a control of its own.
                    String::new()
                } else if let Some(r) = visible_slot(j, scroll) {
                    // In the window: read the live control.
                    widget::field_text(world, form_panel::form_input(r))
                } else {
                    // Off-window: feed its stored value back so `assemble` round-trips
                    // it unchanged rather than blanking it (an empty string would
                    // overwrite a text field).
                    form::current_text(&self.form_args, &f.key)
                }
            })
            .collect();
        self.form_args = form::assemble(&ty, Some(&self.form_args), &self.form_fields, &texts);
    }

    // Close the form panel, discarding its transient state (the browse row's
    // highlight clears with `editing`).
    fn close_form(&mut self) {
        self.selected_type = None;
        self.editing = None;
        self.form_fields.clear();
        self.form_args = serde_json::Map::new();
        self.vec_expanded.clear();
        self.form_scroll = 0;
        self.form_focus = FormFocus::Name;
        self.form_error = None;
        self.field_dropdown = None;
        self.field_dropdown_scroll = 0;
    }

    // Route a resolved panel click. Field-focus transitions mutate the injected
    // `TextInput` components, so this needs the world.
    fn apply_panel(&mut self, action: PanelAction, world: &mut World) {
        match action {
            PanelAction::TogglePicker => {
                if self.combo == Combo::Picker {
                    self.combo = Combo::Closed;
                } else {
                    self.open_combo(Combo::Picker, world);
                }
            }
            PanelAction::ToggleFilter => {
                if self.combo == Combo::Filter {
                    self.combo = Combo::Closed;
                } else {
                    self.open_combo(Combo::Filter, world);
                }
            }
            PanelAction::PickOption(i) => match self.combo {
                Combo::Filter => {
                    if let Some(o) = self.combo_options(world).get(i) {
                        self.type_filter = if o == panel::ALL_LABEL {
                            None
                        } else {
                            Some(o.clone())
                        };
                    }
                    self.list_scroll = 0;
                    self.combo = Combo::Closed;
                }
                Combo::Picker => {
                    if let Some(ty) = self.combo_options(world).get(i).cloned() {
                        // A config singleton edits the world's existing instance if
                        // it has one, else adds it (edit-or-add); a multi-instance
                        // asset always adds a new one.
                        let existing = panel::is_singleton(&ty)
                            .then(|| {
                                self.entries
                                    .iter()
                                    .position(|e| entry_type(e) == Some(ty.as_str()))
                            })
                            .flatten();
                        self.open_form(world, ty, existing);
                    }
                }
                Combo::Closed => {}
            },
            PanelAction::OpenEntry(idx) => {
                if let Some(ty) = self.entries.get(idx).and_then(entry_type).map(String::from) {
                    self.open_form(world, ty, Some(idx));
                }
            }
            PanelAction::OpenRowMenu(entry) => self.row_menu = Some(entry),
            PanelAction::RowDelete => {
                if let Some(idx) = self.row_menu
                    && idx < self.entries.len()
                {
                    self.entries.remove(idx);
                    self.mark_changed();
                    // The open form indexes into `entries`: deleting the edited
                    // entry closes it, and deleting an earlier one shifts it.
                    match self.editing {
                        Some(e) if e == idx => self.close_form(),
                        Some(e) if e > idx => self.editing = Some(e - 1),
                        _ => {}
                    }
                }
                self.row_menu = None;
                let max = self.list_rows().len().saturating_sub(panel::MAX_ROWS);
                self.list_scroll = self.list_scroll.min(max);
            }
            PanelAction::CloseOverlays => {
                self.combo = Combo::Closed;
                self.row_menu = None;
            }
            PanelAction::Consume => {}
        }
    }

    // Route a resolved form-panel click. Field-focus transitions mutate the
    // injected `TextInput` components, so this needs the world.
    fn apply_form(&mut self, action: FormAction, world: &mut World) {
        match action {
            FormAction::FocusName => self.form_focus = FormFocus::Name,
            FormAction::FocusField(i) => self.form_focus = FormFocus::Field(i),
            FormAction::ToggleField(i) => {
                if let Some(f) = self.form_fields.get_mut(i) {
                    f.boolval = !f.boolval;
                }
                self.form_error = None;
            }
            FormAction::CycleField(i) => {
                if let Some(f) = self.form_fields.get_mut(i)
                    && !f.variants.is_empty()
                {
                    f.variant_idx = (f.variant_idx + 1) % f.variants.len();
                }
                self.form_error = None;
            }
            FormAction::OpenFieldDropdown(i) => {
                // Toggle: a second click on the open field's control closes it.
                self.field_dropdown = if self.field_dropdown == Some(i) {
                    None
                } else {
                    Some(i)
                };
                self.field_dropdown_scroll = 0;
                self.form_error = None;
            }
            FormAction::PickFieldOption(opt) => {
                if let Some(open) = self.field_dropdown
                    && let Some(f) = self.form_fields.get_mut(open)
                    && opt < f.variants.len()
                {
                    f.variant_idx = opt;
                }
                self.field_dropdown = None;
                self.form_error = None;
            }
            FormAction::AddArrayElement(j) => {
                self.capture_controls(world);
                if let (Some(ty), Some(path)) = (
                    self.selected_type.clone(),
                    self.form_fields.get(j).map(|f| f.key.clone()),
                ) {
                    form::add_array_elem(&ty, &mut self.form_args, &path);
                    self.form_focus = FormFocus::Name;
                    self.refresh_form(world);
                }
                self.form_error = None;
            }
            FormAction::RemoveArrayElement(j) => {
                self.capture_controls(world);
                if let Some(path) = self.form_fields.get(j).map(|f| f.key.clone()) {
                    form::remove_array_elem(&mut self.form_args, &path);
                    self.form_focus = FormFocus::Name;
                    self.refresh_form(world);
                }
                self.form_error = None;
            }
            FormAction::ToggleVecExpand(j) => {
                // Fold the live controls in first so an in-progress edit survives the
                // field list re-deriving with / without this vector's element leaves.
                self.capture_controls(world);
                if let Some(path) = self.form_fields.get(j).map(|f| f.key.clone()) {
                    if !self.vec_expanded.remove(&path) {
                        self.vec_expanded.insert(path);
                    }
                    self.form_focus = FormFocus::Name;
                    self.refresh_form(world);
                }
                self.form_error = None;
            }
            FormAction::Confirm => self.confirm_form(world),
            FormAction::Close => self.close_form(),
            FormAction::CloseOverlays => self.field_dropdown = None,
            FormAction::Consume => {}
        }
    }

    // Capture the form's controls into the working args, validate, and commit (add
    // a new entry or update the edited one). On a validation error the form stays
    // open with the message shown, so nothing invalid ever reaches world.jsonl.
    fn confirm_form(&mut self, world: &mut World) {
        self.form_error = None;
        let Some(ty) = self.selected_type.clone() else {
            self.close_form();
            return;
        };
        let typed = widget::field_text(world, form_panel::NAME_INPUT);
        // Fold the live control values into the working args (which already holds the
        // structure: nested objects, array lengths), then validate the whole thing.
        self.capture_controls(world);
        let args = self.form_args.clone();
        if let Err(e) = form::validate(&ty, &args) {
            self.form_error = Some(short_error(&e));
            return;
        }
        let args_val = serde_json::Value::Object(args);
        match self.editing {
            Some(idx) => {
                let name = self.finalize_rename(&typed, idx, &ty);
                if let Some(obj) = self.entries.get_mut(idx).and_then(|e| e.as_object_mut()) {
                    obj.insert("name".to_string(), serde_json::Value::String(name));
                    obj.insert("args".to_string(), args_val);
                }
            }
            None => {
                let name = self.finalize_name(&typed, &ty);
                self.entries.push(serde_json::json!({
                    "name": name, "type": ty, "args": args_val,
                }));
            }
        }
        self.mark_changed();
        self.close_form();
    }

    // Move the targeted region's scroll offset in the wheel direction. The tick
    // picks the target from the cursor position (both panels can be open).
    fn scroll(&mut self, delta: f32, target: ScrollTarget, world: &mut World) {
        match target {
            ScrollTarget::List if self.combo == Combo::Closed => {
                let max = self.list_rows().len().saturating_sub(panel::MAX_ROWS);
                self.list_scroll = scroll_step(self.list_scroll, delta, max);
                self.row_menu = None;
            }
            ScrollTarget::List => {
                let max = self
                    .combo_options(world)
                    .len()
                    .saturating_sub(panel::MAX_ROWS);
                self.combo_scroll = scroll_step(self.combo_scroll, delta, max);
            }
            ScrollTarget::Form => {
                if let Some(open) = self.field_dropdown {
                    // An open value dropdown scrolls its own option list.
                    let total = self.form_fields.get(open).map_or(0, |f| f.variants.len());
                    let max = total.saturating_sub(form_panel::MAX_DROP_ROWS);
                    self.field_dropdown_scroll =
                        scroll_step(self.field_dropdown_scroll, delta, max);
                } else {
                    // Scroll the field window: fold the visible controls into the
                    // working args, move the window, then re-seed the newly visible
                    // slots. The same capture / refresh cycle an array add / remove
                    // uses, so no in-progress edit is lost as the window moves.
                    let max = self.form_fields.len().saturating_sub(form::FIELD_POOL);
                    let next = scroll_step(self.form_scroll, delta, max);
                    if next == self.form_scroll {
                        return;
                    }
                    self.capture_controls(world);
                    self.form_scroll = next;
                    self.form_focus = FormFocus::Name;
                    self.refresh_form(world);
                }
            }
            ScrollTarget::TemplateList => {
                if let Some(i) = self.open_template {
                    let max = self
                        .template_rows(i)
                        .len()
                        .saturating_sub(super::asset_list::MAX_ROWS);
                    self.template_list_scroll = scroll_step(self.template_list_scroll, delta, max);
                }
            }
        }
    }

    fn hud_state(&self) -> HudState {
        HudState {
            dirty: self.dirty,
            view_open: self.view_open,
            visible: self.hud_visible,
        }
    }

    // While a title-bar drag is active, follow the cursor (clamped fully on
    // screen); releasing the button ends the drag.
    fn drive_drag(&mut self, input: &FrameInput, vp: [f32; 2]) {
        let Some(drag) = self.drag else {
            return;
        };
        if !input.left_button_down {
            self.drag = None;
            return;
        }
        let pos = [input.mouse_x - drag.grab[0], input.mouse_y - drag.grab[1]];
        match drag.target {
            DragTarget::Assets => {
                self.panel_pos = Some(widget::clamp_origin(pos, panel::size(), vp));
            }
            DragTarget::Edit => {
                let size = form_panel::size(self.form_fields.len());
                self.edit_pos = Some(widget::clamp_origin(pos, size, vp));
            }
            DragTarget::Preview => {
                self.preview_pos = Some(widget::clamp_origin(pos, preview::size(), vp));
            }
            DragTarget::View => {
                self.view_pos = Some(widget::clamp_origin(pos, view::size(), vp));
            }
            DragTarget::Templates => {
                self.templates_pos = Some(widget::clamp_origin(pos, templates::size(), vp));
            }
            DragTarget::TemplateDetail => {
                let n = self
                    .open_template
                    .map_or(1, |i| self.template_rows(i).len());
                let size = template_panel::size(n);
                self.template_detail_pos = Some(widget::clamp_origin(pos, size, vp));
            }
        }
    }

    // Route a press: the top bar first (it draws over the panels), then the panels
    // front-to-back so the frontmost claims a press in an overlap. Whichever panel
    // claims it comes to the front. A press on a panel's title bar starts a drag.
    fn route_click(&mut self, input: &FrameInput, vp: [f32; 2], world: &mut World) {
        let (mx, my) = (input.mouse_x, input.mouse_y);
        if let Some(a) = hud::hit_test(mx, my, true, self.dirty, vp[0]) {
            // SAVE only writes to disk now; it neither rebuilds nor re-injects the
            // world, so an open form is left intact (no blank-field risk).
            self.apply_top(a);
            self.combo = Combo::Closed;
            self.row_menu = None;
            return;
        }
        // Front-to-back: the frontmost shown panel to claim the press handles it and
        // rises to the front (so clicking an exposed sliver of a buried panel brings
        // it forward). `panel_order`'s tail is frontmost.
        let front_to_back: Vec<DragTarget> = self.panel_order.iter().rev().copied().collect();
        for target in front_to_back {
            if self.try_panel_press(target, mx, my, vp, world) {
                return;
            }
        }
    }

    // Try to resolve a press against panel `target`: `false` when it is hidden or
    // the press misses it (the caller tries the next panel back). A hit brings the
    // panel to the front; a title-bar press starts a drag, a body press resolves a
    // control.
    fn try_panel_press(
        &mut self,
        target: DragTarget,
        mx: f32,
        my: f32,
        vp: [f32; 2],
        world: &mut World,
    ) -> bool {
        match target {
            DragTarget::Preview => {
                if !self.preview_open {
                    return false;
                }
                let pv = self.preview_origin(vp);
                if !point_in(mx, my, preview::panel_rect(pv)) {
                    return false;
                }
                self.focus_panel(DragTarget::Preview);
                if point_in(mx, my, preview::close_rect(pv)) {
                    self.preview_open = false;
                } else if point_in(mx, my, preview::title_rect(pv)) {
                    self.drag = Some(Drag {
                        target,
                        grab: [mx - pv[0], my - pv[1]],
                    });
                } else if let Some(PreviewAction::ToggleCapture) = preview::hit_test(mx, my, pv) {
                    self.world_capture = !self.world_capture;
                }
                true
            }
            DragTarget::Edit => {
                // The form is part of the assets UI: interactive only while the
                // browse panel is open.
                if !(self.form_open() && self.panel_open) {
                    return false;
                }
                let fo = self.edit_origin(vp);
                // The X in the title bar closes the form; checked before the
                // title-bar drag so it never starts a drag instead.
                if point_in(mx, my, form_panel::close_rect(fo)) {
                    self.focus_panel(DragTarget::Edit);
                    self.apply_form(FormAction::Close, world);
                    return true;
                }
                if point_in(mx, my, form_panel::title_rect(fo)) {
                    self.focus_panel(DragTarget::Edit);
                    self.drag = Some(Drag {
                        target,
                        grab: [mx - fo[0], my - fo[1]],
                    });
                    return true;
                }
                let action = {
                    let data = self.panel_data(world);
                    let view = self.make_form_view(&data, [mx, my]);
                    form_panel::hit_test(&view, mx, my, fo)
                };
                if let Some(fa) = action {
                    self.focus_panel(DragTarget::Edit);
                    self.apply_form(fa, world);
                    return true;
                }
                false
            }
            DragTarget::Assets => {
                if !self.panel_open {
                    return false;
                }
                let po = self.panel_origin(vp);
                // The X in the title bar closes the Assets panel (state kept, like a
                // View-checkbox untick); checked before the title-bar drag.
                if point_in(mx, my, panel::close_rect(po)) {
                    self.focus_panel(DragTarget::Assets);
                    self.panel_open = false;
                    self.combo = Combo::Closed;
                    self.row_menu = None;
                    return true;
                }
                if point_in(mx, my, panel::title_rect(po)) {
                    self.focus_panel(DragTarget::Assets);
                    self.drag = Some(Drag {
                        target,
                        grab: [mx - po[0], my - po[1]],
                    });
                    return true;
                }
                let action = {
                    let data = self.panel_data(world);
                    let view = self.make_view(&data, [mx, my]);
                    panel::hit_test(&view, mx, my, po)
                };
                if let Some(pa) = action {
                    self.focus_panel(DragTarget::Assets);
                    self.apply_panel(pa, world);
                    return true;
                }
                false
            }
            DragTarget::View => {
                if !self.view_open {
                    return false;
                }
                let vo = self.view_origin(vp);
                if !point_in(mx, my, view::panel_rect(vo)) {
                    return false;
                }
                self.focus_panel(DragTarget::View);
                if point_in(mx, my, view::close_rect(vo)) {
                    self.view_open = false;
                } else if point_in(mx, my, view::title_rect(vo)) {
                    self.drag = Some(Drag {
                        target,
                        grab: [mx - vo[0], my - vo[1]],
                    });
                } else if let Some(a) = view::hit_test(mx, my, vo) {
                    self.apply_view(a);
                }
                true
            }
            DragTarget::Templates => {
                if !self.templates_open {
                    return false;
                }
                let to = self.templates_origin(vp);
                if !point_in(mx, my, templates::panel_rect(to)) {
                    return false;
                }
                self.focus_panel(DragTarget::Templates);
                if point_in(mx, my, templates::close_rect(to)) {
                    self.templates_open = false;
                } else if point_in(mx, my, templates::title_rect(to)) {
                    self.drag = Some(Drag {
                        target,
                        grab: [mx - to[0], my - to[1]],
                    });
                } else if let Some(a) = templates::hit_test(mx, my, to) {
                    self.apply_templates(a);
                }
                true
            }
            DragTarget::TemplateDetail => {
                // The detail panel is part of the Templates UI: interactive only
                // while the Templates list is open and a template is picked.
                let Some(i) = self.open_template.filter(|_| self.templates_open) else {
                    return false;
                };
                let to = self.template_detail_origin(i, vp);
                // The X in the title bar closes the detail; checked before the
                // title-bar drag so it never starts a drag instead.
                if point_in(mx, my, template_panel::close_rect(to)) {
                    self.focus_panel(DragTarget::TemplateDetail);
                    self.close_template_detail();
                    return true;
                }
                if point_in(mx, my, template_panel::title_rect(to)) {
                    self.focus_panel(DragTarget::TemplateDetail);
                    self.drag = Some(Drag {
                        target,
                        grab: [mx - to[0], my - to[1]],
                    });
                    return true;
                }
                let action = {
                    let data = self.template_detail_data(i);
                    let view = self.make_template_view(&data, [mx, my]);
                    template_panel::hit_test(&view, mx, my, to)
                };
                if let Some(a) = action {
                    self.focus_panel(DragTarget::TemplateDetail);
                    self.apply_template_detail(a);
                    return true;
                }
                false
            }
        }
    }
}

impl DebugHook for EditorHook {
    fn tick(&mut self, world: &mut World) {
        let input = world.query::<FrameInput>().last().cloned();
        if let Some(input) = &input {
            // Escape hands the cursor back to the editor (leaves play mode).
            if input.escape {
                self.world_capture = false;
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
                if input.left_click && self.drag.is_none() {
                    self.route_click(input, vp, world);
                }
                // Wheel routing: the frontmost scrollable panel under the cursor
                // takes the wheel (the form's field window or the Assets list). An
                // open value dropdown is modal and can extend past the panel, so it
                // scrolls the form from anywhere while open.
                if input.scroll_delta.abs() > 0.5 {
                    let (mx, my) = (input.mouse_x, input.mouse_y);
                    let form_shown = self.form_open() && self.panel_open;
                    if form_shown && self.field_dropdown.is_some() {
                        self.scroll(input.scroll_delta, ScrollTarget::Form, world);
                    } else {
                        let front_to_back: Vec<DragTarget> =
                            self.panel_order.iter().rev().copied().collect();
                        for target in front_to_back {
                            let hit = match target {
                                DragTarget::Edit if form_shown => {
                                    let data = self.panel_data(world);
                                    let view = self.make_form_view(&data, [mx, my]);
                                    form_panel::cursor_over(&view, mx, my, self.edit_origin(vp))
                                        .then_some(ScrollTarget::Form)
                                }
                                DragTarget::Assets if self.panel_open => {
                                    panel::cursor_over_body(mx, my, self.panel_origin(vp))
                                        .then_some(ScrollTarget::List)
                                }
                                DragTarget::TemplateDetail => self
                                    .open_template
                                    .filter(|_| self.templates_open)
                                    .and_then(|i| {
                                        let data = self.template_detail_data(i);
                                        let view = self.make_template_view(&data, [mx, my]);
                                        template_panel::cursor_over(
                                            &view,
                                            mx,
                                            my,
                                            self.template_detail_origin(i, vp),
                                        )
                                        .then_some(ScrollTarget::TemplateList)
                                    }),
                                _ => None,
                            };
                            if let Some(t) = hit {
                                self.scroll(input.scroll_delta, t, world);
                                break;
                            }
                        }
                    }
                }
            }
        }

        // Drive the world's cursor / freeze state: edit mode (`Some(true)`) frees
        // the cursor and freezes the world; play mode (`Some(false)`) runs it.
        world.insert_resource(MenuOverride(Some(!self.world_capture)));

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
        // Publish the panels' draw layers (focus stack) so the renderer occludes
        // overlaps by focus this frame. Empty while the HUD is hidden, so the
        // renderer skips the overlay sort entirely.
        if shown {
            self.publish_layers(world);
        } else {
            world.insert_resource(HudLayers::default());
        }
        // Preview panel (toggled from the View panel; shown by default).
        if shown && self.preview_open {
            preview::apply(world, self.preview_origin(vp), self.world_capture, mouse);
        } else {
            preview::hide_all(world);
        }
        // View panel (toggled from the top-bar View button).
        if shown && self.view_open {
            view::apply(world, self.view_origin(vp), self.view_state(), mouse);
        } else {
            view::hide_all(world);
        }
        // Templates panel (toggled from the View panel); the picked template's row
        // stays highlighted while its detail panel is open.
        if shown && self.templates_open {
            templates::apply(world, self.templates_origin(vp), self.open_template, mouse);
        } else {
            templates::hide_all(world);
        }
        // Template detail panel: shown only while the Templates list is open and a
        // template is picked; hiding it keeps `open_template` so a later toggle-on
        // restores it.
        let detail = self.open_template.filter(|_| shown && self.templates_open);
        if let Some(i) = detail {
            let data = self.template_detail_data(i);
            let view = self.make_template_view(&data, mouse);
            template_panel::apply(world, Some(&view), self.template_detail_origin(i, vp));
        } else {
            template_panel::apply(world, None, [0.0, 0.0]);
        }
        // Assets panel (toggled from the View panel).
        if shown && self.panel_open {
            let data = self.panel_data(world);
            let view = self.make_view(&data, mouse);
            panel::apply(world, Some(&view), self.panel_origin(vp));
        } else {
            panel::apply(world, None, [0.0, 0.0]);
        }
        // The edit form shows only while the assets UI is on (panel_open); hiding
        // it keeps its state so a later toggle-on restores it.
        let show_form = shown && self.panel_open && self.form_open();
        if show_form {
            let data = self.panel_data(world);
            let view = self.make_form_view(&data, mouse);
            form_panel::apply(world, Some(&view), self.edit_origin(vp));
        } else {
            form_panel::apply(world, None, [0.0, 0.0]);
        }
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
        staged.world_mut().remove_all::<crate::assets::DebugHud>();
        // Carry the editor's typed text (an open form's name + fields, the combo
        // filter) across the fresh HUD injection so it is not blanked.
        let fields = Self::field_snapshot(app.world());
        super::inject::editor_hud(staged.world_mut());
        Self::restore_fields(staged.world_mut(), &fields);
        staged
            .world_mut()
            .insert_resource(MenuOverride(Some(!self.world_capture)));

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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::assets::{Sprite, TextInput, TextLabel};

    fn hook(entries: Vec<serde_json::Value>) -> EditorHook {
        EditorHook::new("unused.jsonl".to_string(), entries)
    }

    // Point the cook's content-addressed cache at a private temp dir for the test
    // process, so the in-memory rebuild tests never touch the working directory.
    fn isolate_state_dir() {
        use std::sync::Once;
        static ONCE: Once = Once::new();
        ONCE.call_once(|| {
            let dir = std::env::temp_dir().join(format!("cn-editor-tests-{}", std::process::id()));
            let _ = std::fs::create_dir_all(&dir);
            concinnity_core::paths::set_root(dir);
        });
    }

    // A world holding just a FrameInput, for driving `tick` directly.
    fn world_with_input(input: FrameInput) -> World {
        let mut world = World::new_empty();
        world.add_component(input);
        world
    }

    // A world with the injected typed fields, for the add / edit flow (the
    // combo filter, the form's name heading, and its arg-input pool).
    fn world_with_fields() -> World {
        let mut world = World::new_empty();
        for id in panel::all_field_ids()
            .into_iter()
            .chain(form_panel::all_field_ids())
        {
            world.add_component(TextInput {
                asset_id: id,
                ..Default::default()
            });
        }
        world
    }

    fn set_field(world: &mut World, id: crate::ecs::asset_id::AssetId, text: &str) {
        for t in world.query_mut::<TextInput>() {
            if t.asset_id == id {
                t.content = text.to_string();
                break;
            }
        }
    }

    fn entry(name: &str, ty: &str) -> serde_json::Value {
        serde_json::json!({"name": name, "type": ty, "args": {}})
    }

    #[test]
    fn starts_in_edit_mode_with_hud_shown() {
        let h = hook(Vec::new());
        assert!(!h.world_capture, "editor holds the cursor at launch");
        assert!(h.hud_visible, "HUD shown at launch");
        // Assets / View / Templates start closed; Preview starts shown.
        assert!(!h.panel_open && !h.view_open && !h.templates_open);
        assert!(h.preview_open, "the Preview panel is shown at launch");
        assert_eq!(h.combo, Combo::Closed);
    }

    // The top-bar View button toggles the View panel; the View panel's rows toggle
    // the Assets, Preview, and Templates panels independently (no mutual exclusion).
    #[test]
    fn view_button_and_view_rows_toggle_the_panels() {
        let mut h = hook(Vec::new());
        h.apply_top(HudAction::ToggleView);
        assert!(h.view_open, "the View button shows the View panel");
        h.apply_top(HudAction::ToggleView);
        assert!(!h.view_open, "a second click hides it");
        // Row 0 -> Assets, row 1 -> Preview, row 2 -> Templates.
        h.apply_view(ViewAction::Toggle(0));
        assert!(h.panel_open, "row 0 shows the Assets panel");
        h.apply_view(ViewAction::Toggle(1));
        assert!(
            !h.preview_open,
            "row 1 hides the (default-shown) Preview panel"
        );
        h.apply_view(ViewAction::Toggle(2));
        assert!(h.templates_open, "row 2 shows the Templates panel");
        assert!(
            h.panel_open,
            "Assets stayed shown -- panels are independent"
        );
    }

    // Picking a template opens its detail panel (nothing is added yet); Apply from
    // the detail layers the assets and closes it; re-applying is idempotent.
    #[test]
    fn template_pick_opens_detail_then_apply_adds_idempotently() {
        let mut h = hook(Vec::new());
        h.apply_templates(TemplatesAction::Pick(0));
        assert_eq!(h.open_template, Some(0), "the detail panel opens on pick");
        assert!(h.entries.is_empty(), "picking adds nothing on its own");

        h.apply_template_detail(TemplateAction::Apply);
        let first = concinnity_templates::TEMPLATES[0].assets().len();
        assert_eq!(h.entries.len(), first, "Apply adds all template entries");
        assert_eq!(h.open_template, None, "Apply closes the detail panel");

        // Re-open and Apply again: no duplicate entries.
        h.apply_templates(TemplatesAction::Pick(0));
        h.apply_template_detail(TemplateAction::Apply);
        assert_eq!(h.entries.len(), first, "re-apply is idempotent");
    }

    // The detail panel's grouped rows come from the shared list model, so they
    // match what the template would add (one row per asset plus a type header
    // each), and the "X" closes the panel without adding anything.
    #[test]
    fn template_detail_rows_and_close() {
        let mut h = hook(Vec::new());
        h.open_template_detail(0);
        let rows = h.template_rows(0);
        let names = rows.iter().filter(|r| !r.is_header).count();
        assert_eq!(
            names,
            concinnity_templates::TEMPLATES[0].assets().len(),
            "one name row per template asset"
        );
        assert!(
            rows.iter().any(|r| r.is_header),
            "grouped under type headers"
        );
        h.apply_template_detail(TemplateAction::Close);
        assert_eq!(h.open_template, None);
        assert!(h.entries.is_empty(), "closing adds nothing");
    }

    // Entry changes drive the live preview: a mutation flags a rebuild AND marks
    // the world dirty (unsaved); a plain View toggle does neither.
    #[test]
    fn entry_changes_request_a_preview_rebuild() {
        let mut h = hook(Vec::new());
        h.apply_top(HudAction::ToggleView);
        assert!(
            !h.rebuild_preview && !h.dirty,
            "a view toggle is not an entry change"
        );
        // Applying a template layers assets: preview rebuild requested + dirty.
        h.open_template_detail(0);
        h.apply_template_detail(TemplateAction::Apply);
        assert!(
            h.rebuild_preview && h.dirty,
            "applying a template updates the live preview and marks unsaved"
        );
    }

    // A live rebuild re-injects a fresh (blank) HUD; the field snapshot carries the
    // editor's typed text (an open form's name, the combo filter) across it so a
    // form open during the swap is not blanked.
    #[test]
    fn field_snapshot_carries_typed_text_across_a_reinjection() {
        let mut old = World::new_empty();
        super::super::inject::editor_hud(&mut old);
        widget::seed_field(&mut old, form_panel::NAME_INPUT, "my_light");
        widget::seed_field(&mut old, panel::FILTER_INPUT, "Point");
        let snapshot = EditorHook::field_snapshot(&old);

        // A fresh HUD injection starts every field blank.
        let mut new = World::new_empty();
        super::super::inject::editor_hud(&mut new);
        assert_eq!(widget::field_text(&new, form_panel::NAME_INPUT), "");

        EditorHook::restore_fields(&mut new, &snapshot);
        assert_eq!(widget::field_text(&new, form_panel::NAME_INPUT), "my_light");
        assert_eq!(widget::field_text(&new, panel::FILTER_INPUT), "Point");
    }

    // The live preview is rebuilt from the in-memory entries with no disk access:
    // authored renderable entries build a rendering world directly, and an empty
    // world is seeded so a window still shows. This is the swap's source of truth
    // now that SAVE only persists.
    #[test]
    fn build_preview_world_renders_from_in_memory_entries() {
        isolate_state_dir();
        // Authored renderable entries (a Room + camera) build a rendering world.
        let h = hook(vec![
            serde_json::json!({"name":"cam","type":"Camera3D","args":{}}),
            serde_json::json!({"name":"room","type":"Room","args":{}}),
        ]);
        assert!(
            h.build_preview_world()
                .expect("authored entries build")
                .renders(),
            "authored renderable entries render without disk"
        );
        // Empty entries: the seed keeps the preview window from going blank.
        let h = hook(Vec::new());
        assert!(
            h.build_preview_world()
                .expect("empty world seeds")
                .renders(),
            "an empty world is seeded so it still renders"
        );
    }

    #[test]
    fn list_rows_group_names_under_type_headers() {
        let h = hook(vec![
            entry("a", "PointLight"),
            entry("b", "Decal"),
            entry("c", "PointLight"),
        ]);
        let rows = h.list_rows();
        // Types sorted: Decal (header, b), then PointLight (header, a, c).
        assert!(rows[0].is_header && rows[0].text == "Decal");
        assert_eq!(rows[1].text, "b");
        assert!(rows[2].is_header && rows[2].text == "PointLight");
        let names: Vec<&str> = rows[3..].iter().map(|r| r.text.as_str()).collect();
        assert_eq!(names, ["a", "c"]);
        // Name rows carry their entry index; headers do not.
        assert_eq!(rows[1].entry, Some(1));
        assert_eq!(rows[0].entry, None);
    }

    #[test]
    fn filter_narrows_the_grouped_list() {
        let mut h = hook(vec![
            entry("a", "PointLight"),
            entry("b", "Decal"),
            entry("c", "PointLight"),
        ]);
        let mut world = world_with_fields();
        h.panel_open = true;
        // Open the filter combo, then pick "PointLight".
        h.open_combo(Combo::Filter, &mut world);
        let opts = h.combo_options(&world);
        assert_eq!(opts[0], panel::ALL_LABEL);
        let pl = opts.iter().position(|o| o == "PointLight").unwrap();
        h.apply_panel(PanelAction::PickOption(pl), &mut world);
        assert_eq!(h.type_filter.as_deref(), Some("PointLight"));
        assert_eq!(h.combo, Combo::Closed);
        let rows = h.list_rows();
        // Only the PointLight group: one header + two names.
        assert_eq!(rows.len(), 3);
        assert!(rows[0].is_header && rows[0].text == "PointLight");
    }

    #[test]
    fn plus_picker_then_name_form_adds_the_entry() {
        let mut h = hook(Vec::new());
        let mut world = world_with_fields();
        h.panel_open = true;
        // "+" opens the picker; the filter field is focused.
        h.apply_panel(PanelAction::TogglePicker, &mut world);
        assert_eq!(h.combo, Combo::Picker);
        assert!(
            world
                .query::<TextInput>()
                .find(|t| t.asset_id == panel::FILTER_INPUT)
                .unwrap()
                .focused
        );
        // Pick the first offered type -> AddForm, name field prefilled + focused.
        let ty = h.combo_options(&world)[0].clone();
        h.apply_panel(PanelAction::PickOption(0), &mut world);
        assert!(h.form_open());
        assert_eq!(h.combo, Combo::Closed);
        assert_eq!(h.selected_type.as_deref(), Some(ty.as_str()));
        assert!(h.editing.is_none());
        let name_field = world
            .query::<TextInput>()
            .find(|t| t.asset_id == form_panel::NAME_INPUT)
            .unwrap();
        assert!(name_field.focused && !name_field.content.is_empty());
        // Edit the name, then confirm.
        set_field(&mut world, form_panel::NAME_INPUT, "my_light");
        h.apply_form(FormAction::Confirm, &mut world);
        assert!(!h.form_open());
        assert!(h.dirty);
        assert_eq!(h.entries.len(), 1);
        assert_eq!(h.entries[0]["name"], "my_light");
        assert_eq!(h.entries[0]["type"], ty.as_str());
    }

    #[test]
    fn row_click_opens_the_edit_form_for_a_rename() {
        let mut h = hook(vec![entry("lamp", "PointLight")]);
        let mut world = world_with_fields();
        h.panel_open = true;
        // Clicking the name row opens the edit form prefilled for a rename.
        h.apply_panel(PanelAction::OpenEntry(0), &mut world);
        assert!(h.form_open());
        assert_eq!(h.editing, Some(0));
        assert_eq!(h.selected_type.as_deref(), Some("PointLight"));
        assert!(h.row_menu.is_none());
        let name_field = world
            .query::<TextInput>()
            .find(|t| t.asset_id == form_panel::NAME_INPUT)
            .unwrap();
        assert_eq!(name_field.content, "lamp", "name prefilled from the entry");
        // Rename and confirm: same entry, no new one.
        set_field(&mut world, form_panel::NAME_INPUT, "streetlamp");
        h.apply_form(FormAction::Confirm, &mut world);
        assert_eq!(h.entries.len(), 1, "edited in place, not appended");
        assert_eq!(h.entries[0]["name"], "streetlamp");
        assert_eq!(h.entries[0]["type"], "PointLight");
        assert!(h.dirty);
    }

    #[test]
    fn row_menu_delete_removes_the_entry() {
        let mut h = hook(vec![entry("a", "Decal"), entry("b", "Decal")]);
        let mut world = world_with_fields();
        h.panel_open = true;
        h.apply_panel(PanelAction::OpenRowMenu(0), &mut world);
        h.apply_panel(PanelAction::RowDelete, &mut world);
        assert_eq!(h.entries.len(), 1);
        assert_eq!(h.entries[0]["name"], "b");
        assert!(h.dirty && h.row_menu.is_none());
    }

    #[test]
    fn edit_rename_to_a_duplicate_is_suffixed() {
        let mut h = hook(vec![entry("a", "Decal"), entry("b", "Decal")]);
        let mut world = world_with_fields();
        h.editing = Some(1);
        h.selected_type = Some("Decal".to_string());
        // Rename "b" to "a": collides with the other entry -> suffixed.
        set_field(&mut world, form_panel::NAME_INPUT, "a");
        h.apply_form(FormAction::Confirm, &mut world);
        assert_eq!(h.entries[1]["name"], "a_1");
    }

    #[test]
    fn confirm_add_with_blank_name_uses_a_generated_one() {
        let mut h = hook(Vec::new());
        let mut world = world_with_fields();
        h.selected_type = Some("PointLight".to_string());
        // Field left blank.
        h.apply_form(FormAction::Confirm, &mut world);
        assert_eq!(h.entries.len(), 1);
        assert_eq!(h.entries[0]["name"], "editor_pointlight");
    }

    #[test]
    fn confirm_add_makes_a_duplicate_name_unique() {
        let mut h = hook(vec![entry("lamp", "PointLight")]);
        let mut world = world_with_fields();
        h.selected_type = Some("PointLight".to_string());
        set_field(&mut world, form_panel::NAME_INPUT, "lamp");
        h.apply_form(FormAction::Confirm, &mut world);
        assert_eq!(h.entries[1]["name"], "lamp_1", "collision is suffixed");
    }

    // Picking a config singleton from the "+" picker edits the world's existing
    // instance if it has one (no second append), and adds one if it does not.
    #[test]
    fn config_singleton_picker_edits_existing_else_adds() {
        // A world that already has a GraphicsConfig: picking it opens an EDIT.
        let mut h = hook(vec![serde_json::json!({
            "name": "gfx", "type": "GraphicsConfig", "args": {}
        })]);
        let mut world = world_with_fields();
        h.panel_open = true;
        h.apply_panel(PanelAction::TogglePicker, &mut world);
        let gi = h
            .combo_options(&world)
            .iter()
            .position(|o| o == "GraphicsConfig")
            .expect("GraphicsConfig is offered in the picker");
        h.apply_panel(PanelAction::PickOption(gi), &mut world);
        assert!(h.form_open());
        assert_eq!(
            h.editing,
            Some(0),
            "picking a present singleton edits it, not a new add"
        );
        h.apply_form(FormAction::Confirm, &mut world);
        assert_eq!(
            h.entries
                .iter()
                .filter(|e| e["type"] == "GraphicsConfig")
                .count(),
            1,
            "the singleton was edited in place, never duplicated"
        );

        // A world WITHOUT the singleton: picking it opens a fresh add.
        let mut h2 = hook(Vec::new());
        let mut world2 = world_with_fields();
        h2.panel_open = true;
        h2.apply_panel(PanelAction::TogglePicker, &mut world2);
        let wi = h2
            .combo_options(&world2)
            .iter()
            .position(|o| o == "Window")
            .expect("Window is offered in the picker");
        h2.apply_panel(PanelAction::PickOption(wi), &mut world2);
        assert!(h2.form_open());
        assert!(h2.editing.is_none(), "no existing Window -> an add form");
        h2.apply_form(FormAction::Confirm, &mut world2);
        assert_eq!(
            h2.entries.iter().filter(|e| e["type"] == "Window").count(),
            1,
            "the missing singleton was added"
        );
    }

    #[test]
    fn cancel_form_returns_to_the_list_without_adding() {
        let mut h = hook(Vec::new());
        let mut world = world_with_fields();
        h.selected_type = Some("Decal".to_string());
        h.apply_form(FormAction::Close, &mut world);
        assert!(!h.form_open());
        assert!(h.selected_type.is_none() && h.editing.is_none());
        assert!(h.entries.is_empty() && !h.dirty);
    }

    #[test]
    fn picker_lists_types_alphabetically() {
        let mut h = hook(Vec::new());
        let world = world_with_fields();
        h.combo = Combo::Picker;
        // A prior browse filter does not reorder the picker: it stays A->Z.
        h.type_filter = Some("Decal".to_string());
        let opts = h.combo_options(&world);
        let mut sorted = opts.clone();
        sorted.sort();
        assert_eq!(opts, sorted, "the picker is alphabetized ascending");
        assert_eq!(
            opts.len(),
            panel::picker_types().count(),
            "every offered type shown (addables + config singletons)"
        );
        // Concretely: AudioCue sorts before Sprite, and a config singleton is mixed
        // in alphabetically (Application sorts before AudioCue).
        let pos = |t: &str| opts.iter().position(|o| o == t).unwrap();
        assert!(pos("AudioCue") < pos("Sprite"));
        assert!(pos("Application") < pos("AudioCue"));
    }

    #[test]
    fn close_overlays_dismisses_combo_and_menu() {
        let mut h = hook(vec![entry("a", "Decal")]);
        let mut world = world_with_fields();
        h.combo = Combo::Filter;
        h.row_menu = Some(0);
        h.apply_panel(PanelAction::CloseOverlays, &mut world);
        assert_eq!(h.combo, Combo::Closed);
        assert!(h.row_menu.is_none());
    }

    #[test]
    fn tick_escape_returns_cursor_to_editor() {
        let mut h = hook(Vec::new());
        h.world_capture = true;
        let mut world = world_with_input(FrameInput {
            escape: true,
            viewport: [1280.0, 720.0],
            ..Default::default()
        });
        h.tick(&mut world);
        assert!(!h.world_capture, "Escape leaves play mode");
    }

    #[test]
    fn tick_f1_toggles_hud_visibility() {
        let mut h = hook(Vec::new());
        let mut world = world_with_input(FrameInput {
            hud_toggle: true,
            viewport: [1280.0, 720.0],
            ..Default::default()
        });
        assert!(h.hud_visible);
        h.tick(&mut world);
        assert!(!h.hud_visible, "first F1 hides the HUD");
        h.tick(&mut world);
        assert!(h.hud_visible, "second F1 shows it again");
    }

    // Overwrite the world's FrameInput in place (tick reads the live component).
    fn set_input(world: &mut World, input: FrameInput) {
        if let Some(i) = world.query_mut::<FrameInput>().last() {
            *i = input;
        } else {
            world.add_component(input);
        }
    }

    // Clicking the Preview panel's capture row hands the cursor to the world;
    // clicking again takes it back.
    #[test]
    fn preview_capture_row_click_toggles_play_mode() {
        let mut h = hook(Vec::new());
        let vp = [1280.0, 720.0];
        let o = h.preview_origin(vp);
        let row_y = o[1] + preview::size()[1] - 5.0;
        let mut world = world_with_input(FrameInput {
            viewport: vp,
            mouse_x: o[0] + 10.0,
            mouse_y: row_y,
            left_click: true,
            left_button_down: true,
            ..Default::default()
        });
        h.tick(&mut world);
        assert!(h.world_capture, "the checkbox click enters play mode");
        set_input(
            &mut world,
            FrameInput {
                viewport: vp,
                mouse_x: o[0] + 10.0,
                mouse_y: row_y,
                left_click: true,
                left_button_down: true,
                ..Default::default()
            },
        );
        h.tick(&mut world);
        assert!(!h.world_capture, "a second click leaves it");
    }

    // Holding a panel's title bar drags it; the origin follows the cursor by the
    // grab offset and hard-stops at the window edges. Release ends the drag.
    #[test]
    fn title_bar_drag_moves_and_clamps_the_assets_panel() {
        let mut h = hook(Vec::new());
        h.panel_open = true;
        let vp = [1280.0, 720.0];
        let start = h.panel_origin(vp);
        // Press on the title bar, 10 px in from its corner.
        let mut world = world_with_input(FrameInput {
            viewport: vp,
            mouse_x: start[0] + 10.0,
            mouse_y: start[1] + 10.0,
            left_click: true,
            left_button_down: true,
            ..Default::default()
        });
        h.tick(&mut world);
        assert!(h.drag.is_some(), "the title press starts a drag");

        // Hold and move: the origin follows, preserving the grab offset.
        set_input(
            &mut world,
            FrameInput {
                viewport: vp,
                mouse_x: 400.0,
                mouse_y: 150.0,
                left_button_down: true,
                ..Default::default()
            },
        );
        h.tick(&mut world);
        assert_eq!(h.panel_origin(vp), [390.0, 140.0]);

        // Drag far past the top-left corner: the panel hard-stops at the edge.
        set_input(
            &mut world,
            FrameInput {
                viewport: vp,
                mouse_x: -500.0,
                mouse_y: -500.0,
                left_button_down: true,
                ..Default::default()
            },
        );
        h.tick(&mut world);
        assert_eq!(h.panel_origin(vp), [0.0, 0.0], "never partially off screen");

        // Release ends the drag; the panel stays where it was dropped.
        set_input(
            &mut world,
            FrameInput {
                viewport: vp,
                left_button_down: false,
                ..Default::default()
            },
        );
        h.tick(&mut world);
        assert!(h.drag.is_none(), "release ends the drag");
        assert_eq!(h.panel_origin(vp), [0.0, 0.0]);
    }

    // The Preview panel drags by its own title bar, clamped to the window's far
    // corner by its own (smaller) footprint.
    #[test]
    fn title_bar_drag_moves_and_clamps_the_preview_panel() {
        let mut h = hook(Vec::new());
        let vp = [1280.0, 720.0];
        let start = h.preview_origin(vp);
        let mut world = world_with_input(FrameInput {
            viewport: vp,
            mouse_x: start[0] + 5.0,
            mouse_y: start[1] + 5.0,
            left_click: true,
            left_button_down: true,
            ..Default::default()
        });
        h.tick(&mut world);
        assert!(h.drag.is_some());
        set_input(
            &mut world,
            FrameInput {
                viewport: vp,
                mouse_x: 5000.0,
                mouse_y: 5000.0,
                left_button_down: true,
                ..Default::default()
            },
        );
        h.tick(&mut world);
        let size = preview::size();
        assert_eq!(
            h.preview_origin(vp),
            [vp[0] - size[0], vp[1] - size[1]],
            "stops flush with the bottom-right corner"
        );
    }

    // While a drag is in progress the press's click must not also resolve to a
    // control underneath on later frames -- e.g. dragging the Assets panel across
    // the Preview checkbox must not toggle play mode.
    #[test]
    fn dragging_does_not_trigger_controls_it_crosses() {
        let mut h = hook(Vec::new());
        h.panel_open = true;
        let vp = [1280.0, 720.0];
        let start = h.panel_origin(vp);
        let mut world = world_with_input(FrameInput {
            viewport: vp,
            mouse_x: start[0] + 10.0,
            mouse_y: start[1] + 10.0,
            left_click: true,
            left_button_down: true,
            ..Default::default()
        });
        h.tick(&mut world);
        // Cross the Preview panel's capture row with the button still held and a
        // stray click edge (e.g. from event coalescing).
        let pv = h.preview_origin(vp);
        set_input(
            &mut world,
            FrameInput {
                viewport: vp,
                mouse_x: pv[0] + 10.0,
                mouse_y: pv[1] + preview::size()[1] - 5.0,
                left_click: true,
                left_button_down: true,
                ..Default::default()
            },
        );
        h.tick(&mut world);
        assert!(!h.world_capture, "the drag swallowed the click");
        assert!(h.drag.is_some(), "still dragging");
    }

    // Clicking an asset's name row in the browse list (not just its row menu)
    // opens the edit-form panel for that entry, and the row stays selected.
    #[test]
    fn clicking_a_list_row_opens_its_edit_form() {
        let mut h = hook(vec![entry("lamp", "PointLight")]);
        h.panel_open = true;
        let vp = [1280.0, 720.0];
        let po = h.panel_origin(vp);
        // Row 0 is the type header; row 1 is the name.
        let row = panel::list_row_rect(po, 1);
        let mut world = World::new_empty();
        super::super::inject::editor_hud(&mut world);
        world.add_component(FrameInput {
            viewport: vp,
            mouse_x: row[0] + 20.0,
            mouse_y: row[1] + 10.0,
            left_click: true,
            left_button_down: true,
            ..Default::default()
        });
        h.tick(&mut world);
        assert!(h.form_open(), "the row click opened the form");
        assert_eq!(h.editing, Some(0));
        assert_eq!(h.selected_type.as_deref(), Some("PointLight"));
        assert_eq!(
            widget::field_text(&world, form_panel::NAME_INPUT),
            "lamp",
            "the name heading is seeded from the entry"
        );
    }

    // Deleting an entry while a form is open keeps the form's entry index valid:
    // deleting the edited entry closes it; deleting an earlier one shifts it.
    #[test]
    fn deleting_entries_fixes_up_the_open_form_index() {
        let mut h = hook(vec![entry("a", "Decal"), entry("b", "Decal")]);
        let mut world = world_with_fields();
        h.panel_open = true;
        // Edit "b" (index 1), then delete "a" (index 0): the form now edits 0.
        h.open_form(&mut world, "Decal".to_string(), Some(1));
        h.apply_panel(PanelAction::OpenRowMenu(0), &mut world);
        h.apply_panel(PanelAction::RowDelete, &mut world);
        assert!(h.form_open(), "the form survives an unrelated delete");
        assert_eq!(h.editing, Some(0), "the edited index shifted down");
        // Confirm still updates the right (renamed-index) entry.
        set_field(&mut world, form_panel::NAME_INPUT, "b2");
        h.apply_form(FormAction::Confirm, &mut world);
        assert_eq!(h.entries.len(), 1);
        assert_eq!(h.entries[0]["name"], "b2");

        // Deleting the edited entry itself closes the form.
        let mut h2 = hook(vec![entry("a", "Decal")]);
        let mut world2 = world_with_fields();
        h2.panel_open = true;
        h2.open_form(&mut world2, "Decal".to_string(), Some(0));
        h2.apply_panel(PanelAction::OpenRowMenu(0), &mut world2);
        h2.apply_panel(PanelAction::RowDelete, &mut world2);
        assert!(!h2.form_open(), "deleting the edited entry closes its form");
    }

    // The edit-form panel drags by its own title bar, independent of the Assets
    // panel.
    #[test]
    fn edit_panel_drags_by_its_title_bar() {
        let mut h = hook(vec![entry("lamp", "PointLight")]);
        let mut world = World::new_empty();
        super::super::inject::editor_hud(&mut world);
        h.panel_open = true;
        h.open_form(&mut world, "PointLight".to_string(), Some(0));
        let vp = [1280.0, 720.0];
        let fo = h.edit_origin(vp);
        world.add_component(FrameInput {
            viewport: vp,
            mouse_x: fo[0] + 12.0,
            mouse_y: fo[1] + 8.0,
            left_click: true,
            left_button_down: true,
            ..Default::default()
        });
        h.tick(&mut world);
        assert!(h.drag.is_some(), "the form title press starts a drag");
        set_input(
            &mut world,
            FrameInput {
                viewport: vp,
                mouse_x: 112.0,
                mouse_y: 208.0,
                left_button_down: true,
                ..Default::default()
            },
        );
        h.tick(&mut world);
        assert_eq!(h.edit_origin(vp), [100.0, 200.0]);
        assert_eq!(
            h.panel_origin(vp),
            panel::default_origin(vp[0]),
            "the Assets panel did not move"
        );
    }

    // Focusing a panel moves it to the front of the stack (drawn on top, first
    // clicked) without duplicating it.
    #[test]
    fn focusing_a_panel_moves_it_to_the_front() {
        let mut h = hook(Vec::new());
        let panels = h.panel_order.len();
        // Default order matches the injected draw order: the Template detail panel
        // frontmost (over the Templates list it spawns from).
        assert_eq!(
            h.panel_order.last().copied(),
            Some(DragTarget::TemplateDetail)
        );
        h.focus_panel(DragTarget::Assets);
        assert_eq!(h.panel_order.last().copied(), Some(DragTarget::Assets));
        assert_eq!(h.panel_order.len(), panels, "no duplicates");
        // Re-focusing the frontmost is a no-op.
        h.focus_panel(DragTarget::Assets);
        assert_eq!(h.panel_order.last().copied(), Some(DragTarget::Assets));
        assert_eq!(h.panel_order.len(), panels);
    }

    // The published HUD layers rank the panels by focus (frontmost highest) and pin
    // the top bar above them all, so the renderer occludes overlaps cleanly.
    #[test]
    fn publish_layers_ranks_panels_below_the_top_bar() {
        let mut h = hook(Vec::new());
        h.focus_panel(DragTarget::Edit); // Edit -> frontmost
        let layers = h.compute_layers();
        let layer = |id| *layers.get(&id).expect("id mapped");
        let edit = layer(form_panel::EDIT_BG);
        let assets = layer(panel::PANEL_BG);
        let preview = layer(preview::TITLE_BG);
        assert!(
            edit > assets && edit > preview,
            "the frontmost panel outranks the others"
        );
        assert!(
            layer(hud::SAVE_BUTTON) > edit,
            "the top bar sits above every panel"
        );
        // A panel's text input shares its panel's layer (it must not sink below it).
        assert_eq!(layer(form_panel::NAME_INPUT), edit);
    }

    // A press on a shown panel brings it to the front and (on its title bar) starts
    // a drag.
    #[test]
    fn a_panel_press_brings_it_to_the_front() {
        let mut h = hook(vec![entry("lamp", "PointLight")]);
        let mut world = world_with_fields();
        h.panel_open = true;
        let vp = [1280.0, 720.0];
        let po = h.panel_origin(vp);
        let t = panel::title_rect(po);
        let claimed = h.try_panel_press(DragTarget::Assets, t[0] + 5.0, t[1] + 5.0, vp, &mut world);
        assert!(claimed, "the press was claimed by the Assets panel");
        assert_eq!(h.panel_order.last().copied(), Some(DragTarget::Assets));
        assert!(h.drag.is_some(), "a title-bar press starts a drag");
    }

    // The X in the edit form's title bar closes the form: the hook routes it before
    // the title-bar drag, so it closes rather than starting a drag.
    #[test]
    fn edit_form_title_bar_x_closes_the_form() {
        let mut h = hook(vec![entry("lamp", "PointLight")]);
        let mut world = world_with_fields();
        h.panel_open = true;
        h.open_form(&mut world, "PointLight".to_string(), Some(0));
        assert!(h.form_open());
        let vp = [1280.0, 720.0];
        let x = form_panel::close_rect(h.edit_origin(vp));
        let claimed = h.try_panel_press(DragTarget::Edit, x[0] + 5.0, x[1] + 5.0, vp, &mut world);
        assert!(claimed, "the X press was claimed");
        assert!(!h.form_open(), "the X closed the form");
        assert!(h.drag.is_none(), "the X did not start a drag");
    }

    // Every floating panel's title-bar X closes it: the press is checked before the
    // title drag, so it closes rather than starting a drag.
    #[test]
    fn every_panel_title_bar_x_closes_it() {
        let vp = [1280.0, 720.0];
        let mut world = world_with_fields();

        // Preview starts shown; its X hides it.
        let mut h = hook(Vec::new());
        let px = preview::close_rect(h.preview_origin(vp));
        assert!(h.try_panel_press(
            DragTarget::Preview,
            px[0] + 5.0,
            px[1] + 5.0,
            vp,
            &mut world
        ));
        assert!(
            !h.preview_open && h.drag.is_none(),
            "Preview X closed it, no drag"
        );

        // Assets.
        let mut h = hook(Vec::new());
        h.panel_open = true;
        let ax = panel::close_rect(h.panel_origin(vp));
        assert!(h.try_panel_press(DragTarget::Assets, ax[0] + 5.0, ax[1] + 5.0, vp, &mut world));
        assert!(
            !h.panel_open && h.drag.is_none(),
            "Assets X closed it, no drag"
        );

        // View.
        let mut h = hook(Vec::new());
        h.view_open = true;
        let vx = view::close_rect(h.view_origin(vp));
        assert!(h.try_panel_press(DragTarget::View, vx[0] + 5.0, vx[1] + 5.0, vp, &mut world));
        assert!(
            !h.view_open && h.drag.is_none(),
            "View X closed it, no drag"
        );

        // Templates.
        let mut h = hook(Vec::new());
        h.templates_open = true;
        let tx = templates::close_rect(h.templates_origin(vp));
        assert!(h.try_panel_press(
            DragTarget::Templates,
            tx[0] + 5.0,
            tx[1] + 5.0,
            vp,
            &mut world
        ));
        assert!(
            !h.templates_open && h.drag.is_none(),
            "Templates X closed it, no drag"
        );
    }

    // A panel toggled off (its View checkbox unticked) is not interactive: a press
    // where it would be falls through instead of being claimed.
    #[test]
    fn a_hidden_panel_is_not_interactive() {
        let mut h = hook(Vec::new());
        let vp = [1280.0, 720.0];
        let mut world = world_with_fields();
        // Preview starts shown: a title-bar press is claimed (starts a drag).
        let pt = preview::title_rect(h.preview_origin(vp));
        assert!(h.try_panel_press(
            DragTarget::Preview,
            pt[0] + 5.0,
            pt[1] + 5.0,
            vp,
            &mut world
        ));
        // Hidden: the same press falls through.
        h.drag = None;
        h.preview_open = false;
        assert!(!h.try_panel_press(
            DragTarget::Preview,
            pt[0] + 5.0,
            pt[1] + 5.0,
            vp,
            &mut world
        ));
        // The View panel starts hidden: its press falls through until it is opened.
        let vt = view::title_rect(h.view_origin(vp));
        assert!(!h.try_panel_press(DragTarget::View, vt[0] + 5.0, vt[1] + 5.0, vp, &mut world));
        h.view_open = true;
        assert!(h.try_panel_press(DragTarget::View, vt[0] + 5.0, vt[1] + 5.0, vp, &mut world));
    }

    // The Templates panel drags by its own title bar and comes to the front on a
    // press, like the other floating panels.
    #[test]
    fn templates_panel_press_drags_and_focuses() {
        let mut h = hook(Vec::new());
        h.templates_open = true;
        let vp = [1280.0, 720.0];
        let mut world = world_with_fields();
        let t = templates::title_rect(h.templates_origin(vp));
        assert!(h.try_panel_press(
            DragTarget::Templates,
            t[0] + 5.0,
            t[1] + 5.0,
            vp,
            &mut world
        ));
        assert!(h.drag.is_some(), "a title-bar press starts a drag");
        assert_eq!(h.panel_order.last().copied(), Some(DragTarget::Templates));
    }

    // End-to-end through `tick` against a fully injected HUD: the top-bar View
    // button opens the View panel, and clicking its "Templates" row opens the
    // Templates panel (the same click path a real session drives).
    #[test]
    fn tick_view_button_opens_view_then_a_row_opens_templates() {
        let vis = |w: &World, id: crate::ecs::asset_id::AssetId| {
            w.query::<Sprite>()
                .find(|s| s.asset_id == id)
                .map(|s| s.visible)
                .unwrap_or(false)
        };
        let rect = |w: &World, id: crate::ecs::asset_id::AssetId| {
            let s = w.query::<Sprite>().find(|s| s.asset_id == id).unwrap();
            [s.x, s.y, s.width, s.height]
        };
        let mut world = World::new_empty();
        super::super::inject::editor_hud(&mut world);
        let vp = [1280.0, 720.0];
        let mut h = hook(Vec::new());

        // Frame 1: no interaction. View + Templates start hidden.
        world.add_component(FrameInput {
            viewport: vp,
            ..Default::default()
        });
        h.tick(&mut world);
        assert!(!vis(&world, view::TITLE_BG) && !vis(&world, templates::TITLE_BG));

        // Frame 2: click the top-bar View button -> the View panel opens.
        let (_, view_btn) = hud::layout(vp[0]);
        set_input(
            &mut world,
            FrameInput {
                viewport: vp,
                mouse_x: view_btn[0] + view_btn[2] * 0.5,
                mouse_y: view_btn[1] + view_btn[3] * 0.5,
                left_click: true,
                left_button_down: true,
                ..Default::default()
            },
        );
        h.tick(&mut world);
        assert!(
            h.view_open && vis(&world, view::TITLE_BG),
            "View panel opened"
        );
        // Its "Templates" row (index 2) is laid out; grab its rect to click it.
        let row = rect(&world, view::row_bg(2));

        // Frame 3: click that row -> the Templates panel opens.
        set_input(
            &mut world,
            FrameInput {
                viewport: vp,
                mouse_x: row[0] + row[2] * 0.5,
                mouse_y: row[1] + row[3] * 0.5,
                left_click: true,
                left_button_down: true,
                ..Default::default()
            },
        );
        h.tick(&mut world);
        assert!(h.templates_open, "the Templates row toggled the panel on");
        assert!(vis(&world, templates::TITLE_BG), "Templates panel shown");
    }

    // Picking a template row spawns the detail panel (title "Template <name>",
    // hidden until then); its Apply button layers the template's assets and closes
    // the detail. Drives the whole flow through `tick` end to end.
    #[test]
    fn tick_picking_a_template_spawns_the_detail_panel_then_apply_adds() {
        let vis = |w: &World, id: crate::ecs::asset_id::AssetId| {
            w.query::<Sprite>()
                .find(|s| s.asset_id == id)
                .map(|s| s.visible)
                .unwrap_or(false)
        };
        let rect = |w: &World, id: crate::ecs::asset_id::AssetId| {
            let s = w.query::<Sprite>().find(|s| s.asset_id == id).unwrap();
            [s.x, s.y, s.width, s.height]
        };
        let mut world = World::new_empty();
        super::super::inject::editor_hud(&mut world);
        let vp = [1280.0, 720.0];
        let mut h = hook(Vec::new());
        // Start with the Templates list already open.
        h.templates_open = true;
        world.add_component(FrameInput {
            viewport: vp,
            ..Default::default()
        });
        h.tick(&mut world);
        assert!(
            vis(&world, templates::TITLE_BG) && !vis(&world, template_panel::PANEL_BG),
            "Templates list shown; detail panel still hidden"
        );

        // Click the first template row -> the detail panel spawns.
        let row = rect(&world, templates::row_bg(0));
        set_input(
            &mut world,
            FrameInput {
                viewport: vp,
                mouse_x: row[0] + row[2] * 0.5,
                mouse_y: row[1] + row[3] * 0.5,
                left_click: true,
                left_button_down: true,
                ..Default::default()
            },
        );
        h.tick(&mut world);
        assert_eq!(h.open_template, Some(0), "the detail panel opened on pick");
        assert!(vis(&world, template_panel::PANEL_BG), "detail panel shown");
        let title = world
            .query::<crate::assets::TextLabel>()
            .find(|l| l.asset_id == template_panel::TITLE_LABEL)
            .unwrap();
        assert!(
            title.content.starts_with("Template "),
            "title bar reads 'Template <name>': {}",
            title.content
        );
        assert!(h.entries.is_empty(), "picking adds nothing yet");

        // Click the detail's Apply button -> the template's assets are added and
        // the detail closes.
        let apply = template_panel::apply_rect(h.template_detail_origin(0, vp));
        set_input(
            &mut world,
            FrameInput {
                viewport: vp,
                mouse_x: apply[0] + apply[2] * 0.5,
                mouse_y: apply[1] + apply[3] * 0.5,
                left_click: true,
                left_button_down: true,
                ..Default::default()
            },
        );
        h.tick(&mut world);
        assert_eq!(h.open_template, None, "Apply closed the detail panel");
        assert!(
            !vis(&world, template_panel::PANEL_BG),
            "detail panel hidden"
        );
        assert_eq!(
            h.entries.len(),
            concinnity_templates::TEMPLATES[0].assets().len(),
            "Apply layered the template's assets"
        );
    }

    #[test]
    fn add_form_writes_edited_arg_values() {
        let mut h = hook(Vec::new());
        let mut world = world_with_fields();
        h.panel_open = true;
        h.apply_panel(PanelAction::TogglePicker, &mut world);
        // Pick a type with a float arg through the real picker->pick path.
        let ty = "PointLight".to_string();
        let idx = h
            .combo_options(&world)
            .iter()
            .position(|o| o == &ty)
            .expect("PointLight is offered");
        h.apply_panel(PanelAction::PickOption(idx), &mut world);
        assert!(h.form_open());
        assert!(!h.form_fields.is_empty(), "the type exposes arg fields");
        // Edit a float field via its input.
        let (j, key) = h
            .form_fields
            .iter()
            .enumerate()
            .find(|(_, f)| matches!(f.kind, form::FieldKind::Float))
            .map(|(j, f)| (j, f.key.clone()))
            .expect("a float arg field");
        set_field(&mut world, form_panel::form_input(j), "3.5");
        set_field(&mut world, form_panel::NAME_INPUT, "lamp");
        h.apply_form(FormAction::Confirm, &mut world);
        assert!(!h.form_open());
        assert_eq!(h.entries.len(), 1);
        assert_eq!(h.entries[0]["name"], "lamp");
        assert_eq!(h.entries[0]["type"], ty.as_str());
        assert_eq!(
            h.entries[0]["args"][&key].as_f64(),
            Some(3.5),
            "the edited float persisted into args"
        );
    }

    #[test]
    fn add_form_writes_an_edited_colour_vector() {
        let mut h = hook(Vec::new());
        let mut world = world_with_fields();
        // VolumetricFog (a newly offered type) has a `color` RGB vector field.
        h.open_form(&mut world, "VolumetricFog".to_string(), None);
        let (j, key) = h
            .form_fields
            .iter()
            .enumerate()
            .find(|(_, f)| matches!(f.kind, form::FieldKind::Vec { color: true, .. }))
            .map(|(j, f)| (j, f.key.clone()))
            .expect("a colour vector field");
        set_field(&mut world, form_panel::form_input(j), "0.1, 0.2, 0.3");
        set_field(&mut world, form_panel::NAME_INPUT, "fog");
        h.apply_form(FormAction::Confirm, &mut world);
        assert!(!h.form_open());
        assert_eq!(h.entries.len(), 1);
        assert_eq!(h.entries[0]["type"], "VolumetricFog");
        assert_eq!(
            h.entries[0]["args"][&key],
            serde_json::json!([0.1, 0.2, 0.3]),
            "the edited colour persisted as a numeric array"
        );
    }

    // Editing a nested (dotted-path) field through the form persists into the
    // sub-object: Camera3D's `controller.move_speed`.
    #[test]
    fn add_form_writes_a_nested_object_field() {
        let mut h = hook(Vec::new());
        let mut world = world_with_fields();
        h.open_form(&mut world, "Camera3D".to_string(), None);
        let j = h
            .form_fields
            .iter()
            .position(|f| f.key == "controller.move_speed")
            .expect("the nested controller.move_speed field is offered");
        assert!(matches!(h.form_fields[j].kind, form::FieldKind::Float));
        set_field(&mut world, form_panel::form_input(j), "12.5");
        set_field(&mut world, form_panel::NAME_INPUT, "cam");
        h.apply_form(FormAction::Confirm, &mut world);
        let cam = h
            .entries
            .iter()
            .find(|e| e["name"] == "cam")
            .expect("the camera was added");
        assert_eq!(cam["type"], "Camera3D");
        assert_eq!(
            cam["args"]["controller"]["move_speed"].as_f64(),
            Some(12.5),
            "the nested edit persisted into args.controller.move_speed"
        );
    }

    #[test]
    fn add_form_writes_string_fields_for_a_new_type() {
        let mut h = hook(Vec::new());
        let mut world = world_with_fields();
        // KeyBinding (a newly offered type) is a pair of string fields.
        h.open_form(&mut world, "KeyBinding".to_string(), None);
        let field_pos = |k: &str| {
            h.form_fields
                .iter()
                .position(|f| f.key == k)
                .unwrap_or_else(|| panic!("{k} field present"))
        };
        let (key_j, action_j) = (field_pos("key"), field_pos("action"));
        assert!(matches!(h.form_fields[key_j].kind, form::FieldKind::Str));
        set_field(&mut world, form_panel::form_input(key_j), "Space");
        set_field(&mut world, form_panel::form_input(action_j), "jump");
        set_field(&mut world, form_panel::NAME_INPUT, "jump_key");
        h.apply_form(FormAction::Confirm, &mut world);
        assert!(!h.form_open());
        assert_eq!(h.entries.len(), 1);
        assert_eq!(h.entries[0]["type"], "KeyBinding");
        assert_eq!(h.entries[0]["args"]["key"], "Space");
        assert_eq!(h.entries[0]["args"]["action"], "jump");
    }

    #[test]
    fn add_form_cycles_and_persists_an_enum_field() {
        let mut h = hook(Vec::new());
        let mut world = world_with_fields();
        // Sprite's `fit` is a string enum -> a cycling picker.
        h.open_form(&mut world, "Sprite".to_string(), None);
        let idx = h
            .form_fields
            .iter()
            .position(|f| f.key == "fit")
            .expect("fit enum field");
        assert!(matches!(h.form_fields[idx].kind, form::FieldKind::Enum));
        let n = h.form_fields[idx].variants.len();
        let start = h.form_fields[idx].variant_idx;
        // Cycle once, then confirm.
        h.apply_form(FormAction::CycleField(idx), &mut world);
        let picked = h.form_fields[idx].variants[(start + 1) % n].clone();
        assert_ne!(
            picked, h.form_fields[idx].variants[start],
            "cycled to a new value"
        );
        set_field(&mut world, form_panel::NAME_INPUT, "spr");
        h.apply_form(FormAction::Confirm, &mut world);
        assert_eq!(h.entries.len(), 1);
        assert_eq!(h.entries[0]["type"], "Sprite");
        assert_eq!(
            h.entries[0]["args"]["fit"], picked,
            "the cycled enum variant persisted into args"
        );
    }

    #[test]
    fn add_form_ref_field_offers_and_persists_an_existing_asset() {
        let mut h = hook(vec![
            entry("grass_tex", "Texture"),
            entry("stone_tex", "Texture"),
        ]);
        let mut world = world_with_fields();
        h.panel_open = true;
        // Add a Decal: its `texture` reference offers the two existing Textures.
        h.open_form(&mut world, "Decal".to_string(), None);
        let idx = h
            .form_fields
            .iter()
            .position(|f| f.key == "texture")
            .expect("texture ref field");
        assert!(
            matches!(h.form_fields[idx].kind, form::FieldKind::Ref { target } if target == "Texture")
        );
        assert_eq!(
            h.form_fields[idx].variants,
            vec![form::NONE_LABEL, "grass_tex", "stone_tex"],
            "options are (none) + the world's Textures"
        );
        assert_eq!(h.form_fields[idx].variant_idx, 0, "starts at (none)");
        // Cycle to the first Texture and confirm.
        h.apply_form(FormAction::CycleField(idx), &mut world);
        assert_eq!(
            h.form_fields[idx].variants[h.form_fields[idx].variant_idx],
            "grass_tex"
        );
        set_field(&mut world, form_panel::NAME_INPUT, "splat");
        h.apply_form(FormAction::Confirm, &mut world);
        let decal = h
            .entries
            .iter()
            .find(|e| e["name"] == "splat")
            .expect("the decal was added");
        assert_eq!(decal["type"], "Decal");
        assert_eq!(
            decal["args"]["texture"], "grass_tex",
            "the reference persisted as the asset's name"
        );
    }

    // A ref field with many candidate assets opens a value dropdown (not a cycle):
    // the dropdown picks an option, which persists as that asset's name.
    #[test]
    fn add_form_ref_field_dropdown_picks_and_persists() {
        // More Textures than the cycle cap, so the picker is a dropdown.
        let mut entries = Vec::new();
        for i in 0..(form_panel::CYCLE_MAX + 3) {
            entries.push(entry(&format!("tex_{i}"), "Texture"));
        }
        let mut h = hook(entries);
        let mut world = world_with_fields();
        h.panel_open = true;
        h.open_form(&mut world, "Decal".to_string(), None);
        let idx = h
            .form_fields
            .iter()
            .position(|f| f.key == "texture")
            .expect("texture ref field");
        // (none) + the textures exceeds CYCLE_MAX, so a click opens a dropdown.
        assert!(h.form_fields[idx].variants.len() > form_panel::CYCLE_MAX);
        h.apply_form(FormAction::OpenFieldDropdown(idx), &mut world);
        assert_eq!(h.field_dropdown, Some(idx), "the dropdown opened");
        // Pick option 3 (a real texture, past (none) at 0).
        let picked = h.form_fields[idx].variants[3].clone();
        h.apply_form(FormAction::PickFieldOption(3), &mut world);
        assert!(h.field_dropdown.is_none(), "picking closes the dropdown");
        assert_eq!(h.form_fields[idx].variant_idx, 3, "the option was selected");
        set_field(&mut world, form_panel::NAME_INPUT, "splat");
        h.apply_form(FormAction::Confirm, &mut world);
        let decal = h.entries.iter().find(|e| e["name"] == "splat").unwrap();
        assert_eq!(
            decal["args"]["texture"], picked,
            "the dropdown-picked reference persisted as the asset's name"
        );
    }

    // A second click on an open dropdown's field toggles it closed; CloseOverlays
    // also dismisses it.
    #[test]
    fn field_dropdown_toggles_and_close_overlays_dismisses_it() {
        let mut h = hook(Vec::new());
        let mut world = world_with_fields();
        h.selected_type = Some("Decal".to_string());
        h.apply_form(FormAction::OpenFieldDropdown(0), &mut world);
        assert_eq!(h.field_dropdown, Some(0));
        // Same field again -> closed.
        h.apply_form(FormAction::OpenFieldDropdown(0), &mut world);
        assert!(h.field_dropdown.is_none(), "a second click closes it");
        // Reopen, then the form's CloseOverlays dismisses it.
        h.apply_form(FormAction::OpenFieldDropdown(0), &mut world);
        h.apply_form(FormAction::CloseOverlays, &mut world);
        assert!(h.field_dropdown.is_none(), "CloseOverlays dismisses it");
    }

    // Wheeling scrolls an open value dropdown (which can extend past the fixed
    // panel body), independent of the cursor-over-body gate.
    #[test]
    fn scrolling_advances_an_open_field_dropdown() {
        let mut entries = Vec::new();
        for i in 0..(form_panel::MAX_DROP_ROWS + 4) {
            entries.push(entry(&format!("tex_{i}"), "Texture"));
        }
        let mut h = hook(entries);
        let mut world = world_with_fields();
        h.panel_open = true;
        h.open_form(&mut world, "Decal".to_string(), None);
        let idx = h
            .form_fields
            .iter()
            .position(|f| f.key == "texture")
            .expect("texture ref field");
        h.apply_form(FormAction::OpenFieldDropdown(idx), &mut world);
        assert_eq!(h.field_dropdown_scroll, 0);
        h.scroll(1.0, ScrollTarget::Form, &mut world);
        assert_eq!(
            h.field_dropdown_scroll, 1,
            "wheel down advances the dropdown"
        );
        h.scroll(-1.0, ScrollTarget::Form, &mut world);
        assert_eq!(h.field_dropdown_scroll, 0, "wheel up rewinds it");
        // It cannot scroll past the last page.
        for _ in 0..50 {
            h.scroll(1.0, ScrollTarget::Form, &mut world);
        }
        let total = h.form_fields[idx].variants.len();
        assert_eq!(
            h.field_dropdown_scroll,
            total - form_panel::MAX_DROP_ROWS,
            "scroll clamps to the last full page"
        );
    }

    // Growing an array through the form's [+] and editing the new element persists:
    // WaterSurface starts with one wave; add a second and set its amplitude.
    #[test]
    fn add_form_grows_an_array_and_edits_the_new_element() {
        let mut h = hook(Vec::new());
        let mut world = world_with_fields();
        h.open_form(&mut world, "WaterSurface".to_string(), None);
        let header = |h: &EditorHook| {
            h.form_fields
                .iter()
                .position(|f| f.key == "waves")
                .expect("waves array header")
        };
        let hj = header(&h);
        assert!(matches!(h.form_fields[hj].kind, form::FieldKind::Array));
        assert_eq!(h.form_fields[hj].variant_idx, 1, "one default wave");
        // [+] grows the array to two waves (fields re-derive).
        h.apply_form(FormAction::AddArrayElement(hj), &mut world);
        assert_eq!(
            h.form_fields[header(&h)].variant_idx,
            2,
            "grew to two waves"
        );
        // Edit the second wave's amplitude, then confirm.
        let ej = h
            .form_fields
            .iter()
            .position(|f| f.key == "waves.1.amplitude")
            .expect("the second wave's amplitude field");
        set_field(&mut world, form_panel::form_input(ej), "4.5");
        set_field(&mut world, form_panel::NAME_INPUT, "sea");
        h.apply_form(FormAction::Confirm, &mut world);
        let ws = h
            .entries
            .iter()
            .find(|e| e["name"] == "sea")
            .expect("the water surface was added");
        assert_eq!(ws["type"], "WaterSurface");
        assert_eq!(
            ws["args"]["waves"].as_array().map(Vec::len),
            Some(2),
            "the grown array persisted with two waves"
        );
        assert_eq!(
            ws["args"]["waves"][1]["amplitude"].as_f64(),
            Some(4.5),
            "the edited new-element value persisted"
        );
    }

    // Removing an array element through the form's [-] shrinks it and persists.
    #[test]
    fn add_form_removes_an_array_element() {
        let mut h = hook(Vec::new());
        let mut world = world_with_fields();
        h.open_form(&mut world, "WaterSurface".to_string(), None);
        let hj = h.form_fields.iter().position(|f| f.key == "waves").unwrap();
        // Grow to two, then remove one back to one.
        h.apply_form(FormAction::AddArrayElement(hj), &mut world);
        let hj = h.form_fields.iter().position(|f| f.key == "waves").unwrap();
        assert_eq!(h.form_fields[hj].variant_idx, 2);
        h.apply_form(FormAction::RemoveArrayElement(hj), &mut world);
        let hj = h.form_fields.iter().position(|f| f.key == "waves").unwrap();
        assert_eq!(h.form_fields[hj].variant_idx, 1, "shrank back to one wave");
        set_field(&mut world, form_panel::NAME_INPUT, "pond");
        h.apply_form(FormAction::Confirm, &mut world);
        let ws = h.entries.iter().find(|e| e["name"] == "pond").unwrap();
        assert_eq!(ws["args"]["waves"].as_array().map(Vec::len), Some(1));
    }

    // A plain vector opens collapsed; disclosing it exposes per-element leaves whose
    // edits write back into the vector (keeping its length) and persist.
    #[test]
    fn form_discloses_a_vector_and_edits_one_element() {
        let mut h = hook(Vec::new());
        let mut world = world_with_fields();
        h.open_form(&mut world, "PointLight".to_string(), None);
        let pos = |h: &EditorHook| {
            h.form_fields
                .iter()
                .position(|f| f.key == "position")
                .expect("a position vector field")
        };
        // Collapsed: no element leaves yet.
        assert!(
            h.form_fields
                .iter()
                .all(|f| !f.key.starts_with("position."))
        );
        // Disclose it: the element leaves appear and the path is tracked expanded.
        h.apply_form(FormAction::ToggleVecExpand(pos(&h)), &mut world);
        assert!(h.vec_expanded.contains("position"));
        let yj = h
            .form_fields
            .iter()
            .position(|f| f.key == "position.1")
            .expect("the y element leaf");
        // Edit y through its control, then confirm.
        let slot = visible_slot(yj, h.form_scroll).expect("y leaf visible");
        set_field(&mut world, form_panel::form_input(slot), "4.5");
        set_field(&mut world, form_panel::NAME_INPUT, "lamp");
        h.apply_form(FormAction::Confirm, &mut world);
        let lamp = h.entries.iter().find(|e| e["name"] == "lamp").unwrap();
        assert_eq!(
            lamp["args"]["position"].as_array().map(Vec::len),
            Some(3),
            "the vector kept its length"
        );
        assert_eq!(lamp["args"]["position"][1].as_f64(), Some(4.5));
    }

    // Collapsing a disclosed vector after editing an element keeps the edit (capture
    // runs before the field list re-derives).
    #[test]
    fn collapsing_a_vector_keeps_its_element_edits() {
        let mut h = hook(Vec::new());
        let mut world = world_with_fields();
        h.open_form(&mut world, "PointLight".to_string(), None);
        let pj = h
            .form_fields
            .iter()
            .position(|f| f.key == "position")
            .unwrap();
        h.apply_form(FormAction::ToggleVecExpand(pj), &mut world);
        let xj = h
            .form_fields
            .iter()
            .position(|f| f.key == "position.0")
            .unwrap();
        let slot = visible_slot(xj, h.form_scroll).unwrap();
        set_field(&mut world, form_panel::form_input(slot), "2.0");
        // Collapse again: the element leaves go away but the edit is folded in.
        let pj = h
            .form_fields
            .iter()
            .position(|f| f.key == "position")
            .unwrap();
        h.apply_form(FormAction::ToggleVecExpand(pj), &mut world);
        assert!(!h.vec_expanded.contains("position"));
        assert!(
            h.form_fields
                .iter()
                .all(|f| !f.key.starts_with("position."))
        );
        set_field(&mut world, form_panel::NAME_INPUT, "lamp");
        h.apply_form(FormAction::Confirm, &mut world);
        let lamp = h.entries.iter().find(|e| e["name"] == "lamp").unwrap();
        assert_eq!(lamp["args"]["position"][0].as_f64(), Some(2.0));
    }

    // A form wider than the control pool scrolls: a field past the window is edited
    // by wheeling down to it. WaterSurface exposes more than a pool's worth of
    // fields, so `roughness` is only reachable after scrolling; its edit must still
    // persist (and the untouched off-window fields keep their defaults).
    #[test]
    fn add_form_scrolls_to_and_edits_an_off_window_field() {
        let mut h = hook(Vec::new());
        let mut world = world_with_fields();
        h.open_form(&mut world, "WaterSurface".to_string(), None);
        assert!(
            h.form_fields.len() > form::FIELD_POOL,
            "WaterSurface overflows the control pool"
        );
        let rj = h
            .form_fields
            .iter()
            .position(|f| f.key == "roughness")
            .expect("a roughness field");
        assert!(
            visible_slot(rj, h.form_scroll).is_none(),
            "roughness starts past the visible window"
        );
        // Wheel to the bottom; roughness scrolls into the window.
        for _ in 0..h.form_fields.len() {
            h.scroll(1.0, ScrollTarget::Form, &mut world);
        }
        let slot = visible_slot(rj, h.form_scroll).expect("roughness scrolled into view");
        // Edit it through its now-visible control and confirm.
        set_field(&mut world, form_panel::form_input(slot), "0.9");
        set_field(&mut world, form_panel::NAME_INPUT, "sea");
        h.apply_form(FormAction::Confirm, &mut world);
        let ws = h
            .entries
            .iter()
            .find(|e| e["name"] == "sea")
            .expect("the water surface was added");
        assert_eq!(
            ws["args"]["roughness"].as_f64(),
            Some(0.9),
            "the off-window field's edit persisted after scrolling to it"
        );
        // An untouched off-window top field kept its default (not blanked on capture).
        assert_eq!(
            ws["args"]["extent"],
            form::base_args("WaterSurface")["extent"],
            "a scrolled-away field kept its value"
        );
    }

    // A reference left at (none) persists as null, not a dangling name.
    #[test]
    fn add_form_ref_field_defaults_to_none() {
        let mut h = hook(vec![entry("grass_tex", "Texture")]);
        let mut world = world_with_fields();
        h.open_form(&mut world, "Decal".to_string(), None);
        set_field(&mut world, form_panel::NAME_INPUT, "bare");
        h.apply_form(FormAction::Confirm, &mut world);
        let decal = h.entries.iter().find(|e| e["name"] == "bare").unwrap();
        assert_eq!(decal["args"]["texture"], serde_json::Value::Null);
    }

    #[test]
    fn invalid_arg_keeps_the_form_open_with_an_error() {
        let mut h = hook(Vec::new());
        let mut world = world_with_fields();
        // Font has a u32 `size_px` field; a negative value cannot re-serialize.
        h.open_form(&mut world, "Font".to_string(), None);
        let j = h
            .form_fields
            .iter()
            .position(|f| f.key == "size_px")
            .expect("size_px field present");
        assert!(matches!(h.form_fields[j].kind, form::FieldKind::Int));
        set_field(&mut world, form_panel::form_input(j), "-5");
        set_field(&mut world, form_panel::NAME_INPUT, "myfont");
        h.apply_form(FormAction::Confirm, &mut world);
        assert!(h.form_open(), "the form stays open on invalid input");
        assert!(h.form_error.is_some(), "an error message is shown");
        assert!(h.entries.is_empty(), "nothing invalid was committed");
    }

    // Toggling the Assets panel off then on (via the View panel) keeps the open
    // form + its browse selection (the state is retained, only hidden), so the same
    // view returns.
    #[test]
    fn toggling_the_assets_panel_keeps_the_open_form_state() {
        let mut h = hook(vec![entry("lamp", "PointLight")]);
        let mut world = world_with_fields();
        h.panel_open = true;
        h.open_form(&mut world, "PointLight".to_string(), Some(0));
        assert!(h.form_open() && h.editing == Some(0));
        // Toggle the assets UI off: the form + selection are kept, not discarded.
        h.apply_view(ViewAction::Toggle(0));
        assert!(!h.panel_open);
        assert!(
            h.form_open(),
            "the form is kept when the panel is toggled off"
        );
        assert_eq!(h.editing, Some(0), "the browse selection is kept");
        // Toggle back on: the same form and selection are restored.
        h.apply_view(ViewAction::Toggle(0));
        assert!(h.panel_open && h.form_open());
        assert_eq!(h.editing, Some(0));
    }

    // Hiding the assets UI hides the form's elements (but keeps its state); showing
    // it again re-renders the form.
    #[test]
    fn a_hidden_assets_panel_hides_the_form_elements() {
        let mut world = World::new_empty();
        super::super::inject::editor_hud(&mut world);
        world.add_component(FrameInput {
            viewport: [1280.0, 720.0],
            ..Default::default()
        });
        let mut h = hook(vec![entry("lamp", "PointLight")]);
        h.panel_open = true;
        h.open_form(&mut world, "PointLight".to_string(), Some(0));
        let form_shown = |w: &World| {
            w.query::<Sprite>()
                .find(|s| s.asset_id == form_panel::EDIT_BG)
                .unwrap()
                .visible
        };
        h.tick(&mut world);
        assert!(form_shown(&world), "form shown while the panel is open");
        // Toggle off: the form elements hide, but its state is retained.
        h.apply_view(ViewAction::Toggle(0));
        h.tick(&mut world);
        assert!(!form_shown(&world), "form elements hidden when toggled off");
        assert!(h.form_open(), "but the form state is retained");
        // Toggle on: the form re-renders.
        h.apply_view(ViewAction::Toggle(0));
        h.tick(&mut world);
        assert!(form_shown(&world), "form shown again on toggle-on");
    }

    #[test]
    fn edit_form_seeds_and_updates_existing_args() {
        let mut h = hook(vec![serde_json::json!({
            "name": "lamp", "type": "PointLight", "args": {}
        })]);
        let mut world = world_with_fields();
        h.panel_open = true;
        h.apply_panel(PanelAction::OpenEntry(0), &mut world);
        assert_eq!(h.editing, Some(0));
        assert!(!h.form_fields.is_empty());
        // The name field was seeded from the entry.
        assert_eq!(widget::field_text(&world, form_panel::NAME_INPUT), "lamp");
        // Edit a float and confirm; the same entry gains a full args object.
        let (j, key) = h
            .form_fields
            .iter()
            .enumerate()
            .find(|(_, f)| matches!(f.kind, form::FieldKind::Float))
            .map(|(j, f)| (j, f.key.clone()))
            .expect("a float arg field");
        set_field(&mut world, form_panel::form_input(j), "9.0");
        h.apply_form(FormAction::Confirm, &mut world);
        assert_eq!(h.entries.len(), 1, "edited in place");
        assert_eq!(h.entries[0]["args"][&key].as_f64(), Some(9.0));
    }

    // Drive `tick` against a fully injected HUD world in each panel body state,
    // exercising the real `panel::apply` layout path (not just the pure hit-test /
    // action logic the other tests cover).
    #[test]
    fn tick_lays_out_the_open_panel_in_every_state() {
        let sprite_visible = |w: &World, id: crate::ecs::asset_id::AssetId| {
            w.query::<Sprite>()
                .find(|s| s.asset_id == id)
                .unwrap()
                .visible
        };
        let label = |w: &World, id: crate::ecs::asset_id::AssetId| {
            w.query::<TextLabel>()
                .find(|l| l.asset_id == id)
                .unwrap()
                .clone()
        };

        let mut world = World::new_empty();
        super::super::inject::editor_hud(&mut world);
        world.add_component(FrameInput {
            viewport: [1280.0, 720.0],
            mouse_x: 1200.0,
            mouse_y: 300.0,
            ..Default::default()
        });
        let mut h = hook(vec![entry("a", "PointLight"), entry("b", "Decal")]);
        h.panel_open = true;

        // Grouped list: panel drawn, first row shows a type sub-header.
        h.tick(&mut world);
        assert!(sprite_visible(&world, panel::PANEL_BG), "panel bg shown");
        let row0 = label(&world, panel::list_row_label(0));
        assert!(
            row0.visible && row0.content == "Decal",
            "first row is a header"
        );

        // Picker combo: the solid backing and the filter field show.
        h.combo = Combo::Picker;
        h.tick(&mut world);
        assert!(
            sprite_visible(&world, panel::COMBO_BG),
            "combo backing shown"
        );
        assert!(
            world
                .query::<TextInput>()
                .find(|t| t.asset_id == panel::FILTER_INPUT)
                .unwrap()
                .visible
        );

        // Row menu: the Delete popup shows over the "a" name row (entry 0).
        h.combo = Combo::Closed;
        h.row_menu = Some(0);
        h.tick(&mut world);
        assert!(sprite_visible(&world, panel::MENU_BG), "row menu shown");
        assert_eq!(label(&world, panel::MENU_DELETE_LABEL).content, "Delete");

        // Form open: the edit panel shows alongside the browse list, with its
        // title bar, name heading, and confirm button.
        h.row_menu = None;
        h.open_form(&mut world, "PointLight".to_string(), None);
        h.tick(&mut world);
        assert!(
            sprite_visible(&world, form_panel::APPLY_BG),
            "confirm button shown"
        );
        assert_eq!(
            label(&world, form_panel::TITLE_LABEL).content,
            "New PointLight"
        );
        assert_eq!(label(&world, form_panel::APPLY_LABEL).content, "Add");
        assert!(
            world
                .query::<TextInput>()
                .find(|t| t.asset_id == form_panel::NAME_INPUT)
                .unwrap()
                .visible,
            "the name heading shows"
        );
        assert!(
            label(&world, panel::list_row_label(0)).visible,
            "the browse list stays visible beside the form"
        );

        // Closing the panel + form blanks both.
        h.panel_open = false;
        h.close_form();
        h.tick(&mut world);
        assert!(!sprite_visible(&world, panel::PANEL_BG), "panel bg hidden");
        assert!(
            !sprite_visible(&world, form_panel::EDIT_BG),
            "form panel hidden"
        );
    }

    #[test]
    fn write_jsonl_persists_entries_atomically() {
        let path = std::env::temp_dir().join("cn_editor_write_jsonl_test.jsonl");
        let path_str = path.to_str().unwrap().to_string();
        let _ = std::fs::remove_file(&path);

        let mut h = hook(vec![serde_json::json!({
            "name": "scene", "type": "GraphicsConfig", "args": {}
        })]);
        h.world_path = path_str.clone();
        let mut world = world_with_fields();
        h.selected_type = Some("PointLight".to_string());
        set_field(&mut world, form_panel::NAME_INPUT, "lamp");
        h.apply_form(FormAction::Confirm, &mut world);
        h.write_jsonl().unwrap();

        let content = std::fs::read_to_string(&path).unwrap();
        let parsed = concinnity_core::world::parse_world_jsonl(&content).unwrap();
        assert_eq!(parsed.len(), 2, "both entries written, one line each");
        assert_eq!(parsed[1]["name"], "lamp");
        assert!(!std::path::Path::new(&format!("{path_str}.tmp")).exists());

        let _ = std::fs::remove_file(&path);
    }
}
