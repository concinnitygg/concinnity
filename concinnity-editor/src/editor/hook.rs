// src/editor/hook.rs
//
// The editor's per-frame drive. Implements the run loop's `DebugHook` seam: each
// frame it hit-tests the editor HUD's controls against the live input, mutates
// the working authored entry list, persists on SAVE, drives the world's cursor /
// freeze state, and re-anchors + recolours the HUD. This is the whole editor: it
// lives in the editor crate (never linked by the shipped runtime), so no editor
// code is compiled into a shipped game.
//
// The top bar (`hud.rs`) owns SAVE, the Templates dropdown, and the capture
// checkbox. The Assets button opens the browse-and-add panel (`panel.rs`): a
// combo (dropdown) that filters the browse list by type, a "+" that opens a typed
// autocomplete of the addable types, a browse list grouped by type with a
// per-name Edit / Delete menu, and a name-first add / edit form. The combo's
// single filter field and the form's name field are real `TextInput` assets
// edited by the engine's text-input system; the hook reads them back.
//
// Cursor control: the editor holds the cursor by default (edit mode -- cursor
// free, world frozen), publishing `MenuOverride(Some(true))`. Ticking the
// capture checkbox hands the cursor to the world (`Some(false)`); Escape takes
// it back. F1 hides / shows the whole HUD.
//
// SAVE re-serializes the authored entry list to world.jsonl and recompiles the
// blobs through the validated cook tail (`build_world_to_disk`). A successful
// SAVE then applies the edit to the running world without recreating the window:
// the live render backend is transplanted out of the world and `apply_world_swap`
// rebuilds the recompiled world onto it (carried in as a `PendingBackend`).

use super::form::{self, FormField};
use super::hud::{self, HudAction, HudState};
use super::panel::{self, Combo, FormFocus, ListRow, PanelAction, PanelView};
use crate::app::state::App;
use crate::assets::FrameInput;
use crate::debug_hook::DebugHook;
use crate::ecs::{MenuOverride, PendingBackend, World};

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
    // Set by a successful SAVE (in `tick`), consumed by `apply_world_swap`.
    swap_requested: bool,
    // Whether the top-bar Templates dropdown is open.
    templates_open: bool,
    // Assets panel state.
    panel_open: bool,
    mode: panel::Mode,
    // The header combo (dropdown) state: closed, filtering, or picking a type.
    combo: Combo,
    // Active type filter for the browse list, or `None` for "all".
    type_filter: Option<String>,
    // First visible row of the (grouped) browse list / the combo option list.
    list_scroll: usize,
    combo_scroll: usize,
    // The type being named in the add / edit form.
    selected_type: Option<String>,
    // When the form is editing an existing entry, its `entries` index (else a new
    // asset is being added).
    editing: Option<usize>,
    // The editable arg fields of the open form (derived from the type's default
    // args). Empty outside AddForm.
    form_fields: Vec<FormField>,
    // Which form input has keyboard focus.
    form_focus: FormFocus,
    // A validation message from the last rejected Add, shown under the form.
    form_error: Option<String>,
    // The `entries` index whose Edit / Delete menu is open, if any.
    row_menu: Option<usize>,
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

// Move a scroll offset one row toward the wheel direction, clamped to `max`.
fn scroll_step(cur: usize, delta: f32, max: usize) -> usize {
    if delta > 0.0 {
        (cur + 1).min(max)
    } else {
        cur.saturating_sub(1)
    }
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

impl EditorHook {
    pub(crate) fn new(world_path: String, entries: Vec<serde_json::Value>) -> Self {
        Self {
            world_path,
            entries,
            dirty: false,
            world_capture: false,
            hud_visible: true,
            swap_requested: false,
            templates_open: false,
            panel_open: false,
            mode: panel::Mode::List,
            combo: Combo::Closed,
            type_filter: None,
            list_scroll: 0,
            combo_scroll: 0,
            selected_type: None,
            editing: None,
            form_fields: Vec::new(),
            form_focus: FormFocus::Name,
            form_error: None,
            row_menu: None,
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

    // Persist the working entries: write world.jsonl, then recompile the blobs.
    // On success the world is clean again and `true` is returned so the caller
    // triggers the live world swap; on failure it stays dirty and `false`.
    fn save(&mut self) -> bool {
        match self.persist() {
            Ok(()) => {
                self.dirty = false;
                tracing::info!("editor: saved {}", self.world_path);
                true
            }
            Err(e) => {
                tracing::error!("editor: save failed: {e}");
                false
            }
        }
    }

    fn persist(&self) -> std::io::Result<()> {
        self.write_jsonl()?;
        concinnity_app::build_world_to_disk(&self.world_path)?;
        Ok(())
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
        let entries = match concinnity_core::world::parse_world_jsonl(t.jsonl) {
            Ok(e) => e,
            Err(e) => {
                tracing::error!("editor: template '{}' failed to parse: {e}", t.name);
                return;
            }
        };
        let mut added = 0;
        for entry in entries {
            if entry_name(&entry).is_some_and(|n| self.name_taken(n)) {
                continue;
            }
            self.entries.push(entry);
            added += 1;
        }
        if added > 0 {
            self.dirty = true;
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

    // The browse list, grouped by type: a sub-header row per type (matching the
    // active filter), then an indented row per asset name carrying its entry
    // index (for the Edit / Delete menu). Insertion order within a type.
    fn list_rows(&self) -> Vec<ListRow> {
        let mut rows = Vec::new();
        for ty in self.distinct_types() {
            if let Some(f) = &self.type_filter
                && &ty != f
            {
                continue;
            }
            let names: Vec<(usize, String)> = self
                .entries
                .iter()
                .enumerate()
                .filter_map(|(i, e)| {
                    let name = entry_name(e)?;
                    (entry_type(e) == Some(ty.as_str())).then(|| (i, name.to_string()))
                })
                .collect();
            if names.is_empty() {
                continue;
            }
            rows.push(ListRow {
                is_header: true,
                text: ty.clone(),
                entry: None,
            });
            for (i, name) in names {
                rows.push(ListRow {
                    is_header: false,
                    text: name,
                    entry: Some(i),
                });
            }
        }
        rows
    }

    // The floating combo option list for the open flavour, narrowed by the typed
    // filter field. Empty when the combo is closed.
    fn combo_options(&self, world: &World) -> Vec<String> {
        let filter = panel::field_text(world, panel::FILTER_INPUT).to_lowercase();
        let matches = |s: &str| filter.is_empty() || s.to_lowercase().contains(&filter);
        match self.combo {
            Combo::Filter => {
                let mut all = vec![panel::ALL_LABEL.to_string()];
                all.extend(self.distinct_types());
                all.into_iter().filter(|o| matches(o)).collect()
            }
            Combo::Picker => {
                let mut opts: Vec<String> = panel::ADD_TYPES
                    .iter()
                    .copied()
                    .filter(|t| matches(t))
                    .map(|t| t.to_string())
                    .collect();
                // Pin the active browse filter to the top when it is offered.
                if let Some(active) = &self.type_filter
                    && let Some(pos) = opts.iter().position(|o| o == active)
                {
                    let pinned = opts.remove(pos);
                    opts.insert(0, pinned);
                }
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
            mode: self.mode,
            combo: self.combo,
            filter_label: &d.filter_label,
            combo_options: &d.combo_options,
            combo_selected: d.combo_selected,
            combo_scroll: self.combo_scroll,
            list_rows: &d.list_rows,
            list_scroll: self.list_scroll,
            row_menu: self.row_menu,
            form_title: &d.form_title,
            form_fields: &self.form_fields,
            form_focus: self.form_focus,
            form_error: self.form_error.as_deref(),
            mouse,
        }
    }

    // -- Action handling --------------------------------------------------------

    // Route a resolved top-bar click. Returns `true` only when a SAVE succeeded,
    // signalling the caller to transplant the backend and request a world swap.
    fn apply_top(&mut self, action: HudAction) -> bool {
        match action {
            HudAction::Save => return self.save(),
            HudAction::ToggleAssets => {
                self.panel_open = !self.panel_open;
                if self.panel_open {
                    self.templates_open = false;
                    self.mode = panel::Mode::List;
                    self.combo = Combo::Closed;
                    self.row_menu = None;
                    self.list_scroll = 0;
                }
            }
            HudAction::ToggleTemplates => {
                self.templates_open = !self.templates_open;
                if self.templates_open {
                    self.panel_open = false;
                }
            }
            HudAction::PickTemplate(i) => {
                self.apply_template(i);
                self.templates_open = false;
            }
            HudAction::ToggleCapture => self.world_capture = !self.world_capture,
            HudAction::CloseTemplates => self.templates_open = false,
        }
        false
    }

    // Open the combo in `flavour`, clearing and focusing the shared filter field.
    fn open_combo(&mut self, flavour: Combo, world: &mut World) {
        self.mode = panel::Mode::List;
        self.combo = flavour;
        self.combo_scroll = 0;
        self.row_menu = None;
        panel::focus_field_with(world, panel::FILTER_INPUT, "");
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
        self.form_fields = form::fields_for(&ty, seed.as_ref());
        self.form_focus = FormFocus::Name;
        self.form_error = None;
        self.selected_type = Some(ty);
        self.editing = editing;
        self.mode = panel::Mode::AddForm;
        self.combo = Combo::Closed;
        self.row_menu = None;
        panel::focus_field_with(world, panel::NAME_INPUT, &name);
        for (j, field) in self.form_fields.iter().enumerate() {
            if !matches!(field.kind, form::FieldKind::Bool) {
                panel::seed_field(world, panel::form_input(j), &field.initial);
            }
        }
    }

    // Leave the form, discarding its transient state, back to the browse list.
    fn close_form(&mut self) {
        self.selected_type = None;
        self.editing = None;
        self.form_fields.clear();
        self.form_focus = FormFocus::Name;
        self.form_error = None;
        self.mode = panel::Mode::List;
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
                        self.open_form(world, ty, None);
                    }
                }
                Combo::Closed => {}
            },
            PanelAction::OpenRowMenu(entry) => self.row_menu = Some(entry),
            PanelAction::RowEdit => {
                if let Some(idx) = self.row_menu
                    && let Some(ty) = self.entries.get(idx).and_then(entry_type).map(String::from)
                {
                    self.open_form(world, ty, Some(idx));
                }
                self.row_menu = None;
            }
            PanelAction::RowDelete => {
                if let Some(idx) = self.row_menu
                    && idx < self.entries.len()
                {
                    self.entries.remove(idx);
                    self.dirty = true;
                }
                self.row_menu = None;
                let max = self.list_rows().len().saturating_sub(panel::MAX_ROWS);
                self.list_scroll = self.list_scroll.min(max);
            }
            PanelAction::FocusName => self.form_focus = FormFocus::Name,
            PanelAction::FocusField(i) => self.form_focus = FormFocus::Field(i),
            PanelAction::ToggleField(i) => {
                if let Some(f) = self.form_fields.get_mut(i) {
                    f.boolval = !f.boolval;
                }
                self.form_error = None;
            }
            PanelAction::ConfirmAdd => self.confirm_form(world),
            PanelAction::CancelForm => self.close_form(),
            PanelAction::CloseOverlays => {
                self.combo = Combo::Closed;
                self.row_menu = None;
            }
            PanelAction::Consume => {}
        }
    }

    // Read the form's controls, assemble + validate the args, and commit (add a
    // new entry or update the edited one). On a validation error the form stays
    // open with the message shown, so nothing invalid ever reaches world.jsonl.
    fn confirm_form(&mut self, world: &mut World) {
        self.form_error = None;
        let Some(ty) = self.selected_type.clone() else {
            self.close_form();
            return;
        };
        let typed = panel::field_text(world, panel::NAME_INPUT);
        let texts: Vec<String> = self
            .form_fields
            .iter()
            .enumerate()
            .map(|(j, f)| {
                if matches!(f.kind, form::FieldKind::Bool) {
                    String::new()
                } else {
                    panel::field_text(world, panel::form_input(j))
                }
            })
            .collect();
        let editing_args = self
            .editing
            .and_then(|idx| self.entries.get(idx))
            .and_then(|e| e.get("args"))
            .and_then(|v| v.as_object())
            .cloned();
        let args = form::assemble(&ty, editing_args.as_ref(), &self.form_fields, &texts);
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
        self.dirty = true;
        self.close_form();
    }

    // Move the active body's scroll offset in the wheel direction.
    fn scroll(&mut self, delta: f32, world: &World) {
        match self.mode {
            panel::Mode::List if self.combo == Combo::Closed => {
                let max = self.list_rows().len().saturating_sub(panel::MAX_ROWS);
                self.list_scroll = scroll_step(self.list_scroll, delta, max);
                self.row_menu = None;
            }
            panel::Mode::List => {
                let max = self
                    .combo_options(world)
                    .len()
                    .saturating_sub(panel::MAX_ROWS);
                self.combo_scroll = scroll_step(self.combo_scroll, delta, max);
            }
            panel::Mode::AddForm => {}
        }
    }

    fn hud_state(&self) -> HudState {
        HudState {
            dirty: self.dirty,
            templates_open: self.templates_open,
            panel_open: self.panel_open,
            world_capture: self.world_capture,
            visible: self.hud_visible,
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
            if self.hud_visible {
                let vw = input.viewport[0];
                // Resolve a click against the top bar first; when it owns none of
                // the click and the panel is open, offer it to the panel.
                if let Some(a) = hud::hit_test(
                    input.mouse_x,
                    input.mouse_y,
                    input.left_click,
                    self.dirty,
                    self.templates_open,
                    vw,
                ) {
                    if self.apply_top(a) {
                        self.swap_requested = true;
                    }
                    // Interacting with the top bar dismisses any panel overlay and
                    // the form: a SAVE swaps the world (re-injecting blank fields),
                    // so a stale form left open would show empty inputs and could
                    // commit lost values.
                    self.combo = Combo::Closed;
                    self.row_menu = None;
                    self.close_form();
                } else if self.panel_open && input.left_click {
                    let action = {
                        let data = self.panel_data(world);
                        let view = self.make_view(&data, [input.mouse_x, input.mouse_y]);
                        panel::hit_test(&view, input.mouse_x, input.mouse_y, vw)
                    };
                    if let Some(pa) = action {
                        self.apply_panel(pa, world);
                    }
                }
                // Wheel over the panel body scrolls the list / combo options.
                if self.panel_open
                    && input.scroll_delta.abs() > 0.5
                    && panel::cursor_over_body(input.mouse_x, input.mouse_y, vw)
                {
                    self.scroll(input.scroll_delta, world);
                }
            }
        }

        // Drive the world's cursor / freeze state: edit mode (`Some(true)`) frees
        // the cursor and freezes the world; play mode (`Some(false)`) runs it.
        world.insert_resource(MenuOverride(Some(!self.world_capture)));

        // Re-anchor + recolour the top bar, then lay out (or hide) the panel.
        hud::apply_layout(world, self.hud_state());
        if self.hud_visible && self.panel_open {
            let (mouse, vw) = input
                .as_ref()
                .map(|i| ([i.mouse_x, i.mouse_y], i.viewport[0]))
                .unwrap_or(([0.0, 0.0], 0.0));
            let data = self.panel_data(world);
            let view = self.make_view(&data, mouse);
            panel::apply(world, Some(&view), vw);
        } else {
            panel::apply(world, None, 0.0);
        }
    }

    // Apply a pending live world swap: rebuild the recompiled world off disk,
    // transplant the running render backend into it, and re-`start` it, so the
    // edit renders without recreating the OS window. Run once per frame by the run
    // loop right after `tick`.
    //
    // The recompiled world is built FIRST, in a throwaway App; only once that
    // succeeds is the backend transplanted out of the live world. So a rebuild
    // failure leaves the live world -- and its window -- fully intact; the next
    // SAVE retries. The backend is never dropped on an error path.
    fn apply_world_swap(&mut self, app: &mut App) {
        if !self.swap_requested {
            return;
        }
        self.swap_requested = false;

        let mut staged = App::new();
        if let Err(e) = super::boot_world(&mut staged, &self.world_path, true) {
            tracing::error!("editor: live swap rebuild failed, keeping current world: {e}");
            return;
        }
        staged.world_mut().remove_all::<crate::assets::DebugHud>();
        super::inject::editor_hud(staged.world_mut());
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
            tracing::error!("editor: live swap start failed: {e:?}");
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

    // A world holding just a FrameInput, for driving `tick` directly.
    fn world_with_input(input: FrameInput) -> World {
        let mut world = World::new_empty();
        world.add_component(input);
        world
    }

    // A world with the injected panel fields, for the add / edit flow (the combo
    // filter, the name, and the form's arg-input pool).
    fn world_with_fields() -> World {
        let mut world = World::new_empty();
        for id in panel::all_field_ids() {
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
        assert!(!h.panel_open && !h.templates_open);
        assert_eq!(h.combo, Combo::Closed);
    }

    #[test]
    fn assets_button_toggles_panel_and_excludes_templates() {
        let mut h = hook(Vec::new());
        h.apply_top(HudAction::ToggleAssets);
        assert!(h.panel_open && h.mode == panel::Mode::List);
        // Opening Templates closes the panel.
        h.apply_top(HudAction::ToggleTemplates);
        assert!(h.templates_open && !h.panel_open);
        // Opening the panel closes Templates.
        h.apply_top(HudAction::ToggleAssets);
        assert!(h.panel_open && !h.templates_open);
    }

    #[test]
    fn templates_pick_applies_and_is_idempotent() {
        let mut h = hook(Vec::new());
        h.apply_top(HudAction::PickTemplate(0));
        let first =
            concinnity_core::world::parse_world_jsonl(concinnity_templates::TEMPLATES[0].jsonl)
                .unwrap()
                .len();
        assert_eq!(h.entries.len(), first, "all template entries added");
        h.apply_top(HudAction::PickTemplate(0));
        assert_eq!(h.entries.len(), first, "re-apply is idempotent");
    }

    #[test]
    fn only_save_signals_a_world_swap() {
        let mut h = hook(Vec::new());
        assert!(!h.apply_top(HudAction::ToggleAssets));
        assert!(!h.apply_top(HudAction::ToggleTemplates));
        assert!(!h.apply_top(HudAction::ToggleCapture));
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
        h.apply_top(HudAction::ToggleAssets);
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
        assert_eq!(h.mode, panel::Mode::AddForm);
        assert_eq!(h.combo, Combo::Closed);
        assert_eq!(h.selected_type.as_deref(), Some(ty.as_str()));
        assert!(h.editing.is_none());
        let name_field = world
            .query::<TextInput>()
            .find(|t| t.asset_id == panel::NAME_INPUT)
            .unwrap();
        assert!(name_field.focused && !name_field.content.is_empty());
        // Edit the name, then confirm.
        set_field(&mut world, panel::NAME_INPUT, "my_light");
        h.apply_panel(PanelAction::ConfirmAdd, &mut world);
        assert_eq!(h.mode, panel::Mode::List);
        assert!(h.dirty);
        assert_eq!(h.entries.len(), 1);
        assert_eq!(h.entries[0]["name"], "my_light");
        assert_eq!(h.entries[0]["type"], ty.as_str());
    }

    #[test]
    fn row_menu_edit_renames_the_existing_entry() {
        let mut h = hook(vec![entry("lamp", "PointLight")]);
        let mut world = world_with_fields();
        h.panel_open = true;
        // Open the row's menu, then Edit -> form prefilled for a rename.
        h.apply_panel(PanelAction::OpenRowMenu(0), &mut world);
        assert_eq!(h.row_menu, Some(0));
        h.apply_panel(PanelAction::RowEdit, &mut world);
        assert_eq!(h.mode, panel::Mode::AddForm);
        assert_eq!(h.editing, Some(0));
        assert_eq!(h.selected_type.as_deref(), Some("PointLight"));
        assert!(h.row_menu.is_none());
        let name_field = world
            .query::<TextInput>()
            .find(|t| t.asset_id == panel::NAME_INPUT)
            .unwrap();
        assert_eq!(name_field.content, "lamp", "name prefilled from the entry");
        // Rename and confirm: same entry, no new one.
        set_field(&mut world, panel::NAME_INPUT, "streetlamp");
        h.apply_panel(PanelAction::ConfirmAdd, &mut world);
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
        h.mode = panel::Mode::AddForm;
        // Rename "b" to "a": collides with the other entry -> suffixed.
        set_field(&mut world, panel::NAME_INPUT, "a");
        h.apply_panel(PanelAction::ConfirmAdd, &mut world);
        assert_eq!(h.entries[1]["name"], "a_1");
    }

    #[test]
    fn confirm_add_with_blank_name_uses_a_generated_one() {
        let mut h = hook(Vec::new());
        let mut world = world_with_fields();
        h.selected_type = Some("PointLight".to_string());
        h.mode = panel::Mode::AddForm;
        // Field left blank.
        h.apply_panel(PanelAction::ConfirmAdd, &mut world);
        assert_eq!(h.entries.len(), 1);
        assert_eq!(h.entries[0]["name"], "editor_pointlight");
    }

    #[test]
    fn confirm_add_makes_a_duplicate_name_unique() {
        let mut h = hook(vec![entry("lamp", "PointLight")]);
        let mut world = world_with_fields();
        h.selected_type = Some("PointLight".to_string());
        h.mode = panel::Mode::AddForm;
        set_field(&mut world, panel::NAME_INPUT, "lamp");
        h.apply_panel(PanelAction::ConfirmAdd, &mut world);
        assert_eq!(h.entries[1]["name"], "lamp_1", "collision is suffixed");
    }

    #[test]
    fn cancel_form_returns_to_the_list_without_adding() {
        let mut h = hook(Vec::new());
        let mut world = world_with_fields();
        h.selected_type = Some("Decal".to_string());
        h.mode = panel::Mode::AddForm;
        h.apply_panel(PanelAction::CancelForm, &mut world);
        assert_eq!(h.mode, panel::Mode::List);
        assert!(h.selected_type.is_none() && h.editing.is_none());
        assert!(h.entries.is_empty() && !h.dirty);
    }

    #[test]
    fn picker_pins_the_active_type_filter_first() {
        let mut h = hook(Vec::new());
        let world = world_with_fields();
        h.combo = Combo::Picker;
        h.type_filter = Some("Decal".to_string());
        let opts = h.combo_options(&world);
        assert_eq!(opts[0], "Decal", "the active filter is pinned to the top");
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

    #[test]
    fn add_form_writes_edited_arg_values() {
        let mut h = hook(Vec::new());
        let mut world = world_with_fields();
        h.apply_top(HudAction::ToggleAssets);
        h.apply_panel(PanelAction::TogglePicker, &mut world);
        let ty = h.combo_options(&world)[0].clone();
        h.apply_panel(PanelAction::PickOption(0), &mut world);
        assert_eq!(h.mode, panel::Mode::AddForm);
        assert!(!h.form_fields.is_empty(), "the type exposes arg fields");
        // Edit a float field via its input.
        let (j, key) = h
            .form_fields
            .iter()
            .enumerate()
            .find(|(_, f)| matches!(f.kind, form::FieldKind::Float))
            .map(|(j, f)| (j, f.key.clone()))
            .expect("a float arg field");
        set_field(&mut world, panel::form_input(j), "3.5");
        set_field(&mut world, panel::NAME_INPUT, "lamp");
        h.apply_panel(PanelAction::ConfirmAdd, &mut world);
        assert_eq!(h.mode, panel::Mode::List);
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
        set_field(&mut world, panel::form_input(j), "0.1, 0.2, 0.3");
        set_field(&mut world, panel::NAME_INPUT, "fog");
        h.apply_panel(PanelAction::ConfirmAdd, &mut world);
        assert_eq!(h.mode, panel::Mode::List);
        assert_eq!(h.entries.len(), 1);
        assert_eq!(h.entries[0]["type"], "VolumetricFog");
        assert_eq!(
            h.entries[0]["args"][&key],
            serde_json::json!([0.1, 0.2, 0.3]),
            "the edited colour persisted as a numeric array"
        );
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
        set_field(&mut world, panel::form_input(j), "-5");
        set_field(&mut world, panel::NAME_INPUT, "myfont");
        h.apply_panel(PanelAction::ConfirmAdd, &mut world);
        assert_eq!(
            h.mode,
            panel::Mode::AddForm,
            "the form stays open on invalid input"
        );
        assert!(h.form_error.is_some(), "an error message is shown");
        assert!(h.entries.is_empty(), "nothing invalid was committed");
    }

    #[test]
    fn a_top_bar_click_closes_an_open_form() {
        let mut h = hook(Vec::new());
        let mut world = world_with_fields();
        h.panel_open = true;
        h.open_form(&mut world, "PointLight".to_string(), None);
        assert_eq!(h.mode, panel::Mode::AddForm);
        // Click the capture checkbox (a top-bar control, no disk I/O) while the
        // form is open.
        let vw = 1280.0;
        let chk = hud::checkbox_rect(vw);
        world.add_component(FrameInput {
            viewport: [vw, 720.0],
            mouse_x: chk[0] + 10.0,
            mouse_y: chk[1] + 10.0,
            left_click: true,
            ..Default::default()
        });
        h.tick(&mut world);
        assert_eq!(h.mode, panel::Mode::List, "the form closed");
        assert!(h.form_fields.is_empty() && h.selected_type.is_none());
    }

    #[test]
    fn edit_form_seeds_and_updates_existing_args() {
        let mut h = hook(vec![serde_json::json!({
            "name": "lamp", "type": "PointLight", "args": {}
        })]);
        let mut world = world_with_fields();
        h.panel_open = true;
        h.apply_panel(PanelAction::OpenRowMenu(0), &mut world);
        h.apply_panel(PanelAction::RowEdit, &mut world);
        assert_eq!(h.editing, Some(0));
        assert!(!h.form_fields.is_empty());
        // The name field was seeded from the entry.
        assert_eq!(panel::field_text(&world, panel::NAME_INPUT), "lamp");
        // Edit a float and confirm; the same entry gains a full args object.
        let (j, key) = h
            .form_fields
            .iter()
            .enumerate()
            .find(|(_, f)| matches!(f.kind, form::FieldKind::Float))
            .map(|(j, f)| (j, f.key.clone()))
            .expect("a float arg field");
        set_field(&mut world, panel::form_input(j), "9.0");
        h.apply_panel(PanelAction::ConfirmAdd, &mut world);
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

        // Row menu: the Edit / Delete popup shows over the "a" name row (entry 0).
        h.combo = Combo::Closed;
        h.row_menu = Some(0);
        h.tick(&mut world);
        assert!(sprite_visible(&world, panel::MENU_BG), "row menu shown");
        assert_eq!(label(&world, panel::MENU_EDIT_LABEL).content, "Edit");
        assert_eq!(label(&world, panel::MENU_DELETE_LABEL).content, "Delete");

        // Add form: the name field + Add button show; the list rows are hidden.
        h.row_menu = None;
        h.mode = panel::Mode::AddForm;
        h.selected_type = Some("PointLight".to_string());
        h.tick(&mut world);
        assert!(
            sprite_visible(&world, panel::FORMADD_BG),
            "Add button shown"
        );
        assert!(
            !label(&world, panel::list_row_label(0)).visible,
            "list rows hidden in the form"
        );

        // Closing the panel blanks the whole thing.
        h.panel_open = false;
        h.tick(&mut world);
        assert!(!sprite_visible(&world, panel::PANEL_BG), "panel bg hidden");
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
        h.mode = panel::Mode::AddForm;
        set_field(&mut world, panel::NAME_INPUT, "lamp");
        h.apply_panel(PanelAction::ConfirmAdd, &mut world);
        h.write_jsonl().unwrap();

        let content = std::fs::read_to_string(&path).unwrap();
        let parsed = concinnity_core::world::parse_world_jsonl(&content).unwrap();
        assert_eq!(parsed.len(), 2, "both entries written, one line each");
        assert_eq!(parsed[1]["name"], "lamp");
        assert!(!std::path::Path::new(&format!("{path_str}.tmp")).exists());

        let _ = std::fs::remove_file(&path);
    }
}
