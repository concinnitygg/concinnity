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
// type-filter dropdown over a scrollable list of the world's existing assets, a
// "+" that opens a typed autocomplete of the addable types, and a name-first add
// form. The panel's two typed fields are real `TextInput` assets edited by the
// engine's text-input system; the hook reads them back.
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

use super::hud::{self, HudAction, HudState};
use super::panel::{self, PanelAction, PanelView};
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
    // Whether the panel's type-filter dropdown is expanded.
    type_dropdown_open: bool,
    // Active type filter for the List mode, or `None` for "all".
    type_filter: Option<String>,
    // First visible row of the (filtered) list / the picker autocomplete.
    list_scroll: usize,
    picker_scroll: usize,
    // The type chosen in the picker, being named in the add form.
    selected_type: Option<String>,
}

// Owned per-tick data backing a `PanelView` (computed from the entries + the live
// filter field, then borrowed for both hit-testing and layout).
struct PanelData {
    filter_options: Vec<String>,
    filter_selected: usize,
    filter_label: String,
    list_items: Vec<(String, String)>,
    picker_options: Vec<String>,
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
            type_dropdown_open: false,
            type_filter: None,
            list_scroll: 0,
            picker_scroll: 0,
            selected_type: None,
        }
    }

    // Whether an entry with this name already exists.
    fn name_taken(&self, n: &str) -> bool {
        self.entries
            .iter()
            .any(|e| e.get("name").and_then(|v| v.as_str()) == Some(n))
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

    // The final name for a form submission: the typed name (trimmed) made unique,
    // or a generated unique name when the field was left blank.
    fn finalize_name(&self, typed: &str, kind: &str) -> String {
        let t = typed.trim();
        if t.is_empty() {
            self.unique_name(kind)
        } else {
            self.unique_from(t)
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
            let name = entry.get("name").and_then(|v| v.as_str());
            if name.is_some_and(|n| self.name_taken(n)) {
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
            if let Some(ty) = e.get("type").and_then(|v| v.as_str())
                && !out.iter().any(|s| s == ty)
            {
                out.push(ty.to_string());
            }
        }
        out.sort();
        out
    }

    // The type-filter dropdown options: the "all" label plus the present types.
    fn filter_options(&self) -> Vec<String> {
        let mut opts = vec![panel::ALL_LABEL.to_string()];
        opts.extend(self.distinct_types());
        opts.truncate(panel::MAX_ROWS);
        opts
    }

    // The existing entries matching the active type filter, as (name, type).
    fn list_items(&self) -> Vec<(String, String)> {
        self.entries
            .iter()
            .filter_map(|e| {
                let name = e.get("name").and_then(|v| v.as_str())?;
                let ty = e.get("type").and_then(|v| v.as_str())?;
                if let Some(f) = &self.type_filter
                    && ty != f
                {
                    return None;
                }
                Some((name.to_string(), ty.to_string()))
            })
            .collect()
    }

    // The addable types offered by the picker, filtered by the typed field, with
    // the active type filter pinned to the top when it is offered.
    fn picker_options(&self, world: &World) -> Vec<String> {
        let filter = panel::field_text(world, panel::FILTER_INPUT).to_lowercase();
        let mut opts: Vec<String> = panel::ADD_TYPES
            .iter()
            .filter(|t| filter.is_empty() || t.to_lowercase().contains(&filter))
            .map(|t| t.to_string())
            .collect();
        if let Some(active) = &self.type_filter
            && let Some(pos) = opts.iter().position(|o| o == active)
        {
            let pinned = opts.remove(pos);
            opts.insert(0, pinned);
        }
        opts
    }

    fn panel_data(&self, world: &World) -> PanelData {
        let filter_options = self.filter_options();
        let filter_selected = match &self.type_filter {
            None => 0,
            Some(t) => filter_options.iter().position(|o| o == t).unwrap_or(0),
        };
        PanelData {
            filter_label: self
                .type_filter
                .clone()
                .unwrap_or_else(|| panel::ALL_LABEL.to_string()),
            filter_selected,
            filter_options,
            list_items: self.list_items(),
            picker_options: self.picker_options(world),
            form_title: match &self.selected_type {
                Some(t) => format!("New {t}"),
                None => "New asset".to_string(),
            },
        }
    }

    fn make_view<'a>(&self, d: &'a PanelData, mouse: [f32; 2]) -> PanelView<'a> {
        PanelView {
            mode: self.mode,
            type_dropdown_open: self.type_dropdown_open,
            filter_label: &d.filter_label,
            filter_options: &d.filter_options,
            filter_selected: d.filter_selected,
            list_items: &d.list_items,
            list_scroll: self.list_scroll,
            picker_options: &d.picker_options,
            picker_scroll: self.picker_scroll,
            form_title: &d.form_title,
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
                    self.type_dropdown_open = false;
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

    // Route a resolved panel click. Field-focus transitions mutate the injected
    // `TextInput` components, so this needs the world.
    fn apply_panel(&mut self, action: PanelAction, world: &mut World) {
        match action {
            PanelAction::TogglePicker => {
                if self.mode == panel::Mode::List {
                    self.mode = panel::Mode::TypePicker;
                    self.picker_scroll = 0;
                    self.type_dropdown_open = false;
                    panel::focus_field_with(world, panel::FILTER_INPUT, "");
                } else {
                    self.mode = panel::Mode::List;
                    self.selected_type = None;
                }
            }
            PanelAction::ToggleTypeDropdown => self.type_dropdown_open = !self.type_dropdown_open,
            PanelAction::PickFilter(i) => {
                if let Some(o) = self.filter_options().get(i) {
                    self.type_filter = if o == panel::ALL_LABEL {
                        None
                    } else {
                        Some(o.clone())
                    };
                }
                self.list_scroll = 0;
                self.type_dropdown_open = false;
            }
            PanelAction::PickType(i) => {
                if let Some(ty) = self.picker_options(world).get(i).cloned() {
                    let name = self.unique_name(&ty);
                    self.selected_type = Some(ty);
                    self.mode = panel::Mode::AddForm;
                    panel::focus_field_with(world, panel::NAME_INPUT, &name);
                }
            }
            PanelAction::ConfirmAdd => {
                if let Some(ty) = self.selected_type.clone() {
                    let typed = panel::field_text(world, panel::NAME_INPUT);
                    let name = self.finalize_name(&typed, &ty);
                    self.entries.push(serde_json::json!({
                        "name": name, "type": ty, "args": {},
                    }));
                    self.dirty = true;
                }
                self.selected_type = None;
                self.mode = panel::Mode::List;
            }
            PanelAction::CancelForm => {
                self.selected_type = None;
                self.mode = panel::Mode::List;
            }
            PanelAction::Consume => {}
        }
    }

    // Move the active body's scroll offset in the wheel direction.
    fn scroll(&mut self, delta: f32, world: &World) {
        match self.mode {
            panel::Mode::List => {
                let max = self.list_items().len().saturating_sub(panel::MAX_ROWS);
                self.list_scroll = scroll_step(self.list_scroll, delta, max);
            }
            panel::Mode::TypePicker => {
                let max = self
                    .picker_options(world)
                    .len()
                    .saturating_sub(panel::PICKER_ROWS);
                self.picker_scroll = scroll_step(self.picker_scroll, delta, max);
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
                // Wheel over the panel body scrolls the list / picker.
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
    use crate::assets::TextInput;

    fn hook(entries: Vec<serde_json::Value>) -> EditorHook {
        EditorHook::new("unused.jsonl".to_string(), entries)
    }

    // A world holding just a FrameInput, for driving `tick` directly.
    fn world_with_input(input: FrameInput) -> World {
        let mut world = World::new_empty();
        world.add_component(input);
        world
    }

    // A world with the injected panel fields, for the add flow.
    fn world_with_fields() -> World {
        let mut world = World::new_empty();
        world.add_component(TextInput {
            asset_id: panel::FILTER_INPUT,
            ..Default::default()
        });
        world.add_component(TextInput {
            asset_id: panel::NAME_INPUT,
            ..Default::default()
        });
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

    #[test]
    fn starts_in_edit_mode_with_hud_shown() {
        let h = hook(Vec::new());
        assert!(!h.world_capture, "editor holds the cursor at launch");
        assert!(h.hud_visible, "HUD shown at launch");
        assert!(!h.panel_open && !h.templates_open);
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
    fn picking_a_filter_narrows_the_list() {
        let mut h = hook(vec![
            serde_json::json!({"name":"a","type":"PointLight","args":{}}),
            serde_json::json!({"name":"b","type":"Decal","args":{}}),
            serde_json::json!({"name":"c","type":"PointLight","args":{}}),
        ]);
        // Filter options: "Assets" + distinct types (sorted): Decal, PointLight.
        let opts = h.filter_options();
        assert_eq!(opts[0], panel::ALL_LABEL);
        assert!(opts.contains(&"PointLight".to_string()));
        // Pick "PointLight".
        let pl = opts.iter().position(|o| o == "PointLight").unwrap();
        let mut world = world_with_fields();
        h.panel_open = true;
        h.apply_panel(PanelAction::PickFilter(pl), &mut world);
        assert_eq!(h.type_filter.as_deref(), Some("PointLight"));
        let items = h.list_items();
        assert_eq!(items.len(), 2, "only the PointLights match");
        assert!(items.iter().all(|(_, t)| t == "PointLight"));
    }

    #[test]
    fn plus_picker_then_name_form_adds_the_entry() {
        let mut h = hook(Vec::new());
        let mut world = world_with_fields();
        h.apply_top(HudAction::ToggleAssets);
        // "+" opens the picker; the filter field is focused.
        h.apply_panel(PanelAction::TogglePicker, &mut world);
        assert_eq!(h.mode, panel::Mode::TypePicker);
        assert!(
            world
                .query::<TextInput>()
                .find(|t| t.asset_id == panel::FILTER_INPUT)
                .unwrap()
                .focused
        );
        // Pick the first offered type -> AddForm, name field prefilled + focused.
        let ty = h.picker_options(&world)[0].clone();
        h.apply_panel(PanelAction::PickType(0), &mut world);
        assert_eq!(h.mode, panel::Mode::AddForm);
        assert_eq!(h.selected_type.as_deref(), Some(ty.as_str()));
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
        let mut h = hook(vec![
            serde_json::json!({"name":"lamp","type":"PointLight","args":{}}),
        ]);
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
        assert!(h.selected_type.is_none());
        assert!(h.entries.is_empty() && !h.dirty);
    }

    #[test]
    fn picker_pins_the_active_type_filter_first() {
        let mut h = hook(Vec::new());
        let world = world_with_fields();
        h.type_filter = Some("Decal".to_string());
        let opts = h.picker_options(&world);
        assert_eq!(opts[0], "Decal", "the active filter is pinned to the top");
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
