// src/editor/hook.rs
//
// The editor's per-frame drive. Implements the run loop's `DebugHook` seam: each
// frame it hit-tests the editor HUD's controls against the live input, mutates
// the working authored entry list, persists on SAVE, drives the world's cursor /
// freeze state, and re-anchors + recolours the HUD. This is the whole editor: it
// lives in the editor crate (never linked by the shipped runtime), so no editor
// code is compiled into a shipped game.
//
// Cursor control: the editor holds the cursor by default (edit mode -- cursor
// free, world frozen), publishing `MenuOverride(Some(true))`. Ticking the
// capture checkbox hands the cursor to the world (play mode, `Some(false)`);
// pressing Escape takes it back. F1 hides/shows the whole HUD.
//
// SAVE persists by re-serializing the authored entry list to world.jsonl and
// recompiling the blobs through the validated cook tail (`build_world_to_disk`),
// never by patching blobs directly. A successful SAVE then applies the edit to
// the running world without recreating the window: the live render backend is
// transplanted out of the world (`take_render_backend`) and `apply_world_swap`
// rebuilds the recompiled world onto it (carried in as a `PendingBackend`), so
// the edit shows immediately. When the backend cannot be transplanted (no live
// GraphicsSystem yet) the SAVE is persist-only and the edit shows next launch.

use super::hud::{self, HudAction, HudState, Menu};
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
    // Which dropdown is open (Add / Templates), or none.
    menu: Option<Menu>,
    // Whether the world currently holds the cursor (play mode). Starts false:
    // the editor owns the cursor at launch so the HUD is immediately usable.
    world_capture: bool,
    // Whether the whole HUD is shown (F1 toggle). Starts shown.
    hud_visible: bool,
    // Set by a successful SAVE (in `tick`), consumed by `apply_world_swap` (run
    // right after) to apply the recompiled world to the running one. The backend
    // is not transplanted until `apply_world_swap` has rebuilt the new world off
    // disk, so a rebuild failure leaves the live world (and its window) untouched.
    swap_requested: bool,
}

impl EditorHook {
    pub(crate) fn new(world_path: String, entries: Vec<serde_json::Value>) -> Self {
        Self {
            world_path,
            entries,
            dirty: false,
            menu: None,
            world_capture: false,
            hud_visible: true,
            swap_requested: false,
        }
    }

    // Append a new asset of `kind` with a generated unique name, marking the
    // world dirty. The args are left empty so the type's registered defaults
    // apply; the dropdown only offers standalone types, so it recompiles cleanly.
    fn add_asset(&mut self, kind: &str) {
        let name = self.unique_name(kind);
        self.entries.push(serde_json::json!({
            "name": name,
            "type": kind,
            "args": {},
        }));
        self.dirty = true;
    }

    // A world-unique name derived from the asset type: `editor_<kind>` plus a
    // numeric suffix bumped until it does not collide with an existing entry.
    fn unique_name(&self, kind: &str) -> String {
        let base = format!("editor_{}", kind.to_ascii_lowercase());
        let taken = |n: &str| {
            self.entries
                .iter()
                .any(|e| e.get("name").and_then(|v| v.as_str()) == Some(n))
        };
        if !taken(&base) {
            return base;
        }
        let mut i = 1;
        loop {
            let candidate = format!("{base}_{i}");
            if !taken(&candidate) {
                return candidate;
            }
            i += 1;
        }
    }

    // Persist the working entries: write world.jsonl, then recompile the blobs.
    // On success the world is clean again and `true` is returned so the caller
    // triggers the live world swap; on failure it stays dirty (so the user can
    // retry) and `false` is returned (no swap).
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
    // already exists in the working world (so re-applying is idempotent). Marks
    // the world dirty if anything was added.
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
            let clashes = name.is_some_and(|n| {
                self.entries
                    .iter()
                    .any(|e| e.get("name").and_then(|v| v.as_str()) == Some(n))
            });
            if clashes {
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

    // Route a resolved HUD click. Split out so the state transitions are
    // unit-testable without a live world or the render loop. Returns `true` only
    // when a SAVE succeeded, signalling the caller to transplant the backend and
    // request a live world swap.
    fn apply(&mut self, action: HudAction) -> bool {
        match action {
            HudAction::Save => return self.save(),
            HudAction::OpenMenu(menu) => self.menu = Some(menu),
            HudAction::CloseMenu => self.menu = None,
            HudAction::ToggleCapture => self.world_capture = !self.world_capture,
            HudAction::Pick(i) => {
                match self.menu {
                    Some(Menu::Add) => {
                        if let Some(kind) = hud::ADD_TYPES.get(i) {
                            self.add_asset(kind);
                        }
                    }
                    Some(Menu::Templates) => self.apply_template(i),
                    None => {}
                }
                self.menu = None;
            }
        }
        false
    }

    fn hud_state(&self) -> HudState {
        HudState {
            dirty: self.dirty,
            menu: self.menu,
            world_capture: self.world_capture,
            visible: self.hud_visible,
        }
    }
}

impl DebugHook for EditorHook {
    fn tick(&mut self, world: &mut World) {
        // Read the latest input snapshot, cloning so the world borrow ends before
        // we mutate it again.
        if let Some(input) = world.query::<FrameInput>().last().cloned() {
            // Escape hands the cursor back to the editor (leaves play mode).
            if input.escape {
                self.world_capture = false;
            }
            // F1 toggles the whole HUD (the editor's replacement for the debug
            // HUD's F1). `hud_toggle` is an edge pulse, so this flips once/press.
            if input.hud_toggle {
                self.hud_visible = !self.hud_visible;
            }
            // Clicks act only while the HUD is shown.
            if self.hud_visible
                && let Some(action) = hud::hit_test(
                    input.mouse_x,
                    input.mouse_y,
                    input.left_click,
                    self.dirty,
                    self.menu,
                    input.viewport[0],
                )
                && self.apply(action)
            {
                // A SAVE just persisted the recompiled blobs. Request the App-level
                // swap (`apply_world_swap`, run right after this tick) to apply the
                // edit to the running world. The backend is transplanted there,
                // only after the new world rebuilds, so a rebuild failure never
                // tears down the live world.
                self.swap_requested = true;
            }
        }

        // Drive the world's cursor / freeze state: edit mode (`Some(true)`) frees
        // the cursor and freezes the world; play mode (`Some(false)`) hands the
        // cursor to the world so its camera runs.
        world.insert_resource(MenuOverride(Some(!self.world_capture)));

        // Re-anchor + recolour the HUD (or hide it when F1-toggled off).
        hud::apply_layout(world, self.hud_state());
    }

    // Apply a pending live world swap: rebuild the recompiled world off disk,
    // transplant the running render backend into it, and re-`start` it, so the
    // edit renders without recreating the OS window. Run once per frame by the run
    // loop right after `tick`.
    //
    // The recompiled world is built FIRST, in a throwaway App; only once that
    // succeeds is the backend transplanted out of the live world and the swap
    // performed. So a rebuild failure (e.g. a bad read of the just-written blobs)
    // leaves the live world -- and its window -- fully intact and rendering; the
    // next SAVE retries. The backend is never dropped on an error path.
    fn apply_world_swap(&mut self, app: &mut App) {
        if !self.swap_requested {
            return;
        }
        self.swap_requested = false;

        // Build the recompiled world off the freshly written blobs (reusing the
        // boot path's build-if-needed / load / empty-seed logic). On failure the
        // live world is left untouched.
        let mut staged = App::new();
        if let Err(e) = super::boot_world(&mut staged, &self.world_path, true) {
            tracing::error!("editor: live swap rebuild failed, keeping current world: {e}");
            return;
        }
        // Drop the baked DebugHud (the editor HUD owns F1), inject the editor HUD,
        // and seed the cursor / freeze state so the freshly started world's first
        // frame already honours edit/play mode (no one-frame cursor grab).
        staged.world_mut().remove_all::<crate::assets::DebugHud>();
        super::inject::editor_hud(staged.world_mut());
        staged
            .world_mut()
            .insert_resource(MenuOverride(Some(!self.world_capture)));

        // Only now transplant the live backend into the rebuilt world (as a
        // `PendingBackend`, so its GraphicsSystem reuses the live window via
        // `reload_world`), drop the pre-edit world (joining its streaming threads
        // before the new world's start spawns its own), and start the new world.
        let Some(backend) = app.world_mut().take_render_backend() else {
            // No live backend to transplant (no started GraphicsSystem): the SAVE
            // is persist-only and the edit shows on the next launch.
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

    fn hook(entries: Vec<serde_json::Value>) -> EditorHook {
        EditorHook::new("unused.jsonl".to_string(), entries)
    }

    // A world holding just a FrameInput, for driving `tick` directly.
    fn world_with_input(input: FrameInput) -> World {
        let mut world = World::new_empty();
        world.add_component(input);
        world
    }

    // The editor starts owning the cursor (edit mode) with the HUD shown and no
    // menu open.
    #[test]
    fn starts_in_edit_mode_with_hud_shown() {
        let h = hook(Vec::new());
        assert!(!h.world_capture, "editor holds the cursor at launch");
        assert!(h.hud_visible, "HUD shown at launch");
        assert!(h.menu.is_none());
    }

    // The Add button opens its menu; a row pick adds that type and closes; a
    // dismiss just closes; the checkbox toggles cursor ownership.
    #[test]
    fn add_menu_click_actions_drive_state() {
        let mut h = hook(Vec::new());

        h.apply(HudAction::OpenMenu(Menu::Add));
        assert_eq!(h.menu, Some(Menu::Add));
        h.apply(HudAction::Pick(0));
        assert!(h.menu.is_none() && h.dirty);
        assert_eq!(h.entries.len(), 1);
        assert_eq!(h.entries[0]["type"], hud::ADD_TYPES[0]);

        h.apply(HudAction::OpenMenu(Menu::Add));
        h.apply(HudAction::CloseMenu);
        assert!(h.menu.is_none());

        assert!(!h.world_capture);
        h.apply(HudAction::ToggleCapture);
        assert!(h.world_capture, "checkbox hands the cursor to the world");
        h.apply(HudAction::ToggleCapture);
        assert!(!h.world_capture, "checkbox takes it back");
    }

    // Picking a template appends all its entries and marks dirty; re-applying it
    // is idempotent (names already present are skipped).
    #[test]
    fn templates_menu_applies_and_is_idempotent() {
        let mut h = hook(Vec::new());
        h.apply(HudAction::OpenMenu(Menu::Templates));
        assert_eq!(h.menu, Some(Menu::Templates));

        h.apply(HudAction::Pick(0));
        assert!(h.menu.is_none() && h.dirty);
        let first =
            concinnity_core::world::parse_world_jsonl(concinnity_templates::TEMPLATES[0].jsonl)
                .unwrap()
                .len();
        assert_eq!(h.entries.len(), first, "all template entries added");

        // Re-applying the same template adds nothing (names already present).
        h.apply(HudAction::OpenMenu(Menu::Templates));
        h.apply(HudAction::Pick(0));
        assert_eq!(h.entries.len(), first, "re-apply is idempotent");
    }

    // Only a successful SAVE signals the caller to transplant the backend and
    // swap the world; every other HUD action returns false (no swap).
    #[test]
    fn only_save_signals_a_world_swap() {
        let mut h = hook(Vec::new());
        assert!(
            !h.apply(HudAction::OpenMenu(Menu::Add)),
            "opening a menu is not a save"
        );
        assert!(!h.apply(HudAction::CloseMenu));
        assert!(!h.apply(HudAction::ToggleCapture));
        h.menu = Some(Menu::Add);
        assert!(
            !h.apply(HudAction::Pick(0)),
            "adding an asset is not a save (it swaps on the next SAVE)"
        );
    }

    // An out-of-range pick index (never produced by the HUD, but guarded) adds
    // nothing and still closes the menu.
    #[test]
    fn out_of_range_pick_is_ignored() {
        let mut h = hook(Vec::new());
        h.menu = Some(Menu::Add);
        h.apply(HudAction::Pick(999));
        assert!(h.menu.is_none());
        assert!(h.entries.is_empty() && !h.dirty);
    }

    #[test]
    fn added_names_are_unique_against_existing_and_each_other() {
        let mut h = hook(vec![serde_json::json!({
            "name": "editor_pointlight", "type": "PointLight", "args": {}
        })]);
        // ADD_TYPES[0] is PointLight, which already exists by name.
        h.menu = Some(Menu::Add);
        h.apply(HudAction::Pick(0));
        h.menu = Some(Menu::Add);
        h.apply(HudAction::Pick(0));
        let names: Vec<&str> = h
            .entries
            .iter()
            .filter_map(|e| e["name"].as_str())
            .collect();
        assert_eq!(
            names,
            [
                "editor_pointlight",
                "editor_pointlight_1",
                "editor_pointlight_2"
            ]
        );
    }

    // Escape while the world holds the cursor returns control to the editor.
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

    // F1 (an edge pulse) toggles the whole HUD on each press.
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

    // write_jsonl serializes the working entries to disk atomically and the
    // result round-trips back through the parser (one line per entry, no temp
    // file left behind).
    #[test]
    fn write_jsonl_persists_entries_atomically() {
        let path = std::env::temp_dir().join("cn_editor_write_jsonl_test.jsonl");
        let path_str = path.to_str().unwrap().to_string();
        let _ = std::fs::remove_file(&path);

        let mut h = hook(vec![serde_json::json!({
            "name": "scene", "type": "GraphicsConfig", "args": {}
        })]);
        h.world_path = path_str.clone();
        h.menu = Some(Menu::Add);
        h.apply(HudAction::Pick(0));
        h.write_jsonl().unwrap();

        let content = std::fs::read_to_string(&path).unwrap();
        let parsed = concinnity_core::world::parse_world_jsonl(&content).unwrap();
        assert_eq!(parsed.len(), 2, "both entries written, one line each");
        assert_eq!(parsed[1]["type"], hud::ADD_TYPES[0]);
        assert!(!std::path::Path::new(&format!("{path_str}.tmp")).exists());

        let _ = std::fs::remove_file(&path);
    }
}
