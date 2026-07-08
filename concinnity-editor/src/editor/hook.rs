// src/editor/hook.rs
//
// The editor's per-frame drive. Implements the run loop's `DebugHook` seam: each
// frame it hit-tests the editor HUD's controls against the live input, mutates
// the working authored entry list, persists on SAVE, and re-anchors + recolours
// the HUD. This is the whole editor: it lives in the editor crate (never linked
// by the shipped runtime), so no editor code is compiled into a shipped game.
//
// SAVE persists by re-serializing the authored entry list to world.jsonl and
// recompiling the blobs through the validated cook tail (`build_world_to_disk`),
// never by patching blobs directly. Phase 1 persists to disk only: the live
// world keeps rendering the pre-edit scene, and added assets appear on the next
// `cn editor` launch.

use super::hud::{self, HudAction};
use crate::assets::FrameInput;
use crate::debug_hook::DebugHook;
use crate::ecs::World;

pub(crate) struct EditorHook {
    // Path to the world.jsonl the edits are written back to.
    world_path: String,
    // The authored entry list (names live here, unlike the compiled blob). Edits
    // mutate this; SAVE serializes it back to `world_path`.
    entries: Vec<serde_json::Value>,
    // Whether `entries` has changes not yet written to disk.
    dirty: bool,
    // Whether the Add dropdown is open.
    menu_open: bool,
}

impl EditorHook {
    pub(crate) fn new(world_path: String, entries: Vec<serde_json::Value>) -> Self {
        Self {
            world_path,
            entries,
            dirty: false,
            menu_open: false,
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
    // On success the world is clean again; on failure it stays dirty so the user
    // can retry.
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

    // Route a resolved HUD click. Split out so the state transitions are
    // unit-testable without a live world or the render loop.
    fn apply(&mut self, action: HudAction) {
        match action {
            HudAction::Save => self.save(),
            HudAction::OpenMenu => self.menu_open = true,
            HudAction::CloseMenu => self.menu_open = false,
            HudAction::PickType(i) => {
                if let Some(kind) = hud::ADD_TYPES.get(i) {
                    self.add_asset(kind);
                }
                self.menu_open = false;
            }
        }
    }
}

impl DebugHook for EditorHook {
    fn tick(&mut self, world: &mut World) {
        // Resolve any HUD click against the latest input snapshot, then re-anchor
        // and recolour the HUD for this frame (before the world step draws it).
        // The input clone ends the world borrow before we touch the world again.
        if let Some(input) = world.query::<FrameInput>().last().cloned()
            && let Some(action) = hud::hit_test(
                input.mouse_x,
                input.mouse_y,
                input.left_click,
                self.dirty,
                self.menu_open,
                input.viewport[0],
            )
        {
            self.apply(action);
        }
        hud::apply_layout(world, self.dirty, self.menu_open);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn hook(entries: Vec<serde_json::Value>) -> EditorHook {
        EditorHook::new("unused.jsonl".to_string(), entries)
    }

    // The Add button opens the menu; a row pick adds that type and closes; a
    // dismiss just closes.
    #[test]
    fn menu_open_pick_and_close_transitions() {
        let mut h = hook(Vec::new());
        assert!(!h.menu_open && !h.dirty);

        h.apply(HudAction::OpenMenu);
        assert!(h.menu_open, "Add opens the menu");

        h.apply(HudAction::PickType(0));
        assert!(!h.menu_open, "picking closes the menu");
        assert!(h.dirty, "picking adds and dirties");
        assert_eq!(h.entries.len(), 1);
        assert_eq!(h.entries[0]["type"], hud::ADD_TYPES[0]);

        h.apply(HudAction::OpenMenu);
        h.apply(HudAction::CloseMenu);
        assert!(!h.menu_open, "dismiss closes without adding");
        assert_eq!(h.entries.len(), 1, "dismiss adds nothing");
    }

    // Each pick appends the type at that dropdown index with empty args.
    #[test]
    fn pick_appends_the_indexed_type() {
        let mut h = hook(Vec::new());
        h.apply(HudAction::PickType(1));
        assert_eq!(h.entries[0]["type"], hud::ADD_TYPES[1]);
        assert_eq!(h.entries[0]["args"], serde_json::json!({}));
        assert_eq!(
            h.entries[0]["name"],
            format!("editor_{}", hud::ADD_TYPES[1].to_ascii_lowercase())
        );
    }

    // An out-of-range pick index (never produced by the HUD, but guarded) adds
    // nothing and still closes the menu.
    #[test]
    fn out_of_range_pick_is_ignored() {
        let mut h = hook(Vec::new());
        h.menu_open = true;
        h.apply(HudAction::PickType(999));
        assert!(!h.menu_open);
        assert!(h.entries.is_empty());
        assert!(!h.dirty);
    }

    #[test]
    fn added_names_are_unique_against_existing_and_each_other() {
        let mut h = hook(vec![serde_json::json!({
            "name": "editor_pointlight", "type": "PointLight", "args": {}
        })]);
        // ADD_TYPES[0] is PointLight, which already exists by name.
        h.apply(HudAction::PickType(0));
        h.apply(HudAction::PickType(0));
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
        h.apply(HudAction::PickType(0));
        h.write_jsonl().unwrap();

        let content = std::fs::read_to_string(&path).unwrap();
        let parsed = concinnity_core::world::parse_world_jsonl(&content).unwrap();
        assert_eq!(parsed.len(), 2, "both entries written, one line each");
        assert_eq!(parsed[1]["type"], hud::ADD_TYPES[0]);
        assert!(!std::path::Path::new(&format!("{path_str}.tmp")).exists());

        let _ = std::fs::remove_file(&path);
    }
}
