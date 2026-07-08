// src/editor/hook.rs
//
// The editor's per-frame drive. Implements the run loop's `DebugHook` seam: each
// frame it hit-tests the editor HUD's buttons against the live input, mutates
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

// The type the phase-1 add button appends. A point light is standalone (no
// source files) and valid with default args in any rendering world, so a SAVE
// after an add always recompiles cleanly.
const ADD_ASSET_TYPE: &str = "PointLight";

pub(crate) struct EditorHook {
    // Path to the world.jsonl the edits are written back to.
    world_path: String,
    // The authored entry list (names live here, unlike the compiled blob). Edits
    // mutate this; SAVE serializes it back to `world_path`.
    entries: Vec<serde_json::Value>,
    // Whether `entries` has changes not yet written to disk.
    dirty: bool,
}

impl EditorHook {
    pub(crate) fn new(world_path: String, entries: Vec<serde_json::Value>) -> Self {
        Self {
            world_path,
            entries,
            dirty: false,
        }
    }

    // Append a new asset of `kind` with a generated unique name, marking the
    // world dirty. The args are left empty so the type's registered defaults
    // apply; a standalone type (e.g. PointLight) then recompiles cleanly.
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

    // Route a resolved HUD click. Split out so the mapping is unit-testable
    // (AddAsset) without a live world or the render loop.
    fn apply(&mut self, action: HudAction) {
        match action {
            HudAction::Add => self.add_asset(ADD_ASSET_TYPE),
            HudAction::Save => self.save(),
        }
    }
}

impl DebugHook for EditorHook {
    fn tick(&mut self, world: &mut World) {
        // Resolve any HUD click against the latest input snapshot, then re-anchor
        // and recolour the buttons for this frame (before the world step draws
        // them). The input clone ends the world borrow before we touch the world
        // again.
        if let Some(input) = world.query::<FrameInput>().last().cloned()
            && let Some(action) = hud::hit_test(
                input.mouse_x,
                input.mouse_y,
                input.left_click,
                self.dirty,
                input.viewport[0],
            )
        {
            self.apply(action);
        }
        hud::apply_layout(world, self.dirty);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn hook(entries: Vec<serde_json::Value>) -> EditorHook {
        EditorHook::new("unused.jsonl".to_string(), entries)
    }

    #[test]
    fn add_action_appends_entry_and_marks_dirty() {
        let mut h = hook(Vec::new());
        assert!(!h.dirty);
        h.apply(HudAction::Add);
        assert!(h.dirty);
        assert_eq!(h.entries.len(), 1);
        assert_eq!(h.entries[0]["type"], ADD_ASSET_TYPE);
        assert_eq!(h.entries[0]["name"], "editor_pointlight");
        assert_eq!(h.entries[0]["args"], serde_json::json!({}));
    }

    #[test]
    fn added_names_are_unique_against_existing_and_each_other() {
        let mut h = hook(vec![serde_json::json!({
            "name": "editor_pointlight", "type": "PointLight", "args": {}
        })]);
        h.apply(HudAction::Add);
        h.apply(HudAction::Add);
        let names: Vec<&str> = h
            .entries
            .iter()
            .filter_map(|e| e["name"].as_str())
            .collect();
        // The pre-existing name plus two distinct generated ones.
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
        h.apply(HudAction::Add);
        h.write_jsonl().unwrap();

        let content = std::fs::read_to_string(&path).unwrap();
        let parsed = concinnity_core::world::parse_world_jsonl(&content).unwrap();
        assert_eq!(parsed.len(), 2, "both entries written, one line each");
        assert_eq!(parsed[1]["name"], "editor_pointlight");
        // The temp file is renamed away, not left behind.
        assert!(!std::path::Path::new(&format!("{path_str}.tmp")).exists());

        let _ = std::fs::remove_file(&path);
    }
}
